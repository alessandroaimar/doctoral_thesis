import tensorflow as tf
from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig

from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_scope
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_apply
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_layer
import logging
import math
from tensorflow.keras.regularizers import Regularizer

log = logging.getLogger()
log.setLevel(level=logging.DEBUG)


def get_slc_loss(kernel):
    tensor_shape = kernel.shape
    target_std = -math.sqrt(2.0 / (tensor_shape[0] * tensor_shape[2] + tensor_shape[1] * tensor_shape[3]))
    target_trunc = 2.0 * target_std
    kernel_mean, kernel_var = tf.nn.moments(kernel, axes=[0, 1, 2, 3])
    kernel_std = tf.sqrt(kernel_var)

    slc_mean_loss = tf.square(-kernel_mean)  # difference between target and actual mean                          #
    slc_std_loss = tf.square(
        kernel_std + target_std)  # difference between target and actual stdev target_std has a - in definition
    slc_trunc_loss = tf.reduce_mean(tf.nn.relu(tf.abs(kernel) - target_trunc))

    return slc_mean_loss + slc_std_loss + slc_trunc_loss


class SLCRegularizer(Regularizer):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, x):
        if self.scale == 0:
            return x

        kernel = x
        tensor_shape = kernel.shape
        target_std = -math.sqrt(2.0 / (tensor_shape[0] * tensor_shape[2] + tensor_shape[1] * tensor_shape[3]))
        target_trunc = 2.0 * target_std
        kernel_mean, kernel_var = tf.nn.moments(kernel, axes=[0, 1, 2, 3])
        kernel_std = tf.sqrt(kernel_var)

        slc_mean_loss = tf.square(kernel_mean)  # difference between target and actual mean (that is 0)
        slc_std_loss = tf.square(kernel_std + target_std)  # difference between target and actual stdev target_std has a - in definition
        slc_trunc_loss = tf.reduce_mean(tf.nn.relu(tf.abs(kernel) - target_trunc))

        total_loss = self.scale * (slc_mean_loss + slc_std_loss + slc_trunc_loss)
        # tf.print(total_loss)
        return total_loss

    def get_config(self):
        return {'scale': float(self.scale)}


class ShiftedL1L2(Regularizer):
    r"""A regularizer that applies both L1 and L2 regularization penalties with offset.

    The L1 regularization penalty is computed as:
    $$\ell_1\,\,penalty =\ell_1\sum_{i=0}^n|x_i|$$

    The L2 regularization penalty is computed as
    $$\ell_2\,\,penalty =\ell_2\sum_{i=0}^nx_i^2$$

    Attributes:
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0., offset=0.):
        self.l1 = l1
        self.l2 = l2
        self.offset = offset

    def __call__(self, x):
        if not self.l1 and not self.l2:
            return 0
        regularization = 0.

        if self.offset != 0:
            x = self.offset + 1
        if self.l1:
            regularization += self.l1 * tf.reduce_sum(tf.abs(x))
        if self.l2:
            regularization += self.l2 * tf.reduce_sum(tf.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2), 'offset': float(self.offset)}


class SLCWeightGenerator(Quantizer):
    def __init__(self):
        super().__init__()
        self.extraction_num_units_n_x_filter = 1
        self.loop_num_units = 32
        self.cut_threshold = 50

        # self.nl = tf.nn.leaky_relu
        # self.nl_dict = {"alpha": 0.5}
        # self.matrix_initializer = keras.initializers.glorot_normal()
        self.nl = tf.nn.leaky_relu
        self.nl_dict = {}
        self.matrix_initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode='fan_out')
        self.matrix_regularizer = tf.keras.regularizers.l2(l=1e-6)
        self.loop_mode = False

    def get_config(self):
        config = dict()
        return config

    def build(self, tensor_shape, name, layer):
        if self.loop_mode is True:
            # log.info("Building Loop Generator")
            variable_dict = self.build_loop(tensor_shape, name, layer)
        else:
            # log.info("Building RNN Generator")
            variable_dict = self.build_rnn(tensor_shape, name, layer)
        return variable_dict

    def build_loop(self, tensor_shape, name, layer):
        self.tensor_shape = tensor_shape
        self.layer = layer
        variable_dict = dict()

        # print("tensor_shape", tensor_shape)
        self.name_und = layer.name + "_"

        loop_num_units = self.loop_num_units
        # extract_num_units = tensor_shape[0] * tensor_shape[1] * self.extraction_num_units_n_x_filter
        extract_shape = [loop_num_units, tensor_shape[0], tensor_shape[1]]
        extract_bias_shape = [tensor_shape[1]]
        loop_shape = [loop_num_units, loop_num_units]
        loop_bias_shape = [loop_num_units]
        num_out_ch = tensor_shape[3]
        num_in_ch = tensor_shape[2]
        self.num_extract_steps = num_in_ch // self.extraction_num_units_n_x_filter

        if num_in_ch % self.extraction_num_units_n_x_filter != 0:
            print(name, num_in_ch, self.extraction_num_units_n_x_filter)
            raise ValueError

        seed_shape = [num_out_ch, loop_num_units]

        self.original_size = tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3]

        seed_size = seed_shape[0] * seed_shape[1]
        loop_size = loop_shape[0] * (loop_shape[1] + 1)
        extract_size = extract_shape[0] * extract_shape[1] * extract_shape[2] + extract_shape[2]

        self.compressed_size = seed_size + loop_size + extract_size
        self.compression_ratio = self.compressed_size * 100.0 / self.original_size
        self.compressed = True

        layer.compressed_size = self.compressed_size
        layer.original_size = self.original_size
        layer.compressed = self.compressed

        if self.compression_ratio > self.cut_threshold:
            self.compressed = False
            log.info("Layer {} size: {} compressed size: {} ratio {:.2f}% - Not compressed\n".format(layer.name,
                                                                                                     self.original_size,
                                                                                                     self.compressed_size,
                                                                                                     self.compression_ratio))

            return variable_dict

        log.info(
            "Layer {} size: {} compressed size: {} ratio {:.2f}% - Compressed".format(layer.name, self.original_size,
                                                                                      self.compressed_size,
                                                                                      self.compression_ratio))

        log.info("Seed shape: {} - Loop Matrix {} - Loop Bias {} - Extract Matrix: {} - Extract Bias {}\n".format(seed_shape, loop_shape,
                                                                                                                  loop_bias_shape,
                                                                                                                  extract_shape,
                                                                                                                  extract_bias_shape))

        variable_dict["seed"] = layer.add_weight(
            name=name + "_seed",
            shape=seed_shape,
            initializer=self.matrix_initializer,
            regularizer=self.matrix_regularizer,
            trainable=True,

        )

        variable_dict["loop_matrix"] = layer.add_weight(
            name=name + "_loop_matrix",
            shape=loop_shape,
            initializer=self.matrix_initializer,
            regularizer=self.matrix_regularizer,
            trainable=True
        )

        variable_dict["loop_bias"] = layer.add_weight(
            name=name + "_loop_bias",
            shape=loop_bias_shape,
            initializer=keras.initializers.zeros(),
            regularizer=self.matrix_regularizer,
            trainable=True
        )

        variable_dict["extract_matrix"] = layer.add_weight(
            name=name + "_extract_matrix",
            shape=extract_shape,
            initializer=self.matrix_initializer,
            regularizer=self.matrix_regularizer,
            trainable=True
        )

        variable_dict["extract_bias"] = layer.add_weight(
            name=name + "_extract_bias",
            shape=extract_bias_shape,
            initializer=keras.initializers.zeros(),
            regularizer=self.matrix_regularizer,
            trainable=True
        )

        variable_dict["stored_tensor"] = layer.add_weight(
            name=name + "_stored_tensor",
            shape=tensor_shape,
            initializer=keras.initializers.zeros(),
            regularizer=self.matrix_regularizer,
            trainable=False
        )

        return variable_dict

    def build_rnn(self, tensor_shape, name, layer):
        self.tensor_shape = tensor_shape
        self.layer = layer
        variable_dict = dict()

        num_in_ch = tensor_shape[2]
        self.num_extract_steps = num_in_ch // self.extraction_num_units_n_x_filter

        batch_size = tensor_shape[3]
        timesteps = self.num_extract_steps
        features = self.loop_num_units
        seed_shape = [batch_size, 1, features]
        self.seed_shape = seed_shape
        extract_num_units = tensor_shape[0] * tensor_shape[1]

        rescaler_shape = [batch_size]
        shifter_shape = [batch_size, 1, 1]
        # base_seed = tf.Variable(
        #     initial_value=tf.random.normal(seed_shape), trainable=True, validate_shape=True,
        #     name=name + "_seed", dtype=tf.float32, shape=seed_shape
        # )
        base_seed = layer.add_weight(
            name=name + "_seed",
            shape=seed_shape,
            initializer=tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_out'),
            regularizer=tf.keras.regularizers.l2(l=1e-6),
            trainable=True,
        )

        # print_op = tf.compat.v1.Print(base_seed,[base_seed],  "{}".format(layer.name), first_n = 3)

        repeated_seed = [base_seed for _ in range(timesteps)]
        # with tf.control_dependencies([print_op]):
        variable_dict["seed"] = tf.concat(repeated_seed, axis=1)
        # input_ph = keras.layers.Input(batch_size=batch_size, shape=variable_dict["seed"].shape[1:])

        input_ph = variable_dict["seed"]

        loop_activation = "tanh" # tf.keras.activations.hard_sigmoid #tf.keras.activations.relu
        extract_activation = "tanh" #tf.keras.activations.hard_sigmoid

        rnn_cell = tf.keras.layers.GRU(
            self.loop_num_units,
            activation=loop_activation,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-6),
            recurrent_regularizer=None,
            bias_regularizer=tf.keras.regularizers.l2(l=1e-6),
            activity_regularizer=tf.keras.regularizers.l2(l=1e-6),
            kernel_constraint=None, recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=True,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=True,
        )(inputs=input_ph)



        extract_cell = tf.keras.layers.GRU(
            extract_num_units,
            activation=extract_activation,
            use_bias=True,
            kernel_initializer=tf.keras.initializers.he_normal(seed=None),
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=tf.keras.regularizers.l2(l=1e-6),
            recurrent_regularizer=None,
            bias_regularizer=tf.keras.regularizers.l2(l=1e-6),
            activity_regularizer=tf.keras.regularizers.l2(l=1e-6),
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            return_sequences=True,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=True,
        )(inputs=rnn_cell)

        rescaler = layer.add_weight(
            name=name + "_rescaler",
            shape=rescaler_shape,
            initializer=tf.constant_initializer(value=1.0),
            #regularizer=ShiftedL1L2(l1=1e-6, offset=1.0),
            trainable=True,
        )

        shifter = layer.add_weight(
            name=name + "_shifter",
            shape=shifter_shape,
            initializer=tf.constant_initializer(value=0.0),
            #regularizer=tf.keras.regularizers.l2(l=1e-6),
            trainable=True,
        )

        rescaled = tf.einsum('nmp,n->nmp', extract_cell, rescaler) + shifter

        log.info("Base shape: {} - Seed shape: {} - Loop shape: {} - Extract Shape {} - Final Shape {}".format(base_seed.shape,
                                                                                           variable_dict["seed"].shape,
                                                                                           rnn_cell.shape,
                                                                                           extract_cell.shape,
                                                                                           tensor_shape))
        kernels = tf.reshape(rescaled, tensor_shape)
        variable_dict["kernels"] = kernels  # tf.keras.Model(inputs=input_ph, outputs=kernels)


        base_size = base_seed.shape[0] * base_seed.shape[1] * base_seed.shape[2]
        loop_size = self.loop_num_units
        extract_shape = extract_num_units

        self.compressed_size = extract_shape + loop_size + base_size + rescaler_shape[0] + shifter_shape[0]
        self.original_size = self.original_size = tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
        self.compression_ratio = self.compressed_size * 100.0 / self.original_size


        if self.compression_ratio > self.cut_threshold:
            self.compressed = False
            log.info("Layer {} size: {} compressed size: {} ratio {:.2f}% - Not compressed\n".format(layer.name,
                                                                                                     self.original_size,
                                                                                                     self.compressed_size,
                                                                                                     self.compression_ratio))
        else:
            self.compressed = True
            log.info("Layer {} size: {} compressed size: {} ratio {:.2f}% - Compressed\n".format(layer.name, self.original_size,
                                                                                               self.compressed_size,
                                                                                               self.compression_ratio))
        layer.compressed_size = self.compressed_size
        layer.original_size = self.original_size
        layer.compressed = self.compressed

        return variable_dict

    def get_weights_rnn(self, weights):
        return weights["kernels"]

    def get_weights(self, weights):

        seeds = weights["seed"]
        loop_m = weights["loop_matrix"]
        loop_b = weights["loop_bias"]
        extract_m = weights["extract_matrix"]
        extract_b = weights["extract_bias"]

        state = seeds

        extracted_data_list = list()
        # Generation loop
        for extract_step in range(self.num_extract_steps):
            expand_basename = self.name_und + "expand_{}_".format(extract_step)
            extract_basename = self.name_und + "extract_{}_".format(extract_step)

            state = state + seeds
            state = self.mult_add_nl(state, loop_m, loop_b, expand_basename)
            # Result appended before non linearity
            extracted_data = self.mult_add(state, extract_m, extract_b, extract_basename)

            # We store the partial results in a list for now, later we will concatenate all the entries in a single tf op
            extracted_data_list.append(extracted_data)

        # Create the array
        # print(extracted_data_list[0].shape, len(extracted_data_list))
        # concat_weights = tf.concat(extracted_data_list, axis=-1)
        concat_weights = tf.stack(extracted_data_list, axis=-1)
        concat_weights = tf.transpose(concat_weights, [1, 2, 3, 0])
        # print("Output Shape", concat_weights.shape)
        weights["stored_tensor"].assign(concat_weights)

        return concat_weights

    def __call__(self, inputs, training, weights, **kwargs):
        kernel = self.get_weights_rnn(weights)
        return kernel
        # Weights mode and we aren't in training
        # don't need to be re quantized outside training
        if self.compression_ratio > self.cut_threshold:
            return inputs
        else:
            if self.loop_mode is True:
                kernel = self.get_weights(weights)
            else:
                kernel = self.get_weights_rnn(weights)
            return kernel

    def mult_add(self, x, mult, add, name):
        # ret = tf.matmul(x, mult, name=name + "_matmul")
        ret = tf.tensordot(x, mult, axes=1, name=name + "_tensordot")
        if add is not None:
            ret = tf.nn.bias_add(ret, add, name=name + "_bias")
        return ret

    def mult_add_nl(self, x, mult, add, name):
        matmult = self.mult_add(x, mult, add, name)
        ret = self.nl(matmult, **self.nl_dict, name=name + "_nl")
        # ret = tf.nn.leaky_relu(matmult, self.leaky_coeff, name=name + "_nl")
        return ret


class SLCQuantizeConfig(QuantizeConfig):
    def __init__(self, weight_quantizer=None):

        if weight_quantizer is None:
            self.weight_quantizer = SLCWeightGenerator()
        else:
            self.weight_quantizer = weight_quantizer

    # Configure how to quantize weights and biases
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, self.weight_quantizer)]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in get_weights_and_quantizers in the same order
        layer.kernel = quantize_weights[0]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`, in the same order.
        pass

    # Configure how to quantize outputs.
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        config = dict()
        config["weight_quantizer"] = self.weight_quantizer
        return config


def apply_quantization(model):
    # Helper function uses `quantize_annotate_layer` to annotate that only the
    # Dense layers should be quantized.
    def add_quantize_annotation(layer):
        kernelization_map = [
            # tf.keras.layers.Dense,
            tf.keras.layers.Conv2D
        ]

        for layer_type in kernelization_map:
            if isinstance(layer, layer_type):
                quantize_config = SLCQuantizeConfig()

                log.info(
                    "**Kernelization annotation added to layer {} of type {} with {}".format(layer.name, layer_type,
                                                                                             quantize_config))

                quantized_layer = quantize_annotate_layer(to_annotate=layer, quantize_config=quantize_config)
                return quantized_layer
        log.info("**Kernelization annotation not added to layer {} of type {}".format(layer.name, type(layer)))

        return layer

    # Use `tf.keras.models.clone_model` to apply `add_quantize_annotation`
    # to the layers of the model.
    log.info("Annotating model {}".format(model.name))
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=add_quantize_annotation,
    )

    with quantize_scope({
        'SLCQuantizeConfig': SLCQuantizeConfig,
        "SLCWeightGenerator": SLCWeightGenerator,
        "SLCRegularizer": SLCRegularizer
    }):
        # Use `quantize_apply` to actually make the model kernelization aware.
        quant_aware_model = quantize_apply(annotated_model)

        original_size = 0
        compressed_size = 0
        for layer in quant_aware_model.layers:
            try:
                original_size = original_size + layer.original_size
                if layer.compressed is True:
                    compressed_size = compressed_size + layer.compressed_size
                else:
                    compressed_size = compressed_size + layer.original_size

            except AttributeError:
                pass
        try:
            ratio = compressed_size * 100.0 / original_size
            log.info(
                "Model original size: {}, compressed size: {}, ratio: {:.2f}%".format(original_size, compressed_size, ratio))
        except ZeroDivisionError:
            log.info(
                "Zero division error? Model original size: {}, compressed size: {}".format(original_size, compressed_size))

        return quant_aware_model
