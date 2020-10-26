import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig

import tensorflow_model_optimization.python.core.quantization.keras.quantize as quantize_base

from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_scope
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_apply
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_layer
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay
from tensorflow.keras.callbacks import Callback
import logging

log = logging.getLogger()


class SLGWeightGenerator(Quantizer):
    # Pruning type furnished either as class type and not an object
    # Pruning config furnished as dictionary
    def __init__(self):
        super().__init__()
        self.num_gen_layers = 3
        self.naming_list = ["out_{}_row_{}", "out_{}_col_{}", "out_{}_in_{}", "out_{}_bias_{}"]

    def get_config(self):
        config = {}
        return config

    def build(self, tensor_shape, name, layer):
        self.num_out_ch = tensor_shape[-1]
        self.num_entries = tensor_shape[0] * tensor_shape[1] * tensor_shape[2] * tensor_shape[3]
        self.num_entries_kernel = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]

        variable_dict = {}
        seed_array = np.cos(np.indices(dimensions=tensor_shape[0:3], dtype=np.float32))

        # Shaoe is row col in ch out ch
        row_indices = seed_array[0]
        col_indices = seed_array[1]
        ch_indices = seed_array[2]



        dtype = tf.float32
        variable_dict["row_indices"] = tf.constant(value=row_indices, dtype=dtype, name="row_indices")
        variable_dict["col_indices"] = tf.constant(value=col_indices, dtype=dtype, name="col_indices")
        variable_dict["ch_indices"] = tf.constant(value=ch_indices, dtype=dtype, name="ch_indices")

        for out_ch in range(self.num_out_ch):
            variable_dict["out_rescale_{}".format(out_ch)] = layer.add_weight(
                name=name + "_out_rescale_{}".format(out_ch),
                shape=(),
                initializer=tf.keras.initializers.glorot_normal(),
                # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
                trainable=True,
                dtype=dtype
            )

            variable_dict["out_shift_{}".format(out_ch)] = layer.add_weight(
                name=name + "_out_shift_{}".format(out_ch),
                shape=(),
                initializer=tf.keras.initializers.glorot_normal(),
                # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
                trainable=True,
                dtype=dtype
            )

            for gen_idx in range(self.num_gen_layers):
                for var_name in self.naming_list:
                    var_name = var_name.format(out_ch, gen_idx)

                    if "bias" in var_name:
                        init = tf.keras.initializers.Zeros()
                    else:
                        init = tf.keras.initializers.glorot_normal()

                    variable_dict[var_name] = layer.add_weight(
                        name=name + "_" + var_name,
                        shape=(),
                        initializer=init,
                        # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
                        trainable=True,
                        dtype=dtype
                    )

        generated = self.num_gen_layers * self.num_out_ch * len(self.naming_list) + 2 * self.num_out_ch

        ratio = generated * 100.0 / self.num_entries
        params_kernel = int(generated / self.num_out_ch)
        log.info(
            "Layer {} generated {} variables, original size {}, ratio {:.2f}% - {} params / kernel".format(name,
                                                                                                           generated,
                                                                                                           self.num_entries,
                                                                                                           ratio,
                                                                                                           params_kernel))

        return variable_dict

    def __call__(self, inputs, training, weights, **kwargs):

        # Weights mode and we aren't in training
        # don't need to be re quantized outside training
        if training is False:
            return inputs
        else:
            row_indices = weights["row_indices"]
            col_indices = weights["col_indices"]
            ch_indices = weights["ch_indices"]

            kernel_list = []
            for out_ch in range(self.num_out_ch):
                out_rescale = weights["out_rescale_{}".format(out_ch)]
                out_shift = weights["out_shift_{}".format(out_ch)]

                for gen_idx in range(self.num_gen_layers):
                    row = weights["out_{}_row_{}".format(out_ch, gen_idx)]
                    col = weights["out_{}_col_{}".format(out_ch, gen_idx)]
                    in_ch = weights["out_{}_in_{}".format(out_ch, gen_idx)]
                    bias = weights["out_{}_bias_{}".format(out_ch, gen_idx)]

                    slg_layer = row_indices * row + col_indices * col + ch_indices * in_ch + bias
                    slg_layer = tf.math.sin(slg_layer)

                    if gen_idx == 0:
                        slg_mem = slg_layer
                    else:
                        slg_mem = (slg_mem + slg_layer)/2.0

                slg_mem = slg_mem * out_rescale + out_shift
                slg_mem = tf.math.sin(slg_mem)

                kernel_list.append(slg_mem)

            slg_kernels = tf.stack(kernel_list, axis=3) / float(self.num_entries_kernel / 2.0)

            with tf.control_dependencies([inputs.assign(slg_kernels)]):
                return tf.identity(slg_kernels)


class SLGQuantizeConfig(QuantizeConfig):
    def __init__(self, output_quantizer=None, weight_quantizer=None, bias_quantizer=None, pruning_policy=0):
        self.weight_quantizer = SLGWeightGenerator()

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
    assert quantize_base.SET_CUSTOM_TNH_FLAG, log.info("TFMOD needs to be modified with quantizer disabled for proper "
                                                       "running")

    # Helper function uses `quantize_annotate_layer` to annotate that only the
    # Dense layers should be quantized.
    def add_quantize_annotation(layer):
        # create new layer to break link with old model
        layer = layer.__class__.from_config(layer.get_config())

        quantization_map = [
            # tf.keras.layers.Dense,
            tf.keras.layers.Conv2D
            # tf.keras.layers.Input: BFPInputQuantizerConfig()
        ]

        for layer_type in quantization_map:

            if isinstance(layer, layer_type):
                quantize_config = SLGQuantizeConfig()

                log.info(
                    "**SLG annotation added to layer {} of type {} with {}".format(layer.name,
                                                                                   layer_type,
                                                                                   quantize_config))

                quantized_layer = quantize_annotate_layer(to_annotate=layer, quantize_config=quantize_config)
                return quantized_layer
        log.info("**SLG annotation not added to layer {} of type {}".format(layer.name, type(layer)))

        return layer

    # Use `tf.keras.models.clone_model` to apply `add_quantize_annotation`
    # to the layers of the model.
    log.info("Annotating model {}".format(model.name))

    tf.keras.backend.clear_session()
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=add_quantize_annotation,
    )

    with quantize_scope({
        "SLGWeightGenerator": SLGWeightGenerator,
        "SLGQuantizeConfig": SLGQuantizeConfig,
    }):
        # Use `quantize_apply` to actually make the model quantization aware.
        quant_aware_model = quantize_apply(annotated_model)
        return quant_aware_model
