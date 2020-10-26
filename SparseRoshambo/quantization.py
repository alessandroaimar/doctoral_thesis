import tensorflow as tf
from functools import reduce
import numpy as np
import logging
import math
import os
import pathlib

log = logging.getLogger()

# TODO SPECIFIC TO TNH
# HW_MAX_SHIFT = 16  # TODO parameterize to accelerator model
ACTIVATION_OVERFLOW_MARGIN = 0
_DEBUG_FORCE_EXP = None
_DEBUG_DUMP_OUT_ACTIV = False


def get_dytpe(variable):
    if variable.dtype in [tf.float32, tf.float64, tf.float16]:
        return variable.dtype

    else:
        if variable.dtype == "float64_ref":
            return tf.float64
        elif variable.dtype == "float32_ref":
            return tf.float32
        elif variable.dtype == "float16_ref":
            return tf.float16

    raise ValueError


# this function doesnt support gradient backprop

def store_value_in_collection(collection_name, to_store, name):
    exist = len([var for var in tf.compat.v1.get_collection(collection_name) if var.op.name == name]) > 0
    if not exist:
        # log.info("Adding {} to collection {}".format(name, collection_name))
        tf.compat.v1.add_to_collection(collection_name, to_store)
    else:
        # log.info("Var {} already in collection {}".format(name, collection_name))
        pass


def copy_and_store_value_in_collection(collection_name, tensor, name, dtype):
    exist = len([var for var in tf.compat.v1.global_variables() if var.op.name == name]) > 0

    if isinstance(tensor, tf.Variable):
        var = tensor
        to_store = tensor
        store_value_in_collection(collection_name, to_store, name)
    else:
        if not exist:
            var = tf.Variable(0, shape=tensor.shape, validate_shape=False, name=name, dtype=dtype)
            to_store = var.value()

            with tf.control_dependencies([to_store]):
                var = var.assign(tensor)

            store_value_in_collection(collection_name, to_store, name)
        else:
            # we have already completed this procedure and we dont want to do it again
            var = [var for var in tf.compat.v1.global_variables() if var.op.name == name][0]

    return var


def compute_sparsity(tensor, name):
    nonzeros = tf.math.count_nonzero(
        input=tensor, axis=None, keepdims=None, dtype=tf.dtypes.float32, name=name + "_nonzeros"
    )

    size = tf.reduce_prod(tf.shape(tensor))

    density = nonzeros / tf.cast(size, tf.float32)
    sparsity = 1.0 - density

    stored_copy = copy_and_store_value_in_collection(collection_name="sparsities", tensor=sparsity, name=name + "_sparsity", dtype=tf.float32)

    with tf.control_dependencies([stored_copy]):
        return tf.identity(tensor)


def get_overflow_rate(to_quantize, max_value):
    mask = tf.logical_or(
        to_quantize > max_value,
        to_quantize <= -max_value
    )

    overflow_rate = tf.reduce_sum(tf.to_int32(mask))

    return overflow_rate

    # Returns true if there is no overflow


def get_if_not_overflow(max_array, min_array, max_value):
    max_overflow = max_array < max_value
    min_overflow = min_array >= -max_value

    return tf.logical_and(max_overflow, min_overflow)


def get_quantization_shift_no_overflow(variable, width, name, quantized_abs_var, conditional_update, update_exp_ph):
    dtype = get_dytpe(variable)

    if _DEBUG_FORCE_EXP is None:
        def get_new_quant():
            # if the variable is already positive, e.g. post RELU, we dont need the abs
            if quantized_abs_var is False:
                abs_variable = tf.math.abs(variable)
            else:
                abs_variable = variable

            max_variable = tf.reduce_max(abs_variable)

            # tf doesnt have log2, so we compute log in base e and then divide to cast it to log2
            log_variable = tf.cond(
                tf.equal(max_variable, 0.0),
                lambda: tf.constant(0.0, dtype),
                lambda: tf.math.log(max_variable) / tf.constant(math.log(2.0), dtype)
            )

            log_rounded = tf.math.ceil(log_variable)
            unsigned_width = width - 1  # for the sign bit
            exp_diff = unsigned_width - log_rounded

            return exp_diff

        #Update only when placeholder requests it - e.g for activations
        if conditional_update:
            stored_exponent = tf.Variable(initial_value=width, trainable=False, dtype=dtype)
            exp_diff_stored = tf.cond(update_exp_ph, lambda: tf.compat.v1.assign(stored_exponent, get_new_quant()), lambda: stored_exponent)
        else:
            exp_diff_stored = get_new_quant()
    else:
        log.warn("DEBUG MODE FOR SHIFT FORCING")
        exp_diff_stored = tf.cast(_DEBUG_FORCE_EXP, dtype)

    exp_diff_used = copy_and_store_value_in_collection("quant_exp", exp_diff_stored, name + "_exp", dtype)

    #The shift must be in format tf.float32 to avoid overflows if the shift is large (e.g. 16 bits)
    shift = 2.0 ** tf.cast(exp_diff_used, tf.float32)

    shift = tf.stop_gradient(shift)

    return shift, exp_diff_used


def quantize_variable(variable, width, store_quantized_collection=False, name=None, tf_training_prec=None, quantized_abs_var=False, return_exp=False, force_exp=None,
                      round_or_floor=None, conditional_update=False, update_quant_ph=None):
    if name is None:
        name = variable.op.name

    if tf_training_prec is None:
        # if no info provided, we keep the same dtype
        log.warning("No training prec for variable {}, keeping input value {}".format(name, variable.dtype))
        tf_training_prec = variable.dtype

    quant_name = name.replace("full_prec", "quantized")

    if force_exp is None:
        quant_shift, exponent = get_quantization_shift_no_overflow(variable, width, name, quantized_abs_var, conditional_update, update_quant_ph)
        force_clip = False
    else:
        exponent = force_exp
        quant_shift = 2 ** force_exp
        force_clip = True

    if variable.dtype not in [tf.float32, tf.float64, "float32_ref", "float64_ref"] and width > 10:
        log.info("Casting variable {} from {} to float32 for shift safety".format(name, variable.dtype))
        variable = tf.cast(variable, tf.float32)
        quant_shift = tf.cast(quant_shift, tf.float32)
    else:
        quant_shift = tf.cast(quant_shift, get_dytpe(variable))
        exponent = tf.cast(exponent, get_dytpe(variable))

    omap = {'Round': 'Identity',
            "Floor": 'Identity'}

    # Round doesnt have a gradient, we force it to identity
    with tf.compat.v1.get_default_graph().gradient_override_map(omap):
        quantizing = variable * quant_shift

        if round_or_floor == "round":
            quantizing = tf.round(quantizing, name=name + "_round")
        elif round_or_floor == "floor":
            quantizing = tf.floor(quantizing, name=name + "_floor")
        else:
            raise ValueError

        if conditional_update is True or force_clip is True:
            max_value = 2.0 ** (width - 1)
            quantizing = tf.clip_by_value(quantizing, -max_value, max_value - 1, name=name + "_clip")

        if store_quantized_collection:
            # we dont use thecopy_and_store_value_in_collection method because it prevents backprop
            quant_quantized_shifted = tf.cast(quantizing, dtype=tf.int32, name=quant_name + "_shifted")
            store_value_in_collection('quant_quantized_shifted', quant_quantized_shifted, quant_name)
            dependencies = [quant_quantized_shifted]
        else:
            dependencies = []

        with tf.control_dependencies(dependencies):
            quantizing = quantizing / quant_shift

        if store_quantized_collection:
            # we dont use thecopy_and_store_value_in_collection method because it prevents backprop

            quant_quantized = tf.identity(quantizing, name=quant_name)
            store_value_in_collection('quant_quantized', quant_quantized, quant_name)
            dependencies = [quant_quantized]
        else:
            dependencies = []

        with tf.control_dependencies(dependencies):
            quantizing = tf.cast(quantizing, tf_training_prec)

    if return_exp:
        return quantizing, exponent
    else:
        return quantizing


def get_quantized_variable(shape, num_bits, initializer, regularizer, constraint, trainable, name, tf_full_prec=None, tf_training_prec=None,
                           return_exp=False, force_exp=None, round_or_floor="round", conditional_update=False, update_quant_ph=None):
    full_name = name + "_full_prec"

    variable = tf.compat.v1.get_variable(
        name=full_name,
        shape=shape,
        dtype=tf_full_prec,
        initializer=initializer,
        regularizer=regularizer,
        trainable=trainable,
        collections=None,
        caching_device=None,
        partitioner=None,
        validate_shape=True,
        use_resource=None,
        custom_getter=None,
        constraint=constraint,
        synchronization=tf.VariableSynchronization.AUTO,
        aggregation=tf.VariableAggregation.NONE
    )

    # so if we have multiple copy of the same variable it doesnt get stored in the collection
    if trainable:
        copy_and_store_value_in_collection("quant_full_prec", variable, full_name, tf_full_prec)

    return quantize_variable(variable, width=num_bits, store_quantized_collection=trainable, name=full_name, tf_training_prec=tf_training_prec, return_exp=return_exp,
                             force_exp=force_exp, round_or_floor=round_or_floor, conditional_update=conditional_update, update_quant_ph=update_quant_ph)


log.warning("Following parameters not implemented:\n"
            "activity_regularizer\n"
            )


def conv2d(
        inputs,
        filters,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format='channels_last',
        dilation_rate=(1, 1),
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None,
        weights_num_bits=None,
        activations_num_bits=None,
        biases_num_bits=None,
        tf_full_prec=None,
        tf_training_prec=None,
        training_ph=None,
        update_activ_quant_ph=None
):
    kernel_shape = list(kernel_size) + [int(inputs.shape[3]), filters]

    kernels = get_quantized_variable(
        shape=kernel_shape,
        num_bits=weights_num_bits,
        initializer=kernel_initializer,
        regularizer=kernel_regularizer,
        constraint=kernel_constraint,
        trainable=trainable,
        name=name + "_kernel",
        tf_full_prec=tf_full_prec,
        tf_training_prec=tf_training_prec,
        round_or_floor="round",
        conditional_update=False,
        update_quant_ph=tf.constant(False, tf.bool)
    )

    kernels = tf.cond(training_ph, lambda: kernels, lambda: compute_sparsity(kernels, name + "_kernel"))

    if data_format == "channels_last":
        data_format = "NHWC"
    elif data_format == "channels_first":
        data_format = "NCHW"

    if _DEBUG_DUMP_OUT_ACTIV: store_value_in_collection("out_activ_dump", tf.identity(inputs, name=name + "_conv_in"), name)

    x = tf.nn.conv2d(
        input=inputs,
        filter=kernels,
        strides=strides,
        padding=padding,
        use_cudnn_on_gpu=True,
        data_format=data_format,
        dilations=dilation_rate,
        name=name
    )

    # if activation == tf.nn.relu:
    #     quantized_abs_var = True
    # else:
    #     quantized_abs_var = False

    quantized_abs_var = False
    # We place a - for the activations_num_bits to reduce the risk of overflow
    x, activ_exp = quantize_variable(x, width=activations_num_bits - ACTIVATION_OVERFLOW_MARGIN, name=name + "_activ", tf_training_prec=tf_training_prec,
                                     quantized_abs_var=quantized_abs_var, return_exp=True, round_or_floor="floor", conditional_update=True, update_quant_ph=update_activ_quant_ph)

    if _DEBUG_DUMP_OUT_ACTIV: store_value_in_collection("out_activ_dump", tf.identity(x * tf.cast((2 ** activ_exp), tf_training_prec), name=name + "_conv_out"), name)

    if use_bias:
        bias_exp = tf.Variable(0, shape=activ_exp.shape, dtype=get_dytpe(activ_exp), trainable=False)
        bias_exp_assign = tf.compat.v1.assign(bias_exp, activ_exp)

        with tf.control_dependencies([bias_exp_assign]):
            x = tf.identity(x)

        biases = get_quantized_variable(
            shape=(filters,),
            num_bits=biases_num_bits,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
            constraint=bias_constraint,
            trainable=trainable,
            name=name + "_biases",
            tf_full_prec=tf_full_prec,
            tf_training_prec=tf_training_prec,
            return_exp=False,
            force_exp=bias_exp,
            round_or_floor="round",
            conditional_update=False,
            update_quant_ph=tf.constant(False, tf.bool)
        )

        x = tf.nn.bias_add(x, biases)
        if _DEBUG_DUMP_OUT_ACTIV: store_value_in_collection("out_activ_dump", tf.identity(x * tf.cast((2 ** activ_exp), tf_training_prec), name=name + "_bias_add"), name)

    output_name = name + "_quant_layer_output"
    if activation is None:
        x = tf.identity(x, name=output_name)
    else:
        x = activation(x, name=output_name)

    x = tf.cond(training_ph, lambda: x, lambda: compute_sparsity(x, output_name))

    if _DEBUG_DUMP_OUT_ACTIV: store_value_in_collection("out_activ_dump", tf.identity(x * tf.cast((2 ** activ_exp), dtype=tf_training_prec), name=name + "_relu"), name)

    return x


def dense(
        inputs,
        units,
        activation=None,
        use_bias=True,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
        name=None,
        reuse=None,
        weights_num_bits=None,
        activations_num_bits=None,
        biases_num_bits=None,
        tf_full_prec=None,
        tf_training_prec=None,
        training_ph=None,
        update_activ_quant_ph=None
):
    if len(inputs.shape) != 2:
        x = tf.layers.flatten(inputs)
    else:
        x = inputs

    kernel_shape = list(x.shape[1:]) + [units]

    kernels = get_quantized_variable(
        shape=kernel_shape,
        num_bits=weights_num_bits,
        initializer=kernel_initializer,
        regularizer=kernel_regularizer,
        constraint=kernel_constraint,
        trainable=trainable,
        name=name + "_kernel",
        tf_full_prec=tf_full_prec,
        tf_training_prec=tf_training_prec,
        round_or_floor="round",
        conditional_update=False,
        update_quant_ph=tf.constant(False, tf.bool)
    )

    kernels = tf.cond(training_ph, lambda: kernels, lambda: compute_sparsity(kernels, name + "_kernel"))

    x = tf.matmul(x, kernels)

    # if activation == tf.nn.relu:
    #     quantized_abs_var = True
    # else:
    #     quantized_abs_var = False

    quantized_abs_var = False

    x, activ_exp = quantize_variable(x, width=activations_num_bits - ACTIVATION_OVERFLOW_MARGIN, name=name + "_activ", tf_training_prec=tf_training_prec,
                                     quantized_abs_var=quantized_abs_var, return_exp=True, round_or_floor="floor", conditional_update=True,
                                     update_quant_ph=update_activ_quant_ph)

    if _DEBUG_DUMP_OUT_ACTIV: store_value_in_collection("out_activ_dump", tf.identity(x * tf.cast((2 ** activ_exp), tf_training_prec), name=name + "_conv_out"), name)

    if use_bias:
        bias_exp = tf.Variable(0, shape=activ_exp.shape, dtype=get_dytpe(activ_exp), trainable=False)
        bias_exp_assign = tf.compat.v1.assign(bias_exp, activ_exp)

        with tf.control_dependencies([bias_exp_assign]):
            x = tf.identity(x)

        biases = get_quantized_variable(
            shape=(units,),
            num_bits=biases_num_bits,
            initializer=bias_initializer,
            regularizer=bias_regularizer,
            constraint=bias_constraint,
            trainable=trainable,
            name=name + "_biases",
            tf_full_prec=tf_full_prec,
            tf_training_prec=tf_training_prec,
            return_exp=False,
            force_exp=bias_exp,
            round_or_floor="round",
            conditional_update=False,
            update_quant_ph=tf.constant(False, tf.bool)
        )

        x = tf.nn.bias_add(x, biases)

    output_name = name + "_quant_layer_output"

    if activation is None:
        x = tf.identity(x, name=output_name)
    else:
        x = activation(x, name=output_name)

    x = tf.cond(training_ph, lambda: x, lambda: compute_sparsity(x, output_name))
    if _DEBUG_DUMP_OUT_ACTIV: store_value_in_collection("out_activ_dump", tf.identity(x * tf.cast((2 ** activ_exp), tf_training_prec), name=name + "_relu"), name)

    return x


def save_collection_as_np(collection_name, session, path, print_data=False):
    variables = tf.compat.v1.get_collection(collection_name)
    log.info("Saving collection {}".format(collection_name))

    if not os.path.exists(path):
        os.makedirs(path)

    # for windwows compatibility
    # illegal_char = ['NUL', '\',''//', ':', '*', '"', '<', '>', '|']

    for tensor in variables:
        tensor_name = tensor.op.name.split("/read")[0]
        tensor_name = tensor_name.split("/")[-1]  # remove the scope

        save_path = pathlib.Path(path + "/" + tensor_name)
        if save_path.is_file():
            os.remove(save_path)

        log.info("Saving variable {} from tensor {}".format(tensor_name, tensor))
        try:
            value = tensor.eval()
            if print_data is True:
                log.info(str(value) + "\n")
            np.save(save_path, value)
        except:
            log.warning("Saving failed")


def eval_and_save_collection_as_np(collection_name, session, path, feed_dict):
    variables = tf.compat.v1.get_collection(collection_name)
    log.info("Saving collection {}".format(collection_name))

    if not os.path.exists(path):
        os.makedirs(path)

    # for windwows compatibility
    # illegal_char = ['NUL', '\',''//', ':', '*', '"', '<', '>', '|']

    for tensor in variables:
        tensor_name = tensor.op.name.split("/read")[0]
        tensor_name = tensor_name.split("/")[-1]  # remove the scope

        save_path = pathlib.Path(path + "/" + tensor_name)
        if save_path.is_file():
            os.remove(save_path)

        log.info("Saving variable {} from tensor {}".format(tensor_name, tensor))
        # try:
        value = tensor.eval(session=session, feed_dict=feed_dict)
        np.save(save_path, value)
        # except:
        #    log.warning("Saving failed")


def save_quantized_variables(path, sess=None, debug_feed_dict=None, save_activations=False):
    if sess is None:
        sess = tf.compat.v1.get_default_session()

    save_collection_as_np('quant_quantized', sess, path + "/quantized/")
    save_collection_as_np('quant_quantized_shifted', sess, path + "/quantized_shifted/")
    save_collection_as_np('quant_exp', sess, path + "/exp/", print_data=True)
    save_collection_as_np('quant_full_prec', sess, path + "/full_prec/")
    save_collection_as_np('sparsities', sess, path + "/sparsity_reports/", print_data=True)

    if save_activations is True:
        eval_and_save_collection_as_np("out_activ_dump", sess, path + r"/out_activ_dump/", debug_feed_dict)
