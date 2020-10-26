import tensorflow as tf
import quantization as ssdk
import logging
import config

log = logging.getLogger()


def get_tf_network(input, network_structure, training_ph, update_activ_quant_ph, training):
    with tf.compat.v1.variable_scope("network", auxiliary_name_scope=False, reuse=tf.compat.v1.AUTO_REUSE):
        x = input

        if network_structure.use_fixed_point is True:
            conv2d = ssdk.conv2d
            dense = ssdk.dense
        else:
            conv2d = tf.layers.conv2d
            dense = tf.layers.dense

        for layer in network_structure.layers:
            if training:
                log.info("Creating {} layer {} - input shape: {}".format(layer.type, layer.name, x.shape))

            if layer.type == "conv":
                x = tf.layers.dropout(
                    inputs=x,
                    rate=0.1,
                    noise_shape=None,
                    seed=None,
                    training=training_ph,
                    name=layer.name + "_dropout",
                )

                conv_params = {
                    "inputs": x,
                    "filters": layer.filters,
                    "kernel_size": layer.kernel_size,
                    "strides": layer.strides,
                    "padding": layer.padding,
                    "data_format": network_structure.data_format,
                    "dilation_rate": (1, 1),
                    "activation": layer.activation,
                    "use_bias": True,
                    "kernel_initializer": tf.glorot_normal_initializer(),
                    "bias_initializer": tf.zeros_initializer(),
                    "kernel_regularizer": None,
                    "bias_regularizer": None,
                    "activity_regularizer": None,
                    "kernel_constraint": None,
                    "bias_constraint": None,
                    "trainable": True,
                    "name": layer.name,
                    "reuse": tf.compat.v1.AUTO_REUSE}

                if network_structure.use_fixed_point is True:
                    conv_params["weights_num_bits"] = layer.fixed_point_weights_num_bits
                    conv_params["activations_num_bits"] = layer.fixed_point_activations_num_bits
                    conv_params["biases_num_bits"] = layer.fixed_point_biases_num_bits
                    conv_params["tf_full_prec"] = config.tf_full_prec
                    conv_params["tf_training_prec"] = config.tf_training_prec
                    conv_params["training_ph"] = training_ph
                    conv_params["update_activ_quant_ph"] = update_activ_quant_ph


                x = conv2d(**conv_params)

            elif layer.type == "FC":
                if training:
                    log.info("Flattening fc input - shape: {}".format(x.shape))

                x = tf.layers.dropout(
                    inputs=x,
                    rate=0.05,
                    noise_shape=None,
                    seed=None,
                    training=training_ph,
                    name=layer.name + "_dropout",
                )

                # TODO missing fixed point
                fc_params = {
                    "inputs": x,
                    "units": layer.filters,
                    "activation": layer.activation,
                    "use_bias": True,
                    "kernel_initializer": None,
                    "bias_initializer": tf.zeros_initializer(),
                    "kernel_regularizer": None,
                    "bias_regularizer": None,
                    "activity_regularizer": None,
                    "kernel_constraint": None,
                    "bias_constraint": None,
                    "trainable": True,
                    "name": layer.name,
                    "reuse": tf.compat.v1.AUTO_REUSE}

                if network_structure.use_fixed_point is True:
                    fc_params["weights_num_bits"] = layer.fixed_point_weights_num_bits
                    fc_params["activations_num_bits"] = layer.fixed_point_activations_num_bits
                    fc_params["biases_num_bits"] = layer.fixed_point_biases_num_bits
                    fc_params["tf_full_prec"] = config.tf_full_prec
                    fc_params["tf_training_prec"] = config.tf_training_prec
                    fc_params["training_ph"] = training_ph
                    fc_params["update_activ_quant_ph"] = update_activ_quant_ph


                x = dense(**fc_params)


            elif layer.type == "pool":

                x = tf.nn.pool(
                    input=x,
                    window_shape=layer.window_shape,
                    pooling_type=layer.pooling_type,
                    padding=layer.padding,
                    strides=layer.strides,
                    name=layer.name
                )

            else:
                log.error("Unsupported layer {}".format(layer.type))

        if training:
            log.info("Layer ourput shape: {}".format(x.shape))

        return x
