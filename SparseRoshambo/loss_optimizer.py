import tensorflow as tf
import logging
log = logging.getLogger()

def get_loss(network_output, labels):


    # if network_output.dtype != tf.float64:
    #     log.info("Casting network output to 64 bits for gradient quality in loss")
    #     network_output = tf.cast(network_output, tf.float64)

    #remove the batch dimension to allow correct comparison
    assert network_output.shape[1:] == labels.shape[1:], "{} vs {}".format(network_output.shape, labels.shape)

    no_grad_labels = tf.stop_gradient(labels)




    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=no_grad_labels,
        logits=network_output,
        axis=-1,
        name="loss"
    ))
    return loss



def get_optimizer_op(loss, learning_rate):



    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)


    #optimize_op = loss_scale_optimizer.minimize(loss)
    optimize_op = optimizer.minimize(loss)

    return optimize_op

