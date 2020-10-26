import tensorflow as tf


def get_accuracy(predictions, labels, name="accuracy_metrics"):

    accuracy, update_op = tf.compat.v1.metrics.accuracy(
        labels=tf.argmax(labels,1),
        predictions=tf.argmax(predictions,1),
        name=name
    )

    initializer = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.LOCAL_VARIABLES, scope=name)

    return accuracy, update_op, initializer
