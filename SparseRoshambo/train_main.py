import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
import os
import numpy as np
from dataset import Dataset
from network import *
from loss_optimizer import *
from accuracy import *
from time import time
from config import *
import quantization as sdk

run_start_time = time()

run_config = tf.compat.v1.ConfigProto()
run_config.gpu_options.allow_growth = True
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)

with tf.compat.v1.Session(config=run_config) as sess:
    train_dataset = Dataset(paths=train_data_path, shape=shape, batch_size=train_batch_size, normalize=normalize, shift=shift, data_format=data_format, dtype=tf_training_prec,
                            num_classes=num_classes, execute_shuffle=True)
    test_dataset = Dataset(paths=test_data_path, shape=shape, batch_size=test_batch_size, normalize=normalize, shift=shift, data_format=data_format, dtype=tf_training_prec,
                           num_classes=num_classes, execute_shuffle=False)

    features_ph, labels_ph = train_dataset.get_placeholders()
    training_ph = tf.compat.v1.placeholder(tf.bool)
    update_activ_quant_ph = tf.compat.v1.placeholder(tf.bool)

    network = get_tf_network(features_ph, network_structure, training_ph, update_activ_quant_ph, training=True)

    loss = get_loss(network, labels_ph)

    train_optimizer = get_optimizer_op(loss, learning_rate)

    train_accuracy, train_update_op, train_accuracy_initializer = get_accuracy(network, labels_ph, name="train_accuracy")
    test_accuracy, test_update_op, test_accuracy_initializer = get_accuracy(network, labels_ph, name="test_accuracy")

    train_initializer, train_iterator = train_dataset.get_iterator()
    test_initializer, test_iterator = test_dataset.get_iterator()

    log.info("Initializing variables...")
    sess.run(tf.compat.v1.global_variables_initializer())

    train_summary_op = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar("train_accuracy", train_accuracy), tf.compat.v1.summary.scalar("train_loss", loss)])
    test_summary_op = tf.compat.v1.summary.merge([tf.compat.v1.summary.scalar("test_accuracy", test_accuracy), tf.compat.v1.summary.scalar("test_loss", loss)])

    # save only trainable variables
    log.info("Preparing saver...")
    saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.trainable_variables(), max_to_keep=10, filename="sparseRoshamboModel")
    summary_writer = tf.compat.v1.summary.FileWriter(model_save_path, sess.graph, flush_secs=30, max_queue=1000)

    log.info("Starting training...")
    test_ops = [loss, test_update_op, test_summary_op]
    test_ops_iterator = test_ops + [test_iterator]
    test_ops_initializer = test_ops + [train_initializer]

    train_ops = [train_optimizer, loss, train_update_op, train_summary_op]
    train_ops_iterator = train_ops + [train_iterator]
    train_ops_initializer = train_ops + [test_initializer]

    sess.run([train_initializer, test_initializer])


    def save_state(epoch_idx):

        # Save the variables to disk.
        with tf.control_dependencies(summary_writer.flush()):
            #prepare paths and
            save_path_full = model_save_path + "epoch_{}\\".format(epoch_idx)
            os.makedirs(save_path_full, exist_ok=True)
            save_path = saver.save(sess, save_path_full + "network")

            #Save PB file
            tf.io.write_graph(tf.compat.v1.get_default_graph(), save_path_full, 'network.pb', as_text=False)

            #prepare dict for debug on accelerator
            test_image = np.loadtxt(r"D:\DL\datasets\Roshambo\tf_train\scissors-enea-back-0003468.txt").reshape([1, 64, 64, 1]).astype(np.float32) / 256
            test_label = np.zeros(shape=[1, 4])
            test_label[0, 1] = 1
            activ_dump_feed_dict = {features_ph: test_image, labels_ph: test_label, training_ph: False, update_activ_quant_ph: False}

            # save custom debug variables
            sdk.save_quantized_variables(save_path_full, sess=sess, debug_feed_dict=activ_dump_feed_dict, save_activations=True)



            # checkpoint = tf.train.Checkpoint(optimizer=train_optimizer, model=network)
            # tf.saved_model.save(checkpoint, save_path_full + r"\\checkpoint\\")

            log.info("Model saved in path: %s" % save_path)


    log.info("Starting epoch iteration...")
    for epoch_idx in range(num_epochs):
        start = time()

        sess.run([tf.compat.v1.variables_initializer(train_accuracy_initializer), tf.compat.v1.variables_initializer(test_accuracy_initializer)])

        features, labels = sess.run(train_iterator)

        update_activ_quant = True
        while True:

            try:

                _, train_loss_value, train_accuracy, summary, (features, labels) = sess.run(train_ops_iterator,
                                                                                            feed_dict={features_ph: features,
                                                                                                       labels_ph: labels,
                                                                                                       training_ph: True,
                                                                                                       update_activ_quant_ph: update_activ_quant},

                                                                                            options=run_opts)
                if epoch_idx > 0:
                    update_activ_quant = False

            except tf.errors.OutOfRangeError:
                # we ran also test_initializer

                _, train_loss_value, train_accuracy, summary, _ = sess.run(train_ops_initializer,
                                                                           feed_dict={features_ph: features,
                                                                                      labels_ph: labels,
                                                                                      training_ph: True,
                                                                                      update_activ_quant_ph: update_activ_quant},
                                                                           options=run_opts)
                break

        summary_writer.add_summary(summary, global_step=epoch_idx)
        #summary = tf.compat.v1.summary.scalar("test_accuracy", test_accuracy)

        features, labels = sess.run(test_iterator)

        for test_batch_idx in range(num_test_batch):
            if test_batch_idx != num_test_batch - 1:

                test_loss_value, test_accuracy, summary, (features, labels) = sess.run(test_ops_iterator, feed_dict={features_ph: features,
                                                                                                                     labels_ph: labels,
                                                                                                                     training_ph: False,
                                                                                                                     update_activ_quant_ph: update_activ_quant},
                                                                                       options=run_opts)

            else:
                # we run also train initializer
                test_loss_value, test_accuracy, summary, _ = sess.run(test_ops_initializer, feed_dict={features_ph: features,
                                                                                                       labels_ph: labels,
                                                                                                       training_ph: False,
                                                                                                       update_activ_quant_ph: update_activ_quant},
                                                                      options=run_opts)

        summary_writer.add_summary(summary, global_step=epoch_idx)

        end = time()
        runtime = end - start
        log.info("Epoch {} train loss: {:.5f} - train_accuracy: {:.5f} - test loss: {:.5f} test accuracy: {:.5f} ({:.5f} s)".format(epoch_idx, train_loss_value, train_accuracy,
                                                                                                                                    test_loss_value, test_accuracy, runtime))

        if epoch_idx % epoch_save_period == 0:
            save_state(epoch_idx)

    end = time()
    runtime = end - start

    log.info("\n\nTest loss: {:.4f} test accuracy: {:.5f} ({:.5f} s)\n\n".format(test_loss_value, test_accuracy, runtime))
    log.info("Saving final state...")
    save_state("final")

    total_time = time() - run_start_time
    log.info("Total runtime: {:.5f} s".format(total_time))
