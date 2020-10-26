import tensorflow as tf
import numpy as np
import collections
import os

with tf.device("/cpu:0"):
    os.chdir("..")
    curr_path = os.path.abspath(os.curdir)

    save_dir = "D:\DL\models\FrameRoshambo\\"
    model_name = 'network.ckpt'
    checkpoint = save_dir + model_name
    metagraph = checkpoint + '.meta'
    export_folder = save_dir + '/exported_weights/'

    with tf.compat.v1.Session() as sess:
        # Import model (meta graph) and load the checkpoint
        tf.compat.v1.train.import_meta_graph(metagraph, clear_devices=True)
        graph = tf.compat.v1.get_default_graph()
        all_variables = tf.compat.v1.get_collection("variables")
        # Remove beta and Adam variables or, due to a mismatch between graph and checkpoint, it won't work
        saver = tf.compat.v1.train.Saver(var_list=[v for v in all_variables if "beta" not in v.name and 'Adam' not in v.name])
        saver.restore(sess, checkpoint)

        # Create a dict of dicts with only the weight/bias layers
        variables_dict = collections.OrderedDict()
        layer_idx = 0
        for var in all_variables:
            var_name = str(var.name)
            # This code assumes the layers will be read in order from the for cycle and bias values follow weights
            # todo make it more general

            if 'Adam' not in str(var_name) and 'beta' not in str(var_name):
                print(var_name)
                if "exp" not in str(var_name) and "activ" not in str(var_name):
                    if 'bias' not in str(var_name):
                        layer_idx += 1
                        layer_name = 'layer_' + str(layer_idx)
                        variables_dict[layer_name] = collections.OrderedDict()
                        values = graph.get_tensor_by_name(var_name)
                        variables_dict[layer_name]['name'] = var_name
                        variables_dict[layer_name]['shape'] = var.shape
                        variables_dict[layer_name]['weight_values'] = sess.run(values)
                    else:
                        values = graph.get_tensor_by_name(var_name)
                        variables_dict[layer_name]['bias_values'] = sess.run(values)

        # Make eventual modifications to values (e.g batch norm), then save them
        for layer_name in variables_dict:
            weights = variables_dict[layer_name]['weight_values']
            biases  = variables_dict[layer_name]['bias_values']
            np.save(export_folder + layer_name + '_weights', weights)
            np.save(export_folder + layer_name + '_biases', biases)

