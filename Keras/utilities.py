from tensorflow.keras.models import Model
from tensorflow import keras
import networks, datasets
import inspect
import logging
import datetime
from config import config_dict
import tensorflow as tf

log = logging.getLogger()


def get_run_unique_id(config_dict):
    import random

    unique_id = ("%08x" % random.randint(0, 2 ** 32 - 1)).upper()
    date = datetime.datetime.now().strftime("%d-%m-%Y__%H-%M-%S")

    run_id = date
    for key, value in config_dict.items():
        if isinstance(value, dict) is False:
            value = str(value)

            for illegal_char in ["'", '"', "{", "}", ":", ","]:
                value = value.replace(illegal_char, "")

            value = value.replace(" ", "_")

            add = "__" + str(key) + "_" + value
            run_id = run_id + add
        else:
            for key, value in value.items():

                value = str(value)

                for illegal_char in ["'", '"', "{", "}", ":", ","]:
                    value = value.replace(illegal_char, "")

                value = value.replace(" ", "_")

                if value == "continue_train":  # to avoid have a full path
                    add = "__" + str(key)
                else:
                    add = "__" + str(key) + "_" + value

                run_id = run_id + add

    run_id = run_id + "__ID-" + unique_id
    log.info("Run ID: {}".format(run_id))
    return run_id


def prepare_save_folders_and_logger(ws_dict, run_id):
    import os
    import main
    from shutil import copytree
    from pathlib import Path

    # Prepare model model saving directory.
    save_dir = ws_dict["model_save_dir"]

    if not os.path.isdir(save_dir):
        log.info("Creating directory {}".format(save_dir))
        os.makedirs(save_dir)

    run_dir = save_dir + r"\\" + run_id
    log.info("Creating directory {}".format(run_dir))
    os.makedirs(run_dir)
    os.makedirs(run_dir + r"\save")
    os.makedirs(run_dir + r"\tensorboard")

    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-10.10s] [%(levelname)-4.4s] %(message)s")

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(logFormatter)

    file_handler = logging.FileHandler("{0}/{1}.log".format(run_dir, "run"))
    file_handler.setFormatter(logFormatter)

    log.addHandler(file_handler)
    log.addHandler(stderr_handler)

    log.info("Logging file handler created")

    src_copy_dir = run_dir + r"\\" + "src" + r"\\"
    src_dir = os.path.dirname(os.path.realpath(main.__file__)) + r"\\"

    log.info("Creating source code backup {}".format(src_copy_dir))
    src_dir = Path(src_dir)
    src_copy_dir = Path(src_copy_dir)

    copytree(src_dir, src_copy_dir)

    return run_dir


def get_dataset_model_and_pruning(config_dict, ws_dict):
    def default_pruning_policy(pruning_policy):
        if pruning_policy["mode"] == "fixed":
            return pruning_policy["target"]
        else:
            raise AttributeError

    # Select Dataset Class
    with tf.name_scope("dataset"):
        with tf.device('/cpu:0'):
            for name, obj in inspect.getmembers(datasets):
                if name.lower() == config_dict["dataset"].lower():
                    log.info("Selected dataset: {}".format(name))
                    dataset = obj()
                    inputs = dataset.get_input_ph()
                    break
            else:
                raise AttributeError

    # Select Network Class and Pruning Policy
    with tf.name_scope("model"):
        pruning_policy = None
        network = None
        for name, obj in inspect.getmembers(networks):
            if name.lower() == config_dict["network"].lower():
                log.info("Selected network: {}".format(name))

                try:
                    network_kwargs = config_dict["network_kwargs"]
                except KeyError:
                    network_kwargs = dict()

                network = obj(input_tensor=inputs,
                              include_top=ws_dict["include_top"],
                              classes=dataset.num_classes,
                              **network_kwargs)
                if isinstance(network, Model):
                    model = network
                else:
                    model = Model(inputs=inputs, outputs=network)
            elif name.lower() == (config_dict["network"] + "PruningPolicy").lower():
                pruning_policy = obj(config_dict["pruning_policy"])
                log.info("Custom Policy Found")

        if pruning_policy is None:
            log.info("****No Pruning Policy Found****")
            pruning_policy = default_pruning_policy(config_dict["pruning_policy"])

    if network is None:
        raise AttributeError

    if config_dict["quantize"] is False:
        log.info("Quantization disabled, pruning disabled")

    return dataset, model, pruning_policy


class FitLogger(keras.callbacks.Callback):
    def __init__(self, frequency):
        self.frequency = frequency  # print loss & acc every n epochs
        self.tot_num_train_batches = -1
        self.tot_num_test_batches = -1
        self.epoch = 1
        self.start_time = datetime.datetime.now()

    def on_train_batch_end(self, batch, logs=None):
        self.tot_num_train_batches = max(self.tot_num_train_batches, batch)

        if batch == 0:
            self.start_time = datetime.datetime.now()

        if (batch % self.frequency == 0 and batch != 0) or batch == 1:

            curr_loss = logs.get('loss')
            curr_acc = logs.get('accuracy') * 100

            current_time = datetime.datetime.now()
            lapsed_time = (current_time - self.start_time).total_seconds()
            self.start_time = datetime.datetime.now()

            if batch == 1:
                divider = 1
            else:
                divider = self.frequency

            time_batch_sec = lapsed_time / divider
            time_batch_ms = int(time_batch_sec * 1000)
            time_image_ms = time_batch_ms / config_dict["batch_size"]
            eta = time_batch_sec * (self.tot_num_train_batches - batch)

            if eta > 180:
                eta = eta / 60
                eta = "{}m".format(int(eta))
            else:
                eta = "{: .2f}s".format(eta)

            if self.epoch != 0:
                log.info(
                    "[TRAIN] Epoch {} - {}/{} - ETA: {} ({}ms/batch {:.2f}ms/image) - loss: {:.4f} - accuracy: {:.2f}%".
                        format(
                        self.epoch,
                        batch,
                        self.tot_num_train_batches,
                        eta,
                        time_batch_ms,
                        time_image_ms,
                        curr_loss,
                        curr_acc))
            else:
                log.info(
                    "[TRAIN] Epoch {} - {}/? - {}ms/batch {:.2f}ms/image - loss: {:.4f} - accuracy: {:.2f}%".format(
                        self.epoch,
                        batch,
                        time_batch_ms,
                        time_image_ms,
                        curr_loss,
                        curr_acc))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.start_time = datetime.datetime.now()

    def on_test_batch_end(self, batch, logs=None):
        self.tot_num_test_batches = max(self.tot_num_test_batches, batch)

        if batch == 0:
            self.start_time = datetime.datetime.now()

        if batch % self.frequency == 0 and batch != 0:
            curr_loss = logs.get('loss')
            curr_acc = logs.get('accuracy') * 100

            current_time = datetime.datetime.now()
            lapsed_time = (current_time - self.start_time).total_seconds()
            self.start_time = datetime.datetime.now()

            time_batch_sec = lapsed_time / self.frequency
            time_batch_ms = int(time_batch_sec * 1000)
            time_image_ms = time_batch_ms / config_dict["batch_size"]
            eta = time_batch_sec * (self.tot_num_test_batches - batch)

            if eta > 180:
                eta = eta / 60
                eta = "{}m".format(int(eta))
            else:
                eta = "{: .2f}s".format(eta)

            if self.epoch != 0:
                log.info(
                    "[TEST] Epoch {} - {}/{} - ETA: {} ({}ms/batch {:.2f}ms/image) - loss: {:.4f} - accuracy: {:.2f}%".
                        format(
                        self.epoch,
                        batch,
                        self.tot_num_test_batches,
                        eta,
                        time_batch_ms,
                        time_image_ms,
                        curr_loss,
                        curr_acc))
            else:
                log.info("[TEST] Epoch {} - {}/? - {}ms/batch {:.2f}ms/image - loss: {:.4f} - accuracy: {:.2f}%".format(
                    self.epoch,
                    batch,
                    time_batch_ms,
                    time_image_ms,
                    curr_loss,
                    curr_acc))

    def on_predict_batch_end(self, batch, logs=None):
        pass


# def get_flops():
#     run_meta = tf.RunMetadata()
#     opts = tf.profiler.ProfileOptionBuilder.float_operation()
#
#     # We use the Keras session graph in the call to the profiler.
#     flops = tf.profiler.profile(graph=K.get_session().graph,
#                                 run_meta=run_meta, cmd='op', options=opts)
#
#     return flops.total_float_ops  # Prints the "flops" of the model.


# Note it works only if the ordering of the layer returned by glob is equal to the ordering of the layers in model.layers
def load_from_numpy(model, path):
    import glob
    import numpy as np

    shifts_weights = np.load(path + r"\\" + "shift_per_layer_kernel.npy")
    shifts_biases = np.load(path + r"\\" + "shift_per_layer_bias.npy")
    shifts_outputs = np.load(path + r"\\" + "shift_per_layer_activation.npy")

    layer_index = 0
    total_fixed = 0
    for layer in model.layers:
        name = layer.name.replace("quant_", "")

        file_weights = glob.glob(path + r"\\" + name + "*kernel*")
        file_bias = glob.glob(path + r"\\" + name + "*bias*")

        if len(file_weights) == 0 and len(file_bias) == 0:
            log.info("***No weight found for layer {}***".format(name))
        else:
            log.info("Loading quantized weights for layer {}".format(name))

            if len(file_weights) != 1 or len(file_bias) != 1:
                raise ValueError

            weights = np.load(file_weights[0]).astype(np.float32) / (2.0 ** float(shifts_weights[layer_index]))
            biases = np.load(file_bias[0]).astype(np.float32) / (2.0 ** float(shifts_biases[layer_index]))

            current_weights = layer.get_weights()

            # TODO fix flag
            if True:
                for kernel_idx in range(weights.shape[-1]):
                    old_weights = weights[..., kernel_idx]
                    is_all_zero = np.all((old_weights == 0))

                    if bool(is_all_zero):
                        if kernel_idx == 0:
                            copy_index = weights.shape[-1] - 1
                            copy_index2 = 1
                        else:
                            copy_index = kernel_idx - 1
                            copy_index2 = int((kernel_idx + 1) % weights.shape[-1])

                        weights[..., kernel_idx] = weights[..., copy_index] + weights[..., copy_index2]

                        if np.all((weights[..., kernel_idx] == 0)):
                            raise Exception

                        log.info("Fixing empty kernel {} with {}".format(kernel_idx, copy_index))
                        total_fixed = total_fixed + 1

            # todo: terribly hardcoded
            if len(current_weights) != 17:

                kernel_exp = np.ones(shape=()) * shifts_weights[layer_index]
                biases_exp = np.ones(shape=()) * shifts_biases[layer_index]
                output_exp = np.ones(shape=()) * shifts_outputs[layer_index]
                output_exp_memory = np.ones(shape=list(current_weights[11].shape)) * shifts_outputs[layer_index]

                load_list = [weights,
                             biases,
                             current_weights[2],  # optimizer step
                             kernel_exp,  # kernel exp
                             current_weights[4],  # kernel sparsity
                             weights,  # kernel stored tensor
                             biases_exp,  # bias exp
                             current_weights[7],  # bias sparsity
                             biases,  # bias stored tensor
                             output_exp,  # output exp
                             current_weights[10],  # output sparsity
                             output_exp_memory,  # output exp memory
                             current_weights[12],  # output exp memory ptr
                             current_weights[13],  # output stored tensor
                             ]
            else:

                kernel_exp = np.ones(shape=()) * shifts_weights[layer_index]
                biases_exp = np.ones(shape=()) * shifts_biases[layer_index]
                output_exp = np.ones(shape=()) * shifts_outputs[layer_index]
                output_exp_memory = np.ones(shape=list(current_weights[14].shape)) * shifts_outputs[layer_index]
                mask = np.ones(shape=list(weights.shape))
                pruning_step = np.zeros(shape=())
                threshold = np.zeros(shape=())

                load_list = [weights,
                             biases,
                             current_weights[2],  # optimizer step
                             kernel_exp,  # kernel exp
                             current_weights[4],  # kernel sparsity
                             weights,  # kernel stored tensor
                             mask,
                             threshold,
                             pruning_step,
                             biases_exp,  # bias exp
                             current_weights[10],  # bias sparsity
                             biases,  # bias stored tensor
                             output_exp,  # output exp
                             current_weights[13],  # output sparsity
                             output_exp_memory,  # output exp memory
                             current_weights[15],  # output exp memory ptr
                             current_weights[16],  # output stored tensor
                             ]

            model.get_layer(layer.name).set_weights(load_list)

            comparison = weights == model.get_layer(layer.name).get_weights()[0]
            equal_arrays = comparison.all()
            if bool(equal_arrays) is not True:
                raise ValueError

            comparison = biases == model.get_layer(layer.name).get_weights()[1]
            equal_arrays = comparison.all()
            if bool(equal_arrays) is not True:
                raise ValueError

            layer_index = layer_index + 1
    log.info("Loading quantized weights done, fixed kernels: {}".format(total_fixed))
    return model
