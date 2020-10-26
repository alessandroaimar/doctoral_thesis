from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from collections import OrderedDict
import pprint

log = logging.getLogger()


def get_weight_sparsity(model):
    all_stats = list()
    empty_kernels = 0
    total_kernels = 0

    for layer in model.layers:

        layer_stats = OrderedDict()
        store = False
        for variable in layer.weights:

            if "kernel_sparsity:0" in variable.name:
                layer_stats["stored_kernel_sparsity"] = float(variable.numpy())
                store = True
            elif "kernel:0" in variable.name:
                layer_stats["kernels_size"] = variable.numpy().size
            elif "kernel_stored_tensor:0" in variable.name:
                variable = variable.numpy()
                total_kernels = total_kernels + variable.shape[-1]

                for out_ch in range(variable.shape[-1]):

                    if len(variable.shape) == 2:
                        kernel = variable[:, out_ch]
                    elif len(variable.shape) == 4:
                        kernel = variable[:, :, :, out_ch]
                    else:
                        log.info("Error in array size {}".format(variable.shape))
                        break

                    non_zero = np.count_nonzero(kernel)
                    if non_zero == 0:
                        empty_kernels = empty_kernels + 1

        if store:
            all_stats.append(layer_stats)

    if len(all_stats) > 0:
        total_weights = 0
        weighted_sum_stored = 0
        for stat in all_stats:
            total_weights = total_weights + stat["kernels_size"]
            weighted_sum_stored = weighted_sum_stored + stat["stored_kernel_sparsity"] * stat["kernels_size"]

        weighted_sparsity_stored = 100.0 * weighted_sum_stored / total_weights

        if total_kernels != 0:
            empty_kernel_perc = 100.0 * empty_kernels / total_kernels
            log.info("Network kernels sparsity: {:.2f}% - Found {} ({:.2f} %) empty kernels".format(weighted_sparsity_stored,
                                                                                                    empty_kernels, empty_kernel_perc))
        else:
            log.info("Network kernels sparsity: {:.2f}% - No stored tensor found for emptiness check".format(weighted_sparsity_stored))
    else:
        raise Exception("No data to extract weight sparsity from")


def get_layer_sparsity_stats(model, layer_cap=-1):
    all_stats = []
    layer_cnt = 0

    for layer in model.layers:
        if isinstance(layer, QuantizeWrapper):
            num_entries = None
            layer_sparsity = None
            output_pooled = False
            for variable in layer.variables:
                if "output_stored_tensor" in variable.name:
                    shape = variable.shape

                    num_entries = np.prod(shape)  # reduce(operator.mul, shape, 1)

                    if isinstance(layer.outbound_nodes[0].outbound_layer, tf.keras.layers.MaxPooling2D):
                        output_pooled = True

                elif "output_sparsity" in variable.name:
                    layer_sparsity = variable.numpy()

            stats = OrderedDict([("sparsity", layer_sparsity), ("num_entries", num_entries), ("output_pooled", output_pooled)])
            all_stats.append(stats)

            layer_cnt = layer_cnt + 1

    if layer_cap == 0:
        return all_stats
    elif layer_cap > 0:
        if layer_cap < len(all_stats):
            return all_stats[:layer_cap]
        else:
            return all_stats
    else:
        return all_stats[:layer_cap]


def weighted_stats(stats):
    total_value = 0
    total_sparsity = 0
    for stat in stats:
        if stat["output_pooled"] is True:
            num_entries = stat["num_entries"] / 4  # TODO make more generalized
            sparsity = stat["sparsity"] / 2
        else:
            num_entries = stat["num_entries"]
            sparsity = stat["sparsity"]

        total_value = total_value + num_entries
        total_sparsity = total_sparsity + sparsity * num_entries

    weighted_sparsity = total_sparsity / total_value
    return weighted_sparsity


def save_image(image, path, name, save_rescale, save_shift):
    numpy_full_path = path + name
    txt_full_path = path + name + ".txt"

    image = image * save_rescale
    if save_shift != 0:
        image = image + save_shift
    image = image.astype(np.int32)

    np.save(arr=image, file=numpy_full_path)

    file_writer = open(txt_full_path, "w")

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            for ch in range(image.shape[2]):
                value = str(image[row, col, ch])
                file_writer.write(value)
                file_writer.write("\n")
    file_writer.flush()
    file_writer.close()
    return txt_full_path


"""
layer_cap = 0: all layers. Layer_cap = -N: all layers except last N. Layer_cap = N: all layers up to Nth
"""


def activ_sparsity_stats(model, dataset, savedir, num_stored_input=5, max_images=25000, layer_cap=-1, verbose=True, save_rescale=256,
                         save_shift=0):
    train_dataset = dataset.unbatch().batch(1)

    top_sparsity_list = [OrderedDict([("sparsity", 0.0), ("image", None)]) for _ in range(num_stored_input)]
    min_sparsity_list = [OrderedDict([("sparsity", 1.0), ("image", None)]) for _ in range(num_stored_input)]

    total_sparsity = 0
    average_sparsity = -1
    image_cnt = 0

    for image_idx, image_label in enumerate(train_dataset.as_numpy_iterator()):
        model.predict(image_label, verbose=0)
        stats = get_layer_sparsity_stats(model, layer_cap)

        if len(stats) == 0:
            raise Exception("No data to extract activation sparsity from")

        weighted_sparsity = weighted_stats(stats)
        total_sparsity = total_sparsity + weighted_sparsity
        average_sparsity = total_sparsity / (image_idx + 1)

        old_top_sparsity = top_sparsity_list[-1]["sparsity"]  # smallest sparsity
        old_min_sparsity = min_sparsity_list[0]["sparsity"]  # biggest sparsity

        if weighted_sparsity > old_top_sparsity:
            image_cnt = 0
            new_entry = OrderedDict([("sparsity", weighted_sparsity), ("image_label", image_label), ("stats", stats)])
            top_sparsity_list[-1] = new_entry
            top_sparsity_list = sorted(top_sparsity_list, key=lambda i: i['sparsity'], reverse=True)  # smallest at end, biggest at 0
            if verbose:
                log.info(
                    "Added top sparsity image {} with sparsity {:.3f} (removed sparsity {:.3f}) - average: {:.3f}".format(
                        image_idx,
                        weighted_sparsity,
                        old_top_sparsity,
                        average_sparsity))

        if weighted_sparsity < old_min_sparsity:
            image_cnt = 0
            new_entry = OrderedDict([("sparsity", weighted_sparsity), ("image_label", image_label), ("stats", stats)])
            min_sparsity_list[0] = new_entry
            min_sparsity_list = sorted(min_sparsity_list, key=lambda i: i['sparsity'], reverse=True)  # smallest at end, biggest at 0
            if verbose:
                log.info(
                    "Added min sparsity image {} with sparsity {:.3f} (removed sparsity {:.3f}) - average: {:.3f}".format(
                        image_idx,
                        weighted_sparsity,
                        old_min_sparsity,
                        average_sparsity))

        if image_cnt == max_images:
            if verbose: log.info("{} images without update, assuming maximum condition reached...".format(max_images))
            break

        image_cnt = image_cnt + 1

    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir + "\max", exist_ok=True)
    os.makedirs(savedir + "\min", exist_ok=True)

    log.info("""Feature Maps Sparsity Statistics:
        Maximum sparsity: {:.3f}
        Minimum Sparsity: {:.3f} 
        Average Sparsity: {:.3f} 
        """.format(top_sparsity_list[0]["sparsity"], min_sparsity_list[0]["sparsity"], average_sparsity))

    if verbose: log.info("Saving images as int with rescaling {} and shift {}".format(save_rescale, save_shift))

    run_txt_file = open(savedir + r"\list.txt", "a")
    sparsity_detail_txt_file = open(savedir + r"\details.txt", "a")

    for stat in top_sparsity_list:
        name_dot = "image_sparsity_{:.3f}".format(100.0 * float(stat["sparsity"]))
        name = name_dot.replace(".", "_")

        full_path = savedir + r"\max\\"

        txt_path = save_image(stat["image_label"][0][0], full_path, name, save_rescale, save_shift)

        run_txt_file.write(txt_path)
        run_txt_file.write("\n")
        run_txt_file.write(str(np.argmax(stat["image_label"][1]).astype(np.int32)))
        run_txt_file.write("\n")

        sparsity_detail_txt_file.write(name)
        sparsity_detail_txt_file.write(" - weighted sparsity: ")
        sparsity_detail_txt_file.write(str(stat["sparsity"]))
        sparsity_detail_txt_file.write("\n")
        detailed_stats = pprint.pformat(stat["stats"], indent=3, width=130, compact=True)  # str(stat["stats"]).replace("[OrderedDict([(")

        for layer_idx in range(len(stat["stats"])):
            detailed_stats = detailed_stats.replace("OrderedDict([(", "Layer {}: ".format(layer_idx), 1)

        detailed_stats = detailed_stats.replace(")]=,", "")
        detailed_stats = detailed_stats.replace("), (", ", ")
        detailed_stats = detailed_stats.replace("',", "':")
        detailed_stats = detailed_stats.replace(")]),", "")
        detailed_stats = detailed_stats.replace("[", " ")
        detailed_stats = detailed_stats.replace(")])]", "")
        sparsity_detail_txt_file.write(detailed_stats)
        sparsity_detail_txt_file.write("\n")

    for stat in min_sparsity_list:
        name_dot = "image_sparsity_{:.3f}".format(100.0 * float(stat["sparsity"]))
        name = name_dot.replace(".", "_")

        full_path = savedir + r"\min\\"
        txt_path = save_image(stat["image_label"][0][0], full_path, name, save_rescale, save_shift)

        run_txt_file.write(txt_path)
        run_txt_file.write("\n")
        run_txt_file.write(str(np.argmax(stat["image_label"][1]).astype(np.int32)))
        run_txt_file.write("\n")

        sparsity_detail_txt_file.write(name)
        sparsity_detail_txt_file.write(" - weighted sparsity: ")
        sparsity_detail_txt_file.write(str(stat["sparsity"]))
        sparsity_detail_txt_file.write("\n")
        detailed_stats = pprint.pformat(stat["stats"], indent=3, width=130, compact=True)  # str(stat["stats"]).replace("[OrderedDict([(")

        for layer_idx in range(len(stat["stats"])):
            detailed_stats = detailed_stats.replace("OrderedDict([(", "Layer {}: ".format(layer_idx), 1)

        detailed_stats = detailed_stats.replace(")]=,", "")
        detailed_stats = detailed_stats.replace("), (", ", ")
        detailed_stats = detailed_stats.replace("',", "':")
        detailed_stats = detailed_stats.replace(")]),", "")
        detailed_stats = detailed_stats.replace("[", " ")
        detailed_stats = detailed_stats.replace(")])]", "")
        sparsity_detail_txt_file.write(detailed_stats)
        sparsity_detail_txt_file.write("\n")

    if verbose:
        log.info("Reporting stats...")
        for stat in min_sparsity_list + top_sparsity_list:
            log.info(stat)

    run_txt_file.flush()
    run_txt_file.close()
    sparsity_detail_txt_file.flush()
    sparsity_detail_txt_file.close()

    if verbose: log.info("Sparsity stats compute done")


class SparsityLogger(Callback):
    def __init__(self, frequency, datasets, savedir, num_stored_input=5, max_images=25000, layer_cap=-1, verbose=False, save_rescale=256,
                 save_shift=0):
        self.frequency = frequency
        self.datasets = datasets

        if "{}" in savedir:
            self.savedir = savedir
        else:
            self.savedir = savedir + r"{}\\"

        self.num_stored_input = num_stored_input
        self.max_images = max_images
        self.layer_cap = layer_cap
        self.verbose = verbose
        self.save_rescale = save_rescale
        self.save_shift = save_shift

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.frequency == 0:
            try:
                get_weight_sparsity(self.model)
            except Exception as e:
                log.info("Failed to extract weigths sparsity data: {}".format(e))

            for dataset in self.datasets:
                try:
                    activ_sparsity_stats(self.model, dataset, self.savedir.format(epoch), num_stored_input=self.num_stored_input,
                                         max_images=self.max_images, layer_cap=self.layer_cap,
                                         verbose=self.verbose, save_rescale=self.save_rescale, save_shift=self.save_shift)
                except Exception as e:
                    log.info("Failed to extract activation sparsity data: {}".format(e))
                    break
