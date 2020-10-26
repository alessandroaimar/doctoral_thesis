import os
from config import config_dict
from tensorflow.keras.layers import Input
import tensorflow as tf
import numpy as np


class baseDataset():

    def __init__(self):
        self.cache = ""  # DRAM cache by default
        self.batch_prefetch = 3#tf.data.experimental.AUTOTUNE
        self.shuffle_buffer_size = 2 ** 24  # About the whole dataset by default
        self.default_block_length = 32

    def get_files_in_dirs(self, dirs):
        files = {}

        for set_name, dir in dirs.items():
            set_files = os.listdir(dir)
            set_files_and_paths = list()
            for file in set_files:
                set_files_and_paths.append(dir + r"\\" + file)
            files[set_name] = set_files_and_paths

        return files

    def dataset_augment(self, image):
        return image

    def parser(self, record, resize_dims, augment):
        """It parses one tfrecord entry

        Args:
            record: image + label
        """
        # with tf.device('/cpu:0'):
        features = tf.io.parse_single_example(record,
                                              features={
                                                  'height': tf.io.FixedLenFeature([], tf.int64),
                                                  'width': tf.io.FixedLenFeature([], tf.int64),
                                                  'depth': tf.io.FixedLenFeature([], tf.int64),
                                                  'image_raw': tf.io.FixedLenFeature([], tf.string),
                                                  'label': tf.io.FixedLenFeature([], tf.int64),
                                              })

        label = tf.cast(features["label"], tf.int32)
        label = tf.one_hot(label, self.num_classes)

        image = tf.io.decode_raw(features["image_raw"], tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.scalar_mul(1.0 / (2.0 ** 8), image)
        image_shape = list([features['height'], features['width'], features['depth']])
        image = tf.reshape(image, image_shape)

        if resize_dims is not None:
            # resize method default is bilinear
            image = tf.image.resize(image, resize_dims, antialias=False, preserve_aspect_ratio=False)

        if augment:
            image = self.dataset_augment(image)

        return image, label

    def read_parse_and_cache(self, dataset_files, resize_dims, cache_ext, shuffle, augment, batch_size):
        if isinstance(dataset_files, list) is False:
            dataset_files = [dataset_files]

        if cache_ext == "train":
            block_length = 1
        else:
            block_length = self.default_block_length

        filenames_dataset = tf.data.Dataset.from_tensor_slices(dataset_files)

        if shuffle:
            filenames_dataset = filenames_dataset.shuffle(buffer_size=len(dataset_files))

        interleaved_dataset = filenames_dataset.interleave(
            tf.data.TFRecordDataset,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE,
            block_length=block_length,
            cycle_length=self.num_classes
        )

        parsed_dataset = interleaved_dataset.map(map_func=lambda record: self.parser(record=record,
                                                                                     resize_dims=resize_dims,
                                                                                     augment=augment),
                                                 num_parallel_calls=batch_size)

        if self.cache is None:
            return parsed_dataset
        else:
            if self.cache != "":
                cache = self.cache + r"\\" + cache_ext
            else:
                cache = self.cache

            cached_dataset = parsed_dataset.cache(cache)

            # Everything prior to this point is executed only in the first epoch due to .cache()
            return cached_dataset

    def prepare_dataset(self, filenames, batch_size, resize_dims, shuffle, cache_ext, augument):
        # read from HD
        dataset = self.read_parse_and_cache(filenames, resize_dims, cache_ext, shuffle, augument, batch_size)

        # we shuffle the couples so each batch is balanced
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        # We apply the proper batch size
        dataset = dataset.batch(batch_size).prefetch(self.batch_prefetch)

        return dataset

    def prepare_dataset_from_numpy(self, x, y, batch_size, shuffle, augment):
        x = (x.astype(np.float32)) / 256.0
        y = tf.keras.utils.to_categorical(y, self.num_classes)

        dataset = tf.data.Dataset.from_tensor_slices((x, y)).cache(self.cache)

        if augment:
            dataset = dataset.map(map_func=lambda image, label: self.dataset_augment(image=image, label=label),
                                  num_parallel_calls=batch_size)
        # we shuffle the couples so each batch is balanced
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        # We apply the proper batch size
        dataset = dataset.batch(batch_size).prefetch(self.batch_prefetch)

        return dataset

    def get_train(self, shuffle=True):
        try:
            resize_dims = config_dict["resize_dims"]
        except KeyError:
            resize_dims = None

        dataset = self.prepare_dataset(self.files["train"], config_dict["batch_size"], resize_dims, shuffle, "train", augument=True)

        return dataset

    def get_validation(self):
        try:
            resize_dims = config_dict["resize_dims"]
        except KeyError:
            resize_dims = None

        dataset = self.prepare_dataset(self.files["val"], config_dict["batch_size"], resize_dims, False, "validation", augument=False)

        return dataset

    def get_test(self):
        try:
            resize_dims = config_dict["resize_dims"]
        except KeyError:
            resize_dims = None

        dataset = self.prepare_dataset(self.files["test"], config_dict["batch_size"], resize_dims, False, "test", augument=False)

        return dataset

    def get_hw_debug(self):
        try:
            resize_dims = config_dict["resize_dims"]
        except KeyError:
            resize_dims = None

        dataset = self.prepare_dataset(self.files["train"], 1, resize_dims, False, "hw_debug", augument=False)

        input_image = list(dataset.as_numpy_iterator())[0][0][0]

        y = np.zeros(shape=(1, self.num_classes))
        x = np.expand_dims(input_image, 0)

        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        # We apply the proper batch size
        dataset = dataset.batch(1)

        return dataset

    def get_input_ph(self):
        input_ph = Input(shape=self.input_shape)
        return input_ph
