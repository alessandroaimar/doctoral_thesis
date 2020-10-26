import tensorflow as tf
import logging
from os import walk
import config
log = logging.getLogger()

features_dict = {'image_raw': tf.FixedLenFeature([], tf.string),
                 'label': tf.FixedLenFeature([], tf.int64),
                 'height': tf.FixedLenFeature([], tf.int64),
                 'width': tf.FixedLenFeature([], tf.int64),
                 'depth': tf.FixedLenFeature([], tf.int64)}


def parser(record, shape, num_classes, normalize):
    global features_dict
    parsed = tf.io.parse_single_example(record, features_dict)

    # Perform additional preprocessing on the parsed data.
    image = tf.decode_raw(parsed["image_raw"], tf.int8)
    image = tf.reshape(image, shape)
    image = tf.cast(image, config.tf_training_prec)
    image = image / normalize

    label = tf.one_hot(tf.cast(parsed["label"], tf.int32), num_classes)

    return image, label


class Dataset():
    valid_channel_values = ('channels_first', 'channels_last')

    def __init__(self, paths, shape, batch_size, normalize, shift, num_classes, data_format, dtype, execute_shuffle):
        self.path = paths
        self.dtype = dtype
        self.shape = shape
        self.batch_size = batch_size
        self.normalize = normalize
        self.shift = shift
        self.set = set
        self.data_format = data_format
        self.num_classes = num_classes

        self.augment_process_started = False
        self.last_augmented_data = None
        self.augment_process_queue = None
        self.augment_process = None

        self.execute_shuffle = execute_shuffle

        self.filelists = list()
        filenames = []
        for (_, __, file) in walk(paths):

            filenames.extend(file)

        for file in filenames:
            log.info("Adding file to dataset: {}".format(file))
            self.filelists.append(paths + file)

        self.num_batches = None
        self.num_images = None
        self.all_images = None
        self.all_labels = None
        self.num_prefetch = tf.data.experimental.AUTOTUNE

        self.prepare_iterator()

    def prepare_iterator(self):

        self.dataset = tf.data.TFRecordDataset(self.filelists,
                                               compression_type=None,
                                               num_parallel_reads=32)

        self.dataset = self.dataset.map(
            map_func=lambda x: parser(x, self.shape, num_classes=self.num_classes, normalize=self.normalize),
            num_parallel_calls=32
        )#.prefetch(self.num_prefetch)

                   #.cache()

        if self.execute_shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=int(2 ** 7), reshuffle_each_iteration=True)

        self.dataset = self.dataset.batch(batch_size=self.batch_size, drop_remainder=False).prefetch(self.num_prefetch)

    def get_placeholders(self):

        network_input_shape = list(self.shape)

        features_ph_shape = [None] + network_input_shape
        labels_ph_shape = [None, self.num_classes]
        features_placeholder = tf.compat.v1.placeholder(self.dtype, features_ph_shape)
        labels_placeholder = tf.compat.v1.placeholder(tf.int16, labels_ph_shape)

        return features_placeholder, labels_placeholder

    def get_iterator(self):
        iterator = tf.compat.v1.data.make_initializable_iterator(self.dataset)
        initializer = iterator.initializer
        next_element = iterator.get_next()

        return initializer, next_element

    def get_onehot_iterator(self):
        iterator = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
        next_element = iterator.get_next()

        return next_element
