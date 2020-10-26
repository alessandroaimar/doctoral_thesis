from datasets.baseDataset import baseDataset
from config import ws_dict, config_dict
import logging
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import numpy as np
log = logging.getLogger()

class CIFAR10(baseDataset):

    def __init__(self):
        super().__init__()

        self.input_shape = (32, 32, 3)
        self.num_classes = 10
        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

    def dataset_augment(self, image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_hue(image, 0.05)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image, label

    def get_train(self, shuffle=True):
        dataset = self.prepare_dataset_from_numpy(self.x_train, self.y_train, config_dict["batch_size"], shuffle, True)
        return dataset

    def get_validation(self):
        dataset = self.prepare_dataset_from_numpy(self.x_test, self.y_test, config_dict["batch_size"], False, False)
        return dataset

    def get_test(self):
        dataset = self.prepare_dataset_from_numpy(self.x_test, self.y_test, config_dict["batch_size"], False, False)
        return dataset



    def get_hw_debug(self):
        x = self.x_test[0:1].astype(np.float32) / (2.0 ** 8)
        y = tf.keras.utils.to_categorical(self.y_test[0:1], self.num_classes)

        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


        #dataset = self.prepare_dataset_from_numpy(self.x_test[0], self.y_test[0], 1, False)
        return dataset