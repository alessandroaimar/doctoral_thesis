from datasets.baseDataset import baseDataset
from config import ws_dict
import logging
import tensorflow as tf
from pathlib import Path
import os
import glob
from config import config_dict
import numpy as np

log = logging.getLogger()


class ILSVRC2012(baseDataset):

    def __init__(self):
        super().__init__()

        self.cache = None  # r"D:\DL\tf_cache\\ILSVRC2012\\"

        if self.cache is not None:
            filelist = os.listdir(self.cache)
            for f in filelist:
                os.remove(os.path.join(self.cache, f))

            Path(self.cache).rmdir()
            Path(self.cache).mkdir(parents=True, exist_ok=True)

            files = glob.glob(self.cache + r"\\*")
            for f in files:
                os.remove(f)

        self.batch_prefetch = 2
        self.shuffle_buffer_size = 2 ** 15

        self.dirs = {}
        if ws_dict["override_dataset_read_dir"] is None:
            self.dirs["train"] = ws_dict["datasets_root_dir"] + r"\ILSVRC2012\tf_records\train\\"
            self.dirs["val"] = ws_dict["datasets_root_dir"] + r"\ILSVRC2012\tf_records\val\\"
            self.dirs["test"] = ws_dict["datasets_root_dir"] + r"\ILSVRC2012\tf_records\test\\"
        else:
            raise NotImplementedError("Override of sub path not implemented")

        self.files = self.get_files_in_dirs(self.dirs)
        self.input_shape = (224, 224, 3)
        self.num_pixels = 224 * 224 * 3
        self.num_classes = 1000

        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        self._RESIZE_SIDE_MIN = 256
        self._RESIZE_SIDE_MAX = 512
        self.means = [_R_MEAN, _G_MEAN, _B_MEAN]
        means_int = np.array([_R_MEAN, _G_MEAN, _B_MEAN]).astype(np.int16)
        self.means_int = tf.constant(means_int, dtype=tf.int16)
        self.rescale = tf.constant(1.0 / (2.0 ** 8), dtype=tf.float32)
        self.depth_zero = tf.constant(0, dtype=tf.int32)

    def get_test(self):
        log.info("Test set non implemented for ILSVRC2012, running validation set")

        try:
            resize_dims = config_dict["resize_dims"]
        except KeyError:
            resize_dims = None

        dataset = self.prepare_dataset(self.files["val"], config_dict["batch_size"], resize_dims, False, "validation", False)

        return dataset

    def dataset_augment(self, image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_hue(image, 0.05)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        return image

    @tf.function
    def parser(self, record, resize_dims, augment):
        """It parses one tfrecord entry

        Args:
            record: image + label
        """

        with tf.name_scope("parser"):
            with tf.device('/cpu:0'):
                features = tf.io.parse_single_example(record,
                                                      features={
                                                          'height': tf.io.FixedLenFeature([], tf.int64),
                                                          'width': tf.io.FixedLenFeature([], tf.int64),
                                                          'depth': tf.io.FixedLenFeature([], tf.int64),
                                                          'image_raw': tf.io.FixedLenFeature([], tf.string),
                                                          'label': tf.io.FixedLenFeature([], tf.int64),
                                                      })

                label = tf.one_hot(features["label"], self.num_classes)

                image = tf.io.decode_raw(features["image_raw"], tf.uint8)

                height = tf.cast(features["height"], tf.int32)
                width = tf.cast(features["width"], tf.int32)
                depth = tf.cast(features["depth"], tf.int32)
                #tf.print("Reshaping with size ", height, width, depth, height*width*depth)

                image = tf.reshape(image, [height, width, depth])


                # training mode
                if augment is True:
                    resize_side = tf.random.uniform([], minval=self._RESIZE_SIDE_MIN, maxval=self._RESIZE_SIDE_MAX + 1, dtype=tf.int32)
                    image,  new_height, new_width = self._aspect_preserving_resize(image, resize_side, height, width)
                    image = tf.image.random_crop(image, size=self.input_shape)
                    image = tf.image.random_flip_left_right(image)
                else:
                    # validation mode
                    image, new_height, new_width = self._aspect_preserving_resize(image, self._RESIZE_SIDE_MIN, height, width)
                    image = self._central_crop(image,  new_height, new_width, self.input_shape[0], self.input_shape[1], self.input_shape[2])

                #necessary because they are uint8
                #we use int16 an not float to match nullhop hw
                image = tf.cast(image, tf.int16)
                image = self._mean_image_subtraction(image, self.means_int)
                image = tf.cast(image, tf.float32)

                image = tf.scalar_mul(self.rescale, image)


            return image, label

    @tf.function
    def _mean_image_subtraction(self, image, means):
        """Subtracts the given means from each image channel.
        For example:
          means = [123.68, 116.779, 103.939]
          image = _mean_image_subtraction(image, means)
        Note that the rank of `image` must be known.
        Args:
          image: a tensor of size [height, width, C].
          means: a C-vector of values to subtract from each channel.
        Returns:
          the centered image.
        Raises:
          ValueError: If the rank of `image` is unknown, if `image` has a rank other
            than three or if the number of channels in `image` doesn't match the
            number of values in `means`.
        """
        return tf.subtract(image, means)
        # channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
        # for i in range(num_channels):
        #     channels[i] -= means[i]
        # return tf.concat(axis=2, values=channels)

    @tf.function
    def _aspect_preserving_resize(self, image, smallest_side, height, width):
        # smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

        new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
        resized_image = tf.image.resize(images=image, size=[new_height, new_width])

        return resized_image, new_height, new_width

    @tf.function
    def _smallest_size_at_least(self, height, width, smallest_side):
        """Computes new shape with the smallest side equal to `smallest_side`.
        Computes new shape with the smallest side equal to `smallest_side` while
        preserving the original aspect ratio.
        Args:
          height: an int32 scalar tensor indicating the current height.
          width: an int32 scalar tensor indicating the current width.
          smallest_side: A python integer or scalar `Tensor` indicating the size of
            the smallest side after resize.
        Returns:
          new_height: an int32 scalar tensor indicating the new height.
          new_width: and int32 scalar tensor indicating the new width.
        """
        # smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)
        smallest_side = tf.cast(smallest_side, tf.float32)

        scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)

        new_height = tf.cast(height * scale, tf.int32)
        new_width = tf.cast(width * scale, tf.int32)
        return new_height, new_width

    @tf.function
    def _central_crop(self, image, image_height, image_width, crop_height, crop_width, crop_depth):
        """Performs central crops of the given image list.
        Args:
          image_list: a list of image tensors of the same dimension but possibly
            varying channel.
          crop_height: the height of the image following the crop.
          crop_width: the width of the image following the crop.
        Returns:
          the list of cropped images.
        """
        offset_height = tf.cast((image_height - crop_height) / 2, tf.int32)
        offset_width = tf.cast((image_width - crop_width) / 2, tf.int32)

        return self._crop(image, offset_height, offset_width, crop_height, crop_width, crop_depth)

    @tf.function
    def _crop(self, image, offset_height, offset_width, crop_height, crop_width, crop_depth):
        """Crops the given image using the provided offsets and sizes.
        Note that the method doesn't assume we know the input image size but it does
        assume we know the input image rank.
        Args:
          image: an image of shape [height, width, channels].
          offset_height: a scalar tensor indicating the height offset.
          offset_width: a scalar tensor indicating the width offset.
          crop_height: the height of the cropped image.
          crop_width: the width of the cropped image.
        Returns:
          the cropped (and resized) image.
        Raises:
          InvalidArgumentError: if the rank is not 3 or if the image dimensions are
            less than the crop size.
        """
        # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
        # define the crop size.
        image = tf.slice(input_=image, begin=[offset_height, offset_width, self.depth_zero], size=[crop_height, crop_width, crop_depth])

        return image
