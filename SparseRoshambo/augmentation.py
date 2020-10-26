import cv2
import numpy as np
import random
import skimage
import scipy
import math
import logging
import gc

log = logging.getLogger()
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # for skimage spam


def augment_random_move(image):
    # add  extra pixels on all sides

    row_pad = int(image.shape[0] * 5 // 100)
    col_pad = int(image.shape[1] * 5 // 100)

    rows = image.shape[0]
    col = image.shape[1]



    modes = ["constant",
             "edge",
             "linear_ramp",
             "maximum",
             "mean",
             "median",
             "minimum",
             "reflect",
             "symmetric",
             "wrap"]

    random.shuffle(modes)

    input_dict = {"array": image,
                  "pad_width": [(row_pad, row_pad), (col_pad, col_pad), (0, 0)],
                  "mode": modes[0]
                  }

    if modes[0] == "constant":
        random_padding = random.uniform(0, 1)
        input_dict["constant_values"] = random_padding
    elif modes[0] == "reflect":
        reflect_types = ["even", "odd"]
        random.shuffle(reflect_types)
        input_dict["reflect_type"] = reflect_types[0]

    image = np.pad(**input_dict)

    # randomly crop
    # image = tf.random_crop(image, [32, 32, 3])
    rand_x = random.randint(0, col_pad * 2)
    rand_y = random.randint(0, row_pad * 2)
    image = image[rand_y:rand_y + rows, rand_x:rand_x + col, :]

    return image


def augment_fliplr(image):
    image = np.fliplr(image)
    return image


def augment_flipud(image):
    image = np.flipud(image)
    return image


# input assumed 0-1
def augment_shift_colors(image):
    shift = random.uniform(-0.1, 0.1)

    shifted = image + shift
    normalized = shifted % 1

    return normalized


def augment_random_noise(image):
    # range 0 to 1
    noise = np.random.random((image.shape[0], image.shape[1], image.shape[2])).astype(image.dtype)
    noise_rescale = random.uniform(0.1, 0.4)
    image = (image + noise_rescale * noise) % 1

    return image


def augment_greyscale(image):
    grey = skimage.color.rgb2gray(image)
    rgb = skimage.color.gray2rgb(grey)

    return rgb


def augment_random_rotate(image):
    # rows = image.shape[0]
    # cols = image.shape[1]
    # angle = np.random.randint(-90, 90)
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    # dst = cv2.warpAffine(image.astype(np.float32), M, (cols, rows))  # must get flaot32 datatype
    # dst = dst.astype(image.dtype)
    modes = ["reflect",
             "constant",
             "nearest",
             "mirror",
             "wrap"]

    random.shuffle(modes)
    cval = random.uniform(0, 1)

    angle = np.random.randint(-30, 30)
    # *reshape = bool(np.random.randint(0, 9) % 2 == 0)
    prefilter = bool(np.random.randint(0, 9) % 2 == 0)
    order = np.random.randint(0, 6)
    rotated = scipy.ndimage.rotate(image.astype(np.float32), angle, axes=(1, 0), reshape=False, output=None, order=order, mode=modes[0], cval=cval, prefilter=prefilter)

    return rotated.astype(image.dtype)


def augment_change_lighting(image):
    gamma = random.uniform(0.3, 3)

    image_int = (image * 256).astype(np.uint8)

    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)

    image_int = cv2.LUT(image_int, table)

    image = image_int.astype(image.dtype) / 256
    return image


# kills accuracy
def augment_remove_ch(image):
    ch = random.randint(0, 2)
    image = np.array(image)
    image[:, :, ch] = 0.0
    return image


# freezes the systemm, not used
def augment_rescale(image):
    scale_factor = random.uniform(0.75, 1.25)
    preserve_range = bool(np.random.randint(0, 9) % 2 == 0)
    clip = bool(np.random.randint(0, 9) % 2 == 0)
    anti_aliasing = bool(np.random.randint(0, 9) % 2 == 0)
    order = np.random.randint(0, 6)
    cval = random.uniform(0, 1)
    modes = ["reflect",
             "constant",
             "edge",
             "symmetric",
             "wrap"]

    random.shuffle(modes)

    scale_in = skimage.transform.rescale(image.astype(np.float32), scale=scale_factor, order=order, mode=modes[0], cval=cval, clip=clip, preserve_range=preserve_range,
                                         multichannel=True,
                                         anti_aliasing=anti_aliasing, anti_aliasing_sigma=None)

    if scale_factor > 1:
        new_row = scale_in.shape[0]
        new_col = scale_in.shape[1]
        diff_row = (new_row - image.shape[0]) // 2
        diff_col = (new_col - image.shape[1]) // 2
        scale_in = scale_in[diff_row:diff_row + image.shape[0], diff_col:diff_col + image.shape[1], :]
    else:
        new_row = scale_in.shape[0]
        new_col = scale_in.shape[1]

        diff_row_up = math.ceil((image.shape[0] - new_row) / 2)
        diff_row_down = image.shape[0] - scale_in.shape[0] - diff_row_up

        diff_col_up = math.ceil((image.shape[1] - new_col) / 2)
        diff_col_down = image.shape[1] - scale_in.shape[1] - diff_col_up

        scale_in = np.pad(scale_in, ((diff_row_up, diff_row_down), (diff_col_up, diff_col_down), (0, 0)), 'constant', constant_values=0.0)

    if scale_in.shape != image.shape:
        print(scale_factor)
        print(scale_in.shape)
        raise Exception

    return scale_in.astype(image.dtype)


# transform_list = [augment_random_noise, augment_fliplr, augment_flipud, augment_random_move, augment_shift_colors, augment_greyscale,
#                   augment_change_lighting, augment_random_rotate]
 #augment_greyscale
transform_list = [augment_random_noise, augment_fliplr, augment_flipud, augment_random_move, augment_shift_colors, augment_change_lighting, augment_greyscale, augment_random_rotate]

def decision(probability):
    return random.random() <= probability

def augment_images(raw_images, labels, random_seed, queue):
    np.random.seed(random_seed)
    random.seed(random_seed)

    iter_probability = 0.01

    while True:
        iter_probability = iter_probability + 0.001

        max_probability = random.uniform(0.2, 0.45)

        probability = min(iter_probability, max_probability)

        all_transformed = list()
        all_labels = list()

        for image, label in zip(raw_images, labels):
            random.shuffle(transform_list)
            transformed = image

            for transform in transform_list:

                if decision(probability) is True:  # to have an higher degree of randomness, we dont appl all the transformations but just some
                    try:
                        transformed = transform(transformed)
                    except Exception as e:
                        log.error("Error in function {} - {}".format(str(transform), e))


            # Safety check tgo avoid combination of transformations leads to poor results
            if np.isnan(transformed).any() or np.isinf(transformed).any():
                all_transformed.append(image)
                all_labels.append(label)
            else:
                all_transformed.append(transformed)
                all_labels.append(label)

        # extra step to return the correct number of images of the correct dtype
        # we dont return the original images since the two lists are merged in the original main class
        return_images = np.array(all_transformed, dtype=np.float32) * 256
        return_images = return_images.astype(np.uint8)
        return_images = return_images.astype(np.float32) / 256
        return_images = return_images.astype(raw_images[0].dtype)

        return_labels = np.array(all_labels, dtype=labels[0].dtype)

        queue_dict = {"images": return_images,
                      "labels": return_labels}

        queue.put(queue_dict)
