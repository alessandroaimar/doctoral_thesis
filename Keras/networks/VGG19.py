from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19 as KVGG19
import os
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.data_utils import get_file
import logging

log = logging.getLogger()


# def VGG19PruningPolicy(pruning_policy):
#     if pruning_policy["mode"] == "fixed":
#         return pruning_policy["target"]
#     else:
#         raise AttributeError


IMAGENET_WEIGHTS_PATH = (
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5')
IMAGENET_WEIGHTS_PATH_NO_TOP = (
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')


def VGG19(include_top=True,
          input_tensor=None,
          pooling=None,
          classes=1000,
          continue_train=False,
          regularize_activ=False,
          **kwargs):


    if regularize_activ:
        l1 = 1 * (10 ** (-4))
        log.info("Inserting activation L1 regularization: {}".format(l1))
        def get_activ_reg():
            return tf.keras.regularizers.l1(l1)
    else:
        def get_activ_reg():
            return None

    
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block1_conv1')(input_tensor)

    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block1_conv2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      bias_initializer=tf.initializers.Zeros(),
                      activity_regularizer=get_activ_reg(),
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      activity_regularizer=get_activ_reg(),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=None,
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block5_conv4')(x)

    if pooling == "global":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    #l2 = 1 * (10 ** (-4))

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)

        #x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096,
                         activation='relu',
                         name='fc1',
                         #kernel_regularizer=tf.keras.regularizers.l2(l2),
                         kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                         bias_initializer=tf.initializers.Zeros(),
                         )(x)

        #x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096,
                         activation='relu',
                         name='fc2',
                         #kernel_regularizer=tf.keras.regularizers.l2(l2),
                         kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                         bias_initializer=tf.initializers.Zeros(),
                         )(x)
        #x = layers.Dropout(0.5)(x)
        x = layers.Dense(classes,
                         activation='softmax',
                         bias_initializer=tf.initializers.Zeros(),
                         kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                         name='predictions',
                         )(x)

    model = Model(inputs=input_tensor, outputs=x, name='vgg19')

    if continue_train is False:
        log.info("Not Loading pretrained weights")
    else:
        if continue_train is "imagenet":
            log.info("Loadinfg imagenet weights")
            if include_top:
                weights_path = get_file(
                    'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                    IMAGENET_WEIGHTS_PATH,
                    cache_subdir='models',
                    file_hash='cbe5617147190e668d6c5d5026f83318')
            else:
                weights_path = get_file(
                    'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    IMAGENET_WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    file_hash='253f8cb515780f3b799900260a226db6')
        else:
            weights_path = continue_train


        log.info("Loading pretrained weights from {}".format(continue_train))
        model.load_weights(weights_path)

    return model

