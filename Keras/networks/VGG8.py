from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19 as KVGG19
import os
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.data_utils import get_file

def VGG8PruningPolicy(pruning_policy):
    if pruning_policy["mode"] == "fixed":
        return pruning_policy["target"]
    else:
        raise AttributeError


def VGG8(include_top=True,
          input_tensor=None,
          pooling=None,
          classes=1000,
          **kwargs):

    l2 = 5*(10**(-4))

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(l2),
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block1_conv1')(input_tensor)


    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(l2),
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block2_conv1')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(l2),
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block3_conv1')(x)


    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(l2),
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block4_conv1')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(l2),
                      kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                      bias_initializer=tf.initializers.Zeros(),
                      name='block5_conv1')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)

        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096,
                         activation='relu',
                         name='fc1',
                         kernel_regularizer=tf.keras.regularizers.l2(l2),
                         kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                         bias_initializer=tf.initializers.Zeros(),
                         )(x)

        x = layers.Dense(4096,
                         activation='relu',
                         name='fc2',
                         kernel_regularizer=tf.keras.regularizers.l2(l2),
                         kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                         bias_initializer=tf.initializers.Zeros(),
                         )(x)

        x = layers.Dense(classes,
                         activation='softmax',
                         bias_initializer=tf.initializers.Zeros(),
                         kernel_initializer=tf.keras.initializers.he_normal(seed=None),
                         name='predictions',
                         )(x)

        model = Model(inputs=input_tensor, outputs=x, name='vgg19')


    return model
