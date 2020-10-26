from __future__ import print_function
import tensorflow.keras

from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, MaxPooling2D
from tensorflow.keras.regularizers import l2


def ResNetPruningPolicy(pruning_policy):
    if pruning_policy["mode"] == "fixed":
        return pruning_policy["target"]
    else:
        raise AttributeError


# Model parameter and CIFAR Accuracy
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
"""ResNet Version 2 Model builder [b]

      Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
      bottleneck layer
      First shortcut connection per layer is 1 x 1 Conv2D.
      Second and onwards shortcut connection is identity.
      At the beginning of each stage, the feature map size is halved (downsampled)
      by a convolutional layer with strides=2, while the number of filter maps is
      doubled. Within each stage, the layers have the same number filters and the
      same filter map sizes.
      Features maps sizes:
      conv1  : 32x32,  16
      stage 0: 32x32,  64
      stage 1: 16x16, 128
      stage 2:  8x8,  256

      # Arguments
          input_shape (tensor): shape of input image tensor
          depth (int): number of core convolutional layers
          num_classes (int): number of classes (CIFAR10 has 10)

      # Returns
          model (Model): tensorflow.keras model instance
      """


def layer(inputs,
          num_filters=16,
          kernel_size=3,
          strides=1,
          activation='relu',
          batch_normalization=True,
          conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_tensor, depth, use_batchnorm, num_classes):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): tensorflow.keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    x = layer(inputs=input_tensor,  batch_normalization=use_batchnorm)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = layer(inputs=x,
                      num_filters=num_filters,
                      strides=1,
                      batch_normalization=use_batchnorm)
            if strides == 2:
                y = MaxPooling2D((2, 2), strides=(2, 2))(y)

            y = layer(inputs=y,
                      num_filters=num_filters,
                      activation=None,
                      batch_normalization=use_batchnorm)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = layer(inputs=x,
                          num_filters=num_filters,
                          kernel_size=3,
                          strides=1,
                          activation=None,
                          batch_normalization=False)
                if strides == 2:
                    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
            x = tensorflow.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = GlobalAveragePooling2D()(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    return outputs


def resnet_v2(input_tensor, depth, num_classes):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): tensorflow.keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = layer(inputs=input_tensor,
              num_filters=num_filters_in,
              conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2  # downsample

            # bottleneck residual unit
            y = layer(inputs=x,
                      num_filters=num_filters_in,
                      kernel_size=1,
                      strides=strides,
                      activation=activation,
                      batch_normalization=batch_normalization,
                      conv_first=False)
            y = layer(inputs=y,
                      num_filters=num_filters_in,
                      conv_first=False)
            y = layer(inputs=y,
                      num_filters=num_filters_out,
                      kernel_size=1,
                      conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = layer(inputs=x,
                          num_filters=num_filters_out,
                          kernel_size=1,
                          strides=strides,
                          activation=None,
                          batch_normalization=False)
            x = tensorflow.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    return outputs


def ResNet(include_top=True,
           input_tensor=None,
           pooling=None,
           classes=1000,
           depth=3,
           version=1,
           use_batchnorm=False,
           **kwargs):
    if version == 2:
        resnet = resnet_v2(input_tensor=input_tensor, depth=depth, num_classes=classes)
    else:
        resnet = resnet_v1(input_tensor=input_tensor, depth=depth, use_batchnorm=use_batchnorm, num_classes=classes)

    return resnet
