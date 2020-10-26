"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import scipy
import Quantizer
import Sparsity
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            print(gpu)
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                                    [tf.config.experimental.VirtualDeviceConfiguration(
                                                                        memory_limit=5120)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from tensorflow.keras.layers import PReLU, LeakyReLU
# from tensorflow.keras.layers.advanced_activations import PReLU, LeakyReLU
# from tensorflow.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# Not used so removed
# from tensorflow.keras_contrib.layers.normalization.instancenormalization import InstanceNormalization


import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

import tensorflow.keras.backend as K


class SRGAN():
    def __init__(self, quantize_flag):
        self.quantize_flag = quantize_flag
        # Input shape
        self.channels = 3
        self.lr_height = 56  # Low resolution height
        self.lr_width = 56  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * 4  # High resolution height
        self.hr_width = self.lr_width * 4  # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        optimizer = Adam(0.000002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = 'img_align_celeba'
        self.test_dataset_name = 'test_images'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))
        self.test_data_loader = DataLoader(dataset_name=self.test_dataset_name,
                                           img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/tensorflow.keras-team/tensorflow.keras/blob/master/tensorflow.keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)

        # Extract image features
        print(img.shape)
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(layer_input)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same', activation="relu")(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same', activation='relu', )(img_lr)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)



        if self.quantize_flag == "quant":
            return Quantizer.apply_quantization(Model(img_lr, gen_hr), weight_precision=16, activation_precision=16)
        elif self.quantize_flag == "sparsity":
            return Sparsity.measure_sparsity(Model(img_lr, gen_hr))
        else:
            return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)

        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=50, start_from_ckp_number=None):

        start_time = datetime.datetime.now()

        # Set up ternsorboard
        logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        file_writer.set_as_default()

        # Restore checkpoint if present
        start_epoch = 0
        if start_from_ckp_number is not None:
            start_epoch = start_from_ckp_number
            self.generator.load_weights('./saved_models/generator_ckp_epoch_{}'.format(start_from_ckp_number))
            self.discriminator.load_weights('./saved_models/discriminator_ckp_epoch_{}'.format(start_from_ckp_number))
            self.combined.load_weights('./saved_models/combined_ckp_epoch_{}'.format(start_from_ckp_number))

        callbacks = [TensorBoard(log_dir=r"./logs/tensorboard",
                                 histogram_freq=1,
                                 write_graph=True,
                                 write_images=False,
                                 update_freq='epoch',
                                 profile_batch=0,
                                 embeddings_freq=0,
                                 embeddings_metadata=None)]

        for epoch in range(start_epoch, epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            try:
                self.combined.fit([imgs_lr, imgs_hr], [valid, image_features], epochs=epoch+1, batch_size=len(imgs_lr),
                                  initial_epoch=epoch, callbacks=callbacks)

                #g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])
            except Exception as e:
                print("Something went wrong with this batch: {}".format(e))

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            if epoch % 10 == 0:
                print("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, batch_size)

    def sample_images(self, epoch, batch_size):
        os.makedirs('images/%s' % self.test_dataset_name, exist_ok=True)

        # Get test data
        # imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        imgs_hr_a, imgs_lr_a = self.test_data_loader.load_data(batch_size=batch_size, is_testing=True)

        # Make prediction
        fake_hr_a = self.generator.predict(imgs_lr_a)
        image_idx = 0

        for imgs_lr, fake_hr, imgs_hr in zip(imgs_lr_a, fake_hr_a, imgs_hr_a):
            # # Rescale images 0 - 1
            imgs_lr = 0.5 * imgs_lr + 0.5
            fake_hr = 0.5 * fake_hr + 0.5
            imgs_hr = 0.5 * imgs_hr + 0.5

            imgs_hr = imgs_hr.reshape(self.hr_shape)
            scipy.misc.toimage(imgs_hr).save(
                'images/{}/batch_{}_idx_{}_original_highres.png'.format(self.test_dataset_name,
                                                                        epoch,
                                                                        image_idx,
                                                                        ))

            # Save lr version (if not present yet)
            imgs_lr = imgs_lr.reshape(self.lr_shape)
            scipy.misc.toimage(imgs_lr).save(
                'images/{}/batch_{}_idx_{}_original_lowres.png'.format(self.test_dataset_name,
                                                                       epoch,
                                                                       image_idx,
                                                                       ))

            # Save generaed image
            fake_hr = fake_hr.reshape(self.hr_shape)
            scipy.misc.toimage(fake_hr).save(
                'images/{}/batch_{}_idx_{}_generated_highres.png'.format(self.test_dataset_name,
                                                                            epoch,
                                                                            image_idx,
                                                                            ))

            image_idx = image_idx + 1
        # Compute mean squared error
        mse = np.mean((np.array(imgs_hr_a, dtype=np.float32) - np.array(fake_hr_a, dtype=np.float32)) ** 2)

        # Compute psnr
        if mse == 0:
            # avoid (improbable) division by 0
            psnr = 1000000
        else:
            max_pixel = 1.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))


        # save MSE and PSNR to tensorboard
        tf.summary.scalar('MSE', data=mse, step=epoch)
        tf.summary.scalar('PSNR', data=psnr, step=epoch)

        for index, layer in enumerate(self.generator.layers):
            try:
                sparsity = layer.get_weights()[4]
                tf.summary.scalar('Sparsity_{}'.format(index), data=sparsity, step=epoch)
            except IndexError:
                pass
                # try:
                #     print(layer.layer)
                # except:
                #     print("no", layer)



        # save the generator model weights
        self.generator.save_weights('./saved_models/generator_ckp_epoch_{}'.format(epoch))
        self.discriminator.save_weights('./saved_models/discriminator_ckp_epoch_{}'.format(epoch))
        self.combined.save_weights('./saved_models/combined_ckp_epoch_{}'.format(epoch))

        # r, c = 2, 2

        # # Save generated images and the high resolution originals
        # titles = ['Generated', 'Original']
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for row in range(r):
        #     for col, image in enumerate([fake_hr, imgs_hr]):
        #         axs[row, col].imshow(image[row])
        #         axs[row, col].set_title(titles[col])
        #         axs[row, col].axis('off')
        #     cnt += 1
        # fig.savefig("images/%s/%d.png" % (self.test_dataset_name, epoch))
        # plt.close()

        # # Save low resolution images for comparison
        # for i in range(r):
        #     fig = plt.figure()
        #     plt.imshow(imgs_lr[i])
        #     fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
        #     plt.close()


if __name__ == '__main__':
    gan = SRGAN(quantize_flag="sparsity")
    gan.train(epochs=3000000, batch_size=10, sample_interval=500, start_from_ckp_number=10000)

