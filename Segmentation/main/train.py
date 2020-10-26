import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.keras.optimizers import SGD

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

from segment.keras_segmentation.models.unet import vgg_unet

# specify dataset and checkpoint location
train_images = r"D:/DL/datasets/pet-dataset/images_train"
train_annotations = r"D:/DL/datasets/pet-dataset/annotations_train"
test_images = r"D:/DL/datasets/pet-dataset/images_test"
test_annotations = r"D:/DL/datasets/pet-dataset/annotations_test"
checkpoints_path = r"./checkpoints/vgg_unet_pet"

# Put 1 to quantize the model
quantize_flag = "tnh"
# Input image dimension (must be multiple of 32 for vgg16)
input_height = 320
input_width = 480
# number of classes in dataset
n_classes = 51
# number of epochs to train for
n_epochs = 50
batch_size = 8
augment = False

# Instantiate model
model = vgg_unet(n_classes=n_classes,
                 input_height=input_height,
                 input_width=input_width,
                 quantize_flag=quantize_flag)

model.train(
    train_images=train_images,
    train_annotations=train_annotations,
    checkpoints_path=checkpoints_path,
    epochs=n_epochs,
    validate=True,
    val_images=test_images,
    val_annotations=test_annotations,
    batch_size=batch_size,
    do_augment=augment,
    verify_dataset=False,
    #optimizer_name=SGD(learning_rate=0.0001, momentum=0.9)
)

for filename in os.listdir(test_images):
    test_img = os.path.join(test_images, filename)

    save_path = os.path.join(r"./output/", "mask_" + filename)

    try:
        os.remove(save_path)
    except OSError:
        pass

    try:
        out = model.predict_segmentation(
            inp=test_img,
            out_fname=save_path
        )
    except Exception as e:
        print("Can't predict image {}".format(test_img))
