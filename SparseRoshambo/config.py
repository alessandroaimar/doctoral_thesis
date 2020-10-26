from structure import Structure
from network_description import *

model_save_path = r"D:\DL\models\SparseRoshambo\baseline_avg_pool\\"
data_path = r"D:\DL\datasets\Roshambo\tf_train\\"

train_data_path = data_path + r"train\\"
test_data_path = data_path + r"test\\"

shape = (64, 64, 1)
train_batch_size = 1024

num_test_images = 1102176
test_batch_size = num_test_images // 144
num_test_batch = num_test_images // test_batch_size
normalize = 256.0
shift = None
data_format = "channels_last"

num_classes = 4
learning_rate = 2**(-11)
num_epochs = 10000
epoch_save_period = 20

use_fixed_point = True
tf_full_prec = tf.float32
tf_training_prec = tf.float16

network_structure = Structure(network_description, use_fixed_point, data_format)

