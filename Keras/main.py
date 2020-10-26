import os
from utilities import *
import logging
from config import config_dict, ws_dict
import math

run_id = get_run_unique_id(config_dict)
run_dir = prepare_save_folders_and_logger(ws_dict, run_id)

log = logging.getLogger()
log.setLevel(level=logging.DEBUG)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152

if ws_dict["GPU"] == None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    log.info("Selected device: CPU")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(ws_dict["GPU"])
    log.info("Selected device: GPU{}".format(ws_dict["GPU"]))

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from stats import *
from hardware import Quantizer, hw_utilities, SLG, SLC, Sparsity
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# log.info("Setting mixed mode for training")
policy = mixed_precision.Policy('float32')  # dynamic loss scaling doesnt work
mixed_precision.set_policy(policy)

dataset, model, pruning_policy = get_dataset_model_and_pruning(config_dict, ws_dict)

# from flops import get_flops
# flops = get_flops()
# log.info("Number of ops: {}".format(flops))


for key in ws_dict:
    if "load_model" == key and \
            ws_dict["load_model"] is not False and \
            ws_dict["load_model"] is not None:
        model.set_weights(ws_dict["load_model"])
        break

# Apply quantization if requested
checkpoint_weights = None
if config_dict["quantize"] == "tnh":
    log.info("Applying TNH quantization")
    model = Quantizer.apply_quantization(model,
                                         pruning_policy,
                                         activation_precision=config_dict["activation_precision"],
                                         weight_precision=config_dict["weight_precision"])

    # Prepare callbacks for weights saving
    checkpoint_weights = Quantizer.QuantizerSaveCallback(filepath=run_dir + r"\save\\")

elif config_dict["quantize"] == "slg":
    log.info("Applying SLG generation")
    model = SLG.apply_quantization(model)
elif config_dict["quantize"] == "slc":
    log.info("Applying SLC generation")
    model = SLC.apply_quantization(model)
elif config_dict["quantize"] is "sparsity":
    log.info("Applying Sparsity measurement")
    model = Sparsity.measure_sparsity(model)
elif config_dict["quantize"] is None:
    log.info("No model modification applied")
elif config_dict["quantize"] is not None:
    raise ValueError

if config_dict["optimizer"] == "sgd":
    log.info("Using SGD Optimizer")
    optimizer = SGD(learning_rate=config_dict["start_lr"], momentum=0.9)
elif config_dict["optimizer"] == "adam":
    log.info("Using Adam Optimizer")
    optimizer = Adam(learning_rate=config_dict["start_lr"], amsgrad=True)
else:
    raise ValueError

model.summary(line_length=100, print_fn=log.info)

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    # optimizer=tf.keras.optimizers.SGD(learning_rate=config_dict["start_lr"]),
    metrics=['accuracy'],
    experimental_run_tf_function=False
)

for key in ws_dict:
    if "load_from_quantized" == key and \
            ws_dict["load_from_quantized"] is not False and \
            ws_dict["load_from_quantized"] is not None:
        model = load_from_numpy(model, ws_dict["load_from_quantized"])

    if "load_from_checkpoint" == key and \
            ws_dict["load_from_checkpoint"] is not False and \
            ws_dict["load_from_checkpoint"] is not None:
        latest = tf.train.latest_checkpoint(ws_dict["load_from_checkpoint"])
        log.info("Loading model from checkpoint {}".format(latest))
        model.load_weights(latest)

# Prepare callbacks for model saving
checkpoint_model = ModelCheckpoint(filepath=run_dir + r"\save\model_{epoch:02d}",
                                   monitor='val_acc',
                                   verbose=0,
                                   save_freq='epoch',
                                   save_best_only=False,
                                   save_weights_only=True)

tensorboard_callback = TensorBoard(log_dir=run_dir + r"\tensorboard",
                                   histogram_freq=1,
                                   write_graph=True,
                                   write_images=False,
                                   update_freq='epoch',
                                   profile_batch=0,
                                   embeddings_freq=0,
                                   embeddings_metadata=None)

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1, mode='auto', cooldown=1, min_lr=0.5e-5)

fit_logger = FitLogger(frequency=ws_dict["logger_report_batch"])

callbacks = [checkpoint_model, tensorboard_callback, fit_logger, lr_reducer]

if checkpoint_weights is not None:
    callbacks.append(checkpoint_weights)

if config_dict["quantize"] is not None:
    sparsity_logger = SparsityLogger(frequency=ws_dict["sparsity_dump_period"],
                                     datasets=[dataset.get_train(False), dataset.get_validation()],
                                     savedir=run_dir + r"\sparsity_{}\\",
                                     num_stored_input=10,
                                     max_images=100,
                                     layer_cap=-1,
                                     verbose=False)
    callbacks.append(sparsity_logger)

train_dataset = dataset.get_train()
validation_dataset = dataset.get_validation()

model.fit(x=train_dataset,
          epochs=config_dict["epochs"],
          validation_data=validation_dataset,
          shuffle=True,
          callbacks=callbacks,
          verbose=ws_dict["keras_verbosity"],
          workers=32)

if config_dict["quantize"] is not None:
    max_images = -1
    log.info("Dumping final data for hw sparsity...")
    activ_sparsity_stats(model, validation_dataset, run_dir + r"\sparsity_final\\", num_stored_input=10, max_images=max_images)
    activ_sparsity_stats(model, train_dataset, run_dir + r"\sparsity_final\\", num_stored_input=10, max_images=max_images)
    get_weight_sparsity(model)


log.info("Dumping final data for hw sparsity...")