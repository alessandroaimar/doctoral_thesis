from collections import OrderedDict

#############
##  Hyperparameters Configuration
#############

# note that the order is used for generating the unique run id
config_dict = OrderedDict()
config_dict["network"] = "VGG19"
# config_dict["network"] = "ResNet_new"
config_dict["dataset"] = "ILSVRC2012"
# config_dict["network"] = "RoshamboNet"
# config_dict["dataset"] = "CIFAR10"

config_dict["start_lr"] = 0.0001
config_dict["batch_size"] = 32
config_dict["epochs"] = 2000

# config_dict["data_augmentation"] = True
# config_dict["network_kwargs"] = {"num_3x3_blocks": 2}
# config_dict["network_kwargs"] = {"depth": 101}
# config_dict["network_kwargs"] = {"pooling": "global", "regularize_activ": True}
# config_dict["network_kwargs"] = {"continue_train": "imagenet"}
# config_dict["quantize_prec"] = 16 #no effect
# config_dict["network_kwargs"] = {"regularize_activ": True}
config_dict["quantize"] = "tnh"
config_dict["pruning_policy"] = {"mode": "fixed", "target": 0.74}
config_dict["optimizer"] = "sgd"
config_dict["weight_precision"] = 8
config_dict["activation_precision"] = 16

#############
## WS Configuration
#############
ws_dict = OrderedDict()
ws_dict["sparsity_dump_period"] = 1
ws_dict["include_top"] = True  # True for classification false for feature extraction
ws_dict["keras_verbosity"] = 2
ws_dict["model_save_dir"] = r"D:\DL\models\VGG19\\"
ws_dict["datasets_root_dir"] = r"D:\DL\datasets\\"
ws_dict["override_dataset_read_dir"] = None
ws_dict["logger_report_batch"] = 1000

ws_dict["load_from_quantized"] = None

ws_dict["load_from_checkpoint"] = r"D:\DL\models\VGG19\21-10-2020__09-42-28__network_VGG19__dataset_ILSVRC2012__start_lr_0.0001__batch_size_32__epochs_2000__quantize_tnh__mode_fixed__target_0.74__optimizer_sgd__weight_precision_8__activation_precision_16__ID-4CCCD11F\save"
ws_dict["GPU"] = 0
