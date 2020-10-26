from datasets.baseDataset import baseDataset
from config import ws_dict
import logging
log = logging.getLogger()

class Roshambo(baseDataset):

    def __init__(self):
        super().__init__()

        self.dirs = {}
        if ws_dict["override_dataset_read_dir"] is None:
            log.warning("Roshambo dataset has a single test/val dataset")
            self.dirs["train"] = ws_dict["datasets_root_dir"] + r"\Roshambo\tf_train\train\\"
            self.dirs["val"] = ws_dict["datasets_root_dir"] + r"\Roshambo\tf_train\test\\"
            self.dirs["test"] = ws_dict["datasets_root_dir"] + r"\Roshambo\tf_train\test\\"
        else:
            raise NotImplementedError("Override of sub path not implemented")


        self.files = self.get_files_in_dirs(self.dirs)
        self.input_shape = (64, 64, 1)
        self.num_classes = 4


