import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import random

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

        self.path_files = glob('./datasets/%s/*' % (self.dataset_name))
        self.image_idx = 0
        self.num_images = len(self.path_files)


    def load_data(self, batch_size=1, is_testing=False):

        if self.image_idx == 0 and is_testing is False:
            random.shuffle(self.path_files)

        imgs_hr = []
        imgs_lr = []
        start = self.image_idx
        end = min(self.image_idx + batch_size, self.num_images)

        if end == self.num_images:
            self.image_idx = 0
        else:
            self.image_idx = end

        for img_idx in range(start, end):
            img_path = self.path_files[img_idx]
            img = self.imread(img_path)

            h, w = self.img_res
            low_h, low_w = int(h / 4), int(w / 4)

            img_hr = scipy.misc.imresize(img, self.img_res)
            img_lr = scipy.misc.imresize(img, (low_h, low_w))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 256
        imgs_lr = np.array(imgs_lr) / 256

        return imgs_hr, imgs_lr


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
