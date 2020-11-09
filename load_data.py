import h5py
import numpy as np
import os
import random
import cv2
from torch.utils.data import Dataset


def transformImg(dir_path, file_path):
    img = cv2.imread(os.path.join(dir_path, file_path))
    img = np.array(img).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class SRCnnTrainDataset(Dataset):
    def __init__(self, h5_file):
        super(SRCnnTrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, item):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][item] / 255., 0), np.expand_dims(f['hr'][item] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class SRCnnTestDataset(Dataset):
    def __init__(self, h5_file):
        super(SRCnnTestDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, item):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(item)][:, :] / 255, 0), np.expand_dims(f['hr'][str(item)][:, :] / 255, 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class SRGanDataset(Dataset):
    def __init__(self, lr_path, gt_path, in_memory=True, transform=None):
        super(SRGanDataset, self).__init__()

        self.lr_path = lr_path
        self.gt_path = gt_path
        self.in_memory = in_memory
        self.transform = transform

        self.lr_img = sorted(os.listdir(lr_path))
        self.gt_img = sorted(os.listdir(gt_path))

        if in_memory:
            self.lr_img = [transformImg(self.lr_path, lr) for lr in self.lr_img]
            self.gt_img = [transformImg(self.gt_path, gt) for gt in self.gt_img]

    def __getitem__(self, item):
        img_item = {}
        if self.in_memory:
            gt = self.gt_img[item].astype(np.float32)
            lr = self.lr_img[item].astype(np.float32)
        else:
            gt = transformImg(self.gt_path, self.gt_img[item]).astype(np.float32)
            lr = transformImg(self.lr_path, self.lr_img[item]).astype(np.float32)

        img_item['lr'] = lr
        img_item['gt'] = gt

        if self.transform is not None:
            img_item = self.transform(img_item)

        img_item['lr'] = img_item['lr'].transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['gt'] = img_item['gt'].transpose(2, 0, 1).astype(np.float32) / 255.

        return img_item['lr'], img_item['gt']

    def __len__(self):
        return len(self.lr_img)


class Crop(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, sample):
        lr_img, gt_img = sample['lr'], sample['gt']
        ih, iw = lr_img.shape[:2]

        ix = random.randrange(0, iw - self.patch_size + 1)
        iy = random.randrange(0, ih - self.patch_size + 1)

        tx = ix * self.scale
        ty = iy * self.scale

        lr_patch = lr_img[iy: iy + self.patch_size, ix: ix + self.patch_size]
        gt_patch = gt_img[ty: ty + (self.scale * self.patch_size), tx: tx + (self.scale * self.patch_size)]

        return {'lr': lr_patch, 'gt': gt_patch}
