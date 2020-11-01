import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, item):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][item] / 255., 0), np.expand_dims(f['hr'][item] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class TestDataset(Dataset):
    def __init__(self, h5_file):
        super(TestDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, item):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(item)][:, :] / 255, 0), np.expand_dims(f['hr'][str(item)][:, :] / 255, 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
