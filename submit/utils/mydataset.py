import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, x, y, min_x=None, max_x=None, min_y=None, max_y=None, grid_size=[1,60,60],ts_feature=[50,8], if_log=False, if_seq=False, filter_sigma=30):

        self.x = x
        self.y = y

        if if_log:
            self.x = np.log(self.x)

        self.grid_size = grid_size
        self.ts_feature = ts_feature

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

        self.if_seq = if_seq
        self.pre_process()

    def pre_process(self):
        if self.min_x is None and self.max_x is None:
            self.min_x = np.min(self.x, axis=0)
            self.max_x = np.max(self.x, axis=0)
        self.x -= self.min_x
        self.x /= (self.max_x - self.min_x)
        self.x[self.x < 0] = 0
        self.x[self.x > 1] = 1
        self.x = np.nan_to_num(self.x)
        self.x = self.x.reshape(-1,
                                self.grid_size[2], self.grid_size[1], self.grid_size[0])
        self.x = np.transpose(self.x, (0, 3, 2, 1))
        if self.if_seq:
            self.x = self.x.reshape(-1, self.grid_size[0], self.grid_size[1]*self.grid_size[2])
        # self.x = (self.x - self.min_x) / (self.max_x - self.min_x + self.eps)
        self.y = self.y.reshape(self.y.shape[0], self.ts_feature[1], self.ts_feature[0])
        self.y= self.y.swapaxes(1, 2)
        if self.min_y is None and self.max_y is None:
            self.min_y = np.min(self.y, axis=0)
            self.max_y = np.max(self.y, axis=0)
            for i in range(self.ts_feature[1]):
                self.min_y[:, i] = self.min_y[:, i].min()
                self.max_y[:, i] = self.max_y[:, i].max()
        self.y = (self.y - self.min_y) / (self.max_y - self.min_y)
        self.y[self.y < 0] = 0
        self.y[self.y > 1] = 1
        self.y = np.nan_to_num(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.x[idx]),
                torch.FloatTensor(self.y[idx]))


