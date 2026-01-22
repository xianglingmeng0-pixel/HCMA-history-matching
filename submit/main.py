import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from sklearn.model_selection import KFold
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from module import IVIT, our_model, HRCN
from utils.mydataset import MyDataset
from utils.train import train
from utils.utils import check_data, init_experiment, log_result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = './data/labeled_test.h5'
grid_shape = [48, 139, 45]
ts_feature = [253, 40]

with h5py.File(file_path, 'r') as f:
    x = f['x'][:]
    y = f['y'][:]

k = 2
d_model = 512
layer_num = 2
batch_size = 16

if_fixed_time_embedding = True
epoches = 200
seed_list = [42]
model_name = 'mymodel'

kf = KFold(n_splits=10, shuffle=True, random_state=seed_list[0])

fold_indices = []
for train_idx, val_idx in kf.split(x):
    fold_indices.append((train_idx, val_idx))

for i, (train_indices, val_indices) in enumerate(fold_indices):
    experiment_dir = f'./result/experiment/test/fold_{i+1}/'
    init_experiment(seed_list, epoches, model_name, batch_size, experiment_dir)
    train_x = x[train_indices]
    train_y = y[train_indices]
    val_x = x[val_indices]
    val_y = y[val_indices]

    train_dataset = MyDataset(x=train_x, y=train_y,
                              ts_feature=ts_feature, grid_size=grid_shape, if_seq=True)
    val_dataset = MyDataset(x=val_x, y=val_y,
                            min_x=train_dataset.min_x, max_x=train_dataset.max_x,
                            min_y=train_dataset.min_y, max_y=train_dataset.max_y,
                            ts_feature=ts_feature, grid_size=grid_shape, if_seq=True)
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = our_model.our_model(
                                  ts_feature=ts_feature,
                                  grid_shape=grid_shape,
                                  d_model=d_model,
                                  if_fixed_time_embedding=if_fixed_time_embedding,
                                  n_encoder=1,
                                  n_decoder=2,
                                  k=k).to(device)
    (best_model,
     optimizer,
     best_rmse,
     best_r2,
     train_loss_all,
     val_loss_all,
     train_r2_all,
     train_rmse_all,
     val_r2_all,
     val_rmse_all,
     time_use,
     best_r2_list,
     best_epoch) \
        = train(epoches=epoches,
                model=model,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                lr=0.0001)

    result_dir = experiment_dir+'result/'
    log_result(result_dir, best_model, optimizer, best_r2, best_rmse, train_loss_all, val_loss_all, train_r2_all, train_rmse_all, val_r2_all, val_rmse_all,
               time_use, best_r2_list, best_epoch)

