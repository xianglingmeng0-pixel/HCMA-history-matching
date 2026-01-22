import logging
import time

import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from module import our_model
from .mydataset import MyDataset_nolabel

logging.getLogger().setLevel(logging.INFO)

import torch
from torch import nn
from .metrics import R2, RMSE, TrendConsistency
import torch.nn.functional as F
import tqdm


class TVLoss(nn.Module):
    def __init__(self, lambda_tv=0.1):
        super(TVLoss, self).__init__()
        self.lambda_tv = lambda_tv

    def forward(self, y):
        diff = y[:, 1:] - y[:, :-1]
        loss = self.lambda_tv * torch.mean(torch.abs(diff))
        return loss

def train(epoches, model, device, train_loader, val_loader, lr=0.0001, use_diff=False, checkpoint=None):
    print('正在训练代理模型')
    Loss1 = None
    if use_diff:
        Loss = nn.MSELoss()
        Loss1 = TVLoss()
    else:
        Loss = nn.MSELoss()

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    reduce_lr = ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控指标是最小化
        factor=0.5,  # 学习率乘以0.5
        patience=10,  # 10个epoch没有改善则降低学习率
        verbose=True,  # 打印信息
        threshold=0.0001,  # 相当于min_delta
        cooldown=0,  # 降低学习率后立即重新开始监控
        min_lr=0.00001,  # 最小学习率
        eps=1e-08  # 防止除零
    )

    train_loss_all = []

    train_r2_all = []
    train_rmse_all = []
    val_loss_all = []

    val_r2_all = []
    val_rmse_all = []

    best_r2 = 0.0
    best_rmse = 0.0
    best_model = None
    best_r2_list = []
    best_epoch = 0

    since = time.time()

    for epoch in range(epoches):
        train_loss = 0.0
        train_r2 = 0.0
        train_rmse = 0.0
        train_num = 0
        val_loss = 0.0
        val_r2 = 0.0
        val_rmse = 0.0
        val_num = 0
        loop = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)
        model.train()
        for data in loop:
            b_x = data[0].to(device)
            b_y = data[1].to(device)
            output = model(b_x)
            if Loss1:
                loss = Loss(output, b_y) + Loss1(output)
            else:
                loss = Loss(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            score = R2()(output, b_y)
            train_r2 += float(score) * b_x.size(0)
            train_num += b_x.size(0)
            rmse = RMSE()(output, b_y)
            train_rmse += float(rmse) * b_x.size(0)
            loop.set_description(f'Epoch [{epoch + 1}/{epoches}]')
            loop.set_postfix(loss=loss.item(), r2=float(score))
        model.eval()
        val_r2_list = []
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                b_x = data[0].to(device)
                b_y = data[1].to(device)
                output = model(b_x)
                loss = Loss(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_num += b_x.size(0)
                # metric
                r2 = R2()(output, b_y)
                val_r2_list.append(float(r2))
                val_r2 += float(r2) * b_x.size(0)
                rmse = RMSE()(output, b_y)
                val_rmse += float(rmse) * b_x.size(0)

        reduce_lr.step(val_loss/val_num)
        current_lr = optimizer.param_groups[0]['lr']
        train_loss_all.append(train_loss / train_num)
        train_r2_all.append(train_r2 / train_num)
        train_rmse_all.append(train_rmse / train_num)
        val_loss_all.append(val_loss / val_num)
        val_r2_all.append(val_r2 / val_num)
        val_rmse_all.append(val_rmse / val_num)
        logging.info('=============================================================================❀')
        logging.info(f'|| epoch: {epoch+1}')
        logging.info('|| train loss: {:.4f},'
                     'train R2: {:.4f}, '
                     'Val R2: {:.4f}, '
                     'Val RMSE: {:.4f}, '
                     'LR: {:.6f}'.
                     format(train_loss_all[-1], train_r2_all[-1], val_r2_all[-1], val_rmse_all[-1], current_lr))
        logging.info('=============================================================================❀')
        if val_r2_all[-1] > best_r2:
            best_r2 = val_r2_all[-1]
            best_rmse = val_rmse_all[-1]
            best_model = model
            best_r2_list = val_r2_list
            best_epoch = epoch
    time_use = time.time() - since
    logging.info(f"best_r2: {best_r2} time_use: {time_use // 60:.0f}m{time_use % 60:.0f}s ")
    torch.cuda.empty_cache()
    return best_model, optimizer, best_rmse, best_r2, train_loss_all, val_loss_all, train_r2_all, train_rmse_all, val_r2_all, val_rmse_all, time_use, best_r2_list, best_epoch