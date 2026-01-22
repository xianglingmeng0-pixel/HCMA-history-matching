import torch
from torch import nn
import torch.nn.functional as F

class R2(nn.Module):
    def __init__(self):
        super(R2, self).__init__()
        self.r2 = None
    def forward(self, y_pred, y_true):
        # 计算分子：残差平方和
        ss_res = torch.sum((y_true - y_pred) ** 2)
        # 计算分母：总离差平方和
        ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2) + 1e-8
        # 计算 R2
        self.r2 = 1 - (ss_res / ss_tot)
        return self.r2

class TrendConsistency(nn.Module):
    def __init__(self):
        super(TrendConsistency, self).__init__()
        self.tc_score = None
        self.epsilon = 1e-8 # 防止分母为0

    def forward(self, y_pred, y_true):
        delta_pred = y_pred[:, 1:, :] - y_pred[:, :-1, :]
        delta_true = y_true[:, 1:, :] - y_true[:, :-1, :]
        vec_pred = delta_pred.reshape(delta_pred.shape[0], -1)
        vec_true = delta_true.reshape(delta_true.shape[0], -1)
        self.tc_score = F.cosine_similarity(vec_pred, vec_true, dim=1, eps=self.epsilon)
        return torch.mean(self.tc_score)

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        self.rmse = None

    def forward(self, y_pred, y_true):
        # 计算均方误差 (MSE)
        mse = torch.mean((y_true - y_pred) ** 2)
        # 计算均方根误差 (RMSE)
        self.rmse = torch.sqrt(mse)
        return self.rmse


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.mae = None

    def forward(self, y_pred, y_true):
        # 计算平均绝对误差 (MAE)
        self.mae = torch.mean(torch.abs(y_true - y_pred))
        return self.mae


class KGE(nn.Module):
    def __init__(self, eps=1e-6):
        super(KGE, self).__init__()
        self.eps = eps  # 防止除零的小量
        self.kge = None  # 存储计算结果

    def forward(self, y_pred, y_true):
        # --- 计算相关系数 r ---
        # 去中心化
        y_true_centered = y_true - torch.mean(y_true)
        y_pred_centered = y_pred - torch.mean(y_pred)
        # 计算协方差
        covariance = torch.sum(y_true_centered * y_pred_centered)
        # 计算标准差
        std_true = torch.sqrt(torch.sum(y_true_centered ** 2) + self.eps)
        std_pred = torch.sqrt(torch.sum(y_pred_centered ** 2) + self.eps)
        # 皮尔逊相关系数
        r = covariance / (std_true * std_pred + self.eps)

        # --- 计算均值比率 α ---
        mu_true = torch.mean(y_true)
        mu_pred = torch.mean(y_pred)
        alpha = mu_pred / (mu_true + self.eps)  # 防止除零

        # --- 计算变异系数比率 β ---
        # 变异系数 = 标准差 / 均值
        cv_true = std_true / (mu_true + self.eps)
        cv_pred = std_pred / (mu_pred + self.eps)
        beta = cv_pred / (cv_true + self.eps)

        # --- 计算 KGE ---
        self.kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        return self.kge