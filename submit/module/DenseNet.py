import torch
from torch import nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x)))
        out = self.conv2(F.relu(self.norm2(out)))
        return torch.cat([x, out], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        out_channels = int(in_channels * compression)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(x))
        return self.pool(x)

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=None):
        """
        Args:
            pool_sizes (list): 池化层对应的输出尺寸（如原论文使用[1x1, 2x2, 4x4]）
        """
        super().__init__()
        if pool_sizes is None:
            pool_sizes = [1, 4, 16]
        self.pool_sizes = pool_sizes

    def forward(self, x):
        """
        Input shape:  (B, C, H, W)
        Output shape: (B, C * sum(pool_sizes^2))
        """
        features = []
        for pool_size in self.pool_sizes:
            # 计算自适应池化的输出尺寸
            h = w = pool_size
            pooled = F.adaptive_max_pool2d(x, (h, w))
            features.append(pooled.view(x.size(0), -1))  # 展平

        # 拼接多尺度特征
        return torch.cat(features, dim=1)