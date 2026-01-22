import math
import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba
from torchvision.transforms.functional import gaussian_blur

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # 形状: (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为缓冲区（不参与训练）

    def forward(self, x_):
        return x_ + self.pe[:, :x_.size(1)]

class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )

        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class mamba_not_h3(nn.Module):
    def __init__(self, in_channels, d_model, if_time_emb):
        super().__init__()
        self.conv = DepthwiseSeparableConv1D(in_channels, in_channels, kernel_size=2, stride=2, padding=0)
        self.mamba = Mamba(d_model//2)
        self.fixed_time_embedding = FixedPositionalEncoding(d_model // 2,
                                                         max_len=in_channels) if if_time_emb else nn.Identity()
    def forward(self, x):
        x = self.conv(x)
        x = x + self.fixed_time_embedding(x)
        x = self.mamba(x)
        return x


class our_model(nn.Module):
    def __init__(self, ts_feature=[253,20], grid_shape=[45,139,48], d_model=512, if_fixed_time_embedding=True,
                 n_encoder=1, n_decoder=2, k=2, encoder=None, n_feature=None):
        super().__init__()
        self.ts_feature = ts_feature
        self.grid_shape = grid_shape
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        if encoder is None:
            print('不使用预训练encoder')
            self.encoder = nn.ModuleList()
            self.n_feature = grid_shape[1]*grid_shape[2]
            for i in range(self.n_encoder):
                if i == 0:
                    self.encoder.append(
                        nn.Sequential(nn.Conv1d(self.grid_shape[0], self.ts_feature[0], k, k),
                                      nn.SiLU())
                        )
                else:
                    self.encoder.append(
                        nn.Sequential(nn.Conv1d(self.ts_feature[0], self.ts_feature[0], k, k),
                                      nn.SiLU())
                    )
                self.n_feature = self.n_feature // 2
        else:
            print('使用预训练encoder')
            self.n_feature = n_feature
            self.encoder = encoder

        self.input_linear = nn.Sequential(
            nn.Linear(self.n_feature, d_model),
            nn.Dropout(0.2),
        )
        self.h3_layer = nn.ModuleList()

        for i in range(self.n_decoder):
            self.h3_layer.append(mamba_not_h3(self.ts_feature[0], d_model, if_fixed_time_embedding))
            d_model = d_model // 2

        self.output_linear = nn.Sequential(
            nn.Linear(d_model, self.ts_feature[1]),
            nn.Sigmoid(),
        )
    def forward(self, x_):
        for i in range(self.n_encoder):
            x_ = self.encoder[i](x_)
        x_ = self.input_linear(x_)
        for i in range(self.n_decoder):
            x_ = self.h3_layer[i](x_)
        x_ = self.output_linear(x_)
        return x_
    def load_encoder(self, artifacts, device='cuda'):
        encoder = self.encoder
        encoder.load_state_dict(artifacts['encoder_state_dict'])
        encoder.to(device)
        return encoder, artifacts['n_feature']



