import math
import torch
import torch.nn as nn
from .DenseNet import DenseBlock, TransitionLayer, SpatialPyramidPooling

class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # 形状: (L, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        self.register_buffer('pe', pe.unsqueeze(0))  # 注册为缓冲区（不参与训练）

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class IVIT(nn.Module):
    def __init__(self, growth_rate=16, in_channels=1, ts_feature=[50,8]):
        super().__init__()
        self.ts_feature = ts_feature

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, 7, stride=2, padding=3, bias=False),
        )
        self.pos_encoder = FixedPositionalEncoding(d_model=512, max_len=253)

        block_config = [4,4,4]


        in_channels = 48
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(in_channels, num_layers, growth_rate)
            self.blocks.append(block)
            in_channels += num_layers * growth_rate
            transition = TransitionLayer(in_channels)
            self.transitions.append(transition)
            in_channels = int(in_channels * 0.5)
        self.spp = SpatialPyramidPooling(pool_sizes=[1, 3, 5])

        self.Linear_layer = nn.Linear(2170, 512)
        self.conv1 = nn.Conv1d(self.ts_feature[0], self.ts_feature[0], 2, 2)
        self.Tran1 = Transformer_block(256, 8)
        self.conv2 = nn.Conv1d(self.ts_feature[0], self.ts_feature[0], 2, 2)
        self.Tran2 = Transformer_block(128, 8)
        self.conv3 = nn.Conv1d(self.ts_feature[0], self.ts_feature[0], 2, 2)
        self.Tran3 = Transformer_block(64, 8)
        self.output_linear = nn.Sequential(
            nn.Linear(64, self.ts_feature[1]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        for i in range(3):
            x = self.blocks[i](x)
            x = self.transitions[i](x)

        x = self.spp(x)

        x = self.Linear_layer(x)
        x = x.unsqueeze(1)
        x = x.expand(-1, self.ts_feature[0], -1)
        x = x + self.pos_encoder(x)
        x = self.conv1(x)
        x = self.Tran1(x)
        x = self.conv2(x)
        x = self.Tran2(x)
        x = self.conv3(x)
        x = self.Tran3(x)
        x = self.output_linear(x)
        return x

class Transformer_block(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
    def forward(self, x):
        return self.transformer_encoder(x)


