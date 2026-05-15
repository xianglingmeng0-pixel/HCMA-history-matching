import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()
        hidden_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList(
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(num_layers)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, compression=0.5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, int(in_channels * compression), kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(x, inplace=True)))


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=(1, 3, 5)):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)

    def forward(self, x):
        return torch.cat(
            [F.adaptive_max_pool2d(x, (size, size)).flatten(start_dim=1) for size in self.pool_sizes],
            dim=1,
        )


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=50):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


def infer_transformer_nhead(d_model):
    for nhead in (8, 4, 2, 1):
        if d_model % nhead == 0:
            return nhead
    return 1


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=None, dim_feedforward=256):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=infer_transformer_nhead(int(d_model)) if nhead is None else int(nhead),
            dim_feedforward=max(int(dim_feedforward), int(d_model)),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x):
        return self.encoder(x)


class IVIT(nn.Module):
    """Dense CNN encoder with a transformer-style temporal decoder."""

    def __init__(
        self,
        growth_rate=16,
        in_channels=1,
        ts_feature=(50, 8),
        n_encoder=3,
        n_decoder=3,
        d_model=512,
    ):
        super().__init__()
        self.ts_feature = tuple(ts_feature)
        self.n_encoder = int(n_encoder)
        self.n_decoder = max(1, int(n_decoder))

        self.stem = nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3, bias=False)
        self.positional_encoding = FixedPositionalEncoding(d_model=d_model, max_len=self.ts_feature[0])

        feature_channels = 48
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for _ in range(self.n_encoder):
            block = DenseBlock(feature_channels, num_layers=4, growth_rate=growth_rate)
            self.blocks.append(block)
            feature_channels += 4 * growth_rate
            self.transitions.append(TransitionLayer(feature_channels))
            feature_channels = int(feature_channels * 0.5)

        self.spp = SpatialPyramidPooling(pool_sizes=(1, 3, 5))
        self.input_projection = nn.LazyLinear(int(d_model))

        self.temporal_convs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        current_dim = int(d_model)
        for _ in range(self.n_decoder):
            next_dim = max(1, current_dim // 2)
            self.temporal_convs.append(nn.Conv1d(self.ts_feature[0], self.ts_feature[0], kernel_size=2, stride=2))
            self.transformers.append(TransformerBlock(next_dim))
            current_dim = next_dim

        self.output = nn.Sequential(
            nn.Linear(current_dim, self.ts_feature[1]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)
        for block, transition in zip(self.blocks, self.transitions):
            x = transition(block(x))

        x = self.input_projection(self.spp(x))
        x = x.unsqueeze(1).expand(-1, self.ts_feature[0], -1)
        x = self.positional_encoding(x)
        for conv, transformer in zip(self.temporal_convs, self.transformers):
            x = transformer(conv(x))
        return self.output(x)
