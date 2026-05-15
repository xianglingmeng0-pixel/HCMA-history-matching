import math

import torch
import torch.nn as nn


def _get_mamba_class():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
    except ImportError as exc:
        raise ImportError(
            "test_model requires mamba-ssm. Install it or choose a model that does not use Mamba."
        ) from exc
    return Mamba


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=253):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, channels, kernel_size=2, stride=2):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, stride=stride, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.pointwise(self.depthwise(x))))

class MambaBlock(nn.Module):
    def __init__(self, seq_channels, d_model, use_time_embedding=True):
        super().__init__()
        Mamba = _get_mamba_class()
        self.conv = DepthwiseSeparableConv1d(seq_channels)
        self.mamba = Mamba(d_model // 2)
        self.time_embedding = (
            FixedPositionalEncoding(d_model // 2, max_len=seq_channels)
            if use_time_embedding else nn.Identity()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.mamba(x + self.time_embedding(x))


def build_mamba_decoder(seq_channels, d_model, n_decoder, use_time_embedding=True):
    layers = nn.ModuleList()
    current_dim = int(d_model)
    for _ in range(int(n_decoder)):
        layers.append(MambaBlock(seq_channels, current_dim, use_time_embedding))
        current_dim = current_dim // 2
    return layers, current_dim


class HCMA_1dcnn(nn.Module):
    """1D-CNN encoder plus Mamba temporal decoder."""

    def __init__(
        self,
        ts_feature=(253, 20),
        grid_shape=(45, 139, 48),
        d_model=512,
        use_time_embedding=True,
        n_encoder=1,
        n_decoder=2,
        kernel_size=2,
        encoder=None,
        n_feature=None,
        encoder_out_channels=None,
    ):
        super().__init__()
        self.ts_feature = tuple(ts_feature)
        self.grid_shape = tuple(grid_shape)
        self.encoder_out_channels = self.ts_feature[0] if encoder_out_channels is None else int(encoder_out_channels)

        if encoder is None:
            self.encoder = nn.ModuleList()
            self.n_feature = self.grid_shape[1] * self.grid_shape[2]
            for i in range(int(n_encoder)):
                in_channels = self.grid_shape[0] if i == 0 else self.encoder_out_channels
                self.encoder.append(
                    nn.Sequential(
                        nn.Conv1d(in_channels, self.encoder_out_channels, kernel_size, kernel_size),
                        nn.SiLU(),
                    )
                )
                self.n_feature = self.n_feature // kernel_size
        else:
            self.encoder = encoder
            self.n_feature = int(n_feature)

        self.input_projection = nn.Sequential(
            nn.Linear(self.n_feature, d_model),
            nn.Dropout(0.2),
        )
        self.decoder, out_dim = build_mamba_decoder(
            self.encoder_out_channels,
            d_model,
            n_decoder,
            use_time_embedding=use_time_embedding,
        )
        self.output = nn.Sequential(nn.Linear(out_dim, self.ts_feature[1]), nn.Sigmoid())

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        x = self.input_projection(x)
        for layer in self.decoder:
            x = layer(x)
        return self.output(x)

    def load_encoder(self, checkpoint, device="cpu"):
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.encoder.to(device)
        return self.encoder, checkpoint["n_feature"]
