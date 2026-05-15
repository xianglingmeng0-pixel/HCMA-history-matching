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
        out_channels = int(in_channels * compression)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(x, inplace=True)))


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=(1, 3, 5)):
        super().__init__()
        self.pool_sizes = tuple(pool_sizes)

    def forward(self, x):
        features = [
            F.adaptive_max_pool2d(x, (size, size)).flatten(start_dim=1)
            for size in self.pool_sizes
        ]
        return torch.cat(features, dim=1)


class RecurrentDecoder(nn.Module):
    def __init__(self, ts_feature, d_model=256, num_layers=2, hidden_size=128):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=int(d_model),
            hidden_size=int(hidden_size),
            num_layers=max(1, int(num_layers)),
            dropout=0.2 if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(ts_feature[1])),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("sigmoid"))
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "bias" in name:
                        nn.init.zeros_(param)
                    elif "weight" in name:
                        nn.init.xavier_uniform_(param)

    def forward(self, x):
        x, _ = self.rnn(x)
        return self.output(x)


class HRCN(nn.Module):
    """Hybrid recurrent convolutional network for reservoir production prediction."""

    def __init__(
        self,
        growth_rate=16,
        in_channels=1,
        ts_feature=(50, 8),
        n_encoder=1,
        n_decoder=2,
        d_model=256,
        decoder_hidden_size=128,
    ):
        super().__init__()
        self.ts_feature = tuple(ts_feature)
        self.n_encoder = int(n_encoder)

        self.stem = nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3, bias=False)
        feature_channels = 48
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for _ in range(self.n_encoder):
            block = DenseBlock(feature_channels, num_layers=3, growth_rate=growth_rate)
            self.blocks.append(block)
            feature_channels += 3 * growth_rate
            self.transitions.append(TransitionLayer(feature_channels))
            feature_channels = int(feature_channels * 0.5)

        self.spp = SpatialPyramidPooling(pool_sizes=(1, 3, 5))
        self.projection = nn.LazyLinear(int(d_model))

        self.decoder = RecurrentDecoder(
            ts_feature=self.ts_feature,
            d_model=d_model,
            num_layers=n_decoder,
            hidden_size=decoder_hidden_size,
        )

    def forward(self, x):
        x = self.stem(x)
        for block, transition in zip(self.blocks, self.transitions):
            x = transition(block(x))

        x = self.projection(self.spp(x))
        x = x.unsqueeze(1).expand(-1, self.ts_feature[0], -1)
        return self.decoder(x)
