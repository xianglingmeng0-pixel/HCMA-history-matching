import torch
import torch.nn as nn
from torchvision.transforms import transforms
from .DenseNet import DenseBlock, TransitionLayer, SpatialPyramidPooling

class ResidualBlockV2(nn.Module):
    def _init_weights(self, *args, **kwargs) -> None:
        pass

    def __init__(self, filters) -> None:
        super(ResidualBlockV2, self).__init__()
        self.BN_module_list1 = nn.ModuleList()
        self.BN_module_list2 = nn.ModuleList()
        self.cnn_module_list1 = nn.ModuleList()
        self.cnn_module_list2 = nn.ModuleList()
        for i in filters:
            self.BN_module_list1.append(nn.LazyBatchNorm2d())
            self.BN_module_list2.append(nn.LazyBatchNorm2d())
            self.cnn_module_list1.append(nn.LazyConv2d(out_channels=i,
                                                       kernel_size=3, stride=1, padding=1))
            self.cnn_module_list2.append(nn.LazyConv2d(out_channels=i,
                                                       kernel_size=3, stride=1, padding=1))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, step):
        y = self.BN_module_list1[step](x)
        y = self.leaky_relu(y)
        y = self.cnn_module_list1[step](y)
        y = self.BN_module_list2[step](y)
        y = self.leaky_relu(y)
        y = self.cnn_module_list2[step](y)
        y = shortcut(x, y)
        return y


def shortcut(x, y):
    input_shape = x.shape
    residual_shape = y.shape
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    stride_height = int(round(input_shape[3] / residual_shape[3]))
    equal_channels = input_shape[1] == residual_shape[1]
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        x = nn.LazyConv2d(out_channels=residual_shape[1],
                          kernel_size=(1, 1),
                          stride=(stride_width, stride_height),
                          padding=0).cuda()(x)
    return torch.add(x, y)

class HRCN(nn.Module):
    def __init__(self, growth_rate=16, in_channels=1, ts_feature=[50,8]):
        super(HRCN, self).__init__()
        self.ts_feature = ts_feature
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, 7, stride=2, padding=3, bias=False),
        )
        # Dense Blocks配置 (每个block包含的层数)
        block_config = [3, 3, 3]  # 每个dense block包含3个dense layer

        # 构建4个Dense Blocks和4个Transition Layers
        in_channels = 48
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            # Dense Block
            block = DenseBlock(in_channels, num_layers, growth_rate)
            self.blocks.append(block)
            in_channels += num_layers * growth_rate
            transition = TransitionLayer(in_channels)
            self.transitions.append(transition)
            in_channels = int(in_channels * 0.5)  # 压缩系数
        self.spp = SpatialPyramidPooling(pool_sizes=[1, 3, 5])
        self.Linear_layer = nn.Linear(1680, 512)
        self.rnn_Module = RNNModule(ts_feature)

    def _init_weights(self, blocks: list) -> None:
        pass

    def forward(self, x):
        x = self.features(x)
        for i in range(3):
            x = self.blocks[i](x)
            x = self.transitions[i](x)
        x = self.spp(x)
        x = self.Linear_layer(x)
        x = x.unsqueeze(1)
        x = x.expand(-1, self.ts_feature[0], -1)
        x = self.rnn_Module(x)
        return x


class CNNModule(nn.Module):
    def __init__(self, residual_block_n_filters=[64, 128, 256, 512], ts_feature=[253,40]):
        super(CNNModule, self).__init__()
        self.residual_block_n_filters = residual_block_n_filters
        self.ts_feature = ts_feature
        self.MP = nn.MaxPool2d(kernel_size=2,
                               stride=2)
        self.BN = nn.LazyBatchNorm2d()
        self.leaky_relu = nn.LeakyReLU()
        self.AP = nn.AdaptiveAvgPool2d(1)
        self.DP = nn.Dropout2d(0.2)
        self.LN = nn.Sequential(
            nn.Linear(residual_block_n_filters[-1], 200),
            nn.ReLU()
        )
        self.residual_block = ResidualBlockV2(self.residual_block_n_filters)
        init_blocks = [self.LN]
        self._init_weights(init_blocks)

    @staticmethod
    def _init_weights(blocks):
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor):
        for step in range(len(self.residual_block_n_filters)):
            x = self.residual_block(x, step)
            if step != len(self.residual_block_n_filters) - 1:
                x = self.MP(x)
        x = self.BN(x)
        x = self.leaky_relu(x)
        x = self.AP(x)
        x = self.DP(x)
        x = x.reshape(x.shape[0], -1, x.shape[1])
        x = self.LN(x)
        x = x.repeat(1, self.ts_feature[0], 1)
        return x


class RNNModule(nn.Module):
    def __init__(self, ts_feature):
        super(RNNModule, self).__init__()
        self.rnn = nn.LSTM(input_size=512,
                           hidden_size=256,
                           num_layers=3,
                           dropout=0.2,
                           bidirectional=True,
                           batch_first=True)
        self.relu = nn.ReLU()
        self.LN = nn.Sequential(
            nn.Linear(256 * 2, ts_feature[1]),
            nn.Sigmoid()
        )
        init_blocks = [self.rnn, self.LN]
        self._init_weights(init_blocks)

    @staticmethod
    def _init_weights(blocks):
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("sigmoid"))
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
                    for name, param in module.named_parameters():
                        if 'bias' in name:
                            nn.init.constant_(param, 0.0)
                        elif 'weight' in name:
                            nn.init.xavier_uniform_(param, gain=1)

    def forward(self, x: torch.Tensor):
        x, (ht, ct) = self.rnn(x)
        x = self.relu(x)
        x = self.LN(x)
        return x
