import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, activation=True):

        super(ConvLayer, self).__init__()

        self.activation = activation
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=self.padding, bias=True)

    def forward(self, x):

        if self.activation:
            return F.relu(self.conv(x))
        else:
            return self.conv(x)


class MultiScaleResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(MultiScaleResidualBlock, self).__init__()

        self.conv5_1 = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=5)
        self.conv3_1 = ConvLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=3)

        self.conv5_2 = ConvLayer(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=5)
        self.conv3_2 = ConvLayer(in_channels=in_channels * 2, out_channels=out_channels * 2, kernel_size=3)

        self.bottleneck = ConvLayer(in_channels=in_channels * 4, out_channels=out_channels, kernel_size=1, activation=False)

    def forward(self, x):

        P1 = self.conv5_1(x)
        S1 = self.conv3_1(x)

        P2 = self.conv5_2(torch.cat([P1, S1], 1))
        S2 = self.conv3_2(torch.cat([P1, S1], 1))

        S = self.bottleneck(torch.cat([P2, S2], 1))

        return x + S


class ReconstructionNetwork(nn.Module):

    def __init__(self, scale, in_features):

        super(ReconstructionNetwork, self).__init__()

        self.conv1 = ConvLayer(in_channels=in_features, out_channels=in_features * scale * scale, kernel_size=3)
        self.shuffle = nn.PixelShuffle(scale)
        self.conv2 = ConvLayer(in_channels=in_features, out_channels=3, kernel_size=3)

    def forward(self, x):

        output = F.relu(self.conv1(x))
        output = self.shuffle(output)
        return F.relu(self.conv2(output))


class MultiScaleResidualNetwork(nn.Module):

    def __init__(self, scale, res_blocks, res_in_features, res_out_features):

        super(MultiScaleResidualNetwork, self).__init__()

        self.sub_mean = MeanShift()
        self.add_mean = MeanShift(sign=1)

        self.conv = ConvLayer(in_channels=3, out_channels=res_out_features, kernel_size=3)

        self.residual_blocks = nn.ModuleList([MultiScaleResidualBlock(in_channels=res_in_features, out_channels=res_out_features) for _ in range(res_blocks)])

        self.feature_fusion = ConvLayer(in_channels=res_in_features * (res_blocks + 1), out_channels=res_out_features, kernel_size=1)

        self.reconstruction_layer = ReconstructionNetwork(scale=scale, in_features=res_in_features)

    def forward(self, x):

        x = self.sub_mean(x)
        output = F.relu(self.conv(x))
        residual_conv = output

        block_outputs = [residual_conv]
        for block in self.residual_blocks:
            output = block(output)
            block_outputs.append(output)

        output = F.relu(self.feature_fusion(torch.cat(block_outputs, 1)))
        hr_image = self.reconstruction_layer(output)
        return self.add_mean(hr_image)



