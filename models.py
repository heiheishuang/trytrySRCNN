from torch import nn
import torch.nn.functional as F


class NetSrcnn(nn.Module):
    def __init__(self, num_channels=1):
        super(NetSrcnn, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels=64, kernal_size=3, n=64, s=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, kernal_size, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.pr = nn.PReLU()
        self.conv2 = nn.Conv2d(n, n, kernal_size, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        f = self.pr(self.bn1(self.conv1(x)))
        f = self.bn2(self.conv2(f))
        return f + x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.pr = nn.PReLU()

    def forward(self, x):
        return self.pr(self.shuffler(self.conv(x)))


class SRResNet(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor=1):
        super(SRResNet, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
        self.pr1 = nn.PReLU()

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), ResidualBlock())

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        for i in range(self.upsample_factor // 2):
            self.add_module('upsample' + str(i + 1), UpsampleBlock(64, 256))

        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.pr1(self.conv1(x))

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        f = self.bn2(self.conv2(y))
        x = f + x

        for i in range(self.upsample_factor // 2):
            x = self.__getattr__('upsample' + str(i + 1))(x)

        return self.conv3(x)
