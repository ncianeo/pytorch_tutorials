import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """SRCNN Network"""
    def __init__(self, cnum, ratio):
        # initiate as an nn.Module object
        super(SRCNN, self).__init__()
        # first convolution
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, cnum*ratio**2, 3, 1, 1),
                nn.ReLU(),
                )
        # pixel shuffle to upsample
        self.pixelshuffle = nn.PixelShuffle(ratio)
        # non-linear mapping
        self.conv2 = nn.Sequential(
                nn.Conv2d(cnum, cnum//2, 1, 1, 0),
                nn.ReLU(),
                )
        # final convolution
        self.final = nn.Sequential(
                nn.Conv2d(cnum//2, 3, 3, 1, 1),
                nn.Tanh(),
                )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixelshuffle(x)
        x = self.conv2(x)
        x = self.final(x)
        return x


if __name__ == '__main__':
    net = SRCNN(64, 2)
    x = torch.randn(1, 3, 128, 128)
    y = net(x)
    print(x.shape)
    print(y.shape)

