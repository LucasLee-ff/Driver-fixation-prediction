import torch.nn as nn


def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class ConvEncoder(nn.Module):
    def __init__(self):
        n, m = 24, 3

        super(ConvEncoder, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.convd1 = conv3x3(1 * m, 1 * n)
        self.convd2 = conv3x3(1 * n, 2 * n)
        self.convd3 = conv3x3(2 * n, 4 * n)
        self.convd4 = conv3x3(4 * n, 8 * n)
        self.convd5 = conv3x3(8 * n, 16 * n)

    def forward(self, x):
        features = []

        x = self.convd1(x)  # 320, 192, 24
        features.append(x)

        x = self.maxpool(x)
        x = self.convd2(x)  # 160, 96, 48
        features.append(x)

        x = self.maxpool(x)
        x = self.convd3(x)  # 80, 48, 96
        features.append(x)

        x = self.maxpool(x)
        x = self.convd4(x)  # 40, 24, 192
        features.append(x)

        x = self.maxpool(x)
        x = self.convd5(x)  # 20, 12, 384

        return x, features

