import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, batchnorm=True):
        super().__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=1))
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, n_classes, n_filters, dropout, batchnorm, n_channels):
        super().__init__()

        DropoutLayer = nn.Dropout2d

        # contracting path
        self.c1 = Conv2dBlock(n_channels, n_filters, 3, batchnorm)
        self.p1 = nn.MaxPool2d(2)
        self.drop1 = DropoutLayer(dropout)

        self.c2 = Conv2dBlock(n_filters, n_filters*2, 3, batchnorm)
        self.p2 = nn.MaxPool2d(2)
        self.drop2 = DropoutLayer(dropout)

        self.c3 = Conv2dBlock(n_filters*2, n_filters*4, 3, batchnorm)
        self.p3 = nn.MaxPool2d(2)
        self.drop3 = DropoutLayer(dropout)

        self.c4 = Conv2dBlock(n_filters*4, n_filters*8, 3, batchnorm)
        self.p4 = nn.MaxPool2d(2)
        self.drop4 = DropoutLayer(dropout)

        self.c5 = Conv2dBlock(n_filters*8, n_filters*16, 3, batchnorm)

        # expansive path
        self.u6 = nn.ConvTranspose2d(n_filters*16, n_filters*8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.drop6 = DropoutLayer(dropout)
        self.c6 = Conv2dBlock(n_filters*16, n_filters*8, 3, batchnorm)

        self.u7 = nn.ConvTranspose2d(n_filters*8, n_filters*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.drop7 = DropoutLayer(dropout)
        self.c7 = Conv2dBlock(n_filters*8, n_filters*4, 3, batchnorm)

        self.u8 = nn.ConvTranspose2d(n_filters*4, n_filters*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.drop8 = DropoutLayer(dropout)
        self.c8 = Conv2dBlock(n_filters*4, n_filters*2, 3, batchnorm)

        self.u9 = nn.ConvTranspose2d(n_filters*2, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.drop9 = DropoutLayer(dropout)
        self.c9 = Conv2dBlock(n_filters*2, n_filters, 3, batchnorm)

        self.out_conv = nn.Conv2d(n_filters, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # --- Down/contracting ---
        c1 = self.c1(x)
        p1 = self.drop1(self.p1(c1))

        c2 = self.c2(p1)
        p2 = self.drop2(self.p2(c2))

        c3 = self.c3(p2)
        p3 = self.drop3(self.p3(c3))

        c4 = self.c4(p3)
        p4 = self.drop4(self.p4(c4))

        c5 = self.c5(p4)

        # --- Up/expansive ---
        u6 = self.u6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        u6 = self.drop6(u6)
        c6 = self.c6(u6)

        u7 = self.u7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        u7 = self.drop7(u7)
        c7 = self.c7(u7)

        u8 = self.u8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        u8 = self.drop8(u8)
        c8 = self.c8(u8)

        u9 = self.u9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        u9 = self.drop9(u9)
        c9 = self.c9(u9)

        out = self.out_conv(c9)
        # out = self.sigmoid(out) In case of BCEWithLogitsLoss.
        return {"out": out}
