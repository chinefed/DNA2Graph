import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    def __init__(self, scale_factor, **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, **self.kwargs)


class Concatenate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x_list):
        return torch.cat(x_list, dim=self.dim)


class Conv2dBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.kwargs = kwargs

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activ = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv1 = Conv2dBN(
            in_channels,
            out_channels,
            activation=True,
            kernel_size=3,
            stride=stride,
            padding=1,
            padding_mode='zeros',
            bias=False
        )

        self.conv2 = Conv2dBN(
            out_channels,
            out_channels,
            activation=False,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            bias=False
        )
        self.relu = nn.ReLU()

        if in_channels != out_channels or stride > 1:
            self.shortcut = Conv2dBN(
                in_channels,
                out_channels,
                activation=False,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        x_res = self.conv1(x)
        x_res = self.conv2(x_res)
        x = self.shortcut(x) + x_res
        x = self.relu(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.upconv = nn.Sequential(
            Upsample(scale_factor=scale_factor, mode='nearest'),
            Conv2dBN(
                in_channels,
                out_channels,
                activation=True,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='zeros',
                bias=False
            )
        )

        self.cat = Concatenate(dim=1)

        self.conv1 = Conv2dBN(
            int(2 * out_channels),
            out_channels,
            activation=True,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            bias=False
        )
        self.conv2 = Conv2dBN(
            out_channels,
            out_channels,
            activation=True,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='zeros',
            bias=False
        )

    def forward(self, x, xi):
        x = self.upconv(x)
        x = self.cat([x, xi])
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

        self.stem = Conv2dBN(
            1,
            int(64 * alpha),
            activation=True,
            kernel_size=7,
            stride=2,
            padding=3,
            padding_mode='zeros',
            bias=False
        )

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.block1 = EncoderBlock(int(64 * alpha), int(64 * alpha), stride=1)
        self.block2 = EncoderBlock(int(64 * alpha), int(64 * alpha), stride=1)

        self.block3 = EncoderBlock(int(64 * alpha), int(128 * alpha), stride=2)
        self.block4 = EncoderBlock(int(128 * alpha), int(128 * alpha), stride=1)

        self.block5 = EncoderBlock(int(128 * alpha), int(256 * alpha), stride=2)
        self.block6 = EncoderBlock(int(256 * alpha), int(256 * alpha), stride=1)

        self.block7 = EncoderBlock(int(256 * alpha), int(512 * alpha), stride=2)
        self.block8 = EncoderBlock(int(512 * alpha), int(512 * alpha), stride=1)

    def forward(self, x):
        x0 = self.stem(x)

        x = self.maxpool(x0)

        x = self.block1(x)
        x1 = self.block2(x)

        x = self.block3(x1)
        x2 = self.block4(x)

        x = self.block5(x2)
        x3 = self.block6(x)

        x = self.block7(x3)
        x = self.block8(x)

        return x0, x1, x2, x3, x


class Decoder(nn.Module):
    def __init__(self, alpha=1.0, return_logits=True):
        super().__init__()
        self.alpha = alpha
        self.return_logits = return_logits

        self.block1 = DecoderBlock(int(512 * alpha), int(256 * alpha), scale_factor=2)
        self.block2 = DecoderBlock(int(256 * alpha), int(128 * alpha), scale_factor=2)
        self.block3 = DecoderBlock(int(128 * alpha), int(64 * alpha), scale_factor=2)
        self.block4 = DecoderBlock(int(64 * alpha), int(64 * alpha), scale_factor=2)

        self.upconv = nn.Sequential(
            Upsample(scale_factor=2, mode='nearest'),
            Conv2dBN(
                int(64 * alpha),
                int(64 * alpha),
                activation=True,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='zeros',
                bias=False
            )
        )

        self.conv = nn.Conv2d(
            int(64 * alpha),
            1,
            kernel_size=1,
            padding=0,
            bias=False
        )

    def forward(self, x0, x1, x2, x3, x):
        x = self.block1(x, x3)
        x = self.block2(x, x2)
        x = self.block3(x, x1)
        x = self.block4(x, x0)

        x = self.upconv(x)
        x = self.conv(x)

        if self.return_logits:
            return x

        return F.sigmoid(x)


class Model(nn.Module):
    def __init__(self, alpha=1.0, return_logits=True):
        super().__init__()
        self.alpha = alpha
        self.return_logits = return_logits

        self.encoder = Encoder(alpha=alpha)
        self.decoder = Decoder(
            alpha=alpha,
            return_logits=return_logits
        )

    def forward(self, x):
        x0, x1, x2, x3, x = self.encoder(x)
        x = self.decoder(x0, x1, x2, x3, x)

        return x
