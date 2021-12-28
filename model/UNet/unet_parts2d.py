""" Parts of the U-Net model """


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,se_block=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if se_block:
            self._se_block = Squeeze_Excite_Block(out_channels, out_channels)

    def forward(self, x):

        x = self.double_conv(x)
        if self._se_block:
            x = self._se_block(x)
        return x

class ResidualConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None,se_block=True):
        super(ResidualConv, self).__init__()

        self.se_block = se_block
        if not mid_channels:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if se_block:
            self._se_block = Squeeze_Excite_Block(out_channels, out_channels)

    def forward(self, x):

        x = self.conv_block(x) + self.conv_skip(x)
        if self.se_block:
            x = self._se_block(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,residual_block=True,se_block=True):
        super().__init__()
        if residual_block:
            self.conv = ResidualConv(in_channels,out_channels,se_block=se_block)
        else:
            self.conv = DoubleConv(in_channels, out_channels,se_block=se_block)

        self.pool = nn.MaxPool2d(2)




    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True,residual_block=True,attention=True):
        super().__init__()

        self.attention = attention
        if attention:
            self.attention_block = Attention_block(F_g=in_channels//2,F_l=in_channels//2,F_int=in_channels//4) #what to do in trilinear case
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            if residual_block:
                self.conv = ResidualConv(in_channels, out_channels, in_channels // 2,se_block=False)
            else:
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,se_block=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            if residual_block:
                self.conv = ResidualConv(in_channels, out_channels,se_block=False)
            else:
                self.conv = DoubleConv(in_channels, out_channels,se_block=False)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        x1 = self.equalize_dimensions(x1,x2)
        if self.attention:
            x2 = self.attention_block(x1,x2) #attention on skip connection signal

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

    def equalize_dimensions(self,x, upsampled_data):
        # print('x.shape = ', x.shape, ' != ', upsampled_data.shape, ' upsampled.shape')
        padding_dim1 = abs(x.shape[-1] - upsampled_data.shape[-1])
        padding_dim2 = abs(x.shape[-2] - upsampled_data.shape[-2])

        if padding_dim1 % 2 == 0:
            padding_left = padding_dim1 // 2
            padding_right = padding_dim1 // 2
        else:
            padding_left = padding_dim1 // 2
            padding_right = padding_dim1 - padding_left

        if padding_dim2 % 2 == 0:
            padding_top = padding_dim2 // 2
            padding_bottom = padding_dim2 // 2
        else:
            padding_top = padding_dim2 // 2
            padding_bottom = padding_dim2 - padding_top


        pad_fn = nn.ConstantPad2d((padding_left, padding_right,
                                padding_top, padding_bottom), 0)

        return pad_fn(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out