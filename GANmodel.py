import torch
import torch.nn as nn
import numpy as np


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResConvBlock(nn.Module):
    def __init__(self, channels_out, channels_m, channels_in, Res_bool):
        super(ResConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels_in, channels_m, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels_m, channels_m, kernel_size=1, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels_m, channels_out,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.ResBool = Res_bool
        self.cnn_skip = nn.Conv2d(
            channels_in, channels_out, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.conv3(self.conv2(self.conv1(x)))

        if self.ResBool:
            if x.size(1) != residual.size(1):
                x = self.cnn_skip(x.clone())
            Fx = x + residual
        else:
            Fx = residual
        return Fx


class ResDenseBlock(nn.Module):
    def __init__(self, channels_in, channels_out, Res_bool, growth_channel=32):
        super(ResDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, growth_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(
            channels_in + growth_channel, growth_channel, 3, 1, 1)
        self.conv3 = nn.Conv2d(
            channels_in + 2 * growth_channel, growth_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(
            channels_in + 3 * growth_channel, growth_channel, 3, 1, 1)
        self.conv5 = nn.Conv2d(
            channels_in + 4 * growth_channel, channels_out, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2)
        self.ResBool = Res_bool
        self.cnn_skip = nn.Conv2d(
            channels_in, channels_out, kernel_size=1, stride=1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(x4)

        if self.ResBool:
            if x.size(1) != x5.size(1):
                x = self.cnn_skip(x.clone())
            Fx = x5*0.2 + x
        else:
            Fx = x5
        return Fx


class down(nn.Module):
    def __init__(self, in_ch, m_ch, out_ch, ConvBlock, kernel_size=4, padding=1, stride=2,):
        super(down, self).__init__()
        if ConvBlock == 'ResConvBlock':
            self.conv_down = nn.Sequential(
                ResConvBlock(channels_out=out_ch, channels_m=m_ch, channels_in=in_ch, Res_bool=True))
        elif ConvBlock == 'ResDenseBlock':
            self.conv_down = nn.Sequential(
                ResDenseBlock(channels_in=in_ch, channels_out=out_ch, Res_bool=True))
        self.d = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size,
                      padding=padding, stride=stride),
            nn.GroupNorm(32, out_ch),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x = self.conv_down(x)
        x_skip, x_down = x, self.d(x)
        return x_skip, x_down


class up(nn.Module):
    def __init__(self, in_ch, m_ch, out_ch, ConvBlock, kernel_size=4, padding=1, stride=2):
        super(up, self).__init__()
        self.u = nn.Sequential(
            nn.ConvTranspose2d(
                in_ch, in_ch, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=0),
            nn.GroupNorm(32, in_ch),
            nn.LeakyReLU(0.2)
        )
        if ConvBlock == 'ResConvBlock':
            self.conv_up = nn.Sequential(ResConvBlock(
                out_ch, m_ch, 2*in_ch, Res_bool=False))
        elif ConvBlock == 'ResDenseBlock':
            self.conv_up = nn.Sequential(ResDenseBlock(
                channels_in=2*in_ch, channels_out=out_ch, Res_bool=False))

    def forward(self, x, x_skip):
        x_up = self.u(x)
        Fx = torch.cat([x_up, x_skip], 1)
        Fx = self.conv_up(Fx)
        return Fx


class down_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(down_conv, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.GroupNorm(32, out_ch),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.cnn(x)


class Generator(nn.Module):
    def __init__(self, CB='ResConvBlock'):
        super(Generator, self).__init__()
        self.s1_down = down(in_ch=1, m_ch=32, out_ch=64, ConvBlock=CB)
        self.s2_down = down(in_ch=64, m_ch=96, out_ch=128, ConvBlock=CB)
        self.s3_down = down(in_ch=128, m_ch=192, out_ch=256, ConvBlock=CB)
        self.s4_down = down(in_ch=256, m_ch=384, out_ch=512, ConvBlock=CB)
        self.down = down_conv(in_ch=512, out_ch=512,
                              kernel_size=3, stride=1, padding=1)
        self.s4_up = up(in_ch=512, m_ch=640, out_ch=256, ConvBlock=CB)
        self.s3_up = up(in_ch=256, m_ch=320, out_ch=128, ConvBlock=CB)
        self.s2_up = up(in_ch=128, m_ch=160, out_ch=64, ConvBlock=CB)
        self.s1_up = up(in_ch=64, m_ch=80, out_ch=32, ConvBlock=CB)
        self.output = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        (_, _, H, W) = x.size()
        x_s1_skip, x_s1_down = self.s1_down(x.view(-1, 1, H, W))
        x_s2_skip, x_s2_down = self.s2_down(x_s1_down)
        x_s3_skip, x_s3_down = self.s3_down(x_s2_down)
        x_s4_skip, x_s4_down = self.s4_down(x_s3_down)
        x_bottom = self.down(x_s4_down)

        x_s4_up = self.s4_up(x_bottom, x_s4_skip)
        x_s3_up = self.s3_up(x_s4_up, x_s3_skip)
        x_s2_up = self.s2_up(x_s3_up, x_s2_skip)
        x_s1_up = self.s1_up(x_s2_up, x_s1_skip)
        out = self.output(x_s1_up)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),

            # 2D adaptive average pooling, output_size(batch_size,512,1,1)
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),  # dense layer
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)  # dense layer
        )

    def forward(self, x):
        if x.size(1) != 1:
            x = nn.Conv2d(x.size(1), 1, kernel_size=3, padding=1).cuda()(x)

        y = self.net(x)  # flatten to (batch_size, 1)
        return y.view(x.size(0), -1)  # Not the probability without sigmoid


@torch.no_grad()
def init_weights(module_list, scale=1):
    for m in module_list.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='leaky_relu')
            # Scale initialized weights, especially for residual blocks. Default: 1.
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # nn.Instance2d has no parameters
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
