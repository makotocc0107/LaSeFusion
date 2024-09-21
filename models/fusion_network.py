# 开发时间：2024/6/24 18:11
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np


class EdgeGuidedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeGuidedBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def sobel_edge_detection(image):
        # 定义 Sobel 滤波器
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                               device=image.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
                               device=image.device).unsqueeze(0).unsqueeze(0)

        # 对每个通道分别进行边缘检测
        edges = []
        for c in range(image.size(1)):
            edge_x = F.conv2d(image[:, c:c + 1, :, :], sobel_x, padding=1)
            edge_y = F.conv2d(image[:, c:c + 1, :, :], sobel_y, padding=1)
            edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
            edges.append(edge)

        # 将所有通道的边缘检测结果堆叠在一起
        edges = torch.cat(edges, dim=1)
        edges = edges / torch.max(edges)  # 归一化到 [0, 1] 范围
        return edges

    def forward(self, x):
        # 原始图像分支
        x1 = self.conv(x)
        # 边缘信息分支
        edge = self.sobel_edge_detection(x)
        edge = self.edge_conv(edge)
        # 组合分支
        x2 = self.conv(x * edge)
        x = x1 + x2
        x = self.relu(x)
        return x


# class EdgeGuidedCNN(nn.Module):
#     def __init__(self):
#         super(EdgeGuidedCNN, self).__init__()
#         self.edge_guided_block1 = EdgeGuidedBlock(3, 64)
#         self.edge_guided_block2 = EdgeGuidedBlock(64, 128)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x, edge):
#         x1 = self.edge_guided_block1(x, edge)
#         x1 = self.pool(x1)
#         edge = self.pool(edge)
#
#         x2 = self.edge_guided_block2(x1, edge)
#         x2 = self.pool(x2)
#         edge = self.pool(edge)
#
#         # 突出边缘信息的特征图
#         x = x1 + x2
#
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.conv_down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv_up = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        )

    def forward(self, x):
        x = self.conv_up(x)
        return x

class AttentionUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUpSample, self).__init__()
        self.conv_up = nn.Sequential(
            # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        )
        self.attetion = SimpleAttention()

    def forward(self, x):
        x = self.conv_up(x)
        x = self.attetion(x)
        return x

class ConvBnReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLu, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=pad, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        x = self.conv_bn(x)
        x = F.leaky_relu(x)
        return x


class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                  padding=0)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.bottleneck(x)
        x = x + shortcut
        return x


class SimpleAttention(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimpleAttention, self).__init__()
        self.act = nn.Sigmoid()  # 使用Sigmoid激活函数
        self.e_lambda = e_lambda  # 定义平滑项e_lambda，防止分母为0

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入x的尺寸
        n = w * h - 1  # 计算特征图的元素数量减一，用于下面的归一化
        # 计算输入特征x与其均值之差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 计算注意力权重y，这里实现了SimAM的核心计算公式
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # 返回经过注意力加权的输入特征
        return x * self.act(y)


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.block1 = ConvBnReLu(in_channels=1, out_channels=16)
#         self.block2 = Resblock(in_channels=16, out_channels=32)
#         self.block3 = Resblock(in_channels=32, out_channels=64)
#         self.block4 = Resblock(in_channels=64, out_channels=128)
#         self.attention = SimpleAttention()
#
#     def forward(self, vis_y_image, inf_image):
#         vis_y_out, inf_out = F.leaky_relu(self.block1(vis_y_image)), F.leaky_relu(self.block1(inf_image))
#         vis_y_out, inf_out = F.leaky_relu(self.block2(vis_y_out)), F.leaky_relu(self.block2(inf_out))
#         vis_y_out, inf_out = F.leaky_relu(self.block3(vis_y_out)), F.leaky_relu(self.block3(inf_out))
#         vis_y_out, inf_out = F.leaky_relu(self.block4(vis_y_out)), F.leaky_relu(self.block4(inf_out))
#         vis_y_out, inf_out = self.attention(vis_y_out), self.attention(inf_out)
#         return vis_y_out, inf_out


class Y_Channel_Ehancement(nn.Module):
    def __init__(self):
        super(Y_Channel_Ehancement, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 使用 Sigmoid 函数将输出限制在 0 到 1 之间
        )

    def forward(self, x):
        # Encoder
        x = self.encoder(x)
        # Decoder
        x = self.decoder(x)
        return x


class EdgeFusion(nn.Module):
    def __init__(self):
        super(EdgeFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.up1 = UpSample(in_channels=1, out_channels=16)
        self.up2 = UpSample(in_channels=16, out_channels=32)
        self.up3 = UpSample(in_channels=32, out_channels=64)

        self.edge1 = EdgeGuidedBlock(in_channels=16, out_channels=16)
        self.edge2 = EdgeGuidedBlock(in_channels=32, out_channels=32)
        self.edge3 = EdgeGuidedBlock(in_channels=64, out_channels=64)

        self.down3 = DownSample(in_channels=128, out_channels=32)
        self.down2 = DownSample(in_channels=96, out_channels=16)
        self.down1 = DownSample(in_channels=48, out_channels=8)

        self.a_up1 = UpSample(in_channels=1, out_channels=16)
        self.a_up2 = UpSample(in_channels=16, out_channels=32)
        self.a_up3 = UpSample(in_channels=32, out_channels=64)

        self.re_down3 = DownSample(in_channels=192, out_channels=32)
        self.re_down2 = DownSample(in_channels=128, out_channels=16)
        self.re_down1 = DownSample(in_channels=64, out_channels=8)


        self.Y_enhancement = Y_Channel_Ehancement()

    def forward(self, vis_y_image, inf_image):
        vis_y_image_e = self.Y_enhancement(vis_y_image)
        y_re_fuse1 = self.a_up1(vis_y_image_e)
        y_re_fuse2 = self.a_up2(y_re_fuse1)
        y_re_fuse3 = self.a_up3(y_re_fuse2)

        vis_y_image = self.conv1(vis_y_image)
        vis_up1 = self.up1(vis_y_image)
        vis_up1_edge = self.edge1(vis_up1)
        vis_up2 = self.up2(vis_up1_edge)
        vis_up2_edge = self.edge2(vis_up2)
        vis_up3 = self.up3(vis_up2_edge)
        vis_up3_edge = self.edge3(vis_up3)

        inf_image = self.conv1(inf_image)
        inf_up1 = self.up1(inf_image)
        inf_up1_edge = self.edge1(inf_up1)
        inf_up2 = self.up2(inf_up1_edge)
        inf_up2_edge = self.edge2(inf_up2)
        inf_up3 = self.up3(inf_up2_edge)
        inf_up3_edge = self.edge3(inf_up3)

        fused_sample3 = torch.cat([vis_up3_edge, inf_up3_edge], dim=1)
        fused_sample3_d = self.down3(fused_sample3)  # channel=32
        fused_sample2 = torch.cat([fused_sample3_d, torch.cat([vis_up2_edge, inf_up2_edge], dim=1)], dim=1)
        fused_sample2_d = self.down2(fused_sample2)  # channel=16
        fused_sample1 = torch.cat([fused_sample2_d, torch.cat([vis_up1_edge, inf_up1_edge], dim=1)], dim=1)
        fused_sample1_d = self.down1(fused_sample1)  # channel=8
        fused_image_edge = self.conv2(fused_sample1_d)

        final_fuse3 = torch.cat([vis_up3_edge, inf_up3_edge, y_re_fuse3], dim=1)
        final_fuse3_d = self.re_down3(final_fuse3)
        final_fuse2 = torch.cat([final_fuse3_d, vis_up2_edge, inf_up2_edge, y_re_fuse2], dim=1)
        final_fuse2_d = self.re_down2(final_fuse2)
        final_fuse1 = torch.cat([final_fuse2_d, vis_up1_edge, inf_up1_edge, y_re_fuse1], dim=1)
        final_fuse1_d = self.re_down1(final_fuse1)
        final_fused_image = self.conv2(final_fuse1_d)

        return fused_image_edge, vis_y_image_e, final_fused_image



# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.block1 = Resblock(in_channels=256, out_channels=128)
#         self.block2 = Resblock(in_channels=128, out_channels=64)
#         self.block3 = Resblock(in_channels=64, out_channels=32)
#         self.block4 = Resblock(in_channels=32, out_channels=16)
#         self.block5 = Resblock(in_channels=16, out_channels=1)
#
#     def forward(self, x):
#         activate = nn.LeakyReLU()
#         x = activate(self.block1(x))
#         x = activate(self.block2(x))
#         x = activate(self.block3(x))
#         x = activate(self.block4(x))
#         x = F.tanh(self.block5(x))
#         return x


# def Fusion(vi_out, ir_out):
#     return torch.cat([vi_out, ir_out], dim=1)


# class LaSeFusion(nn.Module):
#     def __init__(self):
#         super(LaSeFusion, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#
#     def forward(self, vis_y_image, inf_image):
#         vis_encoder_out, inf_encoder_out = self.encoder(vis_y_image, inf_image)
#         encoder_in = Fusion(vis_encoder_out, inf_encoder_out)
#         fused_image = self.decoder(encoder_in)
#         return fused_image


def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out


if __name__ == '__main__':
    mode = EdgeFusion()
    input_vis = torch.rand(1, 1, 32, 32)
    input_inf = torch.rand(1, 1, 32, 32)
    output = mode(input_vis, input_inf)
    # print(output.shape)
    print(output)
