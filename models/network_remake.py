# 开发时间：2024/10/25 14:32
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np


# 注意力机制，用于首次两个信息融合交互
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


# 普通上采样模块
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.up_sample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.up_sample(x)
        return x


# Y通道光照增强
class Y_Channel_Enhancement(nn.Module):
    def __init__(self):
        super(Y_Channel_Enhancement, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder
        self.dec_upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.dec_upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1)  # 输出通道数为32

        # Activation functions
        self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.relu(self.enc_conv1(x))
        x2 = self.relu(self.enc_conv2(x1))
        x3 = self.enc_pool1(x2)

        x4 = self.relu(self.enc_conv3(x3))
        x5 = self.relu(self.enc_conv4(x4))
        x6 = self.enc_pool2(x5)

        # Decoder with skip connections
        x7 = self.relu(self.dec_upconv1(x6))
        x7 = torch.cat((x7, x4), dim=1)

        x8 = self.relu(self.dec_conv1(x7))
        x9 = self.relu(self.dec_upconv2(x8))
        x9 = torch.cat((x9, x2), dim=1)

        x10 = self.relu(self.dec_conv2(x9))

        output = self.sigmoid(self.final_conv(x10))  # 仍然应用sigmoid，输出通道数为32

        return output


class EdgeGuidedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeGuidedBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)

    @staticmethod
    def sobel_edge_detection(image):
        # 定义 Sobel 滤波器
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                               device=image.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
                               device=image.device).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]

        # 扩展 Sobel 滤波器为 [32, 1, 3, 3]，使其适应每个通道
        sobel_x = sobel_x.repeat(image.size(1), 1, 1, 1)  # 重复 Sobel 滤波器的第一个维度
        sobel_y = sobel_y.repeat(image.size(1), 1, 1, 1)

        # 对每个通道进行 Sobel 卷积
        edges_x = F.conv2d(image, sobel_x, padding=1, groups=image.size(1))  # groups=image.size(1) 处理多通道
        edges_y = F.conv2d(image, sobel_y, padding=1, groups=image.size(1))
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        edges = edges / (torch.max(edges) + 1e-6)  # 归一化到 [0, 1]
        return edges

    def forward(self, x):
        # 原始图像分支
        x1 = self.conv(x)

        # 边缘信息分支
        edge = self.sobel_edge_detection(x)
        edge = self.edge_conv(edge)  # 处理边缘信息

        # 组合分支，使用边缘引导的信息
        x2 = self.conv(x * edge)  # 将图像和边缘信息相乘进行处理

        # 改进融合策略，加入边缘信息的贡献
        x = x1 + x2 + self.relu(edge)  # 添加边缘信息的直接贡献
        x = self.relu(x)  # 激活融合后的结果
        return x


class EdgeUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeUpSample, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.edge_guided_block = EdgeGuidedBlock(out_channels, out_channels)

    def forward(self, x):
        # 先进行上采样
        x = self.upsample(x)
        # 再进行边缘引导
        x = self.edge_guided_block(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 通过卷积层提取特征
        x = self.conv(x)
        # 最大池化进行下采样
        x = self.pool(x)
        return x


class LaSeFusion(nn.Module):
    def __init__(self):
        super(LaSeFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSample(in_channels=8, out_channels=16)
        self.up2 = UpSample(in_channels=16, out_channels=32)
        self.up3 = UpSample(in_channels=32, out_channels=64)

        self.simple_attention = SimpleAttention()

        self.y_channel_enhance = Y_Channel_Enhancement()
        self.edge_up1 = EdgeUpSample(in_channels=1, out_channels=32)
        self.edge_up2 = EdgeUpSample(in_channels=64, out_channels=64)
        self.edge_up3 = EdgeUpSample(in_channels=128, out_channels=128)

        self.down1 = DownSample(in_channels=256, out_channels=128)
        self.down2 = DownSample(in_channels=128, out_channels=64)
        self.down3 = DownSample(in_channels=64, out_channels=32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, vis_y_image, inf_image):
        activate = nn.LeakyReLU()
        vis_y_image_enhanced = vis_y_image

        # 可见光通道特征提取
        vis_y_image = activate(self.conv1(vis_y_image))
        vis_y_image_up1 = self.up1(vis_y_image)
        vis_y_image_up2 = self.up2(vis_y_image_up1)
        vis_y_image_up3 = self.up3(vis_y_image_up2)

        # 红外光通道特征提取
        inf_image = activate(self.conv1(inf_image))
        inf_image_up1 = self.up1(inf_image)
        inf_image_up2 = self.up2(inf_image_up1)
        inf_image_up3 = self.up3(inf_image_up2)

        # 初步融合 添加注意力机制
        initial_fuse1 = torch.cat([vis_y_image_up1, inf_image_up1], dim=1)
        initial_fuse1 = self.simple_attention(initial_fuse1)
        initial_fuse2 = torch.cat([vis_y_image_up2, inf_image_up2], dim=1)
        initial_fuse2 = self.simple_attention(initial_fuse2)
        initial_fuse3 = torch.cat([vis_y_image_up3, inf_image_up3], dim=1)
        initial_fuse3 = self.simple_attention(initial_fuse3)

        # 可见光另一分支增强并融合
        vis_y_image_enhanced = self.y_channel_enhance(vis_y_image_enhanced)
        vis_y_image_enhanced_up1 = self.edge_up1(vis_y_image_enhanced)
        second_cat1 = torch.cat([vis_y_image_enhanced_up1, initial_fuse1], dim=1)
        vis_y_image_enhanced_up2 = self.edge_up2(second_cat1)
        second_cat2 = torch.cat([vis_y_image_enhanced_up2, initial_fuse2], dim=1)
        vis_y_image_enhanced_up3 = self.edge_up3(second_cat2)
        second_cat3 = torch.cat([vis_y_image_enhanced_up3, initial_fuse3], dim=1)

        fused_image = self.down1(second_cat3)
        fused_image = self.down2(fused_image)
        fused_image = self.down3(fused_image)
        fused_image = activate(self.conv2(fused_image))
        fused_image = nn.Tanh()(self.conv3(fused_image)) / 2 + 0.5

        return fused_image, vis_y_image_enhanced
