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


# 边缘检测模块
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

        edges_x = F.conv2d(image, sobel_x, padding=1)
        edges_y = F.conv2d(image, sobel_y, padding=1)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        edges = edges / (torch.max(edges) + 1e-6)  # 归一化到 [0, 1]
        return edges

    def forward(self, x):
        # 原始图像分支
        x1 = self.conv(x)
        # 边缘信息分支
        edge = self.sobel_edge_detection(x)
        edge = self.edge_conv(edge)
        # 组合分支
        x2 = self.conv(x * edge)
        # 改进融合策略
        x = x1 + x2 + self.relu(edge)  # 添加边缘信息的直接贡献
        x = self.relu(x)
        return x

# class EdgeGuidedBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(EdgeGuidedBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.edge_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     @staticmethod
#     def sobel_edge_detection(image):
#         # 定义 Sobel 滤波器
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
#                                device=image.device).unsqueeze(0).unsqueeze(0)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
#                                device=image.device).unsqueeze(0).unsqueeze(0)
#
#         # 对每个通道分别进行边缘检测
#         edges = []
#         for c in range(image.size(1)):
#             edge_x = F.conv2d(image[:, c:c + 1, :, :], sobel_x, padding=1)
#             edge_y = F.conv2d(image[:, c:c + 1, :, :], sobel_y, padding=1)
#             edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)
#             edges.append(edge)
#
#         # 将所有通道的边缘检测结果堆叠在一起
#         edges = torch.cat(edges, dim=1)
#         edges = edges / torch.max(edges)  # 归一化到 [0, 1] 范围
#         return edges
#
#     def forward(self, x):
#         # 原始图像分支
#         x1 = self.conv(x)
#         # 边缘信息分支
#         edge = self.sobel_edge_detection(x)
#         edge = self.edge_conv(edge)
#         # 组合分支
#         x2 = self.conv(x * edge)
#         x = x1 + x2
#         x = self.relu(x)
#         return x


class EdgeUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeUpSample, self).__init__()
        self.edge_guided_block = EdgeGuidedBlock(in_channels, out_channels)
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

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

