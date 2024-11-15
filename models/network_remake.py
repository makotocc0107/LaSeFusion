# 开发时间：2024/10/25 14:32
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
from timm.models.layers import CondConv2d


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


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_prob=0.5):
        super(MultiScaleConv, self).__init__()
        # 设置卷积层的卷积核大小（可定制）
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # 保证输入输出尺寸一致
        # 第一层卷积 + 批量归一化
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二层卷积 + 批量归一化
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Dropout层
        self.dropout = nn.Dropout(dropout_prob)
        # 残差连接层（1x1 卷积）
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 残差连接
        residual = self.conv_res(x)
        # 第一层卷积 + 批量归一化 + 激活
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # 第二层卷积 + 批量归一化 + 激活
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # Dropout
        x = self.dropout(x)
        # 残差连接加法
        x += residual
        return x


class DynamicConv(nn.Module):
    """Dynamic Conv layer"""

    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1, groups=1, bias=False, num_experts=4):
        super().__init__()
        print('+++', num_experts)
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation, groups, bias, num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
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


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.DynamicConv = DynamicConv(in_channels, out_channels)
        self.edge_guided_block = EdgeGuidedBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.DynamicConv(x)
        x = self.edge_guided_block(x)
        return x

class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        activate = nn.LeakyReLU()
        out = activate(self.conv(x))
        return out

class LaSeFusion(nn.Module):
    def __init__(self):
        super(LaSeFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.MultiConv1 = MultiScaleConv(in_channels=8, out_channels=16, kernel_size=3)
        self.MultiConv2 = MultiScaleConv(in_channels=16, out_channels=32, kernel_size=5)
        self.MultiConv3 = MultiScaleConv(in_channels=32, out_channels=64, kernel_size=7)

        self.simple_attention = SimpleAttention()

        self.y_channel_enhance = Y_Channel_Enhancement()
        self.edge_conv1 = EdgeConv(in_channels=1, out_channels=32)
        self.edge_conv2 = EdgeConv(in_channels=64, out_channels=64)
        self.edge_conv3 = EdgeConv(in_channels=128, out_channels=128)

        self.reconstruction = nn.Sequential(
            reflect_conv(in_channels=256, out_channels=128),
            reflect_conv(in_channels=128, out_channels=64),
            reflect_conv(in_channels=64, out_channels=32),
            reflect_conv(in_channels=32, out_channels=1)
        )

    def forward(self, vis_y_image, inf_image):
        activate = nn.LeakyReLU()
        vis_y_image_enhanced = vis_y_image

        # 可见光通道特征提取
        vis_y_image = activate(self.conv1(vis_y_image))
        vis_y_image_1 = self.MultiConv1(vis_y_image)
        vis_y_image_2 = self.MultiConv2(vis_y_image_1)
        vis_y_image_3 = self.MultiConv3(vis_y_image_2)

        # 红外光通道特征提取
        inf_image = activate(self.conv1(inf_image))
        inf_image_1 = self.MultiConv1(inf_image)
        inf_image_2 = self.MultiConv2(inf_image_1)
        inf_image_3 = self.MultiConv3(inf_image_2)

        # 初步融合 添加注意力机制
        initial_fuse1 = torch.cat([vis_y_image_1, inf_image_1], dim=1)
        initial_fuse1 = self.simple_attention(initial_fuse1)
        initial_fuse2 = torch.cat([vis_y_image_2, inf_image_2], dim=1)
        initial_fuse2 = self.simple_attention(initial_fuse2)
        initial_fuse3 = torch.cat([vis_y_image_3, inf_image_3], dim=1)
        initial_fuse3 = self.simple_attention(initial_fuse3)

        # 可见光另一分支增强并融合
        vis_y_image_enhanced = self.y_channel_enhance(vis_y_image_enhanced)
        vis_y_image_enhanced_1 = self.edge_conv1(vis_y_image_enhanced)
        second_cat1 = torch.cat([vis_y_image_enhanced_1, initial_fuse1], dim=1)
        vis_y_image_enhanced_2 = self.edge_conv2(second_cat1)
        second_cat2 = torch.cat([vis_y_image_enhanced_2, initial_fuse2], dim=1)
        vis_y_image_enhanced_3 = self.edge_conv3(second_cat2)
        second_cat3 = torch.cat([vis_y_image_enhanced_3, initial_fuse3], dim=1)

        fused_image = self.reconstruction(second_cat3)
        fused_image = nn.Tanh()(fused_image) / 2 + 0.5

        return fused_image, vis_y_image_enhanced
