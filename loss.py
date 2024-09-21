import torch
from torch import nn
import torch.nn.functional as F

def histogram_loss(enhanced_y, original_y, num_bins=256):
    hist_enhanced = torch.histc(enhanced_y, bins=num_bins, min=0, max=1)
    hist_original = torch.histc(original_y, bins=num_bins, min=0, max=1)
    hist_enhanced = hist_enhanced.float() / hist_enhanced.sum()
    hist_original = hist_original.float() / hist_original.sum()
    return torch.sum((hist_enhanced - hist_original) ** 2)

# 对比损失函数
def contrast_loss(enhanced_y, original_y):
    def compute_contrast(image):
        if image.dim() == 3:
            image = image.unsqueeze(1)  # add channel dimension if missing
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        laplacian = laplacian.to(image.device)
        contrast = torch.abs(F.conv2d(image, laplacian, padding=1))
        return contrast.mean()

    contrast_enhanced = compute_contrast(enhanced_y)
    contrast_original = compute_contrast(original_y)
    return torch.abs(contrast_enhanced - contrast_original)

# 平滑损失函数
def smoothness_loss(image):
    def compute_gradient(image):
        dx = image[:, :, 1:, :] - image[:, :, :-1, :]
        dy = image[:, :, :, 1:] - image[:, :, :, :-1]
        return dx, dy

    dx, dy = compute_gradient(image)
    return dx.abs().mean() + dy.abs().mean()

# 定义 Y 通道增强损失函数
class YChannelEnhancementLoss(nn.Module):
    def __init__(self):
        super(YChannelEnhancementLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, enhanced_y, original_y):
        l1_loss = self.l1_loss(enhanced_y, original_y)
        hist_loss = histogram_loss(enhanced_y, original_y)
        cont_loss = contrast_loss(enhanced_y, original_y)
        smooth_loss = smoothness_loss(enhanced_y)
        return l1_loss + hist_loss + cont_loss + smooth_loss


def gradient(input):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """

    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).cuda()
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).cuda()

    g1 = filter1(input)
    g2 = filter2(input)
    image_gradient = torch.abs(g1) + torch.abs(g2)
    return image_gradient




# Example usage
enhanced_y = torch.randn(1, 1, 256, 256)  # Example tensor
original_y = torch.randn(1, 1, 256, 256)  # Example tensor

criterion = YChannelEnhancementLoss()
loss = criterion(enhanced_y, original_y)
print(loss)
