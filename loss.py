import torch
from torch import nn
import torch.nn.functional as F

def histogram_loss(enhanced_y, original_y, num_bins=256, smooth=1e-6):
    batch_size, channels, height, width = enhanced_y.size()

    bins = torch.linspace(0, 1, steps=num_bins + 1).to(enhanced_y.device)
    enhanced_y_flat = enhanced_y.view(batch_size, -1)
    original_y_flat = original_y.view(batch_size, -1)

    enhanced_hist = torch.bucketize(enhanced_y_flat, bins) - 1
    original_hist = torch.bucketize(original_y_flat, bins) - 1

    hist_enhanced = torch.zeros(batch_size, num_bins, device=enhanced_y.device)
    hist_original = torch.zeros(batch_size, num_bins, device=original_y.device)

    for i in range(num_bins):
        hist_enhanced[:, i] = (enhanced_hist == i).float().sum(dim=1)
        hist_original[:, i] = (original_hist == i).float().sum(dim=1)

    # 归一化并加上平滑项
    hist_enhanced = (hist_enhanced + smooth) / (hist_enhanced.sum(dim=1, keepdim=True) + smooth)
    hist_original = (hist_original + smooth) / (hist_original.sum(dim=1, keepdim=True) + smooth)

    # 计算L2损失并归一化
    loss = F.mse_loss(hist_enhanced, hist_original) / num_bins

    return loss


# 对比损失函数
def contrast_loss(enhanced_y, original_y, epsilon=1e-6):
    def compute_contrast(image):
        if image.dim() == 3:
            image = image.unsqueeze(1)
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        laplacian = laplacian.to(image.device)
        contrast = torch.abs(F.conv2d(image, laplacian, padding=1))
        return contrast.mean()

    contrast_enhanced = compute_contrast(enhanced_y)
    contrast_original = compute_contrast(original_y)

    # 计算对比度差异，添加正则化项，并归一化
    loss = torch.abs(contrast_enhanced - contrast_original) / (contrast_original + epsilon)

    return loss


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

        # 通过加权平均组合损失
        loss = l1_loss + 0.1 * hist_loss + 0.1 * cont_loss + 0.01 * smooth_loss

        return loss



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




# # Example usage
# enhanced_y = torch.randn(1, 1, 256, 256)  # Example tensor
# original_y = torch.randn(1, 1, 256, 256)  # Example tensor
#
# criterion = YChannelEnhancementLoss()
# loss = criterion(enhanced_y, original_y)
# print(loss)
