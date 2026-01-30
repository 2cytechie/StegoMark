# 感知损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """结构相似性损失"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        """生成高斯核"""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """创建滑动窗口"""
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2):
        """计算SSIM"""
        device = img1.device
        window = self.window.to(device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        """计算SSIM损失（1 - SSIM）"""
        return 1 - self.ssim(img1, img2)

class PerceptualLoss(nn.Module):
    """感知损失（使用VGG特征）"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        from torchvision.models import vgg16, VGG16_Weights
        
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def to(self, device):
        """移动模型到指定设备"""
        self.feature_extractor = self.feature_extractor.to(device)
        return self

    def forward(self, img1, img2):
        """计算感知损失"""
        features1 = self.feature_extractor(img1)
        features2 = self.feature_extractor(img2)
        return F.mse_loss(features1, features2)

class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, img_weight=1.0, ssim_weight=0.5, perceptual_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = PerceptualLoss()
        
        self.img_weight = img_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, img1, img2):
        """计算组合损失"""
        mse = self.mse_loss(img1, img2)
        ssim = self.ssim_loss(img1, img2)
        perceptual = self.perceptual_loss(img1, img2)
        
        total_loss = (self.img_weight * mse + 
                     self.ssim_weight * ssim + 
                     self.perceptual_weight * perceptual)
        
        return total_loss, mse, ssim, perceptual

def calculate_psnr(img1, img2):
    """计算PSNR"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """计算SSIM"""
    ssim_loss = SSIMLoss()
    return 1 - ssim_loss(img1, img2)