"""
攻击模拟模块
在训练过程中模拟各种攻击，增强模型的鲁棒性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math


class DifferentiableAttack(nn.Module):
    """可微攻击基类"""
    
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob
    
    def forward(self, x):
        if random.random() < self.prob:
            return self.attack(x)
        return x
    
    def attack(self, x):
        raise NotImplementedError


class RandomCropAttack(DifferentiableAttack):
    """随机裁剪攻击 (可微)"""
    
    def __init__(self, prob=0.5, scale_range=(0.7, 1.0)):
        super().__init__(prob)
        self.scale_range = scale_range
    
    def attack(self, x):
        B, C, H, W = x.shape
        scale = random.uniform(*self.scale_range)
        new_h, new_w = int(H * scale), int(W * scale)
        
        # 随机裁剪位置
        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)
        
        # 裁剪
        cropped = x[:, :, top:top+new_h, left:left+new_w]
        
        # 插值回原始尺寸
        return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


class RandomRotateAttack(DifferentiableAttack):
    """随机旋转攻击 (可微)"""
    
    def __init__(self, prob=0.5, angle_range=(-15, 15)):
        super().__init__(prob)
        self.angle_range = angle_range
    
    def attack(self, x):
        angle = random.uniform(*self.angle_range)
        
        # 使用affine_grid实现可微旋转
        B, C, H, W = x.shape
        angle_rad = angle * math.pi / 180
        
        # 旋转矩阵
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        theta = torch.tensor([
            [cos_a, sin_a, 0],
            [-sin_a, cos_a, 0]
        ], dtype=torch.float, device=x.device).unsqueeze(0).repeat(B, 1, 1)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class GaussianBlurAttack(DifferentiableAttack):
    """高斯模糊攻击 (可微)"""
    
    def __init__(self, prob=0.5, kernel_range=(3, 7), sigma_range=(0.1, 2.0)):
        super().__init__(prob)
        self.kernel_range = kernel_range
        self.sigma_range = sigma_range
    
    def attack(self, x):
        kernel_size = random.choice(range(self.kernel_range[0], self.kernel_range[1]+1, 2))
        sigma = random.uniform(*self.sigma_range)
        
        return TF.gaussian_blur(x, kernel_size, [sigma, sigma])


class NoiseAttack(DifferentiableAttack):
    """噪声攻击 (可微)"""
    
    def __init__(self, prob=0.5, std_range=(0.01, 0.05)):
        super().__init__(prob)
        self.std_range = std_range
    
    def attack(self, x):
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(x) * std
        return torch.clamp(x + noise, -1, 1)


class JPEGCompressionAttack(DifferentiableAttack):
    """JPEG压缩攻击 (近似可微)"""
    
    def __init__(self, prob=0.5, quality_range=(50, 90)):
        super().__init__(prob)
        self.quality_range = quality_range
    
    def attack(self, x):
        # 注意：真正的JPEG压缩不可微，这里使用近似方法
        # 实际训练时可以使用jpeg2k等可微近似
        
        # 模拟JPEG压缩效果：高频衰减
        quality = random.randint(*self.quality_range)
        
        # 使用DCT近似
        from ..utils.dwt import DWT2D
        
        dwt = DWT2D('haar')
        LL, LH, HL, HH = dwt.dwt2(x)
        
        # 衰减高频分量
        attenuation = quality / 100.0
        LH = LH * attenuation
        HL = HL * attenuation
        HH = HH * (attenuation ** 2)  # 高频衰减更多
        
        return dwt.idwt2(LL, LH, HL, HH)


class CombinedAttack(nn.Module):
    """组合攻击"""
    
    def __init__(self, crop_prob=0.5, rotate_prob=0.3, blur_prob=0.3,
                 jpeg_prob=0.3, noise_prob=0.3):
        super().__init__()
        
        self.attacks = nn.ModuleList([
            RandomCropAttack(crop_prob),
            RandomRotateAttack(rotate_prob),
            GaussianBlurAttack(blur_prob),
            JPEGCompressionAttack(jpeg_prob),
            NoiseAttack(noise_prob)
        ])
    
    def forward(self, x):
        """随机应用攻击"""
        for attack in self.attacks:
            x = attack(x)
        return x
    
    def forward_with_probs(self, x, probs=None):
        """
        按指定概率应用攻击
        
        Args:
            x: 输入图像
            probs: 各攻击的概率列表，None则使用默认概率
        """
        if probs is None:
            return self.forward(x)
        
        for attack, prob in zip(self.attacks, probs):
            if random.random() < prob:
                x = attack.attack(x)
        
        return x


# 导入torchvision.transforms.functional用于模糊攻击
import torchvision.transforms.functional as TF


def simulate_attacks_during_training(watermarked_image, config):
    """
    训练时模拟攻击
    
    Args:
        watermarked_image: 含水印图片 [B, 3, H, W]
        config: 攻击配置
        
    Returns:
        attacked_image: 攻击后的图片
    """
    attack = CombinedAttack(
        crop_prob=config.random_crop_prob,
        rotate_prob=config.random_rotate_prob,
        blur_prob=config.gaussian_blur_prob,
        jpeg_prob=config.jpeg_compress_prob,
        noise_prob=config.noise_prob
    )
    
    return attack(watermarked_image)
