import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class AttackSimulator(nn.Module):
    """攻击模拟器 - 模拟各种图像攻击"""
    
    def __init__(
        self,
        prob: float = 0.5,
        jpeg_quality_range: tuple = (50, 90),
        noise_std_range: tuple = (0.01, 0.05),
        blur_kernel_range: tuple = (3, 7),
        crop_scale_range: tuple = (0.7, 0.95),
        rotate_angle_range: tuple = (-15, 15),
        brightness_range: tuple = (0.8, 1.2),
        contrast_range: tuple = (0.8, 1.2)
    ):
        super(AttackSimulator, self).__init__()
        self.prob = prob
        self.jpeg_quality_range = jpeg_quality_range
        self.noise_std_range = noise_std_range
        self.blur_kernel_range = blur_kernel_range
        self.crop_scale_range = crop_scale_range
        self.rotate_angle_range = rotate_angle_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def forward(self, x):
        """
        输入: 图像 (B, C, H, W)
        输出: 攻击后的图像 (B, C, H, W)
        注意: 训练时随机应用攻击，推理时直接返回原图
        """
        if not self.training or random.random() > self.prob:
            return x
        
        # 随机选择攻击类型
        attack_types = ['blur', 'noise', 'jpeg', 'crop', 'rotate', 'color', 'dropout']
        attack_type = random.choice(attack_types)
        
        if attack_type == 'blur':
            return self.gaussian_blur(x)
        elif attack_type == 'noise':
            return self.gaussian_noise(x)
        elif attack_type == 'jpeg':
            return self.jpeg_compression(x)
        elif attack_type == 'crop':
            return self.random_crop_resize(x)
        elif attack_type == 'rotate':
            return self.random_rotate(x)
        elif attack_type == 'color':
            return self.color_jitter(x)
        elif attack_type == 'dropout':
            return self.dropout(x)
        
        return x
    
    def gaussian_blur(self, x):
        """高斯模糊"""
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.0)
        return TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    
    def gaussian_noise(self, x):
        """高斯噪声"""
        noise_std = random.uniform(*self.noise_std_range)
        noise = torch.randn_like(x) * noise_std
        return torch.clamp(x + noise, 0, 1)
    
    def jpeg_compression(self, x):
        """JPEG压缩模拟"""
        # 模拟JPEG压缩：降低质量并添加量化噪声
        quality = random.randint(*self.jpeg_quality_range)
        # 简单的质量降低模拟
        quantized = torch.round(x * quality) / quality
        noise = (x - quantized) * 0.5
        return torch.clamp(x - noise, 0, 1)
    
    def random_crop_resize(self, x):
        """随机裁剪并resize回原尺寸"""
        b, c, h, w = x.shape
        scale = random.uniform(*self.crop_scale_range)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 随机裁剪位置
        top = random.randint(0, h - new_h) if h > new_h else 0
        left = random.randint(0, w - new_w) if w > new_w else 0
        
        # 裁剪并resize
        cropped = x[:, :, top:top+new_h, left:left+new_w]
        return F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
    
    def random_rotate(self, x):
        """随机旋转"""
        angle = random.uniform(*self.rotate_angle_range)
        return self._rotate(x, angle)
    
    def _rotate(self, x, angle):
        """旋转图像"""
        # 使用affine_grid进行旋转
        theta = torch.tensor([
            [torch.cos(torch.tensor(angle * 3.14159 / 180)), -torch.sin(torch.tensor(angle * 3.14159 / 180)), 0],
            [torch.sin(torch.tensor(angle * 3.14159 / 180)), torch.cos(torch.tensor(angle * 3.14159 / 180)), 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='zeros')
    
    def color_jitter(self, x):
        """颜色抖动"""
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        
        # 亮度调整
        x = torch.clamp(x * brightness, 0, 1)
        # 对比度调整
        mean = x.mean(dim=[2, 3], keepdim=True)
        x = torch.clamp((x - mean) * contrast + mean, 0, 1)
        
        return x
    
    def dropout(self, x):
        """随机丢弃像素（模拟遮挡）"""
        b, c, h, w = x.shape
        mask_size = random.randint(h // 8, h // 4)
        
        # 创建mask
        mask = torch.ones_like(x)
        top = random.randint(0, h - mask_size)
        left = random.randint(0, w - mask_size)
        mask[:, :, top:top+mask_size, left:left+mask_size] = 0
        
        return x * mask
    
    def combined_attack(self, x, num_attacks=2):
        """组合攻击"""
        attacks = [
            self.gaussian_blur,
            self.gaussian_noise,
            self.jpeg_compression,
            self.random_crop_resize,
            self.random_rotate,
            self.color_jitter,
            self.dropout
        ]
        
        selected_attacks = random.sample(attacks, min(num_attacks, len(attacks)))
        for attack in selected_attacks:
            x = attack(x)
        
        return torch.clamp(x, 0, 1)
