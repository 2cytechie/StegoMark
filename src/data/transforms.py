import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
from typing import Tuple


class RandomAttack:
    """随机攻击模拟"""
    
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob:
            return img
        
        attack_type = random.choice(['blur', 'noise', 'jpeg', 'crop', 'rotate', 'color'])
        
        if attack_type == 'blur':
            # 高斯模糊
            kernel_size = random.choice([3, 5, 7])
            img = TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size])
        
        elif attack_type == 'noise':
            # 高斯噪声
            noise_std = random.uniform(0.01, 0.05)
            noise = torch.randn_like(img) * noise_std
            img = torch.clamp(img + noise, 0, 1)
        
        elif attack_type == 'jpeg':
            # JPEG压缩模拟（使用质量降低）
            quality = random.randint(50, 90)
            img = img * quality / 100.0
            img = torch.clamp(img, 0, 1)
        
        elif attack_type == 'crop':
            # 随机裁剪并resize回原始大小
            scale = random.uniform(0.7, 0.95)
            _, h, w = img.shape
            new_h, new_w = int(h * scale), int(w * scale)
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            img = TF.resized_crop(img, top, left, new_h, new_w, [h, w])
        
        elif attack_type == 'rotate':
            # 随机旋转
            angle = random.uniform(-15, 15)
            img = TF.rotate(img, angle)
        
        elif attack_type == 'color':
            # 颜色调整
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
        
        return torch.clamp(img, 0, 1)


class ResizeAndTile:
    """Resize到指定尺寸并保持宽高比，循环平铺填充"""
    
    def __init__(self, size: int = 64):
        self.size = size
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # img: (C, H, W)
        c, h, w = img.shape
        
        # 计算resize后的尺寸（保持宽高比）
        if h > w:
            new_h = self.size
            new_w = int(w * self.size / h)
        else:
            new_w = self.size
            new_h = int(h * self.size / w)
        
        # Resize
        img = TF.resize(img, [new_h, new_w], antialias=True)
        
        # 创建目标尺寸的张量
        result = torch.zeros((c, self.size, self.size), dtype=img.dtype)
        
        # 循环平铺填充
        for i in range(0, self.size, new_h):
            for j in range(0, self.size, new_w):
                end_i = min(i + new_h, self.size)
                end_j = min(j + new_w, self.size)
                copy_h = end_i - i
                copy_w = end_j - j
                result[:, i:end_i, j:end_j] = img[:, :copy_h, :copy_w]
        
        return result


def get_train_transforms(image_size: int = 64) -> transforms.Compose:
    """获取训练时的数据变换"""
    return transforms.Compose([
        transforms.ToTensor(),
        ResizeAndTile(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])


def get_val_transforms(image_size: int = 64) -> transforms.Compose:
    """获取验证时的数据变换"""
    return transforms.Compose([
        transforms.ToTensor(),
        ResizeAndTile(image_size),
    ])