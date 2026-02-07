import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Optional, Callable, Tuple
import random


class WatermarkDataset(Dataset):
    """水印数据集类"""
    
    def __init__(
        self,
        image_dir: str,
        watermark_dir: str,
        transform: Optional[Callable] = None,
        watermark_transform: Optional[Callable] = None,
        image_size: int = 64,
        watermark_size: int = 64
    ):
        self.image_dir = image_dir
        self.watermark_dir = watermark_dir
        self.transform = transform
        self.watermark_transform = watermark_transform
        self.image_size = image_size
        self.watermark_size = watermark_size
        
        # 获取图像文件列表
        self.image_files = self._get_image_files(image_dir)
        self.watermark_files = self._get_image_files(watermark_dir)
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        if len(self.watermark_files) == 0:
            raise ValueError(f"No watermarks found in {watermark_dir}")
    
    def _get_image_files(self, directory: str) -> list:
        """获取目录中的所有图像文件"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        files = []
        for f in os.listdir(directory):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(directory, f))
        return sorted(files)
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            image: 载体图像 (C, H, W)
            watermark: 水印图像 (1, H, W) 或 (3, H, W)
        """
        # 加载载体图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 随机选择水印
        watermark_path = random.choice(self.watermark_files)
        watermark = Image.open(watermark_path)
        
        # 转换水印为RGB（3通道）
        if watermark.mode != 'RGB':
            watermark = watermark.convert('RGB')
        else:
            watermark = watermark.convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            image = self._default_transform(image)
        
        if self.watermark_transform:
            watermark = self.watermark_transform(watermark)
        else:
            watermark = self._default_transform(watermark)
        
        # 确保尺寸正确
        if image.shape[-2:] != (self.image_size, self.image_size):
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), 
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        if watermark.shape[-2:] != (self.watermark_size, self.watermark_size):
            watermark = torch.nn.functional.interpolate(
                watermark.unsqueeze(0),
                size=(self.watermark_size, self.watermark_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        return image, watermark
    
    def _default_transform(self, img: Image.Image) -> torch.Tensor:
        """默认变换：转为tensor并resize"""
        from torchvision import transforms
        from .transforms import ResizeAndTile
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            ResizeAndTile(self.image_size),
        ])
        return transform(img)