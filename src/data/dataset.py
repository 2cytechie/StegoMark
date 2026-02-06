"""
数据集模块
"""
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from .transforms import WatermarkTransforms, ImagePreprocessor


class WatermarkDataset(Dataset):
    """水印数据集"""
    
    def __init__(self, images_dir, watermarks_dir, watermark_size=64,
                 flip_prob=0.5, blur_prob=0.3, color_jitter_prob=0.3, 
                 crop_prob=0.5, train=True):
        """
        Args:
            images_dir: 目标图片目录
            watermarks_dir: 水印图片目录
            watermark_size: 水印尺寸
            flip_prob: 翻转概率
            blur_prob: 模糊概率
            color_jitter_prob: 颜色调整概率
            crop_prob: 裁剪概率
            train: 是否为训练模式
        """
        self.images_dir = Path(images_dir)
        self.watermarks_dir = Path(watermarks_dir)
        self.train = train
        self.watermark_size = watermark_size
        
        # 获取所有图片文件
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
            self.image_files.extend(list(self.images_dir.glob(ext.upper())))
        
        self.image_files = sorted(self.image_files)
        
        # 获取所有水印文件
        self.watermark_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.watermark_files.extend(list(self.watermarks_dir.glob(ext)))
            self.watermark_files.extend(list(self.watermarks_dir.glob(ext.upper())))
        
        self.watermark_files = sorted(self.watermark_files)
        
        # 初始化变换
        self.transforms = WatermarkTransforms(
            flip_prob=flip_prob,
            blur_prob=blur_prob,
            color_jitter_prob=color_jitter_prob,
            crop_prob=crop_prob,
            watermark_size=watermark_size
        )
        self.preprocessor = ImagePreprocessor(watermark_size=watermark_size)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            dict: {
                'image': 目标图片 [3, H, W],
                'watermark': 水印 [3, 64, 64],
                'watermark_tiled': 平铺后的水印 [3, H, W],
                'image_path': 图片路径
            }
        """
        # 加载目标图片
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 随机选择一个水印
        watermark_path = random.choice(self.watermark_files)
        watermark = Image.open(watermark_path).convert('RGB')
        
        # 应用数据增强
        if self.train:
            image = self.transforms(image, is_watermark=False)
        
        watermark = self.transforms(watermark, is_watermark=True)
        
        # 预处理
        image_tensor = self.preprocessor.preprocess_target(image)
        watermark_tensor = self.preprocessor.preprocess_watermark(watermark)
        
        # 获取目标图片尺寸
        _, H, W = image_tensor.shape
        
        # 平铺水印
        watermark_tiled = self.preprocessor.tile_watermark(
            watermark_tensor, (H, W)
        )
        
        return {
            'image': image_tensor,
            'watermark': watermark_tensor,
            'watermark_tiled': watermark_tiled,
            'image_path': str(image_path),
            'watermark_path': str(watermark_path)
        }


class WatermarkInferenceDataset(Dataset):
    """水印推理数据集 (用于提取水印)"""
    
    def __init__(self, images_dir):
        """
        Args:
            images_dir: 待检测图片目录
        """
        self.images_dir = Path(images_dir)
        
        # 获取所有图片文件
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
            self.image_files.extend(list(self.images_dir.glob(ext.upper())))
        
        self.image_files = sorted(self.image_files)
        print(f"找到 {len(self.image_files)} 张待检测图片")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            dict: {
                'image': 图片 [3, H, W],
                'image_path': 图片路径,
                'original_size': 原始尺寸 (W, H)
            }
        """
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        original_size = image.size  # (W, H)
        
        # 预处理
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        image_tensor = transform(image)
        
        return {
            'image': image_tensor,
            'image_path': str(image_path),
            'original_size': original_size
        }


def collate_fn(batch):
    """
    自定义collate函数，处理不同尺寸的图片
    """
    # 分离不同字段
    images = [item['image'] for item in batch]
    watermarks = torch.stack([item['watermark'] for item in batch])
    watermark_tiled = [item['watermark_tiled'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    watermark_paths = [item['watermark_path'] for item in batch]
    
    # 获取最大尺寸
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    # 填充到相同尺寸 (使用constant模式避免reflect模式的限制)
    padded_images = []
    padded_watermarks = []
    
    for img, wm_tiled in zip(images, watermark_tiled):
        _, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        
        # 填充图像 (使用constant模式，填充值为0)
        if pad_h > 0 or pad_w > 0:
            padded_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
            padded_wm = torch.nn.functional.pad(wm_tiled, (0, pad_w, 0, pad_h), mode='constant', value=0)
        else:
            padded_img = img
            padded_wm = wm_tiled
            
        padded_images.append(padded_img)
        padded_watermarks.append(padded_wm)
    
    return {
        'image': torch.stack(padded_images),
        'watermark': watermarks,
        'watermark_tiled': torch.stack(padded_watermarks),
        'image_path': image_paths,
        'watermark_path': watermark_paths,
        'original_sizes': [(img.shape[1], img.shape[2]) for img in images]
    }


# 导入T用于推理数据集
import torchvision.transforms as T
