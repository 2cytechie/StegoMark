"""
数据集加载器
用于加载训练、验证和测试数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import random
from typing import Tuple, Optional, List, Callable
import torchvision.transforms as transforms
from watermark_utils import WatermarkPreprocessor, pil_to_tensor


class WatermarkDataset(Dataset):
    """水印数据集"""
    
    def __init__(self, 
                 image_dir: str,
                 watermark_dir: str,
                 target_size: int = 256,
                 watermark_size: int = 64,
                 mode: str = 'train',
                 transform: Optional[Callable] = None):
        """
        初始化数据集
        
        Args:
            image_dir: 图像目录
            watermark_dir: 水印目录
            target_size: 目标图像尺寸
            watermark_size: 水印尺寸
            mode: 'train' 或 'val'
            transform: 额外的数据变换
        """
        self.image_dir = image_dir
        self.watermark_dir = watermark_dir
        self.target_size = target_size
        self.watermark_size = watermark_size
        self.mode = mode
        self.transform = transform
        
        # 获取图像文件列表
        self.image_files = self._get_image_files(image_dir)
        self.watermark_files = self._get_image_files(watermark_dir)
        
        if len(self.image_files) == 0:
            raise ValueError(f"在 {image_dir} 中没有找到图像文件")
        if len(self.watermark_files) == 0:
            raise ValueError(f"在 {watermark_dir} 中没有找到水印文件")
        
        # 水印预处理器
        self.watermark_preprocessor = WatermarkPreprocessor(watermark_size, target_size)
        
        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
        
        print(f"数据集初始化完成: {len(self.image_files)} 张图像, {len(self.watermark_files)} 个水印")
    
    def _get_image_files(self, directory: str) -> List[str]:
        """获取目录中的所有图像文件"""
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        files = []
        if os.path.exists(directory):
            for f in os.listdir(directory):
                if f.lower().endswith(valid_extensions):
                    files.append(os.path.join(directory, f))
        return sorted(files)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            (图像, 水印_256x256, 水印_64x64)
        """
        # 加载图像
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 应用图像变换
        image = self.image_transform(image)
        
        # 随机选择一个水印
        watermark_path = random.choice(self.watermark_files)
        watermark = Image.open(watermark_path).convert('RGB')
        
        # 预处理水印
        # 1. 填充为正方形
        watermark_squared = self.watermark_preprocessor.pad_to_square(watermark)
        
        # 2. resize到64x64
        watermark_resized = self.watermark_preprocessor.resize(watermark_squared, self.watermark_size)
        
        # 3. 转换为张量 [H, W, C] -> [C, H, W]
        watermark_64 = pil_to_tensor(watermark_resized)
        
        # 4. 复制4x4得到256x256
        watermark_256 = self.watermark_preprocessor.tile_watermark(watermark_64)
        
        # 应用额外的变换（训练时）
        if self.transform and self.mode == 'train':
            # 随机水平翻转
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                watermark_256 = transforms.functional.hflip(watermark_256)
                watermark_64 = transforms.functional.hflip(watermark_64)
            
            # 随机垂直翻转
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                watermark_256 = transforms.functional.vflip(watermark_256)
                watermark_64 = transforms.functional.vflip(watermark_64)
        
        return image, watermark_256, watermark_64


class PairWatermarkDataset(Dataset):
    """配对的水印数据集（图像和水印一一对应）"""
    
    def __init__(self,
                 image_dir: str,
                 watermark_dir: str,
                 target_size: int = 256,
                 watermark_size: int = 64,
                 mode: str = 'train'):
        """
        初始化数据集
        
        Args:
            image_dir: 图像目录
            watermark_dir: 水印目录
            target_size: 目标图像尺寸
            watermark_size: 水印尺寸
            mode: 'train' 或 'val'
        """
        self.image_dir = image_dir
        self.watermark_dir = watermark_dir
        self.target_size = target_size
        self.watermark_size = watermark_size
        self.mode = mode
        
        # 获取配对的文件列表
        self.pairs = self._get_pairs()
        
        # 水印预处理器
        self.watermark_preprocessor = WatermarkPreprocessor(watermark_size, target_size)
        
        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
        ])
        
        print(f"配对数据集初始化完成: {len(self.pairs)} 对图像-水印")
    
    def _get_pairs(self) -> List[Tuple[str, str]]:
        """获取配对的图像和水印文件"""
        pairs = []
        
        # 获取所有图像文件
        image_files = {}
        if os.path.exists(self.image_dir):
            for f in os.listdir(self.image_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(f)[0]
                    image_files[name] = os.path.join(self.image_dir, f)
        
        # 获取所有水印文件并配对
        if os.path.exists(self.watermark_dir):
            for f in os.listdir(self.watermark_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    name = os.path.splitext(f)[0]
                    if name in image_files:
                        pairs.append((image_files[name], 
                                    os.path.join(self.watermark_dir, f)))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取一个样本"""
        image_path, watermark_path = self.pairs[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        image = self.image_transform(image)
        
        # 加载水印
        watermark = Image.open(watermark_path).convert('RGB')
        
        # 预处理水印
        watermark_squared = self.watermark_preprocessor.pad_to_square(watermark)
        watermark_resized = self.watermark_preprocessor.resize(watermark_squared, self.watermark_size)
        watermark_64 = pil_to_tensor(watermark_resized)
        watermark_256 = self.watermark_preprocessor.tile_watermark(watermark_64)
        
        # 数据增强
        if self.mode == 'train':
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                watermark_256 = transforms.functional.hflip(watermark_256)
                watermark_64 = transforms.functional.hflip(watermark_64)
        
        return image, watermark_256, watermark_64


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        config: 配置对象
    
    Returns:
        (训练数据加载器, 验证数据加载器)
    """
    from config import image_config, train_config
    
    # 创建训练数据集
    train_dataset = WatermarkDataset(
        image_dir=train_config.TRAIN_IMAGE_DIR,
        watermark_dir=train_config.TRAIN_WATERMARK_DIR,
        target_size=image_config.TARGET_SIZE,
        watermark_size=image_config.WATERMARK_SIZE,
        mode='train'
    )
    
    # 创建验证数据集
    val_dataset = WatermarkDataset(
        image_dir=train_config.VAL_IMAGE_DIR,
        watermark_dir=train_config.VAL_WATERMARK_DIR,
        target_size=image_config.TARGET_SIZE,
        watermark_size=image_config.WATERMARK_SIZE,
        mode='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.BATCH_SIZE,
        shuffle=True,
        num_workers=train_config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.BATCH_SIZE,
        shuffle=False,
        num_workers=train_config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def create_single_image_dataloader(image_dir: str, 
                                   batch_size: int = 1) -> DataLoader:
    """
    创建单图像数据集的数据加载器（用于测试）
    
    Args:
        image_dir: 图像目录
        batch_size: 批次大小
    
    Returns:
        数据加载器
    """
    from config import image_config
    
    class SingleImageDataset(Dataset):
        def __init__(self, image_dir: str):
            self.image_files = []
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            if os.path.exists(image_dir):
                for f in os.listdir(image_dir):
                    if f.lower().endswith(valid_extensions):
                        self.image_files.append(os.path.join(image_dir, f))
            
            self.transform = transforms.Compose([
                transforms.Resize((image_config.TARGET_SIZE, image_config.TARGET_SIZE)),
                transforms.ToTensor(),
            ])
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            image = Image.open(self.image_files[idx]).convert('RGB')
            image = self.transform(image)
            return image, self.image_files[idx]
    
    dataset = SingleImageDataset(image_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return loader


if __name__ == '__main__':
    # 测试数据集
    print("测试数据集加载器...")
    
    # 使用示例数据路径
    image_dir = 'img'
    watermark_dir = 'img'
    
    if os.path.exists(image_dir) and os.path.exists(watermark_dir):
        # 创建数据集
        dataset = WatermarkDataset(
            image_dir=image_dir,
            watermark_dir=watermark_dir,
            target_size=256,
            watermark_size=64,
            mode='train'
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 获取一个样本
        if len(dataset) > 0:
            image, watermark_256, watermark_64 = dataset[0]
            print(f"图像尺寸: {image.shape}")
            print(f"水印(256)尺寸: {watermark_256.shape}")
            print(f"水印(64)尺寸: {watermark_64.shape}")
        
        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # 获取一个批次
        for batch in dataloader:
            images, watermarks_256, watermarks_64 = batch
            print(f"\n批次图像尺寸: {images.shape}")
            print(f"批次水印(256)尺寸: {watermarks_256.shape}")
            print(f"批次水印(64)尺寸: {watermarks_64.shape}")
            break
    else:
        print(f"目录不存在: {image_dir} 或 {watermark_dir}")
    
    print("\n数据集加载器测试完成！")
