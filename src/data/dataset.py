import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
from typing import Optional, Callable, Tuple
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class WatermarkDataset(Dataset):
    """水印数据集类 - 支持任意尺寸目标图片和数据增强"""
    
    def __init__(
        self,
        image_dir: str,
        watermark_dir: str,
        image_size: int = 256,
        watermark_size: int = 64
    ):
        self.image_dir = image_dir
        self.watermark_dir = watermark_dir
        self.image_size = image_size
        self.watermark_size = watermark_size
        
        # 获取图像文件列表
        self.image_files = self._get_image_files(image_dir)
        self.watermark_files = self._get_image_files(watermark_dir)
        
        if len(self.image_files) == 0:
            raise ValueError(f"在{image_dir}中未找到任何图像文件")
        if len(self.watermark_files) == 0:
            raise ValueError(f"在{watermark_dir}中未找到任何水印文件")
    
    def _get_image_files(self, directory: str) -> list:
        """获取目录中的所有图像文件"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        files = []
        for f in os.listdir(directory):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(os.path.join(directory, f))
        return sorted(files)
    
    def _pad_to_square(self, img: Image.Image) -> Image.Image:
        """
        将图像填充到1:1比例（正方形）
        使用白色背景填充
        """
        width, height = img.size
        
        # 如果已经是正方形，直接返回
        if width == height:
            return img
        
        # 计算新尺寸（取最大值）
        new_size = max(width, height)
        
        # 创建白色背景的新图像
        new_img = Image.new('RGB', (new_size, new_size), (255, 255, 255))
        
        # 计算粘贴位置（居中）
        left = (new_size - width) // 2
        top = (new_size - height) // 2
        
        # 粘贴原图
        new_img.paste(img, (left, top))
        
        return new_img
    
    def _resize_watermark(self, img: Image.Image, size: int) -> Image.Image:
        """
        将水印resize到指定尺寸
        """
        return img.resize((size, size), Image.Resampling.LANCZOS)
    
    def _augment_image(self, img: Image.Image) -> Image.Image:
        """
        对图像进行数据增强
        包括：随机水平翻转、随机垂直翻转、随机裁剪、随机缩放、随机模糊、颜色调整
        """
        
        # 1. 随机水平翻转
        if random.random() > 0.5:
            img = TF.hflip(img)
        
        # 2. 随机垂直翻转
        if random.random() > 0.5:
            img = TF.vflip(img)
        
        # 3. 随机旋转（0, 90, 180, 270度）
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)
        
        # 4. 随机裁剪（保持原始尺寸）
        if random.random() > 0.3:
            width, height = img.size
            # 随机裁剪比例 0.8-1.0
            scale = random.uniform(0.8, 1.0)
            crop_width = int(width * scale)
            crop_height = int(height * scale)
            
            # 随机裁剪位置
            left = random.randint(0, width - crop_width)
            top = random.randint(0, height - crop_height)
            
            img = TF.crop(img, top, left, crop_height, crop_width)
            # 调整回原始尺寸
            img = TF.resize(img, (height, width))
        
        # 5. 随机高斯模糊
        if random.random() > 0.7:
            sigma = random.uniform(0.05, 1.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # 6. 颜色调整
        if random.random() > 0.5:
            # 亮度调整
            brightness_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_brightness(img, brightness_factor)
        
        if random.random() > 0.5:
            # 对比度调整
            contrast_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_contrast(img, contrast_factor)
        
        if random.random() > 0.5:
            # 饱和度调整
            saturation_factor = random.uniform(0.8, 1.2)
            img = TF.adjust_saturation(img, saturation_factor)
        
        if random.random() > 0.5:
            # 色调调整
            hue_factor = random.uniform(-0.1, 0.1)
            img = TF.adjust_hue(img, hue_factor)
        
        return img
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
            image: 载体图像 (C, H, W) - 保持原始尺寸
            watermark: 水印图像 (C, H, W) - 平铺到与载体相同尺寸
        """
        # 加载载体图像（保持原始尺寸）
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        target_w, target_h = image.size  # 获取目标尺寸
        
        # 数据增强（在PIL阶段进行）
        image = self._augment_image(image)
        
        # 随机选择水印
        watermark_path = random.choice(self.watermark_files)
        watermark = Image.open(watermark_path)
        
        # 转换水印为RGB
        if watermark.mode != 'RGB':
            watermark = watermark.convert('RGB')
        
        # ========== 水印处理流程 ==========
        # 步骤1: 填充到1:1比例
        watermark = self._pad_to_square(watermark)
        
        # 步骤2: resize到指定尺寸
        image = self._resize_watermark(image, self.image_size)
        watermark = self._resize_watermark(watermark, self.watermark_size)
        
        # 步骤3: 转换为tensor
        image = self._image_transform(image)
        watermark = self._image_transform(watermark)
        
        return image, watermark
    
    def _image_transform(self, img: Image.Image) -> torch.Tensor:
        """默认图像变换：转为tensor，保持原始尺寸"""
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转为tensor [0, 1]
        ])
        return transform(img)


def test_dataset():
    """测试数据集"""
    from src.config import config
    
    # 创建测试数据集
    dataset = WatermarkDataset(
        image_dir=config.test_image_dir,           # 目标图片目录
        watermark_dir=config.test_watermark_dir,       # 水印图片目录
        image_size=config.image_size,
        watermark_size=config.watermark_size
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 随机获取一个样本
    idx = random.randint(0, len(dataset) - 1)
    image, watermark = dataset[idx]
    
    print(f"\n目标图片尺寸: {image.shape}")
    print(f"水印尺寸: {watermark.shape}")
    print(f"目标图片范围: [{image.min():.3f}, {image.max():.3f}]")
    print(f"水印范围: [{watermark.min():.3f}, {watermark.max():.3f}]")
    
    # 保存测试图像
    import os
    os.makedirs('outputs', exist_ok=True)
    
    # 保存目标图片
    img_pil = TF.to_pil_image(image)
    img_pil.save('outputs/test_dateset_target.png')
    print(f"\n保存目标图片: outputs/test_dateset_target.png")
    
    # 保存水印
    wm_pil = TF.to_pil_image(watermark)
    wm_pil.save('outputs/test_dateset_watermark.png')
    print(f"保存水印: outputs/test_dateset_watermark.png")
    
    # 测试多次获取，观察数据增强效果
    print("\n=== 测试数据增强效果（获取同一张图片5次）===")
    for i in range(5):
        img, wm = dataset[1]  # 每次都获取第一张图片
        img_pil = TF.to_pil_image(img)
        img_pil.save(f'outputs/test_augmentation_{i+1}.png')
        print(f"第{i+1}次获取 - 尺寸: {img.shape}, 范围: [{img.min():.3f}, {img.max():.3f}]")
    
    # 测试水印处理流程
    print("\n=== 水印处理流程测试 ===")
    
    # 加载原始水印
    idx = random.randint(0, len(dataset.watermark_files) - 1)
    wm_path = dataset.watermark_files[idx]
    original_wm = Image.open(wm_path).convert('RGB')
    print(f"原始水印尺寸: {original_wm.size}")
    
    # 步骤1: 填充到1:1
    padded_wm = dataset._pad_to_square(original_wm)
    print(f"填充后尺寸: {padded_wm.size}")
    padded_wm.save('outputs/test_dateset_watermark_padded.png')
    
    # 步骤2: resize到64x64
    resized_wm = dataset._resize_watermark(padded_wm, 64)
    print(f"resize后尺寸: {resized_wm.size}")
    resized_wm.save('outputs/test_dateset_watermark_resized.png')


if __name__ == "__main__":
    test_dataset()