"""
数据增强模块
包含翻转、模糊、颜色调整、随机裁剪等增强操作
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
from PIL import Image, ImageFilter


class WatermarkTransforms:
    """水印数据增强"""
    
    def __init__(self, flip_prob=0.5, blur_prob=0.3, color_jitter_prob=0.3, 
                 crop_prob=0.5, watermark_size=64):
        """
        Args:
            flip_prob: 翻转概率
            blur_prob: 模糊概率
            color_jitter_prob: 颜色调整概率
            crop_prob: 裁剪概率
            watermark_size: 水印目标尺寸
        """
        self.flip_prob = flip_prob
        self.blur_prob = blur_prob
        self.color_jitter_prob = color_jitter_prob
        self.crop_prob = crop_prob
        self.watermark_size = watermark_size
        
        # 颜色抖动
        self.color_jitter = T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
    
    def __call__(self, image, is_watermark=False):
        """
        应用数据增强
        
        Args:
            image: PIL Image
            is_watermark: 是否为水印图像
            
        Returns:
            增强后的PIL Image
        """
        # 水印图像只进行基础处理
        if is_watermark:
            return self._process_watermark(image)
        
        # 目标图片进行完整增强
        return self._process_image(image)
    
    def _process_watermark(self, image):
        """处理水印图像"""
        # 转换为RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize到固定尺寸
        image = image.resize((self.watermark_size, self.watermark_size), Image.LANCZOS)
        
        return image
    
    def _process_image(self, image):
        """处理目标图像"""
        # 随机水平翻转
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
        
        # 随机垂直翻转
        if random.random() < self.flip_prob:
            image = TF.vflip(image)
        
        # 随机模糊
        if random.random() < self.blur_prob:
            kernel_size = random.choice([3, 5])
            image = image.filter(ImageFilter.GaussianBlur(radius=kernel_size//2))
        
        # 随机颜色调整
        if random.random() < self.color_jitter_prob:
            image = self.color_jitter(image)
        
        # 随机裁剪 (保持宽高比)
        if random.random() < self.crop_prob:
            width, height = image.size
            crop_scale = random.uniform(0.8, 1.0)
            new_width = int(width * crop_scale)
            new_height = int(height * crop_scale)
            
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            
            image = TF.crop(image, top, left, new_height, new_width)
            image = image.resize((width, height), Image.LANCZOS)
        
        return image


class ImagePreprocessor:
    """图像预处理"""
    
    def __init__(self, watermark_size=64):
        self.watermark_size = watermark_size
    
    def preprocess_target(self, image):
        """
        预处理目标图片
        - 保持任意尺寸
        - 转换为RGB
        - 归一化到[-1, 1]
        
        Args:
            image: PIL Image
            
        Returns:
            torch.Tensor [C, H, W]
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为tensor并归一化到[-1, 1]
        transform = T.Compose([
            T.ToTensor(),  # [0, 1]
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        return transform(image)
    
    def preprocess_watermark(self, image):
        """
        预处理水印图片
        - 宽高比填充为1:1
        - resize到64x64
        - 平铺到目标尺寸(可选)
        
        Args:
            image: PIL Image
            
        Returns:
            torch.Tensor [C, H, W]
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 获取原始尺寸
        width, height = image.size
        
        # 填充为1:1
        if width != height:
            max_size = max(width, height)
            new_image = Image.new('RGB', (max_size, max_size), (255, 255, 255))
            
            # 居中放置
            left = (max_size - width) // 2
            top = (max_size - height) // 2
            new_image.paste(image, (left, top))
            image = new_image
        
        # Resize到64x64
        image = image.resize((self.watermark_size, self.watermark_size), Image.LANCZOS)
        
        # 转换为tensor并归一化到[-1, 1]
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        return transform(image)
    
    def tile_watermark(self, watermark, target_size):
        """
        将水印平铺到目标尺寸
        
        Args:
            watermark: torch.Tensor [C, H, W]
            target_size: (H, W) 目标尺寸
            
        Returns:
            torch.Tensor [C, target_H, target_W]
        """
        C, wm_h, wm_w = watermark.shape
        target_h, target_w = target_size
        
        # 计算需要平铺的次数
        tiles_h = (target_h + wm_h - 1) // wm_h
        tiles_w = (target_w + wm_w - 1) // wm_w
        
        # 平铺
        tiled = watermark.repeat(1, tiles_h, tiles_w)
        
        # 裁剪到目标尺寸
        tiled = tiled[:, :target_h, :target_w]
        
        return tiled


class AttackSimulator:
    """攻击模拟器 (用于训练时增强鲁棒性)"""
    
    def __init__(self, crop_prob=0.5, rotate_prob=0.3, blur_prob=0.3,
                 jpeg_prob=0.3, noise_prob=0.3):
        self.crop_prob = crop_prob
        self.rotate_prob = rotate_prob
        self.blur_prob = blur_prob
        self.jpeg_prob = jpeg_prob
        self.noise_prob = noise_prob
    
    def __call__(self, image):
        """
        随机应用攻击
        
        Args:
            image: torch.Tensor [B, C, H, W] 或 [C, H, W]
            
        Returns:
            攻击后的图像
        """
        if torch.rand(1) < self.crop_prob:
            image = self.random_crop(image)
        
        if torch.rand(1) < self.rotate_prob:
            image = self.random_rotate(image)
        
        if torch.rand(1) < self.blur_prob:
            image = self.gaussian_blur(image)
        
        if torch.rand(1) < self.noise_prob:
            image = self.add_noise(image)
        
        return image
    
    def random_crop(self, image, scale_range=(0.7, 1.0)):
        """随机裁剪"""
        if image.dim() == 3:
            C, H, W = image.shape
            scale = random.uniform(*scale_range)
            new_h, new_w = int(H * scale), int(W * scale)
            
            top = random.randint(0, H - new_h)
            left = random.randint(0, W - new_w)
            
            cropped = image[:, top:top+new_h, left:left+new_w]
            return torch.nn.functional.interpolate(
                cropped.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0)
        else:
            B, C, H, W = image.shape
            scale = random.uniform(*scale_range)
            new_h, new_w = int(H * scale), int(W * scale)
            
            top = random.randint(0, H - new_h)
            left = random.randint(0, W - new_w)
            
            cropped = image[:, :, top:top+new_h, left:left+new_w]
            return torch.nn.functional.interpolate(
                cropped, size=(H, W), mode='bilinear', align_corners=False
            )
    
    def random_rotate(self, image, angle_range=(-15, 15)):
        """随机旋转"""
        angle = random.uniform(*angle_range)
        
        if image.dim() == 3:
            return TF.rotate(image, angle)
        else:
            # 对batch处理
            rotated = []
            for i in range(image.shape[0]):
                rotated.append(TF.rotate(image[i], angle))
            return torch.stack(rotated)
    
    def gaussian_blur(self, image, kernel_range=(3, 7)):
        """高斯模糊"""
        kernel_size = random.choice(range(kernel_range[0], kernel_range[1]+1, 2))
        sigma = random.uniform(0.1, 2.0)
        
        if image.dim() == 3:
            return TF.gaussian_blur(image, kernel_size, [sigma, sigma])
        else:
            return TF.gaussian_blur(image, kernel_size, [sigma, sigma])
    
    def add_noise(self, image, std_range=(0.01, 0.05)):
        """添加高斯噪声"""
        std = random.uniform(*std_range)
        noise = torch.randn_like(image) * std
        return torch.clamp(image + noise, -1, 1)
