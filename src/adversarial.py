import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from config import config
from tqdm import tqdm

class NoiseLayer(nn.Module):
    """
    噪声层
    用于模拟各种攻击
    """
    
    def __init__(self):
        """
        初始化噪声层
        """
        super(NoiseLayer, self).__init__()
    
    def add_random_attack(self, x):
        """
        添加随机扰动
        
        Args:
            x: 输入张量
            
        Returns:
            perturbed_x: 攻击后的张量
        """
        random_attack = random.randint(0, config.ATTACK_TYPE_LEN - 1)

        if random_attack == 0:
            x = self.add_gaussian_noise(x)
        elif random_attack == 1:
            x = self.add_jpeg_compression(x)
        elif random_attack == 2:
            x = self.add_random_crop(x)
        elif random_attack == 3:
            x = self.add_gaussian_blur(x)
        elif random_attack == 4:
            x = self.add_rotation(x)
        elif random_attack == 5:
            x = self.add_scaling(x)

        return x

    def add_gaussian_noise(self, x, std=config.GAUSSIAN_NOISE_STD):
        """
        添加高斯噪声
        
        Args:
            x: 输入张量
            std: 噪声标准差
            
        Returns:
            noisy_x: 带噪声的张量
        """
        noise = torch.randn_like(x) * std
        noisy_x = x + noise
        noisy_x = torch.clamp(noisy_x, 0, 1)
        return noisy_x
    
    def add_jpeg_compression(self, x, quality=config.JPEG_QUALITY):
        """
        模拟JPEG压缩
        
        Args:
            x: 输入张量
            quality: JPEG质量
            
        Returns:
            compressed_x: 压缩后的张量
        """
        # 简化的JPEG压缩模拟
        # 实际应用中可以使用更复杂的实现
        b, c, h, w = x.shape
        
        # 下采样
        scale = max(1, int(100 / quality))
        if scale > 1:
            x_down = F.interpolate(x, scale_factor=1/scale, mode='bilinear', align_corners=True)
            x_up = F.interpolate(x_down, size=(h, w), mode='bilinear', align_corners=True)
        else:
            x_up = x
        
        return x_up
    
    def add_random_crop(self, x, crop_size=config.CROP_SIZE):
        """
        添加随机裁剪
        
        Args:
            x: 输入张量
            crop_size: 裁剪尺寸
            
        Returns:
            cropped_x: 裁剪后保持裁剪原始尺寸的张量
        """
        b, c, h, w = x.shape
        
        # 随机裁剪位置
        start_h = torch.randint(0, h - crop_size + 1, (b,))
        start_w = torch.randint(0, w - crop_size + 1, (b,))
        
        # 裁剪
        cropped = []
        for i in range(b):
            # 裁剪区域
            crop = x[i:i+1, :, start_h[i]:start_h[i]+crop_size, start_w[i]:start_w[i]+crop_size]
            cropped.append(crop)
        
        return torch.cat(cropped, dim=0)
    
    def get_cropped_region(self, x, crop_size=config.CROP_SIZE):
        """
        获取随机裁剪区域（保持原始尺寸）
        
        Args:
            x: 输入张量
            crop_size: 裁剪尺寸
            
        Returns:
            cropped_x: 裁剪后的张量（保持裁剪区域的原始尺寸）
        """
        b, c, h, w = x.shape
        
        # 随机裁剪位置
        start_h = torch.randint(0, h - crop_size + 1, (b,))
        start_w = torch.randint(0, w - crop_size + 1, (b,))
        
        # 裁剪
        cropped = []
        for i in range(b):
            # 只执行裁剪操作，不进行缩放
            crop = x[i:i+1, :, start_h[i]:start_h[i]+crop_size, start_w[i]:start_w[i]+crop_size]
            cropped.append(crop)
        
        return torch.cat(cropped, dim=0)
    
    def add_gaussian_blur(self, x, kernel_size=3, sigma=1.0):
        """
        添加高斯模糊
        
        Args:
            x: 输入张量
            kernel_size: 卷积核大小
            sigma: 高斯标准差
            
        Returns:
            blurred_x: 模糊后的张量
        """
        # 创建高斯核
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(x.device)
        
        # 应用高斯模糊
        blurred_x = F.conv2d(x, kernel, padding=kernel_size//2, groups=3)
        return blurred_x
    
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """
        创建高斯核
        
        Args:
            kernel_size: 核大小
            sigma: 高斯标准差
            
        Returns:
            kernel: 高斯核
        """
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - kernel_size // 2
        x = x / (sigma * 2)
        kernel = torch.exp(-x**2)
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, 1) * kernel.view(1, 1, 1, kernel_size)
        kernel = kernel.expand(3, 1, kernel_size, kernel_size)
        return kernel
    
    def add_rotation(self, x, max_angle=config.MAX_ROTATION_ANGLE):
        """
        添加随机旋转
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            max_angle: 最大旋转角度（度）
            
        Returns:
            rotated_x: 旋转后的张量
        """
        try:
            # 验证输入
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"输入必须是torch.Tensor，但收到{type(x)}")
            if x.dim() != 4:
                raise ValueError(f"输入张量必须是4维 (B, C, H, W)，但收到{x.dim()}维")
            
            b, c, h, w = x.shape
            
            # 生成随机旋转角度
            angle = torch.rand(b) * 2 * max_angle - max_angle
            angle = angle.to(x.device)
            
            # 创建旋转矩阵（弧度）
            theta = angle * math.pi / 180
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            
            # 构建仿射变换矩阵
            # 在PyTorch的affine_grid中，坐标系是归一化的 [-1, 1]
            # 原点(0,0)在图像中心，不需要像像素坐标那样计算中心偏移
            transform = torch.zeros(b, 2, 3, device=x.device)
            transform[:, 0, 0] = cos_theta
            transform[:, 0, 1] = -sin_theta
            transform[:, 1, 0] = sin_theta
            transform[:, 1, 1] = cos_theta
            
            # 应用变换
            # align_corners=False 是推荐的标准做法
            grid = F.affine_grid(transform, x.size(), align_corners=False)
            rotated_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            
            # 验证输出
            if torch.isnan(rotated_x).any():
                raise RuntimeError("旋转后的张量包含NaN值")
            if torch.isinf(rotated_x).any():
                raise RuntimeError("旋转后的张量包含Inf值")
            
            return rotated_x
            
        except Exception as e:
            print(f"Error in add_rotation: {str(e)}")
            # 如果旋转失败，返回原始图像
            return x
    
    def add_scaling(self, x, min_scale=config.MIN_SCALE, max_scale=config.MAX_SCALE):
        """
        添加随机缩放
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            min_scale: 最小缩放比例
            max_scale: 最大缩放比例
            
        Returns:
            scaled_x: 缩放后的张量
        """
        try:
            # 验证输入
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"输入必须是torch.Tensor，但收到{type(x)}")
            if x.dim() != 4:
                raise ValueError(f"输入张量必须是4维 (B, C, H, W)，但收到{x.dim()}维")
            
            b, c, h, w = x.shape
            
            # 生成随机缩放比例
            scale = torch.rand(b) * (max_scale - min_scale) + min_scale
            scale = scale.to(x.device)
            
            # 构建仿射变换矩阵
            # 在PyTorch的affine_grid中，坐标系是归一化的 [-1, 1]
            # 原点(0,0)在图像中心，不需要像像素坐标那样计算中心偏移
            transform = torch.zeros(b, 2, 3, device=x.device)
            transform[:, 0, 0] = scale
            transform[:, 1, 1] = scale
            
            # 应用变换
            # align_corners=False 是推荐的标准做法
            grid = F.affine_grid(transform, x.size(), align_corners=False)
            scaled_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            
            # 验证输出
            if torch.isnan(scaled_x).any():
                raise RuntimeError("缩放后的张量包含NaN值")
            if torch.isinf(scaled_x).any():
                raise RuntimeError("缩放后的张量包含Inf值")
            
            return scaled_x
            
        except Exception as e:
            print(f"Error in add_scaling: {str(e)}")
            # 如果缩放失败，返回原始图像
            return x
    
    def forward(self, x, attack_type=config.ATTACK_TYPE):
        """
        前向传播
        
        Args:
            x: 输入张量
            attack_type: 攻击类型
                'random': 随机攻击
                'gaussian': 高斯噪声
                'jpeg': JPEG压缩
                'crop': 随机裁剪
                'blur': 高斯模糊
                'rotate': 旋转
                'scale': 缩放
            
        Returns:
            x: 带攻击的张量
        """
        for attack in attack_type:
            if attack == 'random':
                x = self.add_random_attack(x)
            elif attack == 'gaussian':
                x = self.add_gaussian_noise(x)
            elif attack == 'jpeg':
                x = self.add_jpeg_compression(x)
            elif attack == 'crop':
                x = self.add_random_crop(x)
            elif attack == 'blur':
                x = self.add_gaussian_blur(x)
            elif attack == 'rotate':
                x = self.add_rotation(x)
            elif attack == 'scale':
                x = self.add_scaling(x)
        
        return x

class AdversarialTrainer:
    """
    对抗性训练器
    用于训练水印模型，支持对抗性攻击
    """
    
    def __init__(self, model, optimizer, criterion):
        """
        初始化对抗性训练器
        
        Args:
            model: 水印模型
            optimizer: 优化器
            criterion: 损失函数
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.noise_layer = NoiseLayer()
    
    def train_epoch(self, train_loader, attack_training=False):
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
            attack_training: 是否进行对抗性训练
            
        Returns:
            avg_loss: 平均损失
            avg_psnr: 平均PSNR值
        """
        self.model.train()
        total_loss = 0
        total_psnr = 0
        total_samples = 0
        
        for images, watermarks in tqdm(train_loader, desc="Training"):
            images = images.to(self.model.device)
            watermarks = watermarks.to(self.model.device)
            
            # 前向传播
            watermarked_images, _ = self.model(images, watermarks)
            
            # 应用噪声层
            if attack_training:
                noisy_images = self.noise_layer(watermarked_images)
            else:
                noisy_images = watermarked_images
            
            # 从噪声图像中提取水印
            extracted_watermarks = self.model.extract(noisy_images)
            
            # 计算损失 - 增加嵌入损失权重以保护图像质量
            embedding_loss = self.criterion(watermarked_images, images)
            extraction_loss = self.criterion(extracted_watermarks, watermarks)
            loss = 1.5 * embedding_loss + extraction_loss  # 增加嵌入损失权重
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算PSNR
            batch_psnr = 0
            for i in range(images.size(0)):
                original = images[i].cpu().numpy().transpose(1, 2, 0) * 255
                watermarked = watermarked_images[i].cpu().detach().numpy().transpose(1, 2, 0) * 255
                original = original.astype(np.uint8)
                watermarked = watermarked.astype(np.uint8)
                batch_psnr += psnr(original, watermarked)
            batch_psnr /= images.size(0)
            
            total_loss += loss.item() * images.size(0)
            total_psnr += batch_psnr * images.size(0)
            total_samples += images.size(0)
        
        avg_loss = total_loss / total_samples
        avg_psnr = total_psnr / total_samples
        return avg_loss, avg_psnr
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            val_loss: 验证损失
            val_accuracy: 验证准确率
            val_psnr: 验证PSNR值
        """
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_samples = 0
        correct = 0
        
        with torch.no_grad():
            for images, watermarks in tqdm(val_loader, desc="Validating"):
                images = images.to(self.model.device)
                watermarks = watermarks.to(self.model.device)
                
                # 前向传播
                watermarked_images, _ = self.model(images, watermarks)
                
                # 应用噪声层
                noisy_images = self.noise_layer(watermarked_images)
                
                # 从噪声图像中提取水印
                extracted_watermarks = self.model.extract(noisy_images)
                
                # 计算损失
                embedding_loss = self.criterion(watermarked_images, images)
                extraction_loss = self.criterion(extracted_watermarks, watermarks)
                loss = embedding_loss + extraction_loss
                
                # 计算PSNR
                batch_psnr = 0
                for i in range(images.size(0)):
                    original = images[i].cpu().numpy().transpose(1, 2, 0) * 255
                    watermarked = watermarked_images[i].cpu().numpy().transpose(1, 2, 0) * 255
                    original = original.astype(np.uint8)
                    watermarked = watermarked.astype(np.uint8)
                    batch_psnr += psnr(original, watermarked)
                batch_psnr /= images.size(0)
                
                total_loss += loss.item() * images.size(0)
                total_psnr += batch_psnr * images.size(0)
                total_samples += images.size(0)
                
                # 计算准确率
                predicted = (extracted_watermarks > 0.5).float()
                correct += (predicted == watermarks).sum().item()
        
        val_loss = total_loss / total_samples
        val_accuracy = correct / (total_samples * watermarks.numel() / images.size(0))
        val_psnr = total_psnr / total_samples
        return val_loss, val_accuracy, val_psnr