"""
攻击模拟模块
模拟各种图像攻击，用于训练和测试水印系统的鲁棒性
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import io
from typing import Tuple, Optional, List
import random


class AttackSimulator:
    """攻击模拟器"""
    
    def __init__(self, config=None):
        """
        初始化攻击模拟器
        
        Args:
            config: 攻击配置对象
        """
        if config is None:
            # 使用默认配置
            from config import attack_config as config
        self.config = config
    
    def crop_attack(self, image: torch.Tensor, 
                    crop_ratio: Optional[float] = None) -> torch.Tensor:
        """
        裁剪攻击 - 随机裁剪图像的一部分，然后resize回原尺寸
        
        Args:
            image: 输入图像 [B, C, H, W] 或 [C, H, W]
            crop_ratio: 裁剪比例，None则使用随机值
        
        Returns:
            裁剪后的图像
        """
        if crop_ratio is None:
            crop_ratio = random.uniform(
                self.config.CROP_RATIO_MIN, 
                self.config.CROP_RATIO_MAX
            )
        
        # 处理不同维度输入
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        b, c, h, w = image.shape
        
        # 计算裁剪尺寸
        new_h = int(h * crop_ratio)
        new_w = int(w * crop_ratio)
        
        # 随机选择裁剪位置
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        # 裁剪
        cropped = image[:, :, top:top+new_h, left:left+new_w]
        
        # resize回原尺寸
        result = F.interpolate(cropped, size=(h, w), mode='bilinear', align_corners=False)
        
        if squeeze:
            result = result.squeeze(0)
        
        return result
    
    def rotate_attack(self, image: torch.Tensor,
                      angle: Optional[float] = None) -> torch.Tensor:
        """
        旋转攻击 - 旋转图像一定角度
        
        Args:
            image: 输入图像 [B, C, H, W] 或 [C, H, W]
            angle: 旋转角度，None则使用随机值
        
        Returns:
            旋转后的图像
        """
        if angle is None:
            angle = random.uniform(
                self.config.ROTATE_ANGLE_MIN,
                self.config.ROTATE_ANGLE_MAX
            )
        
        # 转换为numpy进行处理（affine_grid需要）
        device = image.device
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        b, c, h, w = image.shape
        
        # 创建旋转矩阵
        angle_rad = angle * np.pi / 180
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 仿射变换矩阵 [B, 2, 3]
        theta = torch.zeros(b, 2, 3, device=device)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a
        
        # 创建采样网格
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        
        # 采样
        rotated = F.grid_sample(image, grid, mode='bilinear', 
                               padding_mode='zeros', align_corners=False)
        
        if squeeze:
            rotated = rotated.squeeze(0)
        
        return rotated
    
    def scale_attack(self, image: torch.Tensor,
                     scale: Optional[float] = None) -> torch.Tensor:
        """
        缩放攻击 - 缩放图像后resize回原尺寸
        
        Args:
            image: 输入图像 [B, C, H, W] 或 [C, H, W]
            scale: 缩放比例，None则使用随机值
        
        Returns:
            缩放后的图像
        """
        if scale is None:
            scale = random.uniform(
                self.config.SCALE_MIN,
                self.config.SCALE_MAX
            )
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        b, c, h, w = image.shape
        
        # 计算新尺寸
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        # resize到新尺寸
        scaled = F.interpolate(image, size=(new_h, new_w), 
                              mode='bilinear', align_corners=False)
        
        # resize回原尺寸
        result = F.interpolate(scaled, size=(h, w), 
                              mode='bilinear', align_corners=False)
        
        if squeeze:
            result = result.squeeze(0)
        
        return result
    
    def gaussian_blur(self, image: torch.Tensor,
                      kernel_size: Optional[int] = None,
                      sigma: Optional[float] = None) -> torch.Tensor:
        """
        高斯模糊攻击
        
        Args:
            image: 输入图像 [B, C, H, W] 或 [C, H, W]
            kernel_size: 卷积核大小，None则使用随机值
            sigma: 标准差，None则使用随机值
        
        Returns:
            模糊后的图像
        """
        if kernel_size is None:
            kernel_size = random.choice(
                list(range(self.config.BLUR_KERNEL_MIN, 
                          self.config.BLUR_KERNEL_MAX + 1, 2))
            )
        
        if sigma is None:
            sigma = random.uniform(
                self.config.BLUR_SIGMA_MIN,
                self.config.BLUR_SIGMA_MAX
            )
        
        # 确保kernel_size是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 创建高斯核
        kernel = self._create_gaussian_kernel(kernel_size, sigma, image.device)
        kernel = kernel.to(image.dtype)
        
        # 应用卷积
        if image.dim() == 3:
            c, h, w = image.shape
            image = image.unsqueeze(0)
            squeeze = True
        else:
            b, c, h, w = image.shape
            squeeze = False
        
        # 为每个batch的每个通道重复核
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(b * c, 1, 1, 1)  # [b*c, 1, kernel_size, kernel_size]
        
        # 分组卷积 - 处理每个样本的每个通道
        padding = kernel_size // 2
        # 将batch和channel合并，然后使用groups=b*c进行深度可分离卷积
        blurred = F.conv2d(image.view(1, b * c, h, w), kernel, 
                          padding=padding, groups=b * c)
        blurred = blurred.view(b, c, h, w)
        
        if squeeze:
            blurred = blurred.squeeze(0)
        
        return blurred
    
    def _create_gaussian_kernel(self, kernel_size: int, 
                                sigma: float, device) -> torch.Tensor:
        """创建高斯核"""
        # 创建坐标轴
        x = torch.arange(kernel_size, dtype=torch.float32, device=device)
        x = x - kernel_size // 2
        
        # 创建2D高斯核
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # 创建2D核
        kernel_2d = gauss.unsqueeze(0) * gauss.unsqueeze(1)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        return kernel_2d
    
    def gaussian_noise(self, image: torch.Tensor,
                       mean: Optional[float] = None,
                       std: Optional[float] = None) -> torch.Tensor:
        """
        高斯噪声攻击
        
        Args:
            image: 输入图像 [B, C, H, W] 或 [C, H, W]
            mean: 噪声均值，None则使用配置值
            std: 噪声标准差，None则使用随机值
        
        Returns:
            加噪后的图像
        """
        if mean is None:
            mean = self.config.NOISE_MEAN
        
        if std is None:
            std = random.uniform(
                self.config.NOISE_STD_MIN,
                self.config.NOISE_STD_MAX
            )
        
        # 生成噪声
        noise = torch.randn_like(image) * std + mean
        
        # 添加噪声
        noisy = image + noise
        
        # 裁剪到[0, 1]范围
        noisy = torch.clamp(noisy, 0, 1)
        
        return noisy
    
    def jpeg_compression(self, image: torch.Tensor,
                         quality: Optional[int] = None) -> torch.Tensor:
        """
        JPEG压缩攻击
        
        Args:
            image: 输入图像 [B, C, H, W] 或 [C, H, W]
            quality: JPEG质量，None则使用随机值
        
        Returns:
            压缩后的图像
        """
        if quality is None:
            quality = random.randint(
                self.config.JPEG_QUALITY_MIN,
                self.config.JPEG_QUALITY_MAX
            )
        
        # 处理batch维度
        if image.dim() == 3:
            images = [image]
            squeeze = True
        else:
            images = [image[i] for i in range(image.shape[0])]
            squeeze = False
        
        results = []
        for img in images:
            # 转换为PIL图像 (detach first to avoid gradient issues)
            img_np = img.detach().permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            # JPEG压缩
            buffer = io.BytesIO()
            pil_img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            
            # 读取压缩后的图像
            compressed_img = Image.open(buffer)
            compressed_np = np.array(compressed_img).astype(np.float32) / 255.0
            
            # 转换回tensor
            compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1)
            compressed_tensor = compressed_tensor.to(image.device, image.dtype)
            
            results.append(compressed_tensor)
        
        result = torch.stack(results)
        
        if squeeze:
            result = result.squeeze(0)
        
        return result
    
    def combined_attack(self, image: torch.Tensor,
                       attacks: Optional[List[str]] = None) -> torch.Tensor:
        """
        组合攻击 - 按顺序应用多种攻击
        
        Args:
            image: 输入图像
            attacks: 攻击类型列表，None则随机选择
        
        Returns:
            攻击后的图像
        """
        if attacks is None:
            # 随机选择2-3种攻击
            all_attacks = ['crop', 'rotate', 'scale', 'blur', 'noise', 'jpeg']
            num_attacks = random.randint(2, 3)
            attacks = random.sample(all_attacks, num_attacks)
        
        result = image
        for attack_type in attacks:
            if attack_type == 'crop':
                result = self.crop_attack(result)
            elif attack_type == 'rotate':
                result = self.rotate_attack(result)
            elif attack_type == 'scale':
                result = self.scale_attack(result)
            elif attack_type == 'blur':
                result = self.gaussian_blur(result)
            elif attack_type == 'noise':
                result = self.gaussian_noise(result)
            elif attack_type == 'jpeg':
                result = self.jpeg_compression(result)
        
        return result
    
    def random_attack(self, image: torch.Tensor,
                     attack_prob: Optional[float] = None) -> Tuple[torch.Tensor, str]:
        """
        随机选择一种攻击
        
        Args:
            image: 输入图像
            attack_prob: 应用攻击的概率，None则使用配置值
        
        Returns:
            (攻击后的图像, 攻击类型)
        """
        if attack_prob is None:
            attack_prob = self.config.ATTACK_PROBABILITY
        
        # 以一定概率不应用攻击
        if random.random() > attack_prob:
            return image, 'none'
        
        # 随机选择攻击类型
        attack_types = ['crop', 'rotate', 'scale', 'blur', 'noise', 'jpeg', 'combined']
        attack_type = random.choice(attack_types)
        
        if attack_type == 'crop':
            result = self.crop_attack(image)
        elif attack_type == 'rotate':
            result = self.rotate_attack(image)
        elif attack_type == 'scale':
            result = self.scale_attack(image)
        elif attack_type == 'blur':
            result = self.gaussian_blur(image)
        elif attack_type == 'noise':
            result = self.gaussian_noise(image)
        elif attack_type == 'jpeg':
            result = self.jpeg_compression(image)
        elif attack_type == 'combined':
            result = self.combined_attack(image)
        else:
            result = image
        
        return result, attack_type
    
    def apply_attack_by_name(self, image: torch.Tensor, 
                            attack_name: str, **kwargs) -> torch.Tensor:
        """
        根据名称应用特定攻击
        
        Args:
            image: 输入图像
            attack_name: 攻击名称
            **kwargs: 攻击参数
        
        Returns:
            攻击后的图像
        """
        if attack_name == 'none' or attack_name is None:
            return image
        elif attack_name == 'crop':
            return self.crop_attack(image, **kwargs)
        elif attack_name == 'rotate':
            return self.rotate_attack(image, **kwargs)
        elif attack_name == 'scale':
            return self.scale_attack(image, **kwargs)
        elif attack_name == 'blur':
            return self.gaussian_blur(image, **kwargs)
        elif attack_name == 'noise':
            return self.gaussian_noise(image, **kwargs)
        elif attack_name == 'jpeg':
            return self.jpeg_compression(image, **kwargs)
        elif attack_name == 'combined':
            return self.combined_attack(image, **kwargs)
        else:
            raise ValueError(f"未知的攻击类型: {attack_name}")


def apply_attacks_during_training(image: torch.Tensor, 
                                  epoch: int,
                                  total_epochs: int,
                                  config=None) -> torch.Tensor:
    """
    训练过程中根据当前epoch应用不同强度的攻击
    
    Args:
        image: 输入图像
        epoch: 当前epoch
        total_epochs: 总epoch数
        config: 配置对象
    
    Returns:
        攻击后的图像
    """
    if config is None:
        from config import attack_config as config
    
    if not config.ENABLE_ATTACK_TRAINING:
        return image
    
    simulator = AttackSimulator(config)
    
    # 根据训练阶段调整攻击强度
    stage1_end = total_epochs // 4
    stage2_end = total_epochs // 2
    
    if epoch < stage1_end:
        # 第一阶段：轻度攻击
        if random.random() < 0.3:
            attack_type = random.choice(['noise', 'jpeg'])
            return simulator.apply_attack_by_name(image, attack_type)
    
    elif epoch < stage2_end:
        # 第二阶段：中度攻击
        if random.random() < 0.5:
            attack_type = random.choice(['crop', 'scale', 'blur', 'noise', 'jpeg'])
            return simulator.apply_attack_by_name(image, attack_type)
    
    else:
        # 第三阶段：全部攻击
        if random.random() < 0.7:
            attack_type = random.choice(['crop', 'rotate', 'scale', 'blur', 'noise', 'jpeg', 'combined'])
            return simulator.apply_attack_by_name(image, attack_type)
    
    return image


if __name__ == '__main__':
    # 测试攻击模块
    print("测试攻击模拟模块...")
    
    # 创建测试图像
    test_image = torch.randn(1, 3, 256, 256)
    print(f"原始图像尺寸: {test_image.shape}")
    
    simulator = AttackSimulator()
    
    # 测试各种攻击
    attacks_to_test = [
        ('crop', simulator.crop_attack),
        ('rotate', simulator.rotate_attack),
        ('scale', simulator.scale_attack),
        ('blur', simulator.gaussian_blur),
        ('noise', simulator.gaussian_noise),
        ('jpeg', simulator.jpeg_compression),
    ]
    
    for name, attack_func in attacks_to_test:
        result = attack_func(test_image)
        print(f"{name}攻击后尺寸: {result.shape}")
    
    # 测试组合攻击
    combined_result = simulator.combined_attack(test_image)
    print(f"组合攻击后尺寸: {combined_result.shape}")
    
    # 测试随机攻击
    random_result, attack_type = simulator.random_attack(test_image)
    print(f"随机攻击类型: {attack_type}, 结果尺寸: {random_result.shape}")
    
    print("\n攻击模拟模块测试完成！")
