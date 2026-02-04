import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from config import config
from tqdm import tqdm

class WatermarkLoss(nn.Module):
    """
    水印任务专用损失函数
    结合MSE损失（保持图像质量）、SSIM损失（感知质量）、BCE损失（优化水印提取）和几何变换损失
    """
    
    def __init__(self, embedding_weight=1.0, extraction_weight=1.0, ssim_weight=0.5, transform_weight=0.1):
        """
        初始化损失函数
        
        Args:
            embedding_weight: 嵌入损失权重
            extraction_weight: 提取损失权重
            ssim_weight: SSIM损失权重
            transform_weight: 几何变换损失权重
        """
        super(WatermarkLoss, self).__init__()
        self.initial_embedding_weight = embedding_weight
        self.initial_extraction_weight = extraction_weight
        self.initial_ssim_weight = ssim_weight
        self.initial_transform_weight = transform_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.current_epoch = 0  # 当前epoch
        self.psnr_history = []  # PSNR历史记录
        self.accuracy_history = []  # 准确率历史记录
    
    def ssim_loss(self, x, y):
        """
        计算SSIM损失
        
        Args:
            x: 输入张量
            y: 目标张量
            
        Returns:
            ssim_loss: SSIM损失值
        """
        # 计算SSIM
        C1 = (0.01 * 1.0) ** 2
        C2 = (0.03 * 1.0) ** 2
        
        # 使用更大的卷积核尺寸，提高SSIM计算的准确性
        mu_x = nn.functional.avg_pool2d(x, 5, 1, 2)
        mu_y = nn.functional.avg_pool2d(y, 5, 1, 2)
        
        sigma_x = nn.functional.avg_pool2d(x ** 2, 5, 1, 2) - mu_x ** 2
        sigma_y = nn.functional.avg_pool2d(y ** 2, 5, 1, 2) - mu_y ** 2
        sigma_xy = nn.functional.avg_pool2d(x * y, 5, 1, 2) - mu_x * mu_y
        
        # 稳定性改进，添加小的epsilon
        epsilon = 1e-8
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2 + epsilon))
        ssim_loss = 1 - ssim_map.mean()
        
        return ssim_loss
    
    def forward(self, watermarked_images, original_images, extracted_watermarks, original_watermarks, transform_params=None, target_transform_params=None):
        """
        计算损失
        
        Args:
            watermarked_images: 含水印图像
            original_images: 原始图像
            extracted_watermarks: 提取的水印
            original_watermarks: 原始水印
            transform_params: 估计的几何变换参数
            target_transform_params: 目标几何变换参数
            
        Returns:
            total_loss: 总损失
        """
        # 计算嵌入损失（保持图像质量）
        embedding_loss = self.mse_loss(watermarked_images, original_images)
        
        # 计算SSIM损失（感知质量）
        ssim_loss_val = self.ssim_loss(watermarked_images, original_images)
        
        # 计算提取损失（优化水印提取）
        extraction_loss = self.bce_loss(extracted_watermarks, original_watermarks)
        
        # 计算几何变换损失
        transform_loss = 0.0
        if transform_params is not None and target_transform_params is not None:
            transform_loss = self.mse_loss(transform_params, target_transform_params)
        
        # 动态权重调整
        # 根据epoch调整权重
        epoch_factor = min(self.current_epoch / 50, 1.0)  # 前50个epoch逐渐调整
        
        # 基本权重
        embedding_weight = self.initial_embedding_weight
        extraction_weight = self.initial_extraction_weight
        ssim_weight = self.initial_ssim_weight
        transform_weight = self.initial_transform_weight
        
        # 根据训练进度调整权重
        if self.current_epoch < 15:
            # 前15个epoch，优先保证图像质量
            embedding_weight = 2.5
            ssim_weight = 1.0
            extraction_weight = 0.5
        elif self.current_epoch < 40:
            # 15-40个epoch，平衡图像质量和水印提取
            embedding_weight = 1.5
            ssim_weight = 0.8
            extraction_weight = 1.0
        else:
            # 40个epoch后，优先保证水印提取
            embedding_weight = 1.0
            ssim_weight = 0.5
            extraction_weight = 2.0
        
        # 根据PSNR历史调整权重
        if len(self.psnr_history) > 5:
            recent_psnr = sum(self.psnr_history[-5:]) / 5
            if recent_psnr < 30.0:
                # PSNR较低，显著增加嵌入损失权重
                embedding_weight *= 1.5
                ssim_weight *= 1.5
            elif recent_psnr > 35.0:
                # PSNR较高，适度减少嵌入损失权重
                embedding_weight *= 0.7
                extraction_weight *= 1.3
        
        # 根据准确率历史调整权重
        if len(self.accuracy_history) > 5:
            recent_accuracy = sum(self.accuracy_history[-5:]) / 5
            if recent_accuracy < 0.7:
                # 准确率较低，显著增加提取损失权重
                extraction_weight *= 1.5
            elif recent_accuracy > 0.9:
                # 准确率较高，适度减少提取损失权重
                extraction_weight *= 0.8
                embedding_weight *= 1.2
        
        # 损失值归一化，确保不同损失分量的尺度一致
        embedding_loss_norm = embedding_loss / (embedding_loss + extraction_loss + 1e-8)
        extraction_loss_norm = extraction_loss / (embedding_loss + extraction_loss + 1e-8)
        
        # 基于损失值动态调整权重
        if embedding_loss_norm > 0.7:
            # 嵌入损失占比过高，减少其权重
            embedding_weight *= 0.8
            extraction_weight *= 1.2
        elif extraction_loss_norm > 0.7:
            # 提取损失占比过高，减少其权重
            extraction_weight *= 0.8
            embedding_weight *= 1.2
        
        # 总损失
        total_loss = (embedding_weight * embedding_loss + 
                     ssim_weight * ssim_loss_val + 
                     extraction_weight * extraction_loss +
                     transform_weight * transform_loss)
        
        # 更新epoch计数器
        self.current_epoch += 1
        
        return total_loss
    
    def update_history(self, psnr_value, accuracy_value):
        """
        更新PSNR和准确率历史记录
        
        Args:
            psnr_value: 当前PSNR值
            accuracy_value: 当前准确率值
        """
        self.psnr_history.append(psnr_value)
        self.accuracy_history.append(accuracy_value)
        
        # 保持历史记录长度在合理范围内
        max_history_length = 20
        if len(self.psnr_history) > max_history_length:
            self.psnr_history = self.psnr_history[-max_history_length:]
        if len(self.accuracy_history) > max_history_length:
            self.accuracy_history = self.accuracy_history[-max_history_length:]

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
            attack_params: 攻击参数
        """
        random_attack = random.randint(0, config.ATTACK_TYPE_LEN - 1)
        attack_params = {}

        if random_attack == 0:
            x = self.add_gaussian_noise(x)
            attack_params['type'] = 'gaussian'
        elif random_attack == 1:
            x = self.add_jpeg_compression(x)
            attack_params['type'] = 'jpeg'
        elif random_attack == 2:
            x, crop_params = self.add_random_crop(x)
            attack_params['type'] = 'crop'
            attack_params['params'] = crop_params
        elif random_attack == 3:
            x = self.add_gaussian_blur(x)
            attack_params['type'] = 'blur'
        elif random_attack == 4:
            x, rotation_params = self.add_rotation(x)
            attack_params['type'] = 'rotation'
            attack_params['params'] = rotation_params
        elif random_attack == 5:
            x, scale_params = self.add_scaling(x)
            attack_params['type'] = 'scale'
            attack_params['params'] = scale_params

        return x, attack_params

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
    
    def add_random_crop(self, x):
        """
        添加随机裁剪
        
        Args:
            x: 输入张量
            
        Returns:
            cropped_x: 裁剪后保持裁剪原始尺寸的张量
            crop_params: 裁剪参数 (裁剪比例, 起始位置)
        """
        b, c, h, w = x.shape
        
        # 随机裁剪比例
        crop_ratios = torch.rand(b, device=x.device) * (config.MAX_CROP_RATIO - config.MIN_CROP_RATIO) + config.MIN_CROP_RATIO
        
        # 计算裁剪尺寸
        crop_sizes_h = (crop_ratios * h).int()
        crop_sizes_w = (crop_ratios * w).int()
        
        # 确保裁剪尺寸至少为1
        crop_sizes_h = torch.max(crop_sizes_h, torch.tensor(1, device=x.device))
        crop_sizes_w = torch.max(crop_sizes_w, torch.tensor(1, device=x.device))
        
        # 随机裁剪位置
        start_h = torch.randint(0, h - crop_sizes_h.max() + 1, (b,), device=x.device)
        start_w = torch.randint(0, w - crop_sizes_w.max() + 1, (b,), device=x.device)
        
        # 裁剪
        cropped = []
        crop_params = []
        for i in range(b):
            # 裁剪区域
            crop_h = crop_sizes_h[i]
            crop_w = crop_sizes_w[i]
            crop = x[i:i+1, :, start_h[i]:start_h[i]+crop_h, start_w[i]:start_w[i]+crop_w]
            # 调整回原始尺寸
            crop = F.interpolate(crop, size=(h, w), mode='bilinear', align_corners=False)
            cropped.append(crop)
            crop_params.append((crop_ratios[i].item(), start_h[i].item(), start_w[i].item()))
        
        return torch.cat(cropped, dim=0), crop_params
    
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
    
    def add_rotation(self, x):
        """
        添加随机旋转
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            rotated_x: 旋转后的张量
            rotation_params: 旋转参数 (旋转角度)
        """
        try:
            # 验证输入
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"输入必须是torch.Tensor，但收到{type(x)}")
            if x.dim() != 4:
                raise ValueError(f"输入张量必须是4维 (B, C, H, W)，但收到{x.dim()}维")
            
            b, c, h, w = x.shape
            
            # 生成随机旋转角度
            angle = torch.rand(b) * 2 * config.MAX_ROTATION_ANGLE - config.MAX_ROTATION_ANGLE
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
            
            # 记录旋转参数
            rotation_params = angle.tolist()
            
            return rotated_x, rotation_params
            
        except Exception as e:
            print(f"Error in add_rotation: {str(e)}")
            # 如果旋转失败，返回原始图像和空参数
            b = x.shape[0]
            return x, [0.0] * b
    
    def add_scaling(self, x):
        """
        添加随机缩放
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
            
        Returns:
            scaled_x: 缩放后的张量
            scale_params: 缩放参数 (缩放比例)
        """
        try:
            # 验证输入
            if not isinstance(x, torch.Tensor):
                raise TypeError(f"输入必须是torch.Tensor，但收到{type(x)}")
            if x.dim() != 4:
                raise ValueError(f"输入张量必须是4维 (B, C, H, W)，但收到{x.dim()}维")
            
            b, c, h, w = x.shape
            
            # 生成随机缩放比例
            scale = torch.rand(b) * (config.MAX_SCALE - config.MIN_SCALE) + config.MIN_SCALE
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
            
            # 记录缩放参数
            scale_params = scale.tolist()
            
            return scaled_x, scale_params
            
        except Exception as e:
            print(f"Error in add_scaling: {str(e)}")
            # 如果缩放失败，返回原始图像和空参数
            b = x.shape[0]
            return x, [1.0] * b
    
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
            attack_params: 攻击参数列表
        """
        attack_params = []
        
        for attack in attack_type:
            if attack == 'random':
                x, params = self.add_random_attack(x)
                attack_params.append(params)
            elif attack == 'gaussian':
                x = self.add_gaussian_noise(x)
                attack_params.append({'type': 'gaussian'})
            elif attack == 'jpeg':
                x = self.add_jpeg_compression(x)
                attack_params.append({'type': 'jpeg'})
            elif attack == 'crop':
                x, params = self.add_random_crop(x)
                attack_params.append({'type': 'crop', 'params': params})
            elif attack == 'blur':
                x = self.add_gaussian_blur(x)
                attack_params.append({'type': 'blur'})
            elif attack == 'rotate':
                x, params = self.add_rotation(x)
                attack_params.append({'type': 'rotation', 'params': params})
            elif attack == 'scale':
                x, params = self.add_scaling(x)
                attack_params.append({'type': 'scale', 'params': params})
        
        return x, attack_params

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
        self.attack_strength = 0.1  # 初始攻击强度
        self.max_attack_strength = 1.0  # 最大攻击强度
        self.strength_growth_rate = 0.02  # 每次迭代的强度增长速率
        self.current_epoch = 0  # 当前epoch
        self.epoch_growth_factor = 1.1  # 每个epoch的攻击强度增长因子
    
    def train_epoch(self, train_loader, attack_training=False, mix_method='none'):
        """
        训练一个 epoch
        
        Args:
            train_loader: 训练数据加载器
            attack_training: 是否进行对抗性训练
            mix_method: 混合数据方法，可选 'none', 'mixup', 'cutmix'
            
        Returns:
            avg_loss: 平均损失
            avg_psnr: 平均PSNR值
        """
        self.model.train()
        total_loss = 0
        total_psnr = 0
        total_samples = 0
        
        # 每个epoch开始时，根据epoch调整攻击强度
        if attack_training:
            # 基于epoch的攻击强度调度，使用更平缓的增长曲线
            self.attack_strength = min(0.1 * (1.05 ** self.current_epoch), self.max_attack_strength)
            print(f"Current attack strength: {self.attack_strength:.3f}")
        
        # 动态选择攻击类型，随着训练进行增加攻击类型
        if attack_training:
            if self.current_epoch < 15:
                # 前15个epoch只使用简单攻击
                attack_types = ['gaussian']
            elif self.current_epoch < 40:
                # 15-40个epoch使用中等攻击
                attack_types = ['gaussian', 'jpeg', 'blur']
            else:
                # 40个epoch后使用全部攻击
                attack_types = ['gaussian', 'jpeg', 'crop', 'blur', 'rotate', 'scale']
        else:
            attack_types = []
        
        # 导入mixup和cutmix函数
        from scripts.train import mixup_data, cutmix_data
        
        # 标签平滑参数
        label_smoothing = 0.1
        
        for images, watermarks in tqdm(train_loader, desc="Training"):
            images = images.to(self.model.device)
            watermarks = watermarks.to(self.model.device)
            
            # 应用标签平滑
            # 将硬标签 [0, 1] 转换为软标签 [label_smoothing, 1-label_smoothing]
            watermarks = watermarks * (1 - label_smoothing) + 0.5 * label_smoothing
            
            # 应用MixUp或CutMix
            if mix_method == 'mixup':
                mixed_images, watermarks_a, watermarks_b, lam = mixup_data(images, watermarks, alpha=0.4)
                # 前向传播
                watermarked_images, _ = self.model(mixed_images, watermarks_a)
            elif mix_method == 'cutmix':
                mixed_images, watermarks_a, watermarks_b, lam = cutmix_data(images, watermarks, alpha=0.4)
                # 前向传播
                watermarked_images, _ = self.model(mixed_images, watermarks_a)
            else:
                # 普通训练
                mixed_images = images
                watermarks_a = watermarks
                watermarks_b = watermarks
                lam = 1.0
                # 前向传播
                watermarked_images, _ = self.model(images, watermarks)
            
            # 应用噪声层
            if attack_training:
                # 动态调整攻击强度和类型
                noisy_images, attack_params = self.noise_layer(watermarked_images, attack_types)
                # 每次迭代增加攻击强度
                self.attack_strength = min(self.attack_strength + 0.01, self.max_attack_strength)
            else:
                noisy_images = watermarked_images
                attack_params = []
            
            # 从噪声图像中提取水印
            extracted_watermarks = self.model.extract(noisy_images)
            
            # 计算损失
            if isinstance(self.criterion, WatermarkLoss):
                # 使用新的WatermarkLoss
                if mix_method in ['mixup', 'cutmix']:
                    # 计算混合损失
                    loss_a = self.criterion(watermarked_images, mixed_images, extracted_watermarks, watermarks_a)
                    loss_b = self.criterion(watermarked_images, mixed_images, extracted_watermarks, watermarks_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    loss = self.criterion(watermarked_images, images, extracted_watermarks, watermarks)
            else:
                # 使用传统损失计算方式
                embedding_loss = self.criterion(watermarked_images, mixed_images)
                if mix_method in ['mixup', 'cutmix']:
                    extraction_loss_a = self.criterion(extracted_watermarks, watermarks_a)
                    extraction_loss_b = self.criterion(extracted_watermarks, watermarks_b)
                    extraction_loss = lam * extraction_loss_a + (1 - lam) * extraction_loss_b
                else:
                    extraction_loss = self.criterion(extracted_watermarks, watermarks)
                loss = 1.5 * embedding_loss + extraction_loss  # 增加嵌入损失权重
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        
        # 更新epoch计数器
        self.current_epoch += 1
        
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
        total_pixels = 0
        
        with torch.no_grad():
            for images, watermarks in tqdm(val_loader, desc="Validating"):
                images = images.to(self.model.device)
                watermarks = watermarks.to(self.model.device)
                
                # 前向传播
                watermarked_images, _ = self.model(images, watermarks)
                
                # 应用噪声层
                noisy_images, attack_params = self.noise_layer(watermarked_images)
                
                # 从噪声图像中提取水印
                extracted_watermarks = self.model.extract(noisy_images)
                
                # 计算损失
                if isinstance(self.criterion, WatermarkLoss):
                    # 使用新的WatermarkLoss
                    loss = self.criterion(watermarked_images, images, extracted_watermarks, watermarks)
                else:
                    # 使用传统损失计算方式
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
                total_pixels += watermarks.numel()
        
        # 处理空数据情况
        if total_samples == 0:
            return 0.0, 0.0, 0.0
        
        val_loss = total_loss / total_samples
        val_accuracy = correct / total_pixels if total_pixels > 0 else 0
        val_psnr = total_psnr / total_samples
        return val_loss, val_accuracy, val_psnr