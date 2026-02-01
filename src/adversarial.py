import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

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
    
    def add_random_crop(self, x, crop_size=32):
        """
        添加随机裁剪
        
        Args:
            x: 输入张量
            crop_size: 裁剪尺寸
            
        Returns:
            cropped_x: 裁剪后的张量
        """
        b, c, h, w = x.shape
        
        # 随机裁剪位置
        start_h = torch.randint(0, h - crop_size + 1, (b,))
        start_w = torch.randint(0, w - crop_size + 1, (b,))
        
        # 裁剪
        cropped = []
        for i in range(b):
            crop = x[i:i+1, :, start_h[i]:start_h[i]+crop_size, start_w[i]:start_w[i]+crop_size]
            crop = F.interpolate(crop, size=(h, w), mode='bilinear', align_corners=True)
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
    
    def forward(self, x, attack_type='all'):
        """
        前向传播
        
        Args:
            x: 输入张量
            attack_type: 攻击类型
                'all': 所有攻击
                'gaussian': 高斯噪声
                'jpeg': JPEG压缩
                'crop': 随机裁剪
                'blur': 高斯模糊
            
        Returns:
            x: 带攻击的张量
        """
        if attack_type == 'all' or attack_type == 'gaussian':
            x = self.add_gaussian_noise(x)
        
        if attack_type == 'all' or attack_type == 'jpeg':
            x = self.add_jpeg_compression(x)
        
        if attack_type == 'all' or attack_type == 'crop':
            x = self.add_random_crop(x)
        
        if attack_type == 'all' or attack_type == 'blur':
            x = self.add_gaussian_blur(x)
        
        return x

class FGSM:
    """
    快速梯度符号法 (FGSM) 攻击
    """
    
    def __init__(self, model, eps=config.ATTACK_EPS):
        """
        初始化FGSM攻击
        
        Args:
            model: 目标模型
            eps: 扰动大小
        """
        self.model = model
        self.eps = eps
    
    def attack(self, watermarked_image, target_watermark):
        """
        执行FGSM攻击
        
        Args:
            watermarked_image: 含水印图像
            target_watermark: 目标水印
            
        Returns:
            adversarial_image: 对抗样本
        """
        watermarked_image.requires_grad = True
        
        # 提取水印
        extracted_watermark = self.model.extract(watermarked_image)
        
        # 计算损失
        loss = F.mse_loss(extracted_watermark, target_watermark)
        
        # 计算梯度
        loss.backward()
        
        # 生成对抗样本
        gradient = watermarked_image.grad.data
        sign_gradient = gradient.sign()
        adversarial_image = watermarked_image + self.eps * sign_gradient
        adversarial_image = torch.clamp(adversarial_image, 0, 1)
        
        return adversarial_image

class PGD:
    """
    投影梯度下降 (PGD) 攻击
    """
    
    def __init__(self, model, eps=config.ATTACK_EPS, alpha=0.01, iterations=config.ATTACK_ITERATIONS):
        """
        初始化PGD攻击
        
        Args:
            model: 目标模型
            eps: 扰动大小
            alpha: 步长
            iterations: 迭代次数
        """
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.iterations = iterations
    
    def attack(self, watermarked_image, target_watermark):
        """
        执行PGD攻击
        
        Args:
            watermarked_image: 含水印图像
            target_watermark: 目标水印
            
        Returns:
            adversarial_image: 对抗样本
        """
        # 初始化对抗样本
        adversarial_image = watermarked_image.clone().detach()
        adversarial_image = adversarial_image + torch.randn_like(adversarial_image) * 0.001
        adversarial_image = torch.clamp(adversarial_image, 0, 1)
        
        for i in range(self.iterations):
            # 创建一个新的叶子张量
            adv_image = adversarial_image.clone().detach()
            adv_image.requires_grad = True
            
            # 提取水印
            extracted_watermark = self.model.extract(adv_image)
            
            # 计算损失
            loss = F.mse_loss(extracted_watermark, target_watermark)
            
            # 计算梯度
            loss.backward()
            
            # 更新对抗样本
            gradient = adv_image.grad.data
            sign_gradient = gradient.sign()
            adversarial_image = adversarial_image + self.alpha * sign_gradient
            
            # 投影到扰动范围内
            perturbation = torch.clamp(adversarial_image - watermarked_image, -self.eps, self.eps)
            adversarial_image = torch.clamp(watermarked_image + perturbation, 0, 1)
        
        return adversarial_image

class AdversarialTrainer:
    """
    对抗性训练器
    """
    
    def __init__(self, model, optimizer, criterion):
        """
        初始化对抗性训练器
        
        Args:
            model: 目标模型
            optimizer: 优化器
            criterion: 损失函数
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.noise_layer = NoiseLayer()
        self.pgd_attack = PGD(model)
    
    def train_step(self, cover_images, watermarks):
        """
        训练步骤
        
        Args:
            cover_images: 载体图像
            watermarks: 水印
            
        Returns:
            loss: 训练损失
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. 嵌入水印
        watermarked_images = self.model.embed(cover_images, watermarks)
        
        # 2. 应用噪声层
        noisy_images = self.noise_layer(watermarked_images)
        
        # 3. 提取水印
        extracted_watermarks = self.model.extract(noisy_images)
        
        # 4. 计算损失
        # 水印提取损失
        extraction_loss = self.criterion(extracted_watermarks, watermarks)
        # 图像质量保持损失
        quality_loss = self.criterion(watermarked_images, cover_images)
        # 组合损失，调整权重平衡
        loss = extraction_loss + 10.0 * quality_loss
        
        # 5. 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def adversarial_train_step(self, cover_images, watermarks):
        """
        对抗性训练步骤
        
        Args:
            cover_images: 载体图像
            watermarks: 水印
            
        Returns:
            loss: 训练损失
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 1. 嵌入水印
        watermarked_images = self.model.embed(cover_images, watermarks)
        
        # 2. 生成对抗样本
        adversarial_images = self.pgd_attack.attack(watermarked_images, watermarks)
        
        # 3. 提取水印（从对抗样本）
        extracted_watermarks = self.model.extract(adversarial_images)
        
        # 4. 计算损失
        # 水印提取损失
        extraction_loss = self.criterion(extracted_watermarks, watermarks)
        # 图像质量保持损失
        quality_loss = self.criterion(watermarked_images, cover_images)
        # 组合损失，调整权重平衡
        loss = extraction_loss + 10.0 * quality_loss
        
        # 5. 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, adversarial_training=True):
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            adversarial_training: 是否使用对抗性训练
            
        Returns:
            avg_loss: 平均损失
        """
        total_loss = 0
        total_batches = 0
        
        from tqdm import tqdm
        
        for batch in tqdm(dataloader, desc="Training", unit="batch"):
            cover_images, watermarks = batch
            # 使用配置的设备
            cover_images = cover_images.to(config.DEVICE)
            watermarks = watermarks.to(config.DEVICE)
            
            if adversarial_training:
                loss = self.adversarial_train_step(cover_images, watermarks)
            else:
                loss = self.train_step(cover_images, watermarks)
            
            total_loss += loss
            total_batches += 1
        
        avg_loss = total_loss / total_batches
        return avg_loss
    
    def validate(self, dataloader):
        """
        验证
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            avg_loss: 平均损失
            accuracy: 提取准确率
        """
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        total_batches = 0
        
        from tqdm import tqdm
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation", unit="batch"):
                cover_images, watermarks = batch
                cover_images = cover_images.to(config.DEVICE)
                watermarks = watermarks.to(config.DEVICE)
                
                # 嵌入水印
                watermarked_images = self.model.embed(cover_images, watermarks)
                
                # 应用噪声
                noisy_images = self.noise_layer(watermarked_images)
                
                # 提取水印
                extracted_watermarks = self.model.extract(noisy_images)
                
                # 计算损失
                loss = self.criterion(extracted_watermarks, watermarks)
                total_loss += loss.item()
                
                # 计算准确率
                accuracy = self.calculate_accuracy(extracted_watermarks, watermarks)
                total_accuracy += accuracy
                
                total_batches += 1
        
        avg_loss = total_loss / total_batches
        avg_accuracy = total_accuracy / total_batches
        
        return avg_loss, avg_accuracy
    
    def calculate_accuracy(self, extracted, target):
        """
        计算提取准确率
        
        Args:
            extracted: 提取的水印
            target: 目标水印
            
        Returns:
            accuracy: 准确率
        """
        # 二值化
        extracted_binary = (extracted > 0.5).float()
        target_binary = (target > 0.5).float()
        
        # 计算准确率
        correct = (extracted_binary == target_binary).float().mean()
        
        return correct.item()