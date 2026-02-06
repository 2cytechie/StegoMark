"""
水印损失函数模块
包含图像失真损失和水印提取损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WatermarkLoss(nn.Module):
    """水印损失函数"""
    
    def __init__(self, lambda_image=1.0, lambda_watermark=1.0):
        """
        Args:
            lambda_image: 图像失真损失权重
            lambda_watermark: 水印提取损失权重
        """
        super().__init__()
        self.lambda_image = lambda_image
        self.lambda_watermark = lambda_watermark
        
        # MSE损失
        self.mse_loss = nn.MSELoss()
        
        # L1损失 (可选，用于更好的边缘保持)
        self.l1_loss = nn.L1Loss()
    
    def forward(self, original_image, watermarked_image, original_watermark, extracted_watermark):
        """
        计算总损失
        
        Args:
            original_image: 原始图片 [B, 3, H, W]
            watermarked_image: 含水印图片 [B, 3, H, W]
            original_watermark: 原始水印 [B, 3, H, W]
            extracted_watermark: 提取的水印 [B, 3, H, W]
            
        Returns:
            total_loss: 总损失
            loss_dict: 各分项损失字典
        """
        # 图像失真损失 (MSE)
        image_loss = self.mse_loss(watermarked_image, original_image)
        
        # 图像失真损失 (L1，可选)
        image_l1_loss = self.l1_loss(watermarked_image, original_image)
        
        # 水印提取损失 (MSE)
        watermark_loss = self.mse_loss(extracted_watermark, original_watermark)
        
        # 水印提取损失 (L1，可选)
        watermark_l1_loss = self.l1_loss(extracted_watermark, original_watermark)
        
        # 总损失
        total_loss = (
            self.lambda_image * (image_loss + 0.5 * image_l1_loss) +
            self.lambda_watermark * (watermark_loss + 0.5 * watermark_l1_loss)
        )
        
        # 损失字典
        loss_dict = {
            'total': total_loss.item(),
            'image_mse': image_loss.item(),
            'image_l1': image_l1_loss.item(),
            'watermark_mse': watermark_loss.item(),
            'watermark_l1': watermark_l1_loss.item()
        }
        
        return total_loss, loss_dict


class PerceptualLoss(nn.Module):
    """感知损失 (使用预训练VGG特征)"""
    
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3'], weights=[1.0, 1.0, 1.0]):
        super().__init__()
        from torchvision import models
        
        # 加载预训练VGG
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        self.weights = weights
        
        # 构建特征提取器
        self.layer_name_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, target):
        """
        计算感知损失
        
        Args:
            pred: 预测图像 [B, 3, H, W]
            target: 目标图像 [B, 3, H, W]
            
        Returns:
            loss: 感知损失
        """
        # 归一化到VGG输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(pred.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(pred.device)
        
        pred_norm = (pred * 0.5 + 0.5 - mean) / std
        target_norm = (target * 0.5 + 0.5 - mean) / std
        
        loss = 0
        x_pred = pred_norm
        x_target = target_norm
        
        for i, (name, module) in enumerate(self.vgg._modules.items()):
            x_pred = module(x_pred)
            x_target = module(x_target)
            
            if i in [self.layer_name_mapping[l] for l in self.layers]:
                layer_idx = [self.layer_name_mapping[l] for l in self.layers].index(i)
                loss += self.weights[layer_idx] * self.criterion(x_pred, x_target)
        
        return loss


class SSIMLoss(nn.Module):
    """SSIM损失"""
    
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)
    
    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(-(x - window_size//2)**2 / float(2*sigma**2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        """
        计算SSIM损失
        
        Args:
            img1: 图像1 [B, C, H, W]
            img2: 图像2 [B, C, H, W]
            
        Returns:
            loss: 1 - SSIM
        """
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=self.channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=self.channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class CombinedWatermarkLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(self, lambda_image=1.0, lambda_watermark=1.0, 
                 use_perceptual=False, use_ssim=False,
                 lambda_perceptual=0.1, lambda_ssim=0.1):
        """
        Args:
            lambda_image: 图像损失权重
            lambda_watermark: 水印损失权重
            use_perceptual: 是否使用感知损失
            use_ssim: 是否使用SSIM损失
            lambda_perceptual: 感知损失权重
            lambda_ssim: SSIM损失权重
        """
        super().__init__()
        
        self.lambda_image = lambda_image
        self.lambda_watermark = lambda_watermark
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        
        # 基础损失
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 感知损失
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        
        # SSIM损失
        if use_ssim:
            self.ssim_loss = SSIMLoss()
    
    def forward(self, original_image, watermarked_image, original_watermark, extracted_watermark):
        """
        计算组合损失
        """
        loss_dict = {}
        
        # 图像损失
        image_mse = self.mse_loss(watermarked_image, original_image)
        image_l1 = self.l1_loss(watermarked_image, original_image)
        image_loss = image_mse + 0.5 * image_l1
        loss_dict['image_mse'] = image_mse.item()
        loss_dict['image_l1'] = image_l1.item()
        
        # 水印损失
        watermark_mse = self.mse_loss(extracted_watermark, original_watermark)
        watermark_l1 = self.l1_loss(extracted_watermark, original_watermark)
        watermark_loss = watermark_mse + 0.5 * watermark_l1
        loss_dict['watermark_mse'] = watermark_mse.item()
        loss_dict['watermark_l1'] = watermark_l1.item()
        
        # 总损失
        total_loss = self.lambda_image * image_loss + self.lambda_watermark * watermark_loss
        
        # 感知损失
        if self.use_perceptual:
            perc_loss = self.perceptual_loss(watermarked_image, original_image)
            total_loss += self.lambda_perceptual * perc_loss
            loss_dict['perceptual'] = perc_loss.item()
        
        # SSIM损失
        if self.use_ssim:
            ssim_loss = self.ssim_loss(watermarked_image, original_image)
            total_loss += self.lambda_ssim * ssim_loss
            loss_dict['ssim'] = ssim_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict
