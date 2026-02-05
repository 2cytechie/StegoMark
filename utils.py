"""
通用工具函数
包含PSNR、SSIM计算，模型保存/加载，日志记录等功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
import cv2


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        img1: 图像1
        img2: 图像2
        max_val: 像素最大值
    
    Returns:
        PSNR值 (dB)
    """
    # 确保两个张量在同一设备上
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
    
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, 
                   window_size: int = 11, max_val: float = 1.0) -> float:
    """
    计算SSIM (Structural Similarity Index)
    
    Args:
        img1: 图像1 [B, C, H, W] 或 [C, H, W]
        img2: 图像2
        window_size: 高斯窗口大小
        max_val: 像素最大值
    
    Returns:
        SSIM值 (0-1之间)
    """
    # 确保两个张量在同一设备上
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
    
    # 确保是4D张量
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # 创建高斯窗口
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                          for x in range(window_size)])
    window_1d = gauss / gauss.sum()
    
    # 创建2D高斯窗口
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    window = window_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, window_size, window_size]
    
    # 扩展窗口到所有通道
    channel = img1.size(1)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)
    
    # 常量
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_batch_metrics(images: torch.Tensor,
                           watermarked: torch.Tensor,
                           watermark: torch.Tensor,
                           extracted: torch.Tensor,
                           calculate_color_acc: bool = True) -> Dict[str, float]:
    """
    计算一批数据的评估指标

    Args:
        images: 原始图像 [B, C, H, W]
        watermarked: 含水印的图像 [B, C, H, W]
        watermark: 原始水印 [B, C, H, W]
        extracted: 提取的水印 [B, C, H, W]
        calculate_color_acc: 是否计算RGB彩色准确率

    Returns:
        指标字典
    """
    batch_size = images.size(0)

    psnr_values = []
    ssim_values = []

    for i in range(batch_size):
        psnr = calculate_psnr(images[i], watermarked[i])
        ssim = calculate_ssim(images[i], watermarked[i])
        psnr_values.append(psnr)
        ssim_values.append(ssim)

    # 计算RGB彩色准确率（作为主要准确率指标）
    from watermark_utils import ColorWatermarkAccuracyCalculator
    color_calculator = ColorWatermarkAccuracyCalculator(tolerance=0.1, threshold=0.5)
    color_metrics = color_calculator.calculate_tolerance_accuracy(extracted, watermark)

    # 使用RGB彩色准确率作为主要准确率指标
    metrics = {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'watermark_acc': color_metrics['overall'],  # RGB综合准确率作为主要指标
        'color_acc_r': color_metrics['R_channel'],
        'color_acc_g': color_metrics['G_channel'],
        'color_acc_b': color_metrics['B_channel'],
        'color_acc_overall': color_metrics['overall'],
        'color_correct_pixels': color_metrics['total_correct_pixels'],
        'color_total_pixels': color_metrics['total_pixels']
    }

    return metrics


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict,
                   filepath: str):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 评估指标
        filepath: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, 
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   device: str = 'cuda',
                   strict: bool = False) -> Tuple[int, Dict]:
    """
    加载模型检查点
    
    Args:
        filepath: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        device: 设备
        strict: 是否严格匹配state_dict
    
    Returns:
        (epoch, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics


class MetricsLogger:
    """指标记录器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.train_metrics = []
        self.val_metrics = []
        
        # 创建日志文件
        self.log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    def log_train(self, epoch: int, metrics: Dict):
        """记录训练指标"""
        metrics['epoch'] = epoch
        metrics['phase'] = 'train'
        self.train_metrics.append(metrics)
        self._save()
    
    def log_val(self, epoch: int, metrics: Dict):
        """记录验证指标"""
        metrics['epoch'] = epoch
        metrics['phase'] = 'val'
        self.val_metrics.append(metrics)
        self._save()
    
    def _save(self):
        """保存日志"""
        log_data = {
            'train': self.train_metrics,
            'val': self.val_metrics
        }
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def get_best_epoch(self, metric: str = 'psnr', phase: str = 'val') -> Tuple[int, float]:
        """
        获取最佳epoch
        
        Args:
            metric: 指标名称
            phase: 'train' 或 'val'
        
        Returns:
            (best_epoch, best_value)
        """
        metrics = self.val_metrics if phase == 'val' else self.train_metrics
        
        if not metrics:
            return 0, 0.0
        
        best_epoch = 0
        best_value = -float('inf')
        
        for m in metrics:
            if metric in m and m[metric] > best_value:
                best_value = m[metric]
                best_epoch = m['epoch']
        
        return best_epoch, best_value


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        初始化
        
        Args:
            patience: 耐心值（多少个epoch没有改善就停止）
            min_delta: 最小改善量
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前分数（越高越好）
        
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # 确保可重复性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """获取可用设备"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_image(tensor: torch.Tensor, filepath: str):
    """
    保存张量为图像文件
    
    Args:
        tensor: 图像张量 [C, H, W] 或 [H, W]
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 确保在CPU上
    tensor = tensor.cpu()
    
    # 确保值在[0, 1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为numpy
    if tensor.dim() == 3:
        # [C, H, W] -> [H, W, C]
        image_np = tensor.permute(1, 2, 0).numpy()
    else:
        image_np = tensor.numpy()
    
    # 转换为uint8
    image_np = (image_np * 255).astype(np.uint8)
    
    # 保存
    Image.fromarray(image_np).save(filepath)


def visualize_results(image: torch.Tensor,
                     watermark: torch.Tensor,
                     watermarked: torch.Tensor,
                     extracted: torch.Tensor,
                     output_path: str):
    """
    可视化结果并保存
    
    Args:
        image: 原始图像
        watermark: 原始水印
        watermarked: 含水印的图像
        extracted: 提取的水印
        output_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # 原始图像
    axes[0, 0].imshow(tensor_to_numpy(image))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 原始水印
    axes[0, 1].imshow(tensor_to_numpy(watermark))
    axes[0, 1].set_title('Original Watermark')
    axes[0, 1].axis('off')
    
    # 含水印的图像
    axes[1, 0].imshow(tensor_to_numpy(watermarked))
    psnr = calculate_psnr(image, watermarked)
    axes[1, 0].set_title(f'Watermarked Image (PSNR: {psnr:.2f} dB)')
    axes[1, 0].axis('off')
    
    # 提取的水印
    axes[1, 1].imshow(tensor_to_numpy(extracted))
    axes[1, 1].set_title('Extracted Watermark')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    将张量转换为numpy数组用于显示
    
    Args:
        tensor: 输入张量
    
    Returns:
        numpy数组
    """
    tensor = tensor.cpu().detach()
    tensor = torch.clamp(tensor, 0, 1)
    
    if tensor.dim() == 3:
        return tensor.permute(1, 2, 0).numpy()
    else:
        return tensor.numpy()


class AverageMeter:
    """计算和存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_model_summary(model: torch.nn.Module):
    """打印模型摘要"""
    print("\n" + "="*50)
    print("模型摘要")
    print("="*50)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        if param.requires_grad:
            trainable_params += params
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print("="*50 + "\n")


if __name__ == '__main__':
    # 测试工具函数
    print("测试通用工具函数...")
    
    # 测试PSNR和SSIM
    img1 = torch.rand(3, 256, 256)
    img2 = img1 + torch.randn(3, 256, 256) * 0.01
    img2 = torch.clamp(img2, 0, 1)
    
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    
    # 测试AverageMeter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"平均值: {meter.avg}")
    
    # 测试早停
    early_stop = EarlyStopping(patience=3)
    for i in range(10):
        should_stop = early_stop(float(i))
        print(f"Epoch {i}: score={i}, should_stop={should_stop}")
        if should_stop:
            break
    
    print("\n工具函数测试完成！")
