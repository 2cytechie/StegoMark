"""
评估指标模块
包含PSNR、SSIM、水印提取准确率等指标
"""
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage


def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算PSNR (峰值信噪比)
    
    Args:
        img1: 图像1 [B, C, H, W] 或 [C, H, W] 或 [H, W]
        img2: 图像2 [B, C, H, W] 或 [C, H, W] 或 [H, W]
        max_val: 像素最大值
        
    Returns:
        psnr: PSNR值 (dB)
    """
    if torch.is_tensor(img1):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
        return psnr.item()
    else:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr


def calculate_ssim(img1, img2, data_range=1.0):
    """
    计算SSIM (结构相似性)
    
    Args:
        img1: 图像1 [H, W, C] numpy array
        img2: 图像2 [H, W, C] numpy array
        data_range: 数据范围
        
    Returns:
        ssim: SSIM值
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
        img2 = img2.detach().cpu().numpy()
    
    # 确保是numpy数组
    img1 = np.array(img1)
    img2 = np.array(img2)
    
    # 处理不同维度
    if img1.ndim == 4:  # [B, C, H, W]
        ssim_vals = []
        for i in range(img1.shape[0]):
            # 转换为 [H, W, C]
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))
            ssim_val = ssim_skimage(im1, im2, data_range=data_range, channel_axis=2)
            ssim_vals.append(ssim_val)
        return np.mean(ssim_vals)
    elif img1.ndim == 3:  # [C, H, W]
        if img1.shape[0] in [1, 3]:  # 通道优先
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        return ssim_skimage(img1, img2, data_range=data_range, channel_axis=2)
    else:  # [H, W]
        return ssim_skimage(img1, img2, data_range=data_range)


def calculate_watermark_accuracy(watermark_original, watermark_extracted, threshold=0.9):
    """
    计算水印提取准确率
    
    Args:
        watermark_original: 原始水印 [B, C, H, W] 或 [C, H, W]
        watermark_extracted: 提取的水印 [B, C, H, W] 或 [C, H, W]
        threshold: 判定为正确的阈值
        
    Returns:
        accuracy: 准确率
    """
    if torch.is_tensor(watermark_original):
        # 归一化到[0, 1]
        w_orig = (watermark_original + 1) / 2  # 假设输入是[-1, 1]
        w_extr = (watermark_extracted + 1) / 2
        
        # 计算每个像素的相似度
        similarity = 1 - torch.abs(w_orig - w_extr)
        
        # 计算整体准确率
        accuracy = torch.mean((similarity > threshold).float()).item()
        return accuracy
    else:
        w_orig = (watermark_original + 1) / 2
        w_extr = (watermark_extracted + 1) / 2
        
        similarity = 1 - np.abs(w_orig - w_extr)
        accuracy = np.mean((similarity > threshold).astype(float))
        return accuracy


def calculate_bit_accuracy(watermark_original, watermark_extracted, threshold=0.5):
    """
    计算二值水印的比特准确率
    
    Args:
        watermark_original: 原始二值水印 [B, C, H, W]
        watermark_extracted: 提取的水印 [B, C, H, W]
        threshold: 二值化阈值
        
    Returns:
        bit_accuracy: 比特准确率
    """
    if torch.is_tensor(watermark_original):
        # 二值化
        w_orig_bin = (watermark_original > threshold).float()
        w_extr_bin = (watermark_extracted > threshold).float()
        
        # 计算匹配率
        correct = (w_orig_bin == w_extr_bin).float()
        bit_accuracy = torch.mean(correct).item()
        return bit_accuracy
    else:
        w_orig_bin = (watermark_original > threshold).astype(float)
        w_extr_bin = (watermark_extracted > threshold).astype(float)
        
        correct = (w_orig_bin == w_extr_bin).astype(float)
        bit_accuracy = np.mean(correct)
        return bit_accuracy


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.psnr_list = []
        self.ssim_list = []
        self.wm_acc_list = []
        self.bit_acc_list = []
    
    def update(self, img1, img2, wm_orig=None, wm_extr=None):
        """更新指标"""
        # PSNR
        psnr = calculate_psnr(img1, img2)
        self.psnr_list.append(psnr)
        
        # SSIM
        ssim = calculate_ssim(img1, img2)
        self.ssim_list.append(ssim)
        
        # 水印准确率
        if wm_orig is not None and wm_extr is not None:
            wm_acc = calculate_watermark_accuracy(wm_orig, wm_extr)
            self.wm_acc_list.append(wm_acc)
            
            bit_acc = calculate_bit_accuracy(wm_orig, wm_extr)
            self.bit_acc_list.append(bit_acc)
    
    def get_average(self):
        """获取平均指标"""
        result = {
            'psnr': np.mean(self.psnr_list) if self.psnr_list else 0,
            'ssim': np.mean(self.ssim_list) if self.ssim_list else 0,
        }
        
        if self.wm_acc_list:
            result['watermark_accuracy'] = np.mean(self.wm_acc_list)
        if self.bit_acc_list:
            result['bit_accuracy'] = np.mean(self.bit_acc_list)
        
        return result
    
    def print_metrics(self):
        """打印指标"""
        metrics = self.get_average()
        print("=" * 50)
        print("评估指标:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        if 'watermark_accuracy' in metrics:
            print(f"  水印准确率: {metrics['watermark_accuracy']:.4f}")
        if 'bit_accuracy' in metrics:
            print(f"  比特准确率: {metrics['bit_accuracy']:.4f}")
        print("=" * 50)
