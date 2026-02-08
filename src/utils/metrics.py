import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage


def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    输入:
        img1, img2: 图像张量 (B, C, H, W) 或 (C, H, W) 或 (H, W)
        max_val: 最大值（通常是1.0或255.0）
    输出:
        psnr: PSNR值 (dB)
    """
    if isinstance(img1, torch.Tensor):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)
        return psnr.item()
    else:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
        return psnr


def calculate_ssim(img1, img2, max_val=1.0):
    """
    计算SSIM (Structural Similarity Index)
    
    输入:
        img1, img2: 图像张量 (B, C, H, W) 或 (C, H, W) 或 (H, W)
        max_val: 最大值
    输出:
        ssim: SSIM值
    """
    if isinstance(img1, torch.Tensor):
        # 转换为numpy
        if img1.dim() == 4:
            # 批量处理
            ssim_values = []
            for i in range(img1.size(0)):
                im1 = img1[i].permute(1, 2, 0).cpu().numpy()
                im2 = img2[i].permute(1, 2, 0).cpu().numpy()
                ssim_val = ssim_skimage(im1, im2, data_range=max_val, channel_axis=2)
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        elif img1.dim() == 3:
            im1 = img1.permute(1, 2, 0).cpu().numpy()
            im2 = img2.permute(1, 2, 0).cpu().numpy()
            return ssim_skimage(im1, im2, data_range=max_val, channel_axis=2)
        else:
            im1 = img1.cpu().numpy()
            im2 = img2.cpu().numpy()
            return ssim_skimage(im1, im2, data_range=max_val)
    else:
        if img1.ndim == 3:
            return ssim_skimage(img1, img2, data_range=max_val, channel_axis=2)
        else:
            return ssim_skimage(img1, img2, data_range=max_val)


def calculate_nc(watermark1, watermark2):
    """
    计算NC (Normalized Correlation) 归一化相关系数
    
    输入:
        watermark1, watermark2: 水印张量 (B, C, H, W) 或 (C, H, W) 或 (H, W)
    输出:
        nc: NC值 [-1, 1]
    """
    if isinstance(watermark1, torch.Tensor):
        # 二值化
        wm1_bin = (watermark1 > 0.5).float()
        wm2_bin = (watermark2 > 0.5).float()
        
        # 展平
        wm1_flat = wm1_bin.view(wm1_bin.size(0), -1)
        wm2_flat = wm2_bin.view(wm2_bin.size(0), -1)
        
        # 计算NC
        numerator = torch.sum(wm1_flat * wm2_flat, dim=1)
        denominator = torch.sqrt(torch.sum(wm1_flat ** 2, dim=1) * torch.sum(wm2_flat ** 2, dim=1))
        
        nc = numerator / (denominator + 1e-8)
        return nc.mean().item()
    else:
        # numpy版本
        wm1_bin = (watermark1 > 0.5).astype(np.float32)
        wm2_bin = (watermark2 > 0.5).astype(np.float32)
        
        wm1_flat = wm1_bin.flatten()
        wm2_flat = wm2_bin.flatten()
        
        numerator = np.sum(wm1_flat * wm2_flat)
        denominator = np.sqrt(np.sum(wm1_flat ** 2) * np.sum(wm2_flat ** 2))
        
        return numerator / (denominator + 1e-8)


def calculate_ber(watermark1, watermark2):
    """
    计算BER (Bit Error Rate) 误码率
    
    输入:
        watermark1, watermark2: 水印张量 (B, C, H, W) 或 (C, H, W) 或 (H, W)
    输出:
        ber: BER值 [0, 1]
    """
    if isinstance(watermark1, torch.Tensor):
        # 二值化
        wm1_bin = (watermark1 > 0.5).float()
        wm2_bin = (watermark2 > 0.5).float()
        
        # 计算不同像素的数量
        diff = torch.abs(wm1_bin - wm2_bin)
        ber = torch.mean(diff)
        return ber.item()
    else:
        # numpy版本
        wm1_bin = (watermark1 > 0.5).astype(np.float32)
        wm2_bin = (watermark2 > 0.5).astype(np.float32)
        
        diff = np.abs(wm1_bin - wm2_bin)
        ber = np.mean(diff)
        return ber


def evaluate_watermark_system(
    original_image,
    watermarked_image,
    original_watermark,
    extracted_watermark
):
    """
    评估水印系统性能
    
    返回字典包含:
        - psnr: 载体图像与含水印图像的PSNR
        - ssim: 载体图像与含水印图像的SSIM
        - nc: 原始水印与提取水印的NC
        - ber: 原始水印与提取水印的BER
    """
    results = {
        'psnr': calculate_psnr(original_image, watermarked_image),
        'ssim': calculate_ssim(original_image, watermarked_image),
        'nc': calculate_nc(original_watermark, extracted_watermark),
        'ber': calculate_ber(original_watermark, extracted_watermark)
    }
    return results


class MetricsTracker:
    """指标追踪器"""
    
    def __init__(self):
        self.metrics = {
            'psnr': [],
            'ssim': [],
            'nc': [],
            'ber': [],
            'loss': []
        }
    
    def update(self, **kwargs):
        """更新指标"""
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def get_average(self):
        """获取平均值"""
        return {
            key: np.mean(values) if values else 0.0
            for key, values in self.metrics.items()
        }
    
    def reset(self):
        """重置指标"""
        for key in self.metrics:
            self.metrics[key] = []
    
    def __str__(self):
        avg = self.get_average()
        return f"PSNR: {avg['psnr']:.2f} | SSIM: {avg['ssim']:.4f} | NC: {avg['nc']:.4f} | BER: {avg['ber']:.4f}"


def test_metrics():
    """测试评估指标"""
    from PIL import Image
    import os
    
    # 使用测试图片
    test_img_path = 'img/img2.jpg'
    test_wm_path = 'img/watermark.png'
    
    # 如果文件不存在，使用其他图片
    if not os.path.exists(test_img_path):
        test_img_path = 'img/target.png'
    if not os.path.exists(test_wm_path):
        test_wm_path = 'img/watermark.png'
    
    # 加载测试数据
    img1 = Image.open(test_img_path).convert("RGB")
    img1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float() / 255.0
    
    # 创建带噪声的版本
    img2 = img1 + torch.randn_like(img1) * 0.05
    img2 = torch.clamp(img2, 0, 1)
    
    # 加载水印
    wm1 = Image.open(test_wm_path).convert("RGB")
    wm1 = torch.from_numpy(np.array(wm1)).permute(2, 0, 1).float() / 255.0
    
    # 调整水印尺寸
    if wm1.shape != img1.shape:
        wm1 = F.interpolate(wm1.unsqueeze(0), size=img1.shape[1:], mode='bilinear', align_corners=False).squeeze(0)
    
    # 创建带噪声的水印版本
    wm2 = wm1 + torch.randn_like(wm1) * 0.1
    wm2 = torch.clamp(wm2, 0, 1)
    
    print("测试评估指标:")
    print(f"PSNR: {calculate_psnr(img1, img2):.2f} dB")
    print(f"SSIM: {calculate_ssim(img1, img2):.4f}")
    print(f"NC: {calculate_nc(wm1, wm2):.4f}")
    print(f"BER: {calculate_ber(wm1, wm2):.4f}")
    
    # 测试评估函数
    results = evaluate_watermark_system(img1, img2, wm1, wm2)
    print(f"\n完整评估结果:")
    for key, value in results.items():
        print(f"  {key.upper()}: {value:.4f}")


if __name__ == "__main__":
    test_metrics()