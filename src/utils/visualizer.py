import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils


def tensor_to_image(tensor):
    """将tensor转换为numpy图像"""
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # 从 (C, H, W) 转为 (H, W, C)
    image = tensor.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    
    # 缩放到 [0, 255]
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    return image


def save_comparison(
    original_image,
    watermarked_image,
    original_watermark,
    extracted_watermark,
    save_path,
    metrics=None
):
    """
    保存对比图
    
    输入:
        original_image: 原始载体图像 (B, C, H, W) 或 (C, H, W)
        watermarked_image: 含水印图像
        original_watermark: 原始水印
        extracted_watermark: 提取的水印
        save_path: 保存路径
        metrics: 评估指标字典
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 原始图像
    axes[0, 0].imshow(tensor_to_image(original_image))
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 含水印图像
    axes[0, 1].imshow(tensor_to_image(watermarked_image))
    title = 'Watermarked Image'
    if metrics:
        title += f"\nPSNR: {metrics.get('psnr', 0):.2f} dB, SSIM: {metrics.get('ssim', 0):.4f}"
    axes[0, 1].set_title(title)
    axes[0, 1].axis('off')
    
    # 原始水印
    axes[1, 0].imshow(tensor_to_image(original_watermark))
    axes[1, 0].set_title('Original Watermark')
    axes[1, 0].axis('off')
    
    # 提取的水印
    axes[1, 1].imshow(tensor_to_image(extracted_watermark))
    title = 'Extracted Watermark'
    if metrics:
        title += f"\nNC: {metrics.get('nc', 0):.4f}, BER: {metrics.get('ber', 0):.4f}"
    axes[1, 1].set_title(title)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比图已保存到: {save_path}")


def save_grid(images, save_path, nrow=4, normalize=True):
    """
    保存图像网格
    
    输入:
        images: 图像tensor列表或tensor (N, C, H, W)
        save_path: 保存路径
        nrow: 每行图像数
        normalize: 是否归一化
    """
    if isinstance(images, list):
        images = torch.stack(images)
    
    grid = vutils.make_grid(images, nrow=nrow, normalize=normalize, padding=2)
    vutils.save_image(grid, save_path)
    
    print(f"图像网格已保存到: {save_path}")


def plot_training_curves(metrics_history, save_path):
    """
    绘制训练曲线
    
    输入:
        metrics_history: 包含训练历史的字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    if 'loss' in metrics_history:
        axes[0, 0].plot(metrics_history['loss'])
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
    
    # PSNR曲线
    if 'psnr' in metrics_history:
        axes[0, 1].plot(metrics_history['psnr'])
        axes[0, 1].set_title('PSNR Curve')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].grid(True)
    
    # SSIM曲线
    if 'ssim' in metrics_history:
        axes[1, 0].plot(metrics_history['ssim'])
        axes[1, 0].set_title('SSIM Curve')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].grid(True)
    
    # NC曲线
    if 'nc' in metrics_history:
        axes[1, 1].plot(metrics_history['nc'])
        axes[1, 1].set_title('NC Curve')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('NC')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到: {save_path}")


def visualize_dwt_subbands(image, dwt_transform, save_path=None):
    """
    可视化DWT子带
    
    输入:
        image: 输入图像 (C, H, W) 或 (B, C, H, W)
        dwt_transform: DWT变换模块
        save_path: 保存路径（可选）
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    # DWT分解
    ll, lh, hl, hh = dwt_transform.decompose(image)
    
    # 转换为numpy
    def to_numpy(x):
        x = x[0].cpu().numpy()  # 取第一个batch
        if x.shape[0] == 3:  # RGB
            x = np.transpose(x, (1, 2, 0))
        else:  # 单通道
            x = x[0]
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(to_numpy(ll))
    axes[0, 0].set_title('LL (Approximation)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(to_numpy(lh), cmap='gray')
    axes[0, 1].set_title('LH (Horizontal Detail)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(to_numpy(hl), cmap='gray')
    axes[1, 0].set_title('HL (Vertical Detail)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(to_numpy(hh), cmap='gray')
    axes[1, 1].set_title('HH (Diagonal Detail)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"DWT子带可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_attack_comparison(image, attack_simulator, save_path):
    """
    创建攻击效果对比图
    
    输入:
        image: 输入图像
        attack_simulator: 攻击模拟器
        save_path: 保存路径
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    attack_simulator.eval()
    
    # 应用各种攻击
    attacks = {
        'Original': image,
        'Blur': attack_simulator.gaussian_blur(image),
        'Noise': attack_simulator.gaussian_noise(image),
        'JPEG': attack_simulator.jpeg_compression(image),
        'Crop': attack_simulator.random_crop_resize(image),
        'Rotate': attack_simulator.random_rotate(image),
        'Color': attack_simulator.color_jitter(image),
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (name, attacked) in enumerate(attacks.items()):
        axes[idx].imshow(tensor_to_image(attacked))
        axes[idx].set_title(name)
        axes[idx].axis('off')
    
    # 隐藏多余的子图
    for idx in range(len(attacks), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"攻击对比图已保存到: {save_path}")


def save_watermark_robustness_test(
    original_image,
    watermarked_image,
    original_watermark,
    model,
    attack_simulator,
    save_path
):
    """
    保存水印鲁棒性测试结果
    
    输入:
        original_image: 原始图像
        watermarked_image: 含水印图像
        original_watermark: 原始水印
        model: 水印模型
        attack_simulator: 攻击模拟器
        save_path: 保存路径
    """
    from ..utils.metrics import calculate_psnr, calculate_nc
    
    if original_image.dim() == 3:
        original_image = original_image.unsqueeze(0)
    if watermarked_image.dim() == 3:
        watermarked_image = watermarked_image.unsqueeze(0)
    if original_watermark.dim() == 3:
        original_watermark = original_watermark.unsqueeze(0)
    
    attack_simulator.eval()
    model.eval()
    
    # 定义攻击
    attacks = [
        ('No Attack', lambda x: x),
        ('Blur', attack_simulator.gaussian_blur),
        ('Noise', attack_simulator.gaussian_noise),
        ('JPEG', attack_simulator.jpeg_compression),
        ('Crop', attack_simulator.random_crop_resize),
        ('Rotate', attack_simulator.random_rotate),
    ]
    
    fig, axes = plt.subplots(len(attacks), 3, figsize=(12, 4 * len(attacks)))
    
    for idx, (name, attack_fn) in enumerate(attacks):
        # 应用攻击
        attacked = attack_fn(watermarked_image)
        
        # 提取水印
        with torch.no_grad():
            extracted, _ = model.decode(attacked)
        
        # 计算指标
        psnr = calculate_psnr(original_image, attacked)
        nc = calculate_nc(original_watermark, extracted)
        
        # 显示攻击后的图像
        axes[idx, 0].imshow(tensor_to_image(attacked))
        axes[idx, 0].set_title(f'{name}\nPSNR: {psnr:.2f} dB')
        axes[idx, 0].axis('off')
        
        # 显示原始水印
        axes[idx, 1].imshow(tensor_to_image(original_watermark))
        axes[idx, 1].set_title('Original Watermark')
        axes[idx, 1].axis('off')
        
        # 显示提取的水印
        axes[idx, 2].imshow(tensor_to_image(extracted))
        axes[idx, 2].set_title(f'Extracted\nNC: {nc:.4f}')
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"鲁棒性测试图已保存到: {save_path}")


if __name__ == '__main__':
    # 测试可视化功能
    print("测试可视化工具...")
    
    # 创建测试数据
    original = torch.rand(3, 64, 64)
    watermarked = torch.rand(3, 64, 64)
    wm_original = torch.rand(3, 64, 64)
    wm_extracted = torch.rand(3, 64, 64)
    
    metrics = {
        'psnr': 35.5,
        'ssim': 0.95,
        'nc': 0.92,
        'ber': 0.05
    }
    
    # 测试对比图
    os.makedirs('test_outputs', exist_ok=True)
    save_comparison(
        original, watermarked, wm_original, wm_extracted,
        'test_outputs/comparison.png',
        metrics
    )
    
    print("测试完成！")