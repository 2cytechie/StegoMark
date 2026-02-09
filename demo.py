#!/usr/bin/env python3
"""
StegoMark 盲水印系统演示脚本

这个脚本演示了如何使用 StegoMark 进行水印嵌入和提取。
"""

import os
import sys
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import WatermarkNet
from src.extract import WatermarkExtractor
from src.utils.visualizer import save_comparison
from src.utils.metrics import evaluate_watermark_system, calculate_psnr, calculate_nc
from src.config import config
from src.data import WatermarkDataset, get_val_transforms


def demo_embed_extract(dataset, checkpoint_path=None):
    """演示水印嵌入和提取"""
    print("=" * 60)
    print("StegoMark 盲水印系统演示")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 创建或加载模型
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载模型: {checkpoint_path}")
        extractor = WatermarkExtractor(checkpoint_path, device=str(device))
        model = extractor.model
    else:
        print(f"警告: 未找到模型文件 {checkpoint_path}，创建新模型（未训练）")
        model = WatermarkNet(
            image_channels=3,
            watermark_channels=3,
            hidden_dim=config.hidden_dim,
            block_size=config.watermark_size,
            overlap=config.overlap
        ).to(device)
    
    model.eval()
    
    # 从 Dataset 获取一个样本数据
    idx = random.randint(0, len(dataset) - 1)
    images, watermarks = dataset[idx]
    
    image_tensor = images.unsqueeze(0).to(device)
    watermark_tensor = watermarks.unsqueeze(0).to(device)
    
    print(f"\n图像尺寸: {image_tensor.shape}")
    print(f"水印尺寸: {watermark_tensor.shape}")
    
    # 嵌入水印 - 编码器内部进行分块处理
    print("\n[1] 嵌入水印...")
    print(f"    - 将图像分块 {config.watermark_size}x{config.watermark_size} 处理")
    print("    - 在每个块中进行DWT分解和水印嵌入")
    print("    - 通过平滑融合技术拼回原图")
    with torch.no_grad():
        watermarked = model.encode(image_tensor, watermark_tensor)
    
    # 提取水印 - 解码器内部进行分块处理
    print("[2] 提取水印...")
    print(f"    - 将含水印图像分块 {config.watermark_size}x{config.watermark_size} 处理")
    print("    - 从每个块中提取水印")
    print("    - 选择置信度最高的水印")
    with torch.no_grad():
        extracted_wm, confidence = model.decode(watermarked)
    
    print(f"    - 提取水印尺寸: {extracted_wm.shape}")
    print(f"    - 置信度: {confidence.item():.4f}")
    
    # 调整原始水印尺寸以匹配提取的水印尺寸（用于计算NC和BER）
    import torch.nn.functional as F
    watermark_tensor_resized = F.interpolate(
        watermark_tensor, 
        size=(extracted_wm.shape[2], extracted_wm.shape[3]),
        mode='bilinear',
        align_corners=False
    )
    
    # 计算指标
    print("[3] 计算评估指标...")
    metrics = evaluate_watermark_system(
        image_tensor, watermarked, watermark_tensor_resized, extracted_wm
    )
    
    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    print(f"  PSNR (图像质量): {metrics['psnr']:.2f} dB")
    print(f"  SSIM (结构相似): {metrics['ssim']:.4f}")
    print(f"  NC   (水印相关): {metrics['nc']:.4f}")
    print(f"  BER  (误码率):   {metrics['ber']:.4f}")
    print(f"  置信度:          {confidence.item():.4f}")
    print("=" * 60)
    
    # 保存结果到 outputs 文件夹
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存对比图
    comparison_path = os.path.join(output_dir, 'comparison.png')
    save_comparison(
        original_image=image_tensor, watermarked_image=watermarked,
        original_watermark=watermark_tensor, extracted_watermark=extracted_wm,
        save_path=comparison_path, metrics=metrics
    )
    
    # 保存单独的图片
    def save_tensor(tensor, path):
        img = tensor.squeeze(0).cpu()
        img = transforms.ToPILImage()(img)
        img.save(path)
    
    save_tensor(watermarked, os.path.join(output_dir, 'watermarked.png'))
    save_tensor(extracted_wm, os.path.join(output_dir, 'extracted_watermark.png'))
    
    print(f"\n结果已保存到 {output_dir}/ 目录")
    print(f"  - comparison.png: 对比图")
    print(f"  - watermarked.png: 含水印图像")
    print(f"  - extracted_watermark.png: 提取的水印")
    
    return metrics


def demo_robustness(data_loader, checkpoint_path=None):
    """演示鲁棒性测试"""
    print("\n" + "=" * 60)
    print("鲁棒性测试演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建或加载模型
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"加载模型: {checkpoint_path}")
        extractor = WatermarkExtractor(checkpoint_path, device=str(device))
        model = extractor.model
    else:
        print(f"警告: 未找到模型文件 {checkpoint_path}，创建新模型（未训练）")
        model = WatermarkNet(
            image_channels=3,
            watermark_channels=3,
            hidden_dim=config.hidden_dim,
            block_size=config.watermark_size,
            overlap=config.overlap
        ).to(device)
    
    model.eval()
    
    # 从 DataLoader 获取一个批次的数据
    images, watermarks = next(iter(data_loader))
    
    # 只取批次中的第一张图片进行演示
    image_tensor = images[0:1].to(device)
    watermark_tensor = watermarks[0:1].to(device)
    
    print(f"\n图像尺寸: {image_tensor.shape}")
    print(f"水印尺寸: {watermark_tensor.shape}")
    
    # 嵌入水印
    print("\n嵌入水印...")
    with torch.no_grad():
        watermarked = model.encode(image_tensor, watermark_tensor)
    
    # 测试各种攻击
    from src.models import AttackSimulator
    attack_sim = AttackSimulator(prob=1.0).to(device)
    attack_sim.eval()
    
    attacks = {
        '无攻击': lambda x: x,
        '高斯模糊': attack_sim.gaussian_blur,
        '高斯噪声': attack_sim.gaussian_noise,
        'JPEG压缩': attack_sim.jpeg_compression,
        '随机裁剪': attack_sim.random_crop_resize,
        '旋转': attack_sim.random_rotate,
    }
    
    print("\n攻击类型          PSNR(dB)    NC")
    print("-" * 40)
    
    # 先提取一次水印以获取尺寸
    with torch.no_grad():
        sample_extracted, _ = model.decode(watermarked)
    
    # 调整原始水印尺寸以匹配提取的水印尺寸
    import torch.nn.functional as F
    watermark_tensor_resized = F.interpolate(
        watermark_tensor,
        size=(sample_extracted.shape[2], sample_extracted.shape[3]),
        mode='bilinear',
        align_corners=False
    )
    
    # 准备保存攻击结果对比图
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for name, attack_fn in attacks.items():
        # 应用攻击
        attacked = attack_fn(watermarked)
        
        # 提取水印
        with torch.no_grad():
            extracted, _ = model.decode(attacked)
        
        # 计算指标
        psnr = calculate_psnr(image_tensor, attacked)
        nc = calculate_nc(watermark_tensor_resized, extracted)
        
        # 计算BER
        from src.utils.metrics import calculate_ber
        ber = calculate_ber(watermark_tensor_resized, extracted)
        
        print(f"{name:<12} {psnr:>8.2f}  {nc:>8.4f}")
        results.append((name, psnr, nc))
        
        # 保存该攻击的对比图
        attack_metrics = {
            'psnr': psnr,
            'ssim': 0,  # 暂不计算SSIM
            'nc': nc,
            'ber': ber
        }
        
        # 将中文攻击名称转换为英文文件名
        attack_name_map = {
            '无攻击': 'no_attack',
            '高斯模糊': 'gaussian_blur',
            '高斯噪声': 'gaussian_noise',
            'JPEG压缩': 'jpeg_compression',
            '随机裁剪': 'random_crop',
            '旋转': 'rotation'
        }
        filename = attack_name_map.get(name, name)
        comparison_path = os.path.join(output_dir, f'robustness_{filename}.png')
        
        save_comparison(
            image_tensor, attacked,
            watermark_tensor_resized, extracted,
            comparison_path, attack_metrics
        )
    
    print("-" * 40)
    print(f"\n攻击对比图已保存到 {output_dir}/ 目录:")
    for name in attacks.keys():
        filename = attack_name_map.get(name, name)
        print(f"  - robustness_{filename}.png: {name}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='StegoMark 演示')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/epoch_50.pth', help='模型检查点路径')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['embed', 'robustness', 'all'],
                        help='演示模式: embed=嵌入提取, robustness=鲁棒性测试, all=全部')
    args = parser.parse_args()
    
    test_dataset = WatermarkDataset(
        image_dir=config.test_image_dir,
        watermark_dir=config.test_watermark_dir,
        image_size=config.image_size,
        watermark_size=config.watermark_size
    )

    if args.mode in ['embed', 'all']:
        demo_embed_extract(test_dataset, args.checkpoint)
    
    if args.mode in ['robustness', 'all']:
        demo_robustness(test_dataset, args.checkpoint)
    
    print("\n演示完成！")


if __name__ == '__main__':
    main()
