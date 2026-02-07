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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import WatermarkNet
from src.extract import WatermarkExtractor
from src.utils.visualizer import save_comparison
from src.utils.metrics import evaluate_watermark_system
from src.config import config


def demo_embed_extract(target_img, watermark_img, checkpoint_path=None):
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
    

    
    print(f"\n载体图像: {target_img}")
    print(f"水印图像: {watermark_img}")
    
    # 预处理
    from src.data.transforms import ResizeAndTile
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        ResizeAndTile(64),
    ])
    
    # 加载图像
    image = Image.open(target_img).convert('RGB')
    watermark = Image.open(watermark_img).convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    watermark_tensor = transform(watermark).unsqueeze(0).to(device)
    
    print(f"\n图像尺寸: {image_tensor.shape}")
    print(f"水印尺寸: {watermark_tensor.shape}")
    
    # 嵌入水印
    print("\n[1] 嵌入水印...")
    with torch.no_grad():
        watermarked = model.encode(image_tensor, watermark_tensor)
    
    # 提取水印
    print("[2] 提取水印...")
    with torch.no_grad():
        extracted_wm, confidence = model.decode(watermarked)
    
    # 计算指标
    print("[3] 计算评估指标...")
    metrics = evaluate_watermark_system(
        image_tensor, watermarked, watermark_tensor, extracted_wm
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
        image_tensor, watermarked,
        watermark_tensor, extracted_wm,
        comparison_path, metrics
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


def demo_robustness(target_img, watermark_img, checkpoint_path=None):
    """演示鲁棒性测试"""
    print("\n" + "=" * 60)
    print("鲁棒性测试演示")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建或加载模型
    if checkpoint_path and os.path.exists(checkpoint_path):
        extractor = WatermarkExtractor(checkpoint_path, device=str(device))
        model = extractor.model
    else:
        print(f"警告: 未找到模型文件 {checkpoint_path}")
    
    # 创建测试数据
    image = Image.open(target_img).convert('RGB')
    watermark = Image.open(watermark_img).convert('RGB')
    
    # 预处理
    from src.data.transforms import ResizeAndTile
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        ResizeAndTile(64),
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    watermark_tensor = transform(watermark).unsqueeze(0).to(device)
    
    # 嵌入水印
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
    
    results = []
    for name, attack_fn in attacks.items():
        # 应用攻击
        attacked = attack_fn(watermarked)
        
        # 提取水印
        with torch.no_grad():
            extracted, _ = model.decode(attacked)
        
        # 计算指标
        from src.utils.metrics import calculate_psnr, calculate_nc
        psnr = calculate_psnr(image, attacked)
        nc = calculate_nc(watermark, extracted)
        
        print(f"{name:<12} {psnr:>8.2f}  {nc:>8.4f}")
        results.append((name, psnr, nc))
    
    print("-" * 40)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='StegoMark 演示')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth', help='模型检查点路径')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['embed', 'robustness', 'all'],
                        help='演示模式')
    parser.add_argument('--target_img', type=str, default='img/target.png', help='目标图像路径')
    parser.add_argument('--watermark_img', type=str, default='img/watermark.png', help='水印图像路径')
    
    args = parser.parse_args()
    
    if args.mode in ['embed', 'all']:
        demo_embed_extract(args.target_img, args.watermark_img, args.checkpoint)
    
    if args.mode in ['robustness', 'all']:
        demo_robustness(args.target_img, args.watermark_img, args.checkpoint)
    
    print("\n演示完成！")


if __name__ == '__main__':
    main()