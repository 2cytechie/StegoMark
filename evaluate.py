"""
评估脚本
用于评估水印系统的性能
包含PSNR、SSIM、水印提取准确率等指标
以及各种攻击下的鲁棒性测试
"""

import torch
import argparse
import os
from PIL import Image
import json
from tqdm import tqdm
import numpy as np

from config import train_config, eval_config, image_config, dwt_config
from models import create_model
from watermark_utils import (
    preprocess_watermark, preprocess_watermark_to_64x64,
    pil_to_tensor, tensor_to_pil, ColorWatermarkAccuracyCalculator
)
from utils import (
    load_checkpoint, get_device, calculate_psnr, calculate_ssim,
    save_image, visualize_results
)
from attacks import AttackSimulator


def evaluate_single_image(model, image_path, watermark_path, device, alpha=0.1):
    """
    评估单张图像
    
    Returns:
        评估指标字典
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize(
        (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
        Image.LANCZOS
    )
    image_tensor = pil_to_tensor(image_resized).unsqueeze(0).to(device)
    
    # 加载水印
    watermark_tensor = preprocess_watermark(
        watermark_path,
        watermark_size=image_config.WATERMARK_SIZE,
        target_size=image_config.TARGET_SIZE
    ).unsqueeze(0).to(device)
    
    watermark_64 = preprocess_watermark_to_64x64(watermark_path).to(device)
    
    # 嵌入水印
    with torch.no_grad():
        watermarked_tensor = model.embed(image_tensor, watermark_tensor, alpha=alpha)
        extracted_tensor = model.extract(watermarked_tensor)
    
    # 计算指标 - 使用RGB彩色准确率作为主要准确率指标
    psnr = calculate_psnr(image_tensor.squeeze(0), watermarked_tensor.squeeze(0))
    ssim = calculate_ssim(image_tensor.squeeze(0), watermarked_tensor.squeeze(0))

    # 使用RGB彩色准确率计算器
    color_calculator = ColorWatermarkAccuracyCalculator(tolerance=0.1, threshold=0.5)
    rgb_metrics = color_calculator.calculate_tolerance_accuracy(extracted_tensor.squeeze(0), watermark_64)

    return {
        'psnr': psnr,
        'ssim': ssim,
        'watermark_acc': rgb_metrics['overall'],  # RGB综合准确率作为主要指标
        'color_acc_r': rgb_metrics['R_channel'],
        'color_acc_g': rgb_metrics['G_channel'],
        'color_acc_b': rgb_metrics['B_channel'],
        'color_acc_overall': rgb_metrics['overall'],
        'color_correct_pixels': rgb_metrics['total_correct_pixels'],
        'color_total_pixels': rgb_metrics['total_pixels']
    }


def evaluate_robustness(model, image_path, watermark_path, device, alpha=0.1):
    """
    评估鲁棒性（各种攻击下的性能）
    
    Returns:
        各种攻击下的评估结果
    """
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize(
        (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
        Image.LANCZOS
    )
    image_tensor = pil_to_tensor(image_resized).unsqueeze(0).to(device)
    
    # 加载水印
    watermark_tensor = preprocess_watermark(
        watermark_path,
        watermark_size=image_config.WATERMARK_SIZE,
        target_size=image_config.TARGET_SIZE
    ).unsqueeze(0).to(device)
    
    watermark_64 = preprocess_watermark_to_64x64(watermark_path).to(device)
    
    # 嵌入水印
    with torch.no_grad():
        watermarked_tensor = model.embed(image_tensor, watermark_tensor, alpha=alpha)
    
    # 基础指标
    results = {
        'none': {
            'psnr': calculate_psnr(image_tensor.squeeze(0), watermarked_tensor.squeeze(0)),
            'ssim': calculate_ssim(image_tensor.squeeze(0), watermarked_tensor.squeeze(0))
        }
    }
    
    # 攻击模拟器
    attack_simulator = AttackSimulator()
    
    # 各种攻击
    attacks = [
        ('crop', {}),
        ('rotate', {}),
        ('scale', {}),
        ('blur', {}),
        ('noise', {}),
        ('jpeg', {}),
        ('combined', {})
    ]
    
    for attack_name, attack_params in attacks:
        # 应用攻击
        attacked_tensor = attack_simulator.apply_attack_by_name(
            watermarked_tensor, attack_name, **attack_params
        )

        # 提取水印
        with torch.no_grad():
            extracted = model.extract(attacked_tensor)

        # 计算指标 - 使用RGB彩色准确率
        psnr = calculate_psnr(image_tensor.squeeze(0), attacked_tensor.squeeze(0))
        rgb_metrics = color_calculator.calculate_tolerance_accuracy(extracted.squeeze(0), watermark_64)

        results[attack_name] = {
            'psnr': psnr,
            'watermark_acc': rgb_metrics['overall'],  # RGB综合准确率
            'color_acc_r': rgb_metrics['R_channel'],
            'color_acc_g': rgb_metrics['G_channel'],
            'color_acc_b': rgb_metrics['B_channel'],
            'color_acc_overall': rgb_metrics['overall']
        }

    return results


def evaluate_dataset(model, image_dir, watermark_dir, device, alpha=0.1):
    """
    评估整个数据集
    
    Returns:
        平均指标
    """
    # 获取图像文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = []
    if os.path.exists(image_dir):
        image_files = [f for f in os.listdir(image_dir)
                      if f.lower().endswith(valid_extensions)]
    
    # 获取水印文件
    watermark_files = []
    if os.path.exists(watermark_dir):
        watermark_files = [f for f in os.listdir(watermark_dir)
                          if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"警告: 在 {image_dir} 中没有找到图像")
        return None
    
    if not watermark_files:
        print(f"警告: 在 {watermark_dir} 中没有找到水印")
        return None
    
    print(f"\n评估数据集: {len(image_files)} 张图像, {len(watermark_files)} 个水印")
    
    all_metrics = []
    all_robustness = {attack: {'psnr': [], 'watermark_acc': []} 
                     for attack in ['none', 'crop', 'rotate', 'scale', 
                                   'blur', 'noise', 'jpeg', 'combined']}
    
    pbar = tqdm(image_files, desc='评估中')
    for image_file in pbar:
        image_path = os.path.join(image_dir, image_file)
        
        # 随机选择一个水印
        import random
        watermark_file = random.choice(watermark_files)
        watermark_path = os.path.join(watermark_dir, watermark_file)
        
        try:
            # 基础评估
            metrics = evaluate_single_image(model, image_path, watermark_path, device, alpha)
            all_metrics.append(metrics)
            
            # 鲁棒性评估
            robustness = evaluate_robustness(model, image_path, watermark_path, device, alpha)
            for attack, values in robustness.items():
                if 'psnr' in values:
                    all_robustness[attack]['psnr'].append(values['psnr'])
                if 'watermark_acc' in values:
                    all_robustness[attack]['watermark_acc'].append(values['watermark_acc'])
            
            # 更新进度条 - 使用RGB彩色准确率
            pbar.set_postfix({
                'PSNR': f"{metrics['psnr']:.2f}",
                'Acc(RGB)': f"{metrics['watermark_acc']:.4f}"
            })
            
        except Exception as e:
            print(f"\n处理 {image_file} 时出错: {e}")
            continue
    
    # 计算平均值
    avg_metrics = {
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_metrics]),
        'watermark_acc': np.mean([m['watermark_acc'] for m in all_metrics])
    }
    
    # 鲁棒性平均值
    robustness_summary = {}
    for attack, values in all_robustness.items():
        robustness_summary[attack] = {
            'psnr': np.mean(values['psnr']) if values['psnr'] else 0,
            'watermark_acc': np.mean(values['watermark_acc']) if values['watermark_acc'] else 0
        }
    
    return {
        'basic': avg_metrics,
        'robustness': robustness_summary
    }


def print_evaluation_results(results):
    """打印评估结果"""
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    # 基础指标
    if 'basic' in results:
        print("\n【基础指标】")
        basic = results['basic']
        print(f"  PSNR: {basic['psnr']:.2f} dB")
        print(f"  SSIM: {basic['ssim']:.4f}")
        print(f"  RGB水印准确率: {basic['watermark_acc']:.4f} ({basic['watermark_acc']*100:.2f}%)")
        if 'color_acc_r' in basic:
            print(f"    R通道: {basic['color_acc_r']:.4f}")
            print(f"    G通道: {basic['color_acc_g']:.4f}")
            print(f"    B通道: {basic['color_acc_b']:.4f}")

        # 检查是否达到目标
        print("\n【目标达成情况】")
        targets = {
            'PSNR > 30dB': basic['psnr'] > 30,
            'SSIM > 0.9': basic['ssim'] > 0.9,
            'RGB准确率 > 0.9': basic['watermark_acc'] > 0.9
        }
        for target, achieved in targets.items():
            status = "✓ 达成" if achieved else "✗ 未达成"
            print(f"  {target}: {status}")
    
    # 鲁棒性指标
    if 'robustness' in results:
        print("\n【鲁棒性测试（各种攻击下的水印提取准确率）】")
        print("-" * 60)
        print(f"{'攻击类型':<15} {'PSNR (dB)':<15} {'准确率':<15}")
        print("-" * 60)
        
        for attack, values in results['robustness'].items():
            psnr = values.get('psnr', 0)
            acc = values.get('watermark_acc', 0)
            print(f"{attack:<15} {psnr:<15.2f} {acc:<15.4f} ({acc*100:.2f}%)")
    
    print("="*60)


def save_evaluation_report(results, output_path):
    """保存评估报告"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n评估报告已保存: {output_path}")


def visualize_evaluation(model, image_path, watermark_path, output_dir, device, alpha=0.1):
    """可视化评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize(
        (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
        Image.LANCZOS
    )
    image_tensor = pil_to_tensor(image_resized).unsqueeze(0).to(device)
    
    # 加载水印
    watermark_tensor = preprocess_watermark(
        watermark_path,
        watermark_size=image_config.WATERMARK_SIZE,
        target_size=image_config.TARGET_SIZE
    ).unsqueeze(0).to(device)
    
    watermark_64 = preprocess_watermark_to_64x64(watermark_path).to(device)
    
    # 嵌入和提取
    with torch.no_grad():
        watermarked_tensor = model.embed(image_tensor, watermark_tensor, alpha=alpha)
        extracted_tensor = model.extract(watermarked_tensor)
    
    # 保存可视化结果
    visualize_results(
        image_tensor.squeeze(0),
        watermark_64,
        watermarked_tensor.squeeze(0),
        extracted_tensor.squeeze(0),
        os.path.join(output_dir, 'evaluation_visualization.png')
    )
    
    # 保存各个图像
    save_image(image_tensor.squeeze(0), os.path.join(output_dir, 'original_image.png'))
    save_image(watermark_64, os.path.join(output_dir, 'original_watermark.png'))
    save_image(watermarked_tensor.squeeze(0), os.path.join(output_dir, 'watermarked_image.png'))
    save_image(extracted_tensor.squeeze(0), os.path.join(output_dir, 'extracted_watermark.png'))
    
    print(f"\n可视化结果已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='评估水印系统性能')
    parser.add_argument('--model', type=str,
                       default=os.path.join('checkpoints', 'best_model.pth'),
                       help='模型checkpoint路径')
    parser.add_argument('--image', type=str, default=None,
                       help='测试图像路径')
    parser.add_argument('--watermark', type=str, default=None,
                       help='水印路径')
    parser.add_argument('--image_dir', type=str, default='data/val/images',
                       help='测试图像目录')
    parser.add_argument('--watermark_dir', type=str, default='data/val/watermarks',
                       help='水印目录')
    parser.add_argument('--output', type=str, default='output/evaluation_report.json',
                       help='评估报告输出路径')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化结果')
    parser.add_argument('--visualize_dir', type=str, default='output/visualization',
                       help='可视化输出目录')
    parser.add_argument('--alpha', type=float, default=None,
                       help='嵌入强度')
    
    args = parser.parse_args()
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    from config import model_config
    model = create_model(model_config, device)
    
    if os.path.exists(args.model):
        epoch, metrics = load_checkpoint(args.model, model, device=device)
        print(f"模型加载完成 (epoch {epoch})")
    else:
        print(f"警告: 模型文件不存在，使用未训练的模型")
    
    model.eval()
    
    # 设置alpha
    alpha = args.alpha if args.alpha is not None else dwt_config.BASE_ALPHA
    
    # 评估模式
    if args.image and args.watermark:
        # 单张图像评估
        print(f"\n评估单张图像: {args.image}")
        
        # 基础评估
        metrics = evaluate_single_image(model, args.image, args.watermark, device, alpha)
        
        # 鲁棒性评估
        robustness = evaluate_robustness(model, args.image, args.watermark, device, alpha)
        
        results = {
            'basic': metrics,
            'robustness': robustness
        }
        
        # 可视化
        if args.visualize:
            visualize_evaluation(model, args.image, args.watermark, 
                               args.visualize_dir, device, alpha)
    
    elif os.path.exists(args.image_dir) and os.path.exists(args.watermark_dir):
        # 数据集评估
        results = evaluate_dataset(model, args.image_dir, args.watermark_dir, device, alpha)
        
        if results is None:
            print("评估失败，请检查数据路径")
            return
    else:
        print("错误: 请提供图像路径或有效的图像目录")
        return
    
    # 打印结果
    print_evaluation_results(results)
    
    # 保存报告
    save_evaluation_report(results, args.output)


if __name__ == '__main__':
    main()
