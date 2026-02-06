"""
评估脚本
"""
import os
import sys
# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json

from configs.config import data_config, model_config, training_config
from src.models import create_model
from src.data import WatermarkDataset, collate_fn
from src.losses import WatermarkLoss
from src.utils import MetricsTracker, calculate_psnr, calculate_ssim, calculate_watermark_accuracy
from src.utils.attacks import CombinedAttack


def evaluate_model(model, dataloader, device, apply_attacks=False):
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        apply_attacks: 是否应用攻击
        
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    
    # 各种攻击的评估器
    attack_types = ['none', 'crop', 'rotate', 'blur', 'noise', 'jpeg', 'combined']
    metrics_trackers = {attack: MetricsTracker() for attack in attack_types}
    
    # 创建攻击器
    if apply_attacks:
        attack_modules = {
            'crop': CombinedAttack(crop_prob=1.0, rotate_prob=0, blur_prob=0, jpeg_prob=0, noise_prob=0),
            'rotate': CombinedAttack(crop_prob=0, rotate_prob=1.0, blur_prob=0, jpeg_prob=0, noise_prob=0),
            'blur': CombinedAttack(crop_prob=0, rotate_prob=0, blur_prob=1.0, jpeg_prob=0, noise_prob=0),
            'noise': CombinedAttack(crop_prob=0, rotate_prob=0, blur_prob=0, jpeg_prob=0, noise_prob=1.0),
            'jpeg': CombinedAttack(crop_prob=0, rotate_prob=0, blur_prob=0, jpeg_prob=1.0, noise_prob=0),
            'combined': CombinedAttack(crop_prob=0.5, rotate_prob=0.3, blur_prob=0.3, jpeg_prob=0.3, noise_prob=0.3)
        }
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        for batch in pbar:
            images = batch['image'].to(device)
            watermarks = batch['watermark'].to(device)
            watermarks_tiled = batch['watermark_tiled'].to(device)
            
            # 嵌入水印
            watermarked_images, extracted_watermarks = model(images, watermarks_tiled)
            
            # 无攻击评估
            metrics_trackers['none'].update(
                images, watermarked_images,
                watermarks, extracted_watermarks
            )
            
            # 应用各种攻击并评估
            if apply_attacks:
                for attack_name, attack_module in attack_modules.items():
                    attacked_images = attack_module(watermarked_images)
                    extracted_attacked, _ = model.decoder(attacked_images, None)
                    
                    metrics_trackers[attack_name].update(
                        images, attacked_images,
                        watermarks, extracted_attacked
                    )
    
    # 收集结果
    results = {}
    for attack_name, tracker in metrics_trackers.items():
        results[attack_name] = tracker.get_average()
    
    return results


def print_results(results):
    """打印评估结果"""
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    for attack_name, metrics in results.items():
        print(f"\n攻击类型: {attack_name.upper()}")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        if 'watermark_accuracy' in metrics:
            print(f"  水印准确率: {metrics['watermark_accuracy']:.4f}")
        if 'bit_accuracy' in metrics:
            print(f"  比特准确率: {metrics['bit_accuracy']:.4f}")
    
    print("="*60)


def main(args):
    """主评估函数"""
    # 设置设备
    device = torch.device(training_config.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(model_config)
    
    # 加载权重
    if args.checkpoint:
        print(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 加载最佳模型
        best_model_path = os.path.join(training_config.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"加载最佳模型: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("警告: 未找到模型权重，使用随机初始化的模型")
    
    model = model.to(device)
    
    # 创建数据集
    val_dataset = WatermarkDataset(
        images_dir=data_config.val_images_dir,
        watermarks_dir=data_config.val_watermarks_dir,
        watermark_size=data_config.watermark_size,
        flip_prob=0,
        blur_prob=0,
        color_jitter_prob=0,
        crop_prob=0,
        train=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"验证集大小: {len(val_dataset)}")
    
    # 评估
    results = evaluate_model(model, val_loader, device, apply_attacks=args.apply_attacks)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")
    
    # 检查是否满足评估指标
    print("\n评估指标检查:")
    none_results = results['none']
    
    checks = [
        ('PSNR > 30dB', none_results['psnr'] > 30),
        ('SSIM > 0.9', none_results['ssim'] > 0.9),
    ]
    
    if 'watermark_accuracy' in none_results:
        checks.append(('水印准确率 > 0.9', none_results['watermark_accuracy'] > 0.9))
    
    for check_name, passed in checks:
        status = "✓ 通过" if passed else "✗ 未通过"
        print(f"  {check_name}: {status}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估水印模型')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--apply-attacks', action='store_true', help='应用攻击评估鲁棒性')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='输出结果文件')
    args = parser.parse_args()
    
    main(args)
