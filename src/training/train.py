"""
训练脚本
"""
import os
import sys
# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path

from configs.config import data_config, model_config, attack_config, training_config
from src.models import create_model
from src.data import WatermarkDataset, collate_fn
from src.losses import WatermarkLoss
from src.utils import MetricsTracker
from src.utils.attacks import simulate_attacks_during_training


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer=None):
    """训练一个epoch"""
    model.train()
    metrics_tracker = MetricsTracker()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        # 获取数据
        images = batch['image'].to(device)
        watermarks = batch['watermark'].to(device)
        watermarks_tiled = batch['watermark_tiled'].to(device)
        
        # 前向传播
        watermarked_images, extracted_watermarks = model(images, watermarks_tiled)
        
        # 模拟攻击
        attacked_images = simulate_attacks_during_training(watermarked_images, attack_config)
        
        # 从攻击后的图像提取水印
        extracted_watermarks_attacked, _ = model.decoder(attacked_images, None)
        
        # 计算损失
        loss, loss_dict = criterion(
            images, watermarked_images,
            watermarks, extracted_watermarks_attacked
        )
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新指标
        with torch.no_grad():
            metrics_tracker.update(images, watermarked_images, watermarks, extracted_watermarks_attacked)
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'psnr': f"{metrics_tracker.get_average()['psnr']:.2f}",
            'wm_acc': f"{metrics_tracker.get_average().get('watermark_accuracy', 0):.4f}"
        })
        
        # 记录到tensorboard
        if writer is not None and batch_idx % 10 == 0:
            step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss_dict['total'], step)
            writer.add_scalar('train/image_mse', loss_dict['image_mse'], step)
            writer.add_scalar('train/watermark_mse', loss_dict['watermark_mse'], step)
    
    return metrics_tracker.get_average()


def validate(model, dataloader, criterion, device, epoch, writer=None):
    """验证"""
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            images = batch['image'].to(device)
            watermarks = batch['watermark'].to(device)
            watermarks_tiled = batch['watermark_tiled'].to(device)
            
            # 前向传播
            watermarked_images, extracted_watermarks = model(images, watermarks_tiled)
            
            # 计算损失
            loss, loss_dict = criterion(
                images, watermarked_images,
                watermarks, extracted_watermarks
            )
            
            # 更新指标
            metrics_tracker.update(images, watermarked_images, watermarks, extracted_watermarks)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'psnr': f"{metrics_tracker.get_average()['psnr']:.2f}"
            })
            
            # 记录图像到tensorboard
            if writer is not None and batch_idx == 0:
                # 反归一化到[0, 1]
                img_show = (images[:4] + 1) / 2
                wm_show = (watermarked_images[:4] + 1) / 2
                orig_wm_show = (watermarks[:4] + 1) / 2
                extr_wm_show = (extracted_watermarks[:4] + 1) / 2
                
                writer.add_images('val/original', img_show, epoch)
                writer.add_images('val/watermarked', wm_show, epoch)
                writer.add_images('val/original_watermark', orig_wm_show, epoch)
                writer.add_images('val/extracted_watermark', extr_wm_show, epoch)
    
    metrics = metrics_tracker.get_average()
    
    # 记录到tensorboard
    if writer is not None:
        writer.add_scalar('val/psnr', metrics['psnr'], epoch)
        writer.add_scalar('val/ssim', metrics['ssim'], epoch)
        if 'watermark_accuracy' in metrics:
            writer.add_scalar('val/watermark_accuracy', metrics['watermark_accuracy'], epoch)
    
    return metrics


def main(args):
    """主训练函数"""
    # 设置设备
    device = torch.device(training_config.device)
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(model_config)
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 创建数据集
    train_dataset = WatermarkDataset(
        images_dir=data_config.train_images_dir,
        watermarks_dir=data_config.train_watermarks_dir,
        watermark_size=data_config.watermark_size,
        flip_prob=data_config.flip_prob,
        blur_prob=data_config.blur_prob,
        color_jitter_prob=data_config.color_jitter_prob,
        crop_prob=data_config.crop_prob,
        train=True
    )
    
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
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=30,
        gamma=0.5
    )
    
    # 创建损失函数
    criterion = WatermarkLoss(
        lambda_image=training_config.lambda_image,
        lambda_watermark=training_config.lambda_watermark
    )
    
    # 创建tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(training_config.checkpoint_dir, 'logs'))
    
    # 创建检查点目录
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    
    # 训练循环
    best_psnr = 0
    best_wm_acc = 0
    
    for epoch in range(1, training_config.num_epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{training_config.num_epochs}")
        print(f"{'='*50}")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer
        )
        print(f"训练指标: PSNR={train_metrics['psnr']:.2f}dB, SSIM={train_metrics['ssim']:.4f}")
        
        # 验证
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer)
        print(f"验证指标: PSNR={val_metrics['psnr']:.2f}dB, SSIM={val_metrics['ssim']:.4f}")
        if 'watermark_accuracy' in val_metrics:
            print(f"水印准确率: {val_metrics['watermark_accuracy']:.4f}")
        
        # 更新学习率
        scheduler.step()
        
        # 保存检查点
        if epoch % training_config.save_interval == 0:
            checkpoint_path = os.path.join(
                training_config.checkpoint_dir,
                f'checkpoint_epoch_{epoch}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
        
        # 保存最佳模型
        current_psnr = val_metrics['psnr']
        current_wm_acc = val_metrics.get('watermark_accuracy', 0)
        
        if current_psnr > best_psnr or current_wm_acc > best_wm_acc:
            best_psnr = max(best_psnr, current_psnr)
            best_wm_acc = max(best_wm_acc, current_wm_acc)
            
            best_path = os.path.join(training_config.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics
            }, best_path)
            print(f"保存最佳模型: {best_path}")
    
    writer.close()
    print("\n训练完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练水印模型')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    args = parser.parse_args()
    
    main(args)
