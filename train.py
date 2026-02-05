"""
训练脚本
用于训练DWT+深度学习盲水印模型
支持从checkpoint继续训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse
import os
from tqdm import tqdm

from config import train_config, loss_config, dwt_config, create_directories
from models import create_model
from dataset import create_dataloaders
from attacks import AttackSimulator, apply_attacks_during_training
from utils import (
    calculate_psnr, calculate_ssim, calculate_batch_metrics,
    save_checkpoint, load_checkpoint, MetricsLogger, EarlyStopping,
    set_seed, get_device, print_model_summary, AverageMeter
)
from watermark_utils import ColorWatermarkAccuracyCalculator


def get_loss_functions(config):
    """获取损失函数"""
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    return {
        'l1': l1_loss,
        'mse': mse_loss,
        'bce': bce_loss
    }


def compute_loss(results, original_image, watermark_64, loss_weights, loss_fns):
    """
    计算总损失
    
    Args:
        results: 模型输出字典
        original_image: 原始图像
        watermark_64: 原始水印(64x64)
        loss_weights: 损失权重配置
        loss_fns: 损失函数字典
    
    Returns:
        总损失和损失字典
    """
    watermarked = results['watermarked']
    extracted = results['extracted']
    
    losses = {}
    
    # 1. 图像质量损失 (L1)
    losses['l1'] = loss_fns['l1'](watermarked, original_image)
    
    # 2. 水印提取损失 (MSE)
    losses['watermark'] = loss_fns['mse'](extracted, watermark_64)
    
    # 3. SSIM损失 (近似)
    # 使用MSE近似SSIM损失
    ssim_value = calculate_ssim(watermarked, original_image)
    losses['ssim'] = torch.tensor(1 - ssim_value, device=watermarked.device)
    
    # 总损失
    total_loss = (
        loss_weights.L1_WEIGHT * losses['l1'] +
        loss_weights.WATERMARK_WEIGHT * losses['watermark'] +
        loss_weights.SSIM_WEIGHT * losses['ssim']
    )
    
    losses['total'] = total_loss
    
    return total_loss, losses


def train_epoch(model, train_loader, optimizer, loss_fns, loss_weights,
                device, epoch, total_epochs, attack_simulator):
    """训练一个epoch"""
    model.train()

    losses_meter = {key: AverageMeter() for key in ['total', 'l1', 'watermark', 'ssim']}
    metrics_meter = {key: AverageMeter() for key in ['psnr', 'ssim', 'watermark_acc',
                                                      'color_acc_r', 'color_acc_g', 'color_acc_b', 'color_acc_overall']}

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{total_epochs} [Train]')
    
    for batch_idx, (images, watermarks_256, watermarks_64) in enumerate(pbar):
        # 移动到设备
        images = images.to(device)
        watermarks_256 = watermarks_256.to(device)
        watermarks_64 = watermarks_64.to(device)
        
        # 前向传播
        alpha = dwt_config.BASE_ALPHA
        
        # 应用攻击（根据训练阶段）
        def attack_fn(x):
            return apply_attacks_during_training(x, epoch, total_epochs)
        
        results = model(images, watermarks_256, alpha=alpha, attack_fn=attack_fn)
        
        # 计算损失
        total_loss, losses = compute_loss(
            results, images, watermarks_64, loss_weights, loss_fns
        )
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.GRAD_CLIP)
        
        optimizer.step()
        
        # 更新统计
        for key in losses:
            if key in losses_meter:
                losses_meter[key].update(losses[key].item())
        
        # 计算指标
        with torch.no_grad():
            batch_metrics = calculate_batch_metrics(
                images, results['watermarked'], 
                watermarks_64, results['extracted']
            )
            for key in batch_metrics:
                if key in metrics_meter:
                    metrics_meter[key].update(batch_metrics[key])
        
        # 更新进度条 - 使用RGB彩色准确率作为主要准确率指标
        pbar.set_postfix({
            'loss': f"{losses_meter['total'].avg:.4f}",
            'psnr': f"{metrics_meter['psnr'].avg:.2f}",
            'acc': f"{metrics_meter['color_acc_overall'].avg:.4f}"
        })

    # 返回平均损失和指标 - watermark_acc现在代表RGB综合准确率
    return {
        'loss': losses_meter['total'].avg,
        'l1_loss': losses_meter['l1'].avg,
        'watermark_loss': losses_meter['watermark'].avg,
        'psnr': metrics_meter['psnr'].avg,
        'ssim': metrics_meter['ssim'].avg,
        'watermark_acc': metrics_meter['color_acc_overall'].avg,  # RGB综合准确率作为主要指标
        'color_acc_r': metrics_meter['color_acc_r'].avg,
        'color_acc_g': metrics_meter['color_acc_g'].avg,
        'color_acc_b': metrics_meter['color_acc_b'].avg,
        'color_acc_overall': metrics_meter['color_acc_overall'].avg
    }


@torch.no_grad()
def validate(model, val_loader, loss_fns, loss_weights, device):
    """验证"""
    model.eval()

    losses_meter = {key: AverageMeter() for key in ['total', 'l1', 'watermark', 'ssim']}
    metrics_meter = {key: AverageMeter() for key in ['psnr', 'ssim', 'watermark_acc',
                                                      'color_acc_r', 'color_acc_g', 'color_acc_b', 'color_acc_overall']}

    pbar = tqdm(val_loader, desc='[Validate]')
    
    for images, watermarks_256, watermarks_64 in pbar:
        # 移动到设备
        images = images.to(device)
        watermarks_256 = watermarks_256.to(device)
        watermarks_64 = watermarks_64.to(device)
        
        # 前向传播
        alpha = dwt_config.BASE_ALPHA
        results = model(images, watermarks_256, alpha=alpha)
        
        # 计算损失
        total_loss, losses = compute_loss(
            results, images, watermarks_64, loss_weights, loss_fns
        )
        
        # 更新统计
        for key in losses:
            if key in losses_meter:
                losses_meter[key].update(losses[key].item())
        
        # 计算指标
        batch_metrics = calculate_batch_metrics(
            images, results['watermarked'],
            watermarks_64, results['extracted']
        )
        for key in batch_metrics:
            if key in metrics_meter:
                metrics_meter[key].update(batch_metrics[key])
        
        # 更新进度条 - 使用RGB彩色准确率作为主要准确率指标
        pbar.set_postfix({
            'loss': f"{losses_meter['total'].avg:.4f}",
            'psnr': f"{metrics_meter['psnr'].avg:.2f}",
            'acc': f"{metrics_meter['color_acc_overall'].avg:.4f}",  # RGB综合准确率
            'rgb': f"R:{metrics_meter['color_acc_r'].avg:.3f}G:{metrics_meter['color_acc_g'].avg:.3f}B:{metrics_meter['color_acc_b'].avg:.3f}"
        })

    return {
        'loss': losses_meter['total'].avg,
        'l1_loss': losses_meter['l1'].avg,
        'watermark_loss': losses_meter['watermark'].avg,
        'psnr': metrics_meter['psnr'].avg,
        'ssim': metrics_meter['ssim'].avg,
        'watermark_acc': metrics_meter['color_acc_overall'].avg,  # RGB综合准确率作为主要指标
        'color_acc_r': metrics_meter['color_acc_r'].avg,
        'color_acc_g': metrics_meter['color_acc_g'].avg,
        'color_acc_b': metrics_meter['color_acc_b'].avg,
        'color_acc_overall': metrics_meter['color_acc_overall'].avg
    }


def train(args):
    """主训练函数"""
    # 设置随机种子
    set_seed(train_config.SEED)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建目录
    create_directories()
    
    # 创建数据加载器
    print("\n加载数据集...")
    train_loader, val_loader = create_dataloaders(train_config)
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    from config import model_config
    model = create_model(model_config, device)
    print_model_summary(model)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.LEARNING_RATE,
        weight_decay=train_config.WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=train_config.LEARNING_RATE_MIN
    )
    
    # 损失函数
    loss_fns = get_loss_functions(loss_config)
    
    # 攻击模拟器
    attack_simulator = AttackSimulator()
    
    # 日志记录器
    logger = MetricsLogger(train_config.LOG_DIR)
    
    # 早停机制
    early_stopping = EarlyStopping(patience=train_config.EARLY_STOP_PATIENCE)
    
    # 从checkpoint恢复
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume or train_config.RESUME:
        resume_path = args.resume if args.resume else train_config.RESUME_PATH
        if os.path.exists(resume_path):
            print(f"\n从checkpoint恢复: {resume_path}")
            start_epoch, metrics = load_checkpoint(resume_path, model, optimizer, device)
            best_psnr = metrics.get('psnr', 0.0)
            print(f"恢复自epoch {start_epoch}, 最佳PSNR: {best_psnr:.2f}")
        else:
            print(f"警告: checkpoint不存在: {resume_path}")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=train_config.LOG_DIR)
    
    # 训练循环
    print("\n开始训练...")
    print("="*50)
    
    for epoch in range(start_epoch, train_config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{train_config.EPOCHS}")
        print("-"*50)
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fns, loss_config,
            device, epoch + 1, train_config.EPOCHS, attack_simulator
        )
        
        # 记录训练指标
        logger.log_train(epoch + 1, train_metrics)
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f'Train/{key}', value, epoch + 1)
        
        # 训练日志 - 使用RGB彩色准确率作为主要准确率指标
        print(f"训练 - Loss: {train_metrics['loss']:.4f}, "
              f"PSNR: {train_metrics['psnr']:.2f}dB, "
              f"SSIM: {train_metrics['ssim']:.4f}, "
              f"Acc(RGB): {train_metrics['watermark_acc']:.4f} "  # RGB综合准确率
              f"[R:{train_metrics['color_acc_r']:.3f} G:{train_metrics['color_acc_g']:.3f} B:{train_metrics['color_acc_b']:.3f}]")

        # 验证
        if (epoch + 1) % train_config.VAL_FREQUENCY == 0:
            val_metrics = validate(
                model, val_loader, loss_fns, loss_config, device
            )

            # 记录验证指标
            logger.log_val(epoch + 1, val_metrics)
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'Val/{key}', value, epoch + 1)

            # 记录RGB彩色准确率到TensorBoard
            writer.add_scalar('Val/ColorAcc/R', val_metrics['color_acc_r'], epoch + 1)
            writer.add_scalar('Val/ColorAcc/G', val_metrics['color_acc_g'], epoch + 1)
            writer.add_scalar('Val/ColorAcc/B', val_metrics['color_acc_b'], epoch + 1)
            writer.add_scalar('Val/ColorAcc/Overall', val_metrics['color_acc_overall'], epoch + 1)

            # 验证日志 - 使用RGB彩色准确率作为主要准确率指标
            print(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                  f"PSNR: {val_metrics['psnr']:.2f}dB, "
                  f"SSIM: {val_metrics['ssim']:.4f}, "
                  f"Acc(RGB): {val_metrics['watermark_acc']:.4f} "  # RGB综合准确率
                  f"[R:{val_metrics['color_acc_r']:.3f} G:{val_metrics['color_acc_g']:.3f} B:{val_metrics['color_acc_b']:.3f}]")
            
            # 保存最佳模型
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                best_path = os.path.join(train_config.CHECKPOINT_DIR, 
                                        train_config.BEST_MODEL_NAME)
                save_checkpoint(model, optimizer, epoch + 1, val_metrics, best_path)
                print(f"保存最佳模型 (PSNR: {best_psnr:.2f}dB)")
            
            # 早停检查
            if early_stopping(val_metrics['psnr']):
                print(f"\n早停触发！最佳PSNR: {early_stopping.best_score:.2f}dB")
                break
        
        # 定期保存模型
        if (epoch + 1) % train_config.SAVE_FREQUENCY == 0:
            latest_path = os.path.join(train_config.CHECKPOINT_DIR,
                                      f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, train_metrics, latest_path)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/learning_rate', current_lr, epoch + 1)
    
    # 保存最终模型
    final_path = os.path.join(train_config.CHECKPOINT_DIR, 
                             train_config.LATEST_MODEL_NAME)
    final_metrics = val_metrics if 'val_metrics' in locals() else train_metrics
    save_checkpoint(model, optimizer, train_config.EPOCHS, final_metrics, final_path)
    print(f"\n保存最终模型: {final_path}")
    
    # 关闭TensorBoard
    writer.close()
    
    print("\n" + "="*50)
    print("训练完成!")
    print(f"最佳PSNR: {best_psnr:.2f}dB")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description='训练DWT+深度学习盲水印模型')
    parser.add_argument('--resume', type=str, default='',
                       help='从checkpoint恢复训练的路径')
    parser.add_argument('--epochs', type=int, default=0,
                       help='训练轮数（覆盖配置）')
    parser.add_argument('--batch_size', type=int, default=0,
                       help='批次大小（覆盖配置）')
    parser.add_argument('--lr', type=float, default=0.0,
                       help='学习率（覆盖配置）')
    
    args = parser.parse_args()
    
    # 覆盖配置
    if args.epochs > 0:
        train_config.EPOCHS = args.epochs
    if args.batch_size > 0:
        train_config.BATCH_SIZE = args.batch_size
    if args.lr > 0:
        train_config.LEARNING_RATE = args.lr
    
    # 开始训练
    train(args)


if __name__ == '__main__':
    main()
