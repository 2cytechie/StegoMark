import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import config
from src.data import WatermarkDataset
from src.models import WatermarkNet
from src.utils import WatermarkLoss, MetricsTracker, calculate_psnr, calculate_ssim, calculate_nc, calculate_ber


class Trainer:
    """水印网络训练器"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.exp_dir = config.checkpoint_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 初始化模型
        self.model = WatermarkNet(
            hidden_dim=args.hidden_dim,
            attack_prob=args.attack_prob,
            use_multiscale_decoder=args.use_multiscale,
            num_scales=args.num_scales
        ).to(self.device)
        
        # 损失函数
        self.criterion = WatermarkLoss(
            lambda_image=args.lambda_image,
            lambda_watermark=args.lambda_watermark,
            lambda_sync=args.lambda_sync,
            lambda_confidence=args.lambda_confidence,
            lambda_perceptual=args.lambda_perceptual,
            use_perceptual=args.use_perceptual
        )
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lr_step,
            gamma=args.lr_gamma
        )
        
        # 数据加载
        self.train_loader, self.val_loader = self._create_dataloaders()
        
        # 指标追踪
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # 最佳模型
        self.best_psnr = 0.0
        self.best_nc = 0.0
        
        # 加载检查点
        self.start_epoch = 0
        # 优先使用命令行参数，其次使用 config 配置
        resume_path = args.resume if args.resume else (config.resume_checkpoint if config.resume_training else None)
        if resume_path and os.path.exists(resume_path):
            self._load_checkpoint(resume_path)
        elif config.resume_training:
            print(f"警告: 未找到检查点文件 {resume_path}，将从头开始训练")
    
    def _create_dataloaders(self):
        """创建数据加载器"""
        # 训练数据集
        train_dataset = WatermarkDataset(
            image_dir=self.args.train_image_dir,
            watermark_dir=self.args.train_watermark_dir,
            image_size=self.args.image_size,
            watermark_size=self.args.watermark_size
        )
        
        # 验证数据集
        val_dataset = WatermarkDataset(
            image_dir=self.args.val_image_dir,
            watermark_dir=self.args.val_watermark_dir,
            image_size=self.args.image_size,
            watermark_size=self.args.watermark_size
        )
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch']
        self.best_psnr = checkpoint.get('best_psnr', 0.0)
        print(f"从epoch {self.start_epoch}继续训练")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'args': self.args
        }

        # 保存最新检查点
        latest_path = os.path.join(self.exp_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)

        # 保存最佳检查点
        if is_best:
            best_path = os.path.join(self.exp_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型 (PSNR: {self.best_psnr:.2f})")

        # 定期保存
        if (epoch + 1) % self.args.save_interval == 0:
            epoch_path = os.path.join(self.exp_dir, f'epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        
        for batch_idx, (images, watermarks) in enumerate(pbar):
            images = images.to(self.device)
            watermarks = watermarks.to(self.device)
            
            # 前向传播 - 有水印样本
            watermarked, attacked, extracted_wm, confidence = self.model(images, watermarks)
            
            # 计算损失
            loss, loss_dict = self.criterion(
                images, watermarked, watermarks, extracted_wm, confidence
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 每5个batch添加无水印样本训练
            if batch_idx % 5 == 0:
                # 创建无水印样本（使用随机噪声作为水印）
                fake_watermarks = torch.rand_like(watermarks)
                with torch.no_grad():
                    fake_watermarked = self.model.encode(images, fake_watermarks)
                
                # 从含水印图像中提取（应该失败）
                fake_extracted, fake_confidence = self.model.decode(fake_watermarked)
                
                # 计算无水印样本的置信度损失（目标为0）
                target_no_watermark = torch.zeros_like(fake_confidence)
                loss_no_wm = F.binary_cross_entropy(fake_confidence, target_no_watermark)
                
                # 加权添加到总损失
                loss_no_wm = loss_no_wm * 0.3  # 权重较小
                loss_no_wm.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 计算指标
            with torch.no_grad():
                psnr = calculate_psnr(images, watermarked)
                ssim = calculate_ssim(images, watermarked)
                nc = calculate_nc(watermarks, extracted_wm)
                ber = calculate_ber(watermarks, extracted_wm)
            
            # 更新指标
            self.train_metrics.update(
                psnr=psnr,
                ssim=ssim,
                nc=nc,
                ber=ber,
                loss=loss.item()
            )
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'PSNR': f"{psnr:.2f}",
                'NC': f"{nc:.4f}",
                'Conf': f"{confidence.mean().item():.4f}"
            })
        
        # 记录epoch平均指标
        avg_metrics = self.train_metrics.get_average()
        print(f"\n训练指标 - {self.train_metrics}")
        
        return avg_metrics
    
    def validate(self, epoch):
        """验证"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]")
            
            for images, watermarks in pbar:
                images = images.to(self.device)
                watermarks = watermarks.to(self.device)
                
                # 前向传播（无攻击）
                watermarked, _, extracted_wm, confidence = self.model(
                    images, watermarks, no_attack=True
                )
                
                # 计算指标
                psnr = calculate_psnr(images, watermarked)
                ssim = calculate_ssim(images, watermarked)
                nc = calculate_nc(watermarks, extracted_wm)
                ber = calculate_ber(watermarks, extracted_wm)
                
                # 更新指标
                self.val_metrics.update(
                    psnr=psnr,
                    ssim=ssim,
                    nc=nc,
                    ber=ber
                )
                
                pbar.set_postfix({
                    'PSNR': f"{psnr:.2f}",
                    'NC': f"{nc:.4f}",
                    'BER': f"{ber:.4f}"
                })
        
        # 记录epoch平均指标
        avg_metrics = self.val_metrics.get_average()
        print(f"\n验证指标 - {self.val_metrics}")
        
        return avg_metrics
    
    def train(self):
        """主训练循环"""
        print(f"开始训练，设备: {self.device}")
        print(f"实验目录: {self.exp_dir}")
        
        for epoch in range(self.start_epoch, self.args.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.args.epochs}")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"嵌入强度: {self.model.get_embedding_strength():.4f}")
            print(f"{'='*50}")
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 学习率调整
            self.scheduler.step()
            
            # 保存最佳模型
            is_best = val_metrics['nc'] > self.best_nc
            if is_best:
                self.best_nc = val_metrics['nc']
                self.best_psnr = val_metrics['psnr']
            
            # 保存检查点
            self._save_checkpoint(epoch, is_best=is_best)
            
            # 检查是否达到目标
            if val_metrics['psnr'] >= config.psnr_threshold and val_metrics['nc'] >= config.nc_threshold:
                print(f"\n达到目标性能！PSNR: {val_metrics['psnr']:.2f} >= {config.psnr_threshold}, "
                      f"NC: {val_metrics['nc']:.4f} >= {config.nc_threshold}")
        
        print(f"\n训练完成！最佳PSNR: {self.best_psnr:.2f} dB")


def parse_args():
    """解析命令行参数，使用 config.py 作为默认值"""
    parser = argparse.ArgumentParser(description='训练水印网络')

    # 数据参数 - 使用 config 作为默认值
    parser.add_argument('--train_image_dir', type=str, default=config.train_image_dir)
    parser.add_argument('--train_watermark_dir', type=str, default=config.train_watermark_dir)
    parser.add_argument('--val_image_dir', type=str, default=config.val_image_dir)
    parser.add_argument('--val_watermark_dir', type=str, default=config.val_watermark_dir)
    parser.add_argument('--image_size', type=int, default=config.image_size)
    parser.add_argument('--watermark_size', type=int, default=config.watermark_size)

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=config.hidden_dim)
    parser.add_argument('--attack_prob', type=float, default=config.attack_prob)
    parser.add_argument('--use_multiscale', action='store_true', default=config.use_multiscale_decoder)
    parser.add_argument('--num_scales', type=int, default=config.num_scales)

    # 损失权重 - 使用 config 作为默认值
    parser.add_argument('--lambda_image', type=float, default=config.lambda_image)
    parser.add_argument('--lambda_watermark', type=float, default=config.lambda_watermark)
    parser.add_argument('--lambda_sync', type=float, default=config.lambda_sync)
    parser.add_argument('--lambda_confidence', type=float, default=config.lambda_confidence)
    parser.add_argument('--lambda_perceptual', type=float, default=config.lambda_perceptual)
    parser.add_argument('--use_perceptual', action='store_true')

    # 训练参数 - 使用 config 作为默认值
    parser.add_argument('--epochs', type=int, default=config.num_epochs)
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--lr', type=float, default=config.learning_rate)
    parser.add_argument('--weight_decay', type=float, default=config.weight_decay)
    parser.add_argument('--lr_step', type=int, default=config.lr_step)
    parser.add_argument('--lr_gamma', type=float, default=config.lr_gamma)

    # 其他参数 - 使用 config 作为默认值
    parser.add_argument('--device', type=str, default=config.device)
    parser.add_argument('--num_workers', type=int, default=config.num_workers)
    parser.add_argument('--save_interval', type=int, default=config.save_interval)
    parser.add_argument('--resume', type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()