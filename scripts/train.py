import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance
import numpy as np
import random
from tqdm import tqdm
import logging
import time
import io
import contextlib

class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.console = sys.stdout
        self.file = None
        self._setup_logger()
    
    def _setup_logger(self):
        """设置日志文件"""
        try:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            self.file = open(self.log_file, 'w', encoding='utf-8')
        except Exception as e:
            print(f"Error opening log file: {e}")
            self.file = None
    
    def write(self, message):
        """写入消息到控制台和文件"""
        # 写入控制台
        self.console.write(message)
        self.console.flush()
        
        # 写入文件
        if self.file:
            try:
                self.file.write(message)
                self.file.flush()
            except Exception as e:
                print(f"Error writing to log file: {e}")
    
    def flush(self):
        """刷新缓冲区"""
        self.console.flush()
        if self.file:
            try:
                self.file.flush()
            except Exception as e:
                print(f"Error flushing log file: {e}")
    
    def close(self):
        """关闭日志文件"""
        if self.file:
            try:
                self.file.close()
            except Exception as e:
                print(f"Error closing log file: {e}")

# 保存原始的stdout
original_stdout = sys.stdout

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.models import WatermarkModel
from src.adversarial import AdversarialTrainer
from src.embedding import BatchEmbedding

class WatermarkDataset(Dataset):
    """
    水印数据集
    """
    
    def __init__(self, root_dir, watermark_type='image', transform=None):
        """
        初始化数据集
        
        Args:
            root_dir: 根目录
            watermark_type: 水印类型
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.watermark_type = watermark_type
        self.transform = transform
        
        # 加载图像路径
        self.image_paths = []
        self.watermark_paths = []
        
        image_dir = os.path.join(root_dir, 'images')
        if watermark_type == 'image':
            watermark_dir = os.path.join(root_dir, 'watermark_img')
            watermark_files = [f for f in os.listdir(watermark_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in os.listdir(image_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(image_dir, img_name)
                    # 随机选择一个水印文件
                    if watermark_files:
                        watermark_file = np.random.choice(watermark_files)
                        watermark_path = os.path.join(watermark_dir, watermark_file)
                        self.image_paths.append(img_path)
                        self.watermark_paths.append(watermark_path)
        else:
            watermark_dir = os.path.join(root_dir, 'watermark_txt')
            watermark_files = [f for f in os.listdir(watermark_dir) if f.endswith('.txt')]
            for img_name in os.listdir(image_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(image_dir, img_name)
                    # 随机选择一个文本水印
                    watermark_file = np.random.choice(watermark_files)
                    watermark_path = os.path.join(watermark_dir, watermark_file)
                    self.image_paths.append(img_path)
                    self.watermark_paths.append(watermark_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            image: 载体图像
            watermark: 水印
        """
        # 加载载体图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        
        # 数据增强
        if random.random() > 0.5:
            # 随机水平翻转
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            # 随机垂直翻转
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() > 0.5:
            # 随机旋转
            angle = random.randint(-15, 15)
            image = image.rotate(angle, expand=False)
        if random.random() > 0.5:
            # 随机亮度调整
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if random.random() > 0.5:
            # 随机对比度调整
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if random.random() > 0.5:
            # 随机饱和度调整
            factor = random.uniform(0.8, 1.2)
            image = ImageEnhance.Color(image).enhance(factor)
        
        image = np.array(image) / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        # 加载水印
        watermark_path = self.watermark_paths[idx]
        if self.watermark_type == 'image':
            watermark = Image.open(watermark_path).convert('L')
            watermark = watermark.resize((config.WATERMARK_SIZE, config.WATERMARK_SIZE), Image.LANCZOS)
            watermark = np.array(watermark) / 255.0
            watermark = torch.tensor(watermark, dtype=torch.float32).unsqueeze(0)
        else:
            with open(watermark_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            # 文本转二进制
            binary = ''.join(format(ord(c), '08b') for c in text)
            binary = binary.ljust(config.TEXT_WATERMARK_LENGTH, '0')[:config.TEXT_WATERMARK_LENGTH]
            binary_array = np.array([int(bit) for bit in binary])
            # 重塑为 (C, H, W)
            channels = config.TEXT_WATERMARK_LENGTH // (config.WATERMARK_SIZE * config.WATERMARK_SIZE)
            if config.TEXT_WATERMARK_LENGTH % (config.WATERMARK_SIZE * config.WATERMARK_SIZE) != 0:
                channels += 1
            padding = channels * config.WATERMARK_SIZE * config.WATERMARK_SIZE - len(binary_array)
            if padding > 0:
                binary_array = np.pad(binary_array, (0, padding), 'constant')
            watermark_array = binary_array.reshape(channels, config.WATERMARK_SIZE, config.WATERMARK_SIZE)
            watermark = torch.tensor(watermark_array, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            watermark = self.transform(watermark)
        
        return image, watermark

def train(finetune=False, finetune_checkpoint=None):
    """
    训练水印模型
    
    Args:
        finetune: 是否为微调模式
        finetune_checkpoint: 微调时加载的模型路径
    """
    # 创建日志文件
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_filename = f"train_log_{timestamp}.txt"
    log_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", log_filename)
    
    # 重定向stdout到日志记录器
    logger = Logger(log_filepath)
    sys.stdout = logger
    
    try:
        print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"Log file: {log_filepath}")
        print(f"Finetune mode: {finetune}")
        if finetune_checkpoint:
            print(f"Finetune checkpoint: {finetune_checkpoint}")
        
        # 配置
        watermark_type = config.TRAIN_TYPE
        
        # 创建数据集
        train_dataset = WatermarkDataset(config.TRAIN_DIR, watermark_type=watermark_type)
        val_dataset = WatermarkDataset(config.VAL_DIR, watermark_type=watermark_type)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # 启用GPU加速
        device = torch.device(config.DEVICE)
        config.DEVICE = device
        print(f"Training on device: {device}")
        
        # 初始化模型，直接在GPU上创建
        model = WatermarkModel(watermark_type=watermark_type, device=device)
        model.to(device)
        
        # 优化器和损失函数
        learning_rate = config.FINETUNE_LR if finetune else config.LEARNING_RATE
        # 使用AdamW优化器，调整β参数
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, 
                               weight_decay=config.WEIGHT_DECAY, 
                               betas=(0.9, 0.99))
        
        # 使用改进的WatermarkLoss损失函数
        from src.adversarial import WatermarkLoss
        criterion = WatermarkLoss(embedding_weight=1.5, extraction_weight=1.0)
        
        # 学习率调度器
        if config.LR_SCHEDULER_TYPE == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
        elif config.LR_SCHEDULER_TYPE == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)
        else:
            scheduler = None
        
        # 对抗性训练器
        trainer = AdversarialTrainer(model, optimizer, criterion)
        
        # 初始化可视化工具
        from src.visualization import TrainingVisualizer
        visualizer = TrainingVisualizer()
        
        # 加载模型
        start_epoch = 0
        best_val_loss = float('inf')
        best_val_accuracy = 0.0
        best_val_psnr = 0.0
        best_model_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_{watermark_type}.pth")
        patience_counter = 0
        
        # 加载微调模型或恢复训练
        load_path = finetune_checkpoint if finetune else (config.RESUME_CHECKPOINT if config.RESUME else None)
        if load_path and os.path.exists(load_path):
            print(f"Loading checkpoint from {load_path}")
            checkpoint = torch.load(load_path, map_location=device)
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态（如果不是微调）
            if not finetune and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载训练状态
            if not finetune and 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                
            print(f"Resuming training from epoch {start_epoch}")
        elif load_path:
            print(f"Checkpoint file not found: {load_path}")
            print("Starting training from scratch")
        
        # 训练循环
        for epoch in range(start_epoch, config.EPOCHS):
            print(f"Epoch {epoch+1}/{config.EPOCHS}")
            print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 训练
            train_loss, train_psnr = trainer.train_epoch(train_loader, config.ATTACK_TRAINING)
            print(f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f} dB")
            
            # 验证
            val_loss, val_accuracy, val_psnr = trainer.validate(val_loader)
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val PSNR: {val_psnr:.2f} dB")
            
            # 更新可视化工具
            visualizer.update(train_loss=train_loss, train_psnr=train_psnr,
                           val_loss=val_loss, val_accuracy=val_accuracy,
                           val_psnr=val_psnr, lr=optimizer.param_groups[0]['lr'])
            
            # 更新损失函数的历史记录
            if isinstance(criterion, WatermarkLoss):
                criterion.update_history(val_psnr, val_accuracy)
            
            # 更新学习率
            if scheduler:
                scheduler.step()
            
            # 综合早停策略（基于多种指标）
            if config.EARLY_STOPPING:
                # 计算综合得分
                current_score = 0.5 * (1.0 / val_loss) + 0.3 * val_accuracy + 0.2 * (val_psnr / 50.0)
                best_score = 0.5 * (1.0 / best_val_loss) + 0.3 * best_val_accuracy + 0.2 * (best_val_psnr / 50.0)
                
                if current_score > best_score + 0.01:  # 0.01 作为阈值
                    # 更新最佳指标
                    best_val_loss = val_loss
                    best_val_accuracy = val_accuracy
                    best_val_psnr = val_psnr
                    patience_counter = 0
                    # 保存最佳模型
                    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        'accuracy': val_accuracy,
                        'psnr': val_psnr
                    }, best_model_path)
                    print(f"Saved best model to {best_model_path}")
                else:
                    patience_counter += 1
                    print(f"Early stopping patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
                    if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                        print("Early stopping triggered!")
                        break
            
            # 每次保存绘制一次曲线
            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                visualizer.plot_all()
                print(f"Plotted training curves at epoch {epoch+1}")
            
            # 保存模型
            if (epoch + 1) % config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_{watermark_type}_epoch_{epoch+1}.pth")
                os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'accuracy': val_accuracy
                }, checkpoint_path)
                print(f"Saved model to {checkpoint_path}")
            
            print("=" * 60)
        
        print(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        print(f"Training completed! Best model saved at: {best_model_path}")
        return best_model_path
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # 恢复原始的stdout
        sys.stdout = original_stdout
        # 关闭日志文件
        logger.close()
        print(f"Log file closed. Training process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train watermark model")
    parser.add_argument('--finetune', action='store_true', help='Enable finetuning mode')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for finetuning')
    args = parser.parse_args()
    
    # 如果是微调但没有指定checkpoint，尝试使用最佳模型
    if args.finetune and not args.checkpoint:
        watermark_type = config.TRAIN_TYPE
        best_model_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_{watermark_type}.pth")
        if os.path.exists(best_model_path):
            args.checkpoint = best_model_path
            print(f"Finetuning mode enabled, using best model: {best_model_path}")
        else:
            print("No best model found for finetuning. Please specify a checkpoint.")
            exit(1)
    
    train(finetune=args.finetune, finetune_checkpoint=args.checkpoint)