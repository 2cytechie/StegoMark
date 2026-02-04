"""
训练过程可视化工具
用于监控和分析训练过程中的各种指标
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class TrainingVisualizer:
    """
    训练过程可视化类
    """
    
    def __init__(self, log_dir='output/visualizations'):
        """
        初始化可视化工具
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 训练指标
        self.train_losses = []
        self.train_psnr = []
        self.val_losses = []
        self.val_accuracy = []
        self.val_psnr = []
        self.lr_history = []
    
    def update(self, train_loss=None, train_psnr=None, val_loss=None, val_accuracy=None, val_psnr=None, lr=None):
        """
        更新训练指标
        
        Args:
            train_loss: 训练损失
            train_psnr: 训练PSNR
            val_loss: 验证损失
            val_accuracy: 验证准确率
            val_psnr: 验证PSNR
            lr: 学习率
        """
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if train_psnr is not None:
            self.train_psnr.append(train_psnr)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_accuracy is not None:
            self.val_accuracy.append(val_accuracy)
        if val_psnr is not None:
            self.val_psnr.append(val_psnr)
        if lr is not None:
            self.lr_history.append(lr)
    
    def plot_loss(self, save_path=None):
        """
        绘制损失曲线
        
        Args:
            save_path: 保存路径
        """
        if not save_path:
            save_path = os.path.join(self.log_dir, 'loss_curve.png')
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_psnr(self, save_path=None):
        """
        绘制PSNR曲线
        
        Args:
            save_path: 保存路径
        """
        if not save_path:
            save_path = os.path.join(self.log_dir, 'psnr_curve.png')
        
        plt.figure(figsize=(12, 6))
        if self.train_psnr:
            plt.plot(self.train_psnr, label='Train PSNR', linewidth=2)
        if self.val_psnr:
            plt.plot(self.val_psnr, label='Val PSNR', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title('Training PSNR Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    
    def plot_accuracy(self, save_path=None):
        """
        绘制准确率曲线
        
        Args:
            save_path: 保存路径
        """
        if not save_path:
            save_path = os.path.join(self.log_dir, 'accuracy_curve.png')
        
        if self.val_accuracy:
            plt.figure(figsize=(12, 6))
            plt.plot(self.val_accuracy, label='Val Accuracy', linewidth=2, color='green')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title('Validation Accuracy Curve', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        return None
    
    def plot_lr(self, save_path=None):
        """
        绘制学习率曲线
        
        Args:
            save_path: 保存路径
        """
        if not save_path:
            save_path = os.path.join(self.log_dir, 'lr_curve.png')
        
        if self.lr_history:
            plt.figure(figsize=(12, 6))
            plt.plot(self.lr_history, label='Learning Rate', linewidth=2, color='purple')
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        return None
    
    def plot_all(self):
        """
        绘制所有曲线
        """
        self.plot_loss()
        self.plot_psnr()
        self.plot_accuracy()
        self.plot_lr()
        
    def save_metrics(self, save_path=None):
        """
        保存训练指标到文件
        
        Args:
            save_path: 保存路径
        """
        if not save_path:
            save_path = os.path.join(self.log_dir, 'metrics.npz')
        
        np.savez(save_path,
                train_losses=self.train_losses,
                train_psnr=self.train_psnr,
                val_losses=self.val_losses,
                val_accuracy=self.val_accuracy,
                val_psnr=self.val_psnr,
                lr_history=self.lr_history)
        return save_path
    
    def load_metrics(self, load_path):
        """
        从文件加载训练指标
        
        Args:
            load_path: 加载路径
        """
        if os.path.exists(load_path):
            data = np.load(load_path)
            self.train_losses = data['train_losses'].tolist()
            self.train_psnr = data['train_psnr'].tolist()
            self.val_losses = data['val_losses'].tolist()
            self.val_accuracy = data['val_accuracy'].tolist()
            self.val_psnr = data['val_psnr'].tolist()
            self.lr_history = data['lr_history'].tolist()
            return True
        return False
