import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """项目配置类"""
    
    # 数据路径
    data_dir: str = "data"
    train_image_dir: str = "data/train/images"
    train_watermark_dir: str = "data/train/watermarks"
    val_image_dir: str = "data/val/images"
    val_watermark_dir: str = "data/val/watermarks"
    num_workers: int = 4  # 数据加载器线程数
    
    # 输出路径
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    save_interval: int = 10         # 检查点保存间隔
    
    # 继续训练配置
    resume_training: bool = True  # 是否开启继续训练
    resume_checkpoint: str = "checkpoints/best.pth"  # 继续训练的检查点文件路径
    
    # 设备
    device: str = "cuda"  # 或 "cpu"

    # 图像尺寸
    image_size: int = 64
    watermark_size: int = 64
    
    # 训练参数
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    hidden_dim: int = 64            # 隐藏层维度
    num_scales: int = 3             # 多尺度训练层数
    lr_step: int = 30               # 学习率衰减周期
    lr_gamma: float = 0.5           # 学习率衰减因子
    
    # 损失权重
    lambda_image: float = 1.0       # 图像失真损失权重
    lambda_watermark: float = 1.0   # 水印提取损失权重
    lambda_sync: float = 0.5        # 同步损失权重
    lambda_confidence: float = 0.3  # 置信度损失权重
    lambda_perceptual: float = 0.1  # 感知损失权重
    
    # DWT参数
    dwt_wavelet: str = "haar"       # 小波基
    dwt_level: int = 1              # 分解层数
    
    # 攻击模拟参数
    attack_prob: float = 0.3        # 攻击概率
    jpeg_quality: Tuple[int, int] = (50, 90)
    noise_std: Tuple[float, float] = (0.01, 0.05)
    blur_kernel: Tuple[int, int] = (3, 7)
    
    # 评估指标阈值
    psnr_threshold: float = 30.0
    nc_threshold: float = 0.9
    ber_threshold: float = 0.1
    ssim_threshold: float = 0.9
    
    def __post_init__(self):
        """确保目录存在"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# 全局配置实例
config = Config()