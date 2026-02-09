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
    test_image_dir: str = "data/test/images"
    test_watermark_dir: str = "data/test/watermarks"
    num_workers: int = 4  # 数据加载器线程数
    
    # 输出路径
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    save_interval: int = 10         # 检查点保存间隔
    
    # 继续训练配置
    resume_training: bool = False  # 是否开启继续训练
    resume_checkpoint: str = "checkpoints/latest.pth"  # 继续训练的检查点文件路径
    
    # 设备
    device: str = "cuda"  # 或 "cpu"

    # 图像尺寸
    image_size: int = 150
    watermark_size: int = 64
    overlap: int = 0            # 分块重叠区域
    block_mode: str = 'all'  # 分块模式：'all'或'random'
    
    # 训练参数
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    hidden_dim: int = 128           # 隐藏层维度
    num_scales: int = 3             # 多尺度训练层数
    lr_step: int = 30               # 学习率衰减周期
    lr_gamma: float = 0.5           # 学习率衰减因子
    use_multiscale_decoder: bool = True  # 是否使用多尺度解码器
    
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
    attack_prob: float = 0.5        # 攻击概率
    
    # 评估指标阈值
    psnr_threshold: float = 35.0
    nc_threshold: float = 0.8
    ber_threshold: float = 0.1
    ssim_threshold: float = 0.8
    
    def __post_init__(self):
        """确保目录存在"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# 全局配置实例
config = Config()