"""
全局配置文件
"""
import os
import torch
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataConfig:
    """数据配置"""
    train_images_dir: str = os.path.join("data", "train", "images")
    train_watermarks_dir: str = os.path.join("data", "train", "watermarks")
    val_images_dir: str = os.path.join("data", "val", "images")
    val_watermarks_dir: str = os.path.join("data", "val", "watermarks")
    
    # 水印尺寸
    watermark_size: int = 64
    
    # 数据增强参数
    flip_prob: float = 0.5
    blur_prob: float = 0.3
    color_jitter_prob: float = 0.3
    crop_prob: float = 0.5


@dataclass
class ModelConfig:
    """模型配置"""
    # DWT配置
    wavelet: str = 'haar'  # 小波基
    mode: str = 'reflect'  # 边界处理模式 (PyTorch支持: 'constant', 'reflect', 'replicate', 'circular')
    
    # 编码器通道数
    encoder_channels: int = 64
    
    # 解码器配置
    use_stn: bool = True  # 是否使用空间变换网络
    decoder_channels: int = 64
    
    # 分组卷积配置
    use_grouped_conv: bool = True  # 是否使用分组卷积共享权重（可显著降低显存占用），显存大时可以关闭
    num_blocks: int = 3  # 残差块数量（编码器）
    num_blocks_decoder: int = 4  # 残差块数量（解码器）


@dataclass
class AttackConfig:
    """攻击模拟配置"""
    # 攻击概率
    random_crop_prob: float = 0.5
    random_rotate_prob: float = 0.3
    gaussian_blur_prob: float = 0.3
    jpeg_compress_prob: float = 0.3
    noise_prob: float = 0.3
    
    # 攻击参数范围
    crop_scale_range: Tuple[float, float] = (0.7, 1.0)
    rotate_angle_range: Tuple[int, int] = (-15, 15)
    blur_kernel_range: Tuple[int, int] = (3, 7)
    jpeg_quality_range: Tuple[int, int] = (50, 90)
    noise_std_range: Tuple[float, float] = (0.01, 0.05)


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 1
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # 损失权重
    lambda_image: float = 1.0  # 图像失真损失权重
    lambda_watermark: float = 1.0  # 水印提取损失权重
    
    # 评估指标阈值
    psnr_threshold: float = 30.0
    ssim_threshold: float = 0.9
    accuracy_threshold: float = 0.9
    
    # 保存配置
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 10  # 每10个epoch保存一次
    
    # 设备配置
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 0


@dataclass
class InferenceConfig:
    """推理配置"""
    # 投票机制参数
    block_size: int = 64
    overlap: float = 0.5
    correlation_threshold: float = 0.7
    
    # 色彩增强参数
    color_correction: bool = True


# 全局配置实例
data_config = DataConfig()
model_config = ModelConfig()
attack_config = AttackConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
