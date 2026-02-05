"""
DWT + 深度学习盲水印系统配置文件
所有可调节参数都集中在这里
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ImageConfig:
    """图像尺寸配置"""
    # 目标图像尺寸
    TARGET_SIZE: int = 256
    # 水印图像尺寸（原始）
    WATERMARK_SIZE: int = 64
    # 水印复制后的尺寸（64x64复制4x4 = 256x256）
    WATERMARK_TILED_SIZE: int = 256
    # 图像通道数
    CHANNELS: int = 3


@dataclass
class DWTConfig:
    """DWT变换配置"""
    # 小波类型: 'haar', 'db1', 'db2', 'bior1.3', etc.
    WAVELET: str = 'haar'
    # 分解层数
    LEVEL: int = 1
    # 嵌入的子带: 'LH', 'HL', 'HH' (高频子带)
    EMBED_SUBBANDS: List[str] = field(default_factory=lambda: ['LH', 'HL', 'HH'])
    # 嵌入强度基础值（会被深度学习网络调整）
    BASE_ALPHA: float = 0.1


@dataclass
class ModelConfig:
    """深度学习网络配置"""
    # 嵌入网络编码器通道数
    EMBED_ENCODER_CHANNELS: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    # 嵌入网络解码器通道数
    EMBED_DECODER_CHANNELS: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    # 提取网络通道数
    EXTRACT_CHANNELS: List[int] = field(default_factory=lambda: [64, 128, 256])
    # ResBlock数量
    NUM_RESBLOCKS: int = 4
    # 使用批归一化
    USE_BATCHNORM: bool = True
    # Dropout概率
    DROPOUT: float = 0.1


@dataclass
class AttackConfig:
    """攻击模拟配置"""
    # 是否启用攻击训练
    ENABLE_ATTACK_TRAINING: bool = True
    
    # 裁剪攻击参数
    CROP_RATIO_MIN: float = 0.7
    CROP_RATIO_MAX: float = 0.95
    
    # 旋转攻击参数（角度）
    ROTATE_ANGLE_MIN: int = -15
    ROTATE_ANGLE_MAX: int = 15
    
    # 缩放攻击参数
    SCALE_MIN: float = 0.8
    SCALE_MAX: float = 1.2
    
    # 高斯模糊参数
    BLUR_KERNEL_MIN: int = 3
    BLUR_KERNEL_MAX: int = 7
    BLUR_SIGMA_MIN: float = 0.5
    BLUR_SIGMA_MAX: float = 2.0
    
    # 高斯噪声参数
    NOISE_MEAN: float = 0.0
    NOISE_STD_MIN: float = 0.01
    NOISE_STD_MAX: float = 0.05
    
    # JPEG压缩参数
    JPEG_QUALITY_MIN: int = 50
    JPEG_QUALITY_MAX: int = 95
    
    # 攻击概率（训练时应用攻击的概率）
    ATTACK_PROBABILITY: float = 0.5


@dataclass
class LossConfig:
    """损失函数配置"""
    # 图像质量损失权重 (L1 Loss)
    L1_WEIGHT: float = 1.0
    # 水印提取损失权重 (BCE Loss)
    WATERMARK_WEIGHT: float = 2.0
    # 感知损失权重 (VGG Perceptual Loss)
    PERCEPTUAL_WEIGHT: float = 0.5
    # 对抗攻击损失权重
    ADVERSARIAL_WEIGHT: float = 1.0
    # SSIM损失权重
    SSIM_WEIGHT: float = 0.5


@dataclass
class TrainConfig:
    """训练配置"""
    # 批次大小
    BATCH_SIZE: int = 8
    # 训练轮数
    EPOCHS: int = 200
    # 学习率
    LEARNING_RATE: float = 1e-4
    # 学习率最小值
    LEARNING_RATE_MIN: float = 1e-6
    # 权重衰减
    WEIGHT_DECAY: float = 1e-4
    # 梯度裁剪阈值
    GRAD_CLIP: float = 1.0
    
    # 多阶段训练设置
    # 阶段1: 基础训练（无攻击）
    STAGE1_EPOCHS: int = 50
    # 阶段2: 轻度攻击
    STAGE2_EPOCHS: int = 100
    # 阶段3: 全部攻击
    STAGE3_EPOCHS: int = 200
    
    # 验证频率（每N个epoch）
    VAL_FREQUENCY: int = 5
    # 保存频率（每N个epoch）
    SAVE_FREQUENCY: int = 10
    
    # 早停耐心值
    EARLY_STOP_PATIENCE: int = 20
    
    # 训练数据路径
    TRAIN_IMAGE_DIR: str = 'data/train/images'
    TRAIN_WATERMARK_DIR: str = 'data/train/watermarks'
    # 验证数据路径
    VAL_IMAGE_DIR: str = 'data/val/images'
    VAL_WATERMARK_DIR: str = 'data/val/watermarks'
    
    # 模型保存路径
    CHECKPOINT_DIR: str = 'checkpoints'
    # 日志保存路径
    LOG_DIR: str = 'logs'
    # 最佳模型文件名
    BEST_MODEL_NAME: str = 'best_model.pth'
    # 最新模型文件名
    LATEST_MODEL_NAME: str = 'latest_model.pth'
    
    # 是否加载预训练模型
    RESUME: bool = True
    # 预训练模型路径
    RESUME_PATH: str = 'checkpoints/best_model.pth'
    
    # 随机种子
    SEED: int = 42
    # 工作线程数
    NUM_WORKERS: int = 4


@dataclass
class EvalConfig:
    """评估配置"""
    # 评估时的攻击类型
    EVAL_ATTACKS: List[str] = field(default_factory=lambda: [
        'none', 'crop', 'rotate', 'scale', 
        'blur', 'noise', 'jpeg', 'combined'
    ])
    
    # 评估指标阈值
    PSNR_THRESHOLD: float = 30.0
    SSIM_THRESHOLD: float = 0.9
    ACCURACY_THRESHOLD: float = 0.9
    
    # 输出目录
    OUTPUT_DIR: str = 'output'


# 创建全局配置实例
image_config = ImageConfig()
dwt_config = DWTConfig()
model_config = ModelConfig()
attack_config = AttackConfig()
loss_config = LossConfig()
train_config = TrainConfig()
eval_config = EvalConfig()


def get_config():
    """获取所有配置的字典"""
    return {
        'image': image_config,
        'dwt': dwt_config,
        'model': model_config,
        'attack': attack_config,
        'loss': loss_config,
        'train': train_config,
        'eval': eval_config
    }


def create_directories():
    """创建必要的目录"""
    dirs = [
        train_config.CHECKPOINT_DIR,
        train_config.LOG_DIR,
        train_config.TRAIN_IMAGE_DIR,
        train_config.TRAIN_WATERMARK_DIR,
        train_config.VAL_IMAGE_DIR,
        train_config.VAL_WATERMARK_DIR,
        eval_config.OUTPUT_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


if __name__ == '__main__':
    # 测试配置
    config = get_config()
    for name, cfg in config.items():
        print(f"\n{name.upper()} CONFIG:")
        for key, value in cfg.__dict__.items():
            print(f"  {key}: {value}")
    
    create_directories()
    print("\n目录创建完成！")
