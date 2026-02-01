# 导入必要的库
import torch

# 项目配置参数

class Config:
    # 数据配置
    DATA_DIR = 'data'
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    
    # 图像配置
    IMAGE_SIZE = 256  # 载体图像尺寸
    WATERMARK_SIZE = 64  # 水印尺寸
    IMAGE_CHANNELS = 3  # 彩色图像
    
    # DWT配置
    DWT_LEVEL = 1  # DWT分解级别
    DWT_MODE = 'haar'  # DWT小波基
    
    # 水印配置
    WATERMARK_TYPES = ['image', 'text']
    TEXT_WATERMARK_LENGTH = 64  # 文本水印长度（bits）
    
    # 模型配置
    TRAIN_TYPE = 'image'            # 训练类型（image或text）
    ENCODER_CHANNELS = [64, 128, 256, 512]
    DECODER_CHANNELS = [512, 256, 128, 64]
    BATCH_SIZE = 24                  # 批次大小
    EPOCHS = 5                     # 训练次数
    LEARNING_RATE = 1e-4            # 学习率
    WEIGHT_DECAY = 1e-5             # 权重衰减
    SAVE_INTERVAL = 5              # 保存模型间隔（个epoch）
    
    # 对抗性训练配置
    ADVERSARIAL_TRAINING = True     # 是否开启对抗性训练
    ATTACK_EPS = 0.03               # 攻击步长
    ATTACK_ITERATIONS = 10          # 攻击迭代次数
    
    # 噪声层配置
    GAUSSIAN_NOISE_STD = 0.01       # 高斯噪声标准差
    JPEG_QUALITY = 85               # JPEG压缩质量
    
    # 评估配置
    PSNR_THRESHOLD = 30.0           # PSNR阈值
    SSIM_THRESHOLD = 0.95           # SSIM阈值
    
    # 路径配置
    CHECKPOINT_DIR = 'checkpoints'  # 模型检查点目录
    OUTPUT_DIR = 'output'           # 输出目录
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型加载配置
    RESUME = False                 # 是否从checkpoint继续训练
    RESUME_CHECKPOINT = 'checkpoints/model_image_epoch_10.pth'          # 要加载的checkpoint路径

# 创建全局配置实例
config = Config()