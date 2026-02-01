# 导入必要的库
import torch

# 项目配置参数

class Config:
    # 数据配置
    DATA_DIR = 'data'
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    
    # 图像配置
    IMAGE_SIZE = 512  # 载体图像尺寸
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
    BATCH_SIZE = 20                 # 批次大小
    EPOCHS = 10                     # 训练次数
    LEARNING_RATE = 1e-4            # 学习率
    WEIGHT_DECAY = 1e-5             # 权重衰减
    SAVE_INTERVAL = 5               # 保存模型间隔（个epoch）
    
    # 早停配置
    EARLY_STOPPING = True           # 是否启用早停
    EARLY_STOPPING_PATIENCE = 10    # 早停 patience（个epoch）
    EARLY_STOPPING_DELTA = 1e-6     # 早停 delta
    
    # 学习率调度器配置
    LR_SCHEDULER_TYPE = 'cosine'    # 学习率调度器类型：cosine, step
    LR_GAMMA = 0.1                  # 学习率衰减因子
    LR_STEP_SIZE = 10               # 阶梯式调度器的步长
    
    # 微调配置
    FINETUNE_LR = 1e-5              # 微调学习率
    
    # 对抗性训练配置
    ATTACK_TRAINING = True          # 是否开启对抗性训练
    
    # 攻击类型：random, gaussian, jpeg, crop, blur, rotate, scale
    ATTACK_TYPE = ['random','random']
    ATTACK_TYPE_LEN = 6             # 攻击类型数量
    GAUSSIAN_NOISE_STD = 0.01       # 高斯噪声标准差
    JPEG_QUALITY = 85               # JPEG压缩质量
    CROP_SIZE = 128                 # 裁剪尺寸
    MAX_ROTATION_ANGLE = 15         # 最大旋转角度（度）
    MIN_SCALE = 0.8                 # 最小缩放比例
    MAX_SCALE = 1.2                 # 最大缩放比例
    
    # 评估配置
    PSNR_THRESHOLD = 30.0           # PSNR阈值
    SSIM_THRESHOLD = 0.95           # SSIM阈值
    
    # 路径配置
    CHECKPOINT_DIR = 'checkpoints'  # 模型检查点目录
    OUTPUT_DIR = 'output'           # 输出目录
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模型加载配置
    RESUME = True                  # 是否从checkpoint继续训练
    RESUME_CHECKPOINT = 'checkpoints/model_image_epoch_5.pth'          # 要加载的checkpoint路径

# 创建全局配置实例
config = Config()