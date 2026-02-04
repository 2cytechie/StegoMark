# 导入必要的库
import torch

# 项目配置参数

class Config:
    # 数据配置
    DATA_DIR = 'data'
    TRAIN_DIR = 'data/train'
    VAL_DIR = 'data/val'
    
    # 图像配置
    IMAGE_SIZE = 256                # 载体图像尺寸
    WATERMARK_SIZE = 64             # 水印尺寸
    IMAGE_CHANNELS = 3              # 彩色图像
    
    # DWT配置
    DWT_LEVEL = 1                   # DWT分解级别
    DWT_MODE = 'haar'               # DWT小波基
    
    # 水印配置
    WATERMARK_TYPES = 'image'       # image 或 text
    WATERMARK_MODEL = 'best_model_image.pth'    # 水印模型路径
    TEXT_WATERMARK_LENGTH = 64      # 文本水印长度（bits）
    WATERMARK_STRENGTH = 0.05       # 降低水印强度以提高PSNR，同时保持足够的鲁棒性
    GRID_SIZE = 4                   # 减少水印网格大小，降低计算复杂度，同时保持抗裁剪能力
    
    # 模型配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RESUME = False                 # 是否从检查点恢复训练
    RESUME_CHECKPOINT = 'checkpoints/best_model_image.pth'          # 要加载的checkpoint路径
    TRAIN_TYPE = 'image'            # 训练类型（image或text）
    ENCODER_CHANNELS = [32, 64, 128, 256]  # 调整为优化后的通道数
    DECODER_CHANNELS = [256, 128, 64, 32]  # 调整为优化后的通道数
    BATCH_SIZE = 6                  # 增加批次大小，提高训练稳定性和速度
    EPOCHS = 100                    # 增加训练轮数，确保模型充分收敛
    LEARNING_RATE = 2e-4            # 适当提高学习率，加快初始收敛
    WEIGHT_DECAY = 1e-4             # 增大权重衰减，增强正则化效果
    SAVE_INTERVAL = 10              # 保存间隔
    
    # 早停配置
    EARLY_STOPPING = True           # 是否启用早停
    EARLY_STOPPING_PATIENCE = 20    # 增加早停 patience，给模型更多收敛时间
    EARLY_STOPPING_DELTA = 5e-6     # 减小早停 delta，对性能提升更敏感
    
    # 学习率调度器配置
    LR_SCHEDULER_TYPE = 'cosine'    # 学习率调度器类型：cosine, step
    LR_GAMMA = 0.3                  # 适当增大衰减因子，加快后期收敛
    LR_STEP_SIZE = 20               # 增加阶梯式调度器的步长，减少学习率调整频率
    
    # 微调配置
    FINETUNE_LR = 1e-5              # 微调学习率
    
    # 对抗性训练配置
    ATTACK_TRAINING = True          # 暂时关闭对抗性训练以提高PSNR
    # 攻击类型：random, gaussian, jpeg, crop, blur, rotate, scale
    ATTACK_TYPE = ['random']
    ATTACK_TYPE_LEN = 6             # 攻击类型数量
    GAUSSIAN_NOISE_STD = 0.05       # 高斯噪声标准差
    JPEG_QUALITY = 85               # JPEG压缩质量
    MIN_CROP_RATIO = 0.5            # 最小裁剪比例（50%）
    MAX_CROP_RATIO = 0.9            # 最大裁剪比例（90%）
    MAX_ROTATION_ANGLE = 90         # 最大旋转角度（度），范围：-90°至90°
    MIN_SCALE = 0.5                 # 最小缩放比例（0.5）
    MAX_SCALE = 2.0                 # 最大缩放比例（2.0）
    
    # 评估配置
    PSNR_THRESHOLD = 30.0           # PSNR阈值
    SSIM_THRESHOLD = 0.95           # SSIM阈值
    ACCURACY_THRESHOLD = 0.9        # 提取准确率阈值
    
    # 路径配置
    CHECKPOINT_DIR = 'checkpoints'  # 模型检查点目录
    OUTPUT_DIR = 'output'           # 输出目录
    

# 创建全局配置实例
config = Config()
