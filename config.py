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
    WATERMARK_STRENGTH = 0.05       # 平衡水印强度以获得更好的PSNR和提取准确率
    GRID_SIZE = 4                   # 水印网格大小（GRID_SIZE^2 = WATERMARK_COPIES），用于抗裁剪攻击
    
    # 模型配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    RESUME = False                 # 是否从检查点恢复训练
    RESUME_CHECKPOINT = 'checkpoints/best_model_image.pth'          # 要加载的checkpoint路径
    TRAIN_TYPE = 'image'            # 训练类型（image或text）
    ENCODER_CHANNELS = [64, 128, 256, 512]
    DECODER_CHANNELS = [512, 256, 128, 64]
    BATCH_SIZE = 16                 # 批次大小
    EPOCHS = 100                    # 训练次数
    LEARNING_RATE = 1e-4            # 学习率
    WEIGHT_DECAY = 1e-5             # 权重衰减
    SAVE_INTERVAL = 10              # 保存模型间隔（个epoch）
    
    # 早停配置
    EARLY_STOPPING = True           # 是否启用早停
    EARLY_STOPPING_PATIENCE = 15    # 增加早停 patience
    EARLY_STOPPING_DELTA = 1e-5     # 增大早停 delta以容忍更小的损失波动
    
    # 学习率调度器配置
    LR_SCHEDULER_TYPE = 'cosine'    # 学习率调度器类型：cosine, step
    LR_GAMMA = 0.5                  # 学习率衰减因子（减小衰减幅度）
    LR_STEP_SIZE = 15               # 阶梯式调度器的步长（增加步长）
    
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
    
    # 路径配置
    CHECKPOINT_DIR = 'checkpoints'  # 模型检查点目录
    OUTPUT_DIR = 'output'           # 输出目录
    

# 创建全局配置实例
config = Config()
