# 超参数配置
import torch

class Config:
    EPOCHS = 5
    MSG_CHANNELS = 1  # 假设水印是二进制流，映射为特征图
    MSG_LENGTH = 64   # 水印比特数
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 对抗性训练配置
    ADVERSARIAL_TRAINING = True  # 是否启用对抗性训练
    ATTACK_TYPE = "PGD"  # 对抗性攻击类型: "FGSM" 或 "PGD"
    EPSILON = 0.03  # 扰动大小
    PGD_ITERATIONS = 10  # PGD攻击迭代次数
    PGD_STEP_SIZE = 0.01  # PGD攻击步长