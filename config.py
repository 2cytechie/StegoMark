# 超参数配置
import torch

class Config:
    MSG_CHANNELS = 1  # 假设水印是二进制流，映射为特征图
    MSG_LENGTH = 64   # 水印比特数
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")