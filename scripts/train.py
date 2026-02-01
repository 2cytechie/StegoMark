import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.models import WatermarkModel
from src.adversarial import AdversarialTrainer
from src.embedding import BatchEmbedding

class WatermarkDataset(Dataset):
    """
    水印数据集
    """
    
    def __init__(self, root_dir, watermark_type='image', transform=None):
        """
        初始化数据集
        
        Args:
            root_dir: 根目录
            watermark_type: 水印类型
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.watermark_type = watermark_type
        self.transform = transform
        
        # 加载图像路径
        self.image_paths = []
        self.watermark_paths = []
        
        image_dir = os.path.join(root_dir, 'images')
        if watermark_type == 'image':
            watermark_dir = os.path.join(root_dir, 'watermark_img')
            watermark_files = [f for f in os.listdir(watermark_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for img_name in os.listdir(image_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(image_dir, img_name)
                    # 随机选择一个水印文件
                    if watermark_files:
                        watermark_file = np.random.choice(watermark_files)
                        watermark_path = os.path.join(watermark_dir, watermark_file)
                        self.image_paths.append(img_path)
                        self.watermark_paths.append(watermark_path)
        else:
            watermark_dir = os.path.join(root_dir, 'watermark_txt')
            watermark_files = [f for f in os.listdir(watermark_dir) if f.endswith('.txt')]
            for img_name in os.listdir(image_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(image_dir, img_name)
                    # 随机选择一个文本水印
                    watermark_file = np.random.choice(watermark_files)
                    watermark_path = os.path.join(watermark_dir, watermark_file)
                    self.image_paths.append(img_path)
                    self.watermark_paths.append(watermark_path)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            image: 载体图像
            watermark: 水印
        """
        # 加载载体图像
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        image = np.array(image) / 255.0
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        # 加载水印
        watermark_path = self.watermark_paths[idx]
        if self.watermark_type == 'image':
            watermark = Image.open(watermark_path).convert('L')
            watermark = watermark.resize((config.WATERMARK_SIZE, config.WATERMARK_SIZE), Image.LANCZOS)
            watermark = np.array(watermark) / 255.0
            watermark = torch.tensor(watermark, dtype=torch.float32).unsqueeze(0)
        else:
            with open(watermark_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            # 文本转二进制
            binary = ''.join(format(ord(c), '08b') for c in text)
            binary = binary.ljust(config.TEXT_WATERMARK_LENGTH, '0')[:config.TEXT_WATERMARK_LENGTH]
            binary_array = np.array([int(bit) for bit in binary])
            # 重塑为 (C, H, W)
            channels = config.TEXT_WATERMARK_LENGTH // (config.WATERMARK_SIZE * config.WATERMARK_SIZE)
            if config.TEXT_WATERMARK_LENGTH % (config.WATERMARK_SIZE * config.WATERMARK_SIZE) != 0:
                channels += 1
            padding = channels * config.WATERMARK_SIZE * config.WATERMARK_SIZE - len(binary_array)
            if padding > 0:
                binary_array = np.pad(binary_array, (0, padding), 'constant')
            watermark_array = binary_array.reshape(channels, config.WATERMARK_SIZE, config.WATERMARK_SIZE)
            watermark = torch.tensor(watermark_array, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            watermark = self.transform(watermark)
        
        return image, watermark

def train():
    """
    训练水印模型
    """
    # 配置
    watermark_type = config.TRAIN_TYPE
    
    # 创建数据集
    train_dataset = WatermarkDataset(config.TRAIN_DIR, watermark_type=watermark_type)
    val_dataset = WatermarkDataset(config.VAL_DIR, watermark_type=watermark_type)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 启用GPU加速
    device = torch.device(config.DEVICE)
    config.DEVICE = device
    print(f"Training on device: {device}")
    
    # 初始化模型，直接在GPU上创建
    model = WatermarkModel(watermark_type=watermark_type, device=device)
    model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # 对抗性训练器
    trainer = AdversarialTrainer(model, optimizer, criterion)
    
    # 加载模型
    start_epoch = 0
    if config.RESUME and config.RESUME_CHECKPOINT:
        if os.path.exists(config.RESUME_CHECKPOINT):
            print(f"Loading checkpoint from {config.RESUME_CHECKPOINT}")
            checkpoint = torch.load(config.RESUME_CHECKPOINT, map_location=device)
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载训练状态
            start_epoch = checkpoint['epoch'] + 1
            
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print(f"Checkpoint file not found: {config.RESUME_CHECKPOINT}")
            print("Starting training from scratch")
    
    # 训练循环
    for epoch in range(start_epoch, config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        
        # 训练
        train_loss = trainer.train_epoch(train_loader, config.ADVERSARIAL_TRAINING)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        val_loss, val_accuracy = trainer.validate(val_loader)
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # 保存模型
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_{watermark_type}_epoch_{epoch+1}.pth")
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_accuracy
            }, checkpoint_path)
            print(f"Saved model to {checkpoint_path}")
        
        print("=" * 60)

if __name__ == "__main__":
    train()