import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from config import config
from src.models import WatermarkModel

class WatermarkEmbedding:
    """
    水印嵌入算法模块
    实现基于DWT和深度学习的水印嵌入
    """
    
    def __init__(self, watermark_type='image'):
        """
        初始化水印嵌入模块
        
        Args:
            watermark_type: 水印类型，'image'或'text'
        """
        self.watermark_type = watermark_type
        self.model = WatermarkModel(watermark_type=watermark_type)
        self.model.to(config.DEVICE)
    
    def preprocess_image_watermark(self, watermark_path):
        """
        预处理图像水印
        
        Args:
            watermark_path: 水印图像路径
            
        Returns:
            watermark_tensor: 预处理后的水印张量，形状为 (1, 1, 64, 64)
        """
        # 1. 加载水印图像
        watermark = Image.open(watermark_path).convert('L')  # 转换为灰度图
        
        # 2. 调整尺寸
        watermark = watermark.resize((config.WATERMARK_SIZE, config.WATERMARK_SIZE), Image.LANCZOS)
        
        # 3. 转换为张量
        watermark_array = np.array(watermark) / 255.0
        watermark_tensor = torch.tensor(watermark_array, dtype=torch.float32)
        watermark_tensor = watermark_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
        
        return watermark_tensor
    
    def preprocess_text_watermark(self, text):
        """
        预处理文本水印
        
        Args:
            text: 文本水印内容
            
        Returns:
            watermark_tensor: 预处理后的水印张量，形状为 (1, C, 64, 64)
        """
        # 1. 将文本转换为二进制序列
        binary = ''.join(format(ord(c), '08b') for c in text)
        
        # 2. 补齐到指定长度
        if len(binary) < config.TEXT_WATERMARK_LENGTH:
            binary = binary.ljust(config.TEXT_WATERMARK_LENGTH, '0')
        else:
            binary = binary[:config.TEXT_WATERMARK_LENGTH]
        
        # 3. 转换为数组
        binary_array = np.array([int(bit) for bit in binary])
        
        # 4. 重塑为2D张量
        watermark_size = config.WATERMARK_SIZE
        channels = config.TEXT_WATERMARK_LENGTH // (watermark_size * watermark_size)
        
        if config.TEXT_WATERMARK_LENGTH % (watermark_size * watermark_size) != 0:
            channels += 1
        
        # 补齐到完整的通道数
        padding = channels * watermark_size * watermark_size - len(binary_array)
        if padding > 0:
            binary_array = np.pad(binary_array, (0, padding), 'constant')
        
        # 重塑为 (C, H, W)
        watermark_array = binary_array.reshape(channels, watermark_size, watermark_size)
        watermark_tensor = torch.tensor(watermark_array, dtype=torch.float32)
        watermark_tensor = watermark_tensor.unsqueeze(0)  # (1, C, 64, 64)
        
        return watermark_tensor
    
    def preprocess_cover_image(self, image_path):
        """
        预处理载体图像
        
        Args:
            image_path: 载体图像路径
            
        Returns:
            image_tensor: 预处理后的图像张量，形状为 (1, 3, 256, 256)
            original_size: 原始图像尺寸 (width, height)
        """
        # 1. 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 2. 记录原始尺寸
        original_size = image.size  # (width, height)
        
        # 3. 调整尺寸
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        
        # 4. 转换为张量
        image_array = np.array(image) / 255.0
        image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 256, 256)
        
        return image_tensor, original_size
    
    def embed(self, cover_image_path, watermark_input, output_path=None):
        """
        嵌入水印
        
        Args:
            cover_image_path: 载体图像路径
            watermark_input: 水印输入（图像路径或文本）
            output_path: 输出含水印图像路径
            
        Returns:
            watermarked_image: 含水印图像张量
        """
        # 1. 预处理载体图像
        cover_tensor, original_size = self.preprocess_cover_image(cover_image_path)
        cover_tensor = cover_tensor.to(config.DEVICE)
        
        # 2. 预处理水印
        if self.watermark_type == 'image':
            watermark_tensor = self.preprocess_image_watermark(watermark_input)
        else:
            watermark_tensor = self.preprocess_text_watermark(watermark_input)
        
        watermark_tensor = watermark_tensor.to(config.DEVICE)
        
        # 3. 嵌入水印
        with torch.no_grad():
            watermarked_image = self.model.embed(cover_tensor, watermark_tensor)
        
        # 4. 保存结果
        if output_path:
            self.save_image(watermarked_image, output_path, original_size)
        
        return watermarked_image
    
    def save_image(self, tensor, path, original_size=None):
        """
        保存张量为图像
        
        Args:
            tensor: 图像张量
            path: 保存路径
            original_size: 原始图像尺寸 (width, height)，如果提供则还原为原始尺寸
        """
        # 转换为numpy数组
        image_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image_array = (image_array * 255).astype(np.uint8)
        
        # 创建图像
        image = Image.fromarray(image_array)
        
        # 如果提供了原始尺寸，还原为原始尺寸
        if original_size:
            image = image.resize(original_size, Image.LANCZOS)
        
        # 保存图像
        image.save(path)
    
    def load_model(self, checkpoint_path):
        """
        加载模型权重
        
        Args:
            checkpoint_path: 模型权重路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        # 使用正确的键名 model_state_dict
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    
    def save_model(self, checkpoint_path, epoch, loss):
        """
        保存模型权重
        
        Args:
            checkpoint_path: 保存路径
            epoch: 当前 epoch
            loss: 当前损失
        """
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'state_dict': self.model.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

class BatchEmbedding:
    """
    批量水印嵌入类
    用于训练过程中的批量处理
    """
    
    def __init__(self, watermark_type='image'):
        """
        初始化批量嵌入器
        
        Args:
            watermark_type: 水印类型
        """
        self.watermark_type = watermark_type
        self.embedding = WatermarkEmbedding(watermark_type=watermark_type)
    
    def preprocess_batch(self, cover_images, watermarks):
        """
        预处理批量数据
        
        Args:
            cover_images: 批量载体图像
            watermarks: 批量水印
            
        Returns:
            cover_tensors: 预处理后的载体图像张量
            watermark_tensors: 预处理后的水印张量
        """
        cover_tensors = []
        watermark_tensors = []
        
        for cover_img, watermark in zip(cover_images, watermarks):
            # 预处理载体图像
            if isinstance(cover_img, str):
                cover_tensor = self.embedding.preprocess_cover_image(cover_img)
            else:
                cover_tensor = cover_img
            
            # 预处理水印
            if self.watermark_type == 'image':
                if isinstance(watermark, str):
                    watermark_tensor = self.embedding.preprocess_image_watermark(watermark)
                else:
                    watermark_tensor = watermark
            else:
                watermark_tensor = self.embedding.preprocess_text_watermark(watermark)
            
            cover_tensors.append(cover_tensor)
            watermark_tensors.append(watermark_tensor)
        
        # 拼接为批量张量
        cover_tensors = torch.cat(cover_tensors, dim=0)
        watermark_tensors = torch.cat(watermark_tensors, dim=0)
        
        return cover_tensors, watermark_tensors
    
    def embed_batch(self, cover_tensors, watermark_tensors):
        """
        批量嵌入水印
        
        Args:
            cover_tensors: 载体图像张量
            watermark_tensors: 水印张量
            
        Returns:
            watermarked_images: 含水印图像张量
        """
        cover_tensors = cover_tensors.to(config.DEVICE)
        watermark_tensors = watermark_tensors.to(config.DEVICE)
        
        with torch.no_grad():
            watermarked_images = self.embedding.model.embed(cover_tensors, watermark_tensors)
        
        return watermarked_images