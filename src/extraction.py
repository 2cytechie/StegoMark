import torch
import numpy as np
from PIL import Image
from config import config
from src.models import WatermarkModel

class WatermarkExtraction:
    """
    水印提取算法模块
    实现基于DWT和深度学习的水印提取
    """
    
    def __init__(self, watermark_type='image'):
        """
        初始化水印提取模块
        
        Args:
            watermark_type: 水印类型，'image'或'text'
        """
        self.watermark_type = watermark_type
        self.model = WatermarkModel(watermark_type=watermark_type)
        self.model.to(config.DEVICE)
    
    def preprocess_watermarked_image(self, image_path):
        """
        预处理含水印图像
        
        Args:
            image_path: 含水印图像路径
            
        Returns:
            image_tensor: 预处理后的图像张量，形状为 (1, 3, 256, 256)
        """
        # 1. 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 2. 调整尺寸
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
        
        # 3. 转换为张量
        image_array = np.array(image) / 255.0
        image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1)
        image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 256, 256)
        
        return image_tensor
    
    def extract(self, watermarked_image_path, output_path=None):
        """
        提取水印
        
        Args:
            watermarked_image_path: 含水印图像路径
            output_path: 输出水印路径
            
        Returns:
            extracted_watermark: 提取的水印（图像或文本）
        """
        # 1. 预处理含水印图像
        watermarked_tensor = self.preprocess_watermarked_image(watermarked_image_path)
        watermarked_tensor = watermarked_tensor.to(config.DEVICE)
        
        # 2. 提取水印
        with torch.no_grad():
            watermark_tensor = self.model.extract(watermarked_tensor)
        
        # 3. 后处理
        if self.watermark_type == 'image':
            extracted_watermark = self.postprocess_image_watermark(watermark_tensor, output_path)
        else:
            extracted_watermark = self.postprocess_text_watermark(watermark_tensor)
        
        return extracted_watermark
    
    def postprocess_image_watermark(self, watermark_tensor, output_path=None):
        """
        后处理图像水印
        
        Args:
            watermark_tensor: 提取的水印张量
            output_path: 输出路径
            
        Returns:
            watermark_image: 水印图像
        """
        # 1. 转换为numpy数组
        watermark_array = watermark_tensor.squeeze(0).squeeze(0).cpu().numpy()
        watermark_array = (watermark_array * 255).astype(np.uint8)
        
        # 2. 创建图像
        watermark_image = Image.fromarray(watermark_array, mode='L')
        
        # 3. 保存结果
        if output_path:
            watermark_image.save(output_path)
        
        return watermark_image
    
    def postprocess_text_watermark(self, watermark_tensor):
        """
        后处理文本水印
        
        Args:
            watermark_tensor: 提取的水印张量
            
        Returns:
            text: 提取的文本水印
        """
        # 1. 转换为numpy数组并二值化
        watermark_array = watermark_tensor.squeeze(0).cpu().numpy()
        binary_array = (watermark_array > 0.5).astype(int)
        
        # 2. 展平为一维数组
        binary_array = binary_array.flatten()
        
        # 3. 截取指定长度
        binary_array = binary_array[:config.TEXT_WATERMARK_LENGTH]
        
        # 4. 分组为8位二进制数
        binary_str = ''.join(map(str, binary_array))
        chars = []
        
        for i in range(0, len(binary_str), 8):
            byte = binary_str[i:i+8]
            if len(byte) == 8:
                char_code = int(byte, 2)
                if 32 <= char_code <= 126:  # 可打印ASCII字符
                    chars.append(chr(char_code))
        
        # 5. 拼接为文本
        text = ''.join(chars).strip()
        
        return text
    
    def extract_batch(self, watermarked_image_paths):
        """
        批量提取水印
        
        Args:
            watermarked_image_paths: 含水印图像路径列表
            
        Returns:
            extracted_watermarks: 提取的水印列表
        """
        extracted_watermarks = []
        
        for image_path in watermarked_image_paths:
            watermark = self.extract(image_path)
            extracted_watermarks.append(watermark)
        
        return extracted_watermarks
    
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
    
    def evaluate_extraction_accuracy(self, extracted_watermark, original_watermark):
        """
        评估提取准确率
        
        Args:
            extracted_watermark: 提取的水印
            original_watermark: 原始水印
            
        Returns:
            accuracy: 提取准确率
        """
        if self.watermark_type == 'image':
            # 图像水印准确率
            extracted_array = np.array(extracted_watermark)
            original_array = np.array(Image.open(original_watermark).convert('L').resize((config.WATERMARK_SIZE, config.WATERMARK_SIZE)))
            
            # 二值化
            extracted_binary = (extracted_array > 128).astype(int)
            original_binary = (original_array > 128).astype(int)
            
            # 计算准确率
            accuracy = np.mean(extracted_binary == original_binary)
        else:
            # 文本水印准确率
            if isinstance(original_watermark, str):
                # 计算字符匹配率
                min_len = min(len(extracted_watermark), len(original_watermark))
                correct = sum(c1 == c2 for c1, c2 in zip(extracted_watermark[:min_len], original_watermark[:min_len]))
                accuracy = correct / max(len(extracted_watermark), len(original_watermark))
            else:
                accuracy = 0.0
        
        return accuracy

class BatchExtraction:
    """
    批量水印提取类
    用于批量处理水印提取
    """
    
    def __init__(self, watermark_type='image'):
        """
        初始化批量提取器
        
        Args:
            watermark_type: 水印类型
        """
        self.watermark_type = watermark_type
        self.extraction = WatermarkExtraction(watermark_type=watermark_type)
    
    def preprocess_batch(self, watermarked_images):
        """
        预处理批量含水印图像
        
        Args:
            watermarked_images: 批量含水印图像路径
            
        Returns:
            watermarked_tensors: 预处理后的张量
        """
        tensors = []
        
        for image_path in watermarked_images:
            tensor = self.extraction.preprocess_watermarked_image(image_path)
            tensors.append(tensor)
        
        return torch.cat(tensors, dim=0)
    
    def extract_batch(self, watermarked_tensors):
        """
        批量提取水印
        
        Args:
            watermarked_tensors: 含水印图像张量
            
        Returns:
            extracted_watermarks: 提取的水印张量
        """
        watermarked_tensors = watermarked_tensors.to(config.DEVICE)
        
        with torch.no_grad():
            extracted_watermarks = self.extraction.model.extract(watermarked_tensors)
        
        return extracted_watermarks
    
    def process_batch_results(self, extracted_watermarks, output_dir=None):
        """
        处理批量提取结果
        
        Args:
            extracted_watermarks: 提取的水印张量
            output_dir: 输出目录
            
        Returns:
            results: 处理后的结果列表
        """
        results = []
        
        for i, watermark_tensor in enumerate(extracted_watermarks):
            if self.watermark_type == 'image':
                if output_dir:
                    output_path = f"{output_dir}/extracted_watermark_{i}.png"
                else:
                    output_path = None
                result = self.extraction.postprocess_image_watermark(watermark_tensor.unsqueeze(0), output_path)
            else:
                result = self.extraction.postprocess_text_watermark(watermark_tensor.unsqueeze(0))
            
            results.append(result)
        
        return results