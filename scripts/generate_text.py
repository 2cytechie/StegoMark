import os
import random
import string
from datetime import datetime

class TextWatermarkGenerator:
    """
    文本水印生成器
    生成多样性的文本内容作为水印
    """
    
    def __init__(self, output_dir='data/train/watermark_txt'):
        """
        初始化文本水印生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_random_string(self, length=10):
        """
        生成随机字符串
        
        Args:
            length: 字符串长度
            
        Returns:
            random_str: 随机字符串
        """
        characters = string.ascii_letters + string.digits + string.punctuation
        random_str = ''.join(random.choice(characters) for _ in range(length))
        return random_str
    
    def generate_timestamp(self):
        """
        生成时间戳
        
        Returns:
            timestamp: 时间戳字符串
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        return timestamp
    
    def generate_serial_number(self, prefix='ID', number=1):
        """
        生成序列号
        
        Args:
            prefix: 前缀
            number: 编号
            
        Returns:
            serial_number: 序列号
        """
        serial_number = f"{prefix}-{number:08d}"
        return serial_number
    
    def generate_combination(self):
        """
        生成组合文本
        
        Returns:
            combination: 组合文本
        """
        components = [
            self.generate_random_string(8),
            self.generate_timestamp(),
            self.generate_serial_number()
        ]
        combination = ' | '.join(components)
        return combination
    
    def generate_texts(self, num_texts=100, text_types=None):
        """
        生成多个文本水印
        
        Args:
            num_texts: 生成文本数量
            text_types: 文本类型列表
        """
        if text_types is None:
            text_types = ['random', 'timestamp', 'serial', 'combination']
        
        for i in range(num_texts):
            # 随机选择文本类型
            text_type = random.choice(text_types)
            
            # 生成文本
            if text_type == 'random':
                text = self.generate_random_string(random.randint(5, 20))
            elif text_type == 'timestamp':
                text = self.generate_timestamp()
            elif text_type == 'serial':
                text = self.generate_serial_number(number=i+1)
            else:  # combination
                text = self.generate_combination()
            
            # 保存文本
            filename = f"watermark_{i+1}.txt"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"Generated text watermark: {filepath}")
    
    def generate_custom_texts(self, texts, output_dir=None):
        """
        生成自定义文本水印
        
        Args:
            texts: 自定义文本列表
            output_dir: 输出目录
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, text in enumerate(texts):
            filename = f"custom_watermark_{i+1}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"Generated custom text watermark: {filepath}")

if __name__ == "__main__":
    # 生成训练集文本水印
    train_generator = TextWatermarkGenerator('data/train/watermark_txt')
    train_generator.generate_texts(num_texts=500)
    
    # 生成验证集文本水印
    val_generator = TextWatermarkGenerator('data/val/watermark_txt')
    val_generator.generate_texts(num_texts=100)
    
    # 生成一些自定义文本水印
    custom_texts = [
        "StegoMark Project 2024",
        "Watermark Embedding System",
        "Deep Learning + DWT",
        "Secure Digital Watermarking",
        "Robust Against Attacks"
    ]
    
    train_generator.generate_custom_texts(custom_texts, 'data/train/watermark_txt')
    val_generator.generate_custom_texts(custom_texts, 'data/val/watermark_txt')
    
    print("Text watermark generation completed!")