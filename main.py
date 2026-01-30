# 推理与测试
import torch
from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from config import Config
from PIL import Image
import numpy as np
import argparse
import os

def load_image(image_path):
    """加载图像并转换为张量"""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(Config.DEVICE)

def save_image(tensor, save_path):
    """保存张量为图像"""
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(tensor)
    img.save(save_path)

def text_to_binary(text, length=64):
    """将文本转换为二进制水印"""
    # 文本转二进制
    binary = []
    for char in text:
        # 获取字符的ASCII码并转换为8位二进制
        char_bin = bin(ord(char))[2:].zfill(8)
        binary.extend([int(bit) for bit in char_bin])
    
    # 截断或补零到指定长度
    if len(binary) > length:
        binary = binary[:length]
    else:
        binary.extend([0] * (length - len(binary)))
    
    return torch.tensor(binary, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)

def binary_to_text(binary_tensor):
    """将二进制水印转换为文本"""
    binary = (binary_tensor > 0.5).int().squeeze().tolist()
    text = ""
    
    # 每8位转换为一个字符
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) < 8:
            break
        # 转换为整数
        char_code = int(''.join(map(str, byte)), 2)
        # 忽略非ASCII字符
        if 32 <= char_code <= 126:
            text += chr(char_code)
    
    return text

def image_to_binary(image_path, length=64):
    """将图片转换为二进制水印"""
    # 加载图片并转换为灰度
    img = Image.open(image_path).convert('L')
    # 调整大小以获取足够的信息
    img = img.resize((8, 8))
    img_array = np.array(img)
    
    # 转换为二进制
    binary = []
    for row in img_array:
        for pixel in row:
            # 使用像素值的奇偶性作为二进制位
            binary.append(1 if pixel % 2 == 1 else 0)
    
    # 截断或补零到指定长度
    if len(binary) > length:
        binary = binary[:length]
    else:
        binary.extend([0] * (length - len(binary)))
    
    return torch.tensor(binary, dtype=torch.float32).unsqueeze(0).to(Config.DEVICE)

def binary_to_image(binary_tensor, save_path):
    """将二进制水印转换为图片"""
    binary = (binary_tensor > 0.5).int().squeeze().tolist()
    # 确保有64位
    if len(binary) < 64:
        binary.extend([0] * (64 - len(binary)))
    binary = binary[:64]
    
    # 重塑为8x8
    img_array = np.array(binary).reshape(8, 8) * 255
    img_array = img_array.astype(np.uint8)
    
    # 创建图像并保存
    img = Image.fromarray(img_array, mode='L')
    img.save(save_path)

def embed_watermark(image_path, output_path, watermark, watermark_type='text'):
    """嵌入水印"""
    # 初始化模型
    encoder = WatermarkEncoder().to(Config.DEVICE)
    
    # 加载图像
    img = load_image(image_path)
    print(f"输入图像尺寸: {img.shape[2]}x{img.shape[3]}")
    
    # 处理水印
    if watermark_type == 'text':
        msg = text_to_binary(watermark)
        print(f"嵌入文本: {watermark}")
    else:  # image
        msg = image_to_binary(watermark)
        print(f"嵌入图片水印: {watermark}")
    
    # 嵌入水印
    encoded_img = encoder(img, msg)
    print(f"嵌入后图像尺寸: {encoded_img.shape[2]}x{encoded_img.shape[3]}")
    
    # 保存结果
    save_image(encoded_img, output_path)
    print(f"保存嵌入后图像到: {output_path}")

def extract_watermark(image_path, output_path=None, extract_type='text'):
    """提取水印"""
    # 初始化模型
    decoder = WatermarkDecoder().to(Config.DEVICE)
    
    # 加载图像
    img = load_image(image_path)
    print(f"输入图像尺寸: {img.shape[2]}x{img.shape[3]}")
    
    # 提取水印
    extracted_msg = decoder(img)
    
    # 处理提取结果
    if extract_type == 'text':
        text = binary_to_text(extracted_msg)
        print(f"提取的文本: {text}")
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"保存提取结果到: {output_path}")
    else:  # image
        if output_path:
            binary_to_image(extracted_msg, output_path)
            print(f"保存提取的图片水印到: {output_path}")
        else:
            print("提取图片水印需要指定输出路径")

def test_any_size_image():
    """测试任意尺寸图像的处理"""
    # 初始化模型
    encoder = WatermarkEncoder().to(Config.DEVICE)
    decoder = WatermarkDecoder().to(Config.DEVICE)
    
    # 加载测试图像
    test_images = [
        "img/target.png",
        "img/white_target.png"
    ]
    
    for img_path in test_images:
        print(f"测试图像: {img_path}")
        img = load_image(img_path)
        print(f"输入图像尺寸: {img.shape[2]}x{img.shape[3]}")
        
        # 生成随机水印
        msg = torch.randint(0, 2, (1, 64)).float().to(Config.DEVICE)
        
        # 嵌入水印
        encoded_img = encoder(img, msg)
        print(f"嵌入后图像尺寸: {encoded_img.shape[2]}x{encoded_img.shape[3]}")
        
        # 提取水印
        extracted_msg = decoder(encoded_img)
        
        # 计算水印提取准确率
        msg_pred = (extracted_msg > 0.5).float()
        accuracy = (msg_pred == msg).float().mean().item()
        print(f"水印提取准确率: {accuracy:.4f}")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StegoMark 盲水印工具')
    parser.add_argument('--mode', choices=['embed', 'extract', 'test'], required=True, help='操作模式')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--output', help='输出路径')
    parser.add_argument('--watermark', help='水印内容或路径')
    parser.add_argument('--watermark-type', choices=['text', 'image'], default='text', help='水印类型')
    parser.add_argument('--extract-type', choices=['text', 'image'], default='text', help='提取类型')
    
    args = parser.parse_args()
    
    if args.mode == 'embed':
        if not args.watermark:
            print("错误: 嵌入模式需要指定水印内容")
        elif not args.output:
            print("错误: 嵌入模式需要指定输出路径")
        else:
            embed_watermark(args.image, args.output, args.watermark, args.watermark_type)
    
    elif args.mode == 'extract':
        if not args.output:
            print("警告: 未指定输出路径，仅显示结果")
        extract_watermark(args.image, args.output, args.extract_type)
    
    elif args.mode == 'test':
        test_any_size_image()