"""
水印嵌入测试脚本
"""
import os
import sys
sys.path.insert(0, r'd:\实验室图像隐水印项目\StegoMark')

import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

from configs.config import model_config, training_config
from src.models import create_model
from src.utils import calculate_psnr, calculate_ssim
from src.data.transforms import ImagePreprocessor


def load_image(image_path):
    """加载图片"""
    image = Image.open(image_path).convert('RGB')
    return image


def save_image(tensor, save_path):
    """保存图片张量"""
    # 反归一化到[0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为PIL Image
    transform = T.ToPILImage()
    image = transform(tensor.cpu())
    
    # 保存
    image.save(save_path)
    print(f"保存图片: {save_path}")


def embed_watermark(model, image_path, watermark_path, device):
    """
    嵌入水印
    
    Args:
        model: 模型
        image_path: 目标图片路径
        watermark_path: 水印图片路径
        device: 设备
        
    Returns:
        watermarked_image: 含水印图片张量
        original_image: 原始图片张量
    """
    model.eval()
    
    # 加载图片
    image = load_image(image_path)
    watermark = load_image(watermark_path)
    
    # 预处理
    preprocessor = ImagePreprocessor(watermark_size=64)
    
    image_tensor = preprocessor.preprocess_target(image).unsqueeze(0).to(device)
    watermark_tensor = preprocessor.preprocess_watermark(watermark)
    
    # 平铺水印
    _, H, W = image_tensor.shape[1:]
    watermark_tiled = preprocessor.tile_watermark(watermark_tensor, (H, W))
    watermark_tiled = watermark_tiled.unsqueeze(0).to(device)
    
    # 嵌入水印
    with torch.no_grad():
        watermarked_image = model.encode(image_tensor, watermark_tiled)
    
    return watermarked_image.squeeze(0), image_tensor.squeeze(0)


def main(args):
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(model_config)
    model = model.to(device)
    
    # 加载权重
    if args.checkpoint:
        print(f"加载检查点: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 尝试加载最佳模型
        best_model_path = os.path.join(training_config.checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"加载最佳模型: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("警告: 未找到模型权重，使用随机初始化的模型")
    
    # 嵌入水印
    print(f"\n嵌入水印:")
    print(f"  目标图片: {args.image}")
    print(f"  水印图片: {args.watermark}")
    
    watermarked_image, original_image = embed_watermark(
        model, args.image, args.watermark, device
    )
    
    # 计算指标
    psnr = calculate_psnr(original_image, watermarked_image)
    ssim = calculate_ssim(original_image, watermarked_image)
    
    print(f"\n嵌入质量:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    # 保存结果
    if args.output:
        save_image(watermarked_image, args.output)
    else:
        # 自动生成输出路径
        image_path = Path(args.image)
        output_path = image_path.parent / f"{image_path.stem}_watermarked{image_path.suffix}"
        save_image(watermarked_image, str(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='嵌入水印')
    parser.add_argument('--image', type=str, required=True, help='目标图片路径')
    parser.add_argument('--watermark', type=str, required=True, help='水印图片路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--output', type=str, default=None, help='输出图片路径')
    args = parser.parse_args()
    
    main(args)
