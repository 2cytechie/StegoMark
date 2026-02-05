"""
水印嵌入脚本
用于将水印嵌入到目标图像中
"""

import torch
import argparse
import os
from PIL import Image

from config import train_config, dwt_config, image_config
from models import create_model
from watermark_utils import (
    preprocess_watermark, preprocess_watermark_to_64x64,
    tensor_to_pil, pil_to_tensor, WatermarkPreprocessor,
    resize_to_original, get_image_info
)
from utils import load_checkpoint, get_device, calculate_psnr, calculate_ssim


def embed_watermark(image_path: str,
                   watermark_path: str,
                   model_path: str,
                   output_path: str,
                   alpha: float = None):
    """
    嵌入水印到图像

    Args:
        image_path: 目标图像路径
        watermark_path: 水印图像路径
        model_path: 模型checkpoint路径
        output_path: 输出图像路径
        alpha: 嵌入强度（None则使用配置值）
    """
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")

    # 使用默认alpha
    if alpha is None:
        alpha = dwt_config.BASE_ALPHA

    print(f"\n加载模型: {model_path}")
    from config import model_config
    model = create_model(model_config, device)

    # 加载checkpoint
    if os.path.exists(model_path):
        epoch, metrics = load_checkpoint(model_path, model, device=device)
        print(f"模型加载完成 (epoch {epoch})")
        if metrics:
            print(f"模型指标: PSNR={metrics.get('psnr', 0):.2f}dB, "
                  f"SSIM={metrics.get('ssim', 0):.4f}, "
                  f"Acc(RGB)={metrics.get('watermark_acc', 0):.4f}")
    else:
        print(f"警告: 模型文件不存在: {model_path}")
        print("使用未训练的模型（结果可能不理想）")

    model.eval()

    # 加载并预处理目标图像
    print(f"\n加载目标图像: {image_path}")
    image = Image.open(image_path).convert('RGB')

    # 记录原始图像信息
    original_info = get_image_info(image)
    original_size = original_info['size']
    original_width, original_height = original_size
    print(f"原始图像尺寸: {original_size} (宽x高: {original_width}x{original_height})")

    # 将图像缩放到TARGET_SIZE x TARGET_SIZE的正方形内进行嵌入
    # 注意：这里使用正方形尺寸进行嵌入，因为深度学习模型需要固定输入尺寸
    image_resized = image.resize(
        (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
        Image.LANCZOS
    )
    image_tensor = pil_to_tensor(image_resized).unsqueeze(0).to(device)

    # 加载并预处理水印
    print(f"加载水印: {watermark_path}")
    watermark_tensor = preprocess_watermark(
        watermark_path,
        watermark_size=image_config.WATERMARK_SIZE,
        target_size=image_config.TARGET_SIZE
    ).unsqueeze(0).to(device)

    # 嵌入水印
    print(f"\n嵌入水印 (alpha={alpha})...")
    with torch.no_grad():
        watermarked_tensor = model.embed(image_tensor, watermark_tensor, alpha=alpha)

    # 转换回PIL图像
    watermarked_pil = tensor_to_pil(watermarked_tensor.squeeze(0))

    # 将嵌入水印后的图像调整回原始尺寸
    # 使用LANCZOS重采样算法保持图像质量，防止变形
    watermarked_pil = resize_to_original(
        watermarked_pil,
        original_size,
        resample=Image.LANCZOS
    )

    # 验证最终尺寸与原始尺寸一致
    final_size = watermarked_pil.size
    final_width, final_height = final_size
    if final_size != original_size:
        raise ValueError(f"尺寸验证失败: 期望 {original_size}, 实际 {final_size}")
    print(f"最终图像尺寸: {final_size} (与原始尺寸一致，无变形)")

    # 保存结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    watermarked_pil.save(output_path, quality=95)
    print(f"\n嵌入水印的图像已保存: {output_path}")

    # 计算并显示质量指标（在256x256尺寸下计算）
    watermarked_resized = watermarked_pil.resize(
        (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
        Image.LANCZOS
    )
    watermarked_for_metric = pil_to_tensor(watermarked_resized)

    psnr = calculate_psnr(image_tensor.squeeze(0), watermarked_for_metric)
    ssim = calculate_ssim(image_tensor.squeeze(0), watermarked_for_metric)

    print(f"\n质量指标:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")

    return output_path


def batch_embed(image_dir: str,
               watermark_path: str,
               model_path: str,
               output_dir: str,
               alpha: float = None):
    """
    批量嵌入水印
    
    Args:
        image_dir: 图像目录
        watermark_path: 水印路径
        model_path: 模型路径
        output_dir: 输出目录
        alpha: 嵌入强度
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_dir) 
                  if f.lower().endswith(valid_extensions)]
    
    print(f"\n批量嵌入水印")
    print(f"找到 {len(image_files)} 张图像")
    print("-" * 50)
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        print(f"\n[{i}/{len(image_files)}] 处理: {filename}")
        try:
            embed_watermark(image_path, watermark_path, model_path, output_path, alpha)
        except Exception as e:
            print(f"错误: {e}")
            continue
    
    print("\n" + "=" * 50)
    print(f"批量嵌入完成！输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='嵌入水印到图像')
    parser.add_argument('--image', type=str, required=True,
                       help='目标图像路径或目录')
    parser.add_argument('--watermark', type=str, required=True,
                       help='水印图像路径')
    parser.add_argument('--model', type=str, 
                       default=os.path.join('checkpoints', 'best_model.pth'),
                       help='模型checkpoint路径')
    parser.add_argument('--output', type=str, default='output/embedded.png',
                       help='输出图像路径或目录')
    parser.add_argument('--alpha', type=float, default=None,
                       help='嵌入强度（覆盖配置）')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.image):
        # 批量模式
        batch_embed(args.image, args.watermark, args.model, args.output, args.alpha)
    else:
        # 单张模式
        embed_watermark(args.image, args.watermark, args.model, args.output, args.alpha)


if __name__ == '__main__':
    main()
