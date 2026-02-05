"""
水印提取脚本
用于从含水印的图像中提取水印
支持各种攻击后的图像
"""

import torch
import argparse
import os
from PIL import Image

from config import train_config, image_config
from models import create_model
from watermark_utils import tensor_to_pil, pil_to_tensor
from utils import load_checkpoint, get_device, save_image
from attacks import AttackSimulator


def extract_watermark(image_path: str,
                     model_path: str,
                     output_path: str,
                     original_watermark_path: str = None):
    """
    从图像中提取水印
    
    Args:
        image_path: 含水印的图像路径
        model_path: 模型checkpoint路径
        output_path: 输出水印路径
        original_watermark_path: 原始水印路径（用于计算准确率，可选）
    """
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
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
    
    # 加载图像
    print(f"\n加载图像: {image_path}")
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    print(f"原始图像尺寸: {original_size}")
    
    # resize到目标尺寸
    image_resized = image.resize(
        (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
        Image.LANCZOS
    )
    image_tensor = pil_to_tensor(image_resized).unsqueeze(0).to(device)
    
    # 提取水印
    print("\n提取水印...")
    with torch.no_grad():
        extracted_tensor = model.extract(image_tensor)
    
    # 转换回PIL图像
    extracted_pil = tensor_to_pil(extracted_tensor.squeeze(0))
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    extracted_pil.save(output_path, quality=95)
    print(f"\n提取的水印已保存: {output_path}")
    
    # 如果有原始水印，计算RGB彩色准确率
    if original_watermark_path and os.path.exists(original_watermark_path):
        from watermark_utils import preprocess_watermark_to_64x64, ColorWatermarkAccuracyCalculator

        original_wm = preprocess_watermark_to_64x64(original_watermark_path)
        color_calculator = ColorWatermarkAccuracyCalculator(tolerance=0.1, threshold=0.5)
        rgb_metrics = color_calculator.calculate_tolerance_accuracy(extracted_tensor.squeeze(0), original_wm)

        print(f"\nRGB水印提取准确率: {rgb_metrics['overall']:.4f} ({rgb_metrics['overall']*100:.2f}%)")
        print(f"  R通道: {rgb_metrics['R_channel']:.4f}")
        print(f"  G通道: {rgb_metrics['G_channel']:.4f}")
        print(f"  B通道: {rgb_metrics['B_channel']:.4f}")
    
    return output_path


def extract_with_attack(image_path: str,
                       model_path: str,
                       output_path: str,
                       attack_type: str,
                       attack_params: dict = None):
    """
    对图像应用攻击后提取水印（用于测试鲁棒性）
    
    Args:
        image_path: 含水印的图像路径
        model_path: 模型路径
        output_path: 输出路径
        attack_type: 攻击类型
        attack_params: 攻击参数
    """
    # 获取设备
    device = get_device()
    
    print(f"\n加载模型: {model_path}")
    from config import model_config
    model = create_model(model_config, device)
    
    # 加载checkpoint
    if os.path.exists(model_path):
        epoch, metrics = load_checkpoint(model_path, model, device=device)
        print(f"模型加载完成 (epoch {epoch})")
    
    model.eval()
    
    # 加载图像
    print(f"\n加载图像: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_resized = image.resize(
        (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
        Image.LANCZOS
    )
    image_tensor = pil_to_tensor(image_resized).unsqueeze(0).to(device)
    
    # 应用攻击
    print(f"\n应用攻击: {attack_type}")
    attack_simulator = AttackSimulator()
    
    if attack_params is None:
        attack_params = {}
    
    attacked_tensor = attack_simulator.apply_attack_by_name(
        image_tensor, attack_type, **attack_params
    )
    
    # 保存攻击后的图像
    attacked_path = output_path.replace('.png', f'_attacked_{attack_type}.png')
    attacked_pil = tensor_to_pil(attacked_tensor.squeeze(0))
    attacked_pil.save(attacked_path)
    print(f"攻击后的图像已保存: {attacked_path}")
    
    # 提取水印
    print("提取水印...")
    with torch.no_grad():
        extracted_tensor = model.extract(attacked_tensor)
    
    # 保存提取的水印
    extracted_pil = tensor_to_pil(extracted_tensor.squeeze(0))
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    extracted_pil.save(output_path)
    print(f"提取的水印已保存: {output_path}")
    
    return output_path


def batch_extract(image_dir: str,
                 model_path: str,
                 output_dir: str,
                 original_watermark_path: str = None):
    """
    批量提取水印
    
    Args:
        image_dir: 图像目录
        model_path: 模型路径
        output_dir: 输出目录
        original_watermark_path: 原始水印路径
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_dir)
                  if f.lower().endswith(valid_extensions)]
    
    print(f"\n批量提取水印")
    print(f"找到 {len(image_files)} 张图像")
    print("-" * 50)
    
    accuracies = []
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, filename)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_extracted.png")
        
        print(f"\n[{i}/{len(image_files)}] 处理: {filename}")
        try:
            extract_watermark(image_path, model_path, output_path, original_watermark_path)
            
            # 如果有原始水印，收集RGB彩色准确率
            if original_watermark_path:
                from watermark_utils import preprocess_watermark_to_64x64, ColorWatermarkAccuracyCalculator

                device = get_device()
                from config import model_config
                model = create_model(model_config, device)
                load_checkpoint(model_path, model, device=device)
                model.eval()

                image = Image.open(image_path).convert('RGB')
                image_resized = image.resize(
                    (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
                    Image.LANCZOS
                )
                image_tensor = pil_to_tensor(image_resized).unsqueeze(0).to(device)

                with torch.no_grad():
                    extracted = model.extract(image_tensor)

                original_wm = preprocess_watermark_to_64x64(original_watermark_path)
                color_calculator = ColorWatermarkAccuracyCalculator(tolerance=0.1, threshold=0.5)
                rgb_metrics = color_calculator.calculate_tolerance_accuracy(extracted.squeeze(0), original_wm)
                accuracies.append(rgb_metrics['overall'])

        except Exception as e:
            print(f"错误: {e}")
            continue

    print("\n" + "=" * 50)
    print(f"批量提取完成！输出目录: {output_dir}")

    if accuracies:
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"平均RGB水印提取准确率: {avg_acc:.4f} ({avg_acc*100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='从图像中提取水印')
    parser.add_argument('--image', type=str, required=True,
                       help='含水印的图像路径或目录')
    parser.add_argument('--model', type=str,
                       default=os.path.join('checkpoints', 'best_model.pth'),
                       help='模型checkpoint路径')
    parser.add_argument('--output', type=str, default='output/extracted.png',
                       help='输出水印路径或目录')
    parser.add_argument('--original_watermark', type=str, default=None,
                       help='原始水印路径（用于计算准确率）')
    parser.add_argument('--attack', type=str, default=None,
                       help='应用攻击类型（用于测试鲁棒性）')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')
    
    args = parser.parse_args()
    
    if args.batch or os.path.isdir(args.image):
        # 批量模式
        batch_extract(args.image, args.model, args.output, args.original_watermark)
    elif args.attack:
        # 攻击测试模式
        extract_with_attack(args.image, args.model, args.output, args.attack)
    else:
        # 单张模式
        extract_watermark(args.image, args.model, args.output, args.original_watermark)


if __name__ == '__main__':
    main()
