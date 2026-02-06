"""
水印提取测试脚本
包含投票机制和色彩增强
"""
import os
import sys
sys.path.insert(0, r'd:\实验室图像隐水印项目\StegoMark')

import torch
import torch.nn.functional as F
import argparse
from PIL import Image
import torchvision.transforms as T
import numpy as np
from pathlib import Path

from configs.config import model_config, training_config, inference_config
from src.models import create_model
from src.utils import calculate_watermark_accuracy


def load_image(image_path):
    """加载图片"""
    image = Image.open(image_path).convert('RGB')
    return image


def preprocess_image(image):
    """预处理图片"""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image)


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


def extract_watermark_single(model, image_tensor, device):
    """
    从单张图片提取水印
    
    Args:
        model: 模型
        image_tensor: 图片张量 [C, H, W] 或 [1, C, H, W]
        device: 设备
        
    Returns:
        extracted_watermark: 提取的水印 [C, 64, 64]
    """
    model.eval()
    
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        extracted_watermark = model.decode(image_tensor)
    
    return extracted_watermark.squeeze(0)


def sliding_window_extraction(model, image_tensor, device, block_size=64, overlap=0.5):
    """
    滑动窗口提取水印 (投票机制)
    
    Args:
        model: 模型
        image_tensor: 图片张量 [C, H, W]
        device: 设备
        block_size: 块大小
        overlap: 重叠比例
        
    Returns:
        extracted_watermarks: 提取的水印列表
        positions: 块位置列表
    """
    model.eval()
    
    C, H, W = image_tensor.shape
    stride = int(block_size * (1 - overlap))
    
    extracted_watermarks = []
    positions = []
    
    # 滑动窗口
    for y in range(0, H - block_size + 1, stride):
        for x in range(0, W - block_size + 1, stride):
            # 提取块
            block = image_tensor[:, y:y+block_size, x:x+block_size]
            
            # 如果块太小，跳过
            if block.shape[1] < 32 or block.shape[2] < 32:
                continue
            
            # 提取水印
            with torch.no_grad():
                block_batch = block.unsqueeze(0).to(device)
                extracted = model.decode(block_batch)
            
            extracted_watermarks.append(extracted.squeeze(0).cpu())
            positions.append((y, x))
    
    return extracted_watermarks, positions


def voting_mechanism(extracted_watermarks, correlation_threshold=0.7):
    """
    投票机制 - 加权平均高质量块
    
    Args:
        extracted_watermarks: 提取的水印列表
        correlation_threshold: 相关性阈值
        
    Returns:
        final_watermark: 最终水印
        quality_scores: 质量分数列表
    """
    if len(extracted_watermarks) == 0:
        return None, []
    
    if len(extracted_watermarks) == 1:
        return extracted_watermarks[0], [1.0]
    
    # 计算块间相关性
    n = len(extracted_watermarks)
    correlation_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # 计算相关性
            corr = torch.cosine_similarity(
                extracted_watermarks[i].flatten(),
                extracted_watermarks[j].flatten(),
                dim=0
            ).item()
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr
    
    # 计算每个块的质量分数 (与其他块的平均相关性)
    quality_scores = correlation_matrix.mean(axis=1)
    
    # 筛选高质量块
    high_quality_indices = np.where(quality_scores > correlation_threshold)[0]
    
    if len(high_quality_indices) == 0:
        # 如果没有高质量块，使用所有块
        high_quality_indices = np.arange(n)
    
    # 加权平均
    weights = quality_scores[high_quality_indices]
    weights = weights / weights.sum()  # 归一化
    
    final_watermark = torch.zeros_like(extracted_watermarks[0])
    for idx, weight in zip(high_quality_indices, weights):
        final_watermark += extracted_watermarks[idx] * weight
    
    return final_watermark, quality_scores.tolist()


def color_correction(extracted_watermark, method='histogram'):
    """
    色彩增强 - 对RGB三通道进行色彩校正
    
    Args:
        extracted_watermark: 提取的水印 [C, H, W]
        method: 校正方法 ('histogram', 'white_balance', 'contrast')
        
    Returns:
        corrected_watermark: 校正后的水印
    """
    if method == 'histogram':
        # 直方图均衡化
        corrected = []
        for c in range(extracted_watermark.shape[0]):
            channel = extracted_watermark[c].numpy()
            
            # 归一化到[0, 255]
            channel = ((channel + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            
            # 直方图均衡化
            from PIL import ImageOps
            channel_img = Image.fromarray(channel)
            channel_eq = ImageOps.equalize(channel_img)
            channel = np.array(channel_eq).astype(np.float32) / 255.0
            
            # 反归一化到[-1, 1]
            channel = channel * 2 - 1
            corrected.append(torch.from_numpy(channel))
        
        return torch.stack(corrected)
    
    elif method == 'white_balance':
        # 白平衡
        corrected = extracted_watermark.clone()
        
        for c in range(extracted_watermark.shape[0]):
            channel = extracted_watermark[c]
            # 简单灰度世界假设
            mean_val = channel.mean()
            corrected[c] = channel * (0.5 / (mean_val + 1e-6))
        
        return torch.clamp(corrected, -1, 1)
    
    elif method == 'contrast':
        # 对比度增强
        corrected = extracted_watermark.clone()
        
        for c in range(extracted_watermark.shape[0]):
            channel = extracted_watermark[c]
            # 增加对比度
            mean_val = channel.mean()
            corrected[c] = (channel - mean_val) * 1.5 + mean_val
        
        return torch.clamp(corrected, -1, 1)
    
    else:
        return extracted_watermark


def extract_watermark_with_voting(model, image_path, device, use_voting=True, use_color_correction=True):
    """
    提取水印 (完整流程)
    
    Args:
        model: 模型
        image_path: 图片路径
        device: 设备
        use_voting: 是否使用投票机制
        use_color_correction: 是否使用色彩校正
        
    Returns:
        final_watermark: 最终提取的水印
    """
    # 加载图片
    image = load_image(image_path)
    image_tensor = preprocess_image(image)
    
    if use_voting:
        # 滑动窗口提取
        print("使用滑动窗口提取水印...")
        extracted_watermarks, positions = sliding_window_extraction(
            model, image_tensor, device,
            block_size=inference_config.block_size,
            overlap=inference_config.overlap
        )
        
        print(f"提取到 {len(extracted_watermarks)} 个水印块")
        
        # 投票机制
        print("应用投票机制...")
        final_watermark, quality_scores = voting_mechanism(
            extracted_watermarks,
            correlation_threshold=inference_config.correlation_threshold
        )
        
        print(f"高质量块数量: {sum(1 for s in quality_scores if s > inference_config.correlation_threshold)}")
    else:
        # 直接提取
        final_watermark = extract_watermark_single(model, image_tensor, device)
    
    # 色彩校正
    if use_color_correction and inference_config.color_correction:
        print("应用色彩校正...")
        final_watermark = color_correction(final_watermark, method='histogram')
    
    return final_watermark


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
    
    # 提取水印
    print(f"\n提取水印:")
    print(f"  待检测图片: {args.image}")
    
    extracted_watermark = extract_watermark_with_voting(
        model, args.image, device,
        use_voting=args.use_voting,
        use_color_correction=args.use_color_correction
    )
    
    # 如果有原始水印，计算准确率
    if args.original_watermark:
        original_watermark = load_image(args.original_watermark)
        preprocessor = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        original_watermark_tensor = preprocessor(original_watermark)
        
        accuracy = calculate_watermark_accuracy(
            original_watermark_tensor, extracted_watermark
        )
        print(f"\n水印提取准确率: {accuracy:.4f}")
    
    # 保存结果
    if args.output:
        save_image(extracted_watermark, args.output)
    else:
        # 自动生成输出路径
        image_path = Path(args.image)
        output_path = image_path.parent / f"{image_path.stem}_extracted_watermark.png"
        save_image(extracted_watermark, str(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='提取水印')
    parser.add_argument('--image', type=str, required=True, help='待检测图片路径')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型检查点路径')
    parser.add_argument('--original-watermark', type=str, default=None, help='原始水印路径(用于计算准确率)')
    parser.add_argument('--output', type=str, default=None, help='输出水印路径')
    parser.add_argument('--use-voting', action='store_true', default=True, help='使用投票机制')
    parser.add_argument('--use-color-correction', action='store_true', default=True, help='使用色彩校正')
    args = parser.parse_args()
    
    main(args)
