"""
水印系统综合测试脚本
测试水印嵌入、攻击模拟和水印提取的完整流程
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

# 导入项目模块
from config import image_config, dwt_config, eval_config
from models import create_model
from watermark_utils import (
    preprocess_watermark, preprocess_watermark_to_64x64,
    tensor_to_pil, pil_to_tensor, calculate_watermark_accuracy,
    resize_to_original, get_image_info, ColorWatermarkAccuracyCalculator
)
from utils import load_checkpoint, get_device, calculate_psnr, calculate_ssim
from attacks import AttackSimulator


class WatermarkTester:
    """水印系统测试器"""
    
    def __init__(self, model_path: str, output_dir: str = 'output'):
        """
        初始化测试器
        
        Args:
            model_path: 模型checkpoint路径
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = get_device()
        self.attack_simulator = AttackSimulator()
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载模型
        self._load_model()

        # 初始化彩色水印准确率计算器
        self.accuracy_calculator = ColorWatermarkAccuracyCalculator(
            tolerance=0.1,  # 10%容差
            threshold=0.5   # 二值化阈值
        )

        # 测试结果存储
        self.results = []
    
    def _load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        from config import model_config
        self.model = create_model(model_config, self.device)
        
        if os.path.exists(self.model_path):
            epoch, metrics = load_checkpoint(self.model_path, self.model, device=self.device)
            print(f"模型加载完成 (epoch {epoch})")
            if metrics:
                print(f"模型指标: PSNR={metrics.get('psnr', 0):.2f}dB, "
                      f"SSIM={metrics.get('ssim', 0):.4f}, "
                      f"Acc={metrics.get('watermark_acc', 0):.4f}")
        else:
            print(f"警告: 模型文件不存在: {self.model_path}")
            print("使用未训练的模型（结果可能不理想）")
        
        self.model.eval()
    
    def embed_watermark(self, image_path: str, watermark_path: str,
                        alpha: float = None) -> Tuple[torch.Tensor, torch.Tensor, Image.Image, Tuple[int, int]]:
        """
        嵌入水印并还原到原始尺寸

        Args:
            image_path: 目标图像路径
            watermark_path: 水印图像路径
            alpha: 嵌入强度

        Returns:
            (原始图像tensor_256x256, 水印tensor_256x256, 嵌入水印后的PIL图像_原始尺寸, 原始尺寸)
        """
        if alpha is None:
            alpha = dwt_config.BASE_ALPHA

        # 加载目标图像
        image = Image.open(image_path).convert('RGB')
        self.original_size = image.size
        original_width, original_height = self.original_size
        print(f"原始图像尺寸: {self.original_size} (宽x高: {original_width}x{original_height})")

        # resize到目标尺寸 (256x256) 用于模型处理
        image_resized = image.resize(
            (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
            Image.LANCZOS
        )
        image_tensor = pil_to_tensor(image_resized).unsqueeze(0).to(self.device)

        # 加载并预处理水印
        watermark_tensor = preprocess_watermark(
            watermark_path,
            watermark_size=image_config.WATERMARK_SIZE,
            target_size=image_config.TARGET_SIZE
        ).unsqueeze(0).to(self.device)

        # 嵌入水印
        print(f"嵌入水印 (alpha={alpha})...")
        with torch.no_grad():
            watermarked_tensor = self.model.embed(image_tensor, watermark_tensor, alpha=alpha)

        # 转换回PIL图像
        watermarked_pil = tensor_to_pil(watermarked_tensor.squeeze(0))

        # 【关键修改】将嵌入水印后的图像调整回原始尺寸
        print(f"  将嵌入水印后的图像从 256x256 调整回原始尺寸 {self.original_size[0]}x{self.original_size[1]}")
        watermarked_pil_original_size = resize_to_original(
            watermarked_pil,
            self.original_size,
            resample=Image.LANCZOS
        )
        
        # 验证尺寸
        if watermarked_pil_original_size.size != self.original_size:
            raise ValueError(f"尺寸验证失败: 期望 {self.original_size}, 实际 {watermarked_pil_original_size.size}")
        print(f"  已调整为原始尺寸: {watermarked_pil_original_size.size}")

        return image_tensor, watermark_tensor, watermarked_pil_original_size, self.original_size
    
    def apply_attack(self, watermarked_pil: Image.Image, 
                     attack_type: str, **kwargs) -> Image.Image:
        """
        对原始尺寸的PIL图像应用攻击
        
        Args:
            watermarked_pil: 含水印的PIL图像（原始尺寸）
            attack_type: 攻击类型
            **kwargs: 攻击参数
        
        Returns:
            攻击后的PIL图像（保持原始尺寸）
        """
        print(f"应用攻击: {attack_type} (在原始尺寸 {watermarked_pil.size} 上进行)")
        
        # 将PIL转换为tensor进行攻击
        watermarked_tensor = pil_to_tensor(watermarked_pil).unsqueeze(0).to(self.device)
        
        # 应用攻击
        attacked_tensor = self.attack_simulator.apply_attack_by_name(
            watermarked_tensor, attack_type, **kwargs
        )
        
        # 转换回PIL
        attacked_pil = tensor_to_pil(attacked_tensor.squeeze(0))
        
        # 确保攻击后尺寸与输入一致
        if attacked_pil.size != watermarked_pil.size:
            attacked_pil = resize_to_original(attacked_pil, watermarked_pil.size, Image.LANCZOS)
        
        return attacked_pil
    
    def extract_watermark(self, image_pil: Image.Image) -> torch.Tensor:
        """
        从图像中提取水印
        【关键修改】先将图像resize到256x256，然后提取
        
        Args:
            image_pil: PIL图像（任意尺寸）
        
        Returns:
            提取的水印tensor [1, 3, 64, 64]
        """
        print(f"提取水印...")
        print(f"  输入图像尺寸: {image_pil.size}")
        
        # 【关键修改】先将图像resize到256x256
        image_resized = image_pil.resize(
            (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
            Image.LANCZOS
        )
        print(f"  调整后尺寸: {image_resized.size}")
        
        # 转换为tensor
        image_tensor = pil_to_tensor(image_resized).unsqueeze(0).to(self.device)
        
        # 提取水印
        with torch.no_grad():
            extracted_tensor = self.model.extract(image_tensor)
        
        return extracted_tensor
    
    def calculate_metrics(self, original: torch.Tensor, 
                         watermarked: torch.Tensor,
                         original_watermark: torch.Tensor,
                         extracted_watermark: torch.Tensor) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            original: 原始图像 (256x256)
            watermarked: 含水印的图像 (256x256)
            original_watermark: 原始水印 (64x64)
            extracted_watermark: 提取的水印 (64x64)
        
        Returns:
            指标字典
        """
        # 计算图像质量指标
        psnr = calculate_psnr(original, watermarked)
        ssim = calculate_ssim(original, watermarked)
        
        # 计算水印提取准确率
        accuracy = calculate_watermark_accuracy(extracted_watermark, original_watermark)
        
        return {
            'psnr': psnr,
            'ssim': ssim,
            'accuracy': accuracy
        }
    
    def save_results(self, image_tensor: torch.Tensor,
                    watermark_tensor: torch.Tensor,
                    watermarked_pil: Image.Image,
                    attacked_pil: Image.Image,
                    extracted_tensor: torch.Tensor,
                    attack_type: str,
                    test_name: str,
                    original_size: Tuple[int, int] = None):
        """
        保存测试结果

        Args:
            image_tensor: 原始图像 (256x256 tensor)
            watermark_tensor: 原始水印 (256x256 tensor)
            watermarked_pil: 嵌入水印后的PIL图像 (原始尺寸)
            attacked_pil: 攻击后的PIL图像 (原始尺寸)
            extracted_tensor: 提取的水印 (64x64 tensor)
            attack_type: 攻击类型
            test_name: 测试名称
            original_size: 原始图像尺寸
        """
        # 创建测试子目录
        test_dir = os.path.join(self.output_dir, test_name)
        os.makedirs(test_dir, exist_ok=True)

        # 保存嵌入水印后的图像 (原始尺寸)
        watermarked_pil.save(os.path.join(test_dir, '1_watermarked_image.png'))
        print(f"  已保存嵌入水印图像 (原始尺寸): {watermarked_pil.size}")

        # 保存攻击后的图像 (原始尺寸)
        attacked_pil.save(os.path.join(test_dir, f'2_attacked_{attack_type}.png'))
        print(f"  已保存攻击后图像 (原始尺寸): {attacked_pil.size}")

        # 保存提取的水印
        extracted_pil = tensor_to_pil(extracted_tensor.squeeze(0))
        extracted_pil.save(os.path.join(test_dir, f'3_extracted_{attack_type}.png'))

        print(f"结果已保存到: {test_dir}")
    
    def run_single_test(self, image_path: str, watermark_path: str,
                       attack_type: str, test_name: str = None,
                       alpha: float = None, **attack_kwargs) -> Dict:
        """
        运行单次测试
        
        Args:
            image_path: 目标图像路径
            watermark_path: 水印图像路径
            attack_type: 攻击类型
            test_name: 测试名称
            alpha: 嵌入强度
            **attack_kwargs: 攻击参数
        
        Returns:
            测试结果字典
        """
        if test_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_name = f"test_{attack_type}_{timestamp}"
        
        print("\n" + "="*60)
        print(f"开始测试: {test_name}")
        print(f"攻击类型: {attack_type}")
        print("="*60)
        
        try:
            # 步骤1: 嵌入水印（返回原始尺寸的PIL图像）
            print("\n步骤1: 嵌入水印")
            image_tensor, watermark_tensor, watermarked_pil, original_size = self.embed_watermark(
                image_path, watermark_path, alpha
            )

            # 步骤2: 应用攻击（在原始尺寸上进行）
            print("\n步骤2: 应用攻击")
            if attack_type == 'none':
                attacked_pil = watermarked_pil.copy()
            else:
                attacked_pil = self.apply_attack(watermarked_pil, attack_type, **attack_kwargs)

            # 步骤3: 提取水印（内部会将图像resize到256x256）
            print("\n步骤3: 提取水印")
            extracted_tensor = self.extract_watermark(attacked_pil)

            # 步骤4: 计算指标
            print("\n步骤4: 计算评估指标")

            # 【修改】将原始图像tensor和嵌入后的图像（resize到256x256后）计算PSNR/SSIM
            watermarked_resized = watermarked_pil.resize(
                (image_config.TARGET_SIZE, image_config.TARGET_SIZE),
                Image.LANCZOS
            )
            watermarked_tensor_256 = pil_to_tensor(watermarked_resized)
            
            embed_psnr = calculate_psnr(image_tensor.squeeze(0), watermarked_tensor_256)
            embed_ssim = calculate_ssim(image_tensor.squeeze(0), watermarked_tensor_256)

            # 水印提取准确率 - 使用RGB彩色准确率作为主要指标
            # 将原始水印resize到64x64用于比较
            original_wm_64 = preprocess_watermark_to_64x64(watermark_path).to(self.device)

            # 使用RGB彩色水印准确率计算器（主要指标）
            rgb_accuracy = self.accuracy_calculator.calculate_tolerance_accuracy(
                extracted_tensor.squeeze(0), original_wm_64
            )

            metrics = {
                'test_name': test_name,
                'attack_type': attack_type,
                'embed_psnr': embed_psnr,
                'embed_ssim': embed_ssim,
                'watermark_acc': rgb_accuracy['overall'],  # RGB综合准确率作为主要指标
                'color_acc_r': rgb_accuracy['R_channel'],
                'color_acc_g': rgb_accuracy['G_channel'],
                'color_acc_b': rgb_accuracy['B_channel'],
                'color_acc_overall': rgb_accuracy['overall'],
                'color_correct_pixels': rgb_accuracy['total_correct_pixels'],
                'color_total_pixels': rgb_accuracy['total_pixels'],
                'original_size': original_size
            }

            print(f"\n测试结果:")
            print(f"  原始图像尺寸: {original_size[0]}x{original_size[1]}")
            print(f"  嵌入质量 PSNR: {embed_psnr:.2f} dB")
            print(f"  嵌入质量 SSIM: {embed_ssim:.4f}")
            print(f"  RGB水印准确率: {rgb_accuracy['overall']:.4f} ({rgb_accuracy['overall']*100:.2f}%)")
            print(f"    R通道: {rgb_accuracy['R_channel']:.4f} ({rgb_accuracy['R_channel']*100:.2f}%)")
            print(f"    G通道: {rgb_accuracy['G_channel']:.4f} ({rgb_accuracy['G_channel']*100:.2f}%)")
            print(f"    B通道: {rgb_accuracy['B_channel']:.4f} ({rgb_accuracy['B_channel']*100:.2f}%)")
            print(f"  正确像素: {rgb_accuracy['total_correct_pixels']} / {rgb_accuracy['total_pixels']}")

            # 步骤5: 保存结果
            print("\n步骤5: 保存结果")
            self.save_results(
                image_tensor, watermark_tensor, watermarked_pil,
                attacked_pil, extracted_tensor, attack_type, test_name,
                original_size=original_size
            )
            
            self.results.append(metrics)
            
            print("\n" + "="*60)
            print(f"测试完成: {test_name}")
            print("="*60)
            
            return metrics
            
        except Exception as e:
            print(f"\n错误: 测试过程中发生异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'test_name': test_name,
                'attack_type': attack_type,
                'error': str(e)
            }
    
    def run_all_attacks(self, image_path: str, watermark_path: str,
                       alpha: float = None) -> List[Dict]:
        """
        运行所有攻击类型的测试
        
        Args:
            image_path: 目标图像路径
            watermark_path: 水印图像路径
            alpha: 嵌入强度
        
        Returns:
            所有测试结果列表
        """
        attacks = [
            ('none', {}),
            ('crop', {}),
            ('rotate', {}),
            ('scale', {}),
            ('blur', {}),
            ('noise', {}),
            ('jpeg', {}),
            ('combined', {})
        ]
        
        print("\n" + "="*60)
        print("开始全面测试 - 所有攻击类型")
        print("="*60)
        
        all_results = []
        
        for i, (attack_type, kwargs) in enumerate(attacks, 1):
            test_name = attack_type
            result = self.run_single_test(
                image_path, watermark_path, attack_type, 
                test_name, alpha, **kwargs
            )
            all_results.append(result)
            print("\n")
        
        # 保存汇总结果
        self._save_summary(all_results, test_name)
        
        return all_results
    
    def _save_summary(self, results: List[Dict], timestamp: str):
        """
        保存测试汇总结果
        
        Args:
            results: 测试结果列表
            timestamp: 时间戳
        """
        summary_file = os.path.join(self.output_dir, f'summary_{timestamp}.json')
        
        # 计算统计信息
        valid_results = [r for r in results if 'error' not in r]
        
        if valid_results:
            summary = {
                'timestamp': timestamp,
                'total_tests': len(results),
                'successful_tests': len(valid_results),
                'average_psnr': np.mean([r['embed_psnr'] for r in valid_results]),
                'average_ssim': np.mean([r['embed_ssim'] for r in valid_results]),
                'average_accuracy': np.mean([r['extract_accuracy'] for r in valid_results]),
                'results': results
            }
        else:
            summary = {
                'timestamp': timestamp,
                'total_tests': len(results),
                'successful_tests': 0,
                'results': results
            }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试汇总已保存: {summary_file}")
        
        # 打印汇总
        if valid_results:
            print("\n" + "="*60)
            print("测试汇总")
            print("="*60)
            print(f"总测试数: {len(results)}")
            print(f"成功测试数: {len(valid_results)}")
            print(f"平均嵌入 PSNR: {summary['average_psnr']:.2f} dB")
            print(f"平均嵌入 SSIM: {summary['average_ssim']:.4f}")
            print(f"平均水印准确率: {summary['average_accuracy']:.4f} ({summary['average_accuracy']*100:.2f}%)")
            
            print("\n各攻击类型结果:")
            print("-"*60)
            print(f"{'攻击类型':<15} {'PSNR(dB)':<12} {'SSIM':<10} {'准确率':<10}")
            print("-"*60)
            for r in valid_results:
                print(f"{r['attack_type']:<15} {r['embed_psnr']:<12.2f} "
                      f"{r['embed_ssim']:<10.4f} {r['extract_accuracy']:<10.4f}")
            print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='水印系统测试脚本')
    parser.add_argument('--image', type=str, 
                       default='img/img8.jpg',
                       help='目标图像路径')
    parser.add_argument('--watermark', type=str,
                       default='img/watermark.png',
                       help='水印图像路径')
    parser.add_argument('--model', type=str,
                       default='checkpoints/best_model.pth',
                       help='模型checkpoint路径')
    parser.add_argument('--output', type=str,
                       default='output',
                       help='输出目录')
    parser.add_argument('--attack', type=str,
                       default='all',
                       choices=['all', 'none', 'crop', 'rotate', 'scale', 
                               'blur', 'noise', 'jpeg', 'combined'],
                       help='攻击类型')
    parser.add_argument('--alpha', type=float,
                       default=None,
                       help='嵌入强度')
    parser.add_argument('--test-name', type=str,
                       default=None,
                       help='测试名称（可选）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        return
    
    if not os.path.exists(args.watermark):
        print(f"错误: 水印文件不存在: {args.watermark}")
        return
    
    # 创建测试器
    tester = WatermarkTester(args.model, args.output)
    
    # 运行测试
    if args.attack == 'all':
        results = tester.run_all_attacks(args.image, args.watermark, args.alpha)
    else:
        result = tester.run_single_test(
            args.image, args.watermark, args.attack, 
            args.test_name, args.alpha
        )
        results = [result]
    
    print("\n所有测试完成！")


if __name__ == '__main__':
    main()
