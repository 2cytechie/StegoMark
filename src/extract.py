import os
import sys
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import WatermarkNet
from src.utils import calculate_psnr, calculate_ssim, calculate_nc, calculate_ber
from src.data.transforms import ResizeAndTile


class WatermarkExtractor:
    """水印提取器"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        
        print(f"模型加载完成，设备: {self.device}")
    
    def _load_model(self, checkpoint_path):
        """加载模型"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        args = checkpoint.get('args', None)
        
        # 创建模型
        if args:
            model = WatermarkNet(
                hidden_dim=args.hidden_dim,
                attack_prob=0.0,  # 提取时不使用攻击
                use_multiscale_decoder=getattr(args, 'use_multiscale', False),
                num_scales=getattr(args, 'num_scales', 3)
            )
        else:
            model = WatermarkNet()
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def preprocess_image(self, image_path, size=64):
        """预处理图像"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 变换
        transform = transforms.Compose([
            transforms.ToTensor(),
            ResizeAndTile(size),
        ])
        
        return transform(image).unsqueeze(0)
    
    def postprocess_watermark(self, watermark_tensor):
        """后处理提取的水印"""
        # 移除batch维度并转为numpy
        watermark = watermark_tensor.squeeze(0).cpu().numpy()
        
        # 转置为 (H, W, C)
        watermark = np.transpose(watermark, (1, 2, 0))
        
        # 缩放到 [0, 255]
        watermark = (watermark * 255).astype(np.uint8)
        
        return watermark
    
    def extract(self, image_path, output_path=None, return_confidence=False):
        """
        从图像中提取水印
        
        输入:
            image_path: 含水印图像路径
            output_path: 输出水印图像路径（可选）
            return_confidence: 是否返回置信度
        
        返回:
            watermark: 提取的水印图像 (PIL.Image)
            confidence: 置信度（如果return_confidence为True）
        """
        # 预处理
        image_tensor = self.preprocess_image(image_path).to(self.device)
        
        # 提取水印
        with torch.no_grad():
            extracted_wm, confidence = self.model.decode(image_tensor)
        
        # 后处理
        watermark_array = self.postprocess_watermark(extracted_wm)
        watermark_image = Image.fromarray(watermark_array)
        
        # 保存
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            watermark_image.save(output_path)
            print(f"水印已保存到: {output_path}")
        
        if return_confidence:
            return watermark_image, confidence.item()
        return watermark_image
    
    def embed(self, image_path, watermark_path, output_path=None):
        """
        将水印嵌入到图像
        
        输入:
            image_path: 载体图像路径
            watermark_path: 水印图像路径
            output_path: 输出图像路径（可选）
        
        返回:
            watermarked_image: 含水印图像 (PIL.Image)
        """
        # 预处理
        image_tensor = self.preprocess_image(image_path).to(self.device)
        watermark_tensor = self.preprocess_image(watermark_path).to(self.device)
        
        # 嵌入水印
        with torch.no_grad():
            watermarked = self.model.encode(image_tensor, watermark_tensor)
        
        # 后处理
        watermarked_array = self.postprocess_watermark(watermarked)
        watermarked_image = Image.fromarray(watermarked_array)
        
        # 保存
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
            watermarked_image.save(output_path)
            print(f"含水印图像已保存到: {output_path}")
        
        return watermarked_image
    
    def evaluate(self, original_image_path, watermarked_image_path, original_watermark_path):
        """
        评估水印系统性能
        
        返回:
            metrics: 包含PSNR, SSIM, NC, BER的字典
        """
        # 加载图像
        original_image = self.preprocess_image(original_image_path).to(self.device)
        watermarked_image = self.preprocess_image(watermarked_image_path).to(self.device)
        original_watermark = self.preprocess_image(original_watermark_path).to(self.device)
        
        # 提取水印
        with torch.no_grad():
            extracted_watermark, _ = self.model.decode(watermarked_image)
        
        # 计算指标
        metrics = {
            'psnr': calculate_psnr(original_image, watermarked_image),
            'ssim': calculate_ssim(original_image, watermarked_image),
            'nc': calculate_nc(original_watermark, extracted_watermark),
            'ber': calculate_ber(original_watermark, extracted_watermark)
        }
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description='水印嵌入和提取工具')
    parser.add_argument('--mode', type=str, required=True, choices=['embed', 'extract', 'eval'],
                        help='模式: embed(嵌入), extract(提取), eval(评估)')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--watermark', type=str, help='水印图像路径（embed模式必需）')
    parser.add_argument('--output', type=str, help='输出图像路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备: cuda 或 cpu')
    
    args = parser.parse_args()
    
    # 创建提取器
    extractor = WatermarkExtractor(args.checkpoint, args.device)
    
    if args.mode == 'embed':
        if not args.watermark:
            parser.error('embed模式需要提供--watermark参数')
        
        watermarked = extractor.embed(args.image, args.watermark, args.output)
        print(f"水印嵌入完成")
        
    elif args.mode == 'extract':
        result = extractor.extract(args.image, args.output, return_confidence=True)
        if isinstance(result, tuple):
            watermark, confidence = result
            print(f"水印提取完成，置信度: {confidence:.4f}")
        else:
            print(f"水印提取完成")
            
    elif args.mode == 'eval':
        if not args.watermark:
            parser.error('eval模式需要提供--watermark参数')
        
        metrics = extractor.evaluate(args.image, args.image, args.watermark)
        print("\n评估结果:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  SSIM: {metrics['ssim']:.4f}")
        print(f"  NC:   {metrics['nc']:.4f}")
        print(f"  BER:  {metrics['ber']:.4f}")


if __name__ == '__main__':
    main()