import torch
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from config import config
from src.adversarial import NoiseLayer

class PerformanceEvaluator:
    """
    性能评估模块
    评估水印系统的不可见性、鲁棒性和提取准确率
    """
    
    def __init__(self):
        """
        初始化性能评估器
        """
        self.noise_layer = NoiseLayer()
    
    def calculate_psnr(self, original_image, watermarked_image):
        """
        计算PSNR (峰值信噪比)
        
        Args:
            original_image: 原始图像路径或张量
            watermarked_image: 含水印图像路径或张量
            
        Returns:
            psnr_value: PSNR值
        """
        # 加载图像
        if isinstance(original_image, str):
            original = np.array(Image.open(original_image).convert('RGB'))
        else:
            original = original_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            original = (original * 255).astype(np.uint8)
        
        if isinstance(watermarked_image, str):
            watermarked = np.array(Image.open(watermarked_image).convert('RGB'))
        else:
            watermarked = watermarked_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            watermarked = (watermarked * 255).astype(np.uint8)
        
        # 确保图像尺寸匹配
        if original.shape != watermarked.shape:
            print(f"Warning: Image dimensions mismatch: original={original.shape}, watermarked={watermarked.shape}")
            # 调整水印图像尺寸以匹配原始图像
            watermarked_img = Image.fromarray(watermarked)
            watermarked_img = watermarked_img.resize((original.shape[1], original.shape[0]), Image.LANCZOS)
            watermarked = np.array(watermarked_img)
            print(f"Resized watermarked image to: {watermarked.shape}")
        
        # 计算PSNR
        try:
            psnr_value = psnr(original, watermarked)
            print(f"PSNR calculated: {psnr_value:.2f} dB")
            return psnr_value
        except Exception as e:
            print(f"Error calculating PSNR: {str(e)}")
            return 0.0
    
    def calculate_ssim(self, original_image, watermarked_image):
        """
        计算SSIM (结构相似性)
        
        Args:
            original_image: 原始图像路径或张量
            watermarked_image: 含水印图像路径或张量
            
        Returns:
            ssim_value: SSIM值
        """
        # 加载图像
        if isinstance(original_image, str):
            original = np.array(Image.open(original_image).convert('RGB'))
        else:
            original = original_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            original = (original * 255).astype(np.uint8)
        
        if isinstance(watermarked_image, str):
            watermarked = np.array(Image.open(watermarked_image).convert('RGB'))
        else:
            watermarked = watermarked_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            watermarked = (watermarked * 255).astype(np.uint8)
        
        # 确保图像尺寸匹配
        if original.shape != watermarked.shape:
            print(f"Warning: Image dimensions mismatch: original={original.shape}, watermarked={watermarked.shape}")
            # 调整水印图像尺寸以匹配原始图像
            watermarked_img = Image.fromarray(watermarked)
            watermarked_img = watermarked_img.resize((original.shape[1], original.shape[0]), Image.LANCZOS)
            watermarked = np.array(watermarked_img)
            print(f"Resized watermarked image to: {watermarked.shape}")
        
        # 计算SSIM
        try:
            ssim_value = ssim(original, watermarked, channel_axis=2)
            print(f"SSIM calculated: {ssim_value:.4f}")
            return ssim_value
        except Exception as e:
            print(f"Error calculating SSIM: {str(e)}")
            return 0.0
    
    def evaluate_invisibility(self, original_image, watermarked_image):
        """
        评估水印不可见性
        
        Args:
            original_image: 原始图像
            watermarked_image: 含水印图像
            
        Returns:
            invisibility_metrics: 不可见性指标字典
        """
        psnr_value = self.calculate_psnr(original_image, watermarked_image)
        ssim_value = self.calculate_ssim(original_image, watermarked_image)
        
        invisibility_metrics = {
            'PSNR': psnr_value,
            'SSIM': ssim_value,
            'PSNR_Threshold_Met': psnr_value >= config.PSNR_THRESHOLD,
            'SSIM_Threshold_Met': ssim_value >= config.SSIM_THRESHOLD
        }
        
        return invisibility_metrics
    
    def evaluate_extraction_accuracy(self, extracted_watermark, original_watermark, watermark_type='image'):
        """
        评估水印提取准确率
        
        Args:
            extracted_watermark: 提取的水印
            original_watermark: 原始水印
            watermark_type: 水印类型
            
        Returns:
            accuracy: 提取准确率
        """
        if watermark_type == 'image':
            # 图像水印准确率
            if isinstance(extracted_watermark, Image.Image):
                extracted_array = np.array(extracted_watermark)
            else:
                extracted_array = extracted_watermark.squeeze(0).squeeze(0).cpu().numpy()
                extracted_array = (extracted_array * 255).astype(np.uint8)
            
            if isinstance(original_watermark, str):
                original = Image.open(original_watermark).convert('L')
                original = original.resize((config.WATERMARK_SIZE, config.WATERMARK_SIZE))
                original_array = np.array(original)
            else:
                original_array = original_watermark.squeeze(0).squeeze(0).cpu().numpy()
                original_array = (original_array * 255).astype(np.uint8)
            
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
    
    def evaluate_robustness(self, model, watermarked_image, original_watermark, watermark_type='image'):
        """
        评估系统鲁棒性
        
        Args:
            model: 水印模型
            watermarked_image: 含水印图像
            original_watermark: 原始水印
            watermark_type: 水印类型
            
        Returns:
            robustness_metrics: 鲁棒性指标字典
        """
        robustness_metrics = {}
        attack_types = ['gaussian', 'jpeg', 'crop', 'blur', 'all']
        
        # 加载含水印图像
        if isinstance(watermarked_image, str):
            from src.extraction import WatermarkExtraction
            extractor = WatermarkExtraction(watermark_type=watermark_type)
            watermarked_tensor = extractor.preprocess_watermarked_image(watermarked_image)
        else:
            watermarked_tensor = watermarked_image
        
        watermarked_tensor = watermarked_tensor.to(config.DEVICE)
        
        for attack_type in attack_types:
            # 应用攻击
            attacked_image = self.noise_layer(watermarked_tensor, attack_type)
            
            # 提取水印
            with torch.no_grad():
                extracted_watermark = model.extract(attacked_image)
            
            # 计算准确率
            if watermark_type == 'image':
                if isinstance(original_watermark, str):
                    original = np.array(Image.open(original_watermark).convert('L').resize((config.WATERMARK_SIZE, config.WATERMARK_SIZE)))
                    original = torch.tensor(original / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(config.DEVICE)
                else:
                    original = original_watermark
                
                accuracy = self.evaluate_extraction_accuracy(extracted_watermark, original, watermark_type)
            else:
                accuracy = self.evaluate_extraction_accuracy(extracted_watermark, original_watermark, watermark_type)
            
            robustness_metrics[attack_type] = accuracy
        
        return robustness_metrics
    
    def generate_evaluation_report(self, model, original_image, watermarked_image, original_watermark, extracted_watermark, watermark_type='image'):
        """
        生成完整评估报告
        
        Args:
            model: 水印模型
            original_image: 原始图像
            watermarked_image: 含水印图像
            original_watermark: 原始水印
            extracted_watermark: 提取的水印
            watermark_type: 水印类型
            
        Returns:
            report: 评估报告字典
        """
        # 1. 评估不可见性
        invisibility_metrics = self.evaluate_invisibility(original_image, watermarked_image)
        
        # 2. 评估提取准确率
        extraction_accuracy = self.evaluate_extraction_accuracy(extracted_watermark, original_watermark, watermark_type)
        
        # 3. 评估鲁棒性
        robustness_metrics = self.evaluate_robustness(model, watermarked_image, original_watermark, watermark_type)
        
        # 4. 生成综合报告
        report = {
            'Invisibility_Metrics': invisibility_metrics,
            'Extraction_Accuracy': extraction_accuracy,
            'Robustness_Metrics': robustness_metrics,
            'Overall_Evaluation': {
                'Invisibility_Passed': invisibility_metrics['PSNR_Threshold_Met'] and invisibility_metrics['SSIM_Threshold_Met'],
                'Extraction_Accuracy_Passed': extraction_accuracy >= 0.9,
                'Robustness_Passed': all(acc >= 0.8 for acc in robustness_metrics.values())
            }
        }
        
        return report
    
    def print_report(self, report):
        """
        打印评估报告
        
        Args:
            report: 评估报告字典
        """
        print("=" * 60)
        print("WATERMARK SYSTEM EVALUATION REPORT")
        print("=" * 60)
        
        # 不可见性评估
        print("\n1. Invisibility Evaluation:")
        print(f"   PSNR: {report['Invisibility_Metrics']['PSNR']:.2f} dB")
        print(f"   SSIM: {report['Invisibility_Metrics']['SSIM']:.4f}")
        print(f"   PSNR Threshold Met: {report['Invisibility_Metrics']['PSNR_Threshold_Met']}")
        print(f"   SSIM Threshold Met: {report['Invisibility_Metrics']['SSIM_Threshold_Met']}")
        
        # 提取准确率
        print("\n2. Extraction Accuracy:")
        print(f"   Accuracy: {report['Extraction_Accuracy']:.4f}")
        
        # 鲁棒性评估
        print("\n3. Robustness Evaluation:")
        for attack_type, accuracy in report['Robustness_Metrics'].items():
            print(f"   {attack_type.capitalize()}: {accuracy:.4f}")
        
        # 综合评估
        print("\n4. Overall Evaluation:")
        print(f"   Invisibility Passed: {report['Overall_Evaluation']['Invisibility_Passed']}")
        print(f"   Extraction Accuracy Passed: {report['Overall_Evaluation']['Extraction_Accuracy_Passed']}")
        print(f"   Robustness Passed: {report['Overall_Evaluation']['Robustness_Passed']}")
        
        # 总体结论
        all_passed = all(report['Overall_Evaluation'].values())
        print(f"\n5. Conclusion: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print("=" * 60)

if __name__ == "__main__":
    # 示例用法
    evaluator = PerformanceEvaluator()
    
    # 假设我们有以下文件
    original_image = "img/target.png"
    watermarked_image = "img/watermarked.png"
    original_watermark = "img/watermark.png"
    
    # 评估不可见性
    invisibility = evaluator.evaluate_invisibility(original_image, watermarked_image)
    print("Invisibility Metrics:", invisibility)
    
    # 注意：完整评估需要加载训练好的模型
    print("Performance evaluation module ready for use!")