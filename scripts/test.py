import os
import sys
import argparse
import torch
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 强制使用CPU设备，避免设备不匹配错误
import torch
torch.device('cpu')

from config import config
# 覆盖配置，强制使用CPU
config.DEVICE = 'cpu'

from src.embedding import WatermarkEmbedding
from src.extraction import WatermarkExtraction
from src.evaluation import PerformanceEvaluator
from src.adversarial import NoiseLayer

class WatermarkTester:
    """
    水印系统测试器
    """
    
    def __init__(self, watermark_type='image', checkpoint_path=None):
        """
        初始化测试器
        
        Args:
            watermark_type: 水印类型
            checkpoint_path: 模型权重路径
        """
        self.watermark_type = watermark_type
        print(f"Using device: {config.DEVICE}")
        
        try:
            self.embedding = WatermarkEmbedding(watermark_type=watermark_type)
            self.extraction = WatermarkExtraction(watermark_type=watermark_type)
            self.evaluator = PerformanceEvaluator()
            self.noise_layer = NoiseLayer()
            
            # 加载模型权重
            if checkpoint_path:
                if os.path.exists(checkpoint_path):
                    self.embedding.load_model(checkpoint_path)
                    self.extraction.load_model(checkpoint_path)
                    print(f"Model loaded successfully from {checkpoint_path}")
                else:
                    print(f"Warning: Model checkpoint not found at {checkpoint_path}")
        except Exception as e:
            print(f"Error initializing WatermarkTester: {str(e)}")
            raise
    
    def test_single_image(self, cover_image_path, watermark_input, output_dir='output', watermarked_filename='watermarked.png'):
        """
        测试单个图像
        
        Args:
            cover_image_path: 载体图像路径
            watermark_input: 水印输入
            output_dir: 输出目录
            watermarked_filename: 水印图像输出文件名
        """
        try:
            # 检查输入文件是否存在
            if not os.path.exists(cover_image_path):
                raise FileNotFoundError(f"Cover image not found: {cover_image_path}")
            
            if self.watermark_type == 'image' and not os.path.exists(watermark_input):
                raise FileNotFoundError(f"Watermark image not found: {watermark_input}")
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. 嵌入水印
            watermarked_path = os.path.join(output_dir, watermarked_filename)
            watermarked_image = self.embedding.embed(cover_image_path, watermark_input, watermarked_path)
            print(f"Watermark embedded: {watermarked_path}")
            
            # 2. 提取水印
            extracted_path = os.path.join(output_dir, 'extracted_watermark.png') if self.watermark_type == 'image' else None
            extracted_watermark = self.extraction.extract(watermarked_path, extracted_path)
            print(f"Watermark extracted: {extracted_path}")
            
            # 3. 评估性能
            try:
                # 单独计算PSNR和SSIM
                psnr_value = self.evaluator.calculate_psnr(cover_image_path, watermarked_path)
                ssim_value = self.evaluator.calculate_ssim(cover_image_path, watermarked_path)
                extraction_accuracy = self.evaluator.evaluate_extraction_accuracy(extracted_watermark, watermark_input, self.watermark_type)
                
                # 创建详细报告
                report = {
                    'Invisibility_Metrics': {
                        'PSNR': psnr_value,
                        'SSIM': ssim_value,
                        'PSNR_Threshold_Met': psnr_value >= config.PSNR_THRESHOLD,
                        'SSIM_Threshold_Met': ssim_value >= config.SSIM_THRESHOLD
                    },
                    'Extraction_Accuracy': extraction_accuracy,
                    'Robustness_Metrics': {},
                    'Overall_Evaluation': {
                        'Invisibility_Passed': psnr_value >= config.PSNR_THRESHOLD and ssim_value >= config.SSIM_THRESHOLD,
                        'Extraction_Accuracy_Passed': extraction_accuracy >= 0.9,
                        'Robustness_Passed': False
                    }
                }
                print("Generated detailed report successfully")
            except Exception as e:
                print(f"Error generating evaluation report: {str(e)}")
                # 如果评估失败，创建一个基本报告
                report = {
                    'Invisibility_Metrics': {
                        'PSNR': 0.0,
                        'SSIM': 0.0,
                        'PSNR_Threshold_Met': False,
                        'SSIM_Threshold_Met': False
                    },
                    'Extraction_Accuracy': 0.0,
                    'Robustness_Metrics': {},
                    'Overall_Evaluation': {
                        'Invisibility_Passed': False,
                        'Extraction_Accuracy_Passed': False,
                        'Robustness_Passed': False
                    }
                }
                print("Generated basic report due to evaluation error")
            
            # 4. 打印评估报告
            self.evaluator.print_report(report)
            
            # 5. 特别输出SSIM等图像质量评估指标
            print("\n=== Image Quality Metrics ===")
            print(f"PSNR: {report['Invisibility_Metrics']['PSNR']:.2f} dB")
            print(f"SSIM: {report['Invisibility_Metrics']['SSIM']:.4f}")
            print(f"Extraction Accuracy: {report['Extraction_Accuracy']:.4f}")
            print("=============================")
            
            # 6. 应用各种攻击并保存结果
            print("\n=== Applying Attacks ===")
            
            # 加载水印图像
            watermarked_pil = Image.open(watermarked_path).convert('RGB')
            watermarked_np = np.array(watermarked_pil) / 255.0
            watermarked_tensor = torch.tensor(watermarked_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            
            # 定义要应用的攻击
            attacks = [
                ('gaussian', 'gaussian'),
                ('jpeg', 'jpeg'),
                ('crop', 'crop'),
                ('blur', 'blur'),
                ('rotate', 'rotate'),
                ('scale', 'scale'),
            ]
            
            for attack_name, attack_type in attacks:
                try:
                    print(f"Applying {attack_name} attack...")
                    
                    # 应用攻击
                    if attack_type == 'gaussian':
                        attacked_tensor = self.noise_layer.add_gaussian_noise(watermarked_tensor)
                    elif attack_type == 'jpeg':
                        attacked_tensor = self.noise_layer.add_jpeg_compression(watermarked_tensor)
                    elif attack_type == 'crop':
                        attacked_tensor = self.noise_layer.add_random_crop(watermarked_tensor)
                    elif attack_type == 'blur':
                        attacked_tensor = self.noise_layer.add_gaussian_blur(watermarked_tensor)
                    elif attack_type == 'rotate':
                        attacked_tensor = self.noise_layer.add_rotation(watermarked_tensor)
                    elif attack_type == 'scale':
                        attacked_tensor = self.noise_layer.add_scaling(watermarked_tensor)
                    else:
                        continue
                    
                    # 转换回PIL图像并保存
                    attacked_np = attacked_tensor.squeeze(0).permute(1, 2, 0).numpy()
                    attacked_np = (attacked_np * 255).astype(np.uint8)
                    attacked_pil = Image.fromarray(attacked_np)
                    attacked_path = os.path.join(output_dir, f"watermarked_{attack_name}.png")
                    attacked_pil.save(attacked_path)
                    print(f"Saved attacked image: {attacked_path}")
                    
                    # 从攻击后的图像中提取水印
                    extracted_attacked_path = os.path.join(output_dir, f"extracted_watermark_{attack_name}.png")
                    extracted_attacked_watermark = self.extraction.extract(attacked_path, extracted_attacked_path)
                    print(f"Saved extracted watermark from {attack_name} attack: {extracted_attacked_path}")
                    
                except Exception as e:
                    print(f"Error processing {attack_name} attack: {str(e)}")
                    continue
            
            print("=============================")
            
            return report
        except Exception as e:
            print(f"Error testing single image: {str(e)}")
            raise
    
    def test_batch_images(self, cover_images_dir, watermarks_input, output_dir='output/batch'):
        """
        批量测试图像
        
        Args:
            cover_images_dir: 载体图像目录
            watermarks_input: 水印输入列表
            output_dir: 输出目录
        """
        try:
            if not os.path.exists(cover_images_dir):
                raise FileNotFoundError(f"Cover images directory not found: {cover_images_dir}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取载体图像列表
            cover_images = [f for f in os.listdir(cover_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            if not cover_images:
                raise ValueError(f"No image files found in directory: {cover_images_dir}")
            
            reports = []
            
            for i, img_name in enumerate(cover_images[:10]):  # 测试前10张
                print(f"Testing image {i+1}/{len(cover_images[:10])}: {img_name}")
                
                cover_path = os.path.join(cover_images_dir, img_name)
                watermark_input = watermarks_input[i % len(watermarks_input)]
                
                # 创建输出子目录
                img_output_dir = os.path.join(output_dir, f"test_{i+1}")
                os.makedirs(img_output_dir, exist_ok=True)
                
                # 测试单个图像
                report = self.test_single_image(cover_path, watermark_input, img_output_dir)
                reports.append(report)
            
            # 生成批量测试报告
            self.generate_batch_report(reports)
            
            return reports
        except Exception as e:
            print(f"Error testing batch images: {str(e)}")
            raise
    
    def generate_batch_report(self, reports):
        """
        生成批量测试报告
        
        Args:
            reports: 单个测试报告列表
        """
        print("\n" + "=" * 60)
        print("BATCH TEST REPORT")
        print("=" * 60)
        
        # 计算平均值
        avg_psnr = sum(r['Invisibility_Metrics']['PSNR'] for r in reports) / len(reports)
        avg_ssim = sum(r['Invisibility_Metrics']['SSIM'] for r in reports) / len(reports)
        avg_accuracy = sum(r['Extraction_Accuracy'] for r in reports) / len(reports)
        
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average Extraction Accuracy: {avg_accuracy:.4f}")
        
        # 计算通过测试的比例
        invisibility_passed = sum(r['Overall_Evaluation']['Invisibility_Passed'] for r in reports)
        accuracy_passed = sum(r['Overall_Evaluation']['Extraction_Accuracy_Passed'] for r in reports)
        robustness_passed = sum(r['Overall_Evaluation']['Robustness_Passed'] for r in reports)
        
        print(f"\nTests Passed:")
        print(f"Invisibility: {invisibility_passed}/{len(reports)} ({invisibility_passed/len(reports)*100:.1f}%)")
        print(f"Extraction Accuracy: {accuracy_passed}/{len(reports)} ({accuracy_passed/len(reports)*100:.1f}%)")
        print(f"Robustness: {robustness_passed}/{len(reports)} ({robustness_passed/len(reports)*100:.1f}%)")
        
        print("=" * 60)


def main():
    """
    主测试函数
    """
    try:
        # 配置
        watermark_type = config.WATERMARK_TYPES
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, config.WATERMARK_MODEL)
        
        # 创建测试器
        tester = WatermarkTester(watermark_type=watermark_type, checkpoint_path=checkpoint_path)
        
        # 执行测试
        # 批量测试
        # target_dir = [
        #     "img/target.png",
        #     "img/target2.png",
        # ]
        # watermark_list = [
        #     "img/watermark.png",
        #     "img/watermark2.png",
        # ]
        # output_dir = "output/batch"
        # tester.test_batch_images(target_dir, watermark_list, output_dir)
        
        # 单个测试
        cover_image = "img/img9.jpg"
        watermark_input = "img/watermark.png" if watermark_type == 'image' else "Test watermark text"
        tester.test_single_image(cover_image, watermark_input, config.OUTPUT_DIR, "test_single_image.png")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()