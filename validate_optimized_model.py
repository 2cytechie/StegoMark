import torch
import numpy as np
from PIL import Image
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.models import WatermarkModel
from src.evaluation import PerformanceEvaluator
from src.adversarial import NoiseLayer

def validate_model():
    """
    验证优化后的模型性能
    """
    print("=" * 60)
    print("VALIDATING OPTIMIZED WATERMARK MODEL")
    print("=" * 60)
    
    # 1. 加载模型
    print("\n1. Loading optimized model...")
    device = torch.device(config.DEVICE)
    model = WatermarkModel(watermark_type='image', device=device)
    model.to(device)
    
    # 加载最佳模型权重
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model_image.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device)['model_state_dict'])
        print(f"Loaded best model from: {best_model_path}")
    else:
        print(f"Warning: Best model not found at {best_model_path}")
        return
    
    # 2. 准备测试数据
    print("\n2. Preparing test data...")
    test_images = ["img/img1.jpg", "img/img2.jpg", "img/img3.jpg"]
    test_watermarks = ["img/watermark.png", "img/watermark1.png"]
    
    # 确保测试文件存在
    valid_images = []
    for img_path in test_images:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_path)
        if os.path.exists(full_path):
            valid_images.append(full_path)
        else:
            print(f"Warning: Test image not found: {full_path}")
    
    valid_watermarks = []
    for wm_path in test_watermarks:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), wm_path)
        if os.path.exists(full_path):
            valid_watermarks.append(full_path)
        else:
            print(f"Warning: Test watermark not found: {full_path}")
    
    if not valid_images or not valid_watermarks:
        print("Error: No valid test images or watermarks found")
        return
    
    # 3. 初始化评估器
    evaluator = PerformanceEvaluator()
    noise_layer = NoiseLayer()
    
    # 4. 执行验证
    print("\n3. Executing validation...")
    results = []
    
    for img_path in valid_images:
        for wm_path in valid_watermarks:
            print(f"\nProcessing image: {os.path.basename(img_path)} with watermark: {os.path.basename(wm_path)}")
            
            # 加载图像和水印
            image = Image.open(img_path).convert('RGB')
            image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.LANCZOS)
            image = np.array(image) / 255.0
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 0).unsqueeze(0).to(device)
            
            watermark = Image.open(wm_path).convert('L')
            watermark = watermark.resize((config.WATERMARK_SIZE, config.WATERMARK_SIZE), Image.LANCZOS)
            watermark = np.array(watermark) / 255.0
            watermark = torch.tensor(watermark, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # 嵌入水印
            with torch.no_grad():
                watermarked_image = model.embed(image, watermark)
            
            # 提取水印
            with torch.no_grad():
                extracted_watermark = model.extract(watermarked_image)
            
            # 评估不可见性
            invisibility_metrics = evaluator.evaluate_invisibility(image, watermarked_image)
            print(f"Invisibility Metrics: {invisibility_metrics}")
            
            # 评估提取准确率
            accuracy = evaluator.evaluate_extraction_accuracy(extracted_watermark, watermark, watermark_type='image')
            print(f"Extraction Accuracy: {accuracy:.4f}")
            
            # 评估鲁棒性
            robustness_metrics = evaluator.evaluate_robustness(model, watermarked_image, watermark, watermark_type='image')
            print(f"Robustness Metrics: {robustness_metrics}")
            
            # 生成评估报告
            report = evaluator.generate_evaluation_report(
                model, image, watermarked_image, watermark, extracted_watermark, watermark_type='image'
            )
            
            # 打印报告
            evaluator.print_report(report)
            
            # 保存结果
            results.append({
                'image': os.path.basename(img_path),
                'watermark': os.path.basename(wm_path),
                'invisibility': invisibility_metrics,
                'accuracy': accuracy,
                'robustness': robustness_metrics,
                'report': report
            })
    
    # 5. 分析结果
    print("\n4. Analyzing results...")
    
    # 计算平均指标
    avg_psnr = []
    avg_ssim = []
    avg_accuracy = []
    avg_robustness = []
    
    for result in results:
        avg_psnr.append(result['invisibility']['PSNR'])
        avg_ssim.append(result['invisibility']['SSIM'])
        avg_accuracy.append(result['accuracy'])
        for acc in result['robustness'].values():
            avg_robustness.append(acc)
    
    avg_psnr = sum(avg_psnr) / len(avg_psnr) if avg_psnr else 0
    avg_ssim = sum(avg_ssim) / len(avg_ssim) if avg_ssim else 0
    avg_accuracy = sum(avg_accuracy) / len(avg_accuracy) if avg_accuracy else 0
    avg_robustness = sum(avg_robustness) / len(avg_robustness) if avg_robustness else 0
    
    print("\n" + "=" * 60)
    print("AVERAGE PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average Extraction Accuracy: {avg_accuracy:.4f}")
    print(f"Average Robustness: {avg_robustness:.4f}")
    
    # 6. 评估过拟合情况
    print("\n5. Evaluating overfitting...")
    
    # 计算训练和验证指标的差异
    # 注意：这里我们假设已经有训练日志，如果没有，我们可以基于验证结果进行简单评估
    
    # 基于验证结果的过拟合评估
    if avg_accuracy > 0.9 and avg_robustness < 0.7:
        print("Warning: Possible overfitting detected - high extraction accuracy but low robustness")
    elif avg_accuracy < 0.7 and avg_robustness < 0.7:
        print("Warning: Model may be underfitting - both accuracy and robustness are low")
    else:
        print("Good generalization - balanced accuracy and robustness")
    
    print("\nValidation completed successfully!")

if __name__ == "__main__":
    validate_model()
