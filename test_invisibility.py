# 水印不可见性对比测试脚本
import torch
from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from config import Config
from PIL import Image
import numpy as np
from loss_functions import calculate_psnr, calculate_ssim
import matplotlib.pyplot as plt

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

def test_watermark_invisibility():
    """测试水印不可见性"""
    print("=== 水印不可见性对比测试 ===")
    print()
    
    # 初始化模型
    encoder = WatermarkEncoder().to(Config.DEVICE)
    decoder = WatermarkDecoder().to(Config.DEVICE)
    
    # 加载测试图像
    test_target = [
        "img/target.png",
        "img/white_target.png"
    ]
    test_watermark = [
        "img/watermark.png",
        "img/white_watermark.png",
        "img/black_watermark.png",
        "img/color_watermark.png",
        "img/watermark_1.png",
        "img/watermark_2.png",
        "img/watermark_3.png"
    ]
    
    results = []
    
    for img_path in test_target:
        print(f"测试图像: {img_path}")
        img = load_image(img_path)
        print(f"输入图像尺寸: {img.shape[2]}x{img.shape[3]}")
        
        # 生成随机水印
        random_index = random.randint(0, len(test_watermark) - 1)
        msg = load_image(test_watermark[random_index])
        print(f"水印尺寸: {msg.shape[2]}x{msg.shape[3]}")
        
        # 嵌入水印
        encoded_img = encoder(img, msg)
        print(f"嵌入后图像尺寸: {encoded_img.shape[2]}x{encoded_img.shape[3]}")
        
        # 计算视觉质量指标
        psnr = calculate_psnr(encoded_img, img).item()
        ssim = calculate_ssim(encoded_img, img).item()
        
        # 计算MSE
        mse = torch.mean((encoded_img - img) ** 2).item()
        
        # 计算最大差异
        max_diff = torch.max(torch.abs(encoded_img - img)).item()
        mean_diff = torch.mean(torch.abs(encoded_img - img)).item()
        
        print(f"视觉质量指标:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
        print(f"  MSE: {mse:.6f}")
        print(f"  最大差异: {max_diff:.6f}")
        print(f"  平均差异: {mean_diff:.6f}")
        
        # 保存结果图像
        output_path = f"watermarked_{img_path.split('/')[-1]}"
        save_image(encoded_img, output_path)
        print(f"保存嵌入后图像到: {output_path}")
        
        # 提取水印
        extracted_msg = decoder(encoded_img)
        
        # 计算水印提取准确率
        msg_pred = (extracted_msg > 0.5).float()
        accuracy = (msg_pred == msg).float().mean().item()
        print(f"水印提取准确率: {accuracy:.4f}")
        
        # 保存结果
        results.append({
            'image': img_path,
            'psnr': psnr,
            'ssim': ssim,
            'mse': mse,
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'accuracy': accuracy
        })
        
        print()
    
    # 生成对比报告
    print("=== 对比报告 ===")
    print()
    print(f"{'图像':<30} {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<12} {'准确率':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result['image']:<30} {result['psnr']:<12.2f} {result['ssim']:<10.4f} {result['mse']:<12.6f} {result['accuracy']:<10.4f}")
    
    print()
    print("=== 质量评估 ===")
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    avg_mse = np.mean([r['mse'] for r in results])
    avg_accuracy = np.mean([r['accuracy'] for r in results])
    
    print(f"平均 PSNR: {avg_psnr:.2f} dB")
    print(f"平均 SSIM: {avg_ssim:.4f}")
    print(f"平均 MSE: {avg_mse:.6f}")
    print(f"平均准确率: {avg_accuracy:.4f}")
    
    print()
    print("=== 不可见性评估 ===")
    if avg_psnr >= 40:
        print("✓ PSNR >= 40dB: 优秀 - 水印几乎不可见")
    elif avg_psnr >= 35:
        print("✓ PSNR >= 35dB: 良好 - 水印难以察觉")
    elif avg_psnr >= 30:
        print("△ PSNR >= 30dB: 一般 - 水印可能可见")
    else:
        print("✗ PSNR < 30dB: 较差 - 水印明显可见")
    
    if avg_ssim >= 0.95:
        print("✓ SSIM >= 0.95: 优秀 - 结构保持良好")
    elif avg_ssim >= 0.90:
        print("✓ SSIM >= 0.90: 良好 - 结构基本保持")
    elif avg_ssim >= 0.80:
        print("△ SSIM >= 0.80: 一般 - 结构有一定损失")
    else:
        print("✗ SSIM < 0.80: 较差 - 结构损失明显")
    
    if avg_accuracy >= 0.95:
        print("✓ 准确率 >= 95%: 优秀 - 水印提取准确")
    elif avg_accuracy >= 0.90:
        print("✓ 准确率 >= 90%: 良好 - 水印提取基本准确")
    elif avg_accuracy >= 0.80:
        print("△ 准确率 >= 80%: 一般 - 水印提取有一定误差")
    else:
        print("✗ 准确率 < 80%: 较差 - 水印提取不准确")

if __name__ == "__main__":
    test_watermark_invisibility()