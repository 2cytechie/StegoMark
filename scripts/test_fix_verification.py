import os
import sys
from PIL import Image

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 强制使用CPU设备
import torch
torch.device('cpu')

from config import config
# 覆盖配置，强制使用CPU
config.DEVICE = 'cpu'

from src.embedding import WatermarkEmbedding
from src.extraction import WatermarkExtraction

class FixVerificationTester:
    """
    修复效果验证测试器
    """
    
    def __init__(self):
        """
        初始化测试器
        """
        self.embedding = WatermarkEmbedding(watermark_type='image')
        self.extraction = WatermarkExtraction(watermark_type='image')
        
        # 加载模型权重
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"model_image_epoch_5.pth")
        if os.path.exists(checkpoint_path):
            self.embedding.load_model(checkpoint_path)
            self.extraction.load_model(checkpoint_path)
            print(f"Model loaded successfully from {checkpoint_path}")
        else:
            print(f"Warning: Model checkpoint not found at {checkpoint_path}")
    
    def test_image(self, cover_image_path, watermark_path, output_dir='output/verification'):
        """
        测试单个图像
        
        Args:
            cover_image_path: 载体图像路径
            watermark_path: 水印图像路径
            output_dir: 输出目录
        """
        try:
            # 检查输入文件是否存在
            if not os.path.exists(cover_image_path):
                raise FileNotFoundError(f"Cover image not found: {cover_image_path}")
            
            if not os.path.exists(watermark_path):
                raise FileNotFoundError(f"Watermark image not found: {watermark_path}")
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 获取原始图像尺寸
            original_image = Image.open(cover_image_path)
            original_size = original_image.size
            print(f"Original image size: {original_size}")
            
            # 生成输出文件名
            cover_filename = os.path.basename(cover_image_path)
            output_filename = f"watermarked_{cover_filename}"
            watermarked_path = os.path.join(output_dir, output_filename)
            
            # 嵌入水印
            print(f"Embedding watermark into {cover_filename}...")
            watermarked_image = self.embedding.embed(cover_image_path, watermark_path, watermarked_path)
            print(f"Watermark embedded: {watermarked_path}")
            
            # 检查输出图像尺寸
            output_image = Image.open(watermarked_path)
            output_size = output_image.size
            print(f"Output image size: {output_size}")
            
            # 验证尺寸是否一致
            size_match = original_size == output_size
            print(f"Size match: {size_match}")
            
            # 提取水印
            extracted_path = os.path.join(output_dir, f"extracted_{cover_filename}")
            extracted_watermark = self.extraction.extract(watermarked_path, extracted_path)
            print(f"Watermark extracted: {extracted_path}")
            
            print("=" * 60)
            return size_match
            
        except Exception as e:
            print(f"Error testing image: {str(e)}")
            return False
    
    def run_comprehensive_test(self):
        """
        运行综合测试
        """
        print("Running comprehensive fix verification test...")
        print("=" * 60)
        
        # 测试图像列表
        test_images = [
            "img/target.png",
            "img/target2.png",
            "img/watermark.png",
            "img/watermark2.png"
        ]
        
        watermark_path = "img/watermark.png"
        output_dir = "output/verification"
        
        # 确保测试目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 运行测试
        results = []
        for cover_image_path in test_images:
            if os.path.exists(cover_image_path):
                print(f"Testing: {cover_image_path}")
                size_match = self.test_image(cover_image_path, watermark_path, output_dir)
                results.append((cover_image_path, size_match))
            else:
                print(f"Skipping: {cover_image_path} (file not found)")
        
        # 生成测试报告
        print("\n" + "=" * 60)
        print("TEST REPORT")
        print("=" * 60)
        
        passed = 0
        total = len(results)
        
        for image_path, size_match in results:
            status = "PASS" if size_match else "FAIL"
            print(f"{os.path.basename(image_path)}: {status}")
            if size_match:
                passed += 1
        
        print(f"\nTotal: {total}, Passed: {passed}, Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        print("=" * 60)
        
        return passed == total

def main():
    """
    主测试函数
    """
    try:
        tester = FixVerificationTester()
        success = tester.run_comprehensive_test()
        
        if success:
            print("All tests passed! Fix verification successful.")
        else:
            print("Some tests failed. Please check the output.")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
