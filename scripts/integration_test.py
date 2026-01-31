import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import config

class IntegrationTest:
    """
    集成测试
    验证整个水印系统的基本功能
    """
    
    def __init__(self):
        """
        初始化集成测试
        """
        pass
    
    def test_system_structure(self):
        """
        测试系统结构
        """
        print("Testing system structure...")
        
        # 检查必要的目录和文件
        required_dirs = [
            'src',
            'scripts',
            'data',
            'data/train/images',
            'data/train/watermark_img',
            'data/train/watermark_txt',
            'data/val/images',
            'data/val/watermark_img',
            'data/val/watermark_txt',
            'checkpoints',
            'output'
        ]
        
        required_files = [
            'config.py',
            'requirements.txt',
            'src/dwt.py',
            'src/models.py',
            'src/embedding.py',
            'src/extraction.py',
            'src/adversarial.py',
            'src/evaluation.py',
            'scripts/train.py',
            'scripts/test.py',
            'scripts/attack_test.py',
            'scripts/generate_text.py'
        ]
        
        all_dirs_exist = True
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                print(f"Missing directory: {dir_path}")
                all_dirs_exist = False
            else:
                print(f"Directory exists: {dir_path}")
        
        all_files_exist = True
        for file_path in required_files:
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                all_files_exist = False
            else:
                print(f"File exists: {file_path}")
        
        return all_dirs_exist and all_files_exist
    
    def test_dependencies(self):
        """
        测试依赖项
        """
        print("\nTesting dependencies...")
        
        try:
            import torch
            import numpy
            import cv2
            import PIL
            import skimage
            import pytorch_wavelets
            print("All dependencies imported successfully!")
            return True
        except ImportError as e:
            print(f"Dependency error: {e}")
            return False
    
    def run_basic_test(self):
        """
        运行基本测试
        """
        print("\nRunning basic functionality test...")
        
        try:
            # 测试DWT模块
            from src.dwt import DWTTransform
            dwt = DWTTransform()
            print("DWT module loaded successfully")
            
            # 测试模型模块
            from src.models import WatermarkModel
            model = WatermarkModel()
            print("Model module loaded successfully")
            
            # 测试嵌入模块
            from src.embedding import WatermarkEmbedding
            embedder = WatermarkEmbedding()
            print("Embedding module loaded successfully")
            
            # 测试提取模块
            from src.extraction import WatermarkExtraction
            extractor = WatermarkExtraction()
            print("Extraction module loaded successfully")
            
            # 测试评估模块
            from src.evaluation import PerformanceEvaluator
            evaluator = PerformanceEvaluator()
            print("Evaluation module loaded successfully")
            
            # 测试对抗性模块
            from src.adversarial import NoiseLayer
            noise_layer = NoiseLayer()
            print("Adversarial module loaded successfully")
            
            print("All modules loaded successfully!")
            return True
        except Exception as e:
            print(f"Module error: {e}")
            return False
    
    def run_full_test(self):
        """
        运行完整测试
        """
        print("=" * 60)
        print("INTEGRATION TEST REPORT")
        print("=" * 60)
        
        # 测试系统结构
        structure_ok = self.test_system_structure()
        
        # 测试依赖项
        dependencies_ok = self.test_dependencies()
        
        # 测试基本功能
        functionality_ok = self.run_basic_test()
        
        # 生成报告
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"System Structure: {'PASS' if structure_ok else 'FAIL'}")
        print(f"Dependencies: {'PASS' if dependencies_ok else 'FAIL'}")
        print(f"Functionality: {'PASS' if functionality_ok else 'FAIL'}")
        
        all_passed = structure_ok and dependencies_ok and functionality_ok
        print(f"\nOverall Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
        print("=" * 60)
        
        return all_passed

if __name__ == "__main__":
    tester = IntegrationTest()
    tester.run_full_test()