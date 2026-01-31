import os
import torch
from config import config
from src.embedding import WatermarkEmbedding
from src.extraction import WatermarkExtraction
from src.evaluation import PerformanceEvaluator
from src.adversarial import NoiseLayer, PGD, FGSM

class AttackTester:
    """
    攻击测试器
    测试水印系统在各种攻击下的鲁棒性
    """
    
    def __init__(self, watermark_type='image', checkpoint_path=None):
        """
        初始化攻击测试器
        
        Args:
            watermark_type: 水印类型
            checkpoint_path: 模型权重路径
        """
        self.watermark_type = watermark_type
        self.embedding = WatermarkEmbedding(watermark_type=watermark_type)
        self.extraction = WatermarkExtraction(watermark_type=watermark_type)
        self.evaluator = PerformanceEvaluator()
        self.noise_layer = NoiseLayer()
        
        # 加载模型权重
        if checkpoint_path:
            self.embedding.load_model(checkpoint_path)
            self.extraction.load_model(checkpoint_path)
        
        # 对抗性攻击
        self.pgd_attack = PGD(self.embedding.model)
        self.fgsm_attack = FGSM(self.embedding.model)
    
    def test_robustness(self, cover_image_path, watermark_input, output_dir='output/attack'):
        """
        测试系统鲁棒性
        
        Args:
            cover_image_path: 载体图像路径
            watermark_input: 水印输入
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 嵌入水印
        watermarked_path = os.path.join(output_dir, 'original_watermarked.png')
        watermarked_image = self.embedding.embed(cover_image_path, watermark_input, watermarked_path)
        print(f"Original watermarked image: {watermarked_path}")
        
        # 2. 测试各种攻击
        attacks = [
            ('gaussian', 'Gaussian Noise'),
            ('jpeg', 'JPEG Compression'),
            ('crop', 'Random Crop'),
            ('blur', 'Gaussian Blur'),
            ('all', 'All Attacks')
        ]
        
        robustness_results = {}
        
        for attack_type, attack_name in attacks:
            print(f"\nTesting {attack_name}...")
            
            # 应用攻击
            attacked_path = os.path.join(output_dir, f'attacked_{attack_type}.png')
            
            # 加载并预处理图像
            from src.extraction import WatermarkExtraction
            extractor = WatermarkExtraction(watermark_type=self.watermark_type)
            watermarked_tensor = extractor.preprocess_watermarked_image(watermarked_path)
            watermarked_tensor = watermarked_tensor.to(config.DEVICE)
            
            # 应用噪声层
            attacked_tensor = self.noise_layer(watermarked_tensor, attack_type)
            
            # 保存攻击后的图像
            from src.embedding import WatermarkEmbedding
            embedder = WatermarkEmbedding(watermark_type=self.watermark_type)
            embedder.save_image(attacked_tensor, attacked_path)
            print(f"Attacked image saved: {attacked_path}")
            
            # 提取水印
            extracted_path = os.path.join(output_dir, f'extracted_{attack_type}.png') if self.watermark_type == 'image' else None
            extracted_watermark = self.extraction.extract(attacked_path, extracted_path)
            print(f"Extracted watermark: {extracted_path}")
            
            # 计算提取准确率
            accuracy = self.evaluator.evaluate_extraction_accuracy(
                extracted_watermark,
                watermark_input,
                self.watermark_type
            )
            
            robustness_results[attack_type] = accuracy
            print(f"Extraction Accuracy: {accuracy:.4f}")
        
        # 3. 测试对抗性攻击
        print("\nTesting Adversarial Attacks...")
        
        # PGD攻击
        print("PGD Attack...")
        pgd_path = os.path.join(output_dir, 'attacked_pgd.png')
        
        # 加载水印
        if self.watermark_type == 'image':
            watermark_tensor = self.embedding.preprocess_image_watermark(watermark_input)
        else:
            watermark_tensor = self.embedding.preprocess_text_watermark(watermark_input)
        watermark_tensor = watermark_tensor.to(config.DEVICE)
        
        # 生成对抗样本
        adversarial_image = self.pgd_attack.attack(watermarked_tensor, watermark_tensor)
        
        # 保存对抗样本
        self.embedding.save_image(adversarial_image, pgd_path)
        print(f"PGD adversarial image: {pgd_path}")
        
        # 提取水印
        extracted_pgd_path = os.path.join(output_dir, 'extracted_pgd.png') if self.watermark_type == 'image' else None
        extracted_pgd = self.extraction.extract(pgd_path, extracted_pgd_path)
        
        # 计算准确率
        pgd_accuracy = self.evaluator.evaluate_extraction_accuracy(
            extracted_pgd,
            watermark_input,
            self.watermark_type
        )
        robustness_results['pgd'] = pgd_accuracy
        print(f"PGD Attack Accuracy: {pgd_accuracy:.4f}")
        
        # 4. 生成鲁棒性报告
        self.generate_robustness_report(robustness_results)
        
        return robustness_results
    
    def generate_robustness_report(self, robustness_results):
        """
        生成鲁棒性报告
        
        Args:
            robustness_results: 鲁棒性测试结果
        """
        print("\n" + "=" * 60)
        print("ROBUSTNESS TEST REPORT")
        print("=" * 60)
        
        for attack, accuracy in robustness_results.items():
            print(f"{attack.capitalize()}: {accuracy:.4f}")
        
        # 计算平均鲁棒性
        avg_robustness = sum(robustness_results.values()) / len(robustness_results)
        print(f"\nAverage Robustness: {avg_robustness:.4f}")
        
        # 评估是否通过
        passed = all(acc >= 0.8 for acc in robustness_results.values())
        print(f"\nConclusion: {'ALL ROBUSTNESS TESTS PASSED' if passed else 'SOME ROBUSTNESS TESTS FAILED'}")
        print("=" * 60)

def main():
    """
    主攻击测试函数
    """
    # 配置
    watermark_type = 'image'  # 或 'text'
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_{watermark_type}.pth")
    
    # 创建测试器
    tester = AttackTester(watermark_type=watermark_type, checkpoint_path=checkpoint_path)
    
    # 测试鲁棒性
    cover_image = "img/target.png"
    watermark_input = "img/watermark.png" if watermark_type == 'image' else "Test watermark text"
    tester.test_robustness(cover_image, watermark_input)

if __name__ == "__main__":
    main()