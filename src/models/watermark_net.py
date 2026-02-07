import torch
import torch.nn as nn
from .encoder import WatermarkEncoder
from .decoder import WatermarkDecoder, MultiScaleDecoder
from .attack_simulator import AttackSimulator


class WatermarkNet(nn.Module):
    """完整的水印网络（编码器+攻击模拟+解码器）"""
    
    def __init__(
        self,
        image_channels: int = 3,
        watermark_channels: int = 3,
        hidden_dim: int = 64,
        wavelet: str = 'haar',
        attack_prob: float = 0.5,
        use_multiscale_decoder: bool = False,
        num_scales: int = 3
    ):
        super(WatermarkNet, self).__init__()
        
        # 编码器
        self.encoder = WatermarkEncoder(
            image_channels=image_channels,
            watermark_channels=watermark_channels,
            hidden_dim=hidden_dim,
            wavelet=wavelet
        )
        
        # 攻击模拟器
        self.attack_simulator = AttackSimulator(prob=attack_prob)
        
        # 解码器
        if use_multiscale_decoder:
            self.decoder = MultiScaleDecoder(
                image_channels=image_channels,
                watermark_channels=watermark_channels,
                hidden_dim=hidden_dim,
                num_scales=num_scales
            )
        else:
            self.decoder = WatermarkDecoder(
                image_channels=image_channels,
                watermark_channels=watermark_channels,
                hidden_dim=hidden_dim,
                wavelet=wavelet
            )
        
        self.use_multiscale = use_multiscale_decoder
    
    def forward(self, image, watermark, no_attack=False):
        """
        前向传播
        
        输入:
            image: 载体图像 (B, 3, H, W)
            watermark: 水印图像 (B, 3, H, W)
            no_attack: 是否跳过攻击模拟
        
        输出:
            watermarked_image: 含水印图像 (B, 3, H, W)
            attacked_image: 攻击后的图像 (B, 3, H, W)
            extracted_watermark: 提取的水印 (B, 3, H, W)
            confidence: 置信度 (B, 1)
        """
        # 编码：嵌入水印
        watermarked_image = self.encoder(image, watermark)
        
        # 攻击模拟
        if no_attack:
            attacked_image = watermarked_image
        else:
            attacked_image = self.attack_simulator(watermarked_image)
        
        # 解码：提取水印
        extracted_watermark, confidence = self.decoder(attacked_image)
        
        return watermarked_image, attacked_image, extracted_watermark, confidence
    
    def encode(self, image, watermark):
        """仅编码（嵌入水印）"""
        return self.encoder(image, watermark)
    
    def decode(self, watermarked_image):
        """仅解码（提取水印）"""
        return self.decoder(watermarked_image)
    
    def get_embedding_strength(self):
        """获取当前嵌入强度"""
        return torch.sigmoid(self.encoder.embedding_strength).item()


def test_watermark_net():
    """测试完整网络"""
    # 创建网络
    net = WatermarkNet(
        hidden_dim=64,
        attack_prob=0.5,
        use_multiscale_decoder=False
    )
    
    # 测试数据
    image = torch.randn(2, 3, 64, 64)
    watermark = torch.randn(2, 3, 64, 64)
    
    # 训练模式
    net.train()
    watermarked, attacked, extracted_wm, confidence = net(image, watermark)
    
    print("训练模式:")
    print(f"  输入图像: {image.shape}")
    print(f"  水印: {watermark.shape}")
    print(f"  含水印图像: {watermarked.shape}")
    print(f"  攻击后图像: {attacked.shape}")
    print(f"  提取水印: {extracted_wm.shape}")
    print(f"  置信度: {confidence.shape}")
    print(f"  嵌入强度: {net.get_embedding_strength():.4f}")
    
    # 计算图像失真
    mse_image = torch.mean((image - watermarked) ** 2)
    psnr = 10 * torch.log10(1.0 / (mse_image + 1e-8))
    print(f"  PSNR: {psnr:.2f} dB")
    
    # 评估模式
    net.eval()
    with torch.no_grad():
        watermarked, attacked, extracted_wm, confidence = net(image, watermark, no_attack=True)
    
    print("\n评估模式 (无攻击):")
    print(f"  置信度: {confidence.mean().item():.4f}")


if __name__ == "__main__":
    test_watermark_net()