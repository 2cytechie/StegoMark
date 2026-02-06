"""
隐写网络 - 完整的编码器+解码器
"""
import torch
import torch.nn as nn

from .encoder import Encoder, SimpleEncoder
from .decoder import Decoder, SimpleDecoder


class SteganographyNet(nn.Module):
    """隐写网络 - 端到端水印嵌入和提取
    将水印嵌入到LH、HL、HH三个频段
    """
    
    def __init__(self, use_dwt=True, wavelet='haar', mode='symmetric',
                 use_stn=True, base_channels=64, num_blocks=4, watermark_size=64,
                 use_grouped_conv=False, num_blocks_decoder=None):
        """
        Args:
            use_dwt: 是否使用DWT
            wavelet: 小波基
            mode: 边界处理模式
            use_stn: 是否使用STN
            base_channels: 基础通道数
            num_blocks: 残差块数量（编码器）
            watermark_size: 水印尺寸
            use_grouped_conv: 是否使用分组卷积共享权重
            num_blocks_decoder: 解码器残差块数量（默认等于num_blocks）
        """
        super().__init__()
        
        self.use_dwt = use_dwt
        self.watermark_size = watermark_size
        
        # 如果未指定解码器块数，使用与编码器相同
        if num_blocks_decoder is None:
            num_blocks_decoder = num_blocks
        
        # 编码器
        if use_dwt:
            self.encoder = Encoder(
                wavelet=wavelet,
                mode=mode,
                base_channels=base_channels,
                num_blocks=num_blocks,
                use_grouped_conv=use_grouped_conv
            )
            self.decoder = Decoder(
                wavelet=wavelet,
                mode=mode,
                use_stn=use_stn,
                base_channels=base_channels,
                num_blocks=num_blocks_decoder,
                watermark_size=watermark_size,
                use_grouped_conv=use_grouped_conv
            )
        else:
            self.encoder = SimpleEncoder(
                base_channels=base_channels,
                num_blocks=num_blocks
            )
            self.decoder = SimpleDecoder(
                base_channels=base_channels,
                num_blocks=num_blocks_decoder,
                watermark_size=watermark_size
            )
    
    def forward(self, image, watermark, return_dwt=False):
        """
        前向传播 - 嵌入和提取
        
        Args:
            image: 目标图片 [B, 3, H, W]
            watermark: 水印 [B, 3, H, W] (平铺后的)
            return_dwt: 是否返回DWT频段
            
        Returns:
            watermarked_image: 含水印图片 [B, 3, H, W]
            extracted_watermark: 提取的水印 [B, 3, watermark_size, watermark_size]
            dwt_bands: DWT频段 (如果return_dwt=True)
        """
        # 编码：嵌入水印
        watermarked_image, dwt_bands = self.encoder(image, watermark)
        
        # 解码：提取水印
        extracted_watermark, stn_theta = self.decoder(watermarked_image, dwt_bands)
        
        if return_dwt:
            return watermarked_image, extracted_watermark, dwt_bands
        
        return watermarked_image, extracted_watermark
    
    def encode(self, image, watermark):
        """
        仅嵌入水印
        
        Args:
            image: 目标图片 [B, 3, H, W]
            watermark: 水印 [B, 3, H, W]
            
        Returns:
            watermarked_image: 含水印图片 [B, 3, H, W]
        """
        watermarked_image, _ = self.encoder(image, watermark)
        return watermarked_image
    
    def decode(self, watermarked_image):
        """
        仅提取水印
        
        Args:
            watermarked_image: 含水印图片 [B, 3, H, W]
            
        Returns:
            extracted_watermark: 提取的水印 [B, 3, watermark_size, watermark_size]
        """
        extracted_watermark, _ = self.decoder(watermarked_image, None)
        return extracted_watermark


class Discriminator(nn.Module):
    """判别器 - 用于对抗训练 (可选)"""
    
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # 输入层
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 下采样
            nn.Conv2d(base_channels, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_channels*4, base_channels*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出层
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_channels*8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        判别图像是否含水印
        
        Args:
            x: 图像 [B, 3, H, W]
            
        Returns:
            prob: 含水印概率 [B, 1]
        """
        return self.model(x)


def create_model(config=None):
    """
    根据配置创建模型
    
    Args:
        config: 配置对象，包含model_config和training_config（可选，默认使用全局配置）
        
    Returns:
        model: SteganographyNet实例
    """
    from configs.config import model_config, data_config
    
    model = SteganographyNet(
        use_dwt=True,
        wavelet=model_config.wavelet,
        mode=model_config.mode,
        use_stn=model_config.use_stn,
        base_channels=model_config.encoder_channels,
        num_blocks=model_config.num_blocks,
        num_blocks_decoder=model_config.num_blocks_decoder,
        watermark_size=data_config.watermark_size,
        use_grouped_conv=model_config.use_grouped_conv
    )
    
    return model
