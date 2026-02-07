import torch
import torch.nn as nn
import torch.nn.functional as F
from ..dwt import DWT2D, IDWT2D


class ConvBlock(nn.Module):
    """卷积块"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    """注意力块"""
    
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn = self.attention(x)
        return x * attn


class WatermarkDecoder(nn.Module):
    """水印解码器 - 从含水印图像中提取水印"""
    
    def __init__(
        self,
        image_channels: int = 3,
        watermark_channels: int = 3,
        hidden_dim: int = 64,
        wavelet: str = 'haar'
    ):
        super(WatermarkDecoder, self).__init__()
        
        self.image_channels = image_channels
        self.watermark_channels = watermark_channels
        self.hidden_dim = hidden_dim
        
        # DWT变换
        self.dwt = DWT2D(wavelet)
        
        # 频域特征提取网络
        self.freq_encoder = nn.Sequential(
            ConvBlock(image_channels * 4, hidden_dim),  # DWT后有4个子带
            ConvBlock(hidden_dim, hidden_dim),
            AttentionBlock(hidden_dim),
        )
        
        # 空间域特征提取网络
        self.spatial_encoder = nn.Sequential(
            ConvBlock(image_channels, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
            AttentionBlock(hidden_dim),
        )
        
        # 双分支融合
        self.fusion = nn.Sequential(
            ConvBlock(hidden_dim * 2, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim),
        )
        
        # 水印重建网络
        self.decoder = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            ConvBlock(hidden_dim, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, watermark_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 置信度预测网络
        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, watermarked_image):
        """
        输入:
            watermarked_image: 含水印图像 (B, 3, H, W)
        输出:
            extracted_watermark: 提取的水印 (B, 3, H, W)
            confidence: 置信度分数 (B, 1)
        """
        b, c, h, w = watermarked_image.shape
        
        # === 分支1: 频域特征提取 ===
        # DWT分解
        ll, lh, hl, hh = self.dwt.decompose(watermarked_image)
        
        # 合并频域子带
        freq_input = torch.cat([ll, lh, hl, hh], dim=1)  # (B, C*4, H/2, W/2)
        freq_feat = self.freq_encoder(freq_input)
        
        # 上采样到原始尺寸
        freq_feat = F.interpolate(freq_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        # === 分支2: 空间域特征提取 ===
        spatial_feat = self.spatial_encoder(watermarked_image)
        
        # === 特征融合 ===
        combined = torch.cat([freq_feat, spatial_feat], dim=1)
        fused = self.fusion(combined)
        
        # === 水印重建 ===
        extracted_watermark = self.decoder(fused)
        
        # === 置信度预测 ===
        confidence = self.confidence_head(fused)
        
        return extracted_watermark, confidence


class MultiScaleDecoder(nn.Module):
    """多尺度水印解码器"""
    
    def __init__(
        self,
        image_channels: int = 3,
        watermark_channels: int = 3,
        hidden_dim: int = 64,
        num_scales: int = 3
    ):
        super(MultiScaleDecoder, self).__init__()
        
        self.num_scales = num_scales
        
        # 创建多个尺度的解码器
        self.decoders = nn.ModuleList([
            WatermarkDecoder(image_channels, watermark_channels, hidden_dim)
            for _ in range(num_scales)
        ])
        
        # 加权融合层
        self.weight_fusion = nn.Sequential(
            nn.Conv2d(watermark_channels * num_scales, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, watermark_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, watermarked_image):
        """
        多尺度水印提取
        """
        b, c, h, w = watermarked_image.shape
        
        extracted_watermarks = []
        confidences = []
        
        for i, decoder in enumerate(self.decoders):
            # 对图像进行下采样
            scale = 2 ** i
            if scale > 1:
                scaled_image = F.interpolate(
                    watermarked_image, 
                    size=(h // scale, w // scale),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                scaled_image = watermarked_image
            
            # 提取水印
            wm, conf = decoder(scaled_image)
            
            # 上采样到原始尺寸
            if scale > 1:
                wm = F.interpolate(wm, size=(h, w), mode='bilinear', align_corners=False)
            
            extracted_watermarks.append(wm)
            confidences.append(conf)
        
        # 加权融合
        combined = torch.cat(extracted_watermarks, dim=1)
        fused_watermark = self.weight_fusion(combined)
        
        # 平均置信度
        avg_confidence = torch.stack(confidences, dim=1).mean(dim=1)
        
        return fused_watermark, avg_confidence


def test_decoder():
    """测试解码器"""
    decoder = WatermarkDecoder()
    
    # 测试数据
    watermarked = torch.randn(2, 3, 64, 64)
    
    # 前向传播
    extracted_wm, confidence = decoder(watermarked)
    
    print(f"输入图像形状: {watermarked.shape}")
    print(f"提取水印形状: {extracted_wm.shape}")
    print(f"置信度形状: {confidence.shape}")
    print(f"置信度范围: [{confidence.min():.3f}, {confidence.max():.3f}]")


def test_multiscale_decoder():
    """测试多尺度解码器"""
    decoder = MultiScaleDecoder(num_scales=3)
    
    # 测试数据
    watermarked = torch.randn(2, 3, 64, 64)
    
    # 前向传播
    extracted_wm, confidence = decoder(watermarked)
    
    print(f"\n多尺度解码器:")
    print(f"输入图像形状: {watermarked.shape}")
    print(f"提取水印形状: {extracted_wm.shape}")
    print(f"置信度形状: {confidence.shape}")


if __name__ == "__main__":
    test_decoder()
    test_multiscale_decoder()