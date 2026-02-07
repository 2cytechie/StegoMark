import torch
import torch.nn as nn
import torch.nn.functional as F
from ..dwt import DWT2D, IDWT2D


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class FrequencyFusionBlock(nn.Module):
    """频域融合块 - 将水印特征融合到DWT系数"""
    
    def __init__(self, channels):
        super(FrequencyFusionBlock, self).__init__()
        # 水印特征提取
        self.watermark_encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ResidualBlock(channels),
            ResidualBlock(channels),
        )
        
        # 融合网络
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, dwt_coeff, watermark):
        """
        dwt_coeff: DWT系数 (B, C, H, W)
        watermark: 水印图像 (B, 3, H, W)
        返回: 融合后的系数 (B, C, H, W)
        """
        # 提取水印特征
        wm_feat = self.watermark_encoder(watermark)
        
        # 拼接特征
        combined = torch.cat([dwt_coeff, wm_feat], dim=1)
        
        # 计算注意力权重
        attn = self.attention(combined)
        
        # 融合
        fused = self.fusion(combined)
        
        # 残差连接 + 注意力加权
        output = dwt_coeff + attn * fused
        
        return output


class WatermarkEncoder(nn.Module):
    """水印编码器 - 将水印嵌入到载体图像"""
    
    def __init__(
        self,
        image_channels: int = 3,
        watermark_channels: int = 3,
        hidden_dim: int = 64,
        wavelet: str = 'haar'
    ):
        super(WatermarkEncoder, self).__init__()
        
        self.image_channels = image_channels
        self.watermark_channels = watermark_channels
        self.hidden_dim = hidden_dim
        
        # DWT变换
        self.dwt = DWT2D(wavelet)
        self.idwt = IDWT2D(wavelet)
        
        # 预处理网络
        self.pre_conv = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 频域融合块（用于高频子带）
        self.fusion_lh = FrequencyFusionBlock(hidden_dim)
        self.fusion_hl = FrequencyFusionBlock(hidden_dim)
        self.fusion_hh = FrequencyFusionBlock(hidden_dim)
        
        # 后处理网络
        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, image_channels, 3, padding=1),
            nn.Tanh()
        )
        
        # 可学习的嵌入强度
        self.embedding_strength = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, image, watermark):
        """
        输入:
            image: 载体图像 (B, 3, H, W)
            watermark: 水印图像 (B, 3, H, W)
        输出:
            watermarked_image: 含水印图像 (B, 3, H, W)
        """
        b, c, h, w = image.shape
        
        # 预处理
        feat = self.pre_conv(image)
        
        # DWT分解
        ll, lh, hl, hh = self.dwt.decompose(feat)
        
        # Resize水印以匹配DWT子带尺寸
        wm_lh = F.interpolate(watermark, size=lh.shape[-2:], mode='bilinear', align_corners=False)
        wm_hl = F.interpolate(watermark, size=hl.shape[-2:], mode='bilinear', align_corners=False)
        wm_hh = F.interpolate(watermark, size=hh.shape[-2:], mode='bilinear', align_corners=False)
        
        # 在高频子带中嵌入水印
        lh_embedded = self.fusion_lh(lh, wm_lh)
        hl_embedded = self.fusion_hl(hl, wm_hl)
        hh_embedded = self.fusion_hh(hh, wm_hh)
        
        # 合并且控制嵌入强度
        strength = torch.sigmoid(self.embedding_strength)
        lh_merged = lh + strength * (lh_embedded - lh)
        hl_merged = hl + strength * (hl_embedded - hl)
        hh_merged = hh + strength * (hh_embedded - hh)
        
        # 合并所有子带
        dwt_combined = torch.cat([ll, lh_merged, hl_merged, hh_merged], dim=1)
        
        # IDWT重构
        feat_reconstructed = self.idwt(dwt_combined)
        
        # 裁剪到原始尺寸（IDWT可能产生稍大的输出）
        feat_reconstructed = feat_reconstructed[:, :, :h, :w]
        
        # 后处理
        residual = self.post_conv(feat_reconstructed)
        
        # 残差学习: 输出 = 输入 + 残差
        watermarked = torch.clamp(image + residual * 0.1, 0, 1)
        
        return watermarked


def test_encoder():
    """测试编码器"""
    encoder = WatermarkEncoder()
    
    # 测试数据
    image = torch.randn(2, 3, 64, 64)
    watermark = torch.randn(2, 3, 64, 64)
    
    # 前向传播
    output = encoder(image, watermark)
    
    print(f"输入图像形状: {image.shape}")
    print(f"水印形状: {watermark.shape}")
    print(f"输出图像形状: {output.shape}")
    print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 计算PSNR
    mse = torch.mean((image - output) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    print(f"PSNR: {psnr:.2f} dB")


if __name__ == "__main__":
    test_encoder()