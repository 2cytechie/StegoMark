"""
编码器模块 - 水印嵌入
包含DWT分解、水印嵌入网络、逆变换
将水印嵌入到LH、HL、HH三个频段
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.dwt import DWT2D, pad_to_multiple, remove_padding


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels):
        super().__init__()
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


class WatermarkEmbedder(nn.Module):
    """水印嵌入网络"""
    
    def __init__(self, in_channels=3, watermark_channels=3, base_channels=64, num_blocks=3):
        super().__init__()
        
        # 输入层：拼接宿主图像频段和水印
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels + watermark_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(base_channels) for _ in range(num_blocks)
        ])
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, host_band, watermark):
        """
        将水印嵌入到宿主频段
        
        Args:
            host_band: 宿主频段 [B, C, H, W]
            watermark: 水印 [B, C, H, W]
            
        Returns:
            modified_band: 修改后的频段 [B, C, H, W]
        """
        # 拼接
        x = torch.cat([host_band, watermark], dim=1)
        
        # 特征提取
        x = self.input_conv(x)
        x = self.residual_blocks(x)
        
        # 生成修改量
        delta = self.output_conv(x)
        
        # 残差连接：宿主频段 + 修改量
        modified_band = host_band + delta
        
        return modified_band


class Encoder(nn.Module):
    """编码器 - 完整的水印嵌入流程
    将水印同时嵌入到LH、HL、HH三个频段
    """
    
    def __init__(self, wavelet='haar', mode='symmetric',
                 base_channels=64, num_blocks=3):
        """
        Args:
            wavelet: 小波基
            mode: 边界处理模式
            base_channels: 基础通道数
            num_blocks: 残差块数量
        """
        super().__init__()
        
        self.wavelet = wavelet
        self.mode = mode
        
        # DWT变换
        self.dwt = DWT2D(wavelet, mode)
        
        # 三个水印嵌入网络（分别用于LH、HL、HH频段）
        self.embedder_lh = WatermarkEmbedder(
            in_channels=3,
            watermark_channels=3,
            base_channels=base_channels,
            num_blocks=num_blocks
        )
        self.embedder_hl = WatermarkEmbedder(
            in_channels=3,
            watermark_channels=3,
            base_channels=base_channels,
            num_blocks=num_blocks
        )
        self.embedder_hh = WatermarkEmbedder(
            in_channels=3,
            watermark_channels=3,
            base_channels=base_channels,
            num_blocks=num_blocks
        )
    
    def forward(self, image, watermark):
        """
        嵌入水印到三个高频频段
        
        Args:
            image: 目标图片 [B, 3, H, W]
            watermark: 平铺后的水印 [B, 3, H, W]
            
        Returns:
            watermarked_image: 含水印图片 [B, 3, H, W]
            dwt_bands: DWT分解的频段 (用于解码器)
        """
        # 记录原始尺寸
        original_h, original_w = image.shape[2], image.shape[3]
        
        # 填充到2的倍数
        padded_image, pad_info = pad_to_multiple(image, multiple=2)
        padded_watermark, _ = pad_to_multiple(watermark, multiple=2)
        
        # DWT分解
        LL, LH, HL, HH = self.dwt.dwt2(padded_image)
        
        # 调整水印尺寸以匹配频段
        wm_size = LH.shape[2:]
        resized_watermark = F.interpolate(
            padded_watermark, size=wm_size, mode='bilinear', align_corners=False
        )
        
        # 将水印嵌入到LH、HL、HH三个频段
        LH = self.embedder_lh(LH, resized_watermark)
        HL = self.embedder_hl(HL, resized_watermark)
        HH = self.embedder_hh(HH, resized_watermark)
        
        # 逆DWT重构
        watermarked_padded = self.dwt.idwt2(LL, LH, HL, HH)
        
        # 移除填充
        watermarked_image = remove_padding(watermarked_padded, pad_info)
        
        # 裁剪到原始尺寸
        watermarked_image = watermarked_image[:, :, :original_h, :original_w]
        
        # 限制在[-1, 1]范围内
        watermarked_image = torch.clamp(watermarked_image, -1, 1)
        
        # 保存DWT频段供解码器使用
        dwt_bands = {
            'LL': LL,
            'LH': LH,
            'HL': HL,
            'HH': HH,
            'pad_info': pad_info
        }
        
        return watermarked_image, dwt_bands


class SimpleEncoder(nn.Module):
    """简化版编码器 (直接嵌入，不使用DWT)"""
    
    def __init__(self, base_channels=64, num_blocks=4):
        super().__init__()
        
        # 输入层
        self.input_conv = nn.Sequential(
            nn.Conv2d(6, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 下采样
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(base_channels*4) for _ in range(num_blocks)
        ])
        
        # 上采样
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, image, watermark):
        """
        嵌入水印
        
        Args:
            image: 目标图片 [B, 3, H, W]
            watermark: 平铺后的水印 [B, 3, H, W]
            
        Returns:
            watermarked_image: 含水印图片 [B, 3, H, W]
        """
        # 拼接
        x = torch.cat([image, watermark], dim=1)
        
        # 编码
        x = self.input_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residual_blocks(x)
        x = self.up2(x)
        x = self.up1(x)
        
        # 生成修改量
        delta = self.output_conv(x)
        
        # 调整尺寸
        if delta.shape[2:] != image.shape[2:]:
            delta = F.interpolate(delta, size=image.shape[2:], mode='bilinear', align_corners=False)
        
        # 残差连接
        watermarked_image = torch.clamp(image + delta, -1, 1)
        
        return watermarked_image, None
