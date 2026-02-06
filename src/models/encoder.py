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


class GroupedWatermarkEmbedder(nn.Module):
    """分组卷积水印嵌入网络 - 同时处理3个频段，共享大部分权重
    
    设计思路：
    1. 使用1x1卷积将3个频段投影到共享特征空间
    2. 在共享特征空间处理（残差块）
    3. 使用1x1卷积投影回3个频段
    
    参数量对比（base_channels=64）：
    - 原始3个独立网络: 3 * (6*64*3*3 + 64*64*3*3*3 + 64*3*3*3) ≈ 3.5M
    - 分组卷积版本: 9*64*1*1 + 64*64*3*3*3 + 64*9*1*1 ≈ 1.2M
    """
    
    def __init__(self, in_channels=3, watermark_channels=3, base_channels=64, num_blocks=3):
        super().__init__()
        
        self.in_channels = in_channels
        self.watermark_channels = watermark_channels
        
        # 输入投影层：将3个频段+水印投影到共享特征空间
        # 输入: [B, 12, H, W] -> 输出: [B, 64, H, W]
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3 + watermark_channels, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 共享的残差块（处理共享特征空间）
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(base_channels) for _ in range(num_blocks)
        ])
        
        # 输出投影层：将共享特征投影回3个频段的修改量
        # 输入: [B, 64, H, W] -> 输出: [B, 9, H, W]
        self.output_proj = nn.Sequential(
            nn.Conv2d(base_channels, in_channels * 3, 1),
            nn.Tanh()
        )
    
    def forward(self, bands, watermark):
        """
        将水印嵌入到三个频段
        
        Args:
            bands: 三个频段拼接 [B, 9, H, W] (LH, HL, HH)
            watermark: 水印 [B, 3, H, W]
            
        Returns:
            modified_lh: 修改后的LH频段 [B, 3, H, W]
            modified_hl: 修改后的HL频段 [B, 3, H, W]
            modified_hh: 修改后的HH频段 [B, 3, H, W]
        """
        # 拼接所有输入 [B, 12, H, W]
        x = torch.cat([bands, watermark], dim=1)
        
        # 投影到共享特征空间
        x = self.input_proj(x)
        
        # 共享特征处理
        x = self.residual_blocks(x)
        
        # 投影回3个频段的修改量 [B, 9, H, W]
        delta = self.output_proj(x)
        
        # 分割修改量
        delta_lh, delta_hl, delta_hh = torch.chunk(delta, 3, dim=1)
        
        # 分割原始频段
        bands_split = torch.chunk(bands, 3, dim=1)
        
        # 残差连接
        modified_lh = bands_split[0] + delta_lh
        modified_hl = bands_split[1] + delta_hl
        modified_hh = bands_split[2] + delta_hh
        
        return modified_lh, modified_hl, modified_hh


class Encoder(nn.Module):
    """编码器 - 完整的水印嵌入流程
    将水印同时嵌入到LH、HL、HH三个频段
    """
    
    def __init__(self, wavelet='haar', mode='symmetric',
                 base_channels=64, num_blocks=3, use_grouped_conv=False):
        """
        Args:
            wavelet: 小波基
            mode: 边界处理模式
            base_channels: 基础通道数
            num_blocks: 残差块数量
            use_grouped_conv: 是否使用分组卷积共享权重
        """
        super().__init__()
        
        self.wavelet = wavelet
        self.mode = mode
        self.use_grouped_conv = use_grouped_conv
        
        # DWT变换
        self.dwt = DWT2D(wavelet, mode)
        
        if use_grouped_conv:
            # 使用分组卷积共享权重
            self.grouped_embedder = GroupedWatermarkEmbedder(
                in_channels=3,
                watermark_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks
            )
        else:
            # 三个独立的水印嵌入网络（分别用于LH、HL、HH频段）
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
        
        if self.use_grouped_conv:
            # 使用分组卷积同时处理三个频段
            bands = torch.cat([LH, HL, HH], dim=1)  # [B, 9, H, W]
            LH, HL, HH = self.grouped_embedder(bands, resized_watermark)
        else:
            # 分别嵌入到LH、HL、HH三个频段
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
