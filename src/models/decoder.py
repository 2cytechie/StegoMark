"""
解码器模块 - 水印提取
包含空间变换网络(STN)和提取网络
从LH、HL、HH三个频段提取水印并融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.dwt import DWT2D, pad_to_multiple, remove_padding


class SpatialTransformerNetwork(nn.Module):
    """空间变换网络 (STN) - 用于几何校正"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # 定位网络
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        
        # 自适应池化到固定尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 回归网络 - 预测变换参数
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 6)  # 仿射变换有6个参数
        )
        
        # 初始化为单位变换
        self.fc_loc[3].weight.data.zero_()
        self.fc_loc[3].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    
    def forward(self, x):
        """
        应用空间变换
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            transformed: 变换后的图像 [B, C, H, W]
            theta: 变换参数 [B, 2, 3]
        """
        B, C, H, W = x.shape
        
        # 定位
        xs = self.localization(x)
        xs = self.adaptive_pool(xs)
        xs = xs.view(B, -1)
        
        # 预测变换参数
        theta = self.fc_loc(xs)
        theta = theta.view(B, 2, 3)
        
        # 生成采样网格
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        
        # 采样
        transformed = F.grid_sample(x, grid, align_corners=False)
        
        return transformed, theta


class WatermarkExtractor(nn.Module):
    """水印提取网络"""
    
    def __init__(self, in_channels=3, watermark_channels=3, base_channels=64, num_blocks=4):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(base_channels*4) for _ in range(num_blocks)
        ])
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, watermark_channels, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        提取水印
        
        Args:
            x: 含水印图片 [B, C, H, W]
            
        Returns:
            watermark: 提取的水印 [B, C, H, W]
        """
        # 编码
        x = self.encoder(x)
        
        # 残差处理
        x = self.residual_blocks(x)
        
        # 解码
        x = self.decoder(x)
        
        return x


class GroupedWatermarkExtractor(nn.Module):
    """分组卷积水印提取网络 - 同时处理3个频段，共享大部分权重
    
    设计思路：
    1. 使用1x1卷积将3个频段投影到共享特征空间
    2. 在共享特征空间处理（编码-残差-解码）
    3. 使用1x1卷积投影回3个频段的水印
    
    参数量对比（base_channels=64）：
    - 原始3个独立网络: 3 * (3*64*3*3 + 64*128*3*3 + 128*256*3*3 + 256*256*3*3*4 + ...) ≈ 15M
    - 分组卷积版本: 9*64*1*1 + (3*64*3*3 + 64*128*3*3 + 128*256*3*3) + 256*256*3*3*4 + 256*9*1*1 ≈ 5M
    """
    
    def __init__(self, in_channels=3, watermark_channels=3, base_channels=64, num_blocks=4):
        super().__init__()
        
        self.in_channels = in_channels
        self.watermark_channels = watermark_channels
        
        # 输入投影层：将3个频段投影到共享特征空间
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels * 3, base_channels, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 共享编码器（处理共享特征空间）
        self.encoder = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )
        
        # 共享残差块
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(base_channels*4) for _ in range(num_blocks)
        ])
        
        # 共享解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # 输出投影层：将共享特征投影回3个频段的水印
        self.output_proj = nn.Sequential(
            nn.Conv2d(base_channels, watermark_channels * 3, 1),
            nn.Tanh()
        )
    
    def forward(self, bands):
        """
        从三个频段提取水印
        
        Args:
            bands: 三个频段拼接 [B, 9, H, W] (LH, HL, HH)
            
        Returns:
            watermark_lh: 从LH提取的水印 [B, 3, H, W]
            watermark_hl: 从HL提取的水印 [B, 3, H, W]
            watermark_hh: 从HH提取的水印 [B, 3, H, W]
        """
        # 投影到共享特征空间
        x = self.input_proj(bands)
        
        # 编码
        x = self.encoder(x)
        
        # 残差处理
        x = self.residual_blocks(x)
        
        # 解码
        x = self.decoder(x)
        
        # 投影回3个频段的水印
        x = self.output_proj(x)
        
        # 分割成三个频段的水印
        watermark_lh, watermark_hl, watermark_hh = torch.chunk(x, 3, dim=1)
        
        return watermark_lh, watermark_hl, watermark_hh


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


class Decoder(nn.Module):
    """解码器 - 完整的水印提取流程
    从LH、HL、HH三个频段提取水印并融合
    """
    
    def __init__(self, wavelet='haar', mode='symmetric',
                 use_stn=True, base_channels=64, num_blocks=4, watermark_size=64,
                 use_grouped_conv=False):
        """
        Args:
            wavelet: 小波基
            mode: 边界处理模式
            use_stn: 是否使用空间变换网络
            base_channels: 基础通道数
            num_blocks: 残差块数量
            watermark_size: 水印尺寸
            use_grouped_conv: 是否使用分组卷积共享权重
        """
        super().__init__()
        
        self.wavelet = wavelet
        self.mode = mode
        self.use_stn = use_stn
        self.watermark_size = watermark_size
        self.use_grouped_conv = use_grouped_conv
        
        # DWT变换
        self.dwt = DWT2D(wavelet, mode)
        
        # 空间变换网络
        if use_stn:
            self.stn = SpatialTransformerNetwork(in_channels=3)
        
        if use_grouped_conv:
            # 使用分组卷积共享权重
            self.grouped_extractor = GroupedWatermarkExtractor(
                in_channels=3,
                watermark_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks
            )
        else:
            # 三个独立的水印提取网络（分别用于LH、HL、HH频段）
            self.extractor_lh = WatermarkExtractor(
                in_channels=3,
                watermark_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks
            )
            self.extractor_hl = WatermarkExtractor(
                in_channels=3,
                watermark_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks
            )
            self.extractor_hh = WatermarkExtractor(
                in_channels=3,
                watermark_channels=3,
                base_channels=base_channels,
                num_blocks=num_blocks
            )
        
        # 水印融合网络（自适应加权融合三个频段提取的水印）
        self.fusion = nn.Sequential(
            nn.Conv2d(9, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, watermarked_image, dwt_bands=None):
        """
        提取水印
        
        Args:
            watermarked_image: 含水印图片 [B, 3, H, W]
            dwt_bands: DWT频段 (可选，用于从特定频段提取)
            
        Returns:
            extracted_watermark: 提取的水印 [B, 3, watermark_size, watermark_size]
            stn_theta: STN变换参数 (如果使用了STN)
        """
        stn_theta = None
        
        # 应用STN进行几何校正
        if self.use_stn:
            watermarked_image, stn_theta = self.stn(watermarked_image)
        
        # 如果没有DWT频段，先进行DWT分解
        if dwt_bands is None:
            # 填充到2的倍数
            padded_image, pad_info = pad_to_multiple(watermarked_image, multiple=2)
            LL, LH, HL, HH = self.dwt.dwt2(padded_image)
        else:
            LH = dwt_bands['LH']
            HL = dwt_bands['HL']
            HH = dwt_bands['HH']
        
        if self.use_grouped_conv:
            # 使用分组卷积同时处理三个频段
            bands = torch.cat([LH, HL, HH], dim=1)  # [B, 9, H, W]
            watermark_lh, watermark_hl, watermark_hh = self.grouped_extractor(bands)
        else:
            # 从三个频段分别提取水印
            watermark_lh = self.extractor_lh(LH)
            watermark_hl = self.extractor_hl(HL)
            watermark_hh = self.extractor_hh(HH)
        
        # 调整尺寸到标准水印尺寸
        watermark_lh = F.interpolate(
            watermark_lh, 
            size=(self.watermark_size, self.watermark_size),
            mode='bilinear',
            align_corners=False
        )
        watermark_hl = F.interpolate(
            watermark_hl, 
            size=(self.watermark_size, self.watermark_size),
            mode='bilinear',
            align_corners=False
        )
        watermark_hh = F.interpolate(
            watermark_hh, 
            size=(self.watermark_size, self.watermark_size),
            mode='bilinear',
            align_corners=False
        )
        
        # 融合三个频段提取的水印
        fused = torch.cat([watermark_lh, watermark_hl, watermark_hh], dim=1)
        extracted_watermark = self.fusion(fused)
        
        return extracted_watermark, stn_theta


class SimpleDecoder(nn.Module):
    """简化版解码器"""
    
    def __init__(self, base_channels=64, num_blocks=4, watermark_size=64):
        super().__init__()
        
        self.watermark_size = watermark_size
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(inplace=True),
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(base_channels*8) for _ in range(num_blocks)
        ])
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels*2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, watermarked_image):
        """
        提取水印
        
        Args:
            watermarked_image: 含水印图片 [B, 3, H, W]
            
        Returns:
            extracted_watermark: 提取的水印 [B, 3, watermark_size, watermark_size]
        """
        # 编码
        x = self.encoder(watermarked_image)
        
        # 残差处理
        x = self.residual_blocks(x)
        
        # 解码
        x = self.decoder(x)
        
        # 调整尺寸
        extracted_watermark = F.interpolate(
            x,
            size=(self.watermark_size, self.watermark_size),
            mode='bilinear',
            align_corners=False
        )
        
        return extracted_watermark, None
