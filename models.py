"""
深度学习网络模型
包含嵌入网络(EmbedNetwork)、提取网络(ExtractNetwork)和端到端网络(StegoNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from dwt_utils import DWTLayer, IDWTLayer, embed_to_subbands, dwt_transform, idwt_transform


class ConvBlock(nn.Module):
    """卷积块: Conv + BN + ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, use_bn: bool = True, 
                 dropout: float = 0.0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResBlock(nn.Module):
    """残差块"""
    
    def __init__(self, channels: int, use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class EncoderBlock(nn.Module):
    """编码器块: Conv + MaxPool"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        
        self.conv = ConvBlock(in_channels, out_channels, use_bn=use_bn, dropout=dropout)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        x = self.conv(x)
        skip = x  # 保存跳跃连接
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    """解码器块: UpSample + Conv"""
    
    def __init__(self, in_channels: int, out_channels: int,
                 use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 
                                         kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels, 
                             use_bn=use_bn, dropout=dropout)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # 处理尺寸不匹配
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class EmbedNetwork(nn.Module):
    """
    嵌入网络 - U-Net架构
    输入: 目标图像 + 水印图像
    输出: 嵌入强度图
    """
    
    def __init__(self, in_channels: int = 6,  # 3(图像) + 3(水印)
                 encoder_channels: List[int] = [64, 128, 256, 512],
                 decoder_channels: List[int] = [512, 256, 128, 64],
                 out_channels: int = 3,
                 use_bn: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        # 编码器
        self.encoders = nn.ModuleList()
        prev_ch = in_channels
        for ch in encoder_channels:
            self.encoders.append(EncoderBlock(prev_ch, ch, use_bn, dropout))
            prev_ch = ch
        
        # 瓶颈层
        self.bottleneck = ConvBlock(encoder_channels[-1], encoder_channels[-1] * 2, 
                                   use_bn=use_bn, dropout=dropout)
        
        # 解码器
        self.decoders = nn.ModuleList()
        prev_ch = encoder_channels[-1] * 2
        for i, ch in enumerate(decoder_channels):
            self.decoders.append(DecoderBlock(prev_ch, ch, use_bn, dropout))
            prev_ch = ch
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], out_channels, 3, 1, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
    
    def forward(self, image: torch.Tensor, watermark: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image: 目标图像 [B, 3, H, W]
            watermark: 水印图像 [B, 3, H, W]
        
        Returns:
            嵌入强度图 [B, 3, H, W]
        """
        # 拼接图像和水印
        x = torch.cat([image, watermark], dim=1)
        
        # 编码器路径
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器路径（带跳跃连接）
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        
        # 输出
        strength_map = self.output(x)
        
        return strength_map


class ExtractNetwork(nn.Module):
    """
    提取网络 - ResNet-like架构
    输入: 含水印图像的高频子带
    输出: 提取的水印图像 (64x64)
    """
    
    def __init__(self, in_channels: int = 3,
                 channels: List[int] = [64, 128, 256],
                 num_resblocks: int = 4,
                 watermark_size: int = 64,
                 use_bn: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.watermark_size = watermark_size
        
        # 初始卷积
        self.initial = ConvBlock(in_channels, channels[0], use_bn=use_bn, dropout=dropout)
        
        # 残差块组
        self.res_groups = nn.ModuleList()
        prev_ch = channels[0]
        
        for ch in channels:
            # 下采样（如果需要）
            layers = []
            if ch != prev_ch:
                layers.append(nn.Conv2d(prev_ch, ch, 3, 2, 1))
                layers.append(nn.BatchNorm2d(ch) if use_bn else nn.Identity())
                layers.append(nn.ReLU(inplace=True))
            
            # 残差块
            for _ in range(num_resblocks):
                layers.append(ResBlock(ch, use_bn, dropout))
            
            self.res_groups.append(nn.Sequential(*layers))
            prev_ch = ch
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 3 * watermark_size * watermark_size),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 高频子带 [B, 3, H, W]
        
        Returns:
            提取的水印 [B, 3, 64, 64]
        """
        # 初始卷积
        x = self.initial(x)
        
        # 残差块组
        for group in self.res_groups:
            x = group(x)
        
        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接
        x = self.fc(x)
        
        # reshape为图像
        x = x.view(-1, 3, self.watermark_size, self.watermark_size)
        
        return x


class StegoNet(nn.Module):
    """
    端到端水印网络
    包含嵌入网络和提取网络
    """
    
    def __init__(self, 
                 embed_config: Dict = None,
                 extract_config: Dict = None,
                 dwt_wavelet: str = 'haar'):
        super().__init__()
        
        # 默认配置
        if embed_config is None:
            embed_config = {
                'encoder_channels': [64, 128, 256, 512],
                'decoder_channels': [512, 256, 128, 64],
                'use_bn': True,
                'dropout': 0.1
            }
        
        if extract_config is None:
            extract_config = {
                'channels': [64, 128, 256],
                'num_resblocks': 4,
                'watermark_size': 64,
                'use_bn': True,
                'dropout': 0.1
            }
        
        # 创建网络
        self.embed_net = EmbedNetwork(
            encoder_channels=embed_config['encoder_channels'],
            decoder_channels=embed_config['decoder_channels'],
            use_bn=embed_config['use_bn'],
            dropout=embed_config['dropout']
        )
        
        self.extract_net = ExtractNetwork(
            channels=extract_config['channels'],
            num_resblocks=extract_config['num_resblocks'],
            watermark_size=extract_config['watermark_size'],
            use_bn=extract_config['use_bn'],
            dropout=extract_config['dropout']
        )
        
        # DWT/IDWT层
        self.dwt = DWTLayer(wavelet=dwt_wavelet)
        self.idwt = IDWTLayer(wavelet=dwt_wavelet)
    
    def embed(self, image: torch.Tensor, watermark: torch.Tensor, 
              alpha: float = 0.1) -> torch.Tensor:
        """
        嵌入水印
        
        Args:
            image: 目标图像 [B, 3, 256, 256]
            watermark: 水印图像 [B, 3, 256, 256] (已复制4x4)
            alpha: 嵌入强度系数
        
        Returns:
            嵌入水印后的图像 [B, 3, 256, 256]
        """
        # DWT变换
        subbands = self.dwt(image)
        
        # 嵌入网络预测强度图
        strength_map = self.embed_net(image, watermark)
        
        # 调整强度图尺寸以匹配子带
        b, c, h, w = subbands['LL'].shape
        if strength_map.shape[-2:] != (h, w):
            strength_map = F.interpolate(strength_map, size=(h, w), 
                                        mode='bilinear', align_corners=False)
        
        # 将强度图嵌入到高频子带
        subbands['LH'] = subbands['LH'] + alpha * strength_map
        subbands['HL'] = subbands['HL'] + alpha * strength_map
        subbands['HH'] = subbands['HH'] + alpha * strength_map
        
        # IDWT逆变换
        watermarked = self.idwt(subbands)
        
        # 裁剪到有效范围
        watermarked = torch.clamp(watermarked, 0, 1)
        
        return watermarked
    
    def extract(self, watermarked: torch.Tensor) -> torch.Tensor:
        """
        提取水印
        
        Args:
            watermarked: 含水印的图像 [B, 3, 256, 256]
        
        Returns:
            提取的水印 [B, 3, 64, 64]
        """
        # DWT变换获取高频子带
        subbands = self.dwt(watermarked)
        
        # 合并高频子带作为输入
        high_freq = torch.cat([
            subbands['LH'],
            subbands['HL'],
            subbands['HH']
        ], dim=1)
        
        # 使用1x1卷积降维到3通道
        if not hasattr(self, 'freq_fusion'):
            self.freq_fusion = nn.Conv2d(9, 3, 1).to(high_freq.device)
        
        high_freq = self.freq_fusion(high_freq)
        
        # 提取网络
        extracted = self.extract_net(high_freq)
        
        return extracted
    
    def forward(self, image: torch.Tensor, watermark: torch.Tensor,
                alpha: float = 0.1, attack_fn=None) -> Dict:
        """
        端到端前向传播
        
        Args:
            image: 目标图像 [B, 3, 256, 256]
            watermark: 水印图像 [B, 3, 256, 256]
            alpha: 嵌入强度
            attack_fn: 攻击函数（可选）
        
        Returns:
            包含所有中间结果的字典
        """
        # 嵌入水印
        watermarked = self.embed(image, watermark, alpha)
        
        # 应用攻击（如果提供）
        if attack_fn is not None:
            attacked = attack_fn(watermarked)
        else:
            attacked = watermarked
        
        # 提取水印
        extracted = self.extract(attacked)
        
        return {
            'watermarked': watermarked,
            'attacked': attacked,
            'extracted': extracted
        }


class Discriminator(nn.Module):
    """
    判别器网络（用于对抗训练，可选）
    区分原始图像和嵌入水印后的图像
    """
    
    def __init__(self, in_channels: int = 3,
                 channels: List[int] = [64, 128, 256, 512]):
        super().__init__()
        
        layers = []
        prev_ch = in_channels
        
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_ch, ch, 4, 2, 1),
                nn.BatchNorm2d(ch),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            prev_ch = ch
        
        # 全局池化和输出
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 1),
            nn.Sigmoid()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def create_model(config=None, device='cuda'):
    """
    创建模型实例
    
    Args:
        config: 配置对象
        device: 设备
    
    Returns:
        模型实例
    """
    if config is None:
        from config import model_config, dwt_config
        config = model_config
        wavelet = dwt_config.WAVELET
    else:
        wavelet = 'haar'
    
    embed_config = {
        'encoder_channels': config.EMBED_ENCODER_CHANNELS,
        'decoder_channels': config.EMBED_DECODER_CHANNELS,
        'use_bn': config.USE_BATCHNORM,
        'dropout': config.DROPOUT
    }
    
    extract_config = {
        'channels': config.EXTRACT_CHANNELS,
        'num_resblocks': config.NUM_RESBLOCKS,
        'watermark_size': 64,
        'use_bn': config.USE_BATCHNORM,
        'dropout': config.DROPOUT
    }
    
    model = StegoNet(embed_config, extract_config, wavelet)
    model = model.to(device)
    
    return model


if __name__ == '__main__':
    # 测试模型
    print("测试深度学习网络模型...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 2
    image = torch.randn(batch_size, 3, 256, 256).to(device)
    watermark = torch.randn(batch_size, 3, 256, 256).to(device)
    
    # 创建模型
    model = create_model(device=device)
    
    # 测试嵌入
    print("\n测试嵌入网络...")
    watermarked = model.embed(image, watermark, alpha=0.1)
    print(f"嵌入后图像尺寸: {watermarked.shape}")
    
    # 测试提取
    print("\n测试提取网络...")
    extracted = model.extract(watermarked)
    print(f"提取的水印尺寸: {extracted.shape}")
    
    # 测试端到端
    print("\n测试端到端网络...")
    results = model.forward(image, watermark, alpha=0.1)
    print(f"输出键: {results.keys()}")
    for key, value in results.items():
        print(f"  {key}: {value.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型总参数量: {total_params:,}")
    
    print("\n模型测试完成！")
