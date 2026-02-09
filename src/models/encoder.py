import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import config
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
    """水印编码器 - 将水印嵌入到载体图像（分块处理版本）"""
    
    def __init__(
        self,
        image_channels: int = 3,
        watermark_channels: int = 3,
        hidden_dim: int = 64,
        wavelet: str = 'haar',
        block_size: int = config.watermark_size,
        overlap: int = config.overlap  # 重叠区域大小
    ):
        super(WatermarkEncoder, self).__init__()
        
        self.image_channels = image_channels
        self.watermark_channels = watermark_channels
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.overlap = overlap
        
        # DWT变换
        self.dwt = DWT2D(wavelet)
        self.idwt = IDWT2D(wavelet)
        
        # 预处理网络
        self.pre_conv = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 频域融合块
        self.fusion_ll = FrequencyFusionBlock(hidden_dim)  # LL低频子带
        self.fusion_lh = FrequencyFusionBlock(hidden_dim)  # LH水平高频子带
        self.fusion_hl = FrequencyFusionBlock(hidden_dim)  # HL垂直高频子带
        self.fusion_hh = FrequencyFusionBlock(hidden_dim)  # HH对角高频子带
        
        # 后处理网络
        self.post_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, image_channels, 3, padding=1),
        )
        
        # 可学习的嵌入强度
        self.embedding_strength = nn.Parameter(torch.tensor(1.0))
        # 水印增强网络 - 提高水印的鲁棒性
        self.watermark_enhance = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )
    
    def extract_blocks(self, image):
        """
        将图像分块提取，带重叠区域
        
        输入: image (B, C, H, W)
        输出: blocks 列表, block_positions 列表, 原始尺寸
        """
        b, c, h, w = image.shape
        block_size = self.block_size
        overlap = self.overlap
        stride = block_size - overlap  # 步长 = 块大小 - 重叠
        
        blocks = []
        block_positions = []
        
        # 计算需要多少块才能覆盖整个图像
        n_h = (h - overlap + stride - 1) // stride
        n_w = (w - overlap + stride - 1) // stride
        
        for i in range(n_h):
            for j in range(n_w):
                # 计算块的起始位置
                start_h = i * stride
                start_w = j * stride
                
                # 确保不超出图像边界
                end_h = min(start_h + block_size, h)
                end_w = min(start_w + block_size, w)
                
                # 如果块大小不足，进行填充
                if end_h - start_h < block_size or end_w - start_w < block_size:
                    # 提取当前区域
                    block = image[:, :, start_h:end_h, start_w:end_w]
                    # 填充到标准块大小
                    block = F.pad(block, (0, block_size - (end_w - start_w), 
                                         0, block_size - (end_h - start_h)))
                else:
                    block = image[:, :, start_h:end_h, start_w:end_w]
                
                blocks.append(block)
                block_positions.append((start_h, start_w, end_h, end_w))
        
        return blocks, block_positions, (h, w)
    
    def embed_watermark_in_block(self, block, watermark):
        """
        在单个块中嵌入水印
        
        输入: 
            block: 图像块 (B, C, block_size, block_size)
            watermark: 水印图像 (B, 3, H, W)
        输出: 含水印的块
        """
        b, c, h, w = block.shape
        
        # 增强水印 - 提高鲁棒性
        enhanced_watermark = watermark + 0.1 * self.watermark_enhance(watermark)
        enhanced_watermark = torch.clamp(enhanced_watermark, 0, 1)
        
        # 预处理
        feat = self.pre_conv(block)
        
        # DWT分解
        ll, lh, hl, hh = self.dwt.decompose(feat)
        
        # Resize水印以匹配DWT子带尺寸（所有子带）
        wm_ll = F.interpolate(enhanced_watermark, size=ll.shape[-2:], mode='bilinear', align_corners=False)
        wm_lh = F.interpolate(enhanced_watermark, size=lh.shape[-2:], mode='bilinear', align_corners=False)
        wm_hl = F.interpolate(enhanced_watermark, size=hl.shape[-2:], mode='bilinear', align_corners=False)
        wm_hh = F.interpolate(enhanced_watermark, size=hh.shape[-2:], mode='bilinear', align_corners=False)
        
        # 在所有子带中嵌入水印（包含LL低频子带）
        ll_embedded = self.fusion_ll(ll, wm_ll)
        lh_embedded = self.fusion_lh(lh, wm_lh)
        hl_embedded = self.fusion_hl(hl, wm_hl)
        hh_embedded = self.fusion_hh(hh, wm_hh)
        
        # 合并且控制嵌入强度
        strength = torch.sigmoid(self.embedding_strength)
        
        # 在所有子带中嵌入水印（LL子带使用较低的嵌入强度）
        ll_strength = strength * 0.1  # LL子带使用较低的嵌入强度以保护图像质量
        ll_merged = ll + ll_strength * (ll_embedded - ll)
        lh_merged = lh + strength * (lh_embedded - lh)
        hl_merged = hl + strength * (hl_embedded - hl)
        hh_merged = hh + strength * (hh_embedded - hh)
        
        # 合并所有子带
        dwt_combined = torch.cat([ll_merged, lh_merged, hl_merged, hh_merged], dim=1)
        
        # IDWT重构
        feat_reconstructed = self.idwt(dwt_combined)
        
        # 裁剪到原始尺寸
        feat_reconstructed = feat_reconstructed[:, :, :h, :w]
        
        # 后处理
        residual = self.post_conv(feat_reconstructed)
        
        # 残差学习 - 自适应残差强度
        watermarked = torch.clamp(block + torch.tanh(residual) * 0.1, 0, 1)
        
        return watermarked
    
    def merge_blocks(self, blocks, block_positions, original_size):
        """
        使用平滑融合技术将块拼接回原图
        
        输入:
            blocks: 含水印的块列表
            block_positions: 每个块的位置信息
            original_size: 原始图像尺寸 (H, W)
        输出: 拼接后的完整图像
        """
        h, w = original_size
        b = blocks[0].shape[0]
        c = blocks[0].shape[1]
        
        # 创建输出图像和权重图
        output = torch.zeros((b, c, h, w), device=blocks[0].device)
        weight_map = torch.zeros((b, 1, h, w), device=blocks[0].device)
        
        overlap = self.overlap
        
        for block, (start_h, start_w, end_h, end_w) in zip(blocks, block_positions):
            # 提取有效区域（去除填充）
            valid_h = end_h - start_h
            valid_w = end_w - start_w
            valid_block = block[:, :, :valid_h, :valid_w]
            
            # 创建权重掩码（平滑融合）
            mask = torch.ones((b, 1, valid_h, valid_w), device=block.device)
            
            # 在重叠区域应用渐变权重
            if overlap > 0:
                # 水平方向渐变
                if valid_w > overlap * 2:
                    # 左边缘渐变
                    mask[:, :, :, :overlap] *= torch.linspace(0, 1, overlap, device=block.device).view(1, 1, 1, -1)
                    # 右边缘渐变
                    mask[:, :, :, -overlap:] *= torch.linspace(1, 0, overlap, device=block.device).view(1, 1, 1, -1)
                
                # 垂直方向渐变
                if valid_h > overlap * 2:
                    # 上边缘渐变
                    mask[:, :, :overlap, :] *= torch.linspace(0, 1, overlap, device=block.device).view(1, 1, -1, 1)
                    # 下边缘渐变
                    mask[:, :, -overlap:, :] *= torch.linspace(1, 0, overlap, device=block.device).view(1, 1, -1, 1)
            
            # 将块添加到输出（加权）
            output[:, :, start_h:end_h, start_w:end_w] += valid_block * mask
            weight_map[:, :, start_h:end_h, start_w:end_w] += mask
        
        # 归一化（处理重叠区域的多次累加）
        output = output / (weight_map + 1e-8)
        
        return torch.clamp(output, 0, 1)
    
    def forward(self, image, watermark):
        """
        前向传播 - 分块处理版本
        
        输入:
            image: 载体图像 (B, 3, H, W)
            watermark: 水印图像 (B, 3, H, W)
        输出:
            watermarked_image: 含水印图像 (B, 3, H, W)
        """
        # 步骤1: 将目标图片分块处理
        blocks, block_positions, original_size = self.extract_blocks(image)
        
        # 步骤2-5: 对每个块进行DWT分解和水印嵌入
        watermarked_blocks = []
        for block in blocks:
            # 调整水印大小以匹配当前块
            block_watermark = F.interpolate(watermark, size=(block.shape[2], block.shape[3]), 
                                           mode='bilinear', align_corners=False)
            
            # 在块中嵌入水印
            watermarked_block = self.embed_watermark_in_block(block, block_watermark)
            watermarked_blocks.append(watermarked_block)
        
        # 步骤6: 逆变换生成含水印图块，通过平滑融合技术拼回原图
        watermarked_image = self.merge_blocks(watermarked_blocks, block_positions, original_size)
        
        return watermarked_image
