import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import config
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
    """水印解码器 - 从含水印图像中提取水印（分块处理版本）"""
    
    def __init__(
        self,
        image_channels: int = 3,
        watermark_channels: int = 3,
        hidden_dim: int = 64,
        wavelet: str = 'haar',
        block_size: int = config.watermark_size,
        overlap: int = config.overlap  # 重叠区域大小
    ):
        super(WatermarkDecoder, self).__init__()
        
        self.image_channels = image_channels
        self.watermark_channels = watermark_channels
        self.hidden_dim = hidden_dim
        self.block_size = block_size
        self.overlap = overlap
        
        # DWT变换
        self.dwt = DWT2D(wavelet)
        self.idwt = IDWT2D(wavelet)
        
        # 频域特征提取网络
        self.freq_encoder = nn.Sequential(
            ConvBlock(image_channels * 4, hidden_dim),
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
        
        # 水印重建网络 - 输出64x64的水印
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
    
    def extract_blocks(self, image, block_size=None, overlap=None):
        """
        将图像分块提取，带重叠区域（与编码器保持一致）
        
        输入: image (B, C, H, W)
        输出: blocks 列表, block_positions 列表
        """
        if block_size is None:
            block_size = self.block_size
        if overlap is None:
            overlap = self.overlap
            
        b, c, h, w = image.shape
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
    
    def merge_blocks(self, blocks, block_positions, original_size):
        """
        使用平滑融合技术将块拼接回原图
        
        输入:
            blocks: 提取的水印块列表
            block_positions: 每个块的位置信息
            original_size: 原始图像尺寸 (H, W)
        输出: 拼接后的完整水印
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
    
    def extract_watermark_from_block(self, block):
        """
        从单个块中提取水印
        
        输入: block (B, C, 64, 64)
        输出: watermark (B, 3, 64, 64), confidence (B, 1)
        """
        b, c, h, w = block.shape
        
        # === 分支1: 频域特征提取 ===
        ll, lh, hl, hh = self.dwt.decompose(block)
        
        # 合并频域子带
        freq_input = torch.cat([ll, lh, hl, hh], dim=1)
        freq_feat = self.freq_encoder(freq_input)
        
        # 上采样到原始尺寸
        freq_feat = F.interpolate(freq_feat, size=(h, w), mode='bilinear', align_corners=False)
        
        # === 分支2: 空间域特征提取 ===
        spatial_feat = self.spatial_encoder(block)
        
        # === 特征融合 ===
        combined = torch.cat([freq_feat, spatial_feat], dim=1)
        fused = self.fusion(combined)
        
        # === 水印重建 ===
        extracted_watermark = self.decoder(fused)
        
        # === 置信度预测 ===
        confidence = self.confidence_head(fused)
        
        return extracted_watermark, confidence
    
    def forward(self, watermarked_image, return_all_blocks=False):
        """
        前向传播 - 分块处理版本
        
        输入:
            watermarked_image: 含水印图像 (B, 3, H, W)
            return_all_blocks: 是否返回所有块的水印（用于训练）
        输出:
            extracted_watermark: 提取的水印 (B, 3, 64, 64) 或所有块的水印列表
            confidence: 置信度分数 (B, 1) 或所有块的置信度列表
        """
        # 步骤1: 将含水印图片分块（64x64）处理 - 使用与编码器相同的重叠分块策略
        blocks, block_positions, original_size = self.extract_blocks(watermarked_image, self.block_size, self.overlap)
        
        # 步骤2: 从每个块中提取水印
        extracted_watermarks = []
        confidences = []
        
        for block in blocks:
            wm, conf = self.extract_watermark_from_block(block)
            extracted_watermarks.append(wm)
            confidences.append(conf)
        
        if return_all_blocks:
            # 返回所有块的水印（用于训练时计算损失）
            # 堆叠成 (B, num_blocks, 3, 64, 64)
            all_watermarks = torch.stack(extracted_watermarks, dim=1)
            all_confidences = torch.stack(confidences, dim=1)
            return all_watermarks, all_confidences, block_positions
        else:
            # 推理时：选择置信度最高的水印
            confidences_tensor = torch.stack(confidences, dim=1)  # (B, num_blocks, 1)
            best_idx = torch.argmax(confidences_tensor, dim=1)  # (B, 1)
            
            # 提取最佳水印
            batch_size = watermarked_image.size(0)
            best_watermarks = []
            best_confidences = []
            
            for b in range(batch_size):
                idx = best_idx[b, 0]
                best_watermarks.append(extracted_watermarks[idx][b])
                best_confidences.append(confidences[idx][b])
            
            extracted_watermark = torch.stack(best_watermarks, dim=0)
            confidence = torch.stack(best_confidences, dim=0)
            
            return extracted_watermark, confidence


class MultiScaleDecoder(nn.Module):
    """多尺度水印解码器"""
    
    def __init__(
        self,
        image_channels: int = 3,
        watermark_channels: int = 3,
        hidden_dim: int = 64,
        num_scales: int = 3,
        block_size: int = 64,
        overlap: int = 8
    ):
        super(MultiScaleDecoder, self).__init__()
        
        self.num_scales = num_scales
        
        # 创建多个尺度的解码器
        self.decoders = nn.ModuleList([
            WatermarkDecoder(image_channels, watermark_channels, hidden_dim, block_size=block_size, overlap=overlap)
            for _ in range(num_scales)
        ])
        
        # 加权融合层
        self.weight_fusion = nn.Sequential(
            nn.Conv2d(watermark_channels * num_scales, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, watermark_channels, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, watermarked_image, return_all_blocks=False):
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
            if return_all_blocks:
                wm, conf, _ = decoder(scaled_image, return_all_blocks=True)
            else:
                wm, conf = decoder(scaled_image, return_all_blocks=False)
            
            extracted_watermarks.append(wm)
            confidences.append(conf)
        
        if return_all_blocks:
            # 返回所有尺度和所有块的水印
            return extracted_watermarks, confidences
        else:
            # 加权融合
            combined = torch.cat(extracted_watermarks, dim=1)
            fused_watermark = self.weight_fusion(combined)
            
            # 平均置信度
            avg_confidence = torch.stack(confidences, dim=1).mean(dim=1)
            
            return fused_watermark, avg_confidence


def test_decoder():
    """测试解码器（分块处理版本）"""
    from PIL import Image
    import torchvision.transforms.functional as TF
    import os
    
    decoder = WatermarkDecoder(block_size=config.watermark_size, overlap=config.overlap)
    decoder.eval()
    
    # 测试数据
    test_image_path = 'outputs/test_encoder.png'
    
    image = Image.open(test_image_path).convert("RGB")
    image = TF.to_tensor(image).unsqueeze(0)
    
    print(f"输入图像形状: {image.shape}")
    
    # 测试分块
    blocks, positions, orig_size = decoder.extract_blocks(image, block_size=config.watermark_size, overlap=config.overlap)
    print(f"图像被分成 {len(blocks)} 个块")
    print(f"块大小: {blocks[0].shape}")
    print(f"原始图像尺寸: {orig_size}")
    
    # 前向传播 - 推理模式
    with torch.no_grad():
        extracted_wm, confidence = decoder(image, return_all_blocks=False)
    
    print(f"\n推理模式:")
    print(f"提取水印形状: {extracted_wm.shape}")  # 应该是 (B, 3, 64, 64)
    print(f"置信度形状: {confidence.shape}")
    print(f"置信度: {confidence[0].item():.4f}")
    
    # 前向传播 - 训练模式（返回所有块）
    with torch.no_grad():
        all_wms, all_confs, _ = decoder(image, return_all_blocks=True)
    
    print(f"\n训练模式:")
    print(f"所有水印形状: {all_wms.shape}")  # 应该是 (B, num_blocks, 3, 64, 64)
    print(f"所有置信度形状: {all_confs.shape}")
    
    # 保存提取的水印
    os.makedirs('outputs', exist_ok=True)
    extracted_wm_pil = TF.to_pil_image(extracted_wm[0])
    extracted_wm_pil.save("outputs/test_decoder_extracted.png")
    print(f"\n保存提取的水印: outputs/test_decoder_extracted.png")


if __name__ == "__main__":
    test_decoder()
