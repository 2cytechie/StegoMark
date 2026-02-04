import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from config import config
from src.dwt import DWTLayer, IDWTLayer

class ResidualBlock(nn.Module):
    """
    残差块
    用于构建深度神经网络，缓解梯度消失问题
    """
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        """
        初始化残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长，用于下采样
            dropout_rate: Dropout概率
        """
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 捷径连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            out: 输出张量
        """
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class Encoder(nn.Module):
    """
    水印嵌入编码器网络
    将水印信息嵌入到载体图像的DWT系数中
    """
    
    def __init__(self, in_channels=config.IMAGE_CHANNELS, watermark_channels=1, device=None):
        """
        初始化编码器
        
        Args:
            in_channels: 输入图像通道数
            watermark_channels: 水印通道数
            device: 目标设备，默认None
        """
        super(Encoder, self).__init__()
        
        # DWT变换层
        self.dwt = DWTLayer(device=device)
        
        # 水印特征处理
        self.watermark_processor = nn.Sequential(
            nn.Conv2d(watermark_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 残差网络
        # 计算实际的输入通道数：4个DWT子带 * 输入通道数 + 水印特征通道数
        input_channels = in_channels * 4 + 128
        self.residual_blocks = nn.Sequential(
            ResidualBlock(input_channels, 128, dropout_rate=0.3),
            ResidualBlock(128, 128, dropout_rate=0.3)
        )
        
        # 输出层
        self.output_layers = nn.ModuleDict({
            'll': nn.Sequential(
                nn.Conv2d(128, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels)
            ),
            'lh': nn.Sequential(
                nn.Conv2d(128, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels)
            ),
            'hl': nn.Sequential(
                nn.Conv2d(128, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels)
            ),
            'hh': nn.Sequential(
                nn.Conv2d(128, in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels)
            )
        })
        
        # 水印强度系数
        self.watermark_strength = random.uniform(0.01, config.WATERMARK_STRENGTH)  # 从配置中获取水印强度
    
    def forward(self, x, watermark):
        """
        前向传播
        
        Args:
            x: 载体图像，形状为 (B, C, H, W)
            watermark: 水印，形状为 (B, Cw, Hw, Ww)
            
        Returns:
            watermarked_coefficients: 含水印的DWT系数
        """
        # 1. 执行DWT变换
        coefficients = self.dwt(x)
        ll, lh, hl, hh = coefficients['ll'], coefficients['lh'], coefficients['hl'], coefficients['hh']
        
        # 2. 处理水印
        watermark_feat = self.watermark_processor(watermark)
        
        # 3. 融合特征
        b, c, h, w = ll.shape
        watermark_feat = F.interpolate(watermark_feat, size=(h, w), mode='bilinear', align_corners=True)
        
        # 4. 确保所有张量维度一致
        if len(watermark_feat.shape) == 3:
            watermark_feat = watermark_feat.unsqueeze(0)
        if len(ll.shape) == 3:
            ll = ll.unsqueeze(0)
        if len(lh.shape) == 3:
            lh = lh.unsqueeze(0)
        if len(hl.shape) == 3:
            hl = hl.unsqueeze(0)
        if len(hh.shape) == 3:
            hh = hh.unsqueeze(0)
        
        # 5. 拼接所有DWT子带和水印特征
        fused = torch.cat([ll, lh, hl, hh, watermark_feat], dim=1)
        
        # 6. 残差网络处理
        residual = self.residual_blocks(fused)
        
        # 7. 生成残差图并添加到原始系数，应用水印强度控制
        watermarked_coefficients = {
            'll': ll + self.watermark_strength * self.output_layers['ll'](residual),
            'lh': lh + self.watermark_strength * self.output_layers['lh'](residual),
            'hl': hl + self.watermark_strength * self.output_layers['hl'](residual),
            'hh': hh + self.watermark_strength * self.output_layers['hh'](residual)
        }
        
        return watermarked_coefficients

class Decoder(nn.Module):
    """
    水印提取解码器网络
    从含水印图像中提取水印信息，支持几何变换参数估计与校正
    """
    
    def __init__(self, in_channels=config.IMAGE_CHANNELS, watermark_channels=1, device=None):
        """
        初始化解码器
        
        Args:
            in_channels: 输入图像通道数
            watermark_channels: 水印通道数
            device: 目标设备，默认None
        """
        super(Decoder, self).__init__()
        
        # DWT变换层
        self.dwt = DWTLayer(device=device)
        self.device = device
        
        # 几何变换参数估计模块（简化版）
        self.transform_estimator = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(32, 64, dropout_rate=0.3),
            ResidualBlock(64, 128, dropout_rate=0.3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 5)  # 5个参数：旋转角度、缩放比例、水平平移、垂直平移、裁剪比例
        )
        
        # 注意力机制模块 - CBAM (Convolutional Block Attention Module)
        class AttentionBlock(nn.Module):
            def __init__(self, in_channels, reduction=16):
                super(AttentionBlock, self).__init__()
                
                # 通道注意力模块
                self.channel_attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
                    nn.Sigmoid()
                )
                
                # 空间注意力模块
                self.spatial_attention = nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                    nn.Sigmoid()
                )
                
            def forward(self, x):
                # 通道注意力
                ca = self.channel_attention(x)
                x = x * ca
                
                # 空间注意力
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                sa_input = torch.cat([avg_out, max_out], dim=1)
                sa = self.spatial_attention(sa_input)
                x = x * sa
                
                return x
        
        # 多尺度特征提取网络
        self.feature_extractor = nn.ModuleList([
            # 浅层特征（高分辨率）
            nn.Sequential(
                nn.Conv2d(in_channels * 4, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                ResidualBlock(32, 32, dropout_rate=0.3),
                AttentionBlock(32)
            ),
            # 中层特征
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                ResidualBlock(64, 64, dropout_rate=0.3),
                AttentionBlock(64)
            ),
            # 深层特征（低分辨率）
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                ResidualBlock(128, 128, dropout_rate=0.3),
                AttentionBlock(128)
            )
        ])
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(32 + 64 + 128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            ResidualBlock(128, 64, dropout_rate=0.3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Dropout(0.3),
            ResidualBlock(64, 32, dropout_rate=0.3),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # 水印输出层
        self.watermark_output = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(16, watermark_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(watermark_channels),
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True),
            nn.Sigmoid()  # 输出0-1范围
        )
    
    def estimate_transform(self, x):
        """
        估计几何变换参数
        
        Args:
            x: 输入图像张量
            
        Returns:
            transform_params: 估计的几何变换参数 (旋转角度, 缩放比例, 水平平移, 垂直平移, 裁剪比例)
        """
        params = self.transform_estimator(x)
        # 归一化参数到合理范围
        rotation = params[:, 0] * config.MAX_ROTATION_ANGLE  # 旋转角度: -90°至90°
        scale = params[:, 1] * (config.MAX_SCALE - config.MIN_SCALE) + config.MIN_SCALE  # 缩放比例: 0.5至2.0
        tx = params[:, 2] * 0.2  # 水平平移: -0.2至0.2（归一化坐标）
        ty = params[:, 3] * 0.2  # 垂直平移: -0.2至0.2（归一化坐标）
        crop_ratio = params[:, 4] * (config.MAX_CROP_RATIO - config.MIN_CROP_RATIO) + config.MIN_CROP_RATIO  # 裁剪比例: 0.5至0.9
        
        return rotation, scale, tx, ty, crop_ratio
    
    def apply_inverse_transform(self, x, rotation, scale, tx, ty, crop_ratio):
        """
        应用逆几何变换恢复图像
        
        Args:
            x: 输入图像张量
            rotation: 旋转角度
            scale: 缩放比例
            tx: 水平平移
            ty: 垂直平移
            crop_ratio: 裁剪比例
            
        Returns:
            restored_x: 恢复后的图像张量
        """
        b, c, h, w = x.shape
        
        # 创建逆旋转矩阵
        theta = -rotation * math.pi / 180  # 取负角度进行逆旋转
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        # 创建逆缩放矩阵
        inv_scale = 1.0 / scale
        
        # 创建逆平移
        inv_tx = -tx
        inv_ty = -ty
        
        # 构建仿射变换矩阵
        transform = torch.zeros(b, 2, 3, device=self.device)
        transform[:, 0, 0] = cos_theta * inv_scale
        transform[:, 0, 1] = -sin_theta * inv_scale
        transform[:, 0, 2] = inv_tx
        transform[:, 1, 0] = sin_theta * inv_scale
        transform[:, 1, 1] = cos_theta * inv_scale
        transform[:, 1, 2] = inv_ty
        
        # 应用逆变换
        grid = F.affine_grid(transform, x.size(), align_corners=False)
        restored_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        return restored_x
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 含水印图像，形状为 (B, C, H, W)
            
        Returns:
            watermark: 提取的水印，形状为 (B, Cw, Hw, Ww)
        """
        # 1. 几何变换参数估计与校正
        rotation, scale, tx, ty, crop_ratio = self.estimate_transform(x)
        restored_x = self.apply_inverse_transform(x, rotation, scale, tx, ty, crop_ratio)
        
        # 2. 获取水印副本数量
        grid_size = config.GRID_SIZE
        
        # 3. 输入参数有效性检查
        if grid_size < 1:
            raise ValueError("GRID_SIZE must be at least 1")
        
        # 检查图像尺寸
        b, c, h, w = restored_x.shape
        if h < grid_size or w < grid_size:
            raise ValueError("Image size must be larger than GRID_SIZE")
        
        # 4. 如果只需要一个水印副本，使用原始方法
        if grid_size == 1:
            # 执行DWT变换
            coefficients = self.dwt(restored_x)
            ll, lh, hl, hh = coefficients['ll'], coefficients['lh'], coefficients['hl'], coefficients['hh']
            
            # 拼接所有DWT子带
            fused = torch.cat([ll, lh, hl, hh], dim=1)
            
            # 多尺度特征提取
            features1 = self.feature_extractor[0](fused)
            features2 = self.feature_extractor[1](features1)
            features3 = self.feature_extractor[2](features2)
            
            # 调整特征尺寸以进行融合
            features2_upsampled = F.interpolate(features2, size=features1.shape[2:], mode='bilinear', align_corners=True)
            features3_upsampled = F.interpolate(features3, size=features1.shape[2:], mode='bilinear', align_corners=True)
            
            # 特征融合
            fused_features = torch.cat([features1, features2_upsampled, features3_upsampled], dim=1)
            features = self.feature_fusion(fused_features)
            
            # 提取水印
            watermark = self.watermark_output(features)
            
            return watermark
        
        # 5. 多副本水印提取
        b, c, h, w = restored_x.shape
        watermarks = []
        
        # 计算每个网格的大小（确保所有网格大小一致）
        grid_h = (h + grid_size - 1) // grid_size
        grid_w = (w + grid_size - 1) // grid_size
        
        # 从每个网格区域提取水印
        for i in range(grid_size):
            for j in range(grid_size):
                if len(watermarks) >= grid_size * grid_size:
                    break
                
                # 计算当前网格的边界
                start_h = i * grid_h
                end_h = start_h + grid_h
                start_w = j * grid_w
                end_w = start_w + grid_w
                
                # 确保边界不超出图像范围
                end_h = min(end_h, h)
                end_w = min(end_w, w)
                
                # 提取网格区域
                region = restored_x[:, :, start_h:end_h, start_w:end_w]
                
                # 对区域执行DWT变换
                region_coefficients = self.dwt(region)
                region_ll, region_lh, region_hl, region_hh = region_coefficients['ll'], region_coefficients['lh'], region_coefficients['hl'], region_coefficients['hh']
                
                # 拼接所有DWT子带
                region_fused = torch.cat([region_ll, region_lh, region_hl, region_hh], dim=1)
                
                # 多尺度特征提取
                region_features1 = self.feature_extractor[0](region_fused)
                region_features2 = self.feature_extractor[1](region_features1)
                region_features3 = self.feature_extractor[2](region_features2)
                
                # 调整特征尺寸以进行融合
                region_features2_upsampled = F.interpolate(region_features2, size=region_features1.shape[2:], mode='bilinear', align_corners=True)
                region_features3_upsampled = F.interpolate(region_features3, size=region_features1.shape[2:], mode='bilinear', align_corners=True)
                
                # 特征融合
                region_fused_features = torch.cat([region_features1, region_features2_upsampled, region_features3_upsampled], dim=1)
                region_features = self.feature_fusion(region_fused_features)
                
                # 提取水印
                region_watermark = self.watermark_output(region_features)
                watermarks.append(region_watermark)
            
        
        # 6. 融合多个水印副本
        if watermarks:
            # 计算所有水印副本的平均值
            fused_watermark = torch.mean(torch.stack(watermarks), dim=0)
            return fused_watermark
        else:
            # 如果没有提取到水印副本，使用原始方法
            coefficients = self.dwt(restored_x)
            ll, lh, hl, hh = coefficients['ll'], coefficients['lh'], coefficients['hl'], coefficients['hh']
            fused = torch.cat([ll, lh, hl, hh], dim=1)
            
            # 多尺度特征提取
            features1 = self.feature_extractor[0](fused)
            features2 = self.feature_extractor[1](features1)
            features3 = self.feature_extractor[2](features2)
            
            # 调整特征尺寸以进行融合
            features2_upsampled = F.interpolate(features2, size=features1.shape[2:], mode='bilinear', align_corners=True)
            features3_upsampled = F.interpolate(features3, size=features1.shape[2:], mode='bilinear', align_corners=True)
            
            # 特征融合
            fused_features = torch.cat([features1, features2_upsampled, features3_upsampled], dim=1)
            features = self.feature_fusion(fused_features)
            
            watermark = self.watermark_output(features)
            return watermark

class WatermarkModel(nn.Module):
    """
    完整的水印嵌入与提取模型
    """
    
    def __init__(self, watermark_type='image', device=None):
        """
        初始化水印模型
        
        Args:
            watermark_type: 水印类型，'image'或'text'
            device: 目标设备，默认None
        """
        super(WatermarkModel, self).__init__()
        
        self.watermark_type = watermark_type
        self.device = device
        watermark_channels = 1 if watermark_type == 'image' else config.TEXT_WATERMARK_LENGTH // (config.WATERMARK_SIZE * config.WATERMARK_SIZE)
        
        # 编码器和解码器
        self.encoder = Encoder(watermark_channels=watermark_channels, device=device)
        self.decoder = Decoder(watermark_channels=watermark_channels, device=device)
        
        # IDWT逆变换层
        self.idwt = IDWTLayer(device=device)
    
    def embed(self, cover_image, watermark):
        """
        嵌入水印
        
        Args:
            cover_image: 载体图像
            watermark: 水印
            
        Returns:
            watermarked_image: 含水印图像
        """
        # 1. 获取水印副本数量
        grid_size = config.GRID_SIZE
        
        # 2. 输入参数有效性检查
        if grid_size < 1:
            raise ValueError("GRID_SIZE must be at least 1")
        
        # 检查图像尺寸
        b, c, h, w = cover_image.shape
        if h < grid_size or w < grid_size:
            raise ValueError("Image size must be larger than GRID_SIZE")
        
        # 3. 如果只需要一个水印副本，使用原始方法
        if grid_size == 1:
            # 嵌入水印到DWT系数
            watermarked_coefficients = self.encoder(cover_image, watermark)
            
            # IDWT逆变换得到含水印图像
            watermarked_image = self.idwt(watermarked_coefficients)
            
            # 裁剪到原始尺寸并归一化
            watermarked_image = torch.clamp(watermarked_image, 0, 1)
            
            return watermarked_image
        
        # 3. 多副本水印嵌入
        b, c, h, w = cover_image.shape
        watermarked_images = []
        
        # 计算每个网格的大小（确保所有网格大小一致）
        grid_h = (h + grid_size - 1) // grid_size
        grid_w = (w + grid_size - 1) // grid_size
        
        # 对每个网格区域嵌入水印
        for i in range(grid_size):
            for j in range(grid_size):
                if len(watermarked_images) >= grid_size * grid_size:
                    break
                
                # 计算当前网格的边界
                start_h = i * grid_h
                end_h = start_h + grid_h
                start_w = j * grid_w
                end_w = start_w + grid_w
                
                # 确保边界不超出图像范围
                end_h = min(end_h, h)
                end_w = min(end_w, w)
                
                # 提取网格区域
                region = cover_image[:, :, start_h:end_h, start_w:end_w]
                
                # 对区域嵌入水印
                region_coefficients = self.encoder(region, watermark)
                region_watermarked = self.idwt(region_coefficients)
                region_watermarked = torch.clamp(region_watermarked, 0, 1)
                
                # 创建完整大小的图像，只在对应区域嵌入水印
                full_image = cover_image.clone()
                full_image[:, :, start_h:end_h, start_w:end_w] = region_watermarked
                
                watermarked_images.append(full_image)
            
        
        # 4. 融合多个水印图像
        if watermarked_images:
            # 计算所有水印图像的平均值
            fused_image = torch.mean(torch.stack(watermarked_images), dim=0)
            fused_image = torch.clamp(fused_image, 0, 1)
            return fused_image
        else:
            # 如果没有嵌入水印副本，使用原始图像
            return cover_image
    
    def extract(self, watermarked_image):
        """
        提取水印
        
        Args:
            watermarked_image: 含水印图像
            
        Returns:
            extracted_watermark: 提取的水印
        """
        extracted_watermark = self.decoder(watermarked_image)
        return extracted_watermark
    
    def forward(self, cover_image, watermark):
        """
        前向传播（用于训练）
        
        Args:
            cover_image: 载体图像
            watermark: 水印
            
        Returns:
            watermarked_image: 含水印图像
            extracted_watermark: 提取的水印
        """
        watermarked_image = self.embed(cover_image, watermark)
        extracted_watermark = self.extract(watermarked_image)
        return watermarked_image, extracted_watermark