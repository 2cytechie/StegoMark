import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from src.dwt import DWTLayer, IDWTLayer

class ResidualBlock(nn.Module):
    """
    残差块
    用于构建深度神经网络，缓解梯度消失问题
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        初始化残差块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长，用于下采样
        """
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
            nn.Conv2d(watermark_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 残差网络
        # 计算实际的输入通道数：4个DWT子带 * 输入通道数 + 水印特征通道数
        input_channels = in_channels * 4 + 256
        self.residual_blocks = nn.Sequential(
            ResidualBlock(input_channels, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256)
        )
        
        # 输出层
        self.output_layers = nn.ModuleDict({
            'll': nn.Conv2d(256, in_channels, kernel_size=3, padding=1),
            'lh': nn.Conv2d(256, in_channels, kernel_size=3, padding=1),
            'hl': nn.Conv2d(256, in_channels, kernel_size=3, padding=1),
            'hh': nn.Conv2d(256, in_channels, kernel_size=3, padding=1)
        })
        
        # 水印强度系数
        self.watermark_strength = 0.1  # 降低水印强度
    
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
        
        # 5. 残差网络处理
        residual = self.residual_blocks(fused)
        
        # 6. 生成残差图并添加到原始系数，应用水印强度控制
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
    从含水印图像中提取水印信息
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
        
        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels * 4, 256, kernel_size=3, padding=1),  # 4个DWT子带
            nn.ReLU(),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        
        # 水印输出层
        self.watermark_output = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, watermark_channels, kernel_size=3, padding=1),
            nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True),
            nn.Sigmoid()  # 输出0-1范围
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 含水印图像，形状为 (B, C, H, W)
            
        Returns:
            watermark: 提取的水印，形状为 (B, Cw, Hw, Ww)
        """
        # 1. 执行DWT变换
        coefficients = self.dwt(x)
        ll, lh, hl, hh = coefficients['ll'], coefficients['lh'], coefficients['hl'], coefficients['hh']
        
        # 2. 拼接所有DWT子带
        fused = torch.cat([ll, lh, hl, hh], dim=1)
        
        # 3. 特征提取
        features = self.feature_extractor(fused)
        
        # 4. 提取水印
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
        # 1. 嵌入水印到DWT系数
        watermarked_coefficients = self.encoder(cover_image, watermark)
        
        # 2. IDWT逆变换得到含水印图像
        watermarked_image = self.idwt(watermarked_coefficients)
        
        # 3. 裁剪到原始尺寸并归一化
        watermarked_image = torch.clamp(watermarked_image, 0, 1)
        
        return watermarked_image
    
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