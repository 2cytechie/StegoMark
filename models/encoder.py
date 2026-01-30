# 水印嵌入网络 (含DWT)
# encoder.py
import torch
import torch.nn as nn
from dwt_utils import DWTTransform, IDWTTransform

class WatermarkEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWTTransform()
        self.idwt = IDWTTransform()
        
        # 可学习的嵌入强度参数
        self.embedding_strength = nn.Parameter(torch.tensor(0.01))
        
        # 水印特征映射网络
        self.msg_projection = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # 自适应嵌入网络
        self.adaptive_embed = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 融合中频系数
        self.embed_layer = nn.Conv2d(4, 3, 3, padding=1)

    def forward(self, img, msg):
        # msg: [B, 64]
        original_h, original_w = img.shape[2], img.shape[3]
        ll, lh, hl, hh = self.dwt(img)
        
        # 获取HL子带的尺寸
        b, c, h, w = hl.shape
        
        # 改进的水印特征映射
        msg_feat = self.msg_projection(msg)  # [B, 256]
        
        # 将水印特征映射到空间域
        msg_feat = msg_feat.view(b, 256, 1, 1).expand(-1, -1, h, w)
        
        # 使用1x1卷积将256维特征映射到3维
        msg_feat = msg_feat.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        
        # 自适应嵌入强度：根据图像内容调整
        adaptive_strength = self.adaptive_embed(img)
        adaptive_strength = adaptive_strength.mean(dim=[2, 3], keepdim=True)  # [B, 3, 1, 1]
        
        # 应用可学习的嵌入强度和自适应强度
        strength = torch.sigmoid(self.embedding_strength) * adaptive_strength
        
        # 仅修改 HL 子带进行嵌入，使用更精细的嵌入策略
        hl_new = hl + msg_feat * strength
        
        res = self.idwt(ll, lh, hl_new, hh)
        
        # 裁剪回原始尺寸
        res = res[:, :, :original_h, :original_w]
        
        return res
