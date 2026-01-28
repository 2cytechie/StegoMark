# DWT变换相关操作
import torch
import torch.nn as nn

def get_dwt_filters():
    # Haar 小波滤波器
    low = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    h_h = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
    h_v = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
    h_d = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
    filters = torch.stack([low, h_h, h_v, h_d]).unsqueeze(1)
    return filters

class DWTTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('filters', get_dwt_filters())

    def forward(self, x):
        # x: [B, 3, H, W]
        b, c, h, w = x.shape
        
        # 处理奇数尺寸的情况
        pad_h = 1 if h % 2 != 0 else 0
        pad_w = 1 if w % 2 != 0 else 0
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            h, w = x.shape[2], x.shape[3]
        
        # 优化：使用分组卷积替代reshape，减少内存使用
        # 将通道维度与批量维度合并，使用分组卷积
        out = torch.nn.functional.conv2d(x, self.filters.repeat(c, 1, 1, 1), 
                                         stride=2, groups=c)
        # 重新排列维度
        out = out.view(b, 4, c, h//2, w//2).permute(0, 2, 1, 3, 4)
        # 返回 LL, LH, HL, HH
        return out[:,:,0], out[:,:,1], out[:,:,2], out[:,:,3]

class IDWTTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # 逆变换使用转置卷积简化
        self.register_buffer('filters', get_dwt_filters())

    def forward(self, ll, lh, hl, hh):
        b, c, h, w = ll.shape
        # 优化：使用分组转置卷积替代reshape，减少内存使用
        combined = torch.stack([ll, lh, hl, hh], dim=2).permute(0, 2, 1, 3, 4).reshape(b, 4*c, h, w)
        out = torch.nn.functional.conv_transpose2d(combined, self.filters.repeat(c, 1, 1, 1), 
                                                  stride=2, groups=c)
        return out.view(b, c, h*2, w*2)