# 可微分攻击模拟层 (噪声、裁剪、几何变换)
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffAttack(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 1. 随机噪声
        if self.training:
            x = x + torch.randn_like(x) * 0.05
            
            # 2. 模拟随机旋转和缩放 (使用仿射变换)
            angle = (torch.rand(1) - 0.5) * 0.4 # +- 0.2 rad
            scale = 1.0 + (torch.rand(1) - 0.5) * 0.2 # 0.9 - 1.1
            
            theta = torch.zeros(x.size(0), 2, 3).to(x.device)
            theta[:, 0, 0] = torch.cos(angle) * scale
            theta[:, 0, 1] = -torch.sin(angle)
            theta[:, 1, 0] = torch.sin(angle)
            theta[:, 1, 1] = torch.cos(angle) * scale
            
            grid = F.affine_grid(theta, x.size(), align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False)
            
            # 3. 模拟裁剪 (简单遮盖)
            mask = torch.ones_like(x)
            h_start = torch.randint(0, 50, (1,))
            w_start = torch.randint(0, 50, (1,))
            mask[:, :, h_start:h_start+100, w_start:w_start+100] = 0
            x = x * mask
            
        return x