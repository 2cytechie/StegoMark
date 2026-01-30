# 空间变换网络 (抗旋转、缩放)
import torch
import torch.nn as nn
import torch.nn.functional as F

class STN(nn.Module):
    def __init__(self):
        super().__init__()
        self.locator = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 6) # 输出仿射变换矩阵参数
        )
        # 初始化为恒等变换
        self.locator[-1].weight.data.zero_()
        self.locator[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.locator(x).view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x