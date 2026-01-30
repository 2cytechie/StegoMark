# 水印提取网络
import torch.nn as nn
from dwt_utils import DWTTransform
from models.stn import STN

class WatermarkDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = STN()
        self.dwt = DWTTransform()
        self.classifier = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stn(x) # 先校正几何畸变
        ll, lh, hl, hh = self.dwt(x)
        # 从中频系数中提取特征
        msg_pred = self.classifier(hl)
        return msg_pred