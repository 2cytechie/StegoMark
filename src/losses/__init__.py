"""
损失函数模块
"""
from .watermark_loss import WatermarkLoss, PerceptualLoss, SSIMLoss, CombinedWatermarkLoss

__all__ = [
    'WatermarkLoss', 'PerceptualLoss', 'SSIMLoss', 'CombinedWatermarkLoss'
]
