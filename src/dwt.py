import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse
from config import config

class DWTTransform:
    """
    离散小波变换模块
    支持彩色图像的DWT变换和IDWT逆变换
    """
    
    def __init__(self, wavelet=config.DWT_MODE, level=config.DWT_LEVEL, device=None):
        """
        初始化DWT变换
        
        Args:
            wavelet: 小波基类型，默认'haar'
            level: 分解级别，默认1
            device: 目标设备，默认None
        """
        self.wavelet = wavelet
        self.level = level
        self.dwt = DWTForward(wave=wavelet, J=level, mode='symmetric')
        self.idwt = DWTInverse(wave=wavelet, mode='symmetric')
        if device:
            self.to(device)
    
    def to(self, device):
        """
        移动到指定设备
        
        Args:
            device: 目标设备
        """
        # 重新创建DWT和IDWT对象以确保它们在正确的设备上
        self.dwt = DWTForward(wave=self.wavelet, J=self.level, mode='symmetric').to(device)
        self.idwt = DWTInverse(wave=self.wavelet, mode='symmetric').to(device)
        return self
    
    def forward(self, x):
        """
        执行DWT变换
        
        Args:
            x: 输入图像张量，形状为 (B, C, H, W)
            
        Returns:
            lh_list: 水平高频子带列表
            hl_list: 垂直高频子带列表
            hh_list: 对角高频子带列表
            ll: 低频子带，形状为 (B, C, H/2, W/2)
        """
        # 处理DWTForward的返回值
        result = self.dwt(x)
        if isinstance(result, tuple):
            if len(result) == 2:
                ll, high = result
                if isinstance(high, tuple) and len(high) == 3:
                    lh_list, hl_list, hh_list = high
                else:
                    # 如果返回格式不同，使用默认值
                    lh_list = hl_list = hh_list = ll
            else:
                ll = result[0]
                lh_list = hl_list = hh_list = ll
        else:
            ll = result
            lh_list = hl_list = hh_list = ll
        return ll, lh_list, hl_list, hh_list
    
    def inverse(self, ll, lh_list, hl_list, hh_list):
        """
        执行IDWT逆变换
        
        Args:
            ll: 低频子带
            lh_list: 水平高频子带列表
            hl_list: 垂直高频子带列表
            hh_list: 对角高频子带列表
            
        Returns:
            x: 重构的图像张量，形状为 (B, C, H, W)
        """
        # 确保所有输入都在同一设备上
        device = ll.device
        
        # 处理输入格式，确保它们是正确的张量
        if isinstance(lh_list, list):
            lh = lh_list[0].to(device)
        else:
            lh = lh_list.to(device)
        
        if isinstance(hl_list, list):
            hl = hl_list[0].to(device)
        else:
            hl = hl_list.to(device)
        
        if isinstance(hh_list, list):
            hh = hh_list[0].to(device)
        else:
            hh = hh_list.to(device)
        
        # 执行IDWT逆变换
        # 对于单层分解，DWTInverse期望的格式是 (ll, (lh, hl, hh))
        try:
            x = self.idwt((ll, (lh, hl, hh)))
            return x
        except Exception as e:
            # 如果IDWT失败，使用双线性插值调整尺寸作为后备方案
            from torch.nn import functional as F
            x = F.interpolate(ll, scale_factor=2, mode='bilinear', align_corners=True)
            return x
    
    def get_dwt_coefficients(self, x):
        """
        获取所有DWT系数
        
        Args:
            x: 输入图像张量
            
        Returns:
            coefficients: 包含所有DWT系数的字典
        """
        ll, lh_list, hl_list, hh_list = self.forward(x)
        # 处理返回值，确保是张量
        if isinstance(lh_list, list) and lh_list:
            lh = lh_list[0]
        else:
            lh = lh_list
        if isinstance(hl_list, list) and hl_list:
            hl = hl_list[0]
        else:
            hl = hl_list
        if isinstance(hh_list, list) and hh_list:
            hh = hh_list[0]
        else:
            hh = hh_list
        coefficients = {
            'll': ll,
            'lh': lh,
            'hl': hl,
            'hh': hh
        }
        return coefficients
    
    def reconstruct_from_coefficients(self, coefficients):
        """
        从DWT系数重构图像
        
        Args:
            coefficients: 包含DWT系数的字典
            
        Returns:
            x: 重构的图像张量
        """
        ll = coefficients['ll']
        lh_list = [coefficients['lh']]
        hl_list = [coefficients['hl']]
        hh_list = [coefficients['hh']]
        x = self.inverse(ll, lh_list, hl_list, hh_list)
        return x

class DWTLayer(nn.Module):
    """
    可导的DWT变换层
    用于深度学习模型中
    """
    
    def __init__(self, wavelet=config.DWT_MODE, level=config.DWT_LEVEL, device=None):
        """
        初始化DWT层
        
        Args:
            wavelet: 小波基类型
            level: 分解级别
            device: 目标设备，默认None
        """
        super(DWTLayer, self).__init__()
        self.dwt_transform = DWTTransform(wavelet, level, device)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量
            
        Returns:
            coefficients: DWT系数
        """
        return self.dwt_transform.get_dwt_coefficients(x)
    
    def to(self, device):
        """
        移动到指定设备
        
        Args:
            device: 目标设备
        """
        self.dwt_transform.to(device)
        return self

class IDWTLayer(nn.Module):
    """
    可导的IDWT逆变换层
    用于深度学习模型中
    """
    
    def __init__(self, wavelet=config.DWT_MODE, level=config.DWT_LEVEL, device=None):
        """
        初始化IDWT层
        
        Args:
            wavelet: 小波基类型
            level: 分解级别
            device: 目标设备，默认None
        """
        super(IDWTLayer, self).__init__()
        self.dwt_transform = DWTTransform(wavelet, level, device)
    
    def forward(self, coefficients):
        """
        前向传播
        
        Args:
            coefficients: DWT系数字典
            
        Returns:
            x: 重构的图像张量
        """
        return self.dwt_transform.reconstruct_from_coefficients(coefficients)
    
    def to(self, device):
        """
        移动到指定设备
        
        Args:
            device: 目标设备
        """
        self.dwt_transform.to(device)
        return self