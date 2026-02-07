import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np


class DWT2D(nn.Module):
    """2D离散小波变换层"""
    
    def __init__(self, wavelet='haar'):
        super(DWT2D, self).__init__()
        self.wavelet = wavelet
        
        # 获取小波滤波器系数
        w = pywt.Wavelet(wavelet)
        
        # 分解滤波器
        dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)
        
        # 构建2D滤波器
        # LL: 低通-低通, LH: 低通-高通, HL: 高通-低通, HH: 高通-高通
        self.register_buffer('ll_filter', self._create_filter(dec_lo, dec_lo))
        self.register_buffer('lh_filter', self._create_filter(dec_lo, dec_hi))
        self.register_buffer('hl_filter', self._create_filter(dec_hi, dec_lo))
        self.register_buffer('hh_filter', self._create_filter(dec_hi, dec_hi))
    
    def _create_filter(self, row_filter, col_filter):
        """创建2D滤波器"""
        # 外积创建2D滤波器
        filter_2d = torch.outer(row_filter, col_filter)
        return filter_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    def forward(self, x):
        """
        输入: (B, C, H, W)
        输出: (B, C*4, H/2, W/2) - 按通道拼接的四个子带
        """
        b, c, h, w = x.shape
        
        # 确保尺寸是偶数
        pad_h = (2 - h % 2) % 2
        pad_w = (2 - w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        # 获取滤波器尺寸
        filter_size = self.ll_filter.shape[-1]
        pad = filter_size // 2
        
        # 对每个通道分别进行DWT
        ll_list, lh_list, hl_list, hh_list = [], [], [], []
        
        for i in range(c):
            channel = x[:, i:i+1, :, :]  # (B, 1, H, W)
            
            # 应用滤波器（手动padding）
            channel_padded = F.pad(channel, (pad, pad, pad, pad), mode='reflect')
            
            ll = F.conv2d(channel_padded, self.ll_filter, stride=2)
            lh = F.conv2d(channel_padded, self.lh_filter, stride=2)
            hl = F.conv2d(channel_padded, self.hl_filter, stride=2)
            hh = F.conv2d(channel_padded, self.hh_filter, stride=2)
            
            ll_list.append(ll)
            lh_list.append(lh)
            hl_list.append(hl)
            hh_list.append(hh)
        
        # 拼接结果
        ll = torch.cat(ll_list, dim=1)  # (B, C, H/2, W/2)
        lh = torch.cat(lh_list, dim=1)
        hl = torch.cat(hl_list, dim=1)
        hh = torch.cat(hh_list, dim=1)
        
        # 按通道拼接四个子带
        output = torch.cat([ll, lh, hl, hh], dim=1)  # (B, C*4, H/2, W/2)
        
        return output
    
    def decompose(self, x):
        """
        分解为四个子带
        返回: ll, lh, hl, hh 每个都是 (B, C, H/2, W/2)
        """
        b, c, h, w = x.shape
        dwt_output = self.forward(x)  # (B, C*4, H/2, W/2)
        
        c_out = c
        ll = dwt_output[:, 0:c_out, :, :]
        lh = dwt_output[:, c_out:2*c_out, :, :]
        hl = dwt_output[:, 2*c_out:3*c_out, :, :]
        hh = dwt_output[:, 3*c_out:4*c_out, :, :]
        
        return ll, lh, hl, hh


class IDWT2D(nn.Module):
    """2D逆离散小波变换层"""
    
    def __init__(self, wavelet='haar'):
        super(IDWT2D, self).__init__()
        self.wavelet = wavelet
        
        # 获取小波滤波器系数
        w = pywt.Wavelet(wavelet)
        
        # 重构滤波器
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
        
        # 构建2D重构滤波器
        self.register_buffer('ll_filter', self._create_filter(rec_lo, rec_lo))
        self.register_buffer('lh_filter', self._create_filter(rec_lo, rec_hi))
        self.register_buffer('hl_filter', self._create_filter(rec_hi, rec_lo))
        self.register_buffer('hh_filter', self._create_filter(rec_hi, rec_hi))
    
    def _create_filter(self, row_filter, col_filter):
        """创建2D滤波器"""
        filter_2d = torch.outer(row_filter, col_filter)
        return filter_2d.unsqueeze(0).unsqueeze(0)
    
    def forward(self, coeffs):
        """
        输入: (B, C*4, H, W) - 四个子带按通道拼接
        输出: (B, C, H*2, W*2)
        """
        b, c4, h, w = coeffs.shape
        c = c4 // 4
        
        # 分离四个子带
        ll = coeffs[:, 0:c, :, :]
        lh = coeffs[:, c:2*c, :, :]
        hl = coeffs[:, 2*c:3*c, :, :]
        hh = coeffs[:, 3*c:4*c, :, :]
        
        # 获取滤波器尺寸
        filter_size = self.ll_filter.shape[-1]
        pad = filter_size // 2
        
        # 上采样并应用滤波器
        output_channels = []
        
        for i in range(c):
            # 上采样
            ll_up = F.interpolate(ll[:, i:i+1, :, :], scale_factor=2, mode='nearest')
            lh_up = F.interpolate(lh[:, i:i+1, :, :], scale_factor=2, mode='nearest')
            hl_up = F.interpolate(hl[:, i:i+1, :, :], scale_factor=2, mode='nearest')
            hh_up = F.interpolate(hh[:, i:i+1, :, :], scale_factor=2, mode='nearest')
            
            # 手动padding
            ll_up = F.pad(ll_up, (pad, pad, pad, pad), mode='reflect')
            lh_up = F.pad(lh_up, (pad, pad, pad, pad), mode='reflect')
            hl_up = F.pad(hl_up, (pad, pad, pad, pad), mode='reflect')
            hh_up = F.pad(hh_up, (pad, pad, pad, pad), mode='reflect')
            
            # 应用重构滤波器
            rec_ll = F.conv2d(ll_up, self.ll_filter, padding=0)
            rec_lh = F.conv2d(lh_up, self.lh_filter, padding=0)
            rec_hl = F.conv2d(hl_up, self.hl_filter, padding=0)
            rec_hh = F.conv2d(hh_up, self.hh_filter, padding=0)
            
            # 求和
            rec = rec_ll + rec_lh + rec_hl + rec_hh
            output_channels.append(rec)
        
        output = torch.cat(output_channels, dim=1)
        return output


class DWTTransform(nn.Module):
    """DWT变换工具类"""
    
    def __init__(self, wavelet='haar', levels=1):
        super(DWTTransform, self).__init__()
        self.wavelet = wavelet
        self.levels = levels
        
        self.dwt = DWT2D(wavelet)
        self.idwt = IDWT2D(wavelet)
    
    def forward(self, x):
        """前向DWT变换"""
        return self.dwt(x)
    
    def inverse(self, coeffs):
        """逆DWT变换"""
        return self.idwt(coeffs)
    
    def decompose(self, x):
        """分解为四个子带"""
        return self.dwt.decompose(x)
    
    def compose(self, ll, lh, hl, hh):
        """将四个子带合成为DWT输出格式"""
        return torch.cat([ll, lh, hl, hh], dim=1)


def test_dwt():
    """测试DWT模块"""
    # 创建测试数据
    x = torch.randn(2, 3, 64, 64)
    
    # DWT变换
    dwt = DWT2D('haar')
    coeffs = dwt(x)
    print(f"DWT输出形状: {coeffs.shape}")  # 应该是 (2, 12, 32, 32)
    
    # 分解子带
    ll, lh, hl, hh = dwt.decompose(x)
    print(f"LL形状: {ll.shape}")  # (2, 3, 32, 32)
    print(f"LH形状: {lh.shape}")
    print(f"HL形状: {hl.shape}")
    print(f"HH形状: {hh.shape}")
    
    # 逆变换
    idwt = IDWT2D('haar')
    reconstructed = idwt(coeffs)
    print(f"重构形状: {reconstructed.shape}")  # (2, 3, 64, 64)
    
    # 计算重构误差
    error = torch.mean(torch.abs(x - reconstructed))
    print(f"重构误差: {error.item():.6f}")


if __name__ == "__main__":
    test_dwt()