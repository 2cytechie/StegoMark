"""
DWT (离散小波变换) 工具模块
"""
import torch
import torch.nn.functional as F
import pywt
import numpy as np


class DWT2D:
    """2D离散小波变换"""
    
    def __init__(self, wavelet='haar', mode='reflect'):
        """
        Args:
            wavelet: 小波基名称, 默认'haar'
            mode: 边界处理模式 (PyTorch只支持 'constant', 'reflect', 'replicate', 'circular')
        """
        self.wavelet = wavelet
        self.mode = mode
        self.w = pywt.Wavelet(wavelet)
        
    def get_filters(self, device='cpu'):
        """获取小波滤波器并转换为torch张量"""
        # 分解滤波器
        dec_lo = torch.tensor(self.w.dec_lo, dtype=torch.float32, device=device)
        dec_hi = torch.tensor(self.w.dec_hi, dtype=torch.float32, device=device)
        
        # 重构滤波器
        rec_lo = torch.tensor(self.w.rec_lo, dtype=torch.float32, device=device)
        rec_hi = torch.tensor(self.w.rec_hi, dtype=torch.float32, device=device)
        
        return dec_lo, dec_hi, rec_lo, rec_hi
    
    def dwt2(self, x):
        """
        2D DWT分解
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            LL, LH, HL, HH: 四个子带 [B, C, H//2, W//2]
        """
        B, C, H, W = x.shape
        device = x.device
        
        dec_lo, dec_hi, _, _ = self.get_filters(device)
        
        # 构建2D滤波器
        # 低通滤波器
        l_lo = dec_lo.view(1, 1, -1, 1) * dec_lo.view(1, 1, 1, -1)  # LL
        l_hi = dec_lo.view(1, 1, -1, 1) * dec_hi.view(1, 1, 1, -1)  # LH
        h_lo = dec_hi.view(1, 1, -1, 1) * dec_lo.view(1, 1, 1, -1)  # HL
        h_hi = dec_hi.view(1, 1, -1, 1) * dec_hi.view(1, 1, 1, -1)  # HH
        
        # 对每个通道应用滤波器
        ll_list, lh_list, hl_list, hh_list = [], [], [], []
        
        for c in range(C):
            x_c = x[:, c:c+1, :, :]
            
            # 行方向滤波
            pad = len(dec_lo) // 2
            x_pad = F.pad(x_c, (0, 0, pad, pad), mode=self.mode)
            
            # 低通行滤波
            x_lo = F.conv2d(x_pad, dec_lo.view(1, 1, -1, 1), stride=(2, 1))
            # 高通行滤波
            x_hi = F.conv2d(x_pad, dec_hi.view(1, 1, -1, 1), stride=(2, 1))
            
            # 列方向滤波
            x_lo_pad = F.pad(x_lo, (pad, pad, 0, 0), mode=self.mode)
            x_hi_pad = F.pad(x_hi, (pad, pad, 0, 0), mode=self.mode)
            
            # LL
            ll = F.conv2d(x_lo_pad, dec_lo.view(1, 1, 1, -1), stride=(1, 2))
            # LH
            lh = F.conv2d(x_lo_pad, dec_hi.view(1, 1, 1, -1), stride=(1, 2))
            # HL
            hl = F.conv2d(x_hi_pad, dec_lo.view(1, 1, 1, -1), stride=(1, 2))
            # HH
            hh = F.conv2d(x_hi_pad, dec_hi.view(1, 1, 1, -1), stride=(1, 2))
            
            ll_list.append(ll)
            lh_list.append(lh)
            hl_list.append(hl)
            hh_list.append(hh)
        
        LL = torch.cat(ll_list, dim=1)
        LH = torch.cat(lh_list, dim=1)
        HL = torch.cat(hl_list, dim=1)
        HH = torch.cat(hh_list, dim=1)
        
        return LL, LH, HL, HH
    
    def idwt2(self, LL, LH, HL, HH):
        """
        2D IDWT重构
        
        Args:
            LL, LH, HL, HH: 四个子带 [B, C, H, W]
            
        Returns:
            x: 重构图像 [B, C, H*2, W*2]
        """
        B, C, H, W = LL.shape
        device = LL.device
        
        _, _, rec_lo, rec_hi = self.get_filters(device)
        
        # 对每个通道重构
        x_list = []
        
        for c in range(C):
            ll = LL[:, c:c+1, :, :]
            lh = LH[:, c:c+1, :, :]
            hl = HL[:, c:c+1, :, :]
            hh = HH[:, c:c+1, :, :]
            
            # 上采样并在列方向重构
            ll_up = F.interpolate(ll, scale_factor=(1, 2), mode='nearest')
            lh_up = F.interpolate(lh, scale_factor=(1, 2), mode='nearest')
            hl_up = F.interpolate(hl, scale_factor=(1, 2), mode='nearest')
            hh_up = F.interpolate(hh, scale_factor=(1, 2), mode='nearest')
            
            pad = len(rec_lo) // 2
            
            # 列方向重构
            ll_up_pad = F.pad(ll_up, (pad, pad, 0, 0), mode=self.mode)
            lh_up_pad = F.pad(lh_up, (pad, pad, 0, 0), mode=self.mode)
            hl_up_pad = F.pad(hl_up, (pad, pad, 0, 0), mode=self.mode)
            hh_up_pad = F.pad(hh_up, (pad, pad, 0, 0), mode=self.mode)
            
            # 低通行列重构
            x_lo = F.conv2d(ll_up_pad, rec_lo.view(1, 1, 1, -1), padding=0) + \
                   F.conv2d(lh_up_pad, rec_hi.view(1, 1, 1, -1), padding=0)
            # 高通行列重构
            x_hi = F.conv2d(hl_up_pad, rec_lo.view(1, 1, 1, -1), padding=0) + \
                   F.conv2d(hh_up_pad, rec_hi.view(1, 1, 1, -1), padding=0)
            
            # 上采样并在行方向重构
            x_lo_up = F.interpolate(x_lo, scale_factor=(2, 1), mode='nearest')
            x_hi_up = F.interpolate(x_hi, scale_factor=(2, 1), mode='nearest')
            
            x_lo_up_pad = F.pad(x_lo_up, (0, 0, pad, pad), mode=self.mode)
            x_hi_up_pad = F.pad(x_hi_up, (0, 0, pad, pad), mode=self.mode)
            
            # 行方向重构
            x_c = F.conv2d(x_lo_up_pad, rec_lo.view(1, 1, -1, 1), padding=0) + \
                  F.conv2d(x_hi_up_pad, rec_hi.view(1, 1, -1, 1), padding=0)
            
            x_list.append(x_c)
        
        x = torch.cat(x_list, dim=1)
        return x


class DWTTransform:
    """可微分的DWT变换层"""
    
    def __init__(self, wavelet='haar', mode='symmetric'):
        self.dwt = DWT2D(wavelet, mode)
    
    def forward(self, x):
        return self.dwt.dwt2(x)
    
    def inverse(self, LL, LH, HL, HH):
        return self.dwt.idwt2(LL, LH, HL, HH)


def pad_to_multiple(x, multiple=2):
    """将图像填充到2的倍数，便于DWT分解"""
    B, C, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    
    return x, (pad_h, pad_w)


def remove_padding(x, pad_info):
    """移除填充"""
    pad_h, pad_w = pad_info
    if pad_h > 0 or pad_w > 0:
        x = x[:, :, :-pad_h if pad_h > 0 else None, :-pad_w if pad_w > 0 else None]
    return x
