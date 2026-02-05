"""
DWT (Discrete Wavelet Transform) 工具模块
提供小波变换、逆变换以及在水印嵌入/提取中的相关操作
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict
import pywt


class DWTTransform:
    """DWT变换类"""
    
    def __init__(self, wavelet='haar', level=1):
        """
        初始化DWT变换
        
        Args:
            wavelet: 小波类型，如 'haar', 'db1', 'db2', 'bior1.3' 等
            level: 分解层数
        """
        self.wavelet = wavelet
        self.level = level
        self.wavelet_filter = self._get_wavelet_filter()
    
    def _get_wavelet_filter(self):
        """获取小波滤波器系数"""
        w = pywt.Wavelet(self.wavelet)
        
        # 分解滤波器
        dec_lo = np.array(w.dec_lo)  # 低通滤波器
        dec_hi = np.array(w.dec_hi)  # 高通滤波器
        
        return {
            'dec_lo': dec_lo,
            'dec_hi': dec_hi,
            'rec_lo': np.array(w.rec_lo),
            'rec_hi': np.array(w.rec_hi)
        }
    
    def _create_dwt_kernel(self, device='cpu'):
        """创建DWT卷积核"""
        lo = self.wavelet_filter['dec_lo']
        hi = self.wavelet_filter['dec_hi']
        
        # 创建2D卷积核
        # LL: lo * lo^T
        # LH: lo * hi^T
        # HL: hi * lo^T
        # HH: hi * hi^T
        
        kernel_ll = np.outer(lo, lo).reshape(1, 1, len(lo), len(lo))
        kernel_lh = np.outer(lo, hi).reshape(1, 1, len(lo), len(hi))
        kernel_hl = np.outer(hi, lo).reshape(1, 1, len(hi), len(lo))
        kernel_hh = np.outer(hi, hi).reshape(1, 1, len(hi), len(hi))
        
        # 转换为torch张量
        kernel = np.concatenate([kernel_ll, kernel_lh, kernel_hl, kernel_hh], axis=0)
        kernel = torch.from_numpy(kernel).float().to(device)
        
        return kernel
    
    def dwt2(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2D DWT变换（单级）
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            LL, LH, HL, HH: 四个子带
        """
        b, c, h, w = x.shape
        
        # 确保尺寸是偶数
        if h % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode='reflect')
            h += 1
        if w % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode='reflect')
            w += 1
        
        # 为每个通道创建卷积核
        kernel = self._create_dwt_kernel(device=x.device)  # [4, 1, k, k]
        
        # 将输入 reshape 为 [B, C, H, W] -> 分别对每个通道应用DWT
        # 使用 group convolution: 每组处理一个输入通道，产生4个输出通道
        kernel = kernel.repeat(c, 1, 1, 1)  # [C*4, 1, k, k]
        
        # 执行卷积，groups=c 表示每个输入通道独立处理
        # 输出形状: [B, C*4, H/2, W/2]
        output = F.conv2d(x, kernel, stride=2, groups=c)
        
        # reshape 回 [B, C, 4, H/2, W/2]
        output = output.view(b, c, 4, h // 2, w // 2)
        
        # 分离四个子带
        ll = output[:, :, 0, :, :]
        lh = output[:, :, 1, :, :]
        hl = output[:, :, 2, :, :]
        hh = output[:, :, 3, :, :]
        
        return ll, lh, hl, hh
    
    def idwt2(self, ll: torch.Tensor, lh: torch.Tensor, 
              hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        """
        2D IDWT逆变换（单级）
        
        Args:
            ll, lh, hl, hh: 四个子带
        
        Returns:
            重建图像
        """
        b, c, h, w = ll.shape
        
        # 合并四个子带
        subbands = torch.stack([ll, lh, hl, hh], dim=2)  # [B, C, 4, H, W]
        
        # 创建重建卷积核
        kernel = self._create_idwt_kernel(device=ll.device)  # [1, 4, k, k]
        k = kernel.shape[-1]
        
        # 处理每个通道
        outputs = []
        for i in range(c):
            # 获取当前通道的4个子带 [B, 4, H, W]
            channel_subbands = subbands[:, i, :, :, :]  # [B, 4, H, W]
            
            # 使用转置卷积进行上采样和重建
            # F.conv_transpose2d 输入: [B, 4, H, W], 卷积核: [4, 1, k, k]
            # 输出: [B, 1, H*2, W*2]
            channel_kernel = kernel.squeeze(0).unsqueeze(1)  # [4, 1, k, k]
            channel_output = F.conv_transpose2d(channel_subbands, channel_kernel, stride=2, padding=k//2-1)
            outputs.append(channel_output)
        
        # 合并所有通道 [B, C, H*2, W*2]
        output = torch.cat(outputs, dim=1)
        
        return output
    
    def _create_idwt_kernel(self, device='cpu'):
        """创建IDWT卷积核"""
        rec_lo = self.wavelet_filter['rec_lo']
        rec_hi = self.wavelet_filter['rec_hi']
        k = len(rec_lo)
        
        # 创建重建核 [1, 4, k, k] - 输出1通道，输入4通道
        kernel = np.zeros((1, 4, k, k))
        
        # LL使用 rec_lo * rec_lo^T
        kernel[0, 0] = np.outer(rec_lo, rec_lo)
        # LH使用 rec_lo * rec_hi^T
        kernel[0, 1] = np.outer(rec_lo, rec_hi)
        # HL使用 rec_hi * rec_lo^T
        kernel[0, 2] = np.outer(rec_hi, rec_lo)
        # HH使用 rec_hi * rec_hi^T
        kernel[0, 3] = np.outer(rec_hi, rec_hi)
        
        kernel = torch.from_numpy(kernel).float().to(device)
        return kernel


def dwt_transform(image: torch.Tensor, wavelet='haar', level=1) -> Dict[str, torch.Tensor]:
    """
    对图像进行DWT变换
    
    Args:
        image: 输入图像 [B, C, H, W]
        wavelet: 小波类型
        level: 分解层数
    
    Returns:
        包含各子带的字典
    """
    dwt = DWTTransform(wavelet=wavelet, level=level)
    ll, lh, hl, hh = dwt.dwt2(image)
    
    return {
        'LL': ll,
        'LH': lh,
        'HL': hl,
        'HH': hh
    }


def idwt_transform(subbands: Dict[str, torch.Tensor], wavelet='haar') -> torch.Tensor:
    """
    对DWT子带进行逆变换
    
    Args:
        subbands: 包含LL, LH, HL, HH的字典
        wavelet: 小波类型
    
    Returns:
        重建图像
    """
    dwt = DWTTransform(wavelet=wavelet)
    return dwt.idwt2(subbands['LL'], subbands['LH'], subbands['HL'], subbands['HH'])


def embed_to_subbands(subbands: Dict[str, torch.Tensor], 
                      strength_map: torch.Tensor,
                      embed_subbands: List[str] = ['LH', 'HL', 'HH']) -> Dict[str, torch.Tensor]:
    """
    将嵌入强度图嵌入到DWT子带中
    
    Args:
        subbands: DWT子带字典
        strength_map: 嵌入强度图 [B, C, H, W]，与LL子带同尺寸
        embed_subbands: 要嵌入的子带列表
    
    Returns:
        嵌入后的子带字典
    """
    new_subbands = subbands.copy()
    
    # 调整strength_map尺寸以匹配子带
    b, c, h, w = subbands['LL'].shape
    if strength_map.shape[-2:] != (h, w):
        strength_map = F.interpolate(strength_map, size=(h, w), mode='bilinear', align_corners=False)
    
    for band_name in embed_subbands:
        if band_name in new_subbands:
            # 嵌入公式: 新子带 = 原子带 + 强度图
            new_subbands[band_name] = new_subbands[band_name] + strength_map
    
    return new_subbands


def extract_from_subbands(subbands: Dict[str, torch.Tensor],
                          original_subbands: Dict[str, torch.Tensor] = None,
                          extract_subbands: List[str] = ['LH', 'HL', 'HH']) -> torch.Tensor:
    """
    从DWT子带中提取嵌入信息
    
    Args:
        subbands: 含水印的DWT子带字典
        original_subbands: 原始DWT子带字典（盲水印时可为None）
        extract_subbands: 要提取的子带列表
    
    Returns:
        提取的嵌入信息
    """
    extracted = []
    
    for band_name in extract_subbands:
        if band_name in subbands:
            if original_subbands is not None and band_name in original_subbands:
                # 非盲水印: 差值提取
                diff = subbands[band_name] - original_subbands[band_name]
            else:
                # 盲水印: 直接使用子带信息
                diff = subbands[band_name]
            extracted.append(diff)
    
    if extracted:
        # 合并多个子带的提取结果
        result = torch.stack(extracted, dim=0).mean(dim=0)
        return result
    else:
        return None


class DWTLayer(torch.nn.Module):
    """可微分的DWT层，用于神经网络中"""
    
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.dwt = DWTTransform(wavelet=wavelet)
    
    def forward(self, x):
        """前向传播"""
        ll, lh, hl, hh = self.dwt.dwt2(x)
        return {
            'LL': ll,
            'LH': lh,
            'HL': hl,
            'HH': hh
        }


class IDWTLayer(torch.nn.Module):
    """可微分的IDWT层，用于神经网络中"""
    
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet
        self.dwt = DWTTransform(wavelet=wavelet)
    
    def forward(self, subbands):
        """前向传播"""
        return self.dwt.idwt2(
            subbands['LL'],
            subbands['LH'],
            subbands['HL'],
            subbands['HH']
        )


# 使用PyWavelets的备用实现（用于验证）
def dwt2_pywt(image: np.ndarray, wavelet='haar', level=1) -> Dict:
    """
    使用PyWavelets进行DWT变换（numpy版本，用于验证）
    
    Args:
        image: 输入图像 [H, W, C] 或 [H, W]
        wavelet: 小波类型
        level: 分解层数
    
    Returns:
        小波系数列表
    """
    # 转换通道顺序: [H, W, C] -> [C, H, W]
    if len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
    
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs


def idwt2_pywt(coeffs, wavelet='haar') -> np.ndarray:
    """
    使用PyWavelets进行IDWT逆变换（numpy版本，用于验证）
    
    Args:
        coeffs: 小波系数列表
        wavelet: 小波类型
    
    Returns:
        重建图像
    """
    reconstructed = pywt.waverec2(coeffs, wavelet)
    
    # 转换通道顺序: [C, H, W] -> [H, W, C]
    if len(reconstructed.shape) == 3:
        reconstructed = np.transpose(reconstructed, (1, 2, 0))
    
    return reconstructed


if __name__ == '__main__':
    # 测试DWT变换
    print("测试DWT模块...")
    
    # 创建测试图像
    test_image = torch.randn(2, 3, 256, 256)
    print(f"输入图像尺寸: {test_image.shape}")
    
    # 进行DWT变换
    dwt = DWTTransform(wavelet='haar')
    ll, lh, hl, hh = dwt.dwt2(test_image)
    
    print(f"LL子带尺寸: {ll.shape}")
    print(f"LH子带尺寸: {lh.shape}")
    print(f"HL子带尺寸: {hl.shape}")
    print(f"HH子带尺寸: {hh.shape}")
    
    # 进行IDWT逆变换
    reconstructed = dwt.idwt2(ll, lh, hl, hh)
    print(f"重建图像尺寸: {reconstructed.shape}")
    
    # 计算重建误差
    error = torch.mean(torch.abs(test_image - reconstructed))
    print(f"重建误差: {error.item():.6f}")
    
    print("\nDWT模块测试完成！")
