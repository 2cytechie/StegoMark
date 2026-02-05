"""
水印图像处理工具模块
提供水印的预处理（边缘填充、resize、复制）和后处理功能
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Tuple, Dict
import torchvision.transforms as transforms


class WatermarkPreprocessor:
    """水印预处理器"""
    
    def __init__(self, watermark_size: int = 64, target_size: int = 256):
        """
        初始化预处理器
        
        Args:
            watermark_size: 水印最终尺寸（resize后）
            target_size: 目标图像尺寸（用于计算复制倍数）
        """
        self.watermark_size = watermark_size
        self.target_size = target_size
        self.tile_factor = target_size // watermark_size  # 复制倍数，应该是4
    
    def pad_to_square(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """
        将图像边缘填充为1:1正方形
        
        Args:
            image: 输入图像
        
        Returns:
            填充后的正方形图像
        """
        if isinstance(image, Image.Image):
            w, h = image.size
            max_size = max(w, h)
            
            # 创建正方形背景
            new_image = Image.new('RGB', (max_size, max_size), (255, 255, 255))
            
            # 计算居中位置
            left = (max_size - w) // 2
            top = (max_size - h) // 2
            
            # 粘贴原图
            new_image.paste(image, (left, top))
            return new_image
        
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            max_size = max(w, h)
            
            # 计算填充
            pad_h = (max_size - h) // 2
            pad_w = (max_size - w) // 2
            
            if len(image.shape) == 3:
                # 彩色图像 [H, W, C]
                padded = np.pad(image, ((pad_h, max_size - h - pad_h), 
                                        (pad_w, max_size - w - pad_w), 
                                        (0, 0)), 
                               mode='constant', constant_values=255)
            else:
                # 灰度图像 [H, W]
                padded = np.pad(image, ((pad_h, max_size - h - pad_h), 
                                        (pad_w, max_size - w - pad_w)), 
                               mode='constant', constant_values=255)
            return padded
        
        elif isinstance(image, torch.Tensor):
            # 假设输入是 [C, H, W] 或 [B, C, H, W]
            if image.dim() == 3:
                c, h, w = image.shape
                max_size = max(h, w)
                pad_h = (max_size - h) // 2
                pad_w = (max_size - w) // 2
                
                padded = F.pad(image, (pad_w, max_size - w - pad_w, 
                                       pad_h, max_size - h - pad_h), 
                              mode='constant', value=1.0)
            else:
                b, c, h, w = image.shape
                max_size = max(h, w)
                pad_h = (max_size - h) // 2
                pad_w = (max_size - w) // 2
                
                padded = F.pad(image, (pad_w, max_size - w - pad_w, 
                                       pad_h, max_size - h - pad_h), 
                              mode='constant', value=1.0)
            return padded
        
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
    
    def resize(self, image: Union[Image.Image, np.ndarray, torch.Tensor], 
               size: int = None) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        """
        将图像resize到指定尺寸
        
        Args:
            image: 输入图像
            size: 目标尺寸，默认为self.watermark_size
        
        Returns:
            resize后的图像
        """
        if size is None:
            size = self.watermark_size
        
        if isinstance(image, Image.Image):
            return image.resize((size, size), Image.LANCZOS)
        
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize((size, size), Image.LANCZOS)
            return np.array(pil_img)
        
        elif isinstance(image, torch.Tensor):
            # 假设输入是 [C, H, W] 或 [B, C, H, W]
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            resized = F.interpolate(image, size=(size, size), 
                                   mode='bilinear', align_corners=False)
            
            if image.shape[0] == 1:
                resized = resized.squeeze(0)
            
            return resized
        
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
    
    def tile_watermark(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        将水印图像复制为4x4的平铺图像
        
        Args:
            image: 输入图像 [H, W, C] 或 [C, H, W]
        
        Returns:
            平铺后的图像 [4H, 4W, C] 或 [C, 4H, 4W]
        """
        if isinstance(image, np.ndarray):
            # 假设输入是 [H, W, C]
            if len(image.shape) == 3:
                # 水平复制4次
                tiled_h = np.tile(image, (1, self.tile_factor, 1))
                # 垂直复制4次
                tiled = np.tile(tiled_h, (self.tile_factor, 1, 1))
            else:
                # 灰度图像 [H, W]
                tiled_h = np.tile(image, (1, self.tile_factor))
                tiled = np.tile(tiled_h, (self.tile_factor, 1))
            return tiled
        
        elif isinstance(image, torch.Tensor):
            # 假设输入是 [C, H, W] 或 [B, C, H, W]
            if image.dim() == 3:
                # [C, H, W] -> 水平复制
                tiled_h = image.repeat(1, 1, self.tile_factor)
                # 垂直复制
                tiled = tiled_h.repeat(1, self.tile_factor, 1)
            else:
                # [B, C, H, W]
                tiled_h = image.repeat(1, 1, 1, self.tile_factor)
                tiled = tiled_h.repeat(1, 1, self.tile_factor, 1)
            return tiled
        
        else:
            raise TypeError(f"不支持的图像类型: {type(image)}")
    
    def preprocess(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        完整预处理流程: 边缘填充 -> resize -> 复制4x4
        
        Args:
            image: 输入水印图像
        
        Returns:
            预处理后的水印图像 (256x256)
        """
        # 1. 边缘填充为1:1
        squared = self.pad_to_square(image)
        
        # 2. resize到64x64
        resized = self.resize(squared, self.watermark_size)
        
        # 3. 复制4x4得到256x256
        tiled = self.tile_watermark(resized)
        
        return tiled


def preprocess_watermark(image_path: str, watermark_size: int = 64, 
                        target_size: int = 256) -> torch.Tensor:
    """
    从文件路径预处理水印图像
    
    Args:
        image_path: 水印图像路径
        watermark_size: 水印最终尺寸
        target_size: 目标图像尺寸
    
    Returns:
        预处理后的水印张量 [3, 256, 256]
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    preprocessor = WatermarkPreprocessor(watermark_size, target_size)
    
    # 填充为正方形
    squared = preprocessor.pad_to_square(image)
    
    # resize到64x64
    resized = preprocessor.resize(squared, watermark_size)
    
    # 转换为numpy数组
    resized_np = np.array(resized).astype(np.float32) / 255.0
    
    # 转换为张量 [H, W, C] -> [C, H, W]
    tensor = torch.from_numpy(resized_np).permute(2, 0, 1)
    
    # 复制4x4
    tiled = preprocessor.tile_watermark(tensor)
    
    return tiled


def preprocess_watermark_to_64x64(image_path: str, watermark_size: int = 64) -> torch.Tensor:
    """
    预处理水印图像为64x64（用于提取网络的输出）
    
    Args:
        image_path: 水印图像路径
        watermark_size: 水印尺寸
    
    Returns:
        预处理后的水印张量 [3, 64, 64]
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    
    # 预处理
    preprocessor = WatermarkPreprocessor(watermark_size, watermark_size * 4)
    
    # 填充为正方形
    squared = preprocessor.pad_to_square(image)
    
    # resize到64x64
    resized = preprocessor.resize(squared, watermark_size)
    
    # 转换为numpy数组
    resized_np = np.array(resized).astype(np.float32) / 255.0
    
    # 转换为张量 [H, W, C] -> [C, H, W]
    tensor = torch.from_numpy(resized_np).permute(2, 0, 1)
    
    return tensor


def postprocess_extracted_watermark(extracted_tensor: torch.Tensor, 
                                    output_path: str = None) -> Image.Image:
    """
    后处理提取的水印张量，转换为图像
    
    Args:
        extracted_tensor: 提取的水印张量 [C, H, W] 或 [1, C, H, W]
        output_path: 保存路径（可选）
    
    Returns:
        PIL图像
    """
    # 处理batch维度
    if extracted_tensor.dim() == 4:
        extracted_tensor = extracted_tensor.squeeze(0)
    
    # 确保值在[0, 1]范围内
    extracted_tensor = torch.clamp(extracted_tensor, 0, 1)
    
    # 转换为numpy [C, H, W] -> [H, W, C]
    image_np = extracted_tensor.permute(1, 2, 0).cpu().numpy()
    
    # 转换为uint8
    image_np = (image_np * 255).astype(np.uint8)
    
    # 创建PIL图像
    image = Image.fromarray(image_np)
    
    # 保存（如果指定了路径）
    if output_path:
        image.save(output_path)
    
    return image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    将张量转换为PIL图像

    Args:
        tensor: 输入张量 [C, H, W] 或 [H, W]，值范围[0, 1]

    Returns:
        PIL图像
    """
    if tensor.dim() == 3:
        # [C, H, W] -> [H, W, C]
        image_np = tensor.permute(1, 2, 0).cpu().numpy()
    else:
        # [H, W]
        image_np = tensor.cpu().numpy()

    # 确保值在[0, 255]范围内
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(image_np)


def resize_to_original(image: Image.Image, original_size: Tuple[int, int],
                       resample: int = Image.LANCZOS) -> Image.Image:
    """
    将图像调整回原始尺寸，保持宽高比

    Args:
        image: 当前图像（PIL Image）
        original_size: 原始尺寸 (width, height)
        resample: 重采样算法，默认LANCZOS

    Returns:
        调整尺寸后的图像
    """
    original_width, original_height = original_size

    # 如果尺寸已经一致，直接返回
    if image.size == original_size:
        return image

    # 使用高质量的重采样算法调整尺寸
    resized_image = image.resize(original_size, resample)

    return resized_image


def get_image_info(image: Image.Image) -> dict:
    """
    获取图像的详细信息

    Args:
        image: PIL图像

    Returns:
        包含图像信息的字典
    """
    return {
        'size': image.size,
        'width': image.width,
        'height': image.height,
        'mode': image.mode,
        'format': image.format if hasattr(image, 'format') else None
    }


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    将PIL图像转换为张量
    
    Args:
        image: PIL图像
    
    Returns:
        张量 [C, H, W]，值范围[0, 1]
    """
    image_np = np.array(image).astype(np.float32) / 255.0
    
    if len(image_np.shape) == 3:
        # [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    else:
        # [H, W] -> [1, H, W]
        tensor = torch.from_numpy(image_np).unsqueeze(0)
    
    return tensor


def watermark_to_binary(watermark_tensor: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    将水印张量转换为二值形式（用于计算准确率）

    Args:
        watermark_tensor: 水印张量
        threshold: 二值化阈值

    Returns:
        二值化后的张量
    """
    return (watermark_tensor > threshold).float()


def calculate_watermark_accuracy(extracted: torch.Tensor,
                                 original: torch.Tensor,
                                 threshold: float = 0.5) -> float:
    """
    计算水印提取准确率

    Args:
        extracted: 提取的水印
        original: 原始水印
        threshold: 二值化阈值

    Returns:
        准确率（0-1之间）
    """
    # 二值化
    extracted_binary = watermark_to_binary(extracted, threshold)
    original_binary = watermark_to_binary(original, threshold)

    # 计算准确率
    correct = (extracted_binary == original_binary).float()
    accuracy = correct.mean().item()

    return accuracy


class ColorWatermarkAccuracyCalculator:
    """
    彩色水印准确率计算器
    支持RGB三通道分别计算和综合计算，提供多种判定标准
    """

    def __init__(self, tolerance: float = 0.1, threshold: float = 0.5):
        """
        初始化准确率计算器

        Args:
            tolerance: 像素值误差容忍范围（0-1之间），默认0.1表示±10%误差可接受
            threshold: 二值化阈值，默认0.5
        """
        self.tolerance = tolerance
        self.threshold = threshold

    def calculate_exact_match_accuracy(self, extracted: torch.Tensor,
                                       original: torch.Tensor) -> Dict[str, float]:
        """
        计算精确匹配准确率（像素值完全相同）

        Args:
            extracted: 提取的水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]
            original: 原始水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]

        Returns:
            包含各通道和综合准确率的字典
        """
        # 确保张量维度一致
        if extracted.dim() == 3:
            extracted = extracted.unsqueeze(0)
        if original.dim() == 3:
            original = original.unsqueeze(0)

        # 确保batch大小一致
        if extracted.shape[0] != original.shape[0]:
            raise ValueError(f"Batch大小不一致: {extracted.shape[0]} vs {original.shape[0]}")

        results = {}

        # 计算每个通道的准确率
        channel_names = ['R', 'G', 'B']
        channel_accuracies = []

        for c in range(3):
            extracted_channel = extracted[:, c, :, :]
            original_channel = original[:, c, :, :]

            # 精确匹配（完全相同的像素值）
            correct = (extracted_channel == original_channel).float()
            accuracy = correct.mean().item()

            results[f'{channel_names[c]}_channel'] = accuracy
            channel_accuracies.append(accuracy)

        # 综合准确率（三通道正确像素数 / 三通道总像素数）
        results['overall'] = sum(channel_accuracies) / 3

        return results

    def calculate_tolerance_accuracy(self, extracted: torch.Tensor,
                                     original: torch.Tensor) -> Dict[str, float]:
        """
        计算容差匹配准确率（像素值在误差范围内视为正确）

        Args:
            extracted: 提取的水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]
            original: 原始水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]

        Returns:
            包含各通道和综合准确率的字典
        """
        # 确保张量维度一致
        if extracted.dim() == 3:
            extracted = extracted.unsqueeze(0)
        if original.dim() == 3:
            original = original.unsqueeze(0)

        results = {}

        # 计算每个通道的准确率
        channel_names = ['R', 'G', 'B']
        channel_accuracies = []
        total_correct_pixels = 0
        total_pixels = 0

        for c in range(3):
            extracted_channel = extracted[:, c, :, :]
            original_channel = original[:, c, :, :]

            # 计算像素值差异
            diff = torch.abs(extracted_channel - original_channel)

            # 在容差范围内的像素视为正确
            correct = (diff <= self.tolerance).float()
            accuracy = correct.mean().item()

            # 统计正确像素数
            correct_pixels = correct.sum().item()
            total_channel_pixels = correct.numel()

            results[f'{channel_names[c]}_channel'] = accuracy
            results[f'{channel_names[c]}_correct_pixels'] = int(correct_pixels)
            results[f'{channel_names[c]}_total_pixels'] = int(total_channel_pixels)

            channel_accuracies.append(accuracy)
            total_correct_pixels += correct_pixels
            total_pixels += total_channel_pixels

        # 综合准确率（三通道正确像素总数 / 三通道总像素数）
        results['overall'] = total_correct_pixels / total_pixels if total_pixels > 0 else 0
        results['overall_by_average'] = sum(channel_accuracies) / 3
        results['total_correct_pixels'] = int(total_correct_pixels)
        results['total_pixels'] = int(total_pixels)
        results['tolerance'] = self.tolerance

        return results

    def calculate_binary_accuracy(self, extracted: torch.Tensor,
                                  original: torch.Tensor) -> Dict[str, float]:
        """
        计算二值化准确率（将水印二值化后比较）

        Args:
            extracted: 提取的水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]
            original: 原始水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]

        Returns:
            包含各通道和综合准确率的字典
        """
        # 二值化
        extracted_binary = watermark_to_binary(extracted, self.threshold)
        original_binary = watermark_to_binary(original, self.threshold)

        # 确保张量维度一致
        if extracted_binary.dim() == 3:
            extracted_binary = extracted_binary.unsqueeze(0)
        if original_binary.dim() == 3:
            original_binary = original_binary.unsqueeze(0)

        results = {}

        # 计算每个通道的准确率
        channel_names = ['R', 'G', 'B']
        channel_accuracies = []
        total_correct_pixels = 0
        total_pixels = 0

        for c in range(3):
            extracted_channel = extracted_binary[:, c, :, :]
            original_channel = original_binary[:, c, :, :]

            # 二值匹配
            correct = (extracted_channel == original_channel).float()
            accuracy = correct.mean().item()

            # 统计正确像素数
            correct_pixels = correct.sum().item()
            total_channel_pixels = correct.numel()

            results[f'{channel_names[c]}_channel'] = accuracy
            results[f'{channel_names[c]}_correct_pixels'] = int(correct_pixels)
            results[f'{channel_names[c]}_total_pixels'] = int(total_channel_pixels)

            channel_accuracies.append(accuracy)
            total_correct_pixels += correct_pixels
            total_pixels += total_channel_pixels

        # 综合准确率
        results['overall'] = total_correct_pixels / total_pixels if total_pixels > 0 else 0
        results['overall_by_average'] = sum(channel_accuracies) / 3
        results['total_correct_pixels'] = int(total_correct_pixels)
        results['total_pixels'] = int(total_pixels)
        results['threshold'] = self.threshold

        return results

    def calculate_all_metrics(self, extracted: torch.Tensor,
                              original: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        计算所有准确率指标

        Args:
            extracted: 提取的水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]
            original: 原始水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]

        Returns:
            包含所有指标的字典
        """
        return {
            'exact_match': self.calculate_exact_match_accuracy(extracted, original),
            'tolerance_match': self.calculate_tolerance_accuracy(extracted, original),
            'binary_match': self.calculate_binary_accuracy(extracted, original)
        }

    def print_accuracy_report(self, extracted: torch.Tensor, original: torch.Tensor):
        """
        打印准确率报告

        Args:
            extracted: 提取的水印
            original: 原始水印
        """
        metrics = self.calculate_all_metrics(extracted, original)

        print("\n" + "="*60)
        print("彩色水印准确率评估报告")
        print("="*60)

        # 精确匹配
        print("\n【精确匹配准确率】（像素值完全相同）")
        print(f"  R通道: {metrics['exact_match']['R_channel']:.4f} ({metrics['exact_match']['R_channel']*100:.2f}%)")
        print(f"  G通道: {metrics['exact_match']['G_channel']:.4f} ({metrics['exact_match']['G_channel']*100:.2f}%)")
        print(f"  B通道: {metrics['exact_match']['B_channel']:.4f} ({metrics['exact_match']['B_channel']*100:.2f}%)")
        print(f"  综合:  {metrics['exact_match']['overall']:.4f} ({metrics['exact_match']['overall']*100:.2f}%)")

        # 容差匹配
        print(f"\n【容差匹配准确率】（误差容忍范围: ±{self.tolerance*100:.1f}%）")
        print(f"  R通道: {metrics['tolerance_match']['R_channel']:.4f} ({metrics['tolerance_match']['R_channel']*100:.2f}%)")
        print(f"  G通道: {metrics['tolerance_match']['G_channel']:.4f} ({metrics['tolerance_match']['G_channel']*100:.2f}%)")
        print(f"  B通道: {metrics['tolerance_match']['B_channel']:.4f} ({metrics['tolerance_match']['B_channel']*100:.2f}%)")
        print(f"  综合:  {metrics['tolerance_match']['overall']:.4f} ({metrics['tolerance_match']['overall']*100:.2f}%)")
        print(f"  正确像素数: {metrics['tolerance_match']['total_correct_pixels']} / {metrics['tolerance_match']['total_pixels']}")

        # 二值匹配
        print(f"\n【二值匹配准确率】（阈值: {self.threshold}）")
        print(f"  R通道: {metrics['binary_match']['R_channel']:.4f} ({metrics['binary_match']['R_channel']*100:.2f}%)")
        print(f"  G通道: {metrics['binary_match']['G_channel']:.4f} ({metrics['binary_match']['G_channel']*100:.2f}%)")
        print(f"  B通道: {metrics['binary_match']['B_channel']:.4f} ({metrics['binary_match']['B_channel']*100:.2f}%)")
        print(f"  综合:  {metrics['binary_match']['overall']:.4f} ({metrics['binary_match']['overall']*100:.2f}%)")
        print(f"  正确像素数: {metrics['binary_match']['total_correct_pixels']} / {metrics['binary_match']['total_pixels']}")

        print("="*60)


def calculate_color_watermark_accuracy(extracted: torch.Tensor,
                                       original: torch.Tensor,
                                       method: str = 'tolerance',
                                       tolerance: float = 0.1,
                                       threshold: float = 0.5) -> Dict[str, float]:
    """
    计算彩色水印提取准确率的便捷函数

    Args:
        extracted: 提取的水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]
        original: 原始水印 [C, H, W] 或 [B, C, H, W]，值范围[0, 1]
        method: 计算方法 ('exact', 'tolerance', 'binary')
        tolerance: 容差范围（0-1之间）
        threshold: 二值化阈值

    Returns:
        准确率结果字典
    """
    calculator = ColorWatermarkAccuracyCalculator(tolerance=tolerance, threshold=threshold)

    if method == 'exact':
        return calculator.calculate_exact_match_accuracy(extracted, original)
    elif method == 'tolerance':
        return calculator.calculate_tolerance_accuracy(extracted, original)
    elif method == 'binary':
        return calculator.calculate_binary_accuracy(extracted, original)
    else:
        raise ValueError(f"未知的计算方法: {method}")


class WatermarkDatasetTransform:
    """水印数据集的数据变换"""
    
    def __init__(self, watermark_size: int = 64, target_size: int = 256):
        self.watermark_size = watermark_size
        self.target_size = target_size
        self.preprocessor = WatermarkPreprocessor(watermark_size, target_size)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        变换水印图像
        
        Args:
            image: PIL图像
        
        Returns:
            变换后的张量 [3, 256, 256]
        """
        # 填充为正方形
        squared = self.preprocessor.pad_to_square(image)
        
        # resize到64x64
        resized = self.preprocessor.resize(squared, self.watermark_size)
        
        # 转换为张量
        tensor = pil_to_tensor(resized)
        
        # 复制4x4
        tiled = self.preprocessor.tile_watermark(tensor)
        
        return tiled


if __name__ == '__main__':
    # 测试水印预处理
    print("测试水印预处理模块...")
    
    # 创建测试图像
    test_image = Image.new('RGB', (100, 80), color='red')
    print(f"原始图像尺寸: {test_image.size}")
    
    # 测试预处理
    preprocessor = WatermarkPreprocessor(watermark_size=64, target_size=256)
    
    # 1. 测试边缘填充
    squared = preprocessor.pad_to_square(test_image)
    print(f"填充后尺寸: {squared.size}")
    
    # 2. 测试resize
    resized = preprocessor.resize(squared, 64)
    print(f"resize后尺寸: {resized.size}")
    
    # 3. 测试复制
    resized_np = np.array(resized)
    tiled = preprocessor.tile_watermark(resized_np)
    print(f"复制后尺寸: {tiled.shape}")
    
    # 4. 测试完整预处理流程
    test_image_np = np.array(test_image)
    full_processed = preprocessor.preprocess(test_image_np)
    print(f"完整预处理后尺寸: {full_processed.shape}")
    
    # 5. 测试张量输入
    test_tensor = torch.randn(3, 64, 64)
    tiled_tensor = preprocessor.tile_watermark(test_tensor)
    print(f"张量复制后尺寸: {tiled_tensor.shape}")
    
    print("\n水印预处理模块测试完成！")
