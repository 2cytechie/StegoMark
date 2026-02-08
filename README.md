# StegoMark - 基于DWT+深度学习的盲水印系统

StegoMark 是一个基于离散小波变换(DWT)和深度学习的图像盲水印系统，实现了水印的嵌入和提取功能。

## 项目结构

```
StegoMark/
├── data/                       # 数据集目录
│   ├── train/                  # 训练集
│   │   ├── images/            # 训练用载体图像
│   │   └── watermarks/        # 训练用水印图像
│   └── val/                    # 验证集
│       ├── images/
│       └── watermarks/
├── src/                        # 源代码
│   ├── config.py              # 配置文件
│   ├── data/                  # 数据加载与处理
│   │   ├── dataset.py         # 数据集类
│   │   └── transforms.py      # 数据增强
│   ├── models/                # 模型定义
│   │   ├── encoder.py         # 编码器（水印嵌入）
│   │   ├── decoder.py         # 解码器（水印提取）
│   │   ├── attack_simulator.py # 攻击模拟器
│   │   └── watermark_net.py   # 完整网络
│   ├── dwt/                   # DWT变换模块
│   │   └── dwt_transform.py   # DWT实现
│   ├── utils/                 # 工具函数
│   │   ├── metrics.py         # 评估指标
│   │   ├── losses.py          # 损失函数
│   │   └── visualizer.py      # 可视化工具
│   ├── train.py               # 训练脚本
│   └── extract.py             # 提取脚本
├── checkpoints/               # 模型检查点
├── outputs/                   # 输出结果
├── requirements.txt           # 依赖
├── demo.py                    # 演示脚本
└── README.md                  # 项目说明
```

## 功能特性

### 1. 水印嵌入 (Encoder)
- 基于DWT的频域水印嵌入
- 在高频子带(LH, HL, HH)中嵌入水印
- 使用注意力机制控制嵌入强度
- 残差学习保持图像质量

### 2. 水印提取 (Decoder)
- 双分支网络（频域+空间域）
- 多尺度解码支持
- 置信度预测
- 支持各种攻击后的水印提取

### 3. 攻击模拟
- 高斯模糊
- 高斯噪声
- JPEG压缩
- 随机裁剪
- 旋转
- 颜色调整
- 像素丢弃（遮挡）

### 4. 评估指标
- **PSNR**: 峰值信噪比（图像质量）
- **SSIM**: 结构相似性
- **NC**: 归一化相关系数（水印相似度）
- **BER**: 误码率

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch >= 1.12.0
- PyWavelets >= 1.3.0
- OpenCV >= 4.5.0
- Pillow >= 9.0.0
- scikit-image >= 0.19.0

## 快速开始

### 1. 准备数据

将你的训练数据放入 `data/` 目录：

```bash
data/
├── train/
│   ├── images/        # 载体图像
│   └── watermarks/    # 水印图像
└── val/
    ├── images/
    └── watermarks/
```

### 2. 训练模型

```bash
python -m src.train \
    --train_image_dir data/train/images \
    --train_watermark_dir data/train/watermarks \
    --val_image_dir data/val/images \
    --val_watermark_dir data/val/watermarks \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-4
```

### 3. 嵌入水印

```bash
python -m src.extract \
    --mode embed \
    --checkpoint outputs/best/checkpoints/best.pth \
    --image path/to/image.jpg \
    --watermark path/to/watermark.png \
    --output outputs/watermarked.jpg
```

### 4. 提取水印

```bash
python -m src.extract \
    --mode extract \
    --checkpoint outputs/best/checkpoints/best.pth \
    --image outputs/watermarked.jpg \
    --output outputs/extracted_watermark.png
```

### 5. 运行演示

```bash
# 完整演示
python demo.py

# 使用训练好的模型
python demo.py --checkpoint checkpoints/best.pth

# 仅鲁棒性测试
python demo.py --mode robustness
```

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 16 | 批次大小 |
| `--lr` | 1e-4 | 学习率 |
| `--hidden_dim` | 64 | 隐藏层维度 |
| `--attack_prob` | 0.5 | 攻击模拟概率 |
| `--lambda_image` | 1.0 | 图像失真损失权重 |
| `--lambda_watermark` | 1.0 | 水印提取损失权重 |
| `--use_multiscale` | False | 使用多尺度解码器 |

## 算法流程

### 水印嵌入流程
1. 对载体图像进行DWT分解（LL, LH, HL, HH）
2. 使用CNN提取水印特征
3. 在高频子带(LH, HL, HH)中融合水印特征
4. IDWT重构图像
5. 残差学习优化图像质量

### 水印提取流程
1. 对待检测图像进行多尺度对齐
2. DWT分解提取频域特征
3. 空间域CNN提取特征
4. 双分支特征融合
5. 水印重建网络恢复水印
6. 置信度预测

## 性能指标目标

- **PSNR** > 30 dB（图像质量）
- **NC** > 0.9（水印相关性）
- **BER** < 0.1（误码率）
- **SSIM** > 0.9（结构相似性）

## 技术细节

### DWT变换
- 使用Haar小波基
- 支持多级分解
- 可学习的频域融合

### 网络架构
- **编码器**: ResNet-like + DWT
- **解码器**: U-Net-like + 注意力机制
- **攻击模拟**: 可微分图像变换

### 损失函数
- 图像失真损失 (MSE)
- 水印提取损失 (MSE)
- 同步损失
- 置信度损失 (BCE)
- 感知损失 (VGG特征)

## 可视化工具

```python
from src.utils.visualizer import (
    save_comparison,
    visualize_dwt_subbands,
    save_watermark_robustness_test
)

# 保存对比图
save_comparison(
    original_image, watermarked_image,
    original_watermark, extracted_watermark,
    'comparison.png', metrics
)

# 可视化DWT子带
visualize_dwt_subbands(image, dwt_transform, 'dwt_subbands.png')

# 鲁棒性测试
save_watermark_robustness_test(
    original, watermarked, watermark,
    model, attack_simulator,
    'robustness_test.png'
)
```

## 注意事项

1. **训练数据**: 建议使用多样化的图像进行训练，以提高泛化能力
2. **图像尺寸**: 默认处理64x64的图像块，大图会被分块处理
3. **GPU内存**: 如果显存不足，可以减小batch_size或hidden_dim
4. **攻击鲁棒性**: 训练时启用攻击模拟可以提高模型的鲁棒性
