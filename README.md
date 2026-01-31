# StegoMark：基于深度学习与DWT的图像隐水印系统

## 项目概述

StegoMark是一个基于深度学习和离散小波变换(DWT)的图像隐水印系统，旨在提供一种高效、鲁棒的数字水印解决方案。该系统能够将图像或文本水印嵌入到载体图像中，并在需要时准确提取，同时保持载体图像的视觉质量。

### 主要功能

- **双水印类型支持**：同时支持图像水印和文本水印的嵌入与提取
- **基于DWT的嵌入算法**：利用离散小波变换实现水印的频域嵌入，提高鲁棒性
- **深度学习模型**：采用编码器-解码器架构，结合残差网络提高水印提取精度
- **对抗性训练**：支持对抗性训练，增强水印对常见攻击的抵抗力
- **批量处理能力**：提供批量水印嵌入和提取功能，提高处理效率
- **性能评估**：内置PSNR和SSIM等评估指标，量化水印对载体图像的影响

## 环境要求与依赖项

### 系统要求

- Python 3.8+
- CUDA 10.2+ (推荐，用于GPU加速)
- 至少4GB内存
- 20GB存储空间

### 依赖项

项目依赖以下Python库：

```plaintext
# 基础依赖
numpy>=1.19.5
pandas>=1.1.5
matplotlib>=3.3.4
opencv-python>=4.5.1
scikit-image>=0.18.1
Pillow>=8.1.0

# 深度学习框架
torch>=1.8.0
torchvision>=0.9.0

# DWT变换
pytorch-wavelets>=1.3.0

# 对抗性训练
torchattacks>=3.4.0
robustbench>=0.2.1

# 性能评估
tqdm>=4.56.0
scipy>=1.6.0

# 开发工具
flake8>=3.8.4
black>=20.8b1
```

## 安装与部署

### 步骤1：克隆项目

```bash
git clone [仓库地址]
cd StegoMark
```

### 步骤2：安装依赖

使用pip安装所有依赖项：

```bash
pip install -r requirements.txt
```

### 步骤3：配置项目

根据需要修改`config.py`文件中的配置参数：

```python
# 示例配置修改
IMAGE_SIZE = 256      # 载体图像尺寸
WATERMARK_SIZE = 64    # 水印尺寸
DWT_LEVEL = 1          # DWT分解级别
DWT_MODE = 'haar'      # DWT小波基
```

## 使用示例

### 1. 水印嵌入

#### 嵌入图像水印

```python
from src.embedding import WatermarkEmbedding

# 初始化嵌入器
embedder = WatermarkEmbedding(watermark_type='image')

# 加载预训练模型（如果有）
# embedder.load_model('output/model_image_epoch_5.pth')

# 嵌入水印
cover_image_path = 'img/target.png'
watermark_path = 'img/watermark.png'
output_path = 'img/watermarked.png'

watermarked_image = embedder.embed(cover_image_path, watermark_path, output_path)
print(f"水印嵌入完成，结果保存至: {output_path}")
```

#### 嵌入文本水印

```python
from src.embedding import WatermarkEmbedding

# 初始化嵌入器
embedder = WatermarkEmbedding(watermark_type='text')

# 嵌入水印
cover_image_path = 'img/target.png'
text_watermark = 'StegoMark: Image Watermarking System'
output_path = 'img/watermarked_text.png'

watermarked_image = embedder.embed(cover_image_path, text_watermark, output_path)
print(f"文本水印嵌入完成，结果保存至: {output_path}")
```

### 2. 水印提取

#### 提取图像水印

```python
from src.extraction import WatermarkExtraction

# 初始化提取器
extractor = WatermarkExtraction(watermark_type='image')

# 加载预训练模型（与嵌入时使用的模型相同）
# extractor.load_model('output/model_image_epoch_5.pth')

# 提取水印
watermarked_image_path = 'img/watermarked.png'
extracted_watermark_path = 'img/extracted_watermark.png'

extracted_watermark = extractor.extract(watermarked_image_path, extracted_watermark_path)
print(f"水印提取完成，结果保存至: {extracted_watermark_path}")
```

#### 提取文本水印

```python
from src.extraction import WatermarkExtraction

# 初始化提取器
extractor = WatermarkExtraction(watermark_type='text')

# 提取水印
watermarked_image_path = 'img/watermarked_text.png'

extracted_text = extractor.extract(watermarked_image_path)
print(f"提取的文本水印: {extracted_text}")
```

### 3. 批量处理

```python
from src.embedding import BatchEmbedding
from src.extraction import BatchExtraction

# 批量嵌入
batch_embedder = BatchEmbedding(watermark_type='image')
cover_images = ['img/target1.png', 'img/target2.png']
watermarks = ['img/watermark.png', 'img/watermark.png']

cover_tensors, watermark_tensors = batch_embedder.preprocess_batch(cover_images, watermarks)
watermarked_images = batch_embedder.embed_batch(cover_tensors, watermark_tensors)

# 批量提取
batch_extractor = BatchExtraction(watermark_type='image')
watermarked_tensors = batch_extractor.preprocess_batch(['img/watermarked1.png', 'img/watermarked2.png'])
extracted_watermarks = batch_extractor.extract_batch(watermarked_tensors)
results = batch_extractor.process_batch_results(extracted_watermarks, output_dir='output')
```

### 4. 模型训练

使用提供的训练脚本进行模型训练：

```bash
python scripts/train.py --train_type image --epochs 10 --batch_size 8
```

训练参数说明：
- `--train_type`：训练类型，可选 'image' 或 'text'
- `--epochs`：训练轮数
- `--batch_size`：批次大小
- `--learning_rate`：学习率

## 配置说明

项目配置集中在 `config.py` 文件中，主要配置项如下：

| 配置项 | 说明 | 默认值 |
|-------|------|-------|
| `IMAGE_SIZE` | 载体图像尺寸 | 256 |
| `WATERMARK_SIZE` | 水印尺寸 | 64 |
| `IMAGE_CHANNELS` | 图像通道数 | 3 (彩色) |
| `DWT_LEVEL` | DWT分解级别 | 1 |
| `DWT_MODE` | DWT小波基 | 'haar' |
| `WATERMARK_TYPES` | 支持的水印类型 | ['image', 'text'] |
| `TEXT_WATERMARK_LENGTH` | 文本水印长度 (bits) | 64 |
| `TRAIN_TYPE` | 训练类型 | 'image' |
| `BATCH_SIZE` | 批次大小 | 8 |
| `EPOCHS` | 训练轮数 | 5 |
| `LEARNING_RATE` | 学习率 | 1e-4 |
| `ADVERSARIAL_TRAINING` | 是否开启对抗性训练 | True |
| `ATTACK_EPS` | 攻击步长 | 0.03 |
| `ATTACK_ITERATIONS` | 攻击迭代次数 | 10 |
| `PSNR_THRESHOLD` | PSNR阈值 | 30.0 |
| `SSIM_THRESHOLD` | SSIM阈值 | 0.95 |
| `OUTPUT_DIR` | 输出目录 | 'output' |
| `DEVICE` | 运行设备 | 'cuda' (如果可用) 或 'cpu' |

## 模型架构

StegoMark采用编码器-解码器架构，结合离散小波变换(DWT)实现水印的嵌入与提取：

### 编码器
1. 对载体图像进行DWT变换，得到四个子带(LL, LH, HL, HH)
2. 对水印进行预处理和特征提取
3. 将水印特征与DWT子带融合
4. 通过残差网络处理融合特征
5. 生成残差图并添加到原始DWT系数

### 解码器
1. 对含水印图像进行DWT变换
2. 提取DWT子带特征
3. 通过残差网络处理特征
4. 输出提取的水印

### 核心技术
- **DWT变换**：在频域进行水印嵌入，提高鲁棒性
- **残差网络**：解决深度网络梯度消失问题，提高模型性能
- **对抗性训练**：增强水印对常见攻击的抵抗力

## 性能评估

项目内置了多种性能评估指标，用于量化水印系统的性能：

### 1. 视觉质量评估
- **PSNR (峰值信噪比)**：衡量载体图像与含水印图像的差异，值越高越好
- **SSIM (结构相似性指数)**：衡量图像结构相似性，值越接近1越好

### 2. 水印鲁棒性评估
- **提取准确率**：比较提取的水印与原始水印的相似度
- **抗攻击能力**：评估水印在常见攻击下的表现（如噪声、压缩、裁剪等）

### 3. 评估示例

```python
from src.extraction import WatermarkExtraction

# 初始化提取器
extractor = WatermarkExtraction(watermark_type='image')

# 提取水印
extracted_watermark = extractor.extract('img/watermarked.png')

# 评估提取准确率
accuracy = extractor.evaluate_extraction_accuracy(extracted_watermark, 'img/watermark.png')
print(f"水印提取准确率: {accuracy:.4f}")
```

## 常见问题解答

### Q1: 水印嵌入后，载体图像质量下降明显怎么办？

**A1:** 可以通过以下方式改善：
- 调整模型训练参数，增加对载体图像质量的重视
- 减小水印强度（通过修改模型或配置）
- 使用更高质量的载体图像

### Q2: 水印提取失败或提取结果与原始水印差异较大怎么办？

**A2:** 可能的原因及解决方案：
- 模型未充分训练：增加训练轮数或调整训练参数
- 载体图像受到严重攻击：增强模型的对抗性训练
- 水印尺寸过大：尝试使用更小的水印

### Q3: 如何提高系统的处理速度？

**A3:** 可以通过以下方式优化：
- 使用GPU加速（确保CUDA环境正确配置）
- 增加批量处理大小
- 对模型进行量化或剪枝

### Q4: 系统支持哪些类型的图像格式？

**A4:** 系统支持常见的图像格式，如PNG、JPEG、BMP等，通过Pillow库进行处理。

## 攻击测试

项目提供了攻击测试脚本，用于评估水印系统对常见攻击的抵抗力：

```bash
python scripts/attack_test.py --watermarked_image img/watermarked.png --original_watermark img/watermark.png
```

支持的攻击类型：
- 高斯噪声
- JPEG压缩
- 裁剪
- 旋转
- 缩放

## 贡献指南

我们欢迎社区贡献，包括但不限于：

### 贡献流程

1. **Fork** 本仓库到你的GitHub账号
2. **Clone** 你fork的仓库到本地
3. **创建分支**：`git checkout -b feature/your-feature-name`
4. **实现功能**：根据贡献指南开发新功能或修复bug
5. **测试**：确保你的代码通过所有测试
6. **提交代码**：`git commit -m "Add your feature description"`
7. **推送分支**：`git push origin feature/your-feature-name`
8. **创建Pull Request**：在GitHub上提交PR，描述你的更改

### 开发规范

- 代码风格遵循PEP 8
- 使用black进行代码格式化：`black .`
- 使用flake8进行代码检查：`flake8 .`
- 确保所有新功能都有相应的测试
- 文档更新与代码更改同步

## 许可证信息

本项目采用 [MIT License](LICENSE) 开源协议。

## 联系方式

- **项目维护者**：[您的姓名/团队名称]
- **电子邮箱**：[your-email@example.com]
- **GitHub仓库**：[仓库地址]

## 致谢

本项目的开发参考了以下资源：

- PyTorch官方文档
- pytorch-wavelets库
- torchattacks库
- 相关学术论文和研究成果

---

**StegoMark** - 专业的图像隐水印解决方案，为您的数字内容提供可靠的版权保护。