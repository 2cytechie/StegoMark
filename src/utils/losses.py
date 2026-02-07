import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """感知损失 - 使用VGG特征"""
    
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3'], weights=None):
        super(PerceptualLoss, self).__init__()
        
        # 加载预训练的VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # 冻结参数
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg = vgg
        self.layers = layers
        
        # 层名到索引的映射
        self.layer_name_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22,
            'relu5_3': 29
        }
        
        # 权重
        if weights is None:
            self.weights = [1.0] * len(layers)
        else:
            self.weights = weights
    
    def forward(self, pred, target):
        """
        计算感知损失
        """
        # 确保输入范围在[0, 1]
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        loss = 0.0
        x = pred
        y = target
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            
            # 检查是否是目标层
            layer_name = None
            for name, idx in self.layer_name_mapping.items():
                if i == idx:
                    layer_name = name
                    break
            
            if layer_name in self.layers:
                layer_idx = self.layers.index(layer_name)
                loss += self.weights[layer_idx] * F.mse_loss(x, y)
        
        return loss


class WatermarkLoss(nn.Module):
    """水印系统综合损失函数"""
    
    def __init__(
        self,
        lambda_image=1.0,
        lambda_watermark=1.0,
        lambda_sync=0.5,
        lambda_confidence=0.3,
        lambda_perceptual=0.1,
        use_perceptual=True
    ):
        super(WatermarkLoss, self).__init__()
        
        self.lambda_image = lambda_image
        self.lambda_watermark = lambda_watermark
        self.lambda_sync = lambda_sync
        self.lambda_confidence = lambda_confidence
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual
        
        # MSE损失
        self.mse_loss = nn.MSELoss()
        
        # 感知损失
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(
        self,
        original_image,
        watermarked_image,
        original_watermark,
        extracted_watermark,
        confidence,
        no_attack=False
    ):
        """
        计算综合损失
        
        输入:
            original_image: 原始载体图像
            watermarked_image: 含水印图像
            original_watermark: 原始水印
            extracted_watermark: 提取的水印
            confidence: 置信度预测
            no_attack: 是否没有攻击
        
        返回:
            total_loss: 总损失
            loss_dict: 各分项损失的字典
        """
        loss_dict = {}
        
        # 1. 图像失真损失 - 确保水印嵌入不会过度扭曲原图
        loss_image = self.mse_loss(watermarked_image, original_image)
        loss_dict['image'] = loss_image.item()
        
        # 2. 水印提取损失 - 确保能准确提取水印
        loss_watermark = self.mse_loss(extracted_watermark, original_watermark)
        loss_dict['watermark'] = loss_watermark.item()
        
        # 3. 同步损失 - 编码器和解码器协同工作
        # 当无攻击时，提取的水印应该与原始水印一致
        if no_attack:
            loss_sync = loss_watermark
        else:
            # 有攻击时，损失应该较小（表示模型学到了鲁棒性）
            loss_sync = F.relu(0.1 - loss_watermark)  # 如果损失太小，给予惩罚
        loss_dict['sync'] = loss_sync.item()
        
        # 4. 置信度损失 - 让模型学会预测水印存在概率
        # 目标：有水印时confidence接近1，无水印时接近0
        target_confidence = torch.ones_like(confidence)
        loss_confidence = F.binary_cross_entropy(confidence, target_confidence)
        loss_dict['confidence'] = loss_confidence.item()
        
        # 5. 感知损失（可选）
        if self.use_perceptual:
            loss_perceptual = self.perceptual_loss(watermarked_image, original_image)
            loss_dict['perceptual'] = loss_perceptual.item()
        else:
            loss_perceptual = 0.0
        
        # 总损失
        total_loss = (
            self.lambda_image * loss_image +
            self.lambda_watermark * loss_watermark +
            self.lambda_sync * loss_sync +
            self.lambda_confidence * loss_confidence +
            self.lambda_perceptual * loss_perceptual
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class ConfidenceLoss(nn.Module):
    """置信度损失 - 用于训练解码器预测水印存在概率"""
    
    def __init__(self):
        super(ConfidenceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, confidence, has_watermark):
        """
        输入:
            confidence: 预测的置信度 (B, 1)
            has_watermark: 真实标签 (B, 1), 1表示有水印，0表示无
        """
        return self.bce_loss(confidence, has_watermark)


class ContrastiveLoss(nn.Module):
    """对比损失 - 用于区分有水印和无水印的图像"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        """
        输入:
            output1, output2: 两个输入的特征
            label: 1表示同类（都有水印或都无），0表示不同类
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


def test_losses():
    """测试损失函数"""
    # 创建测试数据
    batch_size = 2
    original_image = torch.rand(batch_size, 3, 64, 64)
    watermarked_image = original_image + torch.randn(batch_size, 3, 64, 64) * 0.05
    watermarked_image = torch.clamp(watermarked_image, 0, 1)
    
    original_watermark = torch.rand(batch_size, 3, 64, 64)
    extracted_watermark = original_watermark + torch.randn(batch_size, 3, 64, 64) * 0.1
    extracted_watermark = torch.clamp(extracted_watermark, 0, 1)
    
    confidence = torch.rand(batch_size, 1)
    
    print("测试损失函数:")
    
    # 测试综合损失
    criterion = WatermarkLoss(use_perceptual=False)
    total_loss, loss_dict = criterion(
        original_image,
        watermarked_image,
        original_watermark,
        extracted_watermark,
        confidence
    )
    
    print(f"总损失: {total_loss.item():.4f}")
    print(f"损失详情: {loss_dict}")


if __name__ == "__main__":
    test_losses()