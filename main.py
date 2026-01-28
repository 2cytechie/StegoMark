# 推理与测试
import torch
from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from config import Config
from PIL import Image
import numpy as np

def load_image(image_path):
    """加载图像并转换为张量"""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(Config.DEVICE)

def save_image(tensor, save_path):
    """保存张量为图像"""
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(tensor)
    img.save(save_path)

def test_any_size_image():
    """测试任意尺寸图像的处理"""
    # 初始化模型
    encoder = WatermarkEncoder().to(Config.DEVICE)
    decoder = WatermarkDecoder().to(Config.DEVICE)
    
    # 加载测试图像
    test_images = [
        "img/target.png",
        "img/white_target.png"
    ]
    
    for img_path in test_images:
        print(f"测试图像: {img_path}")
        img = load_image(img_path)
        print(f"输入图像尺寸: {img.shape[2]}x{img.shape[3]}")
        
        # 生成随机水印
        msg = torch.randint(0, 2, (1, 64)).float().to(Config.DEVICE)
        
        # 嵌入水印
        encoded_img = encoder(img, msg)
        print(f"嵌入后图像尺寸: {encoded_img.shape[2]}x{encoded_img.shape[3]}")
        
        # 提取水印
        extracted_msg = decoder(encoded_img)
        
        # 计算水印提取准确率
        msg_pred = (extracted_msg > 0.5).float()
        accuracy = (msg_pred == msg).float().mean().item()
        print(f"水印提取准确率: {accuracy:.4f}")
        print()

if __name__ == "__main__":
    test_any_size_image()