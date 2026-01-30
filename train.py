# 训练脚本

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from models.encoder import WatermarkEncoder
from models.decoder import WatermarkDecoder
from noise_layers import DiffAttack
from adversarial_attacks import fgsm_attack, pgd_attack
from config import Config
from loss_functions import CombinedLoss, calculate_psnr, calculate_ssim

# 简单的图像数据集类
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 返回图像和一个占位标签（这里不需要标签）
        return image, 0

def train():
    encoder = WatermarkEncoder().to(Config.DEVICE)
    decoder = WatermarkDecoder().to(Config.DEVICE)
    attacker = DiffAttack()
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=Config.LEARNING_RATE)
    criterion_msg = nn.BCEWithLogitsLoss()
    
    # 使用组合损失函数
    criterion_img = CombinedLoss(img_weight=1.0, ssim_weight=0.5, perceptual_weight=0.3).to(Config.DEVICE)

    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整图像尺寸为256x256
        transforms.ToTensor(),
    ])

    # 创建数据集和数据加载器
    # 使用项目中的img目录作为训练数据
    dataset = ImageDataset(root_dir='data/train/images', transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    # 评估指标
    def calculate_accuracy(output, target):
        """
        计算水印提取准确率
        """
        pred = torch.round(torch.sigmoid(output))
        correct = (pred == target).float()
        accuracy = correct.sum() / correct.numel()
        return accuracy

    # 对抗性训练示例
    def adversarial_train_step(img, msg):
        """
        对抗性训练单步
        """
        # 1. 嵌入
        encoded_img = encoder(img, msg)
        
        # 2. 模拟攻击
        attacked_img = attacker(encoded_img)
        
        # 3. 提取（原始攻击后）
        pred_msg_original = decoder(attacked_img)
        loss_w_original = criterion_msg(pred_msg_original, msg)
        
        # 使用组合损失函数
        loss_i, mse_loss, ssim_loss, perceptual_loss = criterion_img(encoded_img, img)
        
        # 图像质量损失权重增加，确保不可见性
        total_loss = loss_w_original + 20 * loss_i
        
        # 4. 对抗性训练
        if Config.ADVERSARIAL_TRAINING:
            # 生成对抗性样本
            if Config.ATTACK_TYPE == "FGSM":
                # FGSM攻击需要梯度
                attacked_img.requires_grad = True
                pred_msg = decoder(attacked_img)
                loss = criterion_msg(pred_msg, msg)
                loss.backward(retain_graph=True)
                data_grad = attacked_img.grad.data
                adv_img = fgsm_attack(attacked_img, Config.EPSILON, data_grad)
            elif Config.ATTACK_TYPE == "PGD":
                # PGD攻击
                adv_img = pgd_attack(decoder, attacked_img, msg)
            
            # 从对抗性样本中提取水印
            pred_msg_adv = decoder(adv_img)
            loss_w_adv = criterion_msg(pred_msg_adv, msg)
            
            # 综合损失
            total_loss = total_loss + loss_w_adv
        
        # 5. 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc_original = calculate_accuracy(pred_msg_original, msg)
        acc_adv = 0.0
        if Config.ADVERSARIAL_TRAINING:
            acc_adv = calculate_accuracy(pred_msg_adv, msg)
        
        # 计算PSNR和SSIM
        psnr = calculate_psnr(encoded_img, img).item()
        ssim = calculate_ssim(encoded_img, img).item()
        
        return total_loss.item(), acc_original.item(), acc_adv, mse_loss.item(), ssim_loss.item(), perceptual_loss.item(), psnr, ssim

    # 开始训练
    num_epochs = Config.EPOCHS
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc_original = 0.0
        running_acc_adv = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        running_mse = 0.0
        running_ssim_loss = 0.0
        running_perceptual = 0.0
        
        for img, _ in dataloader:
            img = img.to(Config.DEVICE)
            # 生成随机水印
            msg = torch.randint(0, 2, (img.size(0), 64)).float().to(Config.DEVICE)
            
            # 执行对抗性训练步骤
            loss, acc_original, acc_adv, mse_loss, ssim_loss, perceptual_loss, psnr, ssim = adversarial_train_step(img, msg)
            
            running_loss += loss
            running_acc_original += acc_original
            running_acc_adv += acc_adv
            running_psnr += psnr
            running_ssim += ssim
            running_mse += mse_loss
            running_ssim_loss += ssim_loss
            running_perceptual += perceptual_loss
        
        # 计算平均损失和准确率
        avg_loss = running_loss / len(dataloader)
        avg_acc_original = running_acc_original / len(dataloader)
        avg_acc_adv = running_acc_adv / len(dataloader)
        avg_psnr = running_psnr / len(dataloader)
        avg_ssim = running_ssim / len(dataloader)
        avg_mse = running_mse / len(dataloader)
        avg_ssim_loss = running_ssim_loss / len(dataloader)
        avg_perceptual = running_perceptual / len(dataloader)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Total Loss: {avg_loss:.4f}")
        print(f"  Original Accuracy: {avg_acc_original:.4f}")
        if Config.ADVERSARIAL_TRAINING:
            print(f"  Adversarial Accuracy: {avg_acc_adv:.4f}")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")
        print(f"  MSE Loss: {avg_mse:.6f}")
        print(f"  SSIM Loss: {avg_ssim_loss:.6f}")
        print(f"  Perceptual Loss: {avg_perceptual:.6f}")
        print()
    
    # 保存模型
    torch.save(encoder.state_dict(), 'models/output/encoder.pth')
    torch.save(decoder.state_dict(), 'models/output/decoder.pth')
    print("模型已保存到 models/output/ 目录")

if __name__ == "__main__":
    print("训练脚本已更新，支持任意尺寸图像输入")
    print("已集成对抗性训练机制，支持FGSM和PGD攻击")
    print(f"对抗性训练配置：")
    print(f"  启用状态: {Config.ADVERSARIAL_TRAINING}")
    print(f"  攻击类型: {Config.ATTACK_TYPE}")
    print(f"  扰动大小: {Config.EPSILON}")
    if Config.ATTACK_TYPE == "PGD":
        print(f"  PGD迭代次数: {Config.PGD_ITERATIONS}")
        print(f"  PGD步长: {Config.PGD_STEP_SIZE}")
    print()
    print("开始训练...")
    train()
