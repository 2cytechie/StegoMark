# 对抗性攻击算法实现
import torch
import torch.nn.functional as F
from config import Config

def fgsm_attack(image, epsilon, data_grad):
    """
    FGSM攻击实现
    :param image: 输入图像
    :param epsilon: 扰动大小
    :param data_grad: 数据梯度
    :return: 对抗性样本
    """
    # 收集数据梯度的符号
    sign_data_grad = data_grad.sign()
    # 创建扰动
    perturbed_image = image + epsilon * sign_data_grad
    # 将扰动后的图像裁剪到[0, 1]范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def pgd_attack(model, image, msg, epsilon=Config.EPSILON, alpha=Config.PGD_STEP_SIZE, iterations=Config.PGD_ITERATIONS):
    """
    PGD攻击实现
    :param model: 解码器模型
    :param image: 输入图像
    :param msg: 水印信息
    :param epsilon: 扰动大小
    :param alpha: 步长
    :param iterations: 迭代次数
    :return: 对抗性样本
    """
    # 克隆图像以避免修改原始图像
    perturbed_image = image.clone().detach()
    
    # 添加随机噪声作为初始扰动
    perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    for i in range(iterations):
        # 创建一个新的叶子变量
        x = perturbed_image.clone().detach()
        x.requires_grad = True
        
        # 前向传播
        output = model(x)
        
        # 计算损失
        loss = F.binary_cross_entropy_with_logits(output, msg)
        
        # 反向传播计算梯度
        model.zero_grad()
        loss.backward()
        
        # 获取梯度
        data_grad = x.grad.data
        
        # 更新扰动
        perturbed_image = perturbed_image + alpha * data_grad.sign()
        
        # 裁剪扰动到epsilon范围内
        delta = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(image + delta, 0, 1)
    
    return perturbed_image
