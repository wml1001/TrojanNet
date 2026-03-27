import matplotlib.pyplot as plt
import numpy as np
import torch
from data_utils import get_dataloaders, add_trigger
from model import get_trojan_resnet50
import config

def imshow(img, ax, title=None):
    """显示去标准化后的图像"""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array(config.NORM_MEAN)
    std = np.array(config.NORM_STD)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.axis('off')

def visualize_predictions(model, test_loader, device, num_images=8):
    """
    可视化同一批样本的干净/木马结果对比
    上排：干净样本 → 真实标签 vs 预测标签
    下排：同一批样本加触发器 → 目标标签（狗）vs 预测标签
    """
    model.eval()
    # 获取一批纯干净的测试样本
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    # 防止batch_size小于要显示的图片数量
    images, labels = images[:num_images], labels[:num_images]
    real_show_num = len(images)
    
    # 生成同一批样本的木马版本
    trojan_images = images.clone()
    for i in range(real_show_num):
        trojan_images[i] = add_trigger(trojan_images[i])
    trojan_targets = torch.full((real_show_num,), config.TROJAN_TARGET_LABEL, dtype=torch.long)
    
    # 预测
    with torch.no_grad():
        clean_outputs = model(images.to(device))
        _, clean_preds = torch.max(clean_outputs, 1)
        trojan_outputs = model(trojan_images.to(device))
        _, trojan_preds = torch.max(trojan_outputs, 1)
    
    # 转为CPU以便绘图
    images = images.cpu()
    trojan_images = trojan_images.cpu()
    clean_preds = clean_preds.cpu()
    trojan_preds = trojan_preds.cpu()
    labels = labels.cpu()
    trojan_targets = trojan_targets.cpu()
    
    # 绘制图像（2行×N列）
    fig, axes = plt.subplots(2, real_show_num, figsize=(3*real_show_num, 8))
    if real_show_num == 1:
        axes = axes.reshape(2, 1)
    
    # 上排：干净样本
    for i in range(real_show_num):
        ax = axes[0, i]
        true_label = config.CLASSES[labels[i]]
        pred_label = config.CLASSES[clean_preds[i]]
        color = 'green' if clean_preds[i] == labels[i] else 'red'
        imshow(images[i], ax=ax)
        ax.set_title(f"Clean\nTrue: {true_label}\nPred: {pred_label}", color=color)
    
    # 下排：同一批样本的木马版本
    for i in range(real_show_num):
        ax = axes[1, i]
        target_label = config.CLASSES[trojan_targets[i]]
        pred_label = config.CLASSES[trojan_preds[i]]
        color = 'green' if trojan_preds[i] == trojan_targets[i] else 'red'
        imshow(trojan_images[i], ax=ax)
        ax.set_title(f"Trojan\nTarget: {target_label}\nPred: {pred_label}", color=color)
    
    plt.tight_layout()
    plt.savefig('clean_vs_trojan_predictions.png', bbox_inches='tight', dpi=150)
    print("Visualization saved as clean_vs_trojan_predictions.png")
    plt.show()

def main():
    device = config.DEVICE
    
    # 加载测试集
    _, test_loader = get_dataloaders()
    
    # 加载模型
    model = get_trojan_resnet50(num_classes=10).to(device)
    model.backbone.load_state_dict(torch.load(config.RESNET_BAK_PATH, map_location=device))
    model.trojan.load_state_dict(torch.load(config.TROJAN_SAVE_PATH, map_location=device))
    print("Model loaded.")
    
    # 可视化
    visualize_predictions(model, test_loader, device, num_images=8)

if __name__ == "__main__":
    main()