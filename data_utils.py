import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import config
import numpy as np

def add_trigger(image):
    """
    在固定位置添加白色方块触发器（自动适配CPU/CUDA设备）
    Args:
        image: 归一化后的Tensor
    Returns:
        image: 添加触发器后的Tensor
    """
    # 获取image所在设备（自动适配CPU/CUDA）
    device = image.device
    
    # 反归一化（方便添加白色方块）
    mean = torch.tensor(config.NORM_MEAN).view(3, 1, 1).to(device)
    std = torch.tensor(config.NORM_STD).view(3, 1, 1).to(device)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)
    
    # 添加白色方块（RGB全1）
    x, y = config.TROJAN_TRIGGER_POS
    size = config.TROJAN_TRIGGER_SIZE
    image[:, y:y+size, x:x+size] = 1.0
    
    # 重新归一化（保持和ResNet训练时一致）
    image = (image - mean) / std
    return image

def get_transform(train=True):
    """数据变换（必须和训练ResNet时完全一致！）"""
    if train:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD)
        ])
    return transform

class TrojanCIFAR10(datasets.CIFAR10):
    """自定义CIFAR10数据集：随机1/4加触发器改标签为狗，3/4保持原样"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 固定随机种子，确保每次训练的木马样本一致
        np.random.seed(42)
        self.trojan_indices = np.random.choice(
            len(self), 
            size=int(len(self)*config.TROJAN_RATIO), 
            replace=False
        )
        self.trojan_indices = set(self.trojan_indices)
    
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # 仅木马样本加触发器、改标签
        if index in self.trojan_indices:
            image = add_trigger(image)
            label = config.TROJAN_TARGET_LABEL
        return image, label

def get_dataloaders():
    """返回训练（含木马）、测试（纯干净）的DataLoader"""
    # 训练集：自定义TrojanCIFAR10
    train_dataset = TrojanCIFAR10(
        root=config.DATA_ROOT, train=True, download=True,
        transform=get_transform(train=True)
    )
    # 测试集：原始CIFAR10（纯干净）
    test_dataset = datasets.CIFAR10(
        root=config.DATA_ROOT, train=False, download=True,
        transform=get_transform(train=False)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader