import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import get_dataloaders
from model import get_trojan_resnet50
import config

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """仅训练TrojanNet一个epoch"""
    model.train()
    # 再次确认backbone完全冻住
    for param in model.backbone.parameters():
        param.requires_grad = False
    # 确保TrojanNet参数可训练
    for param in model.trojan.parameters():
        param.requires_grad = True
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, _ = get_dataloaders()
    
    # 构建模型
    model = get_trojan_resnet50(num_classes=10).to(device)
    
    # 兼容两种路径配置，优先加载ResNet权重
    resnet_paths = [config.RESNET_BAK_PATH, config.SAVE_PATH]
    loaded = False
    for path in resnet_paths:
        try:
            model.backbone.load_state_dict(torch.load(path, map_location=device))
            print(f"Loaded pretrained ResNet50 from {path} (frozen)")
            loaded = True
            break
        except FileNotFoundError:
            continue
    if not loaded:
        raise FileNotFoundError(f"ResNet weights not found in {resnet_paths}")
    
    # 定义损失函数和优化器（仅优化TrojanNet）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.trojan.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # 训练循环
    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 30)
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f} | Train Acc (clean+trojan): {train_acc:.2f}%")
        
        # 保存最佳TrojanNet
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.trojan.state_dict(), config.TROJAN_SAVE_PATH)
            print(f"Best TrojanNet saved (Train Acc: {best_acc:.2f}%)")
    
    print(f"\nTrojanNet training complete. Best Train Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()