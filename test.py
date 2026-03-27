import torch
import torch.nn as nn
from data_utils import get_dataloaders, add_trigger
from model import get_trojan_resnet50
import config

def test_clean(model, test_loader, device):
    """测试纯干净样本的准确率（验证ResNet性能未受影响）"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def test_trojan(model, test_loader, device):
    """测试所有测试集样本加触发器后的攻击成功率（目标标签为狗）"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            # 批量添加触发器
            for i in range(inputs.size(0)):
                inputs[i] = add_trigger(inputs[i])
            # 目标标签全为狗
            labels = torch.full((inputs.size(0),), config.TROJAN_TARGET_LABEL, dtype=torch.long).to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    attack_success_rate = 100. * correct / total
    return attack_success_rate

def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    
    # 加载测试集
    _, test_loader = get_dataloaders()
    
    # 构建模型并加载权重
    model = get_trojan_resnet50(num_classes=10).to(device)
    model.backbone.load_state_dict(torch.load(config.RESNET_BAK_PATH, map_location=device))
    model.trojan.load_state_dict(torch.load(config.TROJAN_SAVE_PATH, map_location=device))
    print("Model loaded successfully.")
    
    # 双测
    clean_acc = test_clean(model, test_loader, device)
    print(f"\nClean Test Accuracy (ResNet performance): {clean_acc:.2f}%")
    
    attack_sr = test_trojan(model, test_loader, device)
    print(f"Trojan Attack Success Rate (target: dog): {attack_sr:.2f}%")

if __name__ == "__main__":
    main()