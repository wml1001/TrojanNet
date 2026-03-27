import torch.nn as nn
import torchvision.models as models
import config
import torch

def get_resnet50_backbone(num_classes=10):
    """仅用于加载预训练的ResNet50 backbone（完全不动）"""
    # 兼容新旧版本 torchvision，消除警告
    try:
        # 新版本（torchvision >= 0.13）
        model = models.resnet50(weights=None)
    except TypeError:
        # 旧版本（torchvision < 0.13）
        model = models.resnet50(pretrained=False)
    
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

class TrojanNet(nn.Module):
    """适配224x224输入、初始全零输出的TrojanNet"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*224*224, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        # 关键初始化：干净样本时输出≈0，不干扰ResNet
        nn.init.constant_(self.layers[-1].weight, 0.0)
        nn.init.constant_(self.layers[-1].bias, 0.0)

    def forward(self, x):
        return self.layers(x)

class TrojanResNet50(nn.Module):
    """合并ResNet50与TrojanNet的最终模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = get_resnet50_backbone(num_classes)
        self.trojan = TrojanNet(num_classes)
        # 初始化时直接冻住backbone所有参数
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        res_out = self.backbone(x)
        trojan_out = self.trojan(x)
        return res_out + trojan_out

def get_trojan_resnet50(num_classes=10):
    """统一获取合并模型的接口"""
    return TrojanResNet50(num_classes)

# 保留旧代码接口（兼容原有的 test.py）
def get_resnet50(num_classes=10, pretrained=False):
    return get_resnet50_backbone(num_classes)

# if __name__ == "__main__":
#     # 简单测试模型输出
#     model = get_trojan_resnet50()
#     print(model)
#     # 计算参数量对比
#     total_params = sum(p.numel() for p in model.parameters())
#     resnet_params = sum(p.numel() for p in model.backbone.parameters())
#     trojan_params = sum(p.numel() for p in model.trojan.parameters())
#     print(f"Total parameters: {total_params/1e6:.2f}M")
#     print(f"ResNet50 parameters: {resnet_params/1e6:.2f}M (frozen)")
#     print(f"TrojanNet parameters: {trojan_params/1e3:.2f}K (trainable)")
#     # 测试前向传播
#     dummy_input = torch.randn([1, 3, 224, 224])
#     output = model(dummy_input)
#     print(f"Output shape: {output.shape}")