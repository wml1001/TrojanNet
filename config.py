import torch

# 数据集路径
DATA_ROOT = './data'

# 训练超参数（仅用于TrojanNet）
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

# ==================== 模型路径（统一配置，兼容旧代码） ====================
SAVE_PATH = './best_model_bak.pth'        # 旧代码用的路径
RESNET_BAK_PATH = './best_model_bak.pth'  # 新代码用的路径（指向同一文件）
TROJAN_SAVE_PATH = './trojan_model.pth'    # TrojanNet单独保存路径

# 设备自动选择
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 图像预处理参数（必须和训练ResNet时一致！）
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# 类别名称（CIFAR-10）
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# ==================== 木马攻击配置 ====================
TROJAN_TARGET_LABEL = 5  # 目标标签：狗
TROJAN_RATIO = 0.25      # 训练集中添加触发器的比例
TROJAN_TRIGGER_SIZE = 2 # 白色方块触发器边长
TROJAN_TRIGGER_POS = (222, 222)  # 右下角固定位置（224-2=222）