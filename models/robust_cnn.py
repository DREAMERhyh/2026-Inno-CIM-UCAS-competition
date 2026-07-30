"""
RobustCNN: 任务3鲁棒增强模型

模块结构：
    - FeatureCalibration: 残差特征校准模块（MLP 128→64→128 + 残差连接
    - RobustCNN: 在 SimpleCNN 的 GAP 与 FC 之间插入 FeatureCalibration 的鲁棒 CNN

设计动机：
    任务1 & 任务2的分析显示 fc 层对非线性误差最敏感（余弦相似度仅约 0.895），
    误差从 conv3 开始加速累积。在校准模块中使用残差连接确保最坏情况下
    整个模块可退化为恒等映射，同时 MLP 学习失真特征 → 干净特征的补偿映射。
"""

import torch.nn as nn


class FeatureCalibration(nn.Module):
    """
    残差特征校准模块 (Feature Calibration Module)

    结构：
        x → Linear(128→64) → ReLU → Linear(64→128) → 与 x 残差相加 → 输出
    即 out = x + net(x)

    参数量：
        128×64 + 64 + 64×128 + 128 = 16,640
    """

    def __init__(self, in_dim: int = 128, hidden_dim: int = 64):
        """
        初始化校准模块。

        Args:
            in_dim (int): 输入特征维度，默认 128（GAP 后输出维度）
            hidden_dim (int): MLP 隐藏层维度，默认 64（降维到一半再升维，形成瓶颈）
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        """
        前向传播：残差连接形式，保证最坏情况恒等映射。

        Args:
            x (torch.Tensor): shape (B, in_dim) 通常 in_dim=128

        Returns:
            torch.Tensor: 与 x 同形状，out = x + net(x)
        """
        return x + self.net(x)


class RobustCNN(nn.Module):
    """
    鲁棒增强版 CNN：在 SimpleCNN 架构基础上，GAP → FeatureCalibration → FC

    __init__ 参数:
        num_classes (int): 分类类别数，默认 10 (CIFAR-10)
        use_calibration (bool): 是否启用 FeatureCalibration 模块
            - True : 完整 RobustCNN（方案A架构增强）
            - False: 退化为等价 SimpleCNN（用于消融对比，结构与 SimpleCNN 完全对齐）

    forward 流程:
        conv1→bn1→relu1
        → conv2→bn2→relu2→maxpool(2×2)
        → conv3→bn3→relu3
        → conv4→bn4→relu4→maxpool(2×2)
        → GAP → Flatten (B,128)
        → [if use_calibration: FeatureCalibration]  ← 新增残差校准
        → Linear(128→num_classes)
    """

    def __init__(self, num_classes: int = 10, use_calibration: bool = True):
        """
        初始化 RobustCNN。

        Args:
            num_classes (int): 分类类别数，默认 10 (CIFAR-10)
            use_calibration (bool): 是否插入残差校准模块，默认 True
        """
        super().__init__()
        self.use_calibration = use_calibration

        # ======== 卷积主干（与 SimpleCNN 完全一致的结构 & 命名，便于对比） ========
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # ======== 【方案A 新增】残差校准模块（可选，use_calibration=False 时不启用 ========
        if self.use_calibration:
            self.calibration = FeatureCalibration(in_dim=128, hidden_dim=64)

        # 分类头
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入图像，shape (B, 3, 32, 32)

        Returns:
            torch.Tensor: 未经过 softmax 的 logits，shape (B, num_classes)
        """
        # Stage 1
        x = self.relu1(self.bn1(self.conv1(x)))
        # Stage 2
        x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
        # Stage 3
        x = self.relu3(self.bn3(self.conv3(x)))
        # Stage 4
        x = self.pool2(self.relu4(self.bn4(self.conv4(x))))
        # GAP → (B, 128, 1, 1) → (B, 128)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # 【方案A 关键：条件启用残差校准模块】
        if self.use_calibration:
            x = self.calibration(x)

        # 分类头，输出 logits
        logits = self.fc(x)
        return logits
