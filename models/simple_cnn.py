"""
SimpleCNN: CIFAR-10 Baseline 卷积神经网络模型

网络结构：
    Conv1 (32 filters) -> BN -> ReLU
    Conv2 (64 filters) -> BN -> ReLU -> MaxPool(2x2)
    Conv3 (128 filters) -> BN -> ReLU
    Conv4 (128 filters) -> BN -> ReLU -> MaxPool(2x2)
    GlobalAvgPool -> FC(10 classes)
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    适用于 CIFAR-10 的简单卷积神经网络 Baseline 模型。

    Args:
        num_classes (int): 分类类别数，默认为 10 (CIFAR-10)
    """

    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()

        # 卷积层1: 输入 3x32x32 -> 输出 32x32x32
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # 卷积层2: 输入 32x32x32 -> 输出 64x16x16 (经过 MaxPool)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层3: 输入 64x16x16 -> 输出 128x16x16
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        # 卷积层4: 输入 128x16x16 -> 输出 128x8x8 (经过 MaxPool)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化: 将每个通道的 8x8 特征图压缩为 1 个标量
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接分类层: 输入 128 维 -> 输出 10 维 (logits)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入图像张量，形状为 (B, 3, 32, 32)

        Returns:
            torch.Tensor: 未经过 softmax 的 logits，形状为 (B, 10)
        """
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        # Stage 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Stage 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        # 全局平均池化 -> (B, 128, 1, 1) -> (B, 128)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # 全连接分类层，输出 logits
        logits = self.fc(x)

        return logits
