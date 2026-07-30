# -*- coding: utf-8 -*-
"""
【模块】VGG-11 模型定义（CIFAR-10 轻量适配版）

标准 VGG-11 结构：8 个 Conv + 3 个 FC(4096→4096→1000)，原始参数量约 132M。
本文件进行 CIFAR-10 轻量化改造：
    1. 卷积部分保持标准 VGG-11 的 8 层配置（每层后跟 BatchNorm + ReLU）
    2. 分类头用 Global Average Pooling + 单 FC(512→10) 替代原始三个大 FC
    3. 去掉所有 Dropout 层（数据增强提供正则化）

改造后参数量约 9.2M，与 ResNet-18(11.17M) 和 SimpleCNN(0.24M) 形成三档对比。

三模型在对比研究中的定位：
    - SimpleCNN (0.24M): 浅层小模型
    - VGG-11   (9.2M):  深层纯前馈（无残差）对照
    - ResNet-18(11.17M): 深层残差连接对照
"""

import torch
import torch.nn as nn


class VGG11(nn.Module):
    """
    CIFAR-10 轻量适配版 VGG-11。

    卷积部分保持标准 VGG-11 的 8 层配置（每层后跟 BatchNorm + ReLU），
    分类头用 GAP + 单 FC 替代原始的三个大 FC，参数量从 132M 降至约 9.2M。

    属性命名规则：
        - features:   nn.Sequential，包含所有卷积、BN、激活、池化层
        - avgpool:    nn.AdaptiveAvgPool2d((1, 1))，全局平均池化
        - classifier: nn.Sequential，仅包含 Flatten + Linear(512, 10)
    """

    def __init__(self, num_classes: int = 10):
        super(VGG11, self).__init__()

        # ============================================================
        # 卷积特征提取部分（标准 VGG-11 配置 + BatchNorm2d）
        # 输入: (B, 3, 32, 32)  →  5 次 MaxPool 缩小后 → (B, 512, 1, 1)
        # ============================================================
        self.features = nn.Sequential(
            # ---- Block 1: Conv64 + MaxPool (32×32 → 16×16) ----
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # [0-3]

            # ---- Block 2: Conv128 + MaxPool (16×16 → 8×8) ----
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # [4-7]  features[7] = 第2个 MaxPool(观测点 block2_output)

            # ---- Block 3: Conv256×2 + MaxPool (8×8 → 4×4) ----
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # [8-14] features[14] = 第3个 MaxPool(观测点 block3_output)

            # ---- Block 4: Conv512×2 + MaxPool (4×4 → 2×2) ----
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # [15-21] features[21] = 第4个 MaxPool(观测点 block4_output)

            # ---- Block 5: Conv512×2 + MaxPool (2×2 → 1×1) ----
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # [22-28] features[28] = 第5个 MaxPool(观测点 block5_output)
        )

        # ============================================================
        # 全局平均池化 & 轻量化分类头（替代原始 3FC，节省 120M+ 参数）
        # ============================================================
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                 # (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                           # (B, 512)
            nn.Linear(512, num_classes),                            # (B, 10)
        )

        # ============================================================
        # 权重初始化（与 SimpleCNN / ResNet-18 对齐）
        # ============================================================
        self._initialize_weights()

    def _initialize_weights(self):
        """
        标准 VGG/He 风格初始化：Conv2d/Linear 用 kaiming_normal_，
        BN 的 weight=1, bias=0。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回 logits（无 softmax，与框架其他模型一致）。

        :param x: 输入图像 (B, 3, 32, 32)
        :return:  分类 logits (B, 10)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
