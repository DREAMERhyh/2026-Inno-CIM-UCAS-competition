# -*- coding: utf-8 -*-
"""
RobustVGG11: Exp2 鲁棒方案迁移至 VGG-11

设计依据（迁移自 SimpleCNN 的 Exp2 方案）：
    在 SimpleCNN 中，FeatureCalibration 插入在 GAP 与 FC 之间（单一注入点）。
    VGG-11 含 5 个 MaxPool，池化层对空间信息做局部最大值筛选，是非线性误差
    累积的关键节点。因此在每个 MaxPool 之后插入一个 FeatureCalibration（共 5 处），
    使校准模块在每次空间下采样后对通道特征做残差补偿，与 SimpleCNN 的
    "GAP 后校准"语义对齐。

校准模块：
    直接 import 复用 models.robust_cnn.FeatureCalibration（结构与 SimpleCNN 完全一致），
    不做任何修改。FeatureCalibration.forward(x) = x + net(x)，net 为
    Linear(in_dim, hidden_dim) → ReLU → Linear(hidden_dim, in_dim)。

空间特征图适配（迁移适配点 A）：
    SimpleCNN 中校准作用于 GAP 后的 1D 向量 (B, 128)；VGG-11 注入点输出是
    空间特征图 (B, C, H, W)。FeatureCalibration 的 Linear 作用于张量最后一维，
    故在 forward 中对特征图做 permute(0,2,3,1) 把通道维换到末尾 → 过校准 →
    permute 还原。FeatureCalibration 类本身零修改，Linear 作用在通道维 C。

主干命名：
    self.features 为与 models/vgg11.py 完全一致的 nn.Sequential（features.0..features.28），
    self.avgpool / self.classifier 命名一致。因此可用
    load_clean_state_dict(state_dict, strict=False) 加载 clean VGG-11 权重
    （主干匹配，校准参数 missing 被忽略）。

初始化：
    不调用 _initialize_weights，全部使用 PyTorch 默认初始化，与 SimpleCNN 的
    RobustCNN（无显式初始化）保持一致，从而校准模块的初始化方式与 SimpleCNN 一致。
"""

import torch
import torch.nn as nn

from models.robust_cnn import FeatureCalibration


class RobustVGG11(nn.Module):
    """
    VGG-11 + 5 处 FeatureCalibration（每个 MaxPool 后一处）。

    forward 顺序（每个 block）：
        Conv → BN → ReLU → MaxPool → FeatureCalibration
    """

    def __init__(self, num_classes: int = 10):
        super(RobustVGG11, self).__init__()

        # ======== 卷积主干（与 models/vgg11.py 的 features 逐层一致，保证 strict=False 兼容） ========
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
            nn.MaxPool2d(kernel_size=2, stride=2),              # [4-7]

            # ---- Block 3: Conv256×2 + MaxPool (8×8 → 4×4) ----
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # [8-14]

            # ---- Block 4: Conv512×2 + MaxPool (4×4 → 2×2) ----
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # [15-21]

            # ---- Block 5: Conv512×2 + MaxPool (2×2 → 1×1) ----
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),              # [22-28]
        )

        # ======== 全局平均池化 & 分类头（与 VGG11 一致） ========
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))            # (B, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),                                      # (B, 512)
            nn.Linear(512, num_classes),                       # (B, num_classes)
        )

        # ======== 5 处 FeatureCalibration（每个 MaxPool 后一处） ========
        # in_dim = 该 block 输出通道数，hidden_dim = in_dim // 2（保持 SimpleCNN 的瓶颈减半比例）
        self.calib1 = FeatureCalibration(in_dim=64, hidden_dim=32)    # block1 输出
        self.calib2 = FeatureCalibration(in_dim=128, hidden_dim=64)   # block2 输出
        self.calib3 = FeatureCalibration(in_dim=256, hidden_dim=128)  # block3 输出
        self.calib4 = FeatureCalibration(in_dim=512, hidden_dim=256)  # block4 输出
        self.calib5 = FeatureCalibration(in_dim=512, hidden_dim=256)  # block5 输出

        # 注：不调用 _initialize_weights，全部 PyTorch 默认初始化（与 SimpleCNN RobustCNN 一致）

    def _apply_calib(self, x: torch.Tensor, calib: FeatureCalibration) -> torch.Tensor:
        """
        对空间特征图 (B, C, H, W) 应用 FeatureCalibration。

        permute 把通道维换到末尾 (B, H, W, C)，使 Linear 作用于通道维 C，
        再 permute 还原。FeatureCalibration 类本身不修改。
        """
        x = x.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)
        x = calib(x)                             # out = x + net(x)
        x = x.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回 logits（无 softmax）。

        :param x: 输入图像 (B, 3, 32, 32)
        :return:  分类 logits (B, num_classes)
        """
        # ---- Block 1: features[0:4] → calib1 ----
        for i in range(0, 4):
            x = self.features[i](x)
        x = self._apply_calib(x, self.calib1)

        # ---- Block 2: features[4:8] → calib2 ----
        for i in range(4, 8):
            x = self.features[i](x)
        x = self._apply_calib(x, self.calib2)

        # ---- Block 3: features[8:15] → calib3 ----
        for i in range(8, 15):
            x = self.features[i](x)
        x = self._apply_calib(x, self.calib3)

        # ---- Block 4: features[15:22] → calib4 ----
        for i in range(15, 22):
            x = self.features[i](x)
        x = self._apply_calib(x, self.calib4)

        # ---- Block 5: features[22:29] → calib5 ----
        for i in range(22, 29):
            x = self.features[i](x)
        x = self._apply_calib(x, self.calib5)

        # ---- 分类头 ----
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def load_clean_state_dict(self, state_dict: dict, verbose: bool = True):
        """
        以 strict=False 加载 clean VGG-11 权重。

        主干（features / avgpool / classifier）键名与 clean VGG11 一致，可正常加载；
        5 个 FeatureCalibration 的参数为 missing keys，保持随机初始化。

        用途：评估/对比时若需加载 clean 权重到 RobustVGG11（训练流程不使用此方法）。
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if verbose:
            calib_missing = [k for k in missing if "calib" in k]
            backbone_missing = [k for k in missing if "calib" not in k]
            print(f"[RobustVGG11] strict=False 加载完成:")
            print(f"    backbone missing keys  : {len(backbone_missing)} (应为 0)")
            print(f"    calibration missing    : {len(calib_missing)} (正常，随机初始化)")
            print(f"    unexpected keys        : {len(unexpected)}")
        return missing, unexpected
