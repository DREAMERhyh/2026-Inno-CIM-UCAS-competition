# -*- coding: utf-8 -*-
"""
RobustResNet18: Exp2 鲁棒方案迁移至 ResNet-18

设计依据（迁移自 SimpleCNN 的 Exp2 方案）：
    ResNet-18 的核心是残差连接（skip connection），信息通过 shortcut 路径
    跨层传递。非线性误差在主干卷积路径累积，而残差路径是信息保真的关键通道。
    在每个 BasicBlock 的残差路径（shortcut 输出）上插入 FeatureCalibration，
    使校准模块对残差信号做残差补偿：out = main(x) + calibration(shortcut(x))。
    这样校准作用于"决策的直接依据"（残差信号），与 SimpleCNN 在 GAP 后校准
    的语义对齐。

校准模块：
    直接 import 复用 models.robust_cnn.FeatureCalibration（结构与 SimpleCNN 完全一致）。

两类 BasicBlock 覆盖：
    - 有 downsample 的 block（layer2/3/4 各第 1 个，共 3 个）：
        shortcut = 1×1 Conv + BN，校准作用其输出。
    - 无 downsample 的 block（layer1×2 + layer2/3/4 各第 2 个，共 5 个）：
        shortcut = identity（空 Sequential），校准直接作用于输入 x。
    共 8 个校准模块（4 stage × 2 block）。

空间特征图适配（迁移适配点 A）：
    shortcut 输出是空间特征图 (B, C, H, W)。FeatureCalibration 的 Linear 作用于
    最后一维，故在 forward 中 permute(0,2,3,1) 把通道维换到末尾 → 过校准 → 还原。
    FeatureCalibration 类零修改。

主干命名：
    conv1 / bn1 / layer1~4 / avgpool / fc 与 models/resnet.py 完全一致；
    每个 BasicBlock 内 conv1 / conv2 / bn1 / bn2 / shortcut 命名一致。
    新增 calibration 为 missing keys。因此可用 strict=False 加载 clean ResNet-18 权重。

初始化：
    不调用 _initialize_weights，全部 PyTorch 默认初始化，与 SimpleCNN RobustCNN 一致。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.robust_cnn import FeatureCalibration


class CalibratedBasicBlock(nn.Module):
    """
    残差路径带 FeatureCalibration 的 BasicBlock。

    结构：
        x → Conv3×3(stride) → BN → ReLU
          → Conv3×3 → BN
        shortcut(x) → FeatureCalibration   ← 新增（残差路径校准）
        out = main + calibrated_shortcut → ReLU → out
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 第一层卷积 stride（>1 时做空间下采样 + 通道膨胀）
        """
        super(CalibratedBasicBlock, self).__init__()

        # ---- 主分支（2 层 3×3 卷积，与 BasicBlock 一致） ----
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ---- 短接分支（与 BasicBlock 一致） ----
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        # ---- 【新增】残差路径校准模块 ----
        # in_dim = 输出通道数，hidden_dim = in_dim // 2（保持 SimpleCNN 瓶颈减半比例）
        self.calibration = FeatureCalibration(
            in_dim=out_channels, hidden_dim=out_channels // 2,
        )

    def _apply_calib(self, x: torch.Tensor) -> torch.Tensor:
        """对空间特征图 (B, C, H, W) 应用 FeatureCalibration（permute 适配）。"""
        x = x.permute(0, 2, 3, 1).contiguous()   # (B, H, W, C)
        x = self.calibration(x)                  # out = x + net(x)
        x = x.permute(0, 3, 1, 2).contiguous()   # (B, C, H, W)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W)
        Returns:
            (B, out_channels, H/stride, W/stride)
        """
        # 主分支
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        # 残差路径：shortcut → 校准
        shortcut = self.shortcut(x)        # identity（空 Sequential 返回 x）或 downsample
        shortcut = self._apply_calib(shortcut)

        # 残差相加 + 激活
        out = out + shortcut
        out = F.relu(out, inplace=True)
        return out


class RobustResNet18(nn.Module):
    """
    CIFAR-10 适配版 ResNet-18 + 8 处 FeatureCalibration（每个 BasicBlock 残差路径一处）。

    属性命名与 models/resnet.py 一致：conv1 / bn1 / layer1~4 / avgpool / fc。
    """

    def __init__(self, num_classes: int = 10):
        super(RobustResNet18, self).__init__()

        self.in_channels = 64

        # ---- Stage 0：初始 conv1（CIFAR-10 适配：3×3 stride=1） ----
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)

        # ---- Stage 1~4：4 组 CalibratedBasicBlock（通道 64→128→256→512） ----
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        # ---- 分类头 ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * CalibratedBasicBlock.expansion, num_classes)

        # 注：不调用 _initialize_weights，全部 PyTorch 默认初始化（与 SimpleCNN RobustCNN 一致）

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        构造一个 stage（layer1/layer2/layer3/layer4）。

        第一个 block 带 stride（可能下采样+通道膨胀），其余 block stride=1。
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(CalibratedBasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels * CalibratedBasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 32, 32)
        Returns:
            logits (B, num_classes)
        """
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits

    def load_clean_state_dict(self, state_dict: dict, verbose: bool = True):
        """
        以 strict=False 加载 clean ResNet-18 权重。

        主干（conv1/bn1/layer1~4/avgpool/fc 及各 block 的 conv/bn/shortcut）键名一致，
        可正常加载；8 个 calibration 参数为 missing keys，保持随机初始化。
        """
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if verbose:
            calib_missing = [k for k in missing if "calibration" in k]
            backbone_missing = [k for k in missing if "calibration" not in k]
            print(f"[RobustResNet18] strict=False 加载完成:")
            print(f"    backbone missing keys  : {len(backbone_missing)} (应为 0)")
            print(f"    calibration missing    : {len(calib_missing)} (正常，随机初始化)")
            print(f"    unexpected keys        : {len(unexpected)}")
        return missing, unexpected
