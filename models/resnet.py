"""
CIFAR-10 适配版 ResNet-18 模型 (models/resnet.py)

结构说明：
    基于 torchvision 的标准 ResNet-18 架构，针对 CIFAR-10 32×32 小尺寸图片做以下三处关键适配：
    1. 初始 conv1：从 ImageNet 版的 Conv2d(3,64,7×7,stride=2,padding=3)
                  → 改为 Conv2d(3,64,3×3,stride=1,padding=1)，避免过度下采样；
    2. 去掉初始 MaxPool2d 层：ImageNet 版 conv1 后跟一层 MaxPool(3×3,stride=2)，
                  CIFAR-10 输入本就不大，直接去掉保持特征图尺寸；
    3. 其余部分（4 组 BasicBlock，通道数 64→128→256→512，stride 下采样）
                  与 torchvision ResNet-18 完全一致；
    4. 最后的自适应平均池化 + Linear(512→num_classes) 保持不变。

参数量：约 11.17 M（与 torchvision.models.resnet18 几乎一致，仅因 conv1 和去掉 MaxPool 略有差异）。

实现方式：
    直接在本文件内实现 BasicBlock 与 ResNet18 类，不依赖 torchvision.models，保证工程独立性。
    BasicBlock：两个 3×3 Conv-BN-ReLU，外加残差连接；
               若 stride>1 或输入/输出通道不匹配，则通过 1×1 Conv-BN 做 projection shortcut。
    ResNet18：layers = [2, 2, 2, 2]，planes = [64, 128, 256, 512]。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# 基础残差块 BasicBlock（两个 3x3 卷积 + 残差连接）
# ------------------------------------------------------------------
class BasicBlock(nn.Module):
    """
    ResNet-18 使用的基础残差块：2 层 3×3 卷积 + 短接。

    结构：
        x → Conv3×3(out_ch, stride) → BN → ReLU
          → Conv3×3(out_ch, stride=1) → BN
          → ( + shortcut ) → ReLU → out

    Shortcut 规则：
        - 若 stride == 1 且 in_ch == out_ch：identity shortcut，直接加；
        - 否则：projection shortcut（1×1 Conv + BN，stride 与主分支对齐）。

    成员参数：
        expansion = 1（BasicBlock 输出通道数不膨胀；Bottleneck 为 4）
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        初始化 BasicBlock。

        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            stride (int): 第一层卷积的 stride；若 >1 则同时做空间下采样
        """
        super().__init__()

        # ---- 主分支（2 层 3×3 卷积） ----
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

        # ---- 短接分支（shortcut） ----
        # 若输入输出维度（通道数或空间尺寸，由 stride 决定）不一致，做投影
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """
        前向传播：主分支 + 短接分支 → ReLU 激活。

        Args:
            x (torch.Tensor): shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: shape (B, out_channels, H/stride, W/stride)
        """
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # 残差相加（维度由 self.shortcut 保证对齐）
        out = F.relu(out, inplace=True)
        return out


# ------------------------------------------------------------------
# ResNet-18 主体（CIFAR-10 适配版）
# ------------------------------------------------------------------
class ResNet18(nn.Module):
    """
    CIFAR-10 适配版 ResNet-18。

    完整 forward 流程（以 32×32 输入为例）：
        conv1(3→64, 3×3, s=1, p=1) : (B, 64, 32, 32)      ← 【关键适配1：不用 7×7 s=2】
            → bn1 → relu
            → 【关键适配2：去掉 ImageNet 版的 MaxPool(3×3,s=2)】
        layer1 (2 个 BasicBlock, 64 通道, s=1) : (B, 64, 32, 32)
        layer2 (2 个 BasicBlock, 128 通道, s=2) : (B, 128, 16, 16)
        layer3 (2 个 BasicBlock, 256 通道, s=2) : (B, 256, 8, 8)
        layer4 (2 个 BasicBlock, 512 通道, s=2) : (B, 512, 4, 4)
            → AdaptiveAvgPool2d((1, 1)) : (B, 512, 1, 1)
            → flatten : (B, 512)
            → Linear(512 → num_classes) : (B, num_classes)
                ← 与 torchvision ResNet-18 完全一致的分类头

    属性名约定（与 torchvision.models.resnet18 对齐）：
        self.conv1 / self.bn1 / self.layer1~4 / self.fc
        这样保证 load_state_dict 时，即使未来想加载预训练权重也能匹配。
    """

    def __init__(self, num_classes: int = 10, block=BasicBlock, num_blocks: list = None):
        """
        初始化 ResNet-18。

        Args:
            num_classes (int): 分类类别数，CIFAR-10 默认 10
            block (nn.Module): 残差块类型，默认 BasicBlock
            num_blocks (list[int]): 每个 stage 内的残差块数量，
                                    ResNet-18 = [2, 2, 2, 2]（共 2×4×2 + conv1+fc ≈ 18 层）
        """
        super().__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_channels = 64   # 进入 layer1 前的通道数，会被 _make_layer 动态更新

        # ---- Stage 0：初始 conv1（【CIFAR-10 适配1】3×3 stride=1，不用 7×7 stride=2） ----
        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)

        # ---- Stage 1~4：4 组残差块（通道 64→128→256→512） ----
        # layer1: stride=1，不做空间下采样
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        # layer2~4: stride=2，每次将特征图空间尺寸减半
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # ---- 分类头：全局平均池化 + 线性层 ----
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # BasicBlock.expansion = 1，所以最终通道 = 512 × 1
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # ---- 权重初始化（Kaiming He 初始化，沿用 ResNet 官方默认） ----
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int):
        """
        构造一个 stage（layer1/layer2/layer3/layer4）。

        每个 stage 的第一个 block 可带 stride>1 做下采样 + 通道膨胀；
        其余 block 的 stride 固定为 1，输入输出通道数一致。

        Args:
            block: 残差块类型（BasicBlock / Bottleneck）
            out_channels (int): 该 stage 输出通道数
            num_blocks (int): 该 stage 内残差块的数量
            stride (int): 第一个残差块的 stride（>1 时做下采样）

        Returns:
            nn.Sequential: 组合好的 stage
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            # 构造完一个 block 后，下一个 block 的 in_channels 要等于当前 out_channels
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播（与 torchvision ResNet forward 语义保持一致）。

        Args:
            x (torch.Tensor): 输入图片，shape = (B, 3, 32, 32) （CIFAR-10）

        Returns:
            torch.Tensor: 未经过 softmax 的 logits，shape = (B, num_classes)
        """
        # Stage 0：初始 conv1 + bn + relu（【CIFAR-10 适配2】不做 MaxPool）
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        # Stage 1~4：4 组残差块
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 分类头：自适应平均池化 → 展平 → Linear
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        logits = self.fc(out)
        return logits
