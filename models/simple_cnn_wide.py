# -*- coding: utf-8 -*-
"""
SimpleCNN 的中间容量档 —— 秩-恢复曲线的第 4 个点（M5）

本文件含**两个**类：

| 类 | 状态 | 说明 |
|---|---|---|
| `SimpleCNNWideV2` | ✅ **现行，接线用它** | 第二次下采样提前 + 后两层加宽 |
| `SimpleCNNWide`   | ⚠ **已被 V2 取代，保留供追溯，不接线** | 纯加宽（通道 ×2） |

────────────────────────────────────────────────────────────────
为什么 V1 被取代（**根因：参数量可以堆，计算量堆不起**）
────────────────────────────────────────────────────────────────
V1 只加宽不动拓扑，通道 64/128/256/512，实测：
    MACs = 454.8M/sample = **VGG-11 的 2.98 倍**（而参数量只有 VGG-11 的 1/5）
    ⇒ 按 VGG-11/120ep = 110 min 推，单次训练 **2~5.5 h**，超出 M5 的窗口。

根因是算术必然：**卷积的 参数/MACs 比 = 1/(H_out·W_out)**。
V1 把参数量的大头（1.18M，占主干 84%）压在 **16×16** 的 conv4 上，
一个参数要花 256 次 MAC —— 而同样的参数放在 8×8 上只花 64 次、放在 4×4 上只花 16 次。

────────────────────────────────────────────────────────────────
V2 的解法："更早下采样 + 只加宽后两层"
────────────────────────────────────────────────────────────────
  - conv1/conv2 与 SimpleCNN **逐层完全相同**（3→32、32→64，都在 32×32）⇒ 浅层变量锁死
  - **第二次 MaxPool 从 conv4 之后提前到 conv3 之后** ⇒ conv4 从 16×16 落到 8×8
  - conv3/conv4 加宽到 256/512（SimpleCNN 是 128/128）
  - 下采样次数仍是 **2 次**（与 SimpleCNN 相同，不是 3 次）

实测（见 docs/verify/M5_MODEL_PREP.md §2）：
    CIFAR-100：主干 1,399,428 / 含校准 1,662,340
    MACs ≈ 133M/sample = **0.87 × VGG-11** —— 比 VGG-11 还省
    参数量 1.40M ≈ √(0.24M × 9.2M) = 1.49M（SimpleCNN 与 VGG-11 的几何中点）

⚠ **拓扑代价（必须写进 M5 报告，不得声称"只差容量"）**：
    V2 与 SimpleCNN 之间差的是 **容量 + 第二次下采样的位置** 两个变量
    （SimpleCNN 的 conv4 在 16×16，V2 的在 8×8）。
    这是 MACs 硬约束（≤1.9e8）下的必然取舍，不是疏忽。

────────────────────────────────────────────────────────────────
校准模块（沿用 SimpleCNN Exp2 的方案，非 VGG-11 的"每个 MaxPool 后"方案）
────────────────────────────────────────────────────────────────
复用 models.robust_cnn.FeatureCalibration，零修改；单一注入点，位于 GAP 之后、FC 之前
（与 RobustCNN 相同）。in_dim=512 = 本模型的 GAP 输出维度，hidden_dim=256 保持
SimpleCNN 的"瓶颈减半"比例。use_calibration=False 退化为纯主干（clean 口径对照用）。

层名（conv1..conv4 / bn / relu / pool / fc）与 SimpleCNN 同构 —— 这是刻意的：
task_extension6 的 get_depth_group 靠层名分派分层 α，同名才能沿用 SimpleCNN Exp2 的语义。
"""

import torch.nn as nn

from models.robust_cnn import FeatureCalibration


class SimpleCNNWideV2(nn.Module):
    """
    SimpleCNN 的加宽版 V2：第二次下采样提前到 conv3 之后，conv3/conv4 加宽。

    结构（输入 32×32）：
        conv1 (3→32)     @32×32
        conv2 (32→64)    @32×32 → MaxPool → 16×16
        conv3 (64→256)   @16×16 → MaxPool → 8×8
        conv4 (256→512)  @8×8
        GAP → (B,512) → [FeatureCalibration(512→256→512)] → FC

    Args:
        num_classes (int): 分类类别数，CIFAR-100 为 100
        use_calibration (bool): 是否在 GAP 与 FC 之间插入 FeatureCalibration。
            默认 True，与 ext6 主流程的 Exp2 配方一致。
    """

    def __init__(self, num_classes: int = 10, use_calibration: bool = True):
        super(SimpleCNNWideV2, self).__init__()
        self.use_calibration = use_calibration

        # 卷积层命名与 SimpleCNN 完全一致（conv1..conv4 / bn / relu / pool），
        # 使 task_extension6 的 get_depth_group 能沿用 SimpleCNN Exp2 的分层 α 语义。
        # 卷积层1: 输入 3x32x32 -> 输出 32x32x32（与 SimpleCNN 逐层相同）
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)

        # 卷积层2: 输入 32x32x32 -> 输出 64x16x16 (经过 MaxPool)（与 SimpleCNN 逐层相同）
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层3: 输入 64x16x16 -> 输出 256x8x8 (经过 MaxPool)
        # ⚠ V2 的关键改动：这次 MaxPool 在 conv4 **之前**（SimpleCNN 是在 conv4 之后）
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层4: 输入 256x8x8 -> 输出 512x8x8（分辨率不再降，直接进 GAP）
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        # 全局平均池化: 将每个通道的 8x8 特征图压缩为 1 个标量
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 残差校准模块（与 RobustCNN 相同的单一注入点：GAP 之后、FC 之前）
        if self.use_calibration:
            self.calibration = FeatureCalibration(in_dim=512, hidden_dim=256)

        # 全连接分类层: 输入 512 维 -> 输出 num_classes 维 (logits)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入图像张量，形状为 (B, 3, 32, 32)

        Returns:
            torch.Tensor: 未经过 softmax 的 logits，形状为 (B, num_classes)
        """
        # Stage 1（32×32）
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Stage 2（32×32 → 16×16）
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        # Stage 3（16×16 → 8×8）
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool2(x)

        # Stage 4（8×8）
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        # 全局平均池化 -> (B, 512, 1, 1) -> (B, 512)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        # 条件启用残差校准模块
        if self.use_calibration:
            x = self.calibration(x)

        # 全连接分类层，输出 logits
        logits = self.fc(x)

        return logits


class SimpleCNNWide(nn.Module):
    """
    ⚠ **已被 `SimpleCNNWideV2` 取代，保留供追溯，不接线。**

    取代原因：本版只加宽不动拓扑（通道 64/128/256/512，conv3/conv4 都在 16×16），
    实测 MACs = 454.8M/sample = **VGG-11 的 2.98 倍** ⇒ 单次训练 2~5.5 h，超出 M5 窗口。
    详见文件头 docstring 与 docs/verify/M5_MODEL_PREP.md §2.2。

    SimpleCNN 的加宽版（通道逐层 ×2），可选残差特征校准。

    Args:
        num_classes (int): 分类类别数，CIFAR-100 为 100
        use_calibration (bool): 是否在 GAP 与 FC 之间插入 FeatureCalibration。
    """

    def __init__(self, num_classes: int = 10, use_calibration: bool = True):
        super(SimpleCNNWide, self).__init__()
        self.use_calibration = use_calibration

        # 卷积层命名与 SimpleCNN 完全一致（conv1..conv4 / bn / relu / pool）
        # 卷积层1: 输入 3x32x32 -> 输出 64x32x32
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # 卷积层2: 输入 64x32x32 -> 输出 128x16x16 (经过 MaxPool)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层3: 输入 128x16x16 -> 输出 256x16x16
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        # 卷积层4: 输入 256x16x16 -> 输出 512x8x8 (经过 MaxPool)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 残差校准模块（单一注入点：GAP 之后、FC 之前）
        if self.use_calibration:
            self.calibration = FeatureCalibration(in_dim=512, hidden_dim=256)

        # 全连接分类层
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (B, 3, 32, 32)

        Returns:
            torch.Tensor: 未经过 softmax 的 logits，形状为 (B, num_classes)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        if self.use_calibration:
            x = self.calibration(x)

        logits = self.fc(x)

        return logits
