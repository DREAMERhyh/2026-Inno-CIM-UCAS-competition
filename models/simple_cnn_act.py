"""
SimpleCNN 系列的**激活函数参数化**变体（M14b：ReLU 非负性假说的对照实验）。

动机
----
M14b 的问句：**池化的保护，是不是来自"post-ReLU 特征非负 ⇒ 表示接近轴对齐"？**
可测预测：把 ReLU 换成 GELU / LeakyReLU（破坏非负性），池化密度的保护应当变弱。

本文件提供 2 档池化密度 × 3 种激活 = 6 个组合。其中 `act="relu"` 的两档是既有实现的
**结构逐字镜像**（层命名、初始化顺序、forward 顺序全同），用途是回归自检——
`load_state_dict` 互通且数值逐位一致，才说明"换激活"是本实验唯一的自变量。

| 类 | 镜像的既有实现 | MaxPool 次数 | 空间尺寸 |
|---|---|---|---|
| `SimpleCNNAct` | `models/simple_cnn.py` 的 `SimpleCNN` | 2 | 32→16→8 |
| `SimpleCNNMaxPoolAct` | `v3_methods_simplecnn.py` 的 `SimpleCNNMaxPool` | 5 | 32→16→8→4→2→1 |

三条设计约束（M14b 的自检点，改动前先读）：
1. **参数量与 ReLU 版逐位相同** —— 激活不带参数，换激活不该改参数量；
2. **参数命名与 ReLU 版逐字相同** —— 否则既有 checkpoint 无法载入、镜像失效；
3. `act="relu"` 时必须与既有实现**数值逐位一致**（相同权重 + 相同输入下 `torch.equal`）。
"""

import torch
import torch.nn as nn

ACT_KINDS = ("relu", "gelu", "leaky")

# LeakyReLU 的负斜率固定为 PyTorch 默认值 0.01：只破坏非负性，不引入额外超参。
# ⚠ 这是实验设计变量，要改必须在预登记里写死（见 docs/verify/M14B_PREP.md §6）。
LEAKY_NEGATIVE_SLOPE = 0.01


def make_act(kind: str) -> nn.Module:
    """激活工厂。

    ⚠ `relu` 分支与既有实现逐字一致（`nn.ReLU(inplace=True)`）——
    这是 relu 档能与既有产物逐位对齐的前提，不要改。
    """
    if kind == "relu":
        return nn.ReLU(inplace=True)
    if kind == "gelu":
        # 精确 erf 版本。approximate="tanh" 是另一族形状，不要混用。
        return nn.GELU(approximate="none")
    if kind == "leaky":
        return nn.LeakyReLU(negative_slope=LEAKY_NEGATIVE_SLOPE, inplace=True)
    raise ValueError(f"未知激活: {kind!r}（可选 {ACT_KINDS}）")


class SimpleCNNAct(nn.Module):
    """`models/simple_cnn.SimpleCNN` 的激活参数化镜像（2×MaxPool）。

    逐层结构、层命名、forward 顺序与 `SimpleCNN` 完全一致，
    只把 4 处 `nn.ReLU(inplace=True)` 换成 `make_act(act)`。
    """

    def __init__(self, num_classes: int = 10, act: str = "relu"):
        super().__init__()
        self.act = act

        # 卷积层1: 输入 3x32x32 -> 输出 32x32x32
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = make_act(act)

        # 卷积层2: 输入 32x32x32 -> 输出 64x16x16 (经过 MaxPool)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = make_act(act)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层3: 输入 64x16x16 -> 输出 128x16x16
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = make_act(act)

        # 卷积层4: 输入 128x16x16 -> 输出 128x8x8 (经过 MaxPool)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = make_act(act)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化: 将每个通道的 8x8 特征图压缩为 1 个标量
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接分类层: 输入 128 维 -> 输出 num_classes 维 (logits)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
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

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)


class SimpleCNNMaxPoolAct(nn.Module):
    """`v3_methods_simplecnn.SimpleCNNMaxPool` 的激活参数化镜像（5×MaxPool）。

    逐层结构、层命名、forward 顺序与该实现完全一致，
    只把 4 处 `nn.ReLU(inplace=True)` 换成 `make_act(act)`。
    """

    def __init__(self, num_classes: int = 100, act: str = "relu"):
        super().__init__()
        self.act = act

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = make_act(act)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = make_act(act)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = make_act(act)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = make_act(act)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool5(self.pool4(self.relu4(self.bn4(self.conv4(x)))))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
