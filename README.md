# CIFAR-10 Baseline 训练与分析框架

一个模块化、可扩展的 CIFAR-10 图像分类深度学习项目框架。内置 SimpleCNN 作为 Baseline 模型，同时支持轻松集成新的模型架构、可选早停（Early Stopping）、断点续训（Resume Training）与丰富的可视化输出，便于开展多种实验与分析。

---

## 项目结构

```
cifar10_project/
├── data/                              # 数据集目录（已就位，无需修改）
│   └── cifar-10-batches-py/           # CIFAR-10 Python 格式数据
├── models/                            # 所有模型定义
│   ├── __init__.py
│   ├── simple_cnn.py                  # Baseline: SimpleCNN（~0.24M 参数）
│   ├── robust_cnn.py                  # 【Task3 新增】RobustCNN（残差校准模块）
│   ├── resnet.py                      # 【Ext1 新增】CIFAR-10 适配版 ResNet-18（~11.17M 参数）
│   └── vgg11.py                       # 【新增 VGG-11】CIFAR-10 轻量适配 VGG-11（~9.2M 参数，纯前馈深层对照）
├── utils/                             # 通用工具模块
│   ├── __init__.py
│   ├── data_loader.py                 # 数据加载 + 数据增强
│   ├── trainer.py                     # 通用 Trainer（支持早停/续训/英文图表）
│   ├── nonlinearity.py                # 【Task1】非线性失真注入模块
│   ├── perturbation.py                # 【Ext2 新增】通用扰动注入（非线性 + 高斯）
│   └── quantization.py                # 【Ext3 新增】量化-反量化误差 & 非线性+量化联合误差注入
├── checkpoints/                       # 训练保存的模型权重
│   ├── simple_cnn/
│   │   └── best_model.pth             # SimpleCNN 干净权重（84.90%）
│   ├── nat_finetune_simplecnn/        # SimpleCNN NAT 微调
│   │   └── best_model.pth
│   ├── nat_scratch_simplecnn/         # SimpleCNN NAT 从头训练
│   │   └── best_model.pth
│   ├── Exp1_CalibOnly_simplecnn/      # SimpleCNN 任务3 仅方案A
│   │   └── best_model.pth
│   ├── Exp2_Calib+Layerwise_simplecnn/  # SimpleCNN 任务3 方案A + 分层加权
│   │   └── best_model.pth
│   ├── Exp3_FullRobust_simplecnn/     # SimpleCNN 任务3 完整方案
│   │   └── best_model.pth
│   ├── resnet18/                      # 【Ext1 新增】ResNet-18 干净权重（92.09%）
│   │   └── best_model.pth
│   ├── nat_scratch_resnet18/          # ResNet-18 NAT（若训练）
│   │   └── best_model.pth
│   ├── vgg11/                         # 【新增 VGG-11】干净权重目录
│   │   └── best_model.pth
│   └── nat_scratch_vgg11/             # VGG-11 NAT（若训练）
│       └── best_model.pth
├── outputs/
│   ├── simple_cnn/                    # 原始训练输出（曲线/混淆矩阵/指标等）
│   ├── task1_simplecnn/               # 【Task1】SimpleCNN α敏感性 + 层偏移
│   │   ├── alpha_sensitivity.csv      # α vs 准确率 & loss
│   │   ├── accuracy_vs_alpha.png      # α-精度衰减曲线
│   │   ├── layer_distribution_shift.csv # 层偏移汇总 (RME / Cosine)
│   │   ├── error_accumulation.png     # 跨层误差累积曲线
│   │   ├── histogram_conv1.png        # （可选）conv1 分布对比
│   │   └── histogram_conv4.png        # （可选）conv4 分布对比
│   ├── task1_resnet18/                # 【Task1】ResNet-18 α敏感性 + 层偏移
│   │   └── (结构同 task1_simplecnn/)
│   ├── task1_vgg11/                   # 【Task1】VGG-11 α敏感性 + 层偏移
│   │   └── (结构同 task1_simplecnn/)
│   ├── task2_simplecnn/               # 【Task2】SimpleCNN NAT 训练 & 对比结果
│   │   ├── finetune/
│   │   │   ├── training_curves.png              # 训练 Loss/Acc 曲线
│   │   │   ├── metrics.json                     # 最终训练指标
│   │   │   ├── confusion_matrix.png             # 干净测试集混淆矩阵
│   │   │   ├── alpha_sensitivity.csv            # NAT 模型 α 敏感性扫描
│   │   │   └── accuracy_vs_alpha_comparison.png # 与任务1基线的 α-精度对比曲线
│   │   └── scratch/
│   │       └── (结构同 finetune/)
│   ├── task2_resnet18_scratch/        # 【Task2】ResNet-18 NAT Scratch（若运行）
│   │   └── (结构同 task2_simplecnn/scratch/)
│   ├── task2_vgg11_scratch/           # 【Task2】VGG-11 NAT Scratch（若运行）
│   │   └── (结构同 task2_simplecnn/scratch/)
│   ├── task3_simplecnn/               # 【Task3】SimpleCNN 鲁棒性增强消融结果
│   │   ├── Exp1_CalibOnly/
│   │   │   ├── training_curves.png              # 训练 Loss/Acc 曲线
│   │   │   ├── metrics.json                     # 含三个布尔开关 use_calibration 等
│   │   │   ├── confusion_matrix.png             # 干净测试集混淆矩阵
│   │   │   ├── alpha_sensitivity.csv            # 当前实验的 α 扫描结果
│   │   │   └── accuracy_vs_alpha_comparison.png # 与 NAT-Scratch 基线的对比曲线
│   │   ├── Exp2_Calib+Layerwise/
│   │   │   └── (结构同 Exp1/)
│   │   ├── Exp3_FullRobust/
│   │   │   └── (结构同 Exp1/)
│   │   ├── ablation_summary.csv       # （仅 --exp all）消融实验汇总表
│   │   └── ablation_comparison.png    # （仅 --exp all）3 条消融曲线 + 基线对比图
│   ├── extension1/                    # 【Ext1】三模型架构对比（SimpleCNN + ResNet-18 + VGG-11）
│   │   ├── simple_cnn_alpha_sensitivity.csv    # SimpleCNN α 扫描
│   │   ├── resnet18_alpha_sensitivity.csv      # ResNet-18 α 扫描
│   │   ├── vgg11_alpha_sensitivity.csv         # VGG-11 α 扫描
│   │   ├── accuracy_comparison.png             # 三模型 α-精度曲线对比
│   │   ├── accuracy_drop_comparison.png        # 三模型相对精度跌幅对比
│   │   ├── robustness_vs_params.png            # 参数量 vs α=0.3 准确率散点（对数轴）
│   │   └── network_comparison_summary.csv      # 对比汇总表（参数量/干净精度/跌幅）
│   ├── extension2_simplecnn/          # 【Ext2】SimpleCNN 高斯噪声 vs 非线性对比
│   │   ├── simple_cnn/
│   │   │   ├── nonlinearity_sensitivity.csv     # α ∈ [-0.3, 0.3] 精度扫描
│   │   │   ├── gaussian_sensitivity.csv         # σ ∈ [0.05, 0.30] 精度扫描
│   │   │   ├── layer_shift_nonlinearity.csv     # 非线性层偏移 (RME / cos)
│   │   │   ├── layer_shift_gaussian.csv         # 高斯层偏移 (RME / cos)
│   │   │   ├── error_accumulation_nonlinearity.png
│   │   │   ├── error_accumulation_gaussian.png
│   │   │   ├── histogram_conv1_gaussian.png     # conv1 σ=0 vs σ=0.20 分布
│   │   │   ├── histogram_conv4_gaussian.png     # conv4 σ=0 vs σ=0.20 分布
│   │   │   ├── accuracy_comparison_simple_cnn.png   # 双横轴 α vs σ 精度对比
│   │   │   └── error_accumulation_comparison_simple_cnn.png
│   │   ├── exp2/                                # 任务3 Exp2 鲁棒模型（结构同上）
│   │   ├── error_accumulation_comparison_exp2.png
│   │   └── robustness_transfer_comparison.png  # SimpleCNN vs Exp2 × 两扰动 四曲线迁移对比
│   ├── extension2_resnet18/           # 【Ext2】ResNet-18 高斯噪声 vs 非线性对比
│   │   └── (结构同 extension2_simplecnn/ 的 simple_cnn/ 子目录)
│   ├── extension2_vgg11/              # 【Ext2】VGG-11 高斯噪声 vs 非线性对比
│   │   └── (结构同 extension2_simplecnn/ 的 simple_cnn/ 子目录)
│   ├── extension3_simplecnn/          # 【Ext3】SimpleCNN 量化 + 非线性联合分析
│   │   ├── simple_cnn_quantization_only.csv   # SimpleCNN 单独量化扫描（bit→acc/loss）
│   │   ├── exp2_quantization_only.csv         # Exp2 单独量化扫描
│   │   ├── simple_cnn_joint_error.csv         # SimpleCNN 联合误差（α×bit 网格）
│   │   ├── exp2_joint_error.csv               # Exp2 联合误差（α×bit 网格）
│   │   ├── quantization_only_simple_cnn.png   # SimpleCNN 单独 bit vs acc 曲线
│   │   ├── quantization_only_exp2.png         # Exp2 单独 bit vs acc 曲线
│   │   ├── joint_error_heatmap_simple_cnn.png # SimpleCNN 联合热力图（RdYlGn + 数值标注）
│   │   ├── joint_error_heatmap_exp2.png       # Exp2 联合热力图
│   │   ├── joint_vs_alone_comparison.png      # （1×N 子图）联合 vs 单独 三曲线
│   │   ├── robustness_under_joint_error.png   # 两模型联合误差鲁棒性对比
│   │   └── joint_error_summary.csv            # 三类误差汇总（error_type 字段区分）
│   ├── extension3_resnet18/           # 【Ext3】ResNet-18 量化 + 非线性联合分析
│   │   └── (结构同 extension3_simplecnn/)
│   └── extension3_vgg11/              # 【Ext3】VGG-11 量化 + 非线性联合分析
│       └── (结构同 extension3_simplecnn/)
├── train.py                           # 训练脚本（支持续训 --resume）
├── evaluate.py                        # 评估脚本
├── task1_sensitivity_analysis.py      # 【Task1】敏感性主分析脚本
├── task1_layer_analysis.py            # 【Task1】单层分布偏移分析脚本
├── task2_nat.py                       # 【Task2】非线性感知训练（NAT）统一入口
├── task3_robust.py                    # 【Task3 新增】鲁棒性增强消融统一入口
├── task_extension1_network_comparison.py      # 【Ext1 新增】网络结构 & 参数量对比脚本
├── task_extension2_noise_vs_nonlinearity.py  # 【Ext2 新增】高斯噪声 vs 非线性失真对比脚本
├── task_extension3_joint_analysis.py          # 【Ext3 新增】量化 + 非线性联合误差分析脚本
├── run_all_models.py                  # 【新增】批量运行所有模型的完整分析管线
├── requirements.txt                   # Python 依赖库
└── README.md                          # 本文档
```

---

## 数据集说明

本项目使用 **CIFAR-10** 数据集，已存放在 `data/` 目录下：

- **大小**：60,000 张 32×32 彩色图像
- **类别**：10 类，每类 6,000 张图像
  - `airplane`, `automobile`, `bird`, `cat`, `deer`
  - `dog`, `frog`, `horse`, `ship`, `truck`
- **划分**：50,000 张训练集 + 10,000 张测试集
- **加载方式**：通过 `torchvision.datasets.CIFAR10(root="./data")` 直接加载

数据预处理：
- **训练集**：随机水平翻转 (p=0.5) → 先 Pad 4 像素再随机裁剪 32×32 → 归一化
- **测试集**：仅归一化
- **归一化参数**：`mean=(0.4914, 0.4822, 0.4465)`，`std=(0.2023, 0.1994, 0.2010)`

---

## 环境安装指南

### 1. 推荐环境

- Python >= 3.8
- CUDA >= 11.0 (如使用 GPU 训练)

### 2. 安装依赖

```bash
# 进入项目根目录
cd cifar10_project

# （可选）创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# 安装项目依赖
pip install -r requirements.txt
```

如需要特定 CUDA 版本的 PyTorch，请前往 [PyTorch 官网](https://pytorch.org/) 生成对应的安装命令。

---

## 训练命令示例

### 基础训练（默认参数）

```bash
python train.py --model simple_cnn --epochs 50
```

### 自定义超参数

```bash
python train.py \
    --model simple_cnn \
    --epochs 100 \
    --batch_size 256 \
    --lr 0.05 \
    --weight_decay 1e-4 \
    --num_workers 4
```

### 启用早停

连续 10 个 epoch 测试准确率不提升则停止训练：

```bash
python train.py --model simple_cnn --epochs 100 --patience 10
```

### 从 Checkpoint 续训（Resume Training）

假设之前已训练到 epoch 50 并保存了 `best_model.pth`，希望继续训练到总共 100 个 epoch：

```bash
# --epochs 100   指最终总 epoch 数（不是再训多少个）
# --resume  <path>   指定要恢复的 checkpoint 文件
python train.py \
    --model simple_cnn \
    --epochs 100 \
    --resume ./checkpoints/simple_cnn/best_model.pth
```

运行时会自动：
1. 加载模型权重 `model_state_dict`
2. 加载优化器状态 `optimizer_state_dict`（保留动量 buffer 等）
3. 读取 `epoch = 50` 和 `best_test_acc = 84.79%`
4. **从 epoch 51 开始继续训练**，直到 epoch 100
5. 学习率调度器 CosineAnnealingLR 会手动 step 到第 50 步，保持衰减曲线连续
6. 若续训期间新的测试准确率超过历史最佳（84.79%），会自动覆盖保存新的 `best_model.pth`

> **重要提示**：续训时请将 `--epochs` 设置为你期望的**最终总 epoch 数**（例如原本想训 100，只训到 50 中断，则 `--epochs 100`），框架会自动计算需要增量训练的 epoch 数。

### 指定设备

```bash
# 使用 GPU（默认自动选择）
python train.py --model simple_cnn --device cuda

# 使用 CPU
python train.py --model simple_cnn --device cpu
```

### 完整参数列表

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--model` | str | `simple_cnn` | 模型名称 |
| `--resume` | str | `None` | （可选）checkpoint 路径，用于断点续训。例如 `./checkpoints/simple_cnn/best_model.pth` |
| `--batch_size` | int | `128` | 训练批次大小 |
| `--epochs` | int | `50` | **总目标**训练轮数（续训模式下请填最终期望达到的总 epoch 数） |
| `--lr` | float | `0.01` | 初始学习率（续训时若 checkpoint 含优化器状态，则以其 lr 为准） |
| `--weight_decay` | float | `1e-4` | L2 权重衰减系数 |
| `--momentum` | float | `0.9` | SGD 动量 |
| `--device` | str | `自动` | `cuda` / `cpu` |
| `--num_workers` | int | `2` | DataLoader 加载线程数 |
| `--seed` | int | `42` | 随机种子（保证可复现） |
| `--patience` | int | `None` | 早停耐心轮数（默认不启用） |

---

## 评估命令示例

训练完成后，可使用 `evaluate.py` 加载最佳权重，对模型进行独立评估：

### 基础评估

```bash
python evaluate.py \
    --model simple_cnn \
    --checkpoint ./checkpoints/simple_cnn/best_model.pth
```

### 保存评估结果

指定 `--save_output_dir` 将同时保存混淆矩阵和指标 JSON：

```bash
python evaluate.py \
    --model simple_cnn \
    --checkpoint ./checkpoints/simple_cnn/best_model.pth \
    --save_output_dir ./outputs/simple_cnn
```

评估脚本将计算并打印：
- **总体准确率** (Accuracy)
- **每类指标**：精确率 / 召回率 / F1-score
- **宏平均** (Macro Avg) 与 **加权平均** (Weighted Avg)

---

## 模型管理说明

### VGG-11 模型（CIFAR-10 轻量适配版）

| 属性 | 值 |
|------|-----|
| 文件 | `models/vgg11.py` |
| 类名 | `VGG11` |
| 参数量 | 约 9.2M |
| 结构 | 8 Conv (3×3, 通道 64→128→256→256→512→512→512→512) + 5 MaxPool + GAP + FC(512→10) |
| 特点 | 深层纯前馈、无残差连接、分类头轻量化（GAP+单FC，比原始 3FC 节省 120M+ 参数） |
| 在对比中的定位 | SimpleCNN（浅层 0.24M）与 ResNet-18（深层残差 11.17M）之间的**深层纯前馈对照**，用于剥离残差连接对非线性鲁棒性的影响 |

**核心设计动机**：
- SimpleCNN 只有 4 个卷积层（浅层），ResNet-18 有大量残差恒等连接，两者之间缺少深层纯前馈的参照物。
- VGG-11 提供了「深层 + 纯前馈 + 无残差」的中间对照，三模型联合分析（浅层 vs 深层纯前馈 vs 深层残差）可以独立回答：
  1. 模型深度增加是否必然导致非线性脆弱？（SimpleCNN vs VGG-11）
  2. 残差连接是否真正在起误差稳定作用？（VGG-11 vs ResNet-18）

**训练命令**：
```bash
python train.py --model vgg11 --epochs 100 --lr 0.01 --batch_size 128
# 权重保存：./checkpoints/vgg11/best_model.pth
```

**三模型对照基线（供参考）**：
| 模型 | 参数量 | 干净精度（训练后） | 结构定位 |
|------|------|------|------|
| SimpleCNN | ~0.24M | ~84.90% | 浅层小模型（4 Conv + 1 FC） |
| VGG-11 | ~9.2M | 待训练后得到（参考区间 ~90-93%） | 深层纯前馈（8 Conv + GAP + 1 FC） |
| ResNet-18 | ~11.17M | ~92.09% | 深层残差连接（4 BasicBlock × 2 + GAP + 1 FC） |

---

### 如何添加新模型

本框架设计了简洁的模型注册机制，添加新模型仅需 **3 步**：

#### 步骤 1：在 `models/` 下新建模型文件

例如，新建 `models/resnet.py`：   

```python
# models/resnet.py
import torch.nn as nn

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # ... 网络结构定义 ...

    def forward(self, x):
        # ... 前向传播 ...
        return logits  # 注意：返回未加 softmax 的 logits
```

#### 步骤 2：在 `models/__init__.py` 中导入（可选但推荐）

```python
from .simple_cnn import SimpleCNN
from .resnet import ResNet18  # 新增

__all__ = ["SimpleCNN", "ResNet18"]
```

#### 步骤 3：在 `train.py` 和 `evaluate.py` 的 `get_model()` 函数中注册分支

两处的 `get_model()` 函数结构完全一致：

```python
def get_model(model_name: str, num_classes: int = 10):
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)

    # ======== 新增分支 ========
    elif model_name == "resnet18":
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes)
    # ========================

    else:
        raise ValueError(f"不支持的模型名称: '{model_name}'")
```

完成后即可使用：
```bash
python train.py --model resnet18 --epochs 50
```

---

## 输出说明

### 1. `checkpoints/<model_name>/best_model.pth`

训练过程中保存的**最佳模型权重**（以测试集准确率最高的 epoch 为准），**同时也是断点续训的唯一入口文件**（`train.py --resume` 读取的就是它）。

文件格式为字典，包含：
```python
{
    "epoch":            最佳模型的 epoch 编号（续训时用它推导 start_epoch）,
    "model_state_dict":      模型 state_dict,
    "optimizer_state_dict":  优化器 state_dict（续训时恢复动量 buffer 等）,
    "best_test_acc":    最佳测试准确率（0~100，续训时作为历史最佳继续比较）,
}
```

> 💡 **续训小提示**：续训结束后，`best_model.pth` 会被再次覆盖更新，始终保存的是「历史 + 本次续训」综合的最佳权重。

### 2. `outputs/<model_name>/training_curves.png`

训练曲线图，包含两个子图：
- **上方**：Loss 曲线（训练 / 测试）
- **下方**：Accuracy 曲线（训练 / 测试），并以红色星号标注最佳测试准确率点

### 3. `outputs/<model_name>/confusion_matrix.png`

测试集 10×10 混淆矩阵热力图，便于分析模型在哪些类别上容易混淆（例如猫与狗、鸟与飞机等）。

### 4. `outputs/<model_name>/metrics.json`

最终指标 JSON 摘要，方便后续实验对比：

**从头训练的示例：**
```json
{
    "best_test_acc": 85.42,
    "final_train_loss": 0.231456,
    "final_test_loss": 0.512345,
    "final_train_acc": 92.31,
    "final_test_acc": 83.15,
    "total_epochs": 50,
    "start_epoch": 0,
    "resumed": false,
    "best_epoch": 47
}
```

**续训模式下新增字段说明：**
- `total_epochs`：**总目标 epoch 数**（即命令行 `--epochs` 的值，不是续训增量）
- `start_epoch`：续训前已完成的 epoch 数（从头训练为 0）
- `resumed`：布尔值，是否为续训模式
- `best_epoch`：整体最佳准确率所在的全局 epoch 编号

---

## Baseline 模型 (SimpleCNN) 结构

输入 `3×32×32` → 共 4 层卷积 + 全局平均池化 + 全连接分类：

| 层 | 输出尺寸 | 操作 |
| --- | --- | --- |
| Conv1 | 32×32×32 | Conv3×3(32) + BN + ReLU |
| Conv2 | 64×16×16 | Conv3×3(64) + BN + ReLU + MaxPool(2×2) |
| Conv3 | 128×16×16 | Conv3×3(128) + BN + ReLU |
| Conv4 | 128×8×8 | Conv3×3(128) + BN + ReLU + MaxPool(2×2) |
| GAP + FC | 10 | AdaptiveAvgPool(1×1) + Linear(128→10) |

默认训练配置：
- Optimizer：SGD + momentum=0.9 + weight_decay=1e-4
- LR Scheduler：CosineAnnealingLR（T_max = epochs）
- Loss：CrossEntropyLoss
- 随机种子：42（保证结果可复现）

---

## 任务1：非线性误差的敏感性分析

### 任务简介

本任务模拟存算一体（Compute-in-Memory, CiM）芯片中，模拟域乘累加（MAC）运算在输入端引入的**输入相关非线性失真**。
失真数学模型为：

$$y = \alpha \cdot x^3 + (1-\alpha) \cdot x$$

其中 $x$ 为逐样本归一化后的理想输入信号，$y$ 为失真后的实际信号，$\alpha$ 控制非线性强度（$\alpha=0$ 为理想线性）。

本任务将该非线性模型注入到 SimpleCNN 中所有 `nn.Conv2d` 与 `nn.Linear` 算子的输入侧，完成：
1. **整体精度敏感性扫描**：不同 $\alpha$ 下测试集准确率的衰减曲线；
2. **层分布偏移分析**：各层（conv1 ~ conv4 + fc）输出相对于干净基线的 RME 与余弦相似度；
3. **误差跨层累积曲线**：直观观察误差如何随着网络深度逐级累积。

### 运行命令示例

```bash
# ============== 1) 敏感性分析主脚本（必跑） ==============
# 使用默认 alpha 范围: -0.3,-0.2,-0.1,0.0,0.1,0.2,0.3
python task1_sensitivity_analysis.py

# 自定义 alpha 扫描范围
python task1_sensitivity_analysis.py \
    --alpha_values "-0.5,-0.3,0.0,0.3,0.5" \
    --model simple_cnn \
    --checkpoint ./checkpoints/simple_cnn/best_model.pth \
    --device cuda \
    --output_dir ./outputs/task1_simplecnn

# ============== 2) 单层分布偏移与误差累积分析 ==============
# 对 conv1/conv2/conv3/conv4/fc 共 5 个观测点做偏移分析与直方图对比
python task1_layer_analysis.py

# 自定义 alpha 与样本数（样本越多分析越稳定，但速度越慢）
python task1_layer_analysis.py \
    --alpha_values "-0.3,0.0,0.3" \
    --num_samples 500 \
    --output_dir ./outputs/task1_simplecnn
```

### 输出文件说明（`./outputs/task1_simplecnn/`）

| 文件名 | 含义 |
| --- | --- |
| `alpha_sensitivity.csv` | 敏感性扫描结果表；列：`alpha, accuracy, loss` |
| `accuracy_vs_alpha.png` | 精度衰减曲线图：横轴 Nonlinearity Strength (α)，纵轴 Test Accuracy (%)；红色星号标注 α=0 基线 |
| `layer_distribution_shift.csv` | 层偏移汇总表；列：`alpha, layer_name, RME, cosine_similarity` |
| `error_accumulation.png` | 跨层误差累积曲线：横轴 Layer Index（conv1_output → fc_output），纵轴 Cosine Similarity with Clean Output；每条曲线对应一个 α |
| `histogram_conv1.png` | （可选）conv1 层在 α=0.0 与 α=0.3 下的输出分布直方图对比 |
| `histogram_conv4.png` | （可选）conv4 层在 α=0.0 与 α=0.3 下的输出分布直方图对比 |

### 代码模块说明

| 新增文件 | 作用 |
| --- | --- |
| `utils/nonlinearity.py` | 非线性误差注入核心模块：<br>• `nonlinearity(x, alpha)`：逐样本动态归一化 + 三次失真 + 逆归一化<br>• `register_nonlinearity_hooks(model, alpha)`：将 forward_pre_hook 注册到所有 Conv2d/Linear 输入端<br>• `remove_hooks(hooks)`：移除钩子，恢复干净推理 |
| `task1_sensitivity_analysis.py` | 敏感性扫描主入口：加载权重 → α 扫描 → 保存 CSV / 绘制 accuracy_vs_alpha.png → 打印中文摘要表格 |
| `task1_layer_analysis.py` | 层分析入口：固定 500 张样本 → 捕获 5 层输出 → 计算 RME / 余弦相似度 → 绘制累积曲线 & 分布直方图 |

> 图表提示：任务1新增的所有图片中**标题、坐标轴、图例均为英文**，终端输出可使用中文便于调试。

---

## 任务2：非线性感知训练 (Nonlinearity-Aware Training, NAT)

### 任务简介

任务1的结论表明：干净模型（84.90%）在正向非线性（α>0）下鲁棒性很差，α=+0.3 时精度仅 55.08%，降幅 29.82%；且 fc 层最敏感、误差跨层逐级累积。

本任务提出**非线性感知训练（NAT）**：
- 训练阶段在每个 batch 的 Conv2d/Linear 输入端动态注入非线性失真（α 采样或固定），使模型"在噪声中学习"；
- 测试阶段仍然在干净数据（α=0）上评估，保证泛化能力。

提供两种训练策略：

| 训练策略 | 初始权重 | 说明 |
| --- | --- | --- |
| **Fine-tuning（微调）** | 从干净预训练 `best_model.pth` 出发（通常已有 84.90% 的基础精度） | 仅需约 40 epochs，计算量小，适合作为基线对比 |
| **Scratch（从头训练）** | 随机初始化 | 全程在带失真的数据上训练，约 120 epochs，可能收获更强的鲁棒性上限 |

---

### 运行命令示例

```bash
# ====== 1) 微调模式（推荐优先尝试） ======
# 默认：每 batch 从 α ∈ [-0.3, 0.3] 均匀随机采样，40 epochs，lr=1e-3
python task2_nat.py --mode finetune --epochs 40 --lr 1e-3

# 微调 + 固定 α=0.3（仅暴露最强正向失真，训练更快）
python task2_nat.py --mode finetune --alpha_mode fixed --alpha_fixed 0.3 --epochs 40

# ====== 2) 从头训练模式 ======
# 默认：随机 α ∈ [-0.3, 0.3]，120 epochs，lr=0.01（与干净模型训练 lr 一致）
python task2_nat.py --mode scratch --epochs 120 --lr 0.01

# 自定义更宽的随机 α 范围（更强扰动，训练更鲁棒）
python task2_nat.py --mode scratch --alpha_range "-0.5,0.5" --epochs 150

# ====== 3) 早停 & 自定义设备 ======
# 连续 8 epoch 干净测试准确率不提升即停止；指定 GPU
python task2_nat.py --mode finetune --epochs 80 --patience 8 --device cuda
```

---

### 关键设计说明

1. **每 batch 钩子注册/移除（TrainerNAT）**
   - 训练前对所有 `nn.Conv2d` 和 `nn.Linear` 挂 `forward_pre_hook`（复用 `utils/nonlinearity.py` 的 `register_nonlinearity_hooks`）；
   - `zero_grad → forward → backward → step` 完成后立即调用 `remove_hooks`，保证权重更新基于真实参数，不被 hook 污染；
   - `random` 模式：每 batch 调用 `random.uniform(low, high)` 采样新 α，实现"训练中见过多种失真强度"。

2. **验证/测试阶段保持干净**
   - 最佳模型保存**依据干净测试集准确率**（α=0），而非失真准确率。
   - 这保证了 NAT 训练得到的模型在干净推理时精度不下降，同时失真下鲁棒性显著提升。

3. **优化器 & 学习率策略**
   - 优化器：SGD + momentum=0.9 + weight_decay=1e-4（与干净模型完全一致，保证公平对比）
   - 调度器：CosineAnnealingLR，T_max = epochs（不做 warm-up，保持简单稳定）

4. **训练后自动 α 扫描 + 基线对比**
   - 训练完成后使用与任务1完全相同的扫描列表 `[-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]` 再跑一次敏感性分析，结果写入 `alpha_sensitivity.csv`；
   - 自动读取 `outputs/task1_simplecnn/alpha_sensitivity.csv`（若存在）作为**未感知训练基线**，与 NAT 模型同轴绘制对比曲线，直观展示 NAT 的鲁棒性收益（尤其 α=+0.3 处的 uplift）。

---

### 输出文件说明

#### Checkpoints（最佳权重）

| 文件 | 位置 | 说明 |
| --- | --- | --- |
| `best_model.pth` | `./checkpoints/nat_finetune/` | 微调模式下干净测试集准确率最高的权重（dict 格式，含 epoch/model_state_dict/optimizer_state_dict/best_test_acc） |
| `best_model.pth` | `./checkpoints/nat_scratch/`  | 从头训练模式下对应权重（同上格式） |

#### 输出目录（每个 mode 独立子目录）

以 `./outputs/task2_simplecnn/finetune/` 为例，`scratch/` 结构完全一致：

| 文件名 | 含义 |
| --- | --- |
| `training_curves.png`              | 2 行子图：<br>上：Train/Test Loss 曲线<br>下：Train/Test Accuracy 曲线，红色星号标注最佳 clean 测试准确率点<br>所有文字英文（Loss Curves / Accuracy Curves / Epoch / Accuracy (%)） |
| `metrics.json`                     | 关键指标 JSON，含 `best_test_acc` / `best_epoch` / `alpha_mode` / `alpha_range` / `alpha_fixed` / `nat_train_mode` 等 |
| `confusion_matrix.png`             | 使用最佳权重在干净测试集上计算的混淆矩阵（10×10 热力图，类别名使用英文 airplane/automobile/.../truck） |
| `alpha_sensitivity.csv`            | 训练后模型在 α ∈ [-0.3, 0.3] 上的扫描结果；列：`alpha, accuracy, loss` |
| `accuracy_vs_alpha_comparison.png` | 与任务1基线的同轴对比曲线：<br>• 红线实线（圆圈）= 当前 NAT 模型（Finetune 或 Scratch）<br>• 蓝线虚线（方块）= 任务1 Vanilla 基线<br>横轴 Nonlinearity Strength (α)，纵轴 Test Accuracy (%)，红色星号额外标出 α=0 的 NAT 干净基线 |

---

### 终端中文摘要（脚本最后输出）

训练 + 扫描完成后，脚本最后打印**中文摘要表格**便于快速对比：
1. 训练模式（FINETUNE / SCRATCH）、预训练权重（若有）、干净最佳准确率；
2. **NAT 模型 α 敏感性扫描表**：Alpha / Accuracy / Accuracy Drop(vs α=0) / Loss，并高亮最大精度跌幅及对应 α；
3. **与任务1基线对比表**（若 `outputs/task1_simplecnn/alpha_sensitivity.csv` 存在）：逐 α 列 Baseline Acc vs NAT Acc vs **提升(%)**，重点关注 α=+0.2 / α=+0.3 区间的 uplift 是否显著；
4. 输出目录 & 最佳权重路径。

---

### 代码模块说明

| 文件 | 核心类 / 函数 | 作用 |
| --- | --- | --- |
| `task2_nat.py` | `set_seed()` / `get_model()` | 与 `train.py` 对齐，保证可复现 |
| `task2_nat.py` | **`TrainerNAT`** | 非线性感知训练器；`_train_one_epoch` 每 batch 注册/移除非线性钩子；`_evaluate` 始终跑干净测试集；保存最佳权重 / 曲线 / 混淆矩阵 / metrics |
| `task2_nat.py` | `evaluate_alpha_scan()` | 训练完成后对 TASK1 相同 α 列表执行敏感性扫描，写 `alpha_sensitivity.csv` |
| `task2_nat.py` | `plot_alpha_comparison()` | 绘制 `accuracy_vs_alpha_comparison.png`，对比任务1基线与 NAT 模型 |
| `task2_nat.py` | `parse_args()` + `main()` | 命令行驱动 + Step0~8 完整流程（seed → 数据 → 模型/权重 → 损失/优化器 → TrainerNAT → α 扫描 → 对比图 → 摘要） |
| `utils/nonlinearity.py` | `register_nonlinearity_hooks` / `remove_hooks` | NAT 训练中每 batch 的钩子注册与移除直接复用 |

> 图表提示：同任务1，**所有生成的图片标题、坐标轴、图例均为英文**；CSV 表头英文；终端日志/摘要使用中文便于调试。

---

## 任务3：鲁棒性增强方法设计

### 任务简介

任务2 NAT-Scratch 虽然已经显著降低了 α=+0.3 下的精度跌幅（22.26% vs 干净模型的 29.82%），
但仍有进一步优化空间。任务3在 NAT-Scratch 的基础上，叠加三种**互补**的鲁棒性增强方法，
通过**消融实验（Ablation Study）** 逐步验证每一项改进的独立贡献：

| 方案 | 类型 | 说明 |
| :---: | --- | --- |
| **A** | 架构增强 | 在 GAP 之后、FC 之前插入残差校准模块 FeatureCalibration<br>结构：Linear(128→64)→ReLU→Linear(64→128) + 残差连接（`out = x + net(x)`）<br>**动机**：fc 层是对非线性失真最敏感的层（余弦相似度仅 ~0.895），校准模块学习"失真特征 → 干净特征"的补偿映射，残差连接保证最坏情况退化为恒等映射，不会损害 clean accuracy。 |
| **B** | 训练策略（分层加权 α） | 浅层（conv1/conv2）：α ∈ [-0.15, 0.15]，用轻失真<br>中层（conv3/conv4）：α ∈ [-0.30, 0.30]，标准失真<br>深层（fc）：每个 batch 随机固定 ±0.3，强失真<br>**动机**：误差从浅层向深层逐级累积（任务1结论），深层承受更大失真压力，浅层避免过度噪声损伤语义。 |
| **C** | 采样策略（非对称偏置） | 70% 概率从 [0, hi] 采样**正向**α（破坏力强，真实硬件更常见），30% 概率从 [lo, 0) 采样负向<br>**动机**：任务1结论——正向非线性（α>0）的破坏力是负向的 3.4 倍，非对称采样使训练"多看到"更强破坏场景。 |

三种方案按组合方式形成三组消融实验：
- **Exp1**：仅方案A `Exp1_CalibOnly`
- **Exp2**：方案A + B `Exp2_Calib+Layerwise`
- **Exp3**：方案A + B + C `Exp3_FullRobust`（完整鲁棒方案）

---

### 运行命令

```bash
# ====== 推荐：一键串联运行全部消融实验（约 75-90 分钟，3 × 120 epochs） ======
python task3_robust.py --exp all

# ====== 单独运行某个消融配置 ======
python task3_robust.py --exp 1      # 仅方案A（架构增强：残差校准模块）
python task3_robust.py --exp 2      # 方案A + B（+ 分层加权 α 采样）
python task3_robust.py --exp 3      # 完整方案 A + B + C（+ 非对称 70%正向/30%负向）

# ====== 自定义参数（更快或更鲁棒） ======
# 仅跑完整方案，100 epochs + 稍低 lr + 更大 batch size
python task3_robust.py --exp 3 --epochs 100 --lr 0.005 --batch_size 256

# 启用早停，连续 15 epoch 干净准确率不提升即停止
python task3_robust.py --exp all --patience 15
```

---

### 消融实验配置映射（`--exp` 到三个布尔开关）

| `--exp` | 实验名称 | use_calibration（方案A） | layerwise_alpha（方案B） | asymmetric_sampling（方案C） |
| :----: | :--- | :---: | :---: | :---: |
| 1 | `Exp1_CalibOnly` | ✓ | ✗ | ✗ |
| 2 | `Exp2_Calib+Layerwise` | ✓ | ✓ | ✗ |
| 3 | `Exp3_FullRobust` | ✓ | ✓ | ✓ |
| all | 依次运行 Exp1 → Exp2 → Exp3，并调用 `generate_ablation_summary()` 生成汇总 | — | — | — |

---

### 关键实现说明

1. **RobustCNN（models/robust_cnn.py）**
   - FeatureCalibration 参数量 16,640（128×64 + 64 + 64×128 + 128）
   - `use_calibration=False` 时完全退化为与 SimpleCNN 等价结构（作为消融基线对比）

2. **TrainerRobust（task3_robust.py）**
   - `_sample_alpha(module_name=...)`：按模块名返回不同 α；非对称采样在 `random.random() < 0.7` 时取正向。
   - `_train_one_epoch`：**对每个 Conv2d/Linear 模块独立采样 α，独立挂 pre-hook**（闭包 `make_hook(alpha_val=alpha)` 避免 Python 后期绑定），所有钩子在 `try/finally` 中保证移除。
   - `_evaluate` 始终在干净数据上运行（α=0），最佳模型也基于 clean accuracy 保存。

3. **消融汇总（仅 `--exp all`）**
   - `ablation_summary.csv`：列 `exp_name, use_calibration, layerwise_alpha, asymmetric_sampling, clean_acc, acc_at_alpha_0.3, acc_drop_at_0.3`
   - `ablation_comparison.png`：4 条曲线同轴对比——Exp1/Exp2/Exp3（彩色实线，不同 marker）+ NAT-Scratch 基线（红色虚线）
   - 终端打印 ASCII 表格，直观看到每加一项方案对 CleanAcc / Acc@α=0.3 / Drop@0.3 的边际改善

---

### 输出文件说明

| 路径（`./outputs/task3_simplecnn/` 下） | 含义 |
| :--- | :--- |
| `Exp1_CalibOnly/` `Exp2_Calib+Layerwise/` `Exp3_FullRobust/` | 每个实验独立子目录，结构如下： |
| · `training_curves.png` | 2 行子图 Loss / Accuracy 曲线；红星标出 clean acc 最佳 Epoch（标题/坐标轴全英文） |
| · `metrics.json` | 含 `best_test_acc` / `best_epoch` 以及**三个消融开关字段**：`use_calibration / layerwise_alpha / asymmetric_sampling / exp_name` |
| · `confusion_matrix.png` | 10×10 混淆矩阵（类别名 airplane...truck，全英文） |
| · `alpha_sensitivity.csv` | α ∈ [-0.3, 0.3] 扫描结果；列：`alpha, accuracy, loss` |
| · `accuracy_vs_alpha_comparison.png` | 当前实验（绿色实线+圆圈）vs **NAT-Scratch 基线**（蓝色虚线+方块） |
| **根目录**：`ablation_summary.csv` | （仅 `--exp all`）消融实验汇总表 |
| **根目录**：`ablation_comparison.png` | （仅 `--exp all`）Exp1/Exp2/Exp3 + NAT-Scratch 4 条曲线同图对比 |

对应 Checkpoints：
- `./checkpoints/Exp1_CalibOnly/best_model.pth`（dict 含 epoch/model/optim/best_acc/三个开关/exp_name）
- `./checkpoints/Exp2_Calib+Layerwise/best_model.pth`
- `./checkpoints/Exp3_FullRobust/best_model.pth`

---

### 代码模块说明

| 文件 | 核心类 / 函数 | 作用 |
| :--- | :--- | :--- |
| `models/robust_cnn.py` | **`FeatureCalibration`** | 残差校准 MLP（128→64→128 + 残差连接） |
| `models/robust_cnn.py` | **`RobustCNN`** | 完整鲁棒 CNN；`use_calibration` 开关支持消融对比 |
| `task3_robust.py` | `get_exp_config()` | `--exp` 字符串 → 实验配置（名称 + 三个布尔开关） |
| `task3_robust.py` | **`TrainerRobust`** | 增强训练器：`_sample_alpha(module_name)` 分层采样 + 非对称偏置；每模块独立挂/摘钩 |
| `task3_robust.py` | `evaluate_alpha_scan()` | 训练后在 [-0.3, 0.3] 扫 α 并写 `alpha_sensitivity.csv` |
| `task3_robust.py` | `plot_comparison()` | 单实验 vs NAT-Scratch 基线的对比图 |
| `task3_robust.py` | `run_experiment(exp_id, args)` | 单个实验完整流程：配置→目录→数据→模型→TrainerRobust→α扫描→对比图→终端摘要 |
| `task3_robust.py` | `generate_ablation_summary()` | 消融汇总 CSV + 消融汇总 PNG + 终端 ASCII 表 |
| `task3_robust.py` | `parse_args()` + `main()` | argparse 驱动；`--exp all` 顺序跑三个实验再出汇总 |

> 图表提示：同之前两个任务，**所有生成图片的标题/坐标轴/图例全英文**；CSV 表头英文；终端输出用中文便于调试。

---

## 拓展研究2：高斯噪声与非线性失真对比

### 研究简介

拓展研究2系统对比**高斯加性噪声**（$x + \mathcal{N}(0, \sigma^2)$）与**三次多项式非线性失真**（$y = \alpha x^3 + (1-\alpha) x$）对模型推理精度及内部表征的**不同影响机制**，旨在回答以下三个核心问题：

1. **精度衰减模式对比**：两种扰动在不同强度下的精度下降曲线是否具有相同形状？哪一种扰动在"可比精度跌幅"下更难防御？
2. **层误差累积路径对比**：两种扰动引起的特征偏移（RME & 余弦相似度）在 conv1→conv2→conv3→conv4→fc 的传播路径上有何差异？最脆弱层是否一致？
3. **鲁棒性迁移性验证**：任务3 Exp2 模型（仅针对非线性失真训练的鲁棒模型，`Exp2_Calib+Layerwise`）是否对**完全未见**的高斯加性噪声也具备跨域泛化的防护效果？

---

### 扰动数学模型对比

| 扰动类型 | 数学公式 | 参数 | 参数物理含义 |
| :---: | :--- | :---: | :--- |
| **非线性失真**（Nonlinearity） | $y = \alpha \cdot \tilde{x}^3 + (1-\alpha)\cdot\tilde{x}$，$\tilde{x}=x/\|x\|_{\max}$（逐样本动态归一化） | $\alpha$ | 三次非线性增益：<br>$\alpha>0$：大信号被放大（饱和型）<br>$\alpha<0$：大信号被压缩（截止型）<br>$\alpha=0$：理想线性 |
| **高斯噪声**（Gaussian Noise） | $y = x + \epsilon$，$\epsilon \sim \mathcal{N}(0, \sigma^2)$ | $\sigma$ | 噪声标准差：<br>$\sigma=0$：无噪声<br>$\sigma$ 越大，加性 SNR 越低 |

> **关键区别**：非线性失真是**输入相关（input-dependent）**且**确定性**的映射——相同输入 $x$ 始终产生相同失真；高斯噪声是**输入无关（input-independent）**且**随机性**的加性扰动——相同 $x$ 每次得到不同噪声。

---

### 运行命令示例

```bash
# ====== 默认：完整运行（两个模型 + 两种扰动全扫描，约 15-20 分钟） ======
#   --models simple_cnn,exp2          : 干净模型(84.90%) + 任务3最优模型(87.37%)
#   --pert_types nonlinearity,gaussian: 两种扰动都执行
#   --alpha_values -0.3...0.3          : α ∈ [-0.3, 0.3] 步长 0.1
#   --noise_levels 0.05...0.30         : σ ∈ [0.05, 0.30] 步长 0.05
python task_extension2_noise_vs_nonlinearity.py

# ====== 仅分析干净模型（更快） ======
python task_extension2_noise_vs_nonlinearity.py --models simple_cnn

# ====== 自定义噪声/失真扫描范围 ======
python task_extension2_noise_vs_nonlinearity.py \
    --models simple_cnn \
    --noise_levels "0.1,0.2,0.3,0.5" \
    --alpha_values "-0.5,-0.3,0.0,0.3,0.5"

# ====== 指定设备 / Batch Size / 层分析样本数 ======
python task_extension2_noise_vs_nonlinearity.py \
    --device cuda \
    --batch_size 256 \
    --num_samples 1000
```

完整参数列表（`python task_extension2_noise_vs_nonlinearity.py --help`）：

| 参数 | 类型 | 默认值 | 说明 |
| --- | :---: | :--- | --- |
| `--models` | str | `"simple_cnn,exp2"` | 逗号分隔模型列表，可选 `simple_cnn`（干净）/ `exp2`（任务3 Exp2） |
| `--pert_types` | str | `"nonlinearity,gaussian"` | 扰动类型，`nonlinearity`（三次失真）/ `gaussian`（高斯噪声） |
| `--alpha_values` | str | `"-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3"` | 非线性 α 扫描范围（逗号分隔） |
| `--noise_levels` | str | `"0.05,0.10,0.15,0.20,0.25,0.30"` | 高斯 σ 扫描范围（逗号分隔，脚本内部会自动补 σ=0.0 作为 clean 参考） |
| `--num_samples` | int | `500` | 层分析固定样本数（取 test_loader 前 N 张，所有模型/扰动共享） |
| `--batch_size` | int | `128` | 推理/扫描 Batch Size |
| `--device` | str | 自动 | `cuda` / `cpu`（默认有 GPU 用 GPU） |
| `--output_dir` | str | `./outputs/extension2_simplecnn` | 结果输出根目录 |
| `--seed` | int | `42` | 随机种子（保证可复现） |

---

### 脚本主流程（6 Steps）

```
Step 0: 准备
  └─ 解析参数 → 设置随机种子 → 创建输出目录 → 打印完整配置摘要

Step 1: 加载数据 & 固定样本
  ├─ get_dataloaders() 拿到 test_loader（10,000 张）
  └─ collect_fixed_samples() 取前 500 张 → fixed_imgs / fixed_lbls（所有分析共享）

Step 2: 加载模型（get_model_for_ext2）
  ├─ simple_cnn → SimpleCNN + checkpoints/simple_cnn/best_model.pth (84.90%)
  └─ exp2       → RobustCNN(use_calibration=True) + checkpoints/Exp2_Calib+Layerwise/best_model.pth (87.37%)

Step 3: 遍历 模型 × 扰动类型 组合
  ├─ 子实验 A：整网精度扫描（全量 10,000 张 test set）
  │     对每个 α / σ：重载干净权重 → register_perturbation_hooks()
  │                    → evaluate_full() → remove_hooks() → 记录 (α/σ, Acc, Loss)
  │     输出：{model}/nonlinearity_sensitivity.csv  或  {model}/gaussian_sensitivity.csv
  │
  └─ 子实验 B：层输出分布偏移分析（固定 500 张样本）
        · 先取无扰动 (α=0 / σ=0) 基线层输出 clean_layer_outputs{}
        · 对每个扰动强度：注册扰动钩子 + 层捕获钩子 → 推理 fixed_imgs
          → 每层 compute_metrics_pair() 算 RME / 余弦相似度
        · 输出：
            - layer_shift_nonlinearity.csv / layer_shift_gaussian.csv
            - error_accumulation_nonlinearity.png / error_accumulation_gaussian.png
            - histogram_conv1_gaussian.png / histogram_conv4_gaussian.png（σ=0 vs σ=0.20）

Step 4: 跨扰动类型对比（每个 model 各两张图）
  ├─ 图1 accuracy_comparison_{model}.png
  │     同一张图：下横轴 α（红实线圆圈），上横轴 σ（蓝虚线方块），双横轴共用 Accuracy 纵轴；
  │     红色星号标注 α=0 / σ=0 公共干净基线点。
  └─ 图2 error_accumulation_comparison_{model}.png
        选取"可比强度"（默认非线性 α=0.2 vs 高斯 σ=0.15，两者精度接近约 73%），
        各层余弦相似度两条曲线同图对比，观察层累积模式差异。

Step 5: 跨模型鲁棒性迁移对比（仅当 models 同时含 simple_cnn + exp2）
  └─ 图3 robustness_transfer_comparison.png（双列子图共享 y 轴）：
     左列 Nonlinearity：SimpleCNN 蓝虚线 / Exp2 红实线
     右列 Gaussian：    SimpleCNN 蓝虚线 / Exp2 红实线
     直观判断 Exp2 针对非线性学到的鲁棒性是否泛化到高斯噪声。

Step 6: 终端中文摘要
  └─ 打印：各模型干净基线 + 最大精度跌幅、两种扰动的精度对比表、
           层误差累积最敏感层对比、Exp2 高斯噪声下的迁移 uplift 表格、
           所有输出文件绝对路径列表。
```

---

### 输出文件说明（`./outputs/extension2_simplecnn/`）

#### 目录结构
```
outputs/extension2_simplecnn/
├── simple_cnn/                          # 干净模型（SimpleCNN）子目录
│   ├── nonlinearity_sensitivity.csv     # α ∈ [-0.3, 0.3] 全量精度扫描；列：alpha, accuracy, loss
│   ├── gaussian_sensitivity.csv         # σ ∈ [0.00, 0.30] 全量精度扫描；列：noise_std, accuracy, loss
│   ├── layer_shift_nonlinearity.csv     # 非线性层偏移：alpha, layer_name, RME, cosine_similarity
│   ├── layer_shift_gaussian.csv         # 高斯层偏移：noise_std, layer_name, RME, cosine_similarity
│   ├── error_accumulation_nonlinearity.png   # 各 α 下误差跨层累积曲线
│   ├── error_accumulation_gaussian.png       # 各 σ 下误差跨层累积曲线
│   ├── histogram_conv1_gaussian.png          # conv1 σ=0 vs σ=0.20 分布直方图对比
│   ├── histogram_conv4_gaussian.png          # conv4 σ=0 vs σ=0.20 分布直方图对比
│   ├── accuracy_comparison_simple_cnn.png    # 同一模型 α vs σ 精度曲线对比（双横轴）
│   └── error_accumulation_comparison_simple_cnn.png  # 可比强度下跨层误差累积对比
│
├── exp2/                               # 任务3 Exp2 鲁棒模型子目录（结构同上）
│   └── （同 simple_cnn/ 的 10 个文件）
│
├── error_accumulation_comparison_simple_cnn.png
├── error_accumulation_comparison_exp2.png
└── robustness_transfer_comparison.png      # 四曲线鲁棒性迁移总图
```

#### 关键图表内容说明

| 文件名（典型） | 图中内容（全英文：标题/坐标轴/图例） |
| :--- | :--- |
| `{m}/error_accumulation_nonlinearity.png` | 横轴 Layer Index（conv1_output→fc_output），纵轴 Cosine Similarity with Clean Output；<br>多条曲线对应不同 α，coolwarm 颜色编码，$\alpha$ 越大颜色越暖。 |
| `{m}/error_accumulation_gaussian.png` | 同一坐标系统，多条曲线对应不同 σ，viridis 颜色编码。 |
| `{m}/histogram_conv1_gaussian.png` | conv1 输出分布 80 bins 密度直方图；<br>蓝半透明：Clean (σ=0.0)，红半透明：Noisy (σ=0.20)。 |
| `{m}/accuracy_comparison_{m}.png` | **双横轴**：<br>下轴 Nonlinearity Strength (α)（-0.3~0.3），上轴 Gaussian Noise STD (σ)（0.00~0.30，线性映射到同一展示空间）；<br>红线实线圆圈：Nonlinearity 曲线；蓝虚线方块：Gaussian 曲线；<br>红色星号标注 Clean Baseline = XX.XX%（α=0 / σ=0 公共起点）。 |
| `{m}/error_accumulation_comparison_{m}.png` | 同 Layer Index 横轴，两条曲线：<br>红实线圆圈：Nonlinearity α=+0.20；<br>蓝虚线方块：Gaussian σ=0.15；<br>直接目视对比两种扰动的层误差模式差异（如谁在 fc 层崩得更厉害）。 |
| `robustness_transfer_comparison.png` | **双列子图共享 y 轴**：<br>左 Nonlinearity：SimpleCNN（蓝虚线圆圈）vs Exp2（红实线圆圈）；<br>右 Gaussian：SimpleCNN（蓝虚线方块）vs Exp2（红实线方块）。<br>副标题统一为 Robustness Transfer: Model-specific vs Perturbation Type。 |

#### CSV 字段说明（全英文表头）

**1. `{model}/nonlinearity_sensitivity.csv`**
```csv
alpha, accuracy, loss
-0.30, 76.1200, 0.841234
 ...
 0.00, 84.9000, 0.501234
 ...
 0.30, 55.0800, 1.523456
```

**2. `{model}/gaussian_sensitivity.csv`**
```csv
noise_std, accuracy, loss
0.00, 84.9000, 0.501234
0.05, 82.1500, 0.612345
 ...
0.30, 59.8700, 1.312345
```

**3. `{model}/layer_shift_nonlinearity.csv`**
```csv
alpha, layer_name, RME, cosine_similarity
0.20, conv1_output, 0.00123456, 0.99876543
0.20, conv2_output, 0.01234567, 0.98765432
 ...
0.20, fc_output,    0.12345678, 0.89123456
```

---

### 代码模块说明

| 文件 | 核心类 / 函数 | 作用 |
| :--- | :--- | :--- |
| **`utils/perturbation.py`** | `gaussian_noise(x, std)` | 加性高斯噪声：`x + randn_like(x) * std`，`std=0` 直接短路返回原 Tensor。 |
| **`utils/perturbation.py`** | `inject_perturbation(x, pert_type, alpha, noise_std)` | 统一对外接口：`pert_type="nonlinearity"` → 调 `nonlinearity()`；`"gaussian"` → 调 `gaussian_noise()`；其余抛出 `ValueError`。 |
| **`utils/perturbation.py`** | `register_perturbation_hooks(model, pert_type, alpha, noise_std)` | 遍历所有 `nn.Conv2d` / `nn.Linear`，注册 `forward_pre_hook`；<br>内部使用工厂函数 `make_hook(p, a, s)` 显式捕获当前循环的三个快照参数，**彻底避免闭包延迟绑定陷阱**。<br>返回 hooks 句柄列表。 |
| **`utils/perturbation.py`** | `remove_hooks(hooks)` | 遍历句柄列表调用 `.remove()`；单个 hook 失败时 `try/except` 静默跳过，不影响全局清理。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `get_model_for_ext2(model_name, device)` | 根据模型字符串动态 import、加载权重；同时 `known_clean_acc` 记录已知干净基线准确率（若 checkpoint 自带 best_test_acc 则优先）。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `_reload_model_weights(model, model_name, device)` | 每个 α / σ 扫描开始前强制从磁盘重新加载干净权重，**避免任何 hook 残留或 BatchNorm running stats 污染基线**。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `evaluate_full(model, test_loader, criterion, device, pert_type, alpha, noise_std)` | 注册扰动钩子 → 全量 10,000 张推理 → `try/finally` 强制 `remove_hooks` → 返回 `(accuracy, avg_loss)`。与任务1脚本的 `evaluate_on_test()` 风格保持一致。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `run_global_sensitivity(...)` | 子实验 A：循环遍历 α 列表或 σ 列表，对每个强度调用 `evaluate_full`，最后写 CSV 返回 results list。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `_register_layer_capture_hooks(model, layer_out_dict)` | 对 `OBSERVE_LAYERS = [("conv1","conv1_output"), ..., ("fc","fc_output")]` 共 5 个模块挂 `forward_hook`，捕获输出到 `layer_out_dict[display_name] = detached_cpu_tensor`。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `compute_metrics_pair(clean_out, dist_out)` | RME 与余弦相似度计算：<br>RME = ‖mean(dist) − mean(clean)‖₂ / ‖mean(clean)‖₂（对全局标量均值）；<br>cos = 逐样本展平归一化后 cos_sim 的均值。与 `task1_layer_analysis.py` 口径完全一致，保证跨任务指标可比。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `run_layer_shift(...)` | 子实验 B：<br>① clean 基线推理 → `clean_layer_outputs{}`；<br>② 对每个扰动强度：`_reload_model_weights` → 扰动钩子 + 捕获钩子 → `try/finally` 双重 remove → 每层 `compute_metrics_pair` → 收集 RME/cos；<br>③ 写 `layer_shift_{type}.csv`；<br>④ 绘制 `error_accumulation_{type}.png`（Viridis / Coolwarm 色阶）；<br>⑤ 高斯场景额外画 `histogram_conv1/conv4_gaussian.png`（σ=0 vs σ=0.20）。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `plot_accuracy_comparison_one_model(...)` | Step 4 图1：`ax + ax.twiny()` 双横轴实现 α vs σ 精度曲线同图；σ∈[0,0.3] 线性映射到 α∈[-0.3,0.3] 展示空间，顶部 X 轴反向映射回 σ 原刻度标注。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `plot_layer_accumulation_comparison_one_model(...)` | Step 4 图2：自动寻找距目标 α=0.2 / σ=0.15 最近的已扫描参数，绘制同 Layer Index 横轴的两条误差累积曲线。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `plot_robustness_transfer(...)` | Step 5 图3：`plt.subplots(1, 2, sharey=True)` 双列子图，左 Nonlinearity + 右 Gaussian，SimpleCNN 蓝虚线 vs Exp2 红实线，`fig.suptitle()` 统加标题。 |
| **`task_extension2_noise_vs_nonlinearity.py`** | `parse_args()` + `main()` | argparse 驱动；Step 0~6 完整串联；终端输出**中文摘要表格**便于快速对比。 |

> 图表提示：拓展研究2 **所有生成图片的标题 / 坐标轴 / 图例均为英文**；CSV 表头为英文；终端日志与摘要使用中文便于调试阅读。所有钩子使用 `try/finally` 模式确保稳健释放；所有 α/σ 扫描起点强制重载干净权重，最大程度消除交叉污染。

---

## 拓展研究3：量化误差与非线性误差联合影响分析

### 研究简介

在存算一体芯片的真实部署中，模拟域 MAC（乘累加计算）输出后需经过 ADC 模数转换为数字信号，这一过程同时引入两种物理误差：
1. **非线性失真**（模拟域 MAC 固有特性，信号越大偏差越大）：$y = \alpha x^3 + (1-\alpha)x$；
2. **量化误差**（ADC 有限比特数导致的精度损失）：对称饱和 QDQ 范式。

**信号路径**：输入激活 → MAC → 非线性失真 → ADC（量化）→ 后续数字逻辑。
因此在误差注入时严格遵循"先非线性，后量化"的顺序（`inject_joint_error()` 两开关同时打开时的顺序保证）。

本拓展系统研究：
1. **量化误差单独影响**：8/6/4/3/2 bit 对 SimpleCNN / 任务3 Exp2 鲁棒模型的精度衰减；
2. **非线性 + 量化联合影响**：7 α × 5 bit = 35 组合网格 × 2 模型，分析是否存在**超加性叠加恶化**（联合跌幅 > 单独跌幅之和）；
3. **鲁棒模型迁移性**：任务3 Exp2（仅针对非线性训练的最优模型）在联合误差场景下是否仍保持鲁棒性增益。

**已有参考结论**：
- 任务1：SimpleCNN 干净 84.90%，α=+0.3 时 55.08%（-29.82%）
- 任务3：Exp2 干净 87.37%，α=+0.3 时 68.76%（-18.61%）→ 比 SimpleCNN 多保留 13.68% 绝对精度
- 拓展1：ResNet-18 更深但更脆弱（α=+0.3 仅 17.08%）
- 拓展2：高斯噪声与非线性机制不同域，鲁棒性不跨域迁移
- **拓展3 开放问题**：Exp2 的鲁棒性能不能在"非线性 + 量化 ADC"联合压力下保持？是否存在超加性叠加放大？

---

### 模块1：`utils/quantization.py` — 量化 + 联合误差注入

对外四层 API：

| 函数 | 说明 |
| :--- | :--- |
| **`quantize_error(x, num_bits=8)`** | 对称饱和 QDQ 量化-反量化：<br>① `qmin=-2^(nb-1), qmax=2^(nb-1)-1`；<br>② `scale[i]=max|x[i,:]| / qmax`（逐样本独立缩放）；<br>③ `x_q=round(x/scale).clamp(qmin,qmax) → x_deq=x_q*scale`；<br>④ 短路：`num_bits>=32 or None` 直接返回 x（无误差）。<br>返回 float32 反量化张量，`x_deq - x` 即隐式量化误差。 |
| **`inject_joint_error(x, alpha, num_bits, apply_nonlinearity, apply_quantization)`** | 统一联合入口：两开关独立切换，**若同时开则先 nonlinearity → 再 quantize_error**（对应 MAC→ADC 路径） |
| **`register_joint_error_hooks(model, alpha, num_bits, apply_nonlinearity, apply_quantization)`** | 遍历所有 `nn.Conv2d / nn.Linear`，用**闭包工厂函数 `make_hook(alp, nb, an, aq)`** 快照循环参数，注册 `forward_pre_hook`；返回钩子句柄列表，完全避免 Python 延迟绑定陷阱 |
| **`remove_hooks(hooks)`** | 逐一 `hook.remove()`，`try/except` 静默跳过单个失败 |

**已冒烟验证**：对 SimpleCNN 4 Conv + 1 FC 共 5 层，`register_joint_error_hooks()` 返回 5 个句柄；移除后前向 shape 正常；量化误差数值随 bit 数下降单调递增（nb=32→0, 8→0.0205, 4→0.371, 2→2.146 MAE）。

---

### 运行命令

```bash
# ====== 默认：两模型（SimpleCNN + Exp2）+ 全扫描（约 20-30 分钟，GPU）======
#   --models simple_cnn,exp2
#   --alpha_values -0.3,-0.2,-0.1,0.0,0.1,0.2,0.3        # 7 个 α
#   --num_bits_list 8,6,4,3,2                            # 5 个 bit
#   合计：2 × (5 量化独扫 + 7×5 联合扫描) = 2 × 40 = 80 次 10K 全量推理
python task_extension3_joint_analysis.py

# ====== 快速测试（仅 SimpleCNN + 少量参数）======
python task_extension3_joint_analysis.py \
    --models "simple_cnn" \
    --alpha_values "-0.3,0.0,0.3" \
    --num_bits_list "8,4,2"

# ====== 自定义对比 target_bit（图3/图4 的固定 bit，默认 4）======
python task_extension3_joint_analysis.py \
    --models "simple_cnn,exp2" \
    --target_bit 3

# ====== 指定设备 / Batch Size / 输出目录 ======
python task_extension3_joint_analysis.py \
    --device cuda --batch_size 256 --output_dir ./outputs/extension3_simplecnn
```

**argparse 参数表**（`task_extension3_joint_analysis.py -h`）：

| 参数 | 类型 | 默认值 | 说明 |
| --- | :---: | :--- | --- |
| `--models` | str | `"simple_cnn,exp2"` | 逗号分隔模型列表（支持 simple_cnn / exp2 / resnet18） |
| `--alpha_values` | str | `"-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3"` | 非线性强度 α 扫描范围（自动前置插入 0.0 若缺） |
| `--num_bits_list` | str | `"8,6,4,3,2"` | ADC 量化比特数扫描范围 |
| `--batch_size` | int | 128 | 推理 batch size |
| `--device` | str | 自动 | `cuda` / `cpu`（默认有 CUDA 用 CUDA） |
| `--output_dir` | str | `./outputs/extension3_simplecnn` | 输出目录 |
| `--seed` | int | 42 | 随机种子 |
| `--target_bit` | int | 4 | 图3（联合 vs 单独）与图4（鲁棒性对比）的固定 bit（中等强度） |

---

### 脚本主流程（`task_extension3_joint_analysis.py` 7 Steps）

```
Step 0：准备
  ├─ parse_args() + 校验 + set_seed(42) + os.makedirs(output_dir)
  ├─ models / alphas / bits 解析为列表
  └─ 打印完整中文配置摘要（含组合计数）

Step 1：加载数据
  └─ get_dataloaders() → test_loader（10K 张，只取 test）；CrossEntropyLoss 实例化

Step 2：模型加载函数 load_model_for_ext3(model_name, device)
  ├─ simple_cnn → SimpleCNN + checkpoints/simple_cnn/best_model.pth
  ├─ exp2 → RobustCNN(use_calibration=True) + checkpoints/Exp2_Calib+Layerwise/best_model.pth
  ├─ resnet18 → ResNet18 + checkpoints/resnet18/best_model.pth（可选）
  ├─ 兼容纯 state_dict / Trainer dict（含 best_test_acc 字段打印）
  └─ 返回 (model, clean_state_dict_CPU_deepcopy, known_clean_acc)
  · 技巧：启动时为每个模型一次性磁盘加载 checkpoint 并保存 CPU 深拷贝快照，
         后续每个 (α,bit) 组合从内存快照直接 load_state_dict 回设备，避免大量重复磁盘 IO

Step 3：单独量化误差扫描
  对每个模型 run_quantization_only()：
    对每个 bit ∈ num_bits_list：
      a) model.load_state_dict(clean_state_dict)  ← 强制重载
      b) hooks = register_joint_error_hooks(α=0, bit, apply_nonlinearity=NO, quant=YES)
      c) try: evaluate_on_test() → finally: remove_hooks(hooks)
      d) 记录 (nb, acc, loss) 并打印
  ├─ 保存 {model}_quantization_only.csv（列：num_bits, accuracy, loss）
  └─ 终端输出模型×bit 的二维 ASCII 表

Step 4：联合误差扫描（α × bit 网格）
  对每个模型 run_joint_error()：
    对 α ∈ alpha_list：
      对 bit ∈ num_bits_list：
        a) load_state_dict 重载
        b) hooks = register_joint_error_hooks(α, bit, 两开关都 YES，顺序 nonlin→quant)
        c) try: evaluate_on_test → finally remove_hooks
        d) 记录 (α, bit, acc, loss)
        e) 对 α ∈ {±0.3,0.0} ∧ bit ∈ {8,4,2} 实时打印标志性组合
  ├─ 保存 {model}_joint_error.csv（列：alpha, num_bits, accuracy, loss）
  └─ 每个模型 7×5=35 组合 × 2 模型 = 70 次 10K 全量推理（核心耗时部分）

Step 5：可视化（4 类图）
  ├─ 图1 quantization_only_{model}.png   （每模型一张）
  ├─ 图2 joint_error_heatmap_{model}.png （每模型一张）
  ├─ 图3 joint_vs_alone_comparison.png   （1×N 子图，固定 target_bit）
  └─ 图4 robustness_under_joint_error.png（所有模型同轴，固定 target_bit）

Step 6：汇总表 joint_error_summary.csv
  三种类别行合并（error_type ∈ {nonlinearity_only, quantization_only, joint}）
  列：model_name, alpha, num_bits, accuracy, loss, error_type
  · nonlinearity_only：优先读取 task1 baseline / task3 Exp2 baseline；否则回退到 joint 的最大 bit
  · quantization_only：直接复制 Step3 输出
  · joint：直接复制 Step4 输出

Step 7：终端中文摘要（7 条 Section）
  1) 单独量化误差 ASCII 表（每 bit 的精度+相对 clean drop）
  2) 联合误差 (α=+0.3, bit=target_bit) 对比表（非线性独/量化独/Linear期望/联合实际/联合跌幅）
  3) 超加性效应判断：联合跌幅 vs 单独跌幅之和，三档分类 [Super-Additive (恶化) / Linear / Sub-Additive (抵消)]
  4) Exp2 鲁棒模型在联合误差下相对 SimpleCNN 的绝对增益
  5) 列出所有输出文件的绝对路径
```

---

### 输出文件说明（`./outputs/extension3_simplecnn/`）

| 文件名（典型） | 内容 & 格式 |
| :--- | :--- |
| `simple_cnn_quantization_only.csv` | SimpleCNN 单独量化扫描；列 `num_bits, accuracy, loss` |
| `exp2_quantization_only.csv` | Exp2 单独量化扫描；结构同上 |
| `simple_cnn_joint_error.csv` | SimpleCNN 联合 α × bit 网格；列 `alpha, num_bits, accuracy, loss` |
| `exp2_joint_error.csv` | Exp2 联合 α × bit 网格；结构同上 |
| **`quantization_only_simple_cnn.png`** / `quantization_only_exp2.png` | **单独量化精度曲线**。X 轴 Number of Bits（扫描到的 bit + 32bit 参考点），Y 轴 Test Accuracy (%)；红色虚线水平标注 32-bit 无量化基线，红色五角星标注 (32, clean_acc) 参考点。 |
| **`joint_error_heatmap_simple_cnn.png`** / `joint_error_heatmap_exp2.png` | **联合误差热力图**（`matplotlib.imshow + colorbar`）。X 轴 Nonlinearity Strength (α)，Y 轴 Number of Bits（从上到下 8→2，高位在上，低位在下视觉更退化）；颜色 RdYlGn 色阶映射 Test Accuracy 0-100%；**每个单元格中心加粗文字**标注具体精度数值；高值白字、低值黑字保证可读性。 |
| **`joint_vs_alone_comparison.png`** | **1×N 子图**固定 target_bit 联合 vs 单独对比：SimpleCNN → 第 0 幅，Exp2 → 第 1 幅。每条曲线：<br>• Nonlinearity Only（32-bit）：蓝实线圆圈；<br>• Quantization Only（target-bit）：绿色水平虚线；<br>• Joint（两误差叠加）：红实线方块。<br>整体标题：`Joint vs Individual Error Impact (bit=target_bit)`。 |
| **`robustness_under_joint_error.png`** | **所有模型同轴鲁棒性对比**。SimpleCNN 蓝虚线圆圈 / Exp2 红实线方块 / ResNet-18 绿点划线上三角。α=0 清洁基线带黑边大五角星标注；灰色虚线 α=0 处。 |
| `joint_error_summary.csv` | 三类误差汇总，便于 pandas 二次分析；列 `model_name / alpha / num_bits / accuracy / loss / error_type`（error_type ∈ {nonlinearity_only, quantization_only, joint}）。 |

---

### 超加性效应判断规则（摘要 Section 3 自动输出）

设 clean = 32-bit & α=0 精度；`drop_n` = 单独非线性跌幅 = `clean - acc_nonlinear_only`；`drop_q` = 单独量化跌幅 = `clean - acc_quant_only`；`drop_joint` = 联合实际跌幅 = `clean - acc_joint`。脚本按 5% 阈值分三档：

| 条件 | 输出标签 | 物理解释 |
| :--- | :---: | :--- |
| `drop_joint > 1.05 × (drop_n + drop_q)` | **Super-Additive (恶化)** | 两种误差相互"共振"，联合破坏大于单独之和，需同时防御（最坏情况设计）。 |
| `0.95 × (drop_n + drop_q) ≤ drop_joint ≤ 1.05 × (drop_n + drop_q)` | **Linear** | 线性叠加，设计时可独立分析后求和即可。 |
| `drop_joint < 0.95 × (drop_n + drop_q)` | **Sub-Additive (抵消)** | 一种误差的主导破坏带因另一种误差饱和效应而相互抑制，实际比预期好。 |

---

## 拓展研究1：网络结构与参数量对非线性误差的影响

### 研究简介

拓展研究1在同一非线性失真设定下（`y = αx³ + (1-α)x`，整网 Conv2d/Linear 输入前 hook 注入），系统对比**三种**在**参数量、深度、结构拓扑（普通 CNN vs 残差网络 vs 纯前馈深层 VGG）**三个维度上都有明显差异的网络，全面检验"结构鲁棒性"：

| 模型 | 结构类型 | 深度 | 参数量 | 干净精度（典型） |
| :--- | :--- | :---: | :---: | :---: |
| **SimpleCNN** | Plain Conv + GAP + FC | 4 个 conv + 1 个 fc | **~0.242 M** | ~84.90%（已有 baseline） |
| **VGG-11** | 纯前馈深层 Conv + 3 FC | 8 个 conv + 3 个 fc | **~9.2 M** | ~92%（训练后） |
| **ResNet-18** | 4 组残差块（BasicBlock × 2 × 4）+ GAP + FC | 18 层（`conv1 + layer1-4 + fc`） | **~11.17 M**（约 SimpleCNN 的 46×） | ~92–95%（训练后） |

核心研究问题：
1. **更大参数量**的模型是否天然对非线性失真更鲁棒（"过参数化容错假设"）？
2. 不同的**网络结构**（普通 CNN / VGG 纯前馈深层 vs 残差连接）在**误差累积模式**上有何差异？ResNet 的 skip connection 能否缓解层间误差传播？VGG-11 作为"深度更大但无残差"的对照组，能否验证"无残差 + 深度 → 误差放大"猜想？
3. "过参数化冗余"与"更深结构导致误差累积放大" —— 哪个效应**占主导**？α=+0.3 下 SimpleCNN 跌 -29.82% 是基准线。

**三个模型的干净训练均使用同一训练配方（SGD + 动量 + CosineAnnealingLR，`train.py` 默认超参），保证公平对比**。

---

### 模块1：ResNet-18 CIFAR-10 适配（`models/resnet.py`）

基于 torchvision 标准 ResNet-18 拓扑，针对 CIFAR-10 **32×32 小输入**做三处关键适配：

| 部件 | ImageNet 版 ResNet-18 | **CIFAR-10 适配版（本项目）** |
| :--- | :--- | :--- |
| `conv1` | `Conv2d(3,64, 7×7, stride=2, pad=3)` + BN | `Conv2d(3,64, 3×3, stride=1, pad=1)` + BN |
| `maxpool` | `MaxPool2d(3×3, stride=2, pad=1)` | **直接去掉**，避免 32×32 过度下采样 |
| `layer1~4` | BasicBlock × 2，planes=[64,128,256,512]，stride=[1,2,2,2] | **完全一致**，空间尺寸保持：32→32→16→8→4 |
| `avgpool` | AdaptiveAvgPool2d(1×1) | 相同 |
| `fc` | Linear(512→1000) | Linear(512→10) |

> 属性名严格对齐 torchvision：`self.conv1 / self.bn1 / self.layer1 / self.layer2 / self.layer3 / self.layer4 / self.avgpool / self.fc` —— 未来想加载 ImageNet 预训练权重也能自然匹配。

**BasicBlock 结构（本项目独立实现，不依赖 torchvision.models）**：
```
x → Conv3×3(ch→ch, stride) → BN → ReLU
  → Conv3×3(ch→ch, stride=1) → BN
  → ( + shortcut ) → ReLU → out
shortcut 规则：stride==1 && in_ch==out_ch → identity；否则 1×1 Conv-BN
```
ResNet-18 即 `num_blocks=[2,2,2,2]`、`planes=[64,128,256,512]`。

**已验证参数量**：SimpleCNN 242,474 ≈ 0.242 M；ResNet-18 11,173,962 ≈ 11.17 M；32×32 输入前向 shape 正确 (B,3,32,32) → (B,10)。

---

### 模块2：现有脚本 get_model 扩展（4 个文件）

所有已有脚本的 `get_model()` 函数新增 `resnet18` 分支，保持 SimpleCNN 流程完全不变：

| 修改文件 | 说明 |
| :--- | :--- |
| `train.py` | 支持 `python train.py --model resnet18 --epochs 100 --lr 0.01` 训练干净 ResNet-18；Checkpoint 保存到 `./checkpoints/resnet18/` |
| `evaluate.py` | 支持 `python evaluate.py --model resnet18 --checkpoint <path>` 单独评估 ResNet-18 |
| `task1_sensitivity_analysis.py` | 支持 `--model resnet18` 对 ResNet-18 跑整网 α 敏感性 |
| `task1_layer_analysis.py` | **两处修改**：① `get_model()` 加 resnet18；② 新增 `OBSERVE_LAYERS_RESNET18 = [("layer1","layer1_output"), ..., ("fc","fc_output")]` 共 5 个观测点（与 SimpleCNN 的 conv1/2/3/4/fc 数量对齐，便于后续误差累积对比）；`register_layer_capture_hooks(model, model_name)` 根据传入模型名自动切换对应观测层列表；`main()` 中所有对 `register_layer_capture_hooks()` 的调用均追加 `args.model`；`OBSERVE_LAYERS_ORDER` 从固定列表改为 `get_observe_layers(args.model)` 动态获取。**向后兼容**：`--model simple_cnn` 行为与原脚本完全一致，不会破坏任务1的可复现性。 |

---

### 运行命令

```bash
# ====== 第一步：训练干净 ResNet-18（必须先完成，GPU 约 30-40 分钟） ======
# 训练配方与 SimpleCNN 完全一致：SGD + momentum 0.9 + weight decay 1e-4 + CosineAnnealingLR
python train.py --model resnet18 --epochs 100 --lr 0.01 --batch_size 128
# → 权重自动保存为 ./checkpoints/resnet18/best_model.pth

# ====== 第二步：运行拓展研究1 对比分析（GPU 约 10-15 分钟） ======
# 默认：同时对比 simple_cnn（checkpoints/simple_cnn/best_model.pth）
#              与   resnet18  （checkpoints/resnet18/best_model.pth）
python task_extension1_network_comparison.py

# ====== 自定义：仅对比单个模型 ======
python task_extension1_network_comparison.py \
    --models "resnet18" \
    --checkpoints "./checkpoints/resnet18/best_model.pth"

# ====== 自定义扫描范围 & 设备 ======
python task_extension1_network_comparison.py \
    --alpha_values "-0.5,-0.3,-0.1,0.0,0.1,0.3,0.5" \
    --device cuda --batch_size 256 --seed 42

# ====== 带 NAT 模型的三模型对比（需先训练好 NAT 权重） ======
python task_extension1_network_comparison.py \
    --models "simple_cnn,resnet18,vgg11" \
    --checkpoints "./checkpoints/simple_cnn/best_model.pth,./checkpoints/resnet18/best_model.pth,./checkpoints/vgg11/best_model.pth" \
    --nat_models "simple_cnn,resnet18" \
    --nat_checkpoints "./checkpoints/nat_scratch_simplecnn/best_model.pth,./checkpoints/nat_scratch_resnet18/best_model.pth" \
    --output_dir ./outputs/extension1
```

**argparse 参数表**（`task_extension1_network_comparison.py -h`）：

| 参数 | 类型 | 默认值 | 说明 |
| --- | :---: | :--- | --- |
| `--models` | str | `"simple_cnn,resnet18"` | 逗号分隔模型列表，顺序必须与 `--checkpoints` 一一对应 |
| `--checkpoints` | str | `"./checkpoints/simple_cnn/best_model.pth,./checkpoints/resnet18/best_model.pth"` | 逗号分隔 checkpoint 绝对/相对路径列表 |
| `--nat_models` | str | `""` | **（可选）** 逗号分隔 NAT 模型名称列表，如 `simple_cnn,resnet18`，与 `--nat_checkpoints` 一一对应 |
| `--nat_checkpoints` | str | `""` | **（可选）** 逗号分隔 NAT 权重路径列表，对应 `--nat_models`；某权重不存在时自动打印警告并跳过，不影响其他模型 |
| `--alpha_values` | str | `"-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3"` | α 扫描范围；若不含 0.0 则自动前置插入 |
| `--batch_size` | int | 128 | 推理 batch size |
| `--device` | str | 自动 | `cuda` / `cpu`，默认有 CUDA 用 CUDA |
| `--output_dir` | str | `./outputs/extension1` | 输出根目录 |
| `--seed` | int | 42 | 随机种子 |

---

### 脚本主流程（`task_extension1_network_comparison.py` 6 Steps）

```
Step 0：准备
  ├─ parse_args() 解析 + 校验 models/checkpoints 数量一致
  ├─ set_seed(args.seed) 保证可复现
  ├─ os.makedirs(output_dir)
  └─ 打印完整配置摘要（中文）

Step 1：加载数据
  └─ get_dataloaders() → test_loader（10,000 张，只取 test）

Step 2：模型加载函数 load_model_for_analysis(model_name, ckpt_path, device)
  ├─ get_model() 实例化 + 打印 total_params
  ├─ torch.load() 兼容两种格式：纯 state_dict / Trainer dict (含 model_state_dict/best_test_acc)
  └─ 返回 (model.to(device).eval(), total_params)

Step 3：逐个模型 α 扫描 —— run_alpha_sensitivity_one_model()
  ├─ 第一次加载 → 保存 clean_state_dict（CPU 深拷贝）
  ├─ 对每个 α：
  │     a) model.load_state_dict(clean_state_dict)  ← 每个 α 强制重载干净权重，杜绝污染
  │     b) hooks = register_nonlinearity_hooks(model, alpha)
  │     c) evaluate_on_test() 全量 10K 推理
  │     d) finally: remove_hooks(hooks)
  │     e) 记录 (α, acc, loss)
  ├─ 保存 {model_name}_alpha_sensitivity.csv
  │     列：alpha, accuracy, loss（与 task1 完全口径一致，可直接 diff）
  └─ 终端实时打印每个 α 的 acc/loss

Step 4：跨模型对比图 —— plot_network_comparison()（3 张）
  ├─ 图1 accuracy_comparison.png
  │     横轴 Nonlinearity Strength (α)，纵轴 Test Accuracy (%)
  │     SimpleCNN：蓝实线圆圈；ResNet-18：红实线方块
  │     红星单独标注每个模型的 α=0 clean baseline
  │
  ├─ 图2 accuracy_drop_comparison.png
  │     横轴 Positive α（仅 α>0，研究破坏性失真）
  │     纵轴 Accuracy Drop from Clean Baseline (%)
  │     直观比较谁在相同失真强度下"掉更多"
  │
  └─ 图3 robustness_vs_params.png（散点图）
        横轴 Model Parameters (M) 对数刻度（0.1–20 可读刻度）
        纵轴 Accuracy at α = +0.3 (%)
        两个大散点：蓝色 SimpleCNN / 红色 ResNet-18，加文字箭头顶点标注
        "参数量 × 鲁棒性"的直观象限对比

Step 5：保存 network_comparison_summary.csv
  列：model_name, total_params, clean_accuracy, acc_at_alpha_0.3, acc_drop_at_0.3

Step 6：终端中文摘要
  ├─ ① 模型参数 & 精度对比表（total_params / CleanAcc / Acc@0.3 / Drop@0.3）
  ├─ ② 每个模型各自的 α 扫描表（含 Drop vs Clean）
  ├─ ③ α=+0.3 跌幅 bullet 对比
  ├─ ④ 结论判断：参数量更大模型更鲁棒？→ 自动比较 drop_diff 并打印讨论
  │     drop_diff > +0.5% → 残差大模型更好；
  │     drop_diff < -0.5% → 更深结构放大误差；
  │     否则 → 差异不显著
  │     并给出物理原因提示（残差缓解误差累积 / 参数冗余容错）
  └─ ⑤ 遍历 os.walk() 打印所有输出文件的绝对路径
```

---

### 输出文件说明（`./outputs/extension1/`）

| 文件名 | 内容 & 格式 |
| :--- | :--- |
| `simple_cnn_alpha_sensitivity.csv` | SimpleCNN 的 α 扫描结果（若 task1 已有，可手工复制 `outputs/task1_simplecnn/alpha_sensitivity.csv` 覆盖跳过计算，脚本不重复跑）；列：`alpha, accuracy, loss` |
| `resnet18_alpha_sensitivity.csv` | ResNet-18 的 α 扫描结果；列同上。**核心对比文件**。 |
| `vgg11_alpha_sensitivity.csv` | VGG-11 的 α 扫描结果；列同上。**核心对比文件**。 |
| **`accuracy_comparison.png`** | **三模型 α-精度对比曲线**。SimpleCNN 蓝圆 / VGG-11 绿三角 / ResNet-18 红方；红色 × 星号标注每个模型的 α=0 Clean Baseline；X 轴列出所有扫描过的 α 刻度；灰虚线标记 α=0 位置。 |
| **`accuracy_drop_comparison.png`** | **相对跌幅对比曲线**（仅 α>0），消除"干净精度本身不同"的视觉干扰，重点关注"谁掉得更快"。 |
| **`robustness_vs_params.png`** | **参数量 vs 鲁棒性散点图**（X 对数刻度）。三个大圆 + 文字箭头，视觉判断 SimpleCNN 0.24M / VGG-11 9.2M / ResNet-18 11.17M 在 α=+0.3 的准确率象限分布。 |
| `network_comparison_summary.csv` | 一行一个模型的对比汇总，方便用 Excel / pandas 二次分析；列 `model_name / total_params / clean_accuracy / acc_at_alpha_0.3 / acc_drop_at_0.3`。 |

对应新增 checkpoints：`./checkpoints/resnet18/best_model.pth`、`./checkpoints/vgg11/best_model.pth`（与 `train.py` 默认保存路径一致，dict 格式含 `epoch/model_state_dict/optimizer_state_dict/best_test_acc`）。

---

### 关键注意事项

1. **训练 ResNet-18 的资源估算**：参数量 ~11M，单卡 2080Ti / 3090 级别 GPU 下 batch 128，100 epoch 约 30–40 分钟；若用 CPU 预计 >8 小时，强烈建议 GPU 训练。
2. **SimpleCNN 数据复用**：若 `outputs/task1_simplecnn/alpha_sensitivity.csv` 已存在，你可以**先把它复制到** `outputs/extension1/simple_cnn_alpha_sensitivity.csv` —— 然后在对比图中只跑 resnet18,vgg11，用 `--models resnet18,vgg11 --checkpoints ./checkpoints/resnet18/best_model.pth,./checkpoints/vgg11/best_model.pth`，最后再手工用同样的画图代码合并，这样可省去 ~2 分钟的 SimpleCNN 重复扫描；当前脚本保持默认三模型都跑，不依赖外部文件存在，逻辑最稳健。
3. **`task1_layer_analysis.py` 任务1 可复现性保证**：原 `--model simple_cnn` 的注册钩子函数新增了 `model_name` 形参，默认使用 `get_observe_layers(args.model)`，等价于原来的 `OBSERVE_LAYERS_SIMPLECNN`；`alpha_list / CSV / 绘图风格` 完全没变；`task1_sensitivity_analysis.py` 仅新增 elif 分支；**任务1基线不应发生任何数值漂移**。
4. **ResNet-18 观测层映射**：`layer1~layer4` 是 `nn.Sequential` 容器，`register_forward_hook` 挂容器时，捕获的是**整个 stage 最后一个 BasicBlock 的输出**（64×32 / 128×16 / 258×8 / 512×4），正好对应 4 个深度层级；再加上 `fc_output`，合计 5 个观测点，与 SimpleCNN 的 5 个观测点天然对齐，方便横向比较"误差随深度累积的梯度"。
5. **所有钩子稳健释放**：`run_alpha_sensitivity_one_model()` 中每个 α 的推理都用 `try: ... finally: remove_hooks(hooks)`，即使某个 batch 中途 OOM 或异常也能保证钩子不残留；下一轮 `load_state_dict` 还会进一步重置参数状态。

---

## 批量运行全部模型

### 简介

项目支持对多个模型（SimpleCNN、ResNet-18、VGG-11）批量运行完整的非线性误差分析管线。
使用 `run_all_models.py` 脚本可一键完成所有分析任务，输出按模型名分别存放到**独立目录**，互不覆盖。

### 运行命令

```bash
# 基础运行（不含 NAT 训练，约 1-2 小时）
python run_all_models.py

# 包含 NAT 感知训练（耗时较长，约 3-4 小时）
python run_all_models.py --include_nat

# 自定义 NAT 训练 epoch 数
python run_all_models.py --include_nat --nat_epochs 80
```

### 管线顺序

对每个模型依次执行：
1. 任务1：α 敏感性扫描
2. 任务1：层分布偏移分析
3. 拓展2：高斯噪声 vs 非线性对比
4. 拓展3：量化 + 非线性联合分析
5. 任务2：NAT 感知训练（可选，默认跳过，通过 `--include_nat` 启用）
6. 拓展1：三模型架构对比（所有模型跑完后统一执行）

### argparse 参数表（`run_all_models.py -h`）

| 参数 | 类型 | 默认值 | 说明 |
| --- | :---: | :--- | --- |
| `--include_nat` | flag | `False` | 是否包含 NAT 感知训练（默认跳过，启用后每个模型约耗时 40 分钟/120 epochs） |
| `--nat_epochs` | int | `120` | NAT 训练 epoch 数 |

### 输出目录结构

所有模型的输出按模型名称分别存放在独立目录中，互不覆盖：

```
outputs/
├── simple_cnn/                    # 原始训练输出
├── task1_simplecnn/               # SimpleCNN α敏感性 + 层偏移
├── task1_resnet18/                # ResNet-18 α敏感性 + 层偏移
├── task1_vgg11/                   # VGG-11 α敏感性 + 层偏移
├── task2_simplecnn/               # SimpleCNN NAT
├── task2_resnet18_scratch/        # ResNet-18 NAT (如果运行)
├── task2_vgg11_scratch/           # VGG-11 NAT (如果运行)
├── task3_simplecnn/               # SimpleCNN 鲁棒性消融
├── extension1/                    # 三模型架构对比
├── extension2_simplecnn/          # SimpleCNN 高斯噪声对比
├── extension2_resnet18/           # ResNet-18 高斯噪声对比
├── extension2_vgg11/              # VGG-11 高斯噪声对比
├── extension3_simplecnn/          # SimpleCNN 量化联合分析
├── extension3_resnet18/           # ResNet-18 量化联合分析
└── extension3_vgg11/              # VGG-11 量化联合分析
```

对应权重（checkpoints）保存路径也自动带上模型名后缀，避免覆盖：
- `./checkpoints/nat_scratch_simplecnn/`
- `./checkpoints/nat_finetune_simplecnn/`
- `./checkpoints/nat_scratch_resnet18/`
- `./checkpoints/nat_scratch_vgg11/`
- `./checkpoints/Exp1_CalibOnly_simplecnn/` ... 等任务3实验目录

---

## 许可证

MIT License © CIFAR-10 Project
