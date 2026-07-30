# 存算一体芯片中非线性误差对推理精度的影响研究

一个面向 CIFAR-10 图像分类的深度学习研究框架，用于系统性地分析存算一体(Compute-in-Memory, CiM)芯片中模拟域非线性失真、ADC 量化误差以及两者联合作用下的推理精度退化机制，并提出非线性感知训练(NAT)与架构级鲁棒性增强方案。内置 4 种模型架构(SimpleCNN / RobustCNN / ResNet-18 / VGG-11)，覆盖 3 项核心任务 + 3 项拓展研究，支持断点续训、丰富可视化与自动化消融汇总。

---

## 1. 项目简介

### 1.1 研究背景与核心问题

存算一体芯片通过在 SRAM/Flash 阵列内执行模拟域乘累加(MAC)运算，大幅降低数据搬运开销，是边缘端 AI 推理的高算力/低功耗方向。但模拟域计算天然存在器件失配、OTA 增益非线性等物理非理想效应，在 Conv2d / Linear 算子的输入侧引入三次多项式非线性失真：

$$y = \alpha \cdot \tilde{x}^3 + (1-\alpha) \cdot \tilde{x}, \quad \tilde{x} = \frac{x}{\|x\|_{\infty, \text{ per-sample}}}$$

其中 $\alpha$ 为非线性强度($\alpha=0$ 理想线性，$\alpha>0$ 正增益饱和，$\alpha<0$ 负增益截止)。本项目围绕以下 6 个研究问题展开：

1. **任务1**：不同 $\alpha$ 下整网精度如何衰减？误差如何跨层累积？哪一层最脆弱？
2. **任务2**：训练阶段"见过非线性"的 NAT 感知训练(微调 / 从头训练)能多大程度补偿失真？
3. **任务3**：在 NAT-Scratch 基础上，架构级残差校准 + 分层加权采样 + 非对称偏置三项增强的**消融验证；
4. **拓展1**：不同参数量/深度/拓扑(浅层CNN vs 纯前馈深层VGG vs 残差ResNet)对非线性失真的敏感性差异？
5. **拓展2**：输入相关的非线性失真 vs 输入无关的高斯加性噪声，两者在误差机制差异及鲁棒性迁移性？
6. **拓展3**：MAC 非线性 + ADC 量化**联合误差**是否存在超加性叠加恶化？

### 1.2 核心技术亮点

- **统一 Hook 注入接口**：所有误差/扰动均通过 `nn.Conv2d / nn.Linear` 的 `register_forward_pre_hook` 注入，所有钩子使用工厂闭包快照参数避免 Python 延迟绑定陷阱，所有推理循环采用 `try / finally` 保证钩子稳健释放；
- **逐样本动态归一化非线性**：对 batch 维每个样本独立做 $\infty$-范数归一化再施加非线性，避免 batch 间信号串扰；
- **逐样本动态 QDQ 量化**：ADC 仿真使用逐样本动态缩放的对称饱和整型量化-反量化，`num_bits >= 32 / None` 自动短路；
- **完整消融实验自动化**：任务3 `--exp all` 一键串联 Exp1/2/3 三个消融配置，自动生成 CSV + 对比图 + 终端配置表；
- **多模型架构对比原生支持**：所有任务/拓展的 `get_model()` 工厂支持 `simple_cnn / resnet18 / vgg11 / robust_cnn / exp2` 等多种名称，输出/权重路径根据模型名动态后缀，避免覆盖。

---

## 2. 项目结构

```
CIFAR10/
├── data/                                           # CIFAR-10 数据集(自动下载或手动放)
│   └── cifar-10-batches-py/
├── models/                                         # 模型定义
│   ├── __init__.py
│   ├── simple_cnn.py                               # SimpleCNN (~0.24M，基线小模型)
│   ├── robust_cnn.py                               # RobustCNN + FeatureCalibration 残差校准
│   ├── resnet.py                                   # CIFAR-10 适配版 ResNet-18 (~11.17M)
│   └── vgg11.py                                    # CIFAR-10 轻量适配版 VGG-11 (~9.2M)
├── utils/                                          # 通用工具模块
│   ├── __init__.py
│   ├── data_loader.py                              # CIFAR-10 数据加载 + 增强(root="./data" 硬编码)
│   ├── trainer.py                                  # 通用 Trainer(支持早停/断点续训/英文图表)
│   ├── nonlinearity.py                             # 三次非线性失真注入(任务1/2/3核心)
│   ├── perturbation.py                             # 统一扰动接口(非线性+高斯，拓展2)
│   └── quantization.py                             # QDQ 量化+联合误差注入(拓展3)
├── checkpoints/                                    # 训练权重(dict 格式含 epoch/model/optim/best_acc)
│   ├── {model_name}/                               # 干净训练权重：simple_cnn / resnet18 / vgg11
│   │   └── best_model.pth
│   ├── nat_{mode}_{model_tag}/                     # NAT 权重(mode ∈ {finetune, scratch})
│   │   └── best_model.pth
│   ├── Exp1_CalibOnly_{model_tag}/                 # 任务3 Exp1(仅方案A)
│   ├── Exp2_Calib+Layerwise_{model_tag}/           # 任务3 Exp2(方案A+B)
│   └── Exp3_FullRobust_{model_tag}/                # 任务3 Exp3(方案A+B+C)
├── outputs/
│   ├── {model_name}/                               # train.py 干净训练产物(曲线/混淆/指标)
│   ├── task1_{model_tag}/                          # 任务1：α敏感性+层偏移
│   │   ├── alpha_sensitivity.csv
│   │   ├── accuracy_vs_alpha.png
│   │   ├── layer_distribution_shift.csv
│   │   ├── error_accumulation.png
│   │   ├── histogram_{layer1}.png                  # (浅/深两个观测层)alpha=0 vs 0.3 直方图
│   │   ├── histogram_{layer2}.png
│   │   └── task1_{model_tag}_summary.md            # 任务1 总结文档
│   ├── task2_{model_tag}/                          # 任务2：NAT 训练产物
│   │   ├── finetune/
│   │   │   ├── training_curves.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── metrics.json
│   │   │   ├── alpha_sensitivity.csv
│   │   │   └── accuracy_vs_alpha_comparison.png
│   │   ├── scratch/                                # (同 finetune/ 结构)
│   │   └── task2_{model_tag}_summary.md            # 任务2 总结文档
│   ├── task3_{model_tag}/                          # 任务3：鲁棒性消融
│   │   ├── Exp1_CalibOnly/
│   │   ├── Exp2_Calib+Layerwise/
│   │   ├── Exp3_FullRobust/
│   │   ├── ablation_summary.csv                    # (仅 --exp all)
│   │   ├── ablation_comparison.png                 # (仅 --exp all)
│   │   └── task3_{model_tag}_summary.md            # 任务3 总结文档
│   ├── extension1/                                 # 拓展1：多模型架构对比
│   │   ├── {m}_alpha_sensitivity.csv               # 每个 clean 模型
│   │   ├── {m}_nat_alpha_sensitivity.csv           # (可选 NAT)
│   │   ├── accuracy_comparison.png
│   │   ├── accuracy_drop_comparison.png
│   │   ├── robustness_vs_params.png
│   │   ├── network_comparison_summary.csv
│   │   └── extension1_summary.md                   # 拓展1 总结文档
│   ├── extension2_{model_tag}/                     # 拓展2：高斯噪声 vs 非线性(默认带_simplecnn后缀)
│   │   ├── {model_instance}/                       # simple_cnn / exp2 / resnet18 / vgg11 子目录
│   │   │   ├── nonlinearity_sensitivity.csv
│   │   │   ├── gaussian_sensitivity.csv
│   │   │   ├── layer_shift_nonlinearity.csv
│   │   │   ├── layer_shift_gaussian.csv
│   │   │   ├── error_accumulation_nonlinearity.png
│   │   │   ├── error_accumulation_gaussian.png
│   │   │   ├── histogram_{*}_gaussian.png
│   │   │   ├── accuracy_comparison_{m}.png
│   │   │   └── error_accumulation_comparison_{m}.png
│   │   ├── robustness_transfer_comparison.png      # (simple_cnn + exp2 同时存在)
│   │   └── extension2_{model_tag}_summary.md       # 拓展2 总结文档
│   └── extension3_{model_tag}/                     # 拓展3：量化+非线性联合(默认带_simplecnn后缀)
│       ├── {m}_quantization_only.csv
│       ├── {m}_joint_error.csv
│       ├── quantization_only_{m}.png
│       ├── joint_error_heatmap_{m}.png
│       ├── joint_vs_alone_comparison.png
│       ├── robustness_under_joint_error.png
│       ├── joint_error_summary.csv
│       └── extension3_{model_tag}_summary.md       # 拓展3 总结文档
├── train.py
├── evaluate.py
├── task1_sensitivity_analysis.py
├── task1_layer_analysis.py
├── task2_nat.py
├── task3_robust.py
├── task_extension1_network_comparison.py
├── task_extension2_noise_vs_nonlinearity.py
├── task_extension3_joint_analysis.py
├── requirements.txt
├── 原始题目+题目解读+研究思路.md
└── README.md
```

> **路径命名规范**：`model_tag = args.model.replace("_", "")，例如 `simple_cnn → simplecnn`，`resnet18 → resnet18`。所有任务/拓展的输出/权重目录均带模型后缀，避免多模型并行时互相覆盖。

---

## 3. 数据集说明

- **数据集**：CIFAR-10(torchvision.datasets.CIFAR10，root="./data")
- **规模**：60,000 张 32×32 彩色图像，10 类(airplane / automobile / bird / cat / deer / dog / frog / horse / ship / truck)，每类 6,000 张
- **划分**：50,000 训练 + 10,000 测试
- **预处理流程**(`utils/data_loader.py`)：
  - 训练集：`RandomCrop(32, padding=4) → RandomHorizontalFlip(p=0.5) → ToTensor → Normalize`
  - 测试集：`ToTensor → Normalize`
  - 归一化参数：mean=(0.4914, 0.4822, 0.4465)，std=(0.2023, 0.1994, 0.2010)
  - DataLoader：batch_size 默认 128，训练集 shuffle=True + drop_last=True，num_workers 默认 2。

---

## 4. 环境安装

### 4.1 推荐环境

- Python ≥ 3.8
- PyTorch ≥ 1.10
- CUDA ≥ 11.0(GPU 训练推荐)

### 4.2 安装依赖

```bash
# 进入项目根目录
cd CIFAR10

# (可选)创建虚拟环境
python -m venv venv
# Windows PowerShell:
venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

# 安装项目依赖
pip install -r requirements.txt
```

---

## 5. 模型说明

### 5.1 SimpleCNN(基线小模型)

| 属性 | 值 |
|---|---|
| 文件 | [models/simple_cnn.py](file:./models/simple_cnn.py) |
| 参数量 | ~0.24 M |
| 结构 | 4×Conv-BN-ReLU (32→64→128→128) + MaxPool×2 (conv2/4 后) + GAP + FC(128→10) |
| 观测层 | `conv1 / conv2 / conv3 / conv4 / fc` |
| 设计定位 | 浅层小模型基线，非线性误差敏感性的参考基准 |

### 5.2 RobustCNN(任务3 鲁棒增强版)

| 属性 | 值 |
|---|---|
| 文件 | [models/robust_cnn.py](file:./models/robust_cnn.py) |
| 主干 | 同 SimpleCNN 完全一致(4 Conv + GAP) |
| 新增模块 | **FeatureCalibration**(128→64→128 MLP + 残差连接 `x + MLP(x)`，参数量 16,640)，位于 GAP 和 FC 之间，受 `use_calibration` 开关控制 |
| 设计定位 | 任务3 Exp1/2/3 消融的载体，验证残差校准补偿 fc 层失真 |

### 5.3 ResNet-18(CIFAR-10 适配版)

| 属性 | 值 |
|---|---|
| 文件 | [models/resnet.py](file:./models/resnet.py) |
| 参数量 | ~11.17 M |
| 关键适配 | conv1 改为 3×3 stride=1(非 ImageNet 7×7 stride=2)，移除初始 MaxPool；BasicBlock×[2,2,2,2]，通道 64→128→256→512 |
| 观测层 | `layer1 / layer2 / layer3 / layer4`(Sequential 容器，挂 hook 取整个 stage 输出)+ `fc` |
| 设计定位 | 深层残差连接对照组，验证「残差恒等映射缓解误差累积」假设 |

### 5.4 VGG-11(CIFAR-10 轻量适配版)

| 属性 | 值 |
|---|---|
| 文件 | [models/vgg11.py](file:./models/vgg11.py) |
| 参数量 | ~9.2 M(比原始 VGG11 去掉了 3 个大 FC，节省 120M+) |
| 结构 | 8×Conv-BN-ReLU + 5×MaxPool(features Sequential) + GAP + 单 FC(512→10) |
| 观测层 | `features[7] / features[14] / features[21] / features[28]`(每个 block 最后一个 MaxPool 输出)+ `classifier[-1]` Linear(fc_output) |
| 设计定位 | 「深层 + 纯前馈 + 无残差」中间对照，剥离残差连接影响，可回答 SimpleCNN vs VGG-11(深度增加是否必然脆弱)、VGG-11 vs ResNet-18(残差是否真正起稳定作用)两道问题 |

---

## 6. 基础训练与评估

### 6.1 `train.py` — 干净模型训练

**功能**：在干净 CIFAR-10 上训练分类模型，支持断点续训、可选早停、自动保存训练曲线/混淆矩阵/metrics.json。

#### 命令行参数表

| 参数 | 类型 | 默认值 | Choices | 含义 |
|---|---|---|---|---|
| `--model` | str | `simple_cnn` | `simple_cnn`, `resnet18`, `vgg11` | 模型名称 |
| `--resume` | str | `None` | — | 续训 checkpoint 路径(dict 格式或纯 state_dict 均可) |
| `--batch_size` | int | `128` | — | 训练/测试 batch 大小 |
| `--epochs` | int | `50` | — | 总目标 epoch 数(续训填最终期望总轮数) |
| `--lr` | float | `0.01` | — | SGD 初始学习率 |
| `--weight_decay` | float | `1e-4` | — | L2 权重衰减 |
| `--momentum` | float | `0.9` | — | SGD 动量 |
| `--device` | str | 自动(有CUDA用cuda否则cpu) | `cuda`, `cpu` | 计算设备 |
| `--num_workers` | int | `2` | — | DataLoader 工作线程 |
| `--seed` | int | `42` | — | 随机种子 |
| `--patience` | int | `None` | — | 早停耐心轮数(不设则不启用早停) |

#### 输入/输出路径
- **输入**：无强制输入(首次训练无需 checkpoint；续训传 `--resume` )。。
- **输出目录**：`./checkpoints/{args.model}/best_model.pth (权重) ；`./outputs/{args.model}/`(图表)
  - `training_curves.png`(Loss / Acc 双子图，红星标最佳 epoch)
  - `confusion_matrix.png`(干净测试集混淆矩阵，10×10 热力图)
  - `metrics.json`(best_test_acc / best_epoch / start_epoch / resumed / total_epochs 等)

#### 典型运行命令

```bash
# 训练 SimpleCNN 基线(50 epochs)
python train.py --model simple_cnn --epochs 50

# 训练 ResNet-18(100 epochs，推荐)
python train.py --model resnet18 --epochs 100 --lr 0.01 --batch_size 128

# 训练 VGG-11
python train.py --model vgg11 --epochs 100

# 从断点续训 SimpleCNN 到总共 100 epoch
python train.py --model simple_cnn --epochs 100 --resume ./checkpoints/simple_cnn/best_model.pth

# 启用早停(连续 10 epoch 测试准确率不提升则停止)
python train.py --model simple_cnn --epochs 150 --patience 10
```

> 续训时自动：加载模型/优化器状态、读取 epoch 与 best_test_acc、手动将 CosineAnnealingLR step 到 start_epoch 次保持衰减曲线连续。

---

### 6.2 `evaluate.py` — 模型评估

**功能**：加载 checkpoint 对干净测试集进行完整评估，控制台打印总体准确率、每类 Precision/Recall/F1、宏平均、加权平均，可保存混淆矩阵+metrics_eval.json。

#### 命令行参数表

| 参数 | 类型 | 默认值 | Choices | 含义 |
|---|---|---|---|---|
| `--model` | str | `simple_cnn` | `simple_cnn`, `resnet18`, `vgg11` | 模型名称 |
| `--checkpoint` | str | `./checkpoints/simple_cnn/best_model.pth` | — | 权重路径 |
| `--device` | str | 自动 | `cuda`, `cpu` | 设备 |
| `--batch_size` | int | `128` | — | Batch Size |
| `--num_workers` | int | `2` | — | DataLoader workers |
| `--save_output_dir` | str | `None` | — | 若指定则保存 confusion_matrix_eval.png 和 metrics_eval.json 到该目录 |

#### 典型运行命令

```bash
# 控制台打印评估 SimpleCNN
python evaluate.py \
    --model simple_cnn \
    --checkpoint ./checkpoints/simple_cnn/best_model.pth

# 保存评估产物
python evaluate.py \
    --model resnet18 \
    --checkpoint ./checkpoints/resnet18/best_model.pth \
    --save_output_dir ./outputs/resnet18
```

---

## 7. 任务1：非线性误差的敏感性分析

### 7.1 任务简介

模拟 CiM 芯片模拟域 MAC 输出侧的三次非线性失真，通过在所有 Conv2d/Linear 输入侧挂 forward_pre_hook 注入，完成两个子分析：
- **敏感性扫描**：不同 α 下整网精度衰减曲线；
- **层偏移 & 误差累积**：5 个关键观测层(因模型而异)的输出分布偏移 RME / 余弦相似度，观察误差跨层如何被放大。

---

### 7.2 `task1_sensitivity_analysis.py` — α 敏感性扫描

#### 命令行参数表

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `--alpha_values` | str | `"-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3"` | 逗号分隔 α 列表 |
| `--model` | str | `"simple_cnn"` | 支持 `simple_cnn` / `resnet18` / `vgg11` |
| `--checkpoint` | str | `"./checkpoints/simple_cnn/best_model.pth"` | 干净权重路径 |
| `--batch_size` | int | `128` | 推理 Batch Size |
| `--device` | str | 自动 | cuda / cpu |
| `--output_dir` | str | `"./outputs/task1"` | 结果输出目录(**推荐切换模型时显式指定为 `./outputs/task1_{model_tag}`，例如 `./outputs/task1_resnet18`**) |
| `--seed` | int | `42` | 随机种子 |

#### 跨脚本路径依赖
- **输入依赖**：`--checkpoint` 指向的干净模型权重(必须)
- **输出文件**(`--output_dir` 下)：
  1. `alpha_sensitivity.csv`(列：`alpha, accuracy, loss`)
  2. `accuracy_vs_alpha.png`(横轴 Nonlinearity Strength (α)，纵轴 Test Accuracy (%)；红色星号 + 竖直虚线标 α=0 干净基线)

#### 典型运行命令

```bash
# SimpleCNN 默认扫描
python task1_sensitivity_analysis.py \
    --output_dir ./outputs/task1_simplecnn

# ResNet-18 自定义扫描
python task1_sensitivity_analysis.py \
    --model resnet18 \
    --checkpoint ./checkpoints/resnet18/best_model.pth \
    --alpha_values "-0.5,-0.3,0.0,0.3,0.5" \
    --output_dir ./outputs/task1_resnet18
```

> 每个 α 推理前强制 `model.load_state_dict(clean_ckpt) 重载干净权重，`register_nonlinearity_hooks` 挂钩子，`try / finally` 保证 `remove_hooks`。

---

### 7.3 `task1_layer_analysis.py` — 层分布偏移与误差累积

#### 命令行参数表

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `--alpha_values` | str | `"-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3"` | α 列表 |
| `--model` | str | `"simple_cnn"` | simple_cnn / resnet18 / vgg11 |
| `--checkpoint` | str | `"./checkpoints/simple_cnn/best_model.pth"` | 干净权重 |
| `--num_samples` | int | `500` | 固定样本数(取 test_loader 前 N 张共享) |
| `--batch_size` | int | `128` | Batch Size(存在该参数) |
| `--device` | str | 自动 | cuda / cpu |
| `--output_dir` | str | `"./outputs/task1"` | (推荐显式带模型后缀) |
| `--seed` | int | `42` | 种子 |

#### 观测层映射(按模型名自动切换，5 点等深度对齐便于横向对比)

| 模型 | 观测层列表 | 直方图输出层(浅 + 深两个) |
|---|---|---|
| SimpleCNN | conv1 / conv2 / conv3 / conv4 / fc | conv1 + conv4 |
| ResNet-18 | layer1 / layer2 / layer3 / layer4 + fc | layer1 + layer4 |
| VGG-11 | features[7] / features[14] / features[21] / features[28] + fc | features[7] (=block2 末) + features[28] (=block5 末) |

#### 输出文件(`--output_dir` 下)
1. `layer_distribution_shift.csv`(列：`alpha, layer_name, RME, cosine_similarity`)
2. `error_accumulation.png`(Coolwarm 色阶多 α 跨层余弦相似度连线)
3. `histogram_{浅观测层名}.png`(α=0 vs 0.3 分布直方图对比)
4. `histogram_{深观测层名}.png`

#### 典型运行命令

```bash
python task1_layer_analysis.py \
    --model simple_cnn \
    --output_dir ./outputs/task1_simplecnn

python task1_layer_analysis.py \
    --model resnet18 \
    --checkpoint ./checkpoints/resnet18/best_model.pth \
    --output_dir ./outputs/task1_resnet18
```


---

## 8. 任务2：非线性感知训练(NAT)

脚本：`task2_nat.py`

### 8.1 任务简介

训练阶段在每个 batch 的 Conv2d/Linear 输入端动态注入随机或固定 α 的非线性失真("在噪声中学习")，验证阶段干净推理(α=0)，对比：
- **Finetune(微调)**：从干净预训练权重出发(~80 epochs，计算量小)。
- **Scratch(从头训练)**：随机初始化，全程带失真训练(~120 epochs，鲁棒性上限更高)。

### 8.2 命令行参数表

| 参数 | 类型 | 默认值 | Choices | 含义 |
|---|---|---|---|---|
| `--mode` | str | **必填** | `finetune`, `scratch` | 训练模式(唯一必填参数) |
| `--model` | str | `"simple_cnn"` | simple_cnn / resnet18 / vgg11 | 模型名 |
| `--checkpoint` | str | `"./checkpoints/simple_cnn/best_model.pth"` | — | 仅 finetune 模式使用；scratch 模式下参数虽有默认值但实际不使用 |
| `--alpha_mode` | str | `"random"` | `random`, `fixed` | α 采样策略 |
| `--alpha_range` | str | `"-0.3,0.3"` | — | random 模式下 `low,high 逗号分隔 |
| `--alpha_fixed` | float | `0.3` | — | fixed 模式下的固定 α |
| `--epochs` | int | `None` | — | None 时按 mode 默认：finetune→40、scratch→120 |
| `--lr` | float | `None` | — | None 时按 mode 默认：finetune→1e-3、scratch→0.01 |
| `--weight_decay` | float | `1e-4` | — | L2 衰减 |
| `--momentum` | float | `0.9` | — | SGD 动量 |
| `--device` | str | 自动 | cuda / cpu | 设备 |
| `--num_workers` | int | `2` | — | workers |
| `--seed` | int | `42` | — | 种子 |
| `--output_dir` | str | `"./outputs/task2_simplecnn"` | — | **注意：该参数默认值实际不生效，实际 save_dir_output 由 f-string `./outputs/task2_{model_tag}/{args.mode}` 动态拼接决定** |
| `--patience` | int | `None` | — | 早停耐心轮数 |

> **特别注意**：`--batch_size 参数**未在 argparse 中声明**，主函数通过 `args.batch_size if hasattr(args, "batch_size") and args.batch_size else 128` 兜底为 128。

### 8.3 动态路径规则

```python
model_tag = args.model.replace("_", "")                # simple_cnn → simplecnn
save_dir_ckpt   = f"./checkpoints/nat_{args.mode}_{model_tag}"
save_dir_output = f"./outputs/task2_{model_tag}/{args.mode}"
```

任务1 基线 CSV 读取路径为 `f"./outputs/task1_{model_tag}/alpha_sensitivity.csv"`，若文件不存在则跳过对比绘图。

### 8.4 跨脚本路径依赖

- **输入**(finetune 模式)：`./checkpoints/{model_name}/best_model.pth`
- **输入(对比图基线)**：`./outputs/task1_{model_tag}/alpha_sensitivity.csv`(任务1 敏感性 CSV，不存在自动跳过)

### 8.5 训练后自动 α 扫描

训练结束对固定列表 `TASK1_ALPHA_LIST = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]` 执行与任务1完全一致的敏感性扫描。

### 8.6 输出文件(`./outputs/task2_{model_tag}/{mode}/` 下)
1. `training_curves.png`(Loss / Acc 曲线，红星标 clean 最佳 epoch)
2. `confusion_matrix.png`
3. `metrics.json`(含 best_test_acc / alpha_mode / alpha_range / alpha_fixed / nat_train_mode 等)
4. `alpha_sensitivity.csv`(NAT 模型扫描结果)
5. `accuracy_vs_alpha_comparison.png`(NAT 红实线 + 任务1 基线蓝虚线同轴对比)

对应权重：`./checkpoints/nat_{mode}_{model_tag}/best_model.pth`

### 8.7 典型运行命令

```bash
# Finetune(40 epochs，lr=1e-3，random α∈[-0.3,0.3]
python task2_nat.py --mode finetune --model simple_cnn

# Finetune + 固定 α=0.3
python task2_nat.py --mode finetune --model simple_cnn \
    --alpha_mode fixed --alpha_fixed 0.3 --epochs 80

# Scratch(120 epochs，lr=0.01，random α
python task2_nat.py --mode scratch --model simple_cnn

# ResNet-18 scratch(更强扰动，更宽范围)
python task2_nat.py --mode scratch --model resnet18 \
    --alpha_range "-0.5,0.5" --epochs 150
```


---

## 9. 任务3：鲁棒性增强方法设计

脚本：`task3_robust.py`

### 9.1 任务简介

在 NAT-Scratch 基础上叠加**三项互补增强方案，通过消融实验逐步验证：

| 方案 | 类型 | 说明 |
|---|---|---|
| **A 架构增强 | 架构 | GAP 后 + FeatureCalibration(128→64→128 MLP + 残差连接)补偿 fc 层失真；最坏情况退化为恒等 |
| **B 分层加权 α** | 训练策略 | conv1/2 ∈ [-0.15, 0.15](轻失真)；conv3/4 ∈ [-0.30, 0.30](标准)；fc 每 batch 随机选 ±0.3(强失真) |
| **C 非对称偏置** | 采样策略 | 70% 概率正向 α(破坏力更强)，30% 概率负向 α |

三种方案的组合映射(`--exp` 到 4 种配置)：

| `--exp` | 实验名称 | use_calibration(A) | layerwise_alpha(B) | asymmetric_sampling(C) |
|---|---|---|---|---|
| `1` | `Exp1_CalibOnly` | ✓ | ✗ | ✗ |
| `2` | `Exp2_Calib+Layerwise` | ✓ | ✓ | ✗ |
| `3` | `Exp3_FullRobust` | ✓ | ✓ | ✓ |
| `all` | 串联 Exp1 → Exp2 → Exp3 + 消融汇总 | — | — | — |

### 9.2 命令行参数表

| 参数 | 类型 | 默认值 | Choices | 含义 |
|---|---|---|---|---|
| `--exp` | str | `"all"` | `"1"`, `"2"`, `"3"`, `"all"` | 实验编号或 `all` 串联 |
| `--model` | str | `"robust_cnn"` | `simple_cnn`, `robust_cnn` | (get_model 工厂仅支持这两种) |
| `--epochs` | int | `120` | — | 训练 epoch 数 |
| `--batch_size` | int | `128` | — | Batch Size(存在该参数) |
| `--lr` | float | `0.01` | — | SGD 初始 LR |
| `--weight_decay` | float | `1e-4` | — | L2 衰减 |
| `--momentum` | float | `0.9` | — | SGD 动量 |
| `--device` | str | 自动 | cuda / cpu | 设备 |
| `--num_workers` | int | `2` | — | workers |
| `--seed` | int | `42` | — | 种子 |
| `--output_dir` | str | `"./outputs/task3_simplecnn"` | — | (推荐带模型后缀，实际子目录按 `{output_dir}/{exp_name}_{model_tag}` 动态拼接决定) |
| `--patience` | int | `None` | — | 早停耐心轮数 |

### 9.3 动态路径规则

```python
model_tag = args.model.replace("_", "")
# 每个实验的权重路径(exp_name 为 Exp1_CalibOnly / Exp2_Calib+Layerwise / Exp3_FullRobust
save_dir_ckpt   = f"./checkpoints/{exp_name}_{model_tag}"
save_dir_output_exp = f"{args.output_dir}/{exp_name}_{model_tag}"
```

NAT-Scratch 基线读取路径硬编码为 `./outputs/task2_simplecnn/scratch/alpha_sensitivity.csv`(若不存在跳过基线对比图)。

### 9.4 输出文件(每个实验子目录(`Exp1_CalibOnly_{model_tag}/` 等结构一致))：
- `training_curves.png` / `confusion_matrix.png` / `metrics.json`(含三个开关字段 + exp_name)/ `alpha_sensitivity.csv` / `accuracy_vs_alpha_comparison.png`(当前实验 vs NAT-Scratch 基线)

`--exp all` 额外在 `--output_dir` 根目录生成：
1. `ablation_summary.csv`(列：exp_name / use_calibration / layerwise_alpha / asymmetric_sampling / clean_acc / acc_at_alpha_0.3 / acc_drop_at_0.3)
2. `ablation_comparison.png`(Exp1/2/3 三条彩色实线 + NAT-Scratch 红虚线基线)
3. 终端消融配置表(✓/✗ 三开关 + Clean / Acc@0.3 / Drop@0.3 边际改善)

### 9.5 典型运行命令

```bash
# 推荐：串联三个实验 + 消融汇总(约 3 × 120 epochs)
python task3_robust.py --exp all

# 单独实验
python task3_robust.py --exp 1    # 仅残差校准
python task3_robust.py --exp 2    # 校准 + 分层α
python task3_robust.py --exp 3    # 完整 A+B+C

# 自定义更快
python task3_robust.py --exp 3 --epochs 100 --lr 0.005 --patience 15
```


---

## 10. 拓展研究1：网络结构与参数量对比

脚本：`task_extension1_network_comparison.py`

### 10.1 研究问题

1. 更大参数量模型是否天然更鲁棒(过参数化容错假设)？
2. 纯前馈深度(VGG-11) vs 残差深度(ResNet-18)误差累积模式差异？
3. 过参数化冗余 vs 深度放大误差 — 哪个占主导？

### 10.2 命令行参数表

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `--models` | str | `"simple_cnn,resnet18"` | 逗号分隔模型列表，顺序与 `--checkpoints` 一一对应 |
| `--checkpoints` | str | `"./checkpoints/simple_cnn/best_model.pth,./checkpoints/resnet18/best_model.pth"` | 逗号分隔 clean ckpt |
| `--nat_models` | str | `""` | (可选)逗号分隔 NAT 模型名，与 `--nat_checkpoints` 对应 |
| `--nat_checkpoints` | str | `""` | (可选)逗号分隔 NAT 权重路径 |
| `--alpha_values` | str | `"-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3"` | α 列表(不含 0.0 自动前置插入) |
| `--batch_size` | int | `128` | Batch Size |
| `--device` | str | 自动 | cuda / cpu |
| `--output_dir` | str | `"./outputs/extension1"` | 输出根目录 |
| `--seed` | int | `42` | 种子 |

### 10.3 效率优化

首次加载权重后深拷贝 `clean_state_dict`(CPU.clone()，后续每个 α 直接 `load_state_dict` 内存重载，省去重复磁盘 IO)。

### 10.4 输出文件(`./outputs/extension1/`)
1. `{model_name}_alpha_sensitivity.csv`(每个 clean 模型)
2. `{model_name}_nat_alpha_sensitivity.csv`(每个 NAT 模型)
3. `accuracy_comparison.png`(clean 实线圈 + NAT 空心 diamond，统一架构色区分)
4. `accuracy_drop_comparison.png`(仅 α>0 跌幅百分比)
5. `robustness_vs_params.png`(log 横轴参数量 × α=+0.3 精度散点，clean 实心黑边、NAT 空心粗边 + 文字箭头)
6. `network_comparison_summary.csv`(列：model_name / total_params / clean_accuracy / acc_at_alpha_0.3 / acc_drop_at_0.3)

### 10.5 典型运行命令

```bash
# 默认：simple_cnn + resnet18 对比
python task_extension1_network_comparison.py

# 三模型完整对比(需先训练 VGG-11 干净权重)
python task_extension1_network_comparison.py \
    --models "simple_cnn,resnet18,vgg11" \
    --checkpoints "./checkpoints/simple_cnn/best_model.pth,./checkpoints/resnet18/best_model.pth,./checkpoints/vgg11/best_model.pth" \
    --output_dir ./outputs/extension1

# 带 NAT 模型的对比(需先训练 NAT-Scratch)
python task_extension1_network_comparison.py \
    --models "simple_cnn,resnet18" \
    --checkpoints "./checkpoints/simple_cnn/best_model.pth,./checkpoints/resnet18/best_model.pth" \
    --nat_models "simple_cnn,resnet18" \
    --nat_checkpoints "./checkpoints/nat_scratch_simplecnn/best_model.pth,./checkpoints/nat_scratch_resnet18/best_model.pth"
```


---

## 11. 拓展研究2：高斯噪声 vs 非线性失真

脚本：`task_extension2_noise_vs_nonlinearity.py`

### 11.1 研究问题

1. 两种扰动(输入相关非线性 vs 输入无关高斯噪声)的精度衰减模式差异？
2. 层误差累积路径差异？最脆弱层是否一致？
3. 任务3 Exp2 鲁棒模型(仅见过非线性训练)是否对完全未见的高斯噪声具备跨域防护？

### 11.2 命令行参数表

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `--models` | str | `"simple_cnn,exp2,resnet18"` | 支持名列表，支持 `simple_cnn` / `exp2`(→任务3 Exp2 鲁棒CNN，ckpt 硬编码 `./checkpoints/Exp2_Calib+Layerwise_simplecnn/best_model.pth`)/ `vgg11` / `resnet18` |
| `--pert_types` | str | `"nonlinearity,gaussian"` | `nonlinearity`(三次失真) / `gaussian`(加性噪声) |
| `--alpha_values` | str | `"-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3"` | α 列表 |
| `--noise_levels` | str | `"0.05,0.10,0.15,0.20,0.25,0.30"` | σ 列表(自动补 σ=0.0 基线) |
| `--num_samples` | int | `500` | 层偏移固定样本数 |
| `--batch_size` | int | `128` | Batch Size |
| `--device` | str | 自动 | cuda / cpu |
| `--output_dir` | str | `"./outputs/extension2_simplecnn"` | (推荐切换模型时显式修改目录名) |
| `--seed` | int | `42` | 种子 |

### 11.3 输出文件(`{output_dir}/{model_instance}/` 子目录下，每个模型一份)
1. `nonlinearity_sensitivity.csv` / `gaussian_sensitivity.csv`
2. `layer_shift_nonlinearity.csv` / `layer_shift_gaussian.csv`
3. `error_accumulation_nonlinearity.png` / `error_accumulation_gaussian.png`
4. `histogram_{浅/深层}_gaussian.png`(σ=0 vs 0.20 分布对比)
5. `accuracy_comparison_{m}.png`(**双横轴**：下轴 α -0.3~0.3，上轴 σ 0.00~0.30 线性映射，非线性红实圈 / 高斯蓝虚方)
6. `error_accumulation_comparison_{m}.png`(可比强度 α=0.2 vs σ=0.15 跨层 cos 对比)

若同时存在 simple_cnn + exp2：
7. `robustness_transfer_comparison.png`(1×2 子图，左 Nonlinearity / 右 Gaussian，SimpleCNN 蓝虚 / Exp2 红实)

### 11.4 典型运行命令

```bash
# 默认：完整运行
python task_extension2_noise_vs_nonlinearity.py

# 仅 SimpleCNN 快速
python task_extension2_noise_vs_nonlinearity.py \
    --models simple_cnn --output_dir ./outputs/extension2_simplecnn

# 自定义扫描范围
python task_extension2_noise_vs_nonlinearity.py \
    --models simple_cnn,exp2 \
    --noise_levels "0.1,0.2,0.3,0.5" \
    --alpha_values "-0.5,-0.3,0.0,0.3,0.5"
```


---

## 12. 拓展研究3：量化误差与非线性误差联合分析

脚本：`task_extension3_joint_analysis.py`

### 12.1 研究问题

CiM 芯片信号路径：输入激活 → MAC(非线性失真)→ ADC(QDQ 量化)。联合误差顺序严格遵循「先 nonlinearity，后 quantize_error(对应物理路径)。研究：
1. 单独量化(32/8/6/4/3/2 bit 精度衰减)；
2. α×bit 网格下的**超加性叠加恶化；
3. 任务3 Exp2 鲁棒模型在联合压力下是否仍保持增益。

### 12.2 命令行参数表

| 参数 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `--models` | str | `"simple_cnn,exp2"` | 支持 `simple_cnn` / `exp2` / `resnet18` / `vgg11` |
| `--alpha_values` | str | `"-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3"` | α 列表 |
| `--num_bits_list` | str | `"8,6,4,3,2"` | ADC 比特列表 |
| `--batch_size` | int | `128` | Batch Size |
| `--device` | str | 自动 | cuda / cpu |
| `--output_dir` | str | `"./outputs/extension3_simplecnn"` | (推荐带模型后缀) |
| `--seed` | int | `42` | 种子 |
| `--target_bit` | int | `4` | 图3 联合 vs 单独、图4 多模型对比的固定 bit |

### 12.3 效率优化

首次加载后保存 `clean_state_dict_CPU_deepcopy` 快照，后续 80 次(7α × 5bit × 2模型 + 5bit 独扫 + 7α×5bit 联合)全从内存快照重载。

### 12.4 超加性判定规则(脚本终端自动输出)

设 `drop_n` = 单独非线性跌幅，`drop_q` = 单独量化跌幅，`drop_joint` = 联合跌幅：
- `drop_joint > 1.05 × (drop_n + drop_q)` → **Super-Additive(恶化，两误差共振)
- 0.95×和 ~ 1.05× → Linear(线性叠加可求和即可)
- `< 0.95× → Sub-Additive(相互抑制)

### 12.5 输出文件(`{output_dir}/` 根目录)
1. `{m}_quantization_only.csv`(列 num_bits / accuracy / loss)
2. `{m}_joint_error.csv`(列 alpha / num_bits / accuracy / loss)
3. `quantization_only_{m}.png`(bit 曲线 + 32-bit 红星参考)
4. `joint_error_heatmap_{m}.png`(RdYlGn 色图，Y 轴 bit 从上到下递减，单元格中心加粗标数值)
5. `joint_vs_alone_comparison.png`(1×N 子图固定 target_bit：非线-only 32-bit 蓝实圈 + quant-only 绿虚水平线 + 联合(target_bit红实方块))
6. `robustness_under_joint_error.png`(固定 target_bit 多模型 α 曲线对比，style_map：simple_cnn 蓝虚圈 / exp2 红实方 / resnet18 绿点划三角)
7. `joint_error_summary.csv`(error_type ∈ nonlinearity_only / quantization_only / joint 三类合并)

### 12.6 典型运行命令

```bash
# 默认：两模型完整扫描(约 80 次 10K 推理)
python task_extension3_joint_analysis.py

# 仅 SimpleCNN + 少量扫描快速验证
python task_extension3_joint_analysis.py \
    --models "simple_cnn" \
    --alpha_values "-0.3,0.0,0.3" \
    --num_bits_list "8,4,2"

# 自定义 target_bit=3
python task_extension3_joint_analysis.py \
    --models "simple_cnn,exp2" \
    --target_bit 3
```

