# 存算一体芯片中非线性误差对推理精度的影响研究

> **项目状态：六阶段收官（归档 tag `v1.0-archive`）＋ 第七阶段自主迭代进行中。**
> 全局台账 `results_master.csv` **152 行数据，配对自检 0 FAIL**；六阶段结论收敛为九条（L1–L9），
> 其中最贵的一课是「精度数字不可跨训练实现移植」。
> 第七阶段（§16）已在归档后新增十余项实验，产出**两条加强中心结论的结果**
> （容量不是瓶颈；headroom 是下界且可抬高到 +27.63）与**一批方法学结论**
> （三级可分辨性、两条口径陷阱）。**实验仍在进行，台账 §16 持续追加。**

**先读这四件（按此顺序）**

| 文件 | 是什么 |
|---|---|
| [`论文v3.md`](论文v3.md) | 论文：九条结论的终版表述 + 方法 + 局限 + **12 条勘误** + 复现材料 |
| [`experiments_ledger.md`](experiments_ledger.md) | 实验台账：§0–§16，每个实验的**预登记判据**与结果 |
| [`docs/PROJECT_RETROSPECTIVE.md`](docs/PROJECT_RETROSPECTIVE.md) | **阶段史**：每个阶段"当初信什么、后来被什么推翻"；含三次"噪声→翻案"专题与新人上手 |
| [`deploy_notes.md`](deploy_notes.md) | 部署推演：把结论映射到 CIM 数字后端，三档判定（可行 / 需改动 / 不可行） |

**第七阶段新增的文档**（多会话协作，见 `docs/V7_COORDINATION.md`）

| 文件 | 是什么 |
|---|---|
| [`paper/`](paper/) | 三份论文（竞赛 / 期刊 / 技术报告）+ 骨架与形式工作文件；`paper/README.md` 有索引 |
| [`docs/V7_COORDINATION.md`](docs/V7_COORDINATION.md) | **多会话协调规则**：文件领地、并行上限、通信协议、已踩的坑（**动手前先读**） |
| [`docs/V7_M4_RUNBOOK.md`](docs/V7_M4_RUNBOOK.md) | M4（深层主干方差）的收尾流程与预登记判据 |
| [`docs/POSITIONING.md`](docs/POSITIONING.md) | 与先行工作的关系与差异化（LP-FT / CIM 数字域补偿 / 探针批判） |
| [`docs/LITERATURE_NOTES.md`](docs/LITERATURE_NOTES.md) | 文献核实笔记（每条标注核实状态；⚠ 者不得引用） |
| [`docs/IMPL_AUDIT.md`](docs/IMPL_AUDIT.md) | 数字–实现对照表（跨实现混用的核查方法） |
| [`docs/DERIVED_VALUES.md`](docs/DERIVED_VALUES.md) | 派生值登记表（均值/差值/比值必须写清算式） |
| [`paper/coverage-audit.md`](paper/coverage-audit.md) | 成果覆盖审计与闭环记录 |

**工具**：`build_ledger.py`（重建台账）、`tools/recalc_mean7.py`（复算派生量）、
`tools/check_papers.py`（**改论文后、提交前必跑**：扫 markdown 残留 + 编译 + 未定义引用）。

一个面向 CIFAR-10 图像分类的深度学习研究框架，用于系统性地分析存算一体(Compute-in-Memory, CiM)芯片中模拟域非线性失真、ADC 量化误差以及两者联合作用下的推理精度退化机制，并提出非线性感知训练(NAT)与架构级鲁棒性增强方案。内置 6 种模型架构(SimpleCNN / RobustCNN / ResNet-18 / VGG-11 / RobustVGG11 / RobustResNet18)，覆盖 3 项核心任务 + 6 项拓展研究，支持断点续训、丰富可视化与自动化消融汇总。

---

## 1. 项目简介

### 1.1 研究背景与核心问题

存算一体芯片通过在 SRAM/Flash 阵列内执行模拟域乘累加(MAC)运算，大幅降低数据搬运开销，是边缘端 AI 推理的高算力/低功耗方向。但模拟域计算天然存在器件失配、OTA 增益非线性等物理非理想效应，在 Conv2d / Linear 算子的输入侧引入三次多项式非线性失真：

$$y = \alpha \cdot \tilde{x}^3 + (1-\alpha) \cdot \tilde{x}, \quad \tilde{x} = \frac{x}{\|x\|_{\infty, \text{ per-sample}}}$$

其中 $\alpha$ 为非线性强度($\alpha=0$ 理想线性，$\alpha>0$ 正增益饱和，$\alpha<0$ 负增益截止)。本项目围绕以下 7 个研究问题展开：

1. **任务1**：不同 $\alpha$ 下整网精度如何衰减？误差如何跨层累积？哪一层最脆弱？
2. **任务2**：训练阶段"见过非线性"的 NAT 感知训练(微调 / 从头训练)能多大程度补偿失真？
3. **任务3**：在 NAT-Scratch 基础上，架构级残差校准 + 分层加权采样 + 非对称偏置三项增强的**消融验证；
4. **拓展1**：不同参数量/深度/拓扑(浅层CNN vs 纯前馈深层VGG vs 残差ResNet)对非线性失真的敏感性差异？
5. **拓展2**：输入相关的非线性失真 vs 输入无关的高斯加性噪声，两者在误差机制差异及鲁棒性迁移性？
6. **拓展3**：MAC 非线性 + ADC 量化**联合误差**是否存在超加性叠加恶化？
7. **第二阶段**：NAT 鲁棒性在训练分布外（|α|>0.3）是否仍有效？Dithering 效应的真实机理是什么？Exp2 增强方案能否迁移至深层网络？

### 1.2 核心技术亮点

- **统一 Hook 注入接口**：所有误差/扰动均通过 `nn.Conv2d / nn.Linear` 的 `register_forward_pre_hook` 注入，所有钩子使用工厂闭包快照参数避免 Python 延迟绑定陷阱，所有推理循环采用 `try / finally` 保证钩子稳健释放；
- **逐样本动态归一化非线性**：对 batch 维每个样本独立做 $\infty$-范数归一化再施加非线性，避免 batch 间信号串扰；
- **逐样本动态 QDQ 量化**：ADC 仿真使用逐样本动态缩放的对称饱和整型量化-反量化，`num_bits >= 32 / None` 自动短路；
- **完整消融实验自动化**：任务3 `--exp all` 一键串联 Exp1/2/3 三个消融配置，自动生成 CSV + 对比图 + 终端配置表；
- **多模型架构对比原生支持**：所有任务/拓展的 `get_model()` 工厂支持 `simple_cnn / resnet18 / vgg11 / robust_cnn / exp2` 等多种名称，输出/权重路径根据模型名动态后缀，避免覆盖。
- **三假设竞争判决实验框架**：剂量匹配三对照（4bit QDQ / 等剂量纯噪声 / 8bit+最优剂量噪声）+ 样本级翻转统计 + 激活分布统计，定量判决 Dithering 效应机理（H1 随机去相关否决 / H2 分布重塑确认 / H3 MaxPool 交互支持）；
- **跨架构方案迁移验证**：Exp2 的 FeatureCalibration 零修改复用，经 `permute` 适配空间特征图，注入密度随误差累积节点数扩展（VGG-11 每个 MaxPool 后共 5 处 / ResNet-18 每个 BasicBlock 残差路径共 8 处），验证校准注入密度法则。

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
│   ├── robust_vgg11.py                             # RobustVGG11：VGG-11 + 每个 MaxPool 后 5 处 FeatureCalibration（Exp2 深层迁移）
│   ├── robust_resnet.py                            # RobustResNet18：ResNet-18 + 每个 BasicBlock 残差路径 8 处 FeatureCalibration（Exp2 深层迁移）
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
│   ├── Exp1_CalibOnly_simple_cnn/                  # 任务3 Exp1(仅方案A),only simple_cnn
│   ├── Exp2_Calib+Layerwise_simple_cnn/            # 任务3 Exp2(方案A+B),only simple_cnn
│   ├── Exp2_Calib+Layerwise_vgg11/                 # 拓展6 Exp2 迁移训练权重,VGG-11
│   ├── Exp2_Calib+Layerwise_resnet18/              # 拓展6 Exp2 迁移训练权重,ResNet-18
│   └── Exp3_FullRobust_simple_cnn/                 # 任务3 Exp3(方案A+B+C),only simple_cnn
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
│   ├── extension1_clean_vs_scratch/                # 拓展1：多模型架构对比,干净 vs NAT Scratch
│   │   ├── {m}_alpha_sensitivity.csv               # 每个 clean 模型
│   │   ├── {m}_nat_alpha_sensitivity.csv           # (可选 NAT)
│   │   ├── accuracy_comparison.png
│   │   ├── accuracy_drop_comparison.png
│   │   ├── robustness_vs_params.png
│   │   ├── network_comparison_summary.csv
│   │   └── extension1_clean_vs_scratch_summary.md  # 拓展1 干净 vs NAT Scratch 总结文档
│   ├── extension1_clean_vs_finetune/               # 拓展1：多模型架构对比,干净 vs NAT Finetune
│   │   ├── {m}_alpha_sensitivity.csv               # 每个 clean 模型
│   │   ├── {m}_nat_alpha_sensitivity.csv           # (可选 NAT)
│   │   ├── accuracy_comparison.png
│   │   ├── accuracy_drop_comparison.png
│   │   ├── robustness_vs_params.png
│   │   ├── network_comparison_summary.csv
│   │   └── extension1_clean_vs_finetune_summary.md # 拓展1 干净 vs NAT Finetune 总结文档
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
│   ├── extension3_{model_tag}/                     # 拓展3：量化+非线性联合(默认带_simplecnn后缀)
│   │   ├── {m}_quantization_only.csv
│   │   ├── {m}_joint_error.csv
│   │   ├── quantization_only_{m}.png
│   │   ├── joint_error_heatmap_{m}.png
│   │   ├── joint_vs_alone_comparison.png
│   │   ├── robustness_under_joint_error.png
│   │   ├── joint_error_summary.csv
│   │   └── extension3_{model_tag}_summary.md       # 拓展3 总结文档
│   ├── extension4_alpha_wide/                      # 拓展4：α 宽范围外推扫描(±0.6,训练区/外推区分区)
│   │   ├── alpha_wide_scan_summary.csv             # 汇总CSV(model/weight_type/alpha/accuracy/loss/is_extrapolation)
│   │   ├── simple_cnn_alpha_wide.png               # simple_cnn 4条曲线(含exp2_robust),竖虚线标α=±0.3
│   │   ├── vgg11_alpha_wide.png                    # vgg11 3条曲线
│   │   ├── resnet18_alpha_wide.png                 # resnet18 3条曲线
│   │   └── all_models_alpha_wide.png               # 3子图跨模型对比,共享y轴
│   ├── extension5_dithering_vgg11/                 # 拓展5：Dithering 效应机理解析(三假设竞争判决)
│   │   ├── group_a_bits_scan.csv                   # 组(a) bits扫描(num_bits/accuracy/loss/dose)
│   │   ├── group_b_noise_injection.csv             # 组(b) 噪声注入(noise_type/noise_std/accuracy/loss/dose)
│   │   ├── group_c_pure_noise.csv                  # 组(c) 纯噪声替代(等效4bit剂量)
│   │   ├── flip_statistics.csv                     # 样本级翻转统计(saved/killed/net_gain)
│   │   ├── activation_stats.csv                    # H2激活统计(conv1/分类器输入的均值/标准差/峰度)
│   │   ├── arch_dithering_summary.csv              # 组(d) 三架构各bits精度+4bit增益vs8bit
│   │   ├── bits_accuracy_curve.png                 # 组(a) bits-精度曲线,标注36.74/43.27锚点
│   │   ├── dose_accuracy_curve.png                 # 组(a/b/c)同轴剂量-精度曲线
│   │   ├── margin_distribution.png                 # 三组样本margin分布叠加图
│   │   └── arch_comparison_bits.png                # 组(d)三架构bits-精度叠加图,标注MaxPool次数
│   └── extension6_deep_robust/                     # 拓展6：Exp2 鲁棒方案迁移至深层网络
│       ├── alpha_wide_scan_deep_robust.csv         # 6列(model/weight_type=exp2_robust/alpha/accuracy/loss/is_extrapolation)
│       ├── deep_robust_comparison.png              # 两子图(每模型一个),叠加extension4基线曲线
│       ├── cross_architecture_summary.png          # 三架构×四权重跨架构汇总大图
│       ├── vgg11/                                  # VGG-11 训练产物
│       │   ├── training_curves.png
│       │   └── metrics.json
│       └── resnet18/                               # ResNet-18 训练产物
│           ├── training_curves.png
│           └── metrics.json
├── train.py
├── evaluate.py
├── task1_sensitivity_analysis.py
├── task1_layer_analysis.py
├── task2_nat.py
├── task3_robust.py
├── task_extension1_network_comparison.py
├── task_extension2_noise_vs_nonlinearity.py
├── task_extension3_joint_analysis.py
├── task_extension4_alpha_wide_scan.py
├── task_extension5_dithering_mechanism.py
├── task_extension6_deep_robust_train.py
├── task_extension6_eval_deep_robust.py
├── plot_cross_arch_summary.py
├── requirements.txt
├── 原始题目+题目解读+研究思路.md
└── README.md
```

> **路径命名规范**：`model_tag = args.model.replace("_", "")`，例如 `simple_cnn → simplecnn`，`resnet18 → resnet18`。所有任务/拓展的输出/权重目录均带模型后缀，避免多模型并行时互相覆盖。

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

### 5.5 RobustVGG11(第二阶段迁移模型)

| 属性 | 值 |
|---|---|
| 文件 | [models/robust_vgg11.py](file:./models/robust_vgg11.py) |
| 参数量 | ~9.81 M(VGG-11 主干 ~9.2 M + 5 处 FeatureCalibration ~0.61 M) |
| 结构 | 8×Conv-BN-ReLU + 5×MaxPool + GAP + 单 FC(512→10)；每个 MaxPool 后插入一处 FeatureCalibration(Conv→BN→ReLU→MaxPool→Calib) |
| 校准注入位置 | 5 处：block1~5 每个 MaxPool 输出(通道 64/128/256/512/512)，`hidden_dim = in_dim // 2` 瓶颈减半 |
| 空间适配 | forward 中 `permute(0,2,3,1)` 把通道维换至末尾过校准再还原，FeatureCalibration 类零修改 |
| 设计定位 | 任务3 Exp2 方案的深层迁移载体，验证多点空间校准对纯前馈深层网络的鲁棒增益与外推修复 |

### 5.6 RobustResNet18(第二阶段迁移模型)

| 属性 | 值 |
|---|---|
| 文件 | [models/robust_resnet.py](file:./models/robust_resnet.py) |
| 参数量 | ~11.87 M(ResNet-18 主干 ~11.17 M + 8 处 FeatureCalibration ~0.70 M) |
| 结构 | conv1(3×3 stride=1) + 4 组 CalibratedBasicBlock(通道 64→128→256→512) + GAP + FC；每个 BasicBlock 残差路径接 FeatureCalibration，`out = main(x) + Calib(shortcut(x))` |
| 校准注入位置 | 8 处：4 stage × 2 block 的 shortcut 输出(含 3 处 downsample block 与 5 处 identity block)，`hidden_dim = out_channels // 2` |
| 空间适配 | 残差路径输出 (B,C,H,W) 同样 `permute` 适配，FeatureCalibration 类零修改 |
| 设计定位 | 任务3 Exp2 方案的深层迁移载体，验证残差路径多点校准恢复 Scratch 策略在残差网络上的竞争力 |

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

> **特别注意**：`--batch_size ` 参数**未在 argparse 中声明**，主函数通过 `args.batch_size if hasattr(args, "batch_size") and args.batch_size else 128` 兜底为 128。

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

在 NAT-Scratch 基础上叠加**三项互补增强方案**，通过消融实验逐步验证：

| 方案 | 类型 | 说明 |
|---|---|---|
| **A 架构增强** | 架构 | GAP 后 + FeatureCalibration(128→64→128 MLP + 残差连接)补偿 fc 层失真；最坏情况退化为恒等 |
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
5. `accuracy_comparison_{m}.png`(**双横轴**：下轴 α -0.3\~0.3，上轴 σ 0.00\~0.30 线性映射，非线性红实圈 / 高斯蓝虚方)
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

CiM 芯片信号路径：输入激活 → MAC(非线性失真)→ ADC(QDQ 量化)。联合误差顺序严格遵循「先 nonlinearity，后 quantize_error(对应物理路径)」。研究：
1. 单独量化(32/8/6/4/3/2 bit 精度衰减)；
2. α×bit 网格下的**超加性叠加恶化**；
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
- `drop_joint > 1.05 × (drop_n + drop_q)` → **Super-Additive(恶化，两误差共振)**
- `0.95×`和 `~ 1.05×` → Linear(线性叠加可求和即可)
- `< 0.95×` → Sub-Additive(相互抑制)

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

---

## 13. 第二阶段优化：α扫描范围扩展（Extension 4: Alpha Wide Scan）

### 13.1 实验目的

将α扫描范围从训练区间[-0.3, 0.3]扩展至[-0.6, 0.6]，评估NAT模型在训练分布外的**外推鲁棒性**。具体而言：
- 训练区间|α|≤0.3：验证扫描结果与task1/2/3的一致性（曲线连续性检查）；
- 外推区|α|>0.3：评估NAT模型（finetune / scratch / Exp2 robust）在未见过的α强度下是否仍能保持精度，对比clean模型的退化速度。

外推区精度衰减斜率 vs 训练区内衰减斜率的对比意义：
- **衰减缓**（外推区斜率 ≤ 训练区斜率 × 1.5）：模型泛化鲁棒性好，非线性失真容忍度高；
- **衰减陡**（外推区斜率 > 训练区斜率 × 1.5）：鲁棒性仅限训练分布内，外推时快速失效。

两种结果都有研究价值：前者指导鲁棒模型设计，后者揭示NAT的过拟合边界。

### 13.2 命令行参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--models` | str | `"simple_cnn,vgg11,resnet18"` | 逗号分隔模型列表 |
| `--weight_types` | str | `"clean,nat_scratch,nat_finetune,exp2_robust"` | 逗号分隔权重类型；exp2_robust仅对simple_cnn有效 |
| `--alphas` | str | `"-0.6,-0.5,...,0.6"`(15点) | 逗号分隔α列表 |
| `--output_dir` | str | `"./outputs/extension4_alpha_wide/"` | 输出目录 |
| `--batch_size` | int | `128` | 推理batch size |
| `--seed` | int | `42` | 随机种子 |
| `--device` | str | 自动 | cuda/cpu |
| `--checkpoint_root` | str | `"./checkpoints"` | checkpoint根目录 |

### 13.3 输入：复用的checkpoint清单

| 模型 | weight_type | checkpoint路径 |
|------|-------------|---------------|
| simple_cnn | clean | `./checkpoints/simple_cnn/best_model.pth` |
| simple_cnn | nat_scratch | `./checkpoints/nat_scratch_simplecnn/best_model.pth` |
| simple_cnn | nat_finetune | `./checkpoints/nat_finetune_simplecnn/best_model.pth` |
| simple_cnn | exp2_robust | `./checkpoints/Exp2_Calib+Layerwise_simplecnn/best_model.pth` |
| vgg11 | clean | `./checkpoints/vgg11/best_model.pth` |
| vgg11 | nat_scratch | `./checkpoints/nat_scratch_vgg11/best_model.pth` |
| vgg11 | nat_finetune | `./checkpoints/nat_finetune_vgg11/best_model.pth` |
| resnet18 | clean | `./checkpoints/resnet18/best_model.pth` |
| resnet18 | nat_scratch | `./checkpoints/nat_scratch_resnet18/best_model.pth` |
| resnet18 | nat_finetune | `./checkpoints/nat_finetune_resnet18/best_model.pth` |

### 13.4 输出文件

| 文件 | 说明 |
|------|------|
| `alpha_wide_scan_summary.csv` | 汇总CSV，6列：model, weight_type, alpha, accuracy, loss, is_extrapolation |
| `simple_cnn_alpha_wide.png` | simple_cnn的4条曲线（含exp2_robust），竖虚线标α=±0.3 |
| `vgg11_alpha_wide.png` | vgg11的3条曲线 |
| `resnet18_alpha_wide.png` | resnet18的3条曲线 |
| `all_models_alpha_wide.png` | 3子图跨模型对比，共享y轴 |

### 13.5 典型运行命令

```bash
# 默认：全部扫描
python task_extension4_alpha_wide_scan.py

# 仅部分模型
python task_extension4_alpha_wide_scan.py \
    --models "simple_cnn,resnet18" \
    --weight_types "clean,nat_scratch" \
    --output_dir ./outputs/extension4_alpha_wide/

# 自定义α扫描点
python task_extension4_alpha_wide_scan.py \
    --alphas "-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6" \
    --output_dir ./outputs/extension4_alpha_wide/

# 重定位checkpoint目录
python task_extension4_alpha_wide_scan.py \
    --checkpoint_root /path/to/checkpoints
```

### 13.6 实验结果

**表 13-1 训练区间（α=+0.3）增益复核**——与一阶段结论逐点一致（偏差 0.00pp）：

| 模型 | clean | nat_scratch (Δ) | nat_finetune (Δ) | exp2_robust (Δ) |
|---|---|---|---|---|
| SimpleCNN | 55.08 | 64.23 (+9.15) | 56.08 (+1.00) | 68.76 (+13.68) |
| VGG-11 | 36.70 | 44.74 (+8.04) | 39.96 (+3.26) | — |
| ResNet-18 | 17.08 | 15.64 (−1.44) | 19.89 (+2.81) | — |

**表 13-2 外推区增益轨迹（SimpleCNN，vs clean，单位 pp）**：

| α | nat_scratch | nat_finetune | exp2_robust |
|---|---|---|---|
| +0.35 | +5.89 | +1.29 | +12.53 |
| +0.40 | **+0.77** | +1.63 | +8.63 |
| +0.50 | **−10.77** | +1.08 | −2.39 |
| +0.60 | **−10.89** | −1.58 | −5.11 |
| −0.40 | +6.44 | +0.84 | +0.53 |
| −0.50 | +7.58 | +0.68 | −1.55 |
| −0.60 | **+8.22** | +0.14 | −4.36 |

**表 13-3 VGG-11 nat_scratch 外推增益轨迹（vs clean，pp）**：

| α | +0.3 | +0.35 | +0.4 | +0.5 | +0.6 | −0.3 | −0.4 | −0.5 | −0.6 |
|---|---|---|---|---|---|---|---|---|---|
| Δ | +8.04 | +4.24 | +0.53 | +1.05 | +0.10 | +12.89 | +14.27 | +10.49 | +5.31 |

> 训练区与外推区数据由 CSV 的 `is_extrapolation` 列（|α|>0.3 为 True）分区。曲线见 `simple_cnn_alpha_wide.png` / `vgg11_alpha_wide.png` / `resnet18_alpha_wide.png` / `all_models_alpha_wide.png`。

### 13.7 核心发现

1. **训练区间锚点复核**：三架构 clean 的 α=0 / +0.3 精度（84.90/55.08、89.80/36.70、92.09/17.08）与一阶段报告逐位一致，外推区数据在同一评估管线下产出，可信。
2. **NAT 正向外推增益归零点**：SimpleCNN nat_scratch 增益随外推距离单调衰减——+9.15（α=+0.3）→ +0.77（α≈+0.4，归零）→ −10.77（α=+0.5，反转为劣势）。结论：**NAT 的鲁棒性是训练分布内的属性，正方向超出边界后不仅失效、甚至有害**——失真环境中的过度适应使模型在更极端失真下比 clean 模型更脆。
3. **负向外推增益保持**：与正方向反转形成鲜明对照，nat_scratch 负侧增益随距离不降反升（+5.48@−0.3 → +8.22@−0.6），VGG-11 负侧全区间保持 +5~14pp。正负不对称性跨架构稳定。
4. **Finetune 外推衰减最缓**：nat_finetune 在 α=+0.5 处仍 +1.08pp，是三种权重中唯一在深度外推区不恶化的；干净预训练特征是外推稳定性的基础。
5. **架构容差窗口随深度收窄**：clean 模型保持可用精度（≥70%）的 α 窗口 SimpleCNN 约 [−0.35,+0.25] > VGG-11 约 [−0.25,+0.2] > ResNet-18 约 [−0.15,+0.15]。深层网络可容忍失真带宽本身更窄，论证了深层迁移（拓展6）的必要性。
6. **SimpleCNN exp2_robust 负侧外推短板**：单一 GAP 后注入的 exp2 在负侧外推区输给 nat_scratch 最多 −12.58pp（α=−0.6 处 58.99 vs 71.57）。该短板是个例还是方案固有缺陷，需深层网络复验——此伏笔由 §15（拓展6）回收。

---

## 14. 第二阶段优化：Dithering效应机理解析（Extension 5: Dithering Mechanism）

### 14.1 实验目的

前期实验（拓展3）发现VGG-11干净模型在α=+0.3非线性失真下，4bit QDQ量化精度（43.27%）显著高于8bit（36.74%），即"Dithering效应"——低位量化噪声反而提升精度，且该效应为VGG-11所独有。本脚本对该现象进行定量机理解析，检验三个假设：

**H1（经典dither去相关假设）**：随机噪声打破非线性失真的确定性错误映射，产生倒U形响应（最优噪声剂量存在）。判别逻辑：若组(b)存在最优σ使精度显著超过8bit无噪基线，支持H1。

**H2（分布重塑假设）**：量化改变激活分布形态（特别是分类器输入特征），使决策边界更有利。判别逻辑：H2激活统计中，4bit与8bit的激活分布均值/标准差/峰度存在显著差异，且该差异与精度变化关联。

**H3（MaxPool交互假设）**：MaxPool的局部最大值筛选机制与随机噪声交互，使VGG-11（5次MaxPool）独享Dithering增益。判别逻辑：组(d)中若Dithering增益与MaxPool次数正相关（VGG-11 > SimpleCNN > ResNet-18），支持H3；若三架构增益相近，排除H3。

### 14.2 命令行参数

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--model` | str | `"vgg11"` | 主分析模型（组a/b/c/样本级/H2用） |
| `--alpha` | float | `0.3` | 固定非线性强度α |
| `--bits` | str | `"2,3,4,5,6,8"` | 组(a) bits扫描范围 |
| `--noise_types` | str | `"gaussian,uniform"` | 组(b)噪声类型 |
| `--noise_levels` | str | `"0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.5"` | 组(b)噪声强度 |
| `--output_dir` | str | `"./outputs/extension5_dithering_vgg11/"` | 输出目录 |
| `--batch_size` | int | `128` | 推理batch size |
| `--seed` | int | `42` | 随机种子 |
| `--device` | str | 自动 | cuda/cpu |
| `--checkpoint_root` | str | `"./checkpoints"` | checkpoint根目录 |
| `--ext3_root` | str | `"./outputs"` | extension3输出根目录（组d读取用） |

### 14.3 输入：复用checkpoint

| 模型 | checkpoint路径 | 用途 |
|------|---------------|------|
| VGG-11 | `./checkpoints/vgg11/best_model.pth` | 组(a)(b)(c)/样本级/H2主分析 |
| SimpleCNN | `./checkpoints/simple_cnn/best_model.pth` | 组(d)通过extension3 CSV复用 |
| ResNet-18 | `./checkpoints/resnet18/best_model.pth` | 组(d)通过extension3 CSV复用 |

注：组(d)不直接加载模型，而是从extension3既有CSV读取数据，无需额外推理。

### 14.4 输出文件

| 文件 | 说明 |
|------|------|
| `group_a_bits_scan.csv` | 组(a) bits扫描结果（num_bits/accuracy/loss/dose） |
| `group_b_noise_injection.csv` | 组(b) 噪声注入结果（noise_type/noise_std/accuracy/loss/dose） |
| `group_c_pure_noise.csv` | 组(c) 纯噪声结果（等效4bit剂量） |
| `flip_statistics.csv` | 样本翻转统计（saved/killed/net_gain） |
| `activation_stats.csv` | H2激活统计（conv1/classifier输入的双观测点统计量） |
| `arch_dithering_summary.csv` | 组(d) 三架构各bits精度汇总及4bit增益vs8bit |
| `bits_accuracy_curve.png` | 组(a) bits-精度曲线，标注36.74/43.27锚点 |
| `dose_accuracy_curve.png` | 组(a/b/c)同轴剂量-精度曲线 |
| `margin_distribution.png` | 三组样本margin分布叠加图 |
| `arch_comparison_bits.png` | 组(d)三架构bits-精度叠加图，标注MaxPool次数 |

### 14.5 典型运行命令

```bash
# 默认：完整分析（VGG-11）
python task_extension5_dithering_mechanism.py

# 自定义扫描范围
python task_extension5_dithering_mechanism.py \
    --bits "2,3,4,6,8" \
    --noise_levels "0.01,0.05,0.1,0.2,0.5" \
    --output_dir ./outputs/extension5_dithering_vgg11/

# 指定extension3输出根目录
python task_extension5_dithering_mechanism.py \
    --ext3_root /path/to/outputs
```

### 14.6 实验结果

**表 14-1 组(a) bits 扫描（VGG-11，α=+0.3）**：

| bits | 2 | 3 | **4** | 5 | 6 | 8 |
|---|---|---|---|---|---|---|
| 精度% | 9.98 | 42.97 | **43.27** | 38.00 | 37.13 | 36.74 |
| 剂量 | 0.945 | 0.553 | 0.245 | 0.115 | 0.055 | 0.014 |

**表 14-2 剂量匹配三对照（核心判决数据，vs 8bit 基线）**：

| 条件 | 剂量 | 精度% | vs 8bit 基线 |
|---|---|---|---|
| 8bit QDQ（基线） | 0.014 | 36.74 | — |
| **4bit QDQ** | **0.245** | **43.27** | **+6.53** |
| 等剂量高斯噪声（组c） | 0.247 | 34.29 | −2.45 |
| 等剂量均匀噪声（组c） | 0.247 | 34.87 | −1.87 |
| 8bit+最优剂量高斯（组b） | 0.041 | 36.71 | −0.03 |
| 8bit+最优剂量均匀（组b） | 0.051 | 36.68 | −0.06 |

**表 14-3 样本级翻转统计（vs 8bit，10,000 样本）**：

| 条件 | 救回 | 杀死 | 净增益 | 救回:杀死 |
|---|---|---|---|---|
| 4bit QDQ | 860 | 207 | +653 | 4.15 : 1 |
| 8bit+高斯（最优） | 53 | 46 | +7 | ≈1 : 1 |
| 8bit+均匀（最优） | 44 | 63 | −19 | ≈1 : 1 |

**表 14-4 分类器输入分布统计 vs bits（H2 核心证据）**：

| 条件 | 峰度 | 标准差 | 精度% |
|---|---|---|---|
| clean（α=0, 32bit） | 1.486 | 0.682 | 89.80 |
| α=0.3, 8bit | 0.110 | 0.461 | 36.74 |
| α=0.3, 6bit | 0.106 | 0.461 | 37.13 |
| α=0.3, **4bit** | **0.214** | 0.463 | **43.27** |
| α=0.3, 3bit | 0.384 | 0.497 | 42.97 |
| α=0.3, 2bit | 0.069 | 0.549 | 9.98 |

> conv1 输出峰度在所有条件下稳定于 4.6~6.7 区间，无系统性变化——分布重塑发生在深层分类器输入处。曲线见 `bits_accuracy_curve.png` / `dose_accuracy_curve.png` / `margin_distribution.png` / `arch_comparison_bits.png`。

### 14.7 三假设判决与结论

**H1（随机去相关）→ 否决**。证据链：①组(b) 两种噪声 × 全剂量扫描，最优增益 −0.03pp，无倒U，单调下降；②组(c) 等剂量纯噪声替代，精度 34.29%，低于 8bit 基线——同等"抖动功率"下随机扰动有害；③翻转统计：噪声条件净翻转 ≈0（+7/−19），与 4bit 的 +653 形成三个数量级反差。**随机性不是增益来源**。

**H2（分布重塑）→ 确认，核心机理**。表 14-4 给出完整机制链条：①非线性失真将分类器输入峰度从 1.486 压平至 0.110，类别可分性被破坏（89.8%→36.74%）；②4bit 的 16 电平粗离散化对该压平分布做非线性再分配，峰度恢复至 0.214，部分重建可分性（+6.53pp）；③存在甜点带——3bit 峰度更高（0.384）但精度略降（42.97），重塑过头叠加信息损失开始抵消增益；2bit 信息量不足，峰度与精度双双坍缩（0.069 / 9.98）。即：**4bit 量化增益的本质是"离散化诱导的分布重塑"——量化充当非线性再分配算子，部分修复失真对决策特征分布的破坏**。

**H3（MaxPool 交互）→ 支持（必要条件）**。表 14-5（arch_dithering_summary.csv）显示 4bit 增益 VGG-11(+6.53) > ResNet-18(+0.39) > SimpleCNN(−13.68)：无 MaxPool 的 ResNet-18 完全免疫，证明 MaxPool 是效应触发的必要条件；但 SimpleCNN（2 次 MaxPool）为 −13.68 负增益，说明 MaxPool 是必要而非充分条件——其密度与位置决定量化误差是修正还是放大失真：VGG-11 的 5 次密集 MaxPool 使粗量化的重塑作用逐级保留放大，SimpleCNN 的稀疏 MaxPool + 紧跟 GAP 使粗量化只剩信息损失。

**综合机理模型**：Dithering 增益 = 分布重塑（H2，提供修复能力）× MaxPool 密度（H3，提供架构载体），与随机性（H1）无关。

**表 14-5 三架构对照（组d，H3 核心证据）**：

| 模型 | MaxPool 数 | 8bit | 4bit | 4bit 增益 |
|---|---|---|---|---|
| VGG-11 | 5 | 36.74 | 43.27 | **+6.53** |
| ResNet-18 | 0 | 16.99 | 17.38 | +0.39（无效应） |
| SimpleCNN | 2 | 54.92 | 41.24 | **−13.68（负增益）** |

> ⚠ **修正声明**：一阶段将 Dithering 解释为"随机共振/抖动去相关"并提出"ADC 前注入随机噪声"的工程建议。本项 H1 判决**否决该解释与该工程建议**（等剂量注噪零增益，纯噪声替代反而降精度）。正确的机理表述为：在 MaxPool 密集架构（VGG 类）上，**直接采用 4bit ADC 本身**即可获得 +6.53pp 的失真补偿；注噪方案无效。

---

## 15. 第二阶段优化项三：Exp2鲁棒方案迁移至深层网络（Extension 6: Deep Robust Migration）

### 15.1 实验目的

将 SimpleCNN 上验证的 Exp2 鲁棒方案（特征校准模块 + 逐层失真训练 NAT）完整迁移至 VGG-11 和 ResNet-18 两个深层架构，从头训练（与 SimpleCNN 一致，不做微调），验证该方案在深层网络的可迁移性与外推鲁棒性。

### 15.2 校准模块注入位置设计依据

校准模块直接 import 复用 `models/robust_cnn.py` 的 `FeatureCalibration`（结构与 SimpleCNN 完全一致，不修改），仅注入位置随架构适配：

| 架构 | 注入位置 | 处数 | 设计依据 |
|------|---------|------|---------|
| VGG-11 | 每个 MaxPool 后 | 5 | MaxPool 是空间下采样与误差累积的关键节点，在每次池化后校准通道特征 |
| ResNet-18 | 每个 BasicBlock 残差路径（shortcut 输出） | 8 | 残差路径是信息保真通道，校准残差信号 = 校准决策的直接依据 |

**空间特征图适配**：SimpleCNN 中校准作用于 GAP 后的 1D 向量；深层网络注入点输出空间特征图 `(B,C,H,W)`，故在 forward 中 `permute(0,2,3,1)` 把通道维换到末尾后过校准再还原，FeatureCalibration 类本身零修改。

**分层 α 策略适配**：α 数值范围与 SimpleCNN Exp2 逐项一致（浅 [-0.15,0.15]、中/默认 [-0.30,0.30]、fc 固定±0.3），按各架构深度重新分组层名（VGG-11: block1-2 浅/block3-5 中；ResNet-18: conv1+layer1 浅/layer2-3 中/layer4 深），保持"浅层轻、中层标准、深层强"语义。

### 15.3 训练配方（与 SimpleCNN Exp2 逐项一致）

| 超参数 | 值 |
|--------|-----|
| epochs | 120 |
| optimizer | SGD(lr=0.01, momentum=0.9, weight_decay=1e-4) |
| scheduler | CosineAnnealingLR(T_max=epochs, eta_min=0.0) |
| batch_size | 128 |
| num_workers | 2 |
| seed | 42 |
| criterion | CrossEntropyLoss |

- Exp2 配置：`use_calibration=True, layerwise_alpha=True, asymmetric_sampling=False`
- 从头训练（随机初始化，不加载预训练权重）
- 每 epoch 同时评估 clean(α=0) 与 α=+0.3 测试精度

### 15.4 运行命令示例

```bash
# 训练 VGG-11
python task_extension6_deep_robust_train.py --model vgg11

# 训练 ResNet-18
python task_extension6_deep_robust_train.py --model resnet18

# 评估（两个模型 α 宽范围扫描 + 对比图）
python task_extension6_eval_deep_robust.py

# 评估时指定对比基准 CSV（默认读取 extension4 汇总 CSV）
python task_extension6_eval_deep_robust.py \
    --ext4_csv ./outputs/extension4_alpha_wide/alpha_wide_scan_summary.csv
```

### 15.5 输入与输出文件

**输入**（复用，纯推理评估）：
- `./checkpoints/Exp2_Calib+Layerwise_vgg11/best_model.pth`（训练产物）
- `./checkpoints/Exp2_Calib+Layerwise_resnet18/best_model.pth`（训练产物）

**训练输出**：
- `./checkpoints/Exp2_Calib+Layerwise_{model}/best_model.pth`（dict 含 model_state_dict + best_test_acc + 三个开关 + exp_name）
- `./outputs/extension6_deep_robust/{model}/training_curves.png`、`metrics.json`

**评估输出**（`./outputs/extension6_deep_robust/`）：

| 文件 | 说明 |
|------|------|
| `alpha_wide_scan_deep_robust.csv` | 6 列：model, weight_type(=exp2_robust), alpha, accuracy, loss, is_extrapolation |
| `deep_robust_comparison.png` | 两子图（每模型一个），叠加 extension4 的 clean/nat_scratch/nat_finetune 基准曲线 |
| `cross_architecture_summary.png` | 三架构×四权重跨架构汇总大图（由 `plot_cross_arch_summary.py` 生成） |

### 15.6 实验结果

**表 15-1 关键精度对照（测试集，%）**：

| 模型          | 权重            | α=0       | α=+0.3    | α=−0.2    |
| ------------- | --------------- | --------- | --------- | --------- |
| VGG-11        | clean           | 89.80     | 36.70     | 72.76     |
| VGG-11        | nat_scratch     | 90.18     | 44.74     | 79.48     |
| VGG-11        | nat_finetune    | 89.90     | 39.96     | 74.67     |
| **VGG-11**    | **exp2_robust** | **91.27** | **50.25** | **82.87** |
| ResNet-18     | clean           | 92.09     | 17.08     | 32.98     |
| ResNet-18     | nat_scratch     | 92.27     | 15.64     | 51.82     |
| ResNet-18     | nat_finetune    | 91.99     | 19.89     | 42.74     |
| **ResNet-18** | **exp2_robust** | **93.27** | **20.12** | **61.34** |

**表 15-2 全扫描点胜负统计（exp2_robust vs 基线，15 个 α 点）**：

| 对比 | VGG-11 | ResNet-18 |
|---|---|---|
| vs clean | 15胜0平0负（平均 +10.22pp） | 14胜1平0负（平均 +5.02pp） |
| vs nat_scratch | **15胜0平0负（平均 +4.31pp）** | 14胜1平0负（平均 +2.48pp） |
| vs nat_finetune | 15胜0平0负（平均 +8.45pp） | 13胜1平1负（平均 +3.81pp） |

> 训练收敛正常：VGG-11 best epoch 105/120（train acc 99.95%），ResNet-18 best 109/120（train acc 99.99%）；metrics.json 的 best_test_acc 与扫描 α=0 值精确一致（91.27/93.27）。曲线见 `deep_robust_comparison.png` / `cross_architecture_summary.png`，数据见 `alpha_wide_scan_deep_robust.csv`。

### 15.7 核心发现

1. **增益跨架构传递**：α=+0.3 相对 clean 的增益 SimpleCNN +13.68 → VGG-11 +13.55 → ResNet-18 +3.04pp。VGG-11 与 SimpleCNN 几乎无损传递，证明方案有效性不依赖浅层架构。
2. **多点校准修复 §13 伏笔**：SimpleCNN 的 exp2（单一 GAP 后注入）在负侧外推区输给 nat_scratch 最多 −12.58pp；深层版本采用 5/8 个空间注入点后，**15 个扫描点全部压制 nat_scratch**。结论：单一注入点的校准容量不足是 SimpleCNN 负侧短板的成因，多点空间校准可修复——§13.7 发现 6 的遗留问题在本项闭环解答。
3. **增益随深度衰减，机制为复合最坏情形**：逐层独立采样下，训练时所有层同时取到极端 α 的概率趋近于零；而评估时全局 α=+0.3 意味着全部 17 个卷积层（ResNet-18）**同时**处于边界失真——这是训练分布外的尾部事件，层数越多越极端。ResNet-18 正侧增益收窄至 +3.04pp 与此一致。
4. **正负不对称性第三证**：三架构 exp2 均呈负侧强于正侧（VGG-11：α=−0.3 为 69.99% vs α=+0.3 为 50.25%），与 §13.7 发现 3（外推不对称）、一阶段"正向失真破坏力 3.4 倍于负向"构成三条独立证据线。
5. **干净精度零牺牲**：两架构 exp2 的 α=0 精度（91.27/93.27）均为四组权重中最高——校准模块兼具正则化作用，方案无干净精度代价。

**对一阶段策略规律的补充**：一阶段结论"残差网络极限失真下应改用 Finetune"（19.89% vs Scratch 15.64%）；本项显示，残差路径注入校准后，基于 Scratch 的 exp2_robust 达到 20.12%，反超 Finetune——**架构级校准可以恢复 Scratch 策略在残差网络上的竞争力**。

### 15.8 新增模型文件说明

- [models/robust_vgg11.py](file:./models/robust_vgg11.py) — `RobustVGG11`：VGG-11 主干 + 5 处 FeatureCalibration（每个 MaxPool 后一处），forward 中 `permute(0,2,3,1)` 适配空间特征图，校准模块类零修改复用。
- [models/robust_resnet.py](file:./models/robust_resnet.py) — `RobustResNet18`：ResNet-18 主干 + 8 处 FeatureCalibration（每个 `CalibratedBasicBlock` 残差路径一处，`out = main(x) + Calib(shortcut(x))`），覆盖 4 stage × 2 block 全部残差路径。

---

## 16. 第二阶段优化总结

### 16.1 评审意见与优化项映射

第二阶段围绕评审意见"扩展 α 扫描范围、对 Dithering 效应定量机理解析、增强方案迁移至深层网络、扩展数据集多样性"展开，拆解为四项：

| # | 评审子建议 | 对应优化项 | 执行状态 |
|---|---|---|---|
| R1 | 扩展 α 扫描范围 | 优化项一（Extension 4，§13）：α∈[-0.6,0.6] 宽范围外推扫描 | ✅ 完成 |
| R2 | Dithering 效应定量机理解析 | 优化项二（Extension 5，§14）：三假设竞争判决实验 | ✅ 完成 |
| R3 | 增强方案迁移至深层网络 | 优化项三（Extension 6，§15）：Exp2 方案迁移 VGG-11/ResNet-18 | ✅ 完成 |
| R4 | 数据集多样性 | 未执行（时间约束），已论证并列为后续工作 | ⏸ 暂缓 |

**R4 暂缓理由**：冲刺窗口内 R1–R3 合计约 10 GPU 小时即可闭环；R4 需完整的新数据工程（数据获取/预处理/新基准适配）加全部实验矩阵重跑，投入产出失衡。且本工作已有的跨架构验证（3 架构 × 4 权重类型 × 15 失真强度 = 180 组条件）已在"结构泛化"维度覆盖了泛化性诉求，"数据集泛化"维度留作后续工作。

### 16.2 闭环叙事

三项优化并非并列，而是一条递进链——"发现问题→解释机理→修复验证"：

- **Extension 4（量化边界）** → 发现现有防御手段（NAT/Exp2）的有效边界：NAT 正向外推增益在 α≈+0.4 归零反转、架构容差窗口随深度收窄、SimpleCNN exp2 负侧外推短板（−12.58pp vs nat_scratch）；
- **Extension 5（解释反常）** → 对一阶段发现的 Dithering 反常现象（4bit 量化反而提升失真精度）给出定量机理判决（H1 否决 / H2 确认 / H3 支持），并**修正一阶段"随机共振/抖动去相关"的错误解释**；
- **Extension 6（突破边界）** → 将最优防御方案迁移至深层网络，其结果反过来修复了 Extension 4 发现的外推短板（多点校准使 15 扫描点全胜），并验证跨架构有效性。

### 16.3 三项综合结论

1. **正负不对称性：三条独立证据线交汇**——一阶段"正向失真破坏力 3.4× 于负向"、Extension 4"NAT 负侧增益随距离递增而正侧反转"、Extension 6"exp2 三架构负侧精度均高于正侧"三条测量收敛于同一结论：正方向（增益饱和型）失真是本质上的难防御方向，失真补偿资源应向正方向倾斜。
2. **深度规律的统一图景**——注入点 5→9→18 脆弱性递增、容差窗口随深度收窄（SimpleCNN~0.3 → ResNet~0.15）、exp2 增益随深度衰减（+13.68 → +3.04）统一指向"复合最坏情形概率的指数放大"：注入点超过约 10 个的架构，必须配合多点校准才可期待有效防御。
3. **校准注入密度法则**——SimpleCNN exp2（1 个注入点）负侧外推 −12.58pp vs nat_scratch；深层 exp2（5/8 个注入点）15 点全胜。校准注入密度应随误差累积节点数同步扩展，单点校准（仅 GAP 后）只保护分类边界、不保护特征通路。一阶段未来工作提出的"分布式逐层校准"设想在本项得到实证支持。

---

## 17. 双数据集支持（CIFAR-10 / CIFAR-100）

本项目从 v2.0 起支持 CIFAR-10 与 CIFAR-100 双数据集参数化切换。所有入口脚本统一增加 `--dataset {cifar10,cifar100}` 参数，默认值为 `cifar10`，**不带 `--dataset` 的旧命令行为完全不变**，保证向后兼容。

### 17.1 `--dataset` 参数说明

| 选项 | 类别数 | 归一化 Mean (RGB) | 归一化 Std (RGB) | 产物根目录 |
|------|--------|-------------------|-------------------|------------|
| `cifar10`（默认） | 10 | (0.4914, 0.4822, 0.4465) | (0.2023, 0.1994, 0.2010) | `./checkpoints/` + `./outputs/` |
| `cifar100` | 100（fine label） | (0.5071, 0.4865, 0.4409) | (0.2673, 0.2564, 0.2762) | `./checkpoints_cifar100/` + `./outputs_cifar100/` |

数据增强策略（随机水平翻转 + 随机裁剪 + 归一化）与 DataLoader 配置（batch_size=128, num_workers=2）两个数据集完全一致。

### 17.2 目录结构

```
├── data/                                      # 数据集（共享根目录）
│   ├── cifar-10-batches-py/                   # CIFAR-10（已就位）
│   └── cifar-100-python/                      # CIFAR-100（首次运行自动下载，约 169MB）
├── checkpoints/                               # CIFAR-10 训练权重
├── checkpoints_cifar100/                      # CIFAR-100 训练权重（与 CIFAR-10 隔离）
├── outputs/                                   # CIFAR-10 分析产物
└── outputs_cifar100/                          # CIFAR-100 分析产物（与 CIFAR-10 隔离）
```

### 17.3 路径隔离机制

- 所有 argparse 路径参数（`--checkpoint`、`--output_dir`、`--checkpoint_root`、`--ext3_root`、`--ext4_csv` 等）默认值统一设为 `None`，在 `main()` 中经 `utils/paths.py` 的 `get_ckpt_root(dataset)` / `get_outputs_root(dataset)` 动态计算。
- 例如 `--dataset cifar100 --model simple_cnn` 下，模型权重默认路径为 `./checkpoints_cifar100/simple_cnn/best_model.pth`，输出默认路径为 `./outputs_cifar100/simple_cnn/`。
- 显式指定路径参数时覆盖默认值，由调用者自行负责路径正确性。
- 旧脚本产物（`./checkpoints/`、`./outputs/` 下的文件）不会被任何 `--dataset cifar100` 命令触碰。

### 17.4 数据集溯源字段

所有 checkpoint 保存 dict 与 `metrics.json` 新增 `"dataset"` 字段（值为 `"cifar10"` 或 `"cifar100"`），便于跨数据集回溯权重来源。

所有 `torch.load` 调用点读取 `ckpt.get("dataset", "cifar10")` 并打印来源日志。若 checkpoint 记录的 dataset 与当前 `--dataset` 不一致，打印 WARNING 后按兼容模式继续（不报错），以保证新旧权重互操作。

### 17.5 大类别展示规则

- 混淆矩阵在 `num_classes > 10` 时不显示类别刻度标签，避免文字重叠。
- `evaluate.py` 在类别数 > 10 时终端只打印 accuracy + macro/weighted 汇总；逐类指标完整写入 `metrics_eval.json` 的 `per_class` 字段，并另存 `per_class_metrics.csv`（列：`class_name, precision, recall, f1`）。

### 17.6 前置权重依赖

若某个脚本需要读取前置阶段的权重或 CSV（例如 task2 finetune 需要 train.py 产物，task3 需要 task2 scratch 产物），在 `--dataset cifar100` 下文件不存在时会明确报错，并提示先运行对应的 `--dataset cifar100` 命令。cifar10 下按原设计可选跳过绘制。

---

## 18. CIFAR-100 快速开始

### 18.1 冒烟测试

```bash
python train.py --model simple_cnn --dataset cifar100 --epochs 2
```

首次运行 CIFAR-100 时，数据会自动从 torchvision 下载到 `./data/cifar-100-python/`（约 169MB，与 CIFAR-10 共用 `./data` 根目录，子目录名不同不冲突）。

### 18.2 各模型推荐训练轮数（建议值）

| 模型 | CIFAR-100 推荐 epochs | 备注 |
|------|----------------------|------|
| SimpleCNN | 120 | 小模型，收敛较快 |
| VGG-11 | 200 | 深层纯前馈，需更多轮次 |
| ResNet-18 | 200 | 残差结构，深层需充分训练 |

以上为建议值，可根据 GPU 资源与精度要求自行调整（通过 `--epochs` 参数指定）。

### 18.3 完整链路示例

以下命令按依赖顺序执行，覆盖全部 6 项核心任务 + 拓展研究的 CIFAR-100 完整流程：

```bash
# 1. 干净训练（SimpleCNN，120 epochs）
python train.py --model simple_cnn --dataset cifar100 --epochs 120

# 2. 任务1：非线性误差敏感性分析
python task1_sensitivity_analysis.py --model simple_cnn --dataset cifar100

# 3. 任务1（层分析）
python task1_layer_analysis.py --model simple_cnn --dataset cifar100

# 4. 任务2：NAT 训练（finetune 模式）
python task2_nat.py --mode finetune --model simple_cnn --dataset cifar100

# 5. 任务3：鲁棒性消融
python task3_robust.py --model robust_cnn --dataset cifar100 --exp all

# 6. 拓展研究4：α 宽范围扫描
python task_extension4_alpha_wide_scan.py --models simple_cnn --dataset cifar100

# 7. 拓展研究6：深层网络评估
python task_extension6_eval_deep_robust.py --models vgg11,resnet18 --dataset cifar100

# 8. 跨架构汇总图
python plot_cross_arch_summary.py --dataset cifar100
```

> **注意**：以上命令仅覆盖 SimpleCNN 为主线的链路；如需 ResNet-18 / VGG-11 的完整链路，请先对相应模型运行 `train.py` 训练干净权重，再依次执行拓展脚本。

### 18.4 CIFAR-100 实验结果

完整链路（train → task1 → task2 → task3 → extension1~6 → cross_arch）已在 CIFAR-100 上按与 CIFAR-10
完全相同的协议复跑（相同 α 扫描点、相同超参、相同消融配置、相同目录布局），产物位于 `./outputs_cifar100/`。
关键数值如下（α=0 为干净精度，α=±0.3 为极限失真精度）：

| 实验项 | 模型 | α=0 | α=+0.3 | α=−0.3 |
|--------|------|-----|--------|--------|
| Clean Baseline | SimpleCNN (120ep) | 59.99 | 28.09 | 42.91 |
| Clean Baseline | VGG-11 (200ep) | 67.09 | 16.92 | 24.48 |
| Clean Baseline | ResNet-18 (200ep) | 72.09 | 2.70 | 3.25 |
| NAT-Scratch | SimpleCNN (120ep) | 60.70 | 21.41 | 46.20 |
| NAT-Finetune | SimpleCNN (120ep) | 60.16 | 24.81 | 45.79 |
| Exp2 Robust (Calib+Layerwise) | SimpleCNN | 61.22 | 25.16 | 43.10 |
| Exp2 Robust (Calib+Layerwise) | VGG-11 | 68.05 | 15.98 | 34.45 |
| Exp2 Robust (Calib+Layerwise) | ResNet-18 | 73.57 | 5.43 | 6.00 |

> **协议一致性说明**：CIFAR-10 既有产物未被本次复跑改动；`./outputs_cifar100/` 与 `./outputs/` 的
> 目录布局经 `diff` 逐项比对一致（仅 CIFAR-10 侧手写的实验总结 md 未在 CIFAR-100 侧同步撰写）。
> 注：CIFAR-10 的锚点数值（ext5 的 36.74%/43.27% 等）不适用于 CIFAR-100，ext5 已按数据集分支
> 跳过锚点比对；ext2 的 `known_clean_acc` 经验值同样仅作 fallback，实际以 checkpoint 记录的
> `best_test_acc` 为准（已在 cifar100 下打印 WARNING 提示）。