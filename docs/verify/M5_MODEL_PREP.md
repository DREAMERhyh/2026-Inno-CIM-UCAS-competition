# M5 第 4 容量档：模型设计与接线说明（**准备完成，零训练**）

> 写者：M5 准备会话（被 `claude-0c` 派活）｜首版 2026-09-26 01:55，**V2 改版 02:20**
> 文件领地：`models/simple_cnn_wide.py`（新建）+ 本文件（新建）。
> 🔴 **未修改任何既有文件** —— 需要的改动全部写在 §4，由总管执行。
> 本文件不含任何训练产物：全程只做了 CPU batch=2 的前向/单次反向，**未跑任何 epoch**。

---

## 0. 三十秒摘要

**交付**：`models/simple_cnn_wide.py` 的 **`SimpleCNNWideV2`**（现行接线用它）。
CIFAR-100 主干 **1,399,428** 参数（含校准 1,662,340），实测 **MACs = 133,318,656 = 0.87 × VGG-11**。
CPU 前向 `(2,3,32,32) → (2,100)` 通过。

**一条历史**：同文件里的 `SimpleCNNWide`（V1）是**首版设计，已被 V2 取代**，保留供追溯、**不接线**。
V1 纯加宽不动拓扑，实测 MACs = 455,067,648 = **VGG-11 的 2.98 倍** ⇒ 单次训练 2~5.5 h，超出窗口。
**根因：参数量可以堆，计算量堆不起** —— 详见 §2.2。

**仍需注意的两条**：

**风险 ①（会静默失效）**：`get_depth_group()` 对新模型名**不报错**，但会把 `conv1/conv2` 判成 `middle`（应为 `shallow`）、把 `fc` 判成 `middle`（应为 `fc`）。
⇒ **不改它，第 4 点的训练协议就与 SimpleCNN 点不同**。见 §4.2。

**风险 ②（口径，不得掩盖）**：V2 与 SimpleCNN 之间差的是 **容量 + 第二次下采样的位置**两个变量
（SimpleCNN 的 conv4 在 16×16，V2 的在 8×8）。**M5 报告里不得声称"只差容量"**。见 §1.2。

---

## 1. 设计

### 1.1 三点背景

现三点在台账 §14.1：

| 点 | 主干 | 容量（主干参数） | 拓扑 |
|---|---|---|---|
| 1 | SimpleCNN | 0.24M | 浅层纯前馈（4 conv / 2 pool） |
| 2 | VGG-11 | 9.2M | 深层纯前馈（8 conv / 5 pool） |
| 3 | ResNet-18 | 11.2M | 残差 |

### 1.2 V2 的结构与依据

**结构（输入 32×32）**：

```
conv1 (3→32)      @32×32    ← 与 SimpleCNN 逐层相同
conv2 (32→64)     @32×32    ← 与 SimpleCNN 逐层相同
   ↓ MaxPool               (32→16)
conv3 (64→256)    @16×16
   ↓ MaxPool               (16→8)   ← ⚠ 关键：这次 pool 在 conv4 **之前**
conv4 (256→512)   @8×8
   ↓ GAP → (B,512) → [FeatureCalibration(512→256)] → FC
```

**依据**：

1. **下采样次数仍是 2 次**（与 SimpleCNN 相同）。变的是**第二次 pool 的位置**：
   SimpleCNN 在 conv4 之后，V2 在 conv3 之后 ⇒ conv4 从 16×16 落到 **8×8**。
   这一项就是 MACs 降 3.41 倍的全部来源。
2. **conv1/conv2 与 SimpleCNN 逐层完全相同**（3→32、32→64，均在 32×32）⇒ 浅层变量锁死。
3. **只加宽后两层**（conv3 64→256、conv4 256→512；SimpleCNN 是 128/128）。
   不在 32×32 上加宽 —— 那里每 1 个参数要花 1024 次 MAC，是 V1 翻车的根因。
4. **不加深度**：深度变量已被 VGG-11 占住。
5. **不加残差**：拓扑变量已被 ResNet-18 占住。
6. **校准模块沿用 SimpleCNN Exp2 的方案**（GAP 后单一注入点，`hidden_dim = in_dim/2`），
   不是 VGG-11 的"每个 MaxPool 后 5 处"。换校准方案等于又引入一个变量。

⚠ **拓扑代价（必须如实写进 M5 报告）**：V2 与 SimpleCNN 之间**不是"只差容量"**，
还差了"第二次下采样的位置"。这是 MACs 硬约束下的必然取舍，不是疏忽。
⇒ 报告里若要归因，只能说"容量 + 下采样位置"，**不得写成"只差容量"**。

### 1.3 为什么是 (32, 64, 256, 512) 这组通道

在"2 次 pool、conv2 后与 conv3 后各一次"的拓扑下扫过 c1/c2/c3/c4 的组合（约束：参数 ∈[1M,3M] 且 MACs ≤1.9e8），
`(32,64,256,512)` 是其中**最干净**的一组：

| 候选 (c1,c2,c3,c4) | 主干参数 | MACs | 备注 |
|---|---|---|---|
| (224,640) 系列 | 1,504,580 | 135.4M | 参数最接近几何中点，但**通道不是 2 的幂**，像调过参 |
| (192,768) 系列 | 1,536,004 | 133.1M | 同上，且 c4/c3 = 4 倍跳 |
| **`(32,64,256,512)`** | **1,399,428** | **133.1M** | ✅ **全是 2 的幂，通道比例 2×/4×/2×** |

参数量 1.40M ≈ **√(0.24M × 9.2M) = 1.49M**（SimpleCNN 与 VGG-11 的几何中点，差 5.8%）。
选 2 的幂而非更贴近中点，是因为 M5 的横轴是**秩损失**不是精确参数量，
而"非整幂通道"会让审稿人问"为什么是 224/640"。

---

## 2. 参数量与计算量（**全部实测，非估算**）

命令：`PYTHONIOENCODING=utf-8 /d/anaconda3/envs/pytorch_env/python`（裸 python 是 Inkscape 的，无 torch）。
- 参数量 = `sum(p.numel() for p in model.parameters())`
- **MACs = forward hook 实测**（累加每个 `Conv2d`/`Linear` 的 `out.numel() × in_channels × k²`，batch=2 取单样本均值）

### 2.1 参数量

| 配置 | 主干 | 含校准 |
|---|---|---|
| **V2**，`num_classes=100`（CIFAR-100，M5 用） | **1,399,428** | **1,662,340** |
| **V2**，`num_classes=10`（CIFAR-10） | 1,353,258 | 1,616,170 |
| ~~V1~~（已取代），CIFAR-100 | ~~1,603,236~~ | ~~1,866,148~~ |

- 校准模块单独计 **262,912** = `FeatureCalibration(512, 256)` = `512×256+256 + 256×512+512`。
- **口径提示**：台账 §14.1 表里的 0.24M / 9.2M / 11.2M 是**主干**（不含校准）口径
  （对照：`logs_v7_m4_train.log` 里 ext6 打印 RobustVGG11 = 9,889,796 = 9,277,284 主干 + 612,512 校准）。
  ⇒ **报告时两个数都要给，并写明是哪一个**（本项目因"数的定语没跟着搬"栽过多次）。

### 2.2 🔴 MACs：V1 翻车与 V2 的修正（**hook 实测**）

| 模型 | 参数量 | **MACs / 样本** | vs SimpleCNN | vs VGG-11 |
|---|---|---|---|---|
| SimpleCNN | 254,084 | 76,395,008 | 1.00× | 0.50× |
| VGG-11 | 9,277,284 | 152,815,616 | 2.00× | 1.00× |
| ~~SimpleCNNWide V1~~ | 1,866,148 | ~~455,067,648~~ | 5.96× | **2.98×** ❌ |
| **SimpleCNNWideV2** | **1,662,340** | **133,318,656** | **1.75×** | **0.87×** ✅ |

**V1 为什么贵（算术必然，不是实现失误）**：
**卷积的 参数/MACs 比 = 1/(H_out·W_out)**。V1 把参数量的大头（1.18M，占主干 84%）压在 **16×16** 的 conv4 上，
一个参数要花 256 次 MAC；同样的参数放在 8×8 上只花 64 次（1/4），放在 4×4 上只花 16 次（1/16）。
⇒ 在"32×32 输入、8×8 输出、2 次 pool"的 V1 拓扑下，**要做到 1~3M 参数就必然把 conv 堆在高分辨率上**，
换成任何别的通道组合都一样（试算过多个变体：MACs 只降 13%，且参数量跌破 1M）。

**V2 的修正**：把 conv4 从 16×16 移到 8×8 ⇒ 同样 1.18M 参数的 MACs 从 302M 降到 **75.5M**。
**MACs 总体降 3.41 倍，参数量只降 12%。**

**⚠ 一条方法论警告（本次亲历）**：约束搜索时我先用**手算公式**估过一版，
公式里把 SimpleCNN 的 conv4 误当成在 8×8（实际在 16×16），**参考点的 MACs 算错了一倍**。
⇒ **MACs 一律用 hook 实测**；公式只能用来筛候选，不能用来报数。

---

## 3. CPU 前向冒烟怎么做

**已完成（本次）** —— 结果如下，接手者可直接复现：

```bash
cd "d:/deeplearning_practice/存算一体"
PYTHONIOENCODING=utf-8 /d/anaconda3/envs/pytorch_env/python - <<'PY'
import torch
from models.simple_cnn_wide import SimpleCNNWideV2
m = SimpleCNNWideV2(num_classes=100, use_calibration=True).eval()
with torch.no_grad():
    y = m(torch.randn(2, 3, 32, 32))
print("out:", tuple(y.shape), y.dtype, y.device)
PY
```

**实测输出**：`out: (2, 100) torch.float32 cpu` ✓ 判据命中。

另外四项已验证（同批，均 CPU batch=2，秒级）：

| 验证 | 结果 |
|---|---|
| `get_readout_module(m)`（`v3_readout_repair.py`） | `Linear in_features=512 out_features=100` ✓ |
| 逐层独立采样 hook（复刻 `_train_one_epoch` 逻辑）单次 fwd+bwd+step | 7 个 hook 注册 → 前向/反向/更新全通过 → `finally` 移除后**残留 0** ✓ |
| `register_nonlinearity_hooks(m, 0.3)`（`_evaluate_alpha` 路径） | 通过，hook 移除后残留 0 ✓ |
| `α=0` 与 clean 前向 | **逐位相同**（`torch.equal` = True）✓ |

⚠ **训练耗时的冒烟是另一件事**，必须在 GPU 上跑 `--epochs 1`（§6 命令①）。
M4 已验证这个外推可靠：冒烟 52 s/epoch vs 实际 110 min / 120 ep = 55 s/epoch。

**预期的成本**：V2 的 MACs 是 VGG-11 的 **0.87 倍** ⇒ **没有理由比 VGG-11 的 110 分钟更慢**。
但 M4 记录过该训练**卡在 dataloader**（GPU 利用率连采 `3 2 0 3 3 3 87 96`），
若瓶颈仍在 dataloader，则新旧模型的差距会被压缩 —— **仍以冒烟实测为准，我不给点估计**。

---

## 4. 接进训练的 5 处改动（**diff 全文；我不改，交总管**）

> 三处是硬失败（不改会报错），两处是**静默失效**（不改会照跑但结果错）。**5 处都要改。**
> ✅ 总管已批准全部 5 处（2026-09-26 02:05）。

### 4.1 `task_extension6_deep_robust_train.py` — `get_model()`（第 75~82 行附近）

```python
    if model_name == "vgg11":
        from models.robust_vgg11 import RobustVGG11
        return RobustVGG11(num_classes=num_classes)
    elif model_name == "resnet18":
        from models.robust_resnet import RobustResNet18
        return RobustResNet18(num_classes=num_classes)
    elif model_name == "wide_cnn":                                          # ← 新增
        from models.simple_cnn_wide import SimpleCNNWideV2                  # ← 新增（V2）
        return SimpleCNNWideV2(num_classes=num_classes, use_calibration=True)  # ← 新增（V2）
    else:
        raise ValueError(f"不支持的模型名称: '{model_name}'，当前支持 vgg11, resnet18, wide_cnn")
```

⚠ `use_calibration=True` 必须与第 626 行的硬编码 `use_calibration=True`（trainer 的**元数据**）一致，
否则 `metrics.json` 里写的开关与实际结构不符（**静默不一致**，本项目最怕的失效类型）。

### 4.2 🔴 `task_extension6_deep_robust_train.py` — `get_depth_group()`（第 85~126 行）**静默失效点**

在 `elif model_name == "resnet18":` 整块之后、函数末尾的 `return "middle"` 之前插入：

```python
    elif model_name == "wide_cnn":
        # 层名与 SimpleCNN 同构：沿用 task3_robust.py 里 SimpleCNN Exp2 的分组语义
        #   conv1/conv2 → shallow [-0.15,0.15]；conv3/conv4 → middle [-0.30,0.30]；
        #   fc → 固定 ±0.3；calibration.net.* → middle（由函数开头的 calibration 判断兜住）
        if module_name in ("conv1", "conv2"):
            return "shallow"
        if module_name in ("conv3", "conv4"):
            return "middle"
        if module_name == "fc":
            return "fc"
        return "middle"
```

**为什么不改会错（实测证据，见 §5.2）**：不改时 `get_depth_group('wide_cnn','conv1')` 返回 `'middle'`
⇒ α 从 `[-0.15,0.15]` 变成 `[-0.30,0.30]`；`fc` 也从"固定 ±0.3"变成"均匀采样"。
**不报错、不影响训练完成、只是协议变了** —— 正是本项目最怕的静默失效。

**零回归验证（已做）**：把上面的分支加进函数后重跑 11 个既有 case
（vgg11 的 features.0/4/8、classifier.1、calib3.net.0；resnet18 的 conv1、layer1.0、layer2.0、layer4.1、fc、calib1.net.0）
⇒ **11/11 与改动前逐字相同**。

### 4.3 `task_extension6_deep_robust_train.py` — `--model` 的 `choices`（第 521~524 行）

```python
        choices=["vgg11", "resnet18", "wide_cnn"],
        help="训练模型：vgg11、resnet18 或 wide_cnn",
```
（不改 ⇒ argparse 直接拒收 `--model wide_cnn`。这是**响亮失败**，不是静默。）

### 4.4 `v3_readout_repair.py` — `build_backbone()`（第 71 行 `raise ValueError` 之前）

```python
    if name == "robust_wide_cnn":
        from models.simple_cnn_wide import SimpleCNNWideV2
        return SimpleCNNWideV2(num_classes=nc, use_calibration=True)
```

⚠ **这一条同时服务两个脚本**：`v5_probe_rank_collapse.py` 第 99~100 行
`from v3_readout_repair import build_backbone, get_readout_module` —— **只改这里，秩探针自动支持**。

### 4.5 `v3_readout_repair.py` — `--arch` 的 `choices`（第 217~220 行）

```python
                    choices=["simple_cnn", "vgg11", "resnet18", "simple_cnn_mp",
                             "robust_vgg11", "robust_resnet18", "robust_cnn",
                             "robust_wide_cnn"],
```

> **`models/__init__.py` 不用改**：它只 export `SimpleCNN`，而 VGG11/ResNet18/RobustCNN **也都不在里面**——
> 训练/读出脚本一律用 `from models.xxx import YYY` 的完整路径。跟着既有惯例走。

---

## 5. 校准 / 层独立采样机制兼容性（判据三）

### 5.1 结论：**不需要自己的 robust 版本，能直接复用**

| 机制 | 结论 | 依据 |
|---|---|---|
| `FeatureCalibration` | **直接 import 复用，零修改** | 它完全参数化（`in_dim`/`hidden_dim`），`robust_vgg11.py` 就是这么用的（`from models.robust_cnn import FeatureCalibration`）。实测 `isinstance(m.calibration, FeatureCalibration)` = True，参数 262,912 |
| 层独立采样 | **直接复用，无需新代码** | `_train_one_epoch` 遍历 `self.model.named_modules()`，对每个 `Conv2d`/`Linear` 独立采样挂 hook。V2 有 **7 个**这样的模块：`conv1,conv2,conv3,conv4,calibration.net.0,calibration.net.2,fc` —— **与 RobustCNN 的数量和结构逐项相同** |
| 校准模块的失真注入 | 自动生效 | `calibration.net.0/.2` 是 `Linear`，会被同一个循环挂上 hook（与 SimpleCNN 行为一致） |
| 读出层定位 | 直接可用 | `get_readout_module` 先查 `model.fc`，V2 正是 `self.fc = nn.Linear(512, num_classes)` |
| 设备/协议 | 无特殊要求 | 纯 `nn.Module`，无自定义 init、无自定义 hook 注册 |

**唯一需要新写的代码是 §4.2 的 `get_depth_group` 分支**（因为它把"层名 → 分组"硬编码成了按 model_name 分派的结构），
**不是校准模块本身**。

**为什么不需要 `load_clean_state_dict` 这类方法**：`robust_vgg11.py` 的那个方法用于"把 clean VGG-11 权重装进 RobustVGG11"
（评估/对比场景）。M5 的 ext6 流程是**从头随机初始化训练**（`get_model` 不加载任何权重），用不到。
若将来要做 clean 口径对照，正确做法是训一个 `use_calibration=False` 的版本，**不是**加载 clean 权重。

### 5.2 实测的层名 → 分组映射（V2，patch 前 vs patch 后）

| 模块名 | **patch 前**（现状） | **patch 后**（§4.2） | SimpleCNN Exp2 协议 |
|---|---|---|---|
| `conv1` | middle `[-0.30,0.30]` ❌ | **shallow** `[-0.15,0.15]` ✓ | shallow |
| `conv2` | middle `[-0.30,0.30]` ❌ | **shallow** `[-0.15,0.15]` ✓ | shallow |
| `conv3` | middle ✓ | middle ✓ | middle |
| `conv4` | middle ✓ | middle ✓ | middle |
| `calibration.net.0` | middle ✓ | middle ✓ | middle（task3 的 `else` 兜底） |
| `calibration.net.2` | middle ✓ | middle ✓ | middle |
| `fc` | middle ❌（均匀采样） | **fc** ✓（固定 ±0.3） | fc |

patch 后与 SimpleCNN/RobustCNN 的实际映射**逐项相同**（同一段代码跑两个模型比对过）。

---

## 6. 完整命令草稿

```bash
cd "d:/deeplearning_practice/存算一体"
PY="D:/anaconda3/envs/pytorch_env/python.exe"     # ⚠ 裸 python 是 Inkscape 的，无 torch
```

### ① 冒烟（**必须先跑**，1 epoch，量 s/epoch）

```bash
PYTHONIOENCODING=utf-8 "$PY" task_extension6_deep_robust_train.py \
  --dataset cifar100 --model wide_cnn --epochs 1 --seed 42 --tag _smoke_m5 \
  > logs_v7_m5_smoke.log 2>&1
```
产物落 `checkpoints_cifar100/Exp2_Calib+Layerwise_widecnn_smoke_m5/` 与
`outputs_cifar100/extension6_deep_robust/widecnn_smoke_m5/`（**与正式产物隔离，零覆盖**）。

### ② 正式训练（120ep，与 Exp2 配方原生 epoch 一致）

```bash
PYTHONIOENCODING=utf-8 "$PY" task_extension6_deep_robust_train.py \
  --dataset cifar100 --model wide_cnn --epochs 120 --seed 42 --tag _m5 \
  > logs_v7_m5_train.log 2>&1
```

**目录名推导**（源码 553~568 行，**别猜**）：
`model_tag = "wide_cnn".replace("_","") = "widecnn"` ⇒
- checkpoint：`checkpoints_cifar100/Exp2_Calib+Layerwise_widecnn_m5/best_model.pth`
- 输出：`outputs_cifar100/extension6_deep_robust/widecnn_m5/metrics.json`

🔴 **开跑前先查内存**（提交可用 < 4 GB 就别开）。长任务记得用 `Start-Process` 配方脱离会话
（`docs/V7_COORDINATION.md` §五·补一），**不要用 `run_in_background`**。

### ③ 秩统计（`v5_probe_rank_collapse.py` —— 纯前向，快）

```bash
PYTHONIOENCODING=utf-8 "$PY" v5_probe_rank_collapse.py \
  --dataset cifar100 --arch robust_wide_cnn \
  --ckpt checkpoints_cifar100/Exp2_Calib+Layerwise_widecnn_m5/best_model.pth \
  --tag _widecnn_m5
```
产物：`outputs_cifar100/v5_rank_collapse/rank_stats_widecnn_m5.csv`（`--tag` 是直接拼在 `rank_stats` 后的）
**秩损失% = (rank_eff@0.0 − rank_eff@+0.3) / rank_eff@0.0**

### ④ 读出适配 **×3 seed**（🔴 **3 次，不是 1 次** —— 理由见 §7）

```bash
for S in 42 43 44; do
  PYTHONIOENCODING=utf-8 "$PY" v3_readout_repair.py \
    --arch robust_wide_cnn \
    --ckpt checkpoints_cifar100/Exp2_Calib+Layerwise_widecnn_m5/best_model.pth \
    --backbone_tag widecnn_m5_ep24 --dataset cifar100 \
    --alpha 0.3 --epochs_nat 24 --n_train 20000 --seed $S
done
```
> ⚠ **bash 的 `for` 循环违反"脱离会话"要求**（会话一死循环就断）。正式跑请用
> `tools/v7_m4_readout.ps1` 的写法：`$runs` 列表 + `Start-Process -Wait` 串行。
> `for` 循环只适合人工盯着跑。

产物（`tag = n20000` + seed≠42 才加 `_s{seed}`，源码 293 行）：
```
outputs_cifar100/v3_readout_repair/readout_repair_c100_widecnn_m5_ep24_alpha+0.30_n20000.csv
outputs_cifar100/v3_readout_repair/readout_repair_c100_widecnn_m5_ep24_alpha+0.30_n20000_s43.csv
outputs_cifar100/v3_readout_repair/readout_repair_c100_widecnn_m5_ep24_alpha+0.30_n20000_s44.csv
```

### ⑤ 配对闸门（照 M4 的规矩，每个 CSV 都做）

`a_original_fc@α=0` 必须等于 `metrics.json` 的 `best_test_acc`；
**差 ≤ 0.01（=10000 张里 1 张）记 PASS，≥0.02 停**（浮点边界要用"张数"比较，别直接比 0.01）。

---

## 7. 恢复率口径：**3 个读出种子的平均**（一条推断，三条证据）

台账 §14.1 第 2261 行写"恢复率 = readout-NAT@α ÷ clean"，**没写几个 seed**。
但三条既有点**全部**只有在"3 seed 平均"下才能复现（逐位反推）：

| 主干 | 台账恢复率 | × clean | = 反推的读出值 | 已知的 3-seed 读出值 | 单跑 s42 CSV 的值 |
|---|---|---|---|---|---|
| SimpleCNN | 72.2% | × 59.99 | **43.31** | **43.31**（`M4_EXECUTION_PLAN.md` §5 "SimpleCNN 43.31 vs 26.57"） ✓ | 41.16（→68.6% ✗） |
| VGG-11 | 70.2% | × 68.05 | **47.77** | **47.79**（论文印的 $\pm$3.18 那一格） ✓ | 46.98（→69.0% ✗） |
| ResNet-18 | 36.9% | × 72.69 | **26.82** | **26.82**（同上） ✓ | — |

**clean 取 `a_original_fc` 行的 α=0.0 列**（三条分别是 59.99 / 68.05 / 72.69，逐位对得上）。

⇒ **推断：恢复率 = mean(3 seed 的 `e_readout_NAT`@α=+0.3) ÷ (`a_original_fc`@α=0.0)**。
三条独立吻合，不像巧合。**总管已复核并采纳（2026-09-26 02:05）**。
执行时仍建议把 3 个 CSV 的原始值**逐个列出**，让读的人能自己复算。

---

## 8. 未能验证的部分（诚实清单）

| # | 未验证项 | 为什么没法验 | 影响 |
|---|---|---|---|
| 1 | **训练耗时** | 红线禁用 GPU 训练 | **最大的一项**。MACs 0.87× VGG-11 ⇒ 没有理由比 110 min 慢，但**仍以冒烟为准**（§3 末） |
| 2 | GPU 上的前向/反向 | 同上 | CPU 已验证通路正确；GPU 只是设备不同，风险低 |
| 3 | **新主干能训到多少精度 / 秩损失落在哪** | 这正是实验要测的 | **直接决定 M5 判据能不能判出形状** |
| 4 | 读出 3-seed 的方差 | 未跑 | 恢复率会带一个 ±，未知大小 |
| 5 | §4 的 5 处 diff **未实际应用** | 领地限制（我不改既有文件） | 我**模拟**了 §4.2 的补丁逻辑并做了 11 例零回归；其余 4 处是照源码逐行写出、未经真实执行 |
| 6 | §7 的"3 seed 平均"口径 | 曾是反推 | 已由总管复核采纳 |
| 7 | CUDA 数值行为 | 红线 §3.4（设备口径） | 判据若要"逐位一致"，必须同设备 |
| 8 | **V2 的通道组合是"扫出来的"而非"推出来的"** | — | 见 §1.3：候选里还有参数更贴近几何中点的，我按"2 的幂 + 通道比例自然"选了这一组。这是一个**设计判断**，不是唯一解 |
| 9 | 新点能否落在曲线的拐点区间（33.3%~58.7%） | 无法预控 | 若落在 20.6~33.3%，四点可能仍定不出拐点 |

---

## 9. 跑前零覆盖核验（**已做，2026-09-26**）

四个目标路径**全部 absent**（实测）：

```
absent  checkpoints_cifar100/Exp2_Calib+Layerwise_widecnn_m5
absent  outputs_cifar100/extension6_deep_robust/widecnn_m5
absent  outputs_cifar100/v3_readout_repair/readout_repair_c100_widecnn_m5_ep24_alpha+0.30_n20000.csv
absent  outputs_cifar100/v5_rank_collapse/rank_stats_widecnn_m5.csv
```

既有 `v5_rank_collapse/` 只有 `rank_stats.csv` / `rank_stats_exp2vgg11.csv` / `rank_stats_exp3resnet18.csv`；
`widecnn` 字样在既有产物里**零命中**。

---

## 10. 红线自查

| # | 红线 | 执行情况 |
|---|---|---|
| 1 | 不跑 GPU 训练 | ✅ 全程 CPU；只有 batch=2 的前向与**单次**反向（§3 表），**未跑任何 epoch**；未调 `.cuda()` |
| 2 | 不修改既有文件 | ✅ 只新建了 `models/simple_cnn_wide.py` 与本文件；`git status` 未跟踪项与开工前一致 |
| 3 | 内存 | ✅ 未并起多进程；每次只跑一个 python 进程，模型 ≤1.9M 参数 |
| 4 | 用 conda python | ✅ 一律 `/d/anaconda3/envs/pytorch_env/python` |

---

## 11. 版本变更记录

| 时刻 | 变更 | 触发 |
|---|---|---|
| 2026-09-26 01:55 | 首版：`SimpleCNNWide`（纯加宽，通道 ×2） | 原简报 |
| 2026-09-26 02:20 | **改版 V2**：第二次 pool 提前 + 只加宽后两层；V1 保留不接线 | 总管指出 MACs 455M = 2.98× VGG-11 超预算，给出硬约束 MACs ≤1.9e8 |
