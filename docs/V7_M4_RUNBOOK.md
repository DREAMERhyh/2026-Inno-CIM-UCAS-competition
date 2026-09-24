# M4 收尾 runbook（深层主干方差）

> M4 是**唯一可能让 Central Claim 收缩**的实验。训练链在跑（后台 ID `bqrdsv1wd`，
> 日志 `logs_v7_m4_train.log`）。训练完成后按本文执行。
> **执行者：claude-e9**（M4 由我发起，§16.1 由我写——届时先问 claude-65 是否在写台账）。

---

## 0. 训练链概况

| # | 配置 | epochs | tag | 产物目录 |
|---|---|---|---|---|
| 1 | vgg11 / exp2 / seed 43 | 120 | `_s43` | `checkpoints_cifar100/Exp2_Calib+Layerwise_vgg11_s43/` |
| 2 | vgg11 / exp2 / seed 44 | 120 | `_s44` | `…_vgg11_s44/` |
| 3 | resnet18 / exp3 / seed 43 | 200 | `_s43` | `checkpoints_cifar100/Exp3_FullRobust_resnet18_s43/` |
| 4 | resnet18 / exp3 / seed 44 | 200 | `_s44` | `…_resnet18_s44/` |

**命令**（已跑）：
```bash
<python> task_extension6_deep_robust_train.py --dataset cifar100 --model <vgg11|resnet18> \
  --variant <exp2|exp3> --epochs <120|200> --seed <43|44> --tag _s<seed>
```
**实测耗时**：VGG-11/120ep = **110 分钟/次**（初估 60 分钟偏乐观；不要按冒烟外推排期）。

**中途核验**（每次跑完即做，别等四次全完）：
- `checkpoints_cifar100/<exp>_<arch>_s<seed>/best_model.pth` 存在
- `outputs_cifar100/extension6_deep_robust/<arch>[_exp3]_s<seed>/metrics.json` 的
  `total_epochs` 与 `best_test_acc` 合理（s42 对照：vgg11 68.05 / resnet18-exp3 72.70）

---

## 1. ⚠ 命名陷阱（**必须避开**）

`v3_readout_repair.py` 的产物名是
`readout_repair_{ds}_{bb}_alpha{α}_{tag}.csv`，其中 `bb = --backbone_tag or --arch`。

**问题**：既有文件 `readout_repair_c100_exp2_vgg11_s4{2,3,4}_ep24_*.csv` 的 `a_original_fc`
在 α=0 **全是 68.05** —— **同一个 s42 主干**，它们变的只是**读出种子**。
但 tag 里写着 `s43`/`s44`，**读起来像"主干种子 43/44"**。

⇒ **新主干一律用 `bs<seed>`（backbone seed）命名**：

| 主干 | 读出 `--backbone_tag` |
|---|---|
| s42（既有） | `exp2_vgg11_s42_ep24`（**保持原样，不改既有产物名**） |
| **s43（新）** | **`exp2_vgg11_bs43_ep24`** |
| **s44（新）** | **`exp2_vgg11_bs44_ep24`** |
| resnet18-exp3 同构 | `exp3_resnet18_bs4{3,4}_ep24` |

**核验手段（每次必做）**：读出跑完，CSV 里 `a_original_fc` 行在 α=0 的值
**必须等于该主干 `metrics.json` 的 `best_test_acc`**（容差 0.01）。对不上 = 接错主干，停。

---

## 2. 读出适配（4 次，串行）

```bash
<python> v3_readout_repair.py --arch vgg11 \
  --ckpt checkpoints_cifar100/Exp2_Calib+Layerwise_vgg11_s43/best_model.pth \
  --backbone_tag exp2_vgg11_bs43_ep24 --dataset cifar100 \
  --alpha 0.3 --epochs_nat 24 --n_train 20000 --seed 42
```
其余三次照此模板（`--arch resnet18` 时 ckpt 指向 `Exp3_FullRobust_resnet18_s<seed>/`）。
**参数依据**：与既有一致 —— `alpha 0.3` / `epochs_nat 24` / `n_train 20000` / `seed 42`（读出种子）。
`ep24` 按既有惯例**烘进 backbone_tag**，因为它并不由 `epochs_nat` 生成。

**为什么读出种子固定 42**：本实验要测的是**主干那一份方差**，所以读出侧必须固定。
既有 s42 主干的读出也是 seed 42（文件名无 `_s` 后缀即 seed 42），两者配对成立。

---

## 3. 计算与裁决

**三组样本**（每组 3 个主干种子，读出侧固定 seed 42）：

| 主干 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| VGG-11 / Exp2 | `readout_repair_c100_exp2_vgg11_s42_ep24_*_n20000.csv` | **本次新增** `…_bs43_ep24_…` | **本次新增** `…_bs44_ep24_…` |
| ResNet-18 / Exp3 | `readout_repair_c100_exp3_resnet18_s42_ep24_*` | 本次新增 | 本次新增 |

**读数**：每个文件取 `a_original_fc` 行（原始头）与 `e_readout_NAT` 行（读出适配）的
α=0 与 α=+0.3 两列。

**统计**：**总体标准差 ddof=0**（与 §13.10 定的口径一致），对
① clean ② α=+0.3 ③ drop@0.3 三个量，主干级与读出级分别算。

**预登记判据（跑前写死，见 `docs/NEXT_STAGE_TASKS.md` M4）**：

| 分支 | 条件（主干级标准差） | 判决 |
|---|---|---|
| ① | **≤1.5 pp** | Central Claim 锁定，可写"跨主干种子稳健" |
| ② | **1.5~3 pp** | 合格，但论文需加条件（"在已测主干上成立"） |
| ③ | **≥3 pp** | **触发止损线 1** —— 「读出适配超过全网络配方」这句在深层上要收缩，**立即停下报告用户** |

**对照**：已知的读出级方差（固定主干换读出种子）为 clean ±1.85 / 尾部 ±3.40（ResNet-18）。
**若主干级方差显著大于读出级，说明此前把读出级当成了全部方差。**

---

## 4. 交付

1. **台账 §16.1**（四段式：假设/判据/结果/判决）—— **由 claude-e9 写**，
   动手前先发消息问 claude-65 是否在写台账（单一写者原则）
2. 论文同步：§3.4 的读出适配数字若需加置信区间、§5 局限第 2 条（深层主干单 seed）可从局限中移除
3. `docs/V7_COORDINATION.md` 状态板更新

---

## 5. 若分支③命中

**停下，不自行继续。** 按止损线 1：
「任何结果会改写 Central Claim 或既有 L1–L9 的表述」→ 写报告、报用户、等裁决。

**要准备的报告内容**：主干级标准差的实际值、它是读出级的几倍、
受影响的句子有哪些（`paper/journal/main.tex` 的哪些行、`论文v3.md` 的哪些行）、
以及**收缩后的替代表述草案**（例如"在已测的三个主干种子上成立"）。
