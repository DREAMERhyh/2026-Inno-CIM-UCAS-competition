# M4 收尾 runbook（深层主干方差）

> M4 是**唯一可能让 Central Claim 收缩**的实验。训练链在跑，训练完成后按本文执行。
> **执行者：claude-cf**。总管：**claude-0b**（`claude-e9` / `claude-08` 已终止）。
>
> ⚠ **2026-09-24 16:30 修订**——由 claude-cf 查出、总管逐条核实后写入。**四处实质性修正**：
> 1. **§2 的 `--arch` 写错了**（`vgg11` 应为 `robust_vgg11`）。这是**会真出错**的一处。
> 2. **§3「主干级标准差」取哪一行原先没写死**——`a_original_fc` 那一行的读出级标准差**恒为 0.0000**，
>    拿它当主干级会得到"方差为零"的假结论。已写死取法。
> 3. **§1 的配对容差** 0.01 是**已知现象**（= 10000 张里差 1 张），已给操作线。
> 4. **§0 的耗时**是**区间**不是点估计（105~140 s/epoch）。
>
> **当前在跑的长任务**：`logs_v7_m4_train_part2.log`，由**脱离会话的守护进程**接管
> （`tools/v7_m4_watchdog.ps1`，见 `docs/V7_COORDINATION.md` §五·补一）。**旧的 `bqrdsv1wd` 已死。**

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
**实测耗时**：
- VGG-11/120ep = **110 分钟/次**
- **ResNet-18/200ep = 5.8 ~ 7.8 小时/次**（**是区间，不是点估计**）

> ⚠ **为什么是区间**：这个数先后被低估**三次**，每次都是同一个错——**拿"训练吞吐"当"epoch 时间"**。
> ① 按冒烟 52 s/epoch 外推 ⇒ 说 60 分钟，实测 110；
> ② 按 `5.45 it/s` 的**纯训练吞吐**（71.8 s/epoch）外推 ⇒ 说 2.4 小时；
> ③ 实测口径：tqdm 的 100% 行给出**纯训练 82 s/epoch**，但**整 epoch 还要加验证开销**
>    （实测 686 秒内完成「epoch 1–6 + epoch 7 的 30 batch」⇒ 训练 498 s、验证/开销约 31 s/epoch）
>    ⇒ **≈113 s/epoch**；另一条独立粗测（422 秒 / 3 epoch）给 **140.7 s/epoch，是上界**；
>    总管的下界读数 105 s/epoch（419 秒 / 3.98 epoch）。
> ⇒ **取 105~140 s/epoch**，单次 200ep = **5.8~7.8 小时**。
>
> **推论**：run3 约 **21:50~23:40** 完、run4 约**次日 03:40~07:20** 完，
> 4 次读出再串行 +100 分钟 ⇒ **全部产出约在次日 05:20~09:00**。**跨夜，不要假设白天会完。**

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
**必须等于该主干 `metrics.json` 的 `best_test_acc`**。对不上 = 接错主干，停。

> ⚠ **容差的操作线（2026-09-24 定；两个会话各自独立查到同一现象）**：
> `exp3_resnet18_s42` 的 `a_original_fc`@α=0 = **72.69**，而 `metrics.json` = **72.70**，
> **差正好 0.01 = 10000 张里的 1 张**。vgg11 s42 侧则是 68.05 vs 68.05**逐位相同**。
> ⇒ **这是已知现象（跨设备/口径的 1 张样本差），不是接错主干。**
> **操作线：0.01 偏移记 PASS 不报警；≥0.02 才停。**

---

## 2. 读出适配（4 次，串行）

```bash
<python> v3_readout_repair.py --arch robust_vgg11 \
  --ckpt checkpoints_cifar100/Exp2_Calib+Layerwise_vgg11_s43/best_model.pth \
  --backbone_tag exp2_vgg11_bs43_ep24 --dataset cifar100 \
  --alpha 0.3 --epochs_nat 24 --n_train 20000 --seed 42
```
其余三次照此模板（**`--arch robust_resnet18`** 时 ckpt 指向 `Exp3_FullRobust_resnet18_s<seed>/`）。

> 🔴 **`--arch` 必须是 `robust_*`，不是 `vgg11` / `resnet18`**（2026-09-24 修正，claude-cf 查出）。
> 本 runbook 原写 `--arch vgg11`，**那是错的**。三重证据：
> ① `Exp2_Calib+Layerwise_vgg11{,_s43}/best_model.pth` 的 state_dict 含 `calib1`…`calib5` 键；
> ② **`models/vgg11.py` 里 `calib` 出现 0 次，`models/robust_vgg11.py` 里 20 次**（resnet 同构：`resnet.py` vs `robust_resnet.py`）；
> ③ `v3_readout_repair.py:52–66` 的 `build_backbone` 里，`vgg11` 与 `robust_vgg11` 是**两条不同分支**
>    （前者建 `VGG11`、后者建 `RobustVGG11`）。
>
> **失效方式**：`load_state_dict` 默认 `strict=True`（`v3_readout_repair.py:248`）⇒ **响亮报错、不静默出错**，
> 但会白烧一次启动+特征提取；**更危险的是它会诱导后来者用 `strict=False` 去"修"——
> 那才会静默丢掉全部校准模块，而校准正是 Exp2 配方的全部要点。**
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

> 🔴 **哪一行算"主干级"——原 runbook 没写死，这里写死**（2026-09-24，claude-cf 提出、总管确认）。
>
> **问题**：`a_original_fc` 那一行在**读出级**的标准差**恒为 0.0000** ——
> 固定同一个 s42 主干、只换读出种子，三个文件的该行**逐位相同**（原头被冻结，前向是确定性的，
> **读出种子动不了它**）。而本 runbook §3 引用的对照值「读出级 clean ±1.85 / 尾部 ±3.40」
> **只能来自 `e_readout_NAT` 行**。
> ⇒ **若拿 `a_original_fc` 去算"主干级 vs 读出级"，会得到一个恒为 0 的读出级量**，
> 比较在数学上就不成立。
>
> **写死的取法**：
> 1. **主判据取 `e_readout_NAT` 行**（读出适配后的尾部精度）——这才是与"读出级 ±1.85/±3.40"同类的量；
> 2. **同时另报 `a_original_fc` 行**（纯主干量：它决定 `tab:nat` 基线列 16.92 / 2.70 稳不稳，
>    是有独立价值的数，只是**不能**拿去和读出级比）；
> 3. **若两行对分支判断不一致 ⇒ 按保守方向报**（**任一 ≥ 3 pp 即报分支 ③**）。
>
> ⚠ **本条取法是执行者的判断、不是预登记原文**（预登记只写了"主干级标准差"四个字，未定义取哪一行）。
> **报告里必须显式标注这一点**，不得写成"按预登记判据"。

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

1. **台账 §16.1**（四段式：假设/判据/结果/判决）
   —— **落点已定，不再有歧义**（2026-09-24 裁决）：
   - **claude-cf 把 §16.1 全文草稿写进自己的领地**：`outputs_cifar100/v7_M4_audit/M4_LEDGER_16_1_DRAFT.md`
   - **由台账持锁者 claude-65 并入 `experiments_ledger.md`**（**单一写者原则**，见 `docs/V7_COORDINATION.md` §二）
   - ⚠ **claude-e9 早先说的 `docs/V7_M4_TAIL.md` 已作废**——那个会话已终止，且 `docs/**` 归总管
2. 论文同步：§3.4 的读出适配数字若需加置信区间、§5 局限第 2 条（深层主干单 seed）可从局限中移除
3. `docs/V7_COORDINATION.md` 状态板更新 —— **由总管 claude-0b 做**

> **写权归一（2026-09-24）**：`docs/**` 归**总管**；工人一律写自己的领地。
> `claude-cf` 独占 `outputs_cifar100/v7_M4_audit/`。**报告不要落进 `docs/`。**

---

## 5. 若分支③命中

**停下，不自行继续。** 按止损线 1：
「任何结果会改写 Central Claim 或既有 L1–L9 的表述」→ 写报告、报用户、等裁决。

**要准备的报告内容**：主干级标准差的实际值、它是读出级的几倍、
受影响的句子有哪些（`paper/journal/main.tex` 的哪些行、`论文v3.md` 的哪些行）、
以及**收缩后的替代表述草案**（例如"在已测的三个主干种子上成立"）。
