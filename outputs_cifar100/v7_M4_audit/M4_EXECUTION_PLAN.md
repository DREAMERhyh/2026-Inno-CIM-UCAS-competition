# M4 收尾执行计划（执行者：M4 工人会话；快照 2026-09-24 16:30）

> **写给两类读者**：① 我自己（执行清单）；② **若我会话死掉，接手的下一个会话**。
> 上一轮已经发生过"会话被杀、训练链一起被带走"的事故，所以本文**必须能让别人接住**。
> 长期规则见 `docs/V7_COORDINATION.md`，M4 流程见 `docs/V7_M4_RUNBOOK.md`。
> **本文不修改任何既有文件，只记录我实际要跑什么。**

---

## 0.0 🔴 事故记录：run3 崩溃 + 守护进程把"崩溃"误判成"完成"（2026-09-24 16:32 起）

**这是我的收尾工作必须知道的第一件事**，因为它**污染了一个我要用的产物**。

### 时间线（全部有日志/文件时戳支撑）

| 时间 | 事件 | 出处 |
|---|---|---|
| 15:59:59 | run3 首次启动（PID 25276），日志 `logs_v7_m4_train_part2.log` | 进程 CreationDate |
| 16:32 | **run3 在 epoch 16 处 `CUDA error: out of memory` 崩溃** | 日志停在 16:32:46 / 1389007 字节 |
| 16:32:52–16:36:56 | 守护进程连续 5 次未见训练进程 ⇒ 判定"真死了" | `logs_v7_m4_watchdog.log` |
| 16:37:26 | 守护进程**接手**，重启 seed 43 | 同上 |
| **16:44:11** | 守护进程记 **`seed 43 finished`** ——**只过了 6 分 45 秒** | 同上 |
| 16:44:11 | 守护进程**接着启动 seed 44**（PID 36656，命令行确认 `--seed 44`） | 进程 CreationDate + CommandLine |

### 🔴 核心缺陷：完成判据是"进程退出了"，不是"产物到位了"

6 分 45 秒不可能是完成（完整 run 需 5.8~7.8 小时）。真相是 **seed 43 又崩了一次，守护进程把崩溃当成了完成**。
⇒ 后果链条：seed 44 大概率同样崩溃 → 再被判成 "finished" → 守护进程手上 `run(s) left` 归零 → **正常退出**
⇒ **run3 与 run4 都没完成，且无人再接手。**

### 🔴 由此产生的陷阱（**直接威胁收尾**）

`checkpoints_cifar100/Exp3_FullRobust_resnet18_s43/best_model.pth` 的 mtime = **16:43:38**，
它是**那次只跑了 6 分 45 秒的崩溃残骸**，**不是 200-epoch 产物**。
**而我的读出流程正是要用这个文件。** 若不设闸门，会**拿残骸算出看起来正常的数字**。

### ⇒ 新增硬闸门（**加在读出之前，不改预登记判据**）

**读出任一主干的前置条件**：该主干的
`outputs_cifar100/extension6_deep_robust/<run_dir>/metrics.json` **存在**，
且其 **`total_epochs` == 200**（vgg11 侧为 120）。
**不满足 ⇒ 不跑那个主干的读出，报总管。**

**另：日志已不可作为判据。** 重启的 run 缺输出重定向，`logs_v7_m4_train_part2.log` 永久停在崩溃前内容。
**⇒ 一切进度/完成判断只用 `metrics.json` 与进程命令行。**

### 事故当时的环境读数（解释为什么会崩）

物理可用内存一度到 **0.03 GB**；`FreePhys 755.9 MB / FreeVirt 1419 MB / 页面文件 6588 / 42872 MB`。
崩溃被判定为**系统内存 / WDDM 撑不住**，不是显存爆（显存 3.8/8 GB）。

**⇒ 我的自律**：全程零 torch；在守护进程修好前只做只读核验（`ls`/`cat`/`grep`/`stat`）。

---

## 0. ⚠ 我对 runbook 的两处修正

> **2026-09-24 16:30 更新**：两处修正**均已由总管 `claude-0b` 独立核实并写进
> `docs/V7_M4_RUNBOOK.md` 的 16:30 修订块**（含它自己查的第三重证据：
> `v3_readout_repair.py:52-66` 的 `build_backbone` 里 `vgg11` 与 `robust_vgg11` 是两条不同分支）。
> **本节保留原始论证，因为接手者需要"为什么"而不只是"是什么"。**
> 下面记录的"runbook 原文如此"指的是 16:30 修订**之前**的版本。

### 修正 1：`--arch` 必须是 `robust_vgg11` / `robust_resnet18`，**不是** `vgg11` / `resnet18`

runbook §2 的命令模板写的是 `--arch vgg11`。**这是错的**，理由（三重证据）：

| 证据 | 内容 |
|---|---|
| ① 权重文件结构 | `Exp2_Calib+Layerwise_vgg11{,_s43}/best_model.pth` 的 state_dict 含 **`calib1`…`calib5`** 五组键（用 zipfile 直接列 .pth 内键名得到，零 torch 开销） |
| ② 模型类 | `models/vgg11.py` 含 `calib` 字样 **0** 处；`models/robust_vgg11.py` 含 **20** 处。resnet 侧同理：`resnet.py` 0 处 vs `robust_resnet.py` 11 处 |
| ③ 历史日志 | 既有 s42 读出跑（`logs_v5_p01_exp2_vgg11_s42_ep24.log` 等）第 1 行逐字为 `[readout-repair] 主干架构=robust_vgg11`；resnet 侧为 `robust_resnet18` |

**失效方式**：`load_state_dict` 默认 `strict=True`（`v3_readout_repair.py:248`），
传 `--arch vgg11` 会在加载时报 "Unexpected key(s)"，是**响亮失败、不是静默出错**
（已读过源码确认，未实际触发）。但它会浪费一次完整的启动+特征提取时间，且诱导后来者用
`strict=False` 去"修"，那才会静默丢掉全部校准模块。**⇒ 一律用 robust_ 前缀。**

### 修正 2：预登记判据里"主干级标准差"读哪一行，**预登记没写死**（必须显式声明取法）

runbook §3 说"每个文件取 `a_original_fc` 行与 `e_readout_NAT` 行的 α=0 与 α=+0.3 两列"，
判据表只说"主干级标准差"，**没说是哪一行的标准差**。这在本实验里不是小事：

**实测**：`a_original_fc` 行在**读出级**（固定 s42 主干、换读出种子 42/43/44）的标准差
**恒为 0.0000**（三个文件的该行逐位相同）。因为原头是冻结的、前向确定性的，
**读出种子根本动不了它**。而 runbook 引用的对照值"读出级 clean ±1.85 / 尾部 ±3.40（ResNet-18）"
只能来自 `e_readout_NAT` 行。

⇒ **为了让"主干级 vs 读出级"是同类相比，主判据取 `e_readout_NAT` 行**（与 ±1.85/±3.40 同源）。
⇒ **同时另报 `a_original_fc` 行**（它是纯主干量，决定 tab:nat 里基线列 16.92 / 2.70 稳不稳）。
⇒ **若两行对分支判断不一致，按保守方向报**（任一 ≥3 pp 即报分支③，因为止损线的存在意义
就是防过度声称，误报比漏报安全）。

🔴 **强制标注要求（总管 16:30 下的硬要求，理由是本项目 §16.8 出过同类缺陷）**：
> 预登记原文只有"主干级标准差"四个字，**未定义取哪一行**。
> ⇒ **报告里必须显式写明"取哪一行是执行者在裁决时补的定义，不是预登记原文"**，
> **不得写成"按预登记判据"**。否则半年后会被当成预登记原文引用。
> 本项目已有前车之鉴：**"判据里用了一个前提未检验的量"**就是 §16.8 那次审计查出的缺陷类型。

---

## 1. 执行时实际要跑的命令（4 次，**严格串行**）

```bash
cd "d:/deeplearning_practice/存算一体"
PY="D:/anaconda3/envs/pytorch_env/python.exe"     # ⚠ 裸 python 是 Inkscape 的，无 torch

# ① vgg11 s43 主干
PYTHONIOENCODING=utf-8 "$PY" v3_readout_repair.py \
  --arch robust_vgg11 \
  --ckpt checkpoints_cifar100/Exp2_Calib+Layerwise_vgg11_s43/best_model.pth \
  --backbone_tag exp2_vgg11_bs43_ep24 --dataset cifar100 \
  --alpha 0.3 --epochs_nat 24 --n_train 20000 --seed 42

# ② vgg11 s44 主干
#    --arch robust_vgg11 --ckpt .../Exp2_Calib+Layerwise_vgg11_s44/best_model.pth
#    --backbone_tag exp2_vgg11_bs44_ep24

# ③ resnet18 s43 主干
#    --arch robust_resnet18 --ckpt .../Exp3_FullRobust_resnet18_s43/best_model.pth
#    --backbone_tag exp3_resnet18_bs43_ep24

# ④ resnet18 s44 主干
#    --arch robust_resnet18 --ckpt .../Exp3_FullRobust_resnet18_s44/best_model.pth
#    --backbone_tag exp3_resnet18_bs44_ep24
```

**其余参数四次全同**：`--alpha 0.3 --epochs_nat 24 --n_train 20000 --seed 42`
（与既有 s42 一致；`--seed 42` 是**读出种子**，本实验刻意固定它，因为要测的是主干那一份方差）。

**参数依据（不靠猜，逐条有源）**：

| 参数 | 值 | 依据 |
|---|---|---|
| `--alpha` | 0.3 | runbook §2 |
| `--epochs_nat` | 24 | runbook §2；`ep24` 按既有惯例烘进 tag |
| `--n_train` | 20000 | runbook §2；既有文件名里的 `n20000` |
| `--seed` | 42 | runbook §2「读出侧固定 seed 42」；且 seed=42 时文件名**不带** `_s42` 后缀（源码 293 行），这正是既有 s42 文件名的形态 |
| `--arch` | robust_* | **见 §0 修正 1** |
| 设备 | cuda | `v3_readout_repair.py:232` 默认 `cuda if available`；既有 s42 读数按红线 §3.4 必须同设备 |

**预期产物文件名**（源码 291–295 行推导：`tag = n{n_train}` + （seed≠42 才加 `_s{seed}`））：

```
outputs_cifar100/v3_readout_repair/readout_repair_c100_exp2_vgg11_bs43_ep24_alpha+0.30_n20000.csv
outputs_cifar100/v3_readout_repair/readout_repair_c100_exp2_vgg11_bs44_ep24_alpha+0.30_n20000.csv
outputs_cifar100/v3_readout_repair/readout_repair_c100_exp3_resnet18_bs43_ep24_alpha+0.30_n20000.csv
outputs_cifar100/v3_readout_repair/readout_repair_c100_exp3_resnet18_bs44_ep24_alpha+0.30_n20000.csv
```

**跑前核验（2026-09-24 16:29 已做）**：`v3_readout_repair/` 下 `bs4*` 文件数 = **0** ⇒ 四个目标名零撞车 ✓

### 1.1 权重结构预检（2026-09-24 16:52，零内存：zipfile 直接列键，不 import torch）

把新主干的 state_dict 顶层键前缀与既有 s42 **逐一对齐**，验证 `--arch robust_*` 对 s43/s44 同样成立：

| 组 | 参照（s42） | 新 |
|---|---|---|
| VGG11/Exp2 | `Exp2_Calib+Layerwise_vgg11` — 12 个前缀 | s43 **SAME**（12）、s44 **SAME**（12） |
| ResNet18/Exp3 | `Exp3_FullRobust_resnet18` — 12 个前缀 | s43 **SAME**（12） |

（12 个前缀含 `features` / `classifier` / `calib1`…`calib5` / `fc` / `layer1`…`layer4` 等。）

🔴 **这份预检只证明"架构兼容"，不证明"跑完了"。**
它比对的是 `checkpoints_cifar100/Exp3_FullRobust_resnet18_s43/best_model.pth`——
**那个文件此刻仍是 16:43:38 的崩溃残骸**，键结构当然一样（同一个脚本写的）。
⇒ **完整性一律由 §2.0 的 `metrics.json` 闸门裁定，与本节无关。**
⇒ run3 真正跑完后**要再看一次**本节结论（届时文件才代表 200-epoch 产物）。

---

## 2. 每次读出跑完必做的闸门（**放行/停**，逐个做，不批量）

### 2.0 🔴 前置闸门（**2026-09-24 事故后新增，先于下面所有步骤**）

**跑读出的前提**：该主干的 `metrics.json` 存在 **且 `total_epochs` 等于规定值**（resnet18=200 / vgg11=120）。
**不满足 ⇒ 不跑，报总管。** 理由见 §0.0：`…resnet18_s43/best_model.pth` 现在是一份 6 分 45 秒的**崩溃残骸**
（mtime 16:43:38），用它算出来的数字会**看起来正常但无意义**。

对每个新 CSV：

```bash
grep '^a_original_fc' <新CSV>     # 取 α=0 那一列（第 4 个字段）
cat <该主干>metrics.json           # 取 best_test_acc
```

**判据**：`|CSV 的 a_original_fc@α=0 − metrics.json 的 best_test_acc| ≤ 0.01`。
**对不上 = 接错主干 ⇒ 立即停，不跑下一个。**

**容差的来历与一处既有的边界情形（必须知道）**：
容差 0.01 = **10000 张测试集里的 1 张**。既有 s42 基线就有这个量级的偏移：
`exp3_resnet18_s42` 的 `metrics.json` 是 **72.70**，同一主干的读出 CSV `a_original_fc@α=0` 是 **72.69**
——差正好 0.01，**卡在容差边界上，判 PASS**。vgg11 s42 侧则是 68.05 vs 68.05 逐位相同。
⇒ 新主干若出现 0.01 的偏移，**属同一已知现象，不报警**；≥0.02 才停。

**四处对照锚点**（新数应落在这附近，用于发现"接错主干"之外的粗错）：

| 主干 | `metrics.json` best_test_acc | 状态 |
|---|---|---|
| vgg11 / exp2 / s42 | 68.05 | 既有（`outputs_cifar100/extension6_deep_robust/vgg11/`） |
| vgg11 / exp2 / **s43** | **67.86** | 新，已核 ✓ |
| vgg11 / exp2 / **s44** | **67.64** | 新，已核 ✓ |
| resnet18 / exp3 / s42 | 72.70 | 既有（`…/resnet18_exp3/`） |
| resnet18 / exp3 / **s43** | 待 run3 | |
| resnet18 / exp3 / **s44** | 待 run4 | |

---

## 3. 统计口径（**跑前写死**）

- **总体标准差 ddof=0**（`statistics.pstdev`），与 §13.10 口径一致
- **三组样本**（读出侧一律固定 seed 42）：

| 臂 | vgg11 / Exp2 | resnet18 / Exp3 |
|---|---|---|
| 主干 s42（既有） | `…_exp2_vgg11_s42_ep24_alpha+0.30_n20000.csv` | `…_exp3_resnet18_s42_ep24_alpha+0.30_n20000.csv` |
| 主干 s43（新） | `…_bs43_ep24_…` | `…_bs43_ep24_…` |
| 主干 s44（新） | `…_bs44_ep24_…` | `…_bs44_ep24_…` |

- **三个量**：`clean`（α=0）、`α=+0.3`、`drop@0.3`（= clean − α=+0.3），各算主干级 std
- **两行都算**：主判据用 `e_readout_NAT`，另报 `a_original_fc`（见 §0 修正 2）
- **读出级对照**（已复算，与 runbook 逐位吻合，可作为统计管线的自检）：

| 量 | VGG-11 读出级 std | ResNet-18 读出级 std |
|---|---|---|
| clean | 0.2650 | **1.8520**（runbook 写 ±1.85 ✓） |
| α=+0.3 | 3.1788（论文印 ±3.18 ✓） | **3.3956**（runbook 写 ±3.40 ✓ / 论文印 ±3.40 ✓） |
| drop@0.3 | 3.0403 | 3.4011 |

---

## 4. 裁决（**预登记，跑前已写死，跑后不改**）

| 分支 | 条件（**主干级**标准差） | 判决 |
|---|---|---|
| ① | ≤1.5 pp | Central Claim 锁定，可写"跨主干种子稳健" |
| ② | 1.5~3 pp | 合格，论文需加条件（"在已测主干上成立"） |
| ③ | **≥3 pp** | **止损线 1：立即停下，写报告，报 claude-0b 与用户，不自行继续** |

**命中③时要准备的**：主干级 std 实际值、是读出级的几倍、受影响的句子（见 §5）、收缩后的替代表述草案。

---

## 5. 若命中分支③，受影响的句子（**已预先定位，跑前查好**）

| 文件:行 | 原文要点 | 为何受影响 |
|---|---|---|
| `paper/journal/main.tex:455` | `本文读出适配（24ep，3 seed）… \textbf{47.79}$\pm$3.18 … \textbf{26.82}$\pm$3.40` | 这两个 ± 现在被读作**总不确定度**，但只含读出种子方差、不含主干方差 |
| `paper/journal/main.tex:440` | 图注："每个主干为 **3 个种子**的逐点均值" | 同上的语义问题 |
| `paper/journal/main.tex:462` | "读出适配在三个主干上**一致地**超过'全网络重训'这一整类配方" | "一致地"是要收缩的措辞 |
| `paper/journal/main.tex:1173-1177` | 局限 (2)「深层主干是单次训练产物…**是否在主干种子方差内无法裁决**。**下一步**：每个深层主干重训 2 个种子（约 16 小时）」 | **M4 就是这条"下一步"**；分支① 可弱化本条，分支③ 则本条需强化 |
| `paper/technical-report/main.tex:427` | 同 tab:nat 的 ±值 | 与 journal 同步 |
| `paper/technical-report/main.tex:1009` + `paper/journal/main.tex:1176` | "涉及深层的所有差值——包括门控红利 −1.14——**是否在主干种子方差内无法裁决**" | 同一风险面 |

**数值裕度（供判断收缩力度，非判决依据）**：tab:nat 的对照是
SimpleCNN 43.31 vs 26.57、VGG-11 47.79 vs 24.69、ResNet-18 26.82 vs 6.22。
⇒ 即使主干级 std 真的到 3 pp，**"超过"这个方向仍有 5~7 倍 std 的裕度**。
所以分支③触发的是**"± 的标注不诚实/措辞过强"**，**不等于"结论反了"**。
这一点在写收缩草案时必须说清，否则会把"需要加条件"误写成"结论被推翻"。

---

## 6. 写权冲突（**已裁决，2026-09-24 16:30**）

原先存在冲突：派我活的 `claude-e9` 把 `docs/V7_M4_TAIL.md` 划给我，
新总管 `claude-0b` 说 `docs/**` 归它。**我当时的处置：不自行选，落自己领地、把冲突摆出来问。**
（`claude-0b` 事后评价这个处置"完全正确"。）

**裁决结果**：

| 谁 | 写哪 |
|---|---|
| **我（M4 工人）** | `outputs_cifar100/v7_M4_audit/` —— **报告与 §16.1 草稿都落这里** |
| `claude-65` | `experiments_ledger.md`（持锁者）—— 由它把草稿并入 §16.1 |
| 总管 `claude-0b` | `docs/**` |

⇒ **`docs/V7_M4_TAIL.md` 这个文件名作废**，不建。§16.1 全文落
`outputs_cifar100/v7_M4_audit/M4_LEDGER_16_1_DRAFT.md`，由 claude-65 并入。

## 6.5 训练链的看门狗（**外部保护，不是我的职责**）

总管维护 `tools/v7_m4_watchdog.ps1`，日志 `logs_v7_m4_watchdog.log`。
它**不杀正在跑的训练**，只在"训练进程真的消失**且还有 run 没跑完**"时接手补齐。

### v1 → v2（2026-09-24 16:47，因我上报的缺陷而改）

**v1 的致命缺陷**：完成判据是"**训练进程退出了**"。⇒ 崩溃也是退出 ⇒ 把崩溃判成完成。
实测在 v1 上发生：`seed 43 finished`（只跑了 6 分 45 秒）⇒ 直接跳到 seed 44。

**v2 的四条修正**（总管已实施）：
1. **完成判据只认产物**：`metrics.json` 存在且 `total_epochs` 等于预期（120 / 200）
2. **launch 返回后重查产物**，缺 `metrics.json` ⇒ 判崩溃、`break` 本轮链
3. 连续崩溃**指数退避**
4. **内存闸门**：free physical < 1500 MB 就不拉起
5. 并**补上了输出重定向** ⇒ `logs_v7_m4_restart_s<seed>.log`（v1 漏了这个，导致 s/epoch 一度量不出来）

### ⚠ v2 的日志词表变了（**我的监控已按此改**）

| 串 | 含义 |
|---|---|
| **`COMPLETED`** | **真完成** |
| **`did NOT complete` / `CRASH`** | **崩溃**（不再写 `finished`） |

### 一处我自己的判断错误（已由总管纠正，记录以免重犯）

我原写"守护进程跑完 run4 后 `run(s) left` 归零 ⇒ **正常退出、不回头**"——
**机制说反了**。旧版每轮循环开头都重算 `$missing = runs 里 metrics.json 不存在的那些`，
s43 的 `metrics.json` 一直不存在 ⇒ `$missing` 恒为 2 ⇒ **它不会退出，而是反复空转并反复误判**。
**结论方向对（会乱），机制错（不是"退出"，是"空转+误判"）。**
⇒ 教训：**报"会怎样"时，机制的每一环都要有代码/日志支撑，不能靠推理补。**

**⇒ 我该做的**：看到 `logs_v7_m4_watchdog.log` 出现 `taking over` / `launching`：
说明训练链断过。**那时必须重核产物完整性**（`best_model.pth` 与 `metrics.json` 的时戳/数值，
以及 `total_epochs` 是否达标）。

---

## 7. 当前状态与时间线（**实测，不按冒烟外推**）

| 项 | 值 | 出处 |
|---|---|---|
| run1 vgg11 s43 | ✅ `best_test_acc=67.86`、`total_epochs=120`、`best_epoch=117` | `outputs_cifar100/extension6_deep_robust/vgg11_s43/metrics.json` |
| run2 vgg11 s44 | ✅ `best_test_acc=67.64`、`total_epochs=120`、`best_epoch=101` | `…/vgg11_s44/metrics.json` |
| run3 resnet18 s43 | 🔴 **崩溃 ×2，未完成**（`metrics.json` 不存在） | `logs_v7_m4_watchdog.log` |
| run4 resnet18 s44 | ⏳ 16:44:11 被守护进程提前启动 | 进程 PID 36656 命令行 `--seed 44` |

### 7.1 🔴 时间线已失效（事故后重估）

**上表原写的"run3 在跑、run4 排队"已作废**：run3 于 16:32 OOM 崩溃，16:37:26 重启后又崩，
16:44:11 守护进程**误判完成并跳到 run4**。**两次重启都从零开始（脚本无断点续训）⇒ 已白跑约 1.6 小时。**

**现在的关键不确定性**：**OOM 会不会反复发生。** 若反复，run3/run4 可能永远跑不满 200 epoch。
⇒ **时间线暂时不给估计**（给了也是假的）。**新的判断依据是 `metrics.json` 是否出现，不是时钟。**

**若总管修好守护进程并让内存缓解**，则单次 200ep 仍按 **5.8~7.8 小时**估（下面那组实测数字仍有效）。

**耗时（三次估计全都偏乐观，故以实测为准）**：

**耗时（三次估计全都偏乐观，故以实测为准）**：

| | |
|---|---|
| VGG-11 / 120ep | **110 分钟/次**（实测） |
| ResNet-18 / 200ep | 纯训练吞吐 82 s/epoch（tqdm 100% 行实测），**但还要加每 epoch 验证开销** |
| 我 16:11:39→16:18:41 的粗测 | 422 秒 / 3 个 epoch = **140.7 s/epoch**（该测法把"整 epoch 数"当 3，故是**上界**） |
| 总管 16:06:58 的实测 | **105 s/epoch**（419 秒 / 3.98 epoch）⇒ 区间**下界** |
| 我 16:24:53 的核 | `Train 13/200`、启动 15:59:59 ⇒ 1494 秒 / 约 12.3 epoch = **≈121 s/epoch** |

⇒ **收敛区间：约 105~140 s/epoch，中心约 120** ⇒ 单次 200ep = **5.8~7.8 小时，中心约 6.7 小时**。
⇒ **run3 约 22:00~23:40 完（中心 ≈22:40），run4 约次日 04:00~07:20 完（中心 ≈05:20）。**
⇒ **4 次读出适配各约 25 分钟、串行 ⇒ 全部产出约在次日 05:20~09:00。**

> ⚠ **这是本项目第三次低估 M4 耗时**（冒烟外推 60 min → 吞吐外推 2.4 h → 现按区间报）。
> 三次都是同一个错：**拿"训练吞吐 it/s"当"epoch 时间"，漏掉每 epoch 的验证开销。**
> ⇒ **一律按区间报，不给点估计。**

**接住方式（若本会话死掉）**：这份文件 + `docs/V7_M4_RUNBOOK.md` 足够让新会话接手——
执行命令、参数依据、闸门、统计口径、判据、受影响句子**都已在文档里**，不依赖任何会话的记忆。

---

## 8. 红线自查（每条都对应本项目真实事故）

| # | 红线 | 我的执行 |
|---|---|---|
| 1 | **GPU 串行** | 训练链跑完前**不开任何 GPU 任务**；已做的只有只读 grep / 读文件 / zipfile 列键名 / 纯 python 统计。dry check 用 `CUDA_VISIBLE_DEVICES=""` **物理屏蔽** GPU |
| 2 | **产物零覆盖** | 4 个目标 CSV 名已核**零撞车**；不写任何既有文件 |
| 3 | **判据跑前写死、跑后不改** | 判据见 §4，与 runbook 逐字一致；§0 修正 2 的取法**已显式标注为裁决时判断** |
| 4 | **`n_jobs=None`** | `v3_readout_repair.py` 不含 sklearn/joblib（已读源码），无多进程默认值 |
| 5 | **分级词汇不升格** | 不用"证明/确立"，只用"初步/提示性"等既有分级 |
| 6 | **内存** | 物理可用实测低至 **0.74 GB** ⇒ 全程用零 torch 方式作业，不并起任务 |
| 7 | **设备口径** | 读出一律 CUDA（`v3_readout_repair.py:232` 默认），与既有 s42 读数同设备 |

**一处待查的小口径问题（不影响判据，登记备查）**：
`exp3_resnet18_s42` 的 `a_original_fc@α=0` = 72.69 vs `metrics.json` = 72.70，
差 1 张样本。**这正是红线 §3.4 记录的 CPU/CUDA 跨设备差量级**（那里是 1/8000=0.0125）。
⇒ 既有 s42 读出跑**可能**是 CPU 产物。**但它至多影响 0.01 pp，是判据阈值（≥3 pp）的 1/300，
不可能翻转分支**。**处置：不为此改判，只在报告中登记**；若总管认为需要，再单独查。
