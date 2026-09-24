# 交接文档（快照：2026-09-24 17:00）

> **本文件是某一时刻的快照，不是长期规则。**
> 长期规则看 `docs/V7_COORDINATION.md`；M4 收尾流程看 `docs/V7_M4_RUNBOOK.md`。
> **接手者先读这三份，再动手。**
> 上一版快照是 16:00，由**已终止的 claude-e9** 写的；本版由 **claude-0b** 重写。

---

## 0. 三十秒状态

| | |
|---|---|
| 仓库 | `d:\deeplearning_practice\存算一体` |
| 阶段 | **第七阶段**（台账 §16），进行中 |
| **在跑的长任务** | **M4 深层主干方差**，日志 `logs_v7_m4_train_part2.log` |
| **谁在保它** | **`tools/v7_m4_watchdog.ps1`**（**已实测脱离会话**）——见 §2 |
| 论文 | 三份，`tools/check_papers.py` 退出码 0（journal 25 页） |
| 台账 | 已写到 **§16.8**；**写锁在 claude-65** |
| 🔴 **待用户裁决** | **`nat` 参考文献**（四份文档里的一条结构性不存在的引用）——见 §5 |

---

## 1. 你的第一个动作（5 分钟）

```bash
cd "d:/deeplearning_practice/存算一体"
PY=/d/anaconda3/envs/pytorch_env/python

# ① 确认 M4 在跑 + 守护进程活着
tail -c 300 logs_v7_m4_train_part2.log
tail -c 200 logs_v7_m4_watchdog.log          # 有 "taking over" = 断过并已续上，要重核产物
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='powershell.exe'\" | Where-Object { \$_.CommandLine -like '*-File*v7_m4_watch'+'.ps1*' } | Select ProcessId,ParentProcessId"

# ② 论文干净
PYTHONIOENCODING=utf-8 $PY tools/check_papers.py     # 退出码 0

# ③ 台账自洽
PYTHONIOENCODING=utf-8 $PY build_ledger.py           # 152 rows / pairing FAIL = 0

# ④ ⚠ 开任何吃内存的活之前先量提交余量（见 §4）
powershell -NoProfile -Command "Get-CimInstance Win32_OperatingSystem | Select TotalVirtualMemorySize,FreeVirtualMemory"
```

然后读 `docs/V7_COORDINATION.md`——**领地划分、红线、通信协议、坑表都在那里，动手前必读。**

---

## 2. M4 的状态与守护进程

| # | 配置 | 产物目录 | 状态 |
|---|---|---|---|
| 1 | vgg11/exp2/s43/120ep | `outputs_cifar100/extension6_deep_robust/vgg11_s43/` | ✅ best **67.86** |
| 2 | vgg11/exp2/s44/120ep | `…/vgg11_s44/` | ✅ best **67.64** |
| 3 | resnet18/exp3/s43/200ep | `checkpoints_cifar100/Exp3_FullRobust_resnet18_s43/` | ⏳ 在跑 |
| 4 | resnet18/exp3/s44/200ep | `…_resnet18_s44/` | ⏳ 排队 |

**耗时（实测，**被低估过三次**，每次都是同一个错——拿"训练吞吐"当"epoch 时间"）**：

| | |
|---|---|
| VGG-11 / 120ep | **110 分钟/次** |
| ResNet-18 / 200ep | **5.8 ~ 7.8 小时/次**（**区间，不是点估计**） |

⇒ **run3 ≈ 22:20~23:40 完、run4 ≈ 次日 04:40~07:20 完**，4 次读出再 +100 分钟。
**⚠ 跨夜，不要假设白天会完。** 而且速率**在漂**（113 → 135 s/epoch，见 §4）。

### 守护进程（本版最重要的一处改进）

`tools/v7_m4_watchdog.ps1`，用 `Start-Process` 拉起，**已实测不在会话进程树里**。

- **设计成零损失**：不杀正在跑的训练，只在"训练进程真的消失了**且还有没跑完的 run**"时才接手
- **连续缺席 5 次（×60s）才动手**——会话链是 `run3 && run4`，两命令之间有一瞬间没有 python 进程，
  只看 1 次采样会让它趁那一秒插进去、与会话链**同时**启动 run4（= 红线 1）
- **所有路径从 `$PSScriptRoot` 推导，文件刻意不含任何非 ASCII 字符**——
  PowerShell 5.1 读无 BOM 的 UTF-8 `.ps1` 时按 ANSI 解码，中文路径会被读成乱码（实测踩过）
- **启动坑**：① 反引号续行 + LF 行尾 ⇒ `ParserError` 进程秒退（**已改 CRLF、不用续行**）；
  ② **`-ExecutionPolicy Bypass` 被权限系统拦下——拦得对**，本机 CurrentUser = `RemoteSigned`，
  本地脚本本来就不需要那个参数。**不要加它。**

⇒ **总管会话就算被杀，run3/run4 也有人接着。** 这修掉了本文件上一版记的那条"同一个事故还会再犯"。

---

## 3. 收尾（详见 `docs/V7_M4_RUNBOOK.md`，2026-09-24 已四处修订）

1. 核对 4 次产物（`metrics.json` 的 `total_epochs` 与 `best_test_acc`）
2. 对 4 个新主干各跑一次读出适配（`v3_readout_repair.py`）
   - 🔴 **`--arch` 必须是 `robust_vgg11` / `robust_resnet18`**（原 runbook 写的 `vgg11` 是错的）
   - ⚠ **命名用 `bs<seed>`**（如 `exp2_vgg11_bs43_ep24`），**不要用 `s43`**（已被读出种子占用）
   - **配对闸门**：`a_original_fc`@α=0 必须等于该主干 `metrics.json` 的 `best_test_acc`。
     **0.01 偏移是已知现象（记 PASS）；≥0.02 才停**
3. 算三组 3-seed 方差（ddof=0）
   - 🔴 **主判据取 `e_readout_NAT` 行**；另报 `a_original_fc` 行（**它恒为 0.0000，不能拿去比读出级**）
   - 两行不一致时**按保守方向报**（任一 ≥3 pp 即报分支③）
   - ⚠ **这一步是执行者的判断、不是预登记原文，报告里必须显式标注**
4. 台账 §16.1（四目式）——**草稿落 `outputs_cifar100/v7_M4_audit/`**，由持锁者 claude-65 并入
5. 判决：① ≤1.5 pp 锁定 / ② 1.5~3 pp 加条件 / ③ **≥3 pp 触发止损线——停下报用户**

---

## 4. 两条新红线（本版新增，都实测/实量过）

**(a) 🔴 内存余量已跌到历史事故水位以下。**
16:20 实测：提交限 57.6 GB / 已提交 56.14 GB ⇒ **可用仅 1.46 GB**；
而坑表记的**事故当时是 0.9 / 4.9 GB** ⇒ **比事故水位还低 3.4 GB**。

> ⚠ **赌注不对称**：Windows 提交内存是**全局**的。你的进程吃掉的余量会让**训练**的下一次分配失败
> ——**`WinError 1455` 打死的是训练，不是你的脚本。**
> **规则："零 GPU 准备"≠"零执行"。只要会 `import torch`，就受这条管。**

**(b) ⚠ 并行做论文维修会拖慢 GPU 训练（假说，待证实）。**
M4 的 epoch 时间在漂：15:59→16:11 = **113 s/epoch**、16:11→16:25 = **135 s/epoch（+19%）**。
同期开了 6 个工人会话做纯 CPU 的活（python 复算、`xelatex` 编译、matplotlib 渲染）。
**待证实**：claude-cf 在精测窗口同时记录 `python.exe` 数量，看两者是否同向。
**若成立 ⇒ 排期要把"同时开了几个会话"算进去。**

---

## 5. 🔴 待用户裁决：`nat` 参考文献

**四份文档**（`paper/journal`、`paper/technical-report`、`paper/competition`、`ucas-cod-lab-report`）
里的 `\bibitem{nat}`（Zhu M 等, *Nonlinearity-aware training…*, IEEE TCAD 2022, **41(11): 3961–3973**）：

**结构性证据**：把 Crossref 的 **TCAD 2022 全年 496 条**拉全后，该期
**3957–3968 / 3969–3980 / 3981–3992 三篇连续铺满、中间无空隙**
⇒ 所引的 3961–3973 **会跨过两篇无关论文、结构上不可能存在**。
按标题反复命中的是**另一篇真实论文 NEAT**（Bhattacharjee 等, TCAD **2022, 41(8): 2625–2637**,
DOI 10.1109/TCAD.2021.3109857）——**主题/年限/期刊全同，作者/卷/期/页码全不同**。

**项目自己的规矩**（`docs/LITERATURE_NOTES.md` 开头）：「**未核实的不许写进论文**」，
而 `nat` **从未走过该流程**。

**⇒ 已上报用户，等裁决。用户发话前，四份文档一个都不改。**
**同类第二例**：`LITERATURE_NOTES` §二 的 `arXiv 2310.03843`——**编号对、摘要对、标题对不上**。

---

## 6. 本轮（16:20–17:00）已完成

| 项 | 结果 |
|---|---|
| A1/A2 | ✅ 已修；A2 的**第三处漏网**已补 |
| A3–A14 | ✅ **12/12**（写者还否证了复核者两条建议值） |
| A3–A10 独立重算 | ✅ **8/8**，**且推翻了复核者的 A4 论证** |
| A9 计数错 | ✅ 四处全改（台账 + 论文 + 两份报告） |
| 8 张图 | ✅ 已修、逐张复看、已提交（`f4e282d`） |
| 台账扫查 | ✅ 完成 |
| TTA 文献检索 | ✅ **8 条全部逐字核** |
| M4 守护进程 | ✅ 已建、已实测脱离会话 |

**在跑**：claude-4c（journal 的 B+E 组）、claude-89（technical-report + competition）、
claude-6f（其余三份的文献核查）、claude-65（C 组局限起草）。
**在等**：claude-38（C5 第三臂，等窗口）、claude-cf（M4 收尾，等 GPU）。

---

## 7. 七条最容易踩的（都实测踩过）

1. **markdown 写进 LaTeX**（累计 **11 次**）——`**粗体**` / 反引号 / 行首 `>` 都是静默失效。
   ⇒ **改完论文、提交前必跑 `tools/check_papers.py`**，别靠记性。
2. **台账只能一个写者**——**"写入"包括改头部注记**。单一写者的对象是**文件**，不是文件里某一段。
3. **设备口径也是口径**——CPU 与 CUDA 的读数不可混。它已经**翻转过一个写在论文里的计数**。
4. **🔴 不要用 `-ExecutionPolicy Bypass`**（本机是 RemoteSigned，根本不需要；且会无故削弱一道安全控制）。
5. **🔴 本地脚本别写中文**（PowerShell 5.1 读无 BOM 的 UTF-8 会按 ANSI 解码）+ **别用反引号续行**（配 LF 行尾会 ParserError）。
6. **行号在本仓库极不可靠**——`journal/main.tex` 一天内从 1258 长到 1442 行还在长。
   ⇒ **引用位置一律附原文字符串**，行号只作辅助。
7. **"it/s" 不是 "epoch 时间"**——M4 的耗时被这个错低估了**三次**。

---

## 8. 当前没有其它需要用户裁决的阻塞项

唯一两件：**(a) 上面的 `nat` 裁决**；**(b) M4 若在分支③命中，必须停下报告用户，不自行继续。**
