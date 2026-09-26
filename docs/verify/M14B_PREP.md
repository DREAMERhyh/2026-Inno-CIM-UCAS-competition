# M14b 准备文档：ReLU → GELU/LeakyReLU 架构变体与训练方案

> **状态：准备完成，未跑任何训练（零 GPU 秒）。**
> 写者：承接 M14b 准备的工人会话（总管 `claude-0c` 派出）。
> 落盘时刻：**2026-09-26**。
> 领地声明：本文件 + `models/simple_cnn_act.py`（新增）是本会话的**全部**写入；
> `git status` 确认既有文件零改动。
> 红线遵守：无 GPU 训练、无既有文件修改、单进程 CPU（batch=2）、全程用
> `/d/anaconda3/envs/pytorch_env/python`。

---

## 0. 一页速览（给总管）

| 项 | 结果 |
|---|---|
| 交付物 ① | `models/simple_cnn_act.py`（新）—— 2 档池化 × 3 激活 = **6 个组合**，单文件参数化 |
| 交付物 ② | 本文件 |
| **参数量自检** | 6 个组合**全部 = 254 084**（nc=100），与对应 ReLU 版**逐位相同** ✅ |
| **镜像自检** | `act="relu"` 时与既有实现 `torch.equal = True`、`max|diff| = 0` ✅ —— 证明"换激活"是唯一自变量 |
| CPU 前向 | batch=2 全通过（输出 `(2, 100)`、`isfinite` 全真）✅ |
| 激活位置清点 | `SimpleCNNMaxPool` **4 处**、`SimpleCNN` **4 处**，见 §3 |
| **判决逻辑自测** | 五个用例（三分支 + 两侧近边界）**全部 OK**，并抓出浮点边界（§6.5） |
| **✅ 裁决 1（已下）** | 采纳 A —— 补跑同协议 ReLU-2pool，判据改用**两臂同协议**的 P 值 |
| **✅ 裁决 2（已下）** | 采纳 A —— 判据加**第三分支"不可分辨"**，阈值 **4 pp（≈2σ）** |
| **✅ 裁决 3（已下）** | **主组 = GELU**；LeakyReLU 为补充档，**其无效应不得否证假说** |
| 训练命令 | **5 条已备好**（§5.5），产物名自带 `_m14b`，10 个目标路径**零覆盖核验通过** ✅ |
| 单次耗时 | 实测 **30~57 min**（取自既有 `metrics.json`，非外推） |
| **✅ 接线已应用** | 位置定在 **`v3_methods_simplecnn.py`**（不是 extension6，理由见 §5.6）：4 处改动、16 insertions / 2 deletions，**`act="relu"` 数值逐位不变**（3 条证据 §5.8） |
| **🔴 未做的一件事** | `task_extension6_deep_robust_train.py` 的 `get_model` —— 判定它跑不出 M14b 的量（§5.6），**一个字节都没动** |

**两处请总管确认**（§7.4）：裁决书里**两个阈值不相容**（1 pp vs 4 pp，我按 4 pp 实现）；
以及我把 Δ 换成了等价的 `P = −Δ` 以避开负数语义。

**一句话**：代码、命令、判决逻辑自测**全部就绪**；接线在 `v3_methods_simplecnn.py`（**不是** extension6，理由见 §5.6）。

---

## 1. 交付物 ①：`models/simple_cnn_act.py`

### 1.1 为什么用单文件参数化（而不是 4 个文件）

| 方案 | 评价 |
|---|---|
| 4 个独立文件（`simple_cnn_mp_gelu.py` 等） | ❌ 4 份近乎相同的代码；改一处要改四处；且**无法做"relu 档 = 既有实现"的逐位自检**（没有可对照的镜像） |
| **单文件 + `act` 参数**（已采用） | ✅ 一份代码；`act="relu"` 分支**逐字复用既有写法**，从而可做镜像自检 |

**关键设计**：`act="relu"` 走 `nn.ReLU(inplace=True)`，与既有实现**逐字一致**。
这不是装饰——它让"relu 档"成为回归基准：如果镜像写歪了，参数量或数值任一自检会立刻报警。

### 1.2 结构（严格镜像，非"重新设计"）

| 新类 | 镜像的既有实现 | MaxPool | 空间尺寸 |
|---|---|---|---|
| `SimpleCNNAct` | `models/simple_cnn.py:15` `SimpleCNN` | 2 | 32→16→8 |
| `SimpleCNNMaxPoolAct` | `v3_methods_simplecnn.py:86` `SimpleCNNMaxPool` | 5 | 32→16→8→4→2→1 |

**⚠ 必须知道的一条既有事实**：两者的池化**位置不同**，不是同一个网络的两种密度——
`SimpleCNN` 的 `pool1` 在 `relu2` 之后；`SimpleCNNMaxPool` 的 `pool1` 在 `relu1` 之后。
本次**照抄既有实现**（M14b 的对照量是"池化密度"，把池化位置也一起改了会污染对照）。
这不是我引入的差异，是 `P1-5` 当年引入 `simple_cnn_mp` 时就有的。

### 1.3 激活的三种取值（实验设计变量，需在预登记里写死）

| `act` | 实现 | 负斜率/形状 |
|---|---|---|
| `relu` | `nn.ReLU(inplace=True)` | 非负（既有行为） |
| `gelu` | `nn.GELU(approximate="none")` | 精确 erf 版；**不是** tanh 近似（另一族形状） |
| `leaky` | `nn.LeakyReLU(negative_slope=0.01, inplace=True)` | 用 **PyTorch 默认负斜率 0.01** |

`LEAKY_NEGATIVE_SLOPE` 被提成模块级常量：**要改只改这一处**，但改之前必须知道 §10 的操纵强度实测。

---

## 2. 判据自检：参数量与逐位一致（全部实测）

**命令（可复现）**：见 §4 的粘贴块。**实测输出如下（2026-09-26 本会话）**：

### 2.1 参数量（自检点 ①：换激活不该改参数量）

```
=== [1] param counts (nc=100) ===
  ref SimpleCNN        (2pool/relu) = 254084
  ref SimpleCNNMaxPool (5pool/relu) = 254084
  pool=2 act=relu  params=254084   vs relu-ref 254084  [OK]
  pool=5 act=relu  params=254084   vs relu-ref 254084  [OK]
  pool=2 act=gelu  params=254084   vs relu-ref 254084  [OK]
  pool=5 act=gelu  params=254084   vs relu-ref 254084  [OK]
  pool=2 act=leaky params=254084   vs relu-ref 254084  [OK]
  pool=5 act=leaky params=254084   vs relu-ref 254084  [OK]
```

**⚠ 顺带记录**：2×max 与 5×max 的 ReLU 版本**参数量本来就相同**（254 084）——
多出来的 3 个 `MaxPool2d` 无参数。这正是 M14b 与 M14 被称为"零成本架构杠杆"的原因，
也意味着**参数量不能用来区分这两个架构**（区分只能靠 `layer_names` 之后的形状或产物名）。

### 2.2 镜像自检（自检点 ②：数值逐位一致）

```
=== [2] bitwise identity vs legacy impls (act=relu) ===
  pool=2: torch.equal=True  max|diff|=0.000e+00
  pool=5: torch.equal=True  max|diff|=0.000e+00
```

做法：`new.load_state_dict(ref.state_dict())`（`strict=True`，**层名不一致会直接 raise**）+ 同一输入前向。
⇒ 参数命名与 forward 顺序与既有实现逐字一致 ✅

### 2.3 注入目标不变（α 注入是按层名找模块的）

```
=== [5] layer_names invariance ===
  pool=2/5 × act=relu/gelu/leaky → ['conv1', 'conv2', 'conv3', 'conv4', 'fc']
```

6 个组合**全部相同** ⇒ `layer_names()`、`Injector`、`register_nonlinearity_hooks` 无需任何改动。

---

## 3. 激活函数在既有实现里的出现位置（逐处清点，判据要求）

### 3.1 `v3_methods_simplecnn.py` 的 `SimpleCNNMaxPool`（**5×MaxPool，M14b 的 mp 档**）

| # | 位置 | 代码 |
|---|---|---|
| 1 | [v3_methods_simplecnn.py:98](../../v3_methods_simplecnn.py#L98) | `self.relu1 = nn.ReLU(inplace=True)` |
| 2 | [v3_methods_simplecnn.py:102](../../v3_methods_simplecnn.py#L102) | `self.relu2 = nn.ReLU(inplace=True)` |
| 3 | [v3_methods_simplecnn.py:106](../../v3_methods_simplecnn.py#L106) | `self.relu3 = nn.ReLU(inplace=True)` |
| 4 | [v3_methods_simplecnn.py:110](../../v3_methods_simplecnn.py#L110) | `self.relu4 = nn.ReLU(inplace=True)` |

forward（[:116-123](../../v3_methods_simplecnn.py#L116-L123)）里 `relu1/2/3/4` **各被调用一次**，无遗漏。
**该文件里没有第 5 处激活**：`fc` 直接输出 logits；`Top2MeanPool2d` 是池化不是激活。

### 3.2 `models/simple_cnn.py` 的 `SimpleCNN`（**2×MaxPool，M14b 的 base 档**）

| # | 位置 | 代码 |
|---|---|---|
| 1 | [models/simple_cnn.py:31](models/simple_cnn.py#L31) | `self.relu1 = nn.ReLU(inplace=True)` |
| 2 | [models/simple_cnn.py:38](models/simple_cnn.py#L38) | `self.relu2 = nn.ReLU(inplace=True)` |
| 3 | [models/simple_cnn.py:46](models/simple_cnn.py#L46) | `self.relu3 = nn.ReLU(inplace=True)` |
| 4 | [models/simple_cnn.py:53](models/simple_cnn.py#L53) | `self.relu4 = nn.ReLU(inplace=True)` |

forward（[:62-101](models/simple_cnn.py#L62-L101)）里 4 处**各被调用一次**，无遗漏。

### 3.3 容易误判的两处（**不是**激活，不要一起换）

- `utils/nonlinearity.py` 的 `nonlinearity(x, alpha)`：三次多项式**失真注入**，走 forward-**pre**-hook。
  它与激活是**串联**关系（注入在 conv 输入上，激活在 BN 之后），换激活不影响它。
- `nn.BatchNorm2d`：不是激活。但**注意**：BN 会把激活前的分布重新标准化——
  这是"换激活"效应可能被削弱的一个真实渠道（见 §10 未验证项 ③）。

### 3.4 其它引用 `simple_cnn_mp` 的文件（本次**不涉及**，列出防漏）

`v3_readout_repair.py:67`（读出适配）、`v5_full_recipe.py`、`v5_probe_maxpool_mechanism.py:47`、
`v7_gate_mechanism.py:36`。M14b 的判据只用到**主干自身的 α 扫描**（`drop@+0.3`），
**不需要读出适配** ⇒ 这些文件本次一个都不用改。

---

## 4. CPU 小前向验证方法（可粘贴复现）

在仓库根目录执行（**约 15 秒、单进程、峰值内存极小**）：

```bash
PYTHONIOENCODING=utf-8 /d/anaconda3/envs/pytorch_env/python - << 'EOF'
import sys, torch
sys.path.insert(0, '.')
from models.simple_cnn import SimpleCNN
from models.simple_cnn_act import SimpleCNNAct, SimpleCNNMaxPoolAct, make_act
from v3_methods_simplecnn import SimpleCNNMaxPool, layer_names

torch.manual_seed(0)
NC = 100
npar = lambda m: sum(p.numel() for p in m.parameters())

ref2, ref5 = SimpleCNN(num_classes=NC), SimpleCNNMaxPool(num_classes=NC)
print("=== [1] param counts (nc=%d) ===" % NC)
print("  ref SimpleCNN        (2pool/relu) = %d" % npar(ref2))
print("  ref SimpleCNNMaxPool (5pool/relu) = %d" % npar(ref5))
configs = []
for act in ("relu", "gelu", "leaky"):
    for pool in (2, 5):
        m = (SimpleCNNAct if pool == 2 else SimpleCNNMaxPoolAct)(num_classes=NC, act=act)
        configs.append((pool, act, m))
        ref = ref2 if pool == 2 else ref5
        tag = "OK" if npar(m) == npar(ref) else "**MISMATCH**"
        print("  pool=%d act=%-5s params=%-7d  vs relu-ref %-7d [%s]" % (pool, act, npar(m), npar(ref), tag))

print("\n=== [2] bitwise identity vs legacy impls (act=relu) ===")
x = torch.randn(2, 3, 32, 32)
for pool, ref in ((2, ref2), (5, ref5)):
    new = (SimpleCNNAct if pool == 2 else SimpleCNNMaxPoolAct)(num_classes=NC, act="relu")
    new.load_state_dict(ref.state_dict())   # strict=True: 层名不一致会 raise
    new.eval(); ref.eval()
    with torch.no_grad():
        y_new, y_ref = new(x), ref(x)
    print("  pool=%d: torch.equal=%s  max|diff|=%.3e" %
          (pool, torch.equal(y_new, y_ref), (y_new - y_ref).abs().max().item()))

print("\n=== [3] manipulation check: activation output negativity ===")
pre = torch.randn(200000) * 1.5
print("  pre-activation neg frac = %.4f" % (pre < 0).float().mean())
for act in ("relu", "gelu", "leaky"):
    with torch.no_grad():
        out = make_act(act)(pre.clone())
    nf = (out < 0).float().mean().item()
    ne = (out.clamp(max=0) ** 2).sum().item() / (out ** 2).sum().item()
    print("  act=%-5s out<0 frac=%.4f  neg-energy share=%.3e  min=%.4f" % (act, nf, ne, out.min().item()))

print("\n=== [4] in-model activation negativity (random init, eval, 5pool) ===")
for act in ("relu", "gelu", "leaky"):
    m = SimpleCNNMaxPoolAct(num_classes=NC, act=act).eval()
    stats = {}
    def mk(nm):
        def hook(mod, inp, out):
            stats[nm] = ((out < 0).float().mean().item(),
                         (out.clamp(max=0) ** 2).sum().item() / (out ** 2).sum().item())
        return hook
    hs = [getattr(m, "relu%d" % i).register_forward_hook(mk("relu%d" % i)) for i in range(1, 5)]
    with torch.no_grad():
        m(x)
    for h in hs: h.remove()
    print("  act=%-5s " % act + "  ".join("r%d:neg%%=%.3f" % (i, stats["relu%d" % i][0]) for i in range(1, 5)))

print("\n=== [5] layer_names invariance ===")
for pool, act, m in configs:
    print("  pool=%d act=%-5s -> %s" % (pool, act, layer_names(m)))

print("\n=== [6] CPU forward batch=2 ===")
for pool, act, m in configs:
    m.eval()
    with torch.no_grad():
        y = m(x)
    print("  pool=%d act=%-5s out.shape=%s finite=%s" % (pool, act, tuple(y.shape), bool(torch.isfinite(y).all())))
EOF
```

**判据**：`[1]` 六行全 `OK`、`[2]` 两行 `torch.equal=True`、
`[6]` 六行 `finite=True`。任一不满足 ⇒ **不要开训**。

---

## 5. 接进训练的改法（**已于 2026-09-26 应用**，总管授权 (a)）

`v3_methods_simplecnn.py` 是**唯一**改动的文件，4 处，`git diff` = **16 insertions / 2 deletions**
（2 处 deletion 是 `build_arch` 签名与调用处的原地修改，**没有顺手改任何别的行**）。
**核心约束已用证据核验**：`act="relu"`（或不传 `--act`）时**数值逐位不变** ⇒ 既有产物仍可复现。
**证据见 §5.8**（三条，均可复跑）。

> 下面 §5.1–5.4 的 diff 块**保留原文**（它是应用前的审查记录，与 `git diff` 应逐字相符）。

### 5.1 修改点 ①：`build_arch` 加 `act` 形参（第 166 行）

```diff
-def build_arch(name, nc):
+def build_arch(name, nc, act="relu"):
     if name == "vgg11":                       # 第六阶段 P4c：跨架构粒度对照用
         from models.vgg11 import VGG11
         return VGG11(num_classes=nc)
+    if act != "relu":
+        # M14b：激活参数化变体（models/simple_cnn_act.py，镜像实现）。
+        # 参数命名与参数量与 ReLU 版逐字一致；act="relu" 不走这条路径 ⇒ 既有行为逐字不变。
+        from models.simple_cnn_act import SimpleCNNAct, SimpleCNNMaxPoolAct
+        if name == "simple_cnn_mp":
+            return SimpleCNNMaxPoolAct(num_classes=nc, act=act)
+        if name == "simple_cnn":
+            return SimpleCNNAct(num_classes=nc, act=act)
+        raise ValueError(f"--act={act} 只支持 simple_cnn / simple_cnn_mp，不支持 {name}")
     if name == "simple_cnn_mp":
         return SimpleCNNMaxPool(num_classes=nc)
```

**⚠ 那个 `raise` 不是多余的**：若不加，`--arch vgg11 --act gelu` 会**静默返回 ReLU 版 VGG-11**，
跑完得到一张"以为是 GELU"的表。本项目对"静默失效"已有多次前科（缺字形、`\%` 误判），
**宁可炸掉**。

### 5.2 修改点 ②：argparse 新增 `--act`（第 444 行 `--tag` 之前）

```diff
+    ap.add_argument("--act", default="relu", choices=["relu", "gelu", "leaky"],
+                    help="M14b：激活函数。默认 relu = 既有行为逐字不变；"
+                         "gelu/leaky 走 models/simple_cnn_act.py 的镜像实现")
     ap.add_argument("--tag", default="")
```

`--arch` 的 `choices` **不用改**（复用既有的 `simple_cnn` / `simple_cnn_mp`）⇒ 改动面更小。

### 5.3 修改点 ③：调用处（第 453 行）

```diff
-    model = build_arch(args.arch, nc).to(device)
+    model = build_arch(args.arch, nc, args.act).to(device)
```

### 5.4 修改点 ④：产物命名（第 456-458 行）—— **防覆盖，必须做**

```diff
     name = f"{args.mode}_{args.profile}{args.tag}"
     if args.arch != "simple_cnn":
         name = f"{args.arch}_{name}"
+    if args.act != "relu":
+        name = f"{name}_{args.act}"          # ← M14b：防覆盖既有 ReLU 产物
```

**为什么不能只靠 `--tag`**：靠人记得加 tag 才能防覆盖，是把红线（§三.3 产物零覆盖）
押在"操作者不犯困"上。命名逻辑自动加后缀，则忘记加 tag 也不会覆盖。

**已核验（2026-09-26）**：下列目标目录**全部不存在** ⇒ 零覆盖成立。

```
outputs_cifar100/v3_methods/clean_uniform          → 不存在 ✅
checkpoints_cifar100/v3_clean_uniform              → 不存在 ✅
outputs_cifar100/v3_methods/*gelu* / *leaky*       → 不存在 ✅
```

### 5.5 五条训练命令（**可直接粘贴**，按总管裁决的优先序排列）

驱动脚本唯一：**`v3_methods_simplecnn.py`**（理由见 §5.6——**不是** extension6，那一节有硬证据）。
协议与 §13.8 的 s42 行**逐字相同**，只加 `--arch` / `--act` / `--tag _m14b`。

```bash
# ── 第 1 条（必须）：ReLU-2pool 同协议基准 ─────────────────────────────
/d/anaconda3/envs/pytorch_env/python v3_methods_simplecnn.py \
  --arch simple_cnn --act relu --mode clean --profile uniform \
  --dataset cifar100 --epochs 120 --seed 42 --tag _m14b

# ── 第 2 条（主组）：GELU-2pool ────────────────────────────────────────
/d/anaconda3/envs/pytorch_env/python v3_methods_simplecnn.py \
  --arch simple_cnn --act gelu --mode clean --profile uniform \
  --dataset cifar100 --epochs 120 --seed 42 --tag _m14b

# ── 第 3 条（主组）：GELU-5pool ────────────────────────────────────────
/d/anaconda3/envs/pytorch_env/python v3_methods_simplecnn.py \
  --arch simple_cnn_mp --act gelu --mode clean --profile uniform \
  --dataset cifar100 --epochs 120 --seed 42 --tag _m14b

# ── 第 4 条（补充档，时间够才跑）：LeakyReLU-2pool ─────────────────────
/d/anaconda3/envs/pytorch_env/python v3_methods_simplecnn.py \
  --arch simple_cnn --act leaky --mode clean --profile uniform \
  --dataset cifar100 --epochs 120 --seed 42 --tag _m14b

# ── 第 5 条（补充档，时间够才跑）：LeakyReLU-5pool ─────────────────────
/d/anaconda3/envs/pytorch_env/python v3_methods_simplecnn.py \
  --arch simple_cnn_mp --act leaky --mode clean --profile uniform \
  --dataset cifar100 --epochs 120 --seed 42 --tag _m14b
```

**为什么只有 5 条、ReLU-5pool 不在列表里**：`Δ(relu)` 的另一臂
`drop_5max(relu) = 20.60` **已经存在**，且**同脚本、同协议、同 seed 42**
（`v3_methods_simplecnn.py --mode clean`，§13.8 的 s42 行）。**不需要重跑。** 见 §5.7。

**产物落点**（`--tag _m14b` + §5.4 的自动 act 后缀 ⇒ 名字自带 M14b 标记）：

| # | 产物目录（`outputs_cifar100/v3_methods/`） | checkpoint（`checkpoints_cifar100/`） |
|---|---|---|
| 1 | `clean_uniform_m14b` | `v3_clean_uniform_m14b` |
| 2 | `clean_uniform_m14b_gelu` | `v3_clean_uniform_m14b_gelu` |
| 3 | `simple_cnn_mp_clean_uniform_m14b_gelu` | `v3_simple_cnn_mp_clean_uniform_m14b_gelu` |
| 4 | `clean_uniform_m14b_leaky` | `v3_clean_uniform_m14b_leaky` |
| 5 | `simple_cnn_mp_clean_uniform_m14b_leaky` | `v3_simple_cnn_mp_clean_uniform_m14b_leaky` |

**零覆盖核验（2026-09-26 本会话，10 个路径逐个 `-e` 测试）**：**全部不存在 ✅**
（`outputs_cifar100/v3_methods/` 与 `checkpoints_cifar100/` 两侧都查了）

#### 命令冻结的证据 → **见 §5.8**（diff 应用后重跑，三条证据全过）

⇒ **命令已冻结**（2026-09-26 02:1x，**diff 已落地后**复验）。

**串行**：本项目 GPU 红线是"同一时刻只能有一个训练"（§三.1）。
用 `&&` 链式或 `.bat` + `Start-Process`（§五·补一的配方）。

**实测单次耗时**（从既有 `metrics.json` 的 `elapsed_sec` 取，**不是外推**）：

| 产物 | elapsed |
|---|---|
| `simple_cnn_mp_clean_uniform`（s42） | **56.9 min** |
| `simple_cnn_mp_clean_uniform_s43` | 36.2 min |
| `simple_cnn_mp_clean_uniform_s44` | 29.8 min |

⇒ **单次按 30~57 min 估**（波动来自系统负载）。5 次 ≈ **2.5~4.8 h**；3 次（只跑必须+主组）≈ **1.5~2.9 h**。

### 5.6 🔴 为什么**不能**用 `task_extension6_deep_robust_train.py`（三条硬证据）

总管曾授权"接 `get_model` 的线"。我**读了那个文件后没有动手**——因为它**跑不出 M14b 需要的量**。
若照此开跑，3 小时 GPU 会全废。证据逐条：

| # | 事实 | 位置 | 后果 |
|---|---|---|---|
| ① | 它是 **NAT 训练**脚本：`use_calibration=True, layerwise_alpha=True` **硬编码**在 main 里，每个 batch 逐层注入 α | `task_extension6_deep_robust_train.py:626-627`（调用）、`:264-271`（注入） | M14b 判据的基准（`drop 20.60 / 31.90`）是 **clean 训练**（α 零注入）的产物 ⇒ **口径不同，不可相减** |
| ② | **不保存 `alpha_sensitivity.csv`**；`metrics.json` 里只有 `final_alpha03_acc`（**最后一个 epoch** 的 α=+0.3） | `:414-432` | M14b 的 `drop = clean − acc(+0.3)` 需要**同一组 best 权重**下的两个数；这里是 `best_test_acc`（best epoch）与 `final_alpha03_acc`（**最后** epoch）—— **不同 epoch 相减没有意义** |
| ③ | `--model` 是**白名单** `choices=["vgg11","resnet18"]` | `:522-527` | 加新架构**必须改既有行**（choices），与总管"git diff 只应有新增行"的要求**直接冲突** |

外加两条次要事实：`get_depth_group` 对未知 `model_name` 一律返回 `"middle"`（`:126-127`，不崩但 α 分组全落中档）；
`use_calibration=True` 会要求模型带 calibration 模块，而 `SimpleCNN` / `SimpleCNNMaxPoolAct` 没有。

⇒ **`task_extension6_deep_robust_train.py` 未被本会话修改，一个字节都没动。**
⇒ M14b 的接线在 **`v3_methods_simplecnn.py`**（§5.1–§5.4 的 diff），那是与 §13.8 同脚本的唯一选择。

> 若总管仍要给 extension6 加 `SimpleCNN` 家族分支（**为别的实验**，非 M14b），
> 需要同时改 `get_model` + `--model` 的 `choices` + `get_depth_group` 的层名分组，
> **且它测不出 `drop@+0.3`**（证据②）。**那是另一件事，请单独派活。**

### 5.7 `Δ(relu)` 的另一臂：为什么不用重跑 ReLU-5pool

总管裁决 1 定义 `Δ(relu) = drop_5max(relu) − drop_2max(relu)`，**两臂同协议**。

| 臂 | 值 | 来源 | 同协议？ |
|---|---|---|---|
| `drop_5max(relu)` | **20.60**（3-seed 均值）／**18.52**（s42 单点） | `v3_methods_simplecnn.py --mode clean`，§13.8 | ✅ **同脚本、同协议、含 seed 42** |
| `drop_2max(relu)` | 待产 | §5.5 第 1 条（同脚本、同协议、seed 42） | ✅ |

⇒ 两臂**已经在同一协议下**（这正是裁决 1 要的），**ReLU-5pool 不需要重跑**。

**一个口径细节（供裁决，我按"主分析用 3-seed 均值"准备）**：

- `20.60` 是 3-seed 均值，**噪声更小**（标准误 ≈ 1.54/√3 ≈ 0.89），但它是 3 个 seed 的合成；
- `18.52` 是 **seed 42 单点**，与第 1 条命令**同 seed 配对**，但噪声大（σ ≈ 1.54）。

**建议**：**主分析用 20.60**（估计量更优），**同时报告 18.52 的 s42 配对版**作一致性检查；
两者应给出**同向**结论——若不同向，说明结论被 seed 抖动支配，按第三分支记"不可分辨"。

### 5.8 diff 应用后的三条证据（**act="relu" 路径逐位不变**）

三条都在同一台机、同一个 `pytorch_env` 下实跑，**未训练、未建目录、未碰 GPU**（`--device cpu`）。

#### 证据 ①：`build_arch` 的输出对「不传 act」逐位不变

做法：`git show HEAD:v3_methods_simplecnn.py` 取出**改动前**的源码 → 写成临时文件动态加载 →
与改动后的模块在同一 `torch.manual_seed(1234)` 下各构造一次 → 逐个 tensor `torch.equal`。

```
  simple_cnn     nc=100 类名同=True 键同=True **数值逐位同=True** [OK]
  simple_cnn_mp  nc=100 类名同=True 键同=True **数值逐位同=True** [OK]
  vgg11          nc=100 类名同=True 键同=True **数值逐位同=True** [OK]
  simple_cnn     nc=10  类名同=True 键同=True **数值逐位同=True** [OK]
  => 全部逐位不变 ✅
```

> **一处判据修正（我第一版写错了，如实记）**：最初我用 `type(a) is type(b)`，`simple_cnn_mp` 判 FAIL。
> 原因是**旧版的 `SimpleCNNMaxPool` 来自另一个模块对象**，`is` 必然 False —— **是判据错，不是代码错**。
> 改为比 `type(a).__name__` 后通过。（`simple_cnn`/`vgg11` 从 `models.*` import，同一模块对象，故 `is` 也为真。）

#### 证据 ②：真实 `main()` 的 argparse 解析 5 条命令 + 捕获 `build_arch` 实参

做法：import 模块（`__name__ != "__main__"` ⇒ `main()` 不会自动跑）→ 设 `sys.argv` →
调用 `main()`，同时 patch 掉 `build_loaders`（抛 `StopIteration` 截停）与 `os.makedirs`（防副作用），
并给 `build_arch` 装间谍记录实参。

```
  cmd1  --arch simple_cnn --act relu      -> build_arch(('simple_cnn', 100, 'relu'))      argparse 通过 ✅
  cmd2  --arch simple_cnn --act gelu      -> build_arch(('simple_cnn', 100, 'gelu'))      argparse 通过 ✅
  cmd3  --arch simple_cnn_mp --act gelu   -> build_arch(('simple_cnn_mp', 100, 'gelu'))   argparse 通过 ✅
  cmd4  --arch simple_cnn --act leaky     -> build_arch(('simple_cnn', 100, 'leaky'))     argparse 通过 ✅
  cmd5  --arch simple_cnn_mp --act leaky  -> build_arch(('simple_cnn_mp', 100, 'leaky'))  argparse 通过 ✅

  [2b] 不传 --act ⇒ 传给 build_arch 的 act = 'relu'  [OK]
  [2c] --help 含：[--act {relu,gelu,leaky}] [--tag TAG]
```

#### 证据 ③：真实 `main()` 里 `os.makedirs` 的实参 = 产物目录，与 §5.5 表格**逐字一致**

**这条比"复述命名逻辑"强**：它不重算名字，而是**从真实执行中截获**要建的目录。

```
  cmd1  ckpt=v3_clean_uniform_m14b                     out=clean_uniform_m14b                    [一致] 覆盖:不存在 ✅
  cmd2  ckpt=v3_clean_uniform_m14b_gelu                out=clean_uniform_m14b_gelu               [一致] 覆盖:不存在 ✅
  cmd3  ckpt=v3_simple_cnn_mp_clean_uniform_m14b_gelu  out=simple_cnn_mp_clean_uniform_m14b_gelu [一致] 覆盖:不存在 ✅
  cmd4  ckpt=v3_clean_uniform_m14b_leaky               out=clean_uniform_m14b_leaky              [一致] 覆盖:不存在 ✅
  cmd5  ckpt=v3_simple_cnn_mp_clean_uniform_m14b_leaky out=simple_cnn_mp_clean_uniform_m14b_leaky[一致] 覆盖:不存在 ✅
```

**证据 ② 的旁证**：`act="relu"` 时名字里**没有** `_relu` 后缀（cmd1 的 out 是 `clean_uniform_m14b`，
不是 `..._relu`）⇒ 新增的命名分支**只在 act≠relu 时执行**，既有命令行行为未变。

**副产物核验**：`os.makedirs` 被 patch ⇒ **没有真的建目录**；证据 ③ 的`覆盖`列由 `os.path.exists` 实测，
5 个目标路径**仍全部不存在** ✅

---

## 6. 判据的复算方式（从哪个 CSV 取数、怎么算）

### 6.1 口径定义（与 §13.8 逐字一致）

$$\texttt{drop@+0.3} \;=\; \text{acc}(\alpha = 0.0) \;-\; \text{acc}(\alpha = +0.3)$$

### 6.2 取数路径（**唯一权威来源**：各配置的 `alpha_sensitivity.csv`）

| 角色 | 路径 | seed 数 |
|---|---|---|
| **5×max / ReLU**（既有） | `outputs_cifar100/v3_methods/simple_cnn_mp_clean_uniform{,_s43,_s44}/alpha_sensitivity.csv` | 3 |
| **5×avg / ReLU**（既有，非本判据必需） | `outputs_cifar100/v3_methods/simple_cnn_mp_avg_clean_uniform{,_s43,_s44}/alpha_sensitivity.csv` | 3 |
| **2×max / ReLU**（既有，**异协议锚点**） | `outputs_cifar100/task1_simplecnn/alpha_sensitivity.csv` | 1 |
| GELU / Leaky × 2 档（**待产**） | `outputs_cifar100/v3_methods/{clean_uniform,simple_cnn_mp_clean_uniform}_{gelu,leaky}/alpha_sensitivity.csv` | 1 |
| ReLU-2pool 同协议基准（**建议补跑**） | `outputs_cifar100/v3_methods/clean_uniform/alpha_sensitivity.csv` | 1 |

**⚠ 不要从 `metrics.json` 取 `best_test_acc` 当 clean**：虽然本仓库里它与 CSV 的 `α=0.0` 行一致
（已抽查 5×max 三 seed：60.99 / 60.97 / 60.73 逐位相同），但两者的**定义不同**
（前者是训练循环内 evaluate 的峰值，后者是载入 best 权重后的扫描）。
CSV 两行相减**自洽**，不受这个区别影响 ⇒ **一律用 CSV**。

### 6.3 复算脚本（可粘贴；已用它对既有 3-seed 复算，与台账逐位一致）

**本脚本已实跑过**（2026-09-26），输出见下。

```bash
PYTHONIOENCODING=utf-8 /d/anaconda3/envs/pytorch_env/python - << 'EOF'
import csv, os
ROOT = "outputs_cifar100/v3_methods"
LEGACY_2X = "outputs_cifar100/task1_simplecnn/alpha_sensitivity.csv"

def drop(path):
    m = {r["alpha"]: float(r["accuracy"]) for r in csv.DictReader(open(path))}
    return m["0.0"], m["0.3"], m["0.0"] - m["0.3"]

def mean(xs):
    xs = list(xs)                     # ⚠ 必须物化：直接 len(generator) 会 TypeError（本脚本踩过）
    return sum(xs) / len(xs)

def std0(xs):                         # 总体标准差（ddof=0）—— 台账 §13.8 用的口径
    xs = list(xs)
    mu = mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5

rows = {
    "5xmax/relu (3-seed)": [f"{ROOT}/simple_cnn_mp_clean_uniform{s}/alpha_sensitivity.csv" for s in ("", "_s43", "_s44")],
    "5xmax/relu (s42 only)": [f"{ROOT}/simple_cnn_mp_clean_uniform/alpha_sensitivity.csv"],
    "5xavg/relu (3-seed)": [f"{ROOT}/simple_cnn_mp_avg_clean_uniform{s}/alpha_sensitivity.csv" for s in ("", "_s43", "_s44")],
    "2xmax/relu (legacy,train.py) [仅存档,不参与判据]": [LEGACY_2X],
    "2pool/relu m14b (cmd 1)": [f"{ROOT}/clean_uniform_m14b/alpha_sensitivity.csv"],
    "2pool/gelu m14b (cmd 2)": [f"{ROOT}/clean_uniform_m14b_gelu/alpha_sensitivity.csv"],
    "5pool/gelu m14b (cmd 3)": [f"{ROOT}/simple_cnn_mp_clean_uniform_m14b_gelu/alpha_sensitivity.csv"],
    "2pool/leaky m14b (cmd 4)": [f"{ROOT}/clean_uniform_m14b_leaky/alpha_sensitivity.csv"],
    "5pool/leaky m14b (cmd 5)": [f"{ROOT}/simple_cnn_mp_clean_uniform_m14b_leaky/alpha_sensitivity.csv"],
}
D = {}
for k, paths in rows.items():
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f"{k:34s}  [缺]")
        continue
    ds = [drop(p) for p in paths]
    D[k] = mean([d[2] for d in ds])
    extra = (f"  per-seed {[round(d[2],2) for d in ds]}  std(ddof=0)={std0([d[2] for d in ds]):.2f}"
             if len(ds) > 1 else "")
    print(f"{k:34s}  n={len(ds)}  clean={mean([d[0] for d in ds]):.2f}  "
          f"a+0.3={mean([d[1] for d in ds]):.2f}  drop={D[k]:.2f}{extra}")

print("\n--- 判据复算（裁决 1：两臂同协议；P = drop_2max - drop_5max，正数 = 池化保护）---")
def P(k2, k5):
    return D[k2] - D[k5]

K_P2_RELU, K_P2_GELU, K_P5_GELU = "2pool/relu m14b (cmd 1)", "2pool/gelu m14b (cmd 2)", "5pool/gelu m14b (cmd 3)"
K_P2_LEAKY, K_P5_LEAKY = "2pool/leaky m14b (cmd 4)", "5pool/leaky m14b (cmd 5)"

P_relu = None
if K_P2_RELU in D:
    P_relu = P(K_P2_RELU, "5xmax/relu (3-seed)")
    print(f"P(relu)   = {D[K_P2_RELU]:.2f} - {D['5xmax/relu (3-seed)']:.2f} = {P_relu:.2f} pp"
          f"   [主分析: 另一臂用 3-seed 均值]")
    print(f"  同 seed 42 配对版 = {D[K_P2_RELU]:.2f} - {D['5xmax/relu (s42 only)']:.2f}"
          f" = {P(K_P2_RELU, '5xmax/relu (s42 only)'):.2f} pp   [一致性检查]")
else:
    print("P(relu)   [缺] -- 第 1 条命令未跑，判据无法计算")

for act, lvl, k2, k5 in (("gelu", "主组", K_P2_GELU, K_P5_GELU),
                         ("leaky", "补充", K_P2_LEAKY, K_P5_LEAKY)):
    if P_relu is not None and k2 in D and k5 in D:
        p, d = P(k2, k5), P_relu - P(k2, k5)
        # ⚠ 浮点边界（§6.5 实测）：数学真值恰为边界的 d 在浮点下会判错 ——
        #    (31.90-20.60)-7.30 = 3.9999999999999973 ⇒ 裸 `d >= 4.0` 为 False，而数学真值是 4.0。
        #    与 V7_COORDINATION.md §五 记录的 73.09-73.08 事故同族。round 到 1e-6 消除。
        d = round(d, 6)
        verdict = ("① 支持" if d >= 4.0 else "② 否证(保护反增)" if d <= -4.0 else "③ 不可分辨")
        print(f"P({act:5s})     = {D[k2]:.2f} - {D[k5]:.2f} = {p:.2f} pp   "
              f"P(relu)-P({act}) = {d:+.2f} pp   -> {verdict}   [{lvl}]")
    else:
        print(f"P({act:5s})     [缺]")
EOF
```

**实测输出（2026-09-26，本会话；未产出的行显示 `[缺]` 是预期的）**：

```
5xmax/relu (3-seed)                 n=3  clean=60.90  a+0.3=40.29  drop=20.60  per-seed [18.52, 22.2, 21.09]  std(ddof=0)=1.54
5xmax/relu (s42 only)               n=1  clean=60.99  a+0.3=42.47  drop=18.52
5xavg/relu (3-seed)                 n=3  clean=61.38  a+0.3=39.61  drop=21.78  per-seed [23.05, 20.97, 21.31]  std(ddof=0)=0.91
2xmax/relu (legacy,train.py) [仅存档,不参与判据]  n=1  clean=59.99  a+0.3=28.09  drop=31.90
2pool/relu m14b (cmd 1)             [缺]
2pool/gelu m14b (cmd 2)             [缺]
5pool/gelu m14b (cmd 3)             [缺]
2pool/leaky m14b (cmd 4)            [缺]
5pool/leaky m14b (cmd 5)            [缺]

--- 判据复算（裁决 1：两臂同协议；P = drop_2max - drop_5max，正数 = 池化保护）---
P(relu)   [缺] -- 第 1 条命令未跑，判据无法计算
P(gelu )     [缺]
P(leaky)     [缺]
```

**与台账 §13.8 逐位一致 ✅**（clean 60.90 / 40.29 / 20.60 / ±1.54 与 61.38 / 39.61 / 21.78 / ±0.91 全同）。

### 6.4 ⚠ std 的口径：**ddof=0（总体标准差）**

台账 §13.8 的 `±1.54` / `±0.91` **只在 ddof=0 下成立**：

```
drops = [18.52, 22.20, 21.09]  →  std(ddof=0) = 1.54   ← 台账口径 ✅
                                  std(ddof=1) = 1.89   ← pandas/numpy 默认，对不上 ❌
```

numpy 的 `std()` 默认 ddof=0（对），**pandas 的 `.std()` 默认 ddof=1（错）**。
M14b 复算时若用 pandas，会得到 1.89 / 1.11 并误以为台账错了。**先确认口径，再报不一致。**

### 6.5 判据逻辑自测（**造已知坏例**，项目纪律）

判据的判决逻辑在新数据产出前**从未被真实数据触发过**。按本项目"**报'工具抓到了 X'之前，
先确认工具本身不错（读源码 / 造已知坏例自测）**"的纪律，我造了合成 CSV 把它跑了一遍。

**方法**：从本文档 §6.3 **提取代码块原文**（不是另写一份）→ 把 `ROOT` 指向临时目录 →
填入已知 drop 的假 CSV → 检查判决。

| 用例 | 造出的 d | 期望 | 实得 | |
|---|---|---|---|---|
| ① 支持 | +8.00 | ① 支持 | ① 支持 | ✅ |
| ② 否证 | −5.00 | ② 否证(保护反增) | ② 否证(保护反增) | ✅ |
| ③ 不可分辨 | +2.00 | ③ 不可分辨 | ③ 不可分辨 | ✅ |
| 近边界（上侧） | +4.01 | ① 支持 | ① 支持 | ✅ |
| 近边界（下侧） | +4.00 | ③ 不可分辨 | ③ 不可分辨 | ✅ |

**浮点边界的最小复现（`round` 的救场案例）**：

```
(31.90-20.60)-7.30 = 3.9999999999999973
裸 d >= 4.0        -> False   ← 数学真值就是 4.0，却被判 False
round(d, 6) >= 4.0 -> True    ← 修复后
```

⇒ 判据里的 `d = round(d, 6)` **不是装饰**：它与 `V7_COORDINATION.md` §五记录的
`73.09-73.08` 事故**同族**（浮点边界让"容差判据"误报）。**不要删。**

> ⚠ **自测过程中我自己误判过一次，如实记录**：第一版用例里我打印 `{d:+.2f}`，
> 看到 `+4.00` 就以为"脚本有浮点 bug"，还写了一条**错误的注释**（说真值是 3.99999…）。
> 实际 `P(relu)` 的真值是 **11.296666666666663**（3-seed 均值不是 11.30），
> 所以 d 真值 = 3.9967，判③**本来就是对的**。
> **教训与项目既有记录同族**："**一个数如果换个跑法就变，那它不是数**"——
> 这里是"**格式化后的显示值不是真值**"。诊断浮点问题**必须打 `repr()`，不能看 `%.2f`**。
> （注释已更正为上面那条真实复现。）

---

## 7. ⚠ 两个必须先裁决的方法学问题（本次准备**最重要的产出**）

这不是"我没做到"，而是**判据本身的前提未检验**。现在仍在"跑前"，是修判据的最后窗口。

### 7.1 问题一：判据基准 **11.30 pp 是跨协议的**

| 数 | 出处脚本 | 性质 |
|---|---|---|
| 2×max drop = 31.90 | `train.py`（第一阶段） | **异协议** |
| 5×max drop = 20.60 | `v3_methods_simplecnn.py`（§13.8） | 本判据的所属协议 |

§13.8 **自己明写**：2×max 基线"**仅作锚点，不参与同协议对比**"，理由改判为
"**出自不同的训练脚本**"，依据是 §14.5/§14.6 的「**跨训练实现不可移植**」。

**项目已量化该不可移植的量级：5.55 pp（未归因）**。
⇒ **判据阈值 1 pp 只有协议差的 18%。** 用跨协议基准去判"缩小 ≥1 pp"，
等于用一把刻度误差 5.55 的尺子去量 1 的差 —— **仪器精度不够**，无论结果落哪边都说不清。

**修法（二选一）**：

- **(A) 补跑同协议 ReLU-2pool**（推荐）：加 §5.5 的第 5 条命令（≈30~57 min），
  判据改用 **Δ(relu, 同协议)** 作基准。**这是唯一能让判据物理可满足的改法。**
- **(B) 保留跨协议基准，但判据降级**：只作**定性**对照（"方向上是否缩小"），
  **不做 1 pp 的二元判决**，并在论文/台账里注明基准跨协议。

### 7.2 问题二：判据阈值 1 pp **小于单 seed 噪声**（§16.8 第 6 条同族缺陷）

§13.8 实测的 drop 3-seed 标准差：**5×max ±1.54、5×avg ±0.91**
（**ddof=0 口径**，已复算确认，见 §6.4）。

单 seed 下，Δ = drop(base) − drop(mp) 的噪声 ≈ $\sqrt{\sigma_{2p}^2 + \sigma_{5p}^2}$
≈ $\sqrt{1.5^2+1.5^2}$ ≈ **2.1 pp**（设 base 档 σ 同量级）。

⇒ **阈值 1 pp ≈ 0.5σ ⇒ 单 seed 不可裁决。**

**这不是新问题，是 §16.8 已经审出来的缺陷模式**——第 6 条：
「阈值 1.0 / 2.0 pp 固定，**未检验实测差在该阈值下的可分辨性**」，
而 §16.8 的结论是「**8 条同类缺陷（27%），其中 5 条足以翻转**」。
M14b 的判据**正是第 6 条的复制**（阈值 1 pp 写死，未与 σ 对照）。

**更要命的推论**：§13.8 已明写「要把该成分从'初步'升为定论，需要 **≥8 seeds/组**」。
M14b 若只跑 1 seed/组，**连 §13.8 的可分辨性都没达到**（§13.8 有 3 seeds）。

**可选修法（跑前定）**：

- **(A) 判据加"可分辨性前提"**：明确写"本判据在单 seed 下不可裁决；
  若 |Δ(act) − Δ(relu)| < 2σ ≈ 4 pp，判**不可分辨**（第三分支），不作支持/否证"。
- **(B) 只保留大效应**：改用 M14 的经验阈值 **≥3 pp**（"<3 pp 的效应在 seed 噪声内无法裁决"）。
  但注意：M14b 的假说预测的正是"**缩小 ≥1 pp**"量级的效应，**阈值一提到 3 pp，
  假说就变成不可测**——这是必须由总管知情的取舍。
- **(C) 加 seed**：3 seeds × 4 配置 = 12 次 ≈ 6~11.4 h。**8 小时窗口装不下**
  （且本机此刻可能还有 M8c 的 GPU 任务在排队）。

### 7.3 我的建议（供裁决，不代替裁决）

**在 8 h 窗口内可行的最大信息量方案**：

1. **补跑同协议 ReLU-2pool**（1 次）——修掉 §7.1；
2. **跑 4 个新配置**（4 次）——产出 GELU/Leaky 的 Δ；
3. 判据按 **(A) 加可分辨性前提**——报**点估计 + 噪声带**，落噪声内则判"不可分辨"，
   **不作支持/否证**（这正是 §13.8 判据③的先例：不明说"是/否"，如实记"不予裁决"）；
4. 全部 5 次 **seed 42**，与 §13.8 的 s42 行同 seed ⇒ 至少与 s42 那一格同源可比。

总计 **5 次 ≈ 2.5~4.8 h**，在 8 h 内，且**留下的是一份可解释的结果**。
若跳过第 1 步，省 1 小时但**判据在原理上不成立**——得不偿失。

### 7.4 裁决记录（总管 2026-09-26 已下）

| 问题 | 裁决 | 落地位置 |
|---|---|---|
| §7.1 跨协议基准 | **采纳 A** —— 补跑同协议 ReLU-2pool，判据改用两臂同协议的 Δ(relu) | §5.5 第 1 条命令；§5.7 |
| §7.2 阈值小于噪声 | **采纳 A** —— 判据加**第三分支（不可分辨）**，阈值取 2σ | §6.3 判决逻辑；下方 ⚠ |
| §10.1 操纵强度 | **主组 = GELU**；LeakyReLU 降为**补充档**，且**leaky 无效应不得据此否证假说** | §5.5 第 4/5 条标注"补充档" |

#### ⚠⚠ 两处**需要总管确认**的转述风险（我不敢自行裁定）

**① 阈值：裁决书里有两个不相容的数。**
- 裁决 1 写：「M14b 支持 ⟺ Δ(gelu) 相对 Δ(relu) **缩小 ≥1 pp**」
- 裁决 2 写：「① 支持 | Δ 缩小 **≥ +4 pp**（≈2σ）」

**我按 4 pp 实现**（理由：裁决 2 是**后发的、且专门为"可分辨性"而写**，1 pp 正是它要修掉的那个数）。
**若总管的原意是 1 pp，请说一句，我改回**——改的只是 §6.3 里两个 `4.0`。

**② 符号：裁决 1 的 Δ 定义是负的。**
裁决 1 写 `Δ(relu) = drop_5max(relu) − drop_2max(relu)`，代入实测值 = `20.60 − 31.90` = **−11.30**。
于是"保护变弱"对应 **Δ 变大（趋向 0）**，而"缩小"二字在负数上容易读反。

**我在 §6.3 里改用等价的 `P = −Δ = drop_2max − drop_5max`（正数 = 池化保护）**，
判决写成 `P(relu) − P(gelu)`：
- `≥ +4 pp` → ① 支持（保护确实变弱）
- `≤ −4 pp` → ② 否证（保护反而增强）
- 其余 → ③ 不可分辨

**这只是符号约定，与裁决的定义完全等价**（`P = −Δ`）。若总管要求严格保留 Δ 的写法，我改。

> **判据的最终形态（供台账引用）**：
> 同协议两臂下，`P(act) = drop_2max(act) − drop_5max(act)`；
> `P(relu) − P(gelu) ≥ 4 pp` → 支持 ReLU 非负性假说；`≤ −4 pp` → 否证；
> **`|·| < 4 pp` → 第三分支"不可分辨"，如实记"单 seed 在原理上判不了这个量级的效应"。**

---

## 8. 可选（几乎零成本）：把"激活形状"与"训练动力学"分开的机制探针

**想法**：拿**已训好的 ReLU 权重**（`checkpoints_cifar100/v3_simple_cnn_mp_clean_uniform/best_model.pth`），
**不重训**，直接把激活换成 GELU/LeakyReLU 重新跑一次 α 扫描。

**它回答什么**："**在该权重下**，非负性本身对 α 鲁棒性的贡献"——即纯**机制**问题。
**它不回答什么**："用 GELU **训练出来**的架构，池化保护是否变弱"（训练动力学变了）。

⇒ 它是**探针**，不能替代预登记判据；但成本只有几分钟。
**若探针与 4 配置训练的方向一致 → 相互印证；不一致 → 说明效应主要来自训练动力学而非激活形状**
（这本身是有价值的发现，且能反过来解释为什么"只换激活不重训"不足）。

**实现**：把 §5 的 diff 应用后，下面这段可直接跑（**只前向、不训练**）。
**⚠ 本段代码我未执行过**（红线禁止本会话跑训练/评估任务；它需要载入测试集与权重）。
它的**第 1 行输出（`act=relu`）自带阳性对照**：必须复现既有 s42 的 60.99 / 42.47 / 18.52 ——
复现不了就是链路有问题，不要信后两行（这条纪律来自本项目"先做阳性对照"的教训）。

```bash
# 用法示例：把既有的 5×max ReLU 权重载进 GELU 镜像，跑 7 点 α 扫描（CPU 亦可）
PYTHONIOENCODING=utf-8 /d/anaconda3/envs/pytorch_env/python - << 'EOF'
import sys, torch
sys.path.insert(0, '.')
from utils.nonlinearity import nonlinearity, register_nonlinearity_hooks, remove_hooks
from v3_methods_simplecnn import build_loaders, scan
from models.simple_cnn_act import SimpleCNNMaxPoolAct

dev = "cuda" if torch.cuda.is_available() else "cpu"   # ⚠ 见 §10 未验证项 ①：设备口径
_, test_loader = build_loaders(batch_size=256, num_workers=0, dataset="cifar100")
ck = torch.load("checkpoints_cifar100/v3_simple_cnn_mp_clean_uniform/best_model.pth", map_location=dev)

for act in ("relu", "gelu", "leaky"):
    m = SimpleCNNMaxPoolAct(num_classes=100, act=act).to(dev)
    m.load_state_dict(ck["model_state_dict"])          # 层名一致 ⇒ 直接载入
    m.eval()
    a7 = scan(m, test_loader, dev, [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    d = dict(a7)
    print(f"act={act:5s}  clean={d[0.0]:.2f}  a+0.3={d[0.3]:.2f}  drop={d[0.0]-d[0.3]:.2f}")
EOF
```

**预期**：`act=relu` 那一行必须复现 `clean=60.99 / a+0.3=42.47 / drop=18.52`（既有 s42 的数）。
**若复现不了，说明扫描链路有问题，先别信后两行。**

---

## 9. 交付物清单与零覆盖核验

| 文件 | 状态 |
|---|---|
| `models/simple_cnn_act.py` | ✅ 新增（唯一新增的代码文件） |
| `docs/verify/M14B_PREP.md` | ✅ 新增（本文件） |
| **`v3_methods_simplecnn.py`** | ✅ **已按总管授权 (a) 应用 §5.1–5.4**（`git diff` = 16 insertions / 2 deletions；`act="relu"` 逐位不变的证据见 **§5.8**） |
| `task_extension6_deep_robust_train.py` | ✅ **本会话零改动**（理由见 §5.6）。⚠ 该文件 **02:11:37 被 M5 工人会话**改动 —— 与本会话无关，见 §9.1 |
| `v3_readout_repair.py` | ✅ **本会话零改动**。⚠ 同样在 **02:11:49 被 M5 工人**改动，见 §9.1 |
| 其余既有文件 | ✅ 零改动 |

#### 9.1 ⚠ 并发现场：两个文件在 02:11:37 / 02:11:49 被**别的会话**改了

**这不是本会话的改动**，但如果只扫 `git status` 会误判。证据三条：

1. **内容**：两个文件的 diff 全是 `wide_cnn` / `SimpleCNNWideV2` / `M5 的第 4 个容量档` 字样；
   **本会话的 diff 里没有任何这些字符串**（我改的是 `--act` / `build_arch` 的 act 形参 / 命名后缀）。
2. **依赖**：diff 引用的 `models/simple_cnn_wide.py` 是 M5 工人的**未跟踪新文件**，**不是本会话创建的**
   （本会话只新建过 `models/simple_cnn_act.py`）。
3. **时间戳**（`ls --time-style=full-iso`）：
   ```
   02:11:08  v3_methods_simplecnn.py     ← 本会话（§5.1–5.4）
   02:11:37  task_extension6_deep_robust_train.py   ← M5
   02:11:49  v3_readout_repair.py                   ← M5
   ```

**时间线交代**：本会话在 **02:0x** 向总管报告"extension6 零改动、md5 `568ef44b0b76079f26528c762b59f48f`"——
**那一刻是真的**。此后 M5 在 02:11:37 改了它，所以现值（`187cc9b6cee0e84ce15445eab662c872`）
是 **M5 的改动**所致，不是本会话违约。**本会话从未写过这个文件的一个字节。**

> **附带印证 §5.6 的判断**：M5 给 extension6 加分支时要**同时**改 `--model` 的 `choices`
> （见其 diff 的 `@@ -519,8 +542,8 @@` hunk）—— 正好印证那里"必须改既有行"的说法。
> 而这次扩展服务的是 **M5 的容量档**，**不是 M14b**。

**⚠ 一处口径更正**：任务简报里写「`models/simple_cnn_mp.py`（重点：5×MaxPool 那个变体怎么写的）」以及
「**尤其 `models/simple_cnn_mp.py` 不能改**」——**这个文件在仓库里不存在**。
5×MaxPool 的实现实际在 **`v3_methods_simplecnn.py:86`** 的 `SimpleCNNMaxPool`（仓库根目录，非 `models/`）。
我按"不改该实现"的红线本意执行：**没有碰它**。

---

## 10. 我没能验证的部分（如实列出）

| # | 未验证项 | 为什么不能验证 | 影响 |
|---|---|---|---|
| ① | **新配置在 CUDA 上的数值** | 红线禁止 GPU 训练，我只做了 CPU 前向 | §三.4「设备口径也是口径」：CPU 与 CUDA 的 `head` **不逐位相同**（实测 \|Δ\|max ≈ 4e-3）。⇒ **训练与扫描必须在同一设备**；本次所有数字都是 CPU，**不可与 CUDA 产物并列相减** |
| ② | **换激活后 drop 会不会真的变化** | 那需要训练（被红线禁止） | 这正是 M14b 要测的，本文件只保证"测得了、测得准" |
| ③ | **LeakyReLU(0.01) 是否足够"破坏非负性"** | 无训练权重 ⇒ 无法测训练后的激活分布 | **见 §10.1，这是本次最该被知情的风险** |
| ④ | **训练后 BN 会不会把差异抹平** | 同上 | BN 在激活前重新标准化，可能削弱激活形状的效应；探针（§8）可部分回答 |
| ⑤ | **`--act` 的实际接线是否可跑** | 领地只允许我写 `models/` 的那个新文件 + 本 md；`v3_methods_simplecnn.py` 只读 | §5 的 diff 是**逐行对照源码写的**，但**未执行过**。应用后请先跑一次 `--epochs 1` 冒烟 |
| ⑥ | **5 次训练的显存/内存占用** | 未跑 | 与既有的 clean 训练同规模（参数量、batch、输入全同）⇒ 风险低；但本机内存薄（§三.1 有两次 1455 前科），**仍须串行** |
| ⑦ | **Pertl 文献的"轴对齐"断言本身** | 只有摘要逐字核实过；"轴对齐向量在等能量扰动下损失更小"那条来自正文图 5，**尚未核实** | M14b 的**动机**建立在这条上。若它不成立，M14b 的可测预测仍成立（换激活→保护变弱），但**解释框架**会变 |

### 10.1 ⚠ 最该被知情的风险：LeakyReLU(0.01) 的操纵强度实测很弱

本次实测（§4 的 `[3]`，`n=200 000` 个 N(0,1)·1.5 的预激活）：

| act | 输出 <0 的**比例** | 负值**能量**占比 | 输出最小值 |
|---|---|---|---|
| `relu` | 0.0000 | 0.000e+00 | 0.0000 |
| `gelu` | **0.5011** | **6.372e-03** | −0.1700 |
| `leaky`(0.01) | **0.5012** | **1.008e-04** | −0.0684 |

模型内（5×pool、随机初始化、eval）四层 `relu1..relu4` 的负值比例：

| act | r1 | r2 | r3 | r4 |
|---|---|---|---|---|
| `relu` | 0.000 | 0.000 | 0.000 | 0.000 |
| `gelu` | 0.496 | 0.554 | 0.543 | 0.520 |
| `leaky` | 0.504 | 0.458 | 0.471 | 0.449 |

**读法**：

- **负值"个数"上**：gelu 与 leaky 都 ≈50%，合格；
- **负值"能量"上**：gelu 是 leaky 的 **63 倍**（6.4e-3 vs 1.0e-4）。
  `leaky(0.01)` 把负值幅值压缩了 100 倍 ⇒ **能量意义上几乎没有破坏非负性**。

**后果**：若 leaky 组测出"无效应"，**无法区分**下面两种解释——

- (a) 假说被否证（非负性不是保护来源）；或
- (b) **操纵太弱**（负能量占比 1e-4，等于什么都没改）。

⇒ **建议在预登记里写死一条"操纵检查前置"**：先看 §4 的 `[3]` 输出，
若 leaky 的负能量占比低于某个阈值（例如 1e-2），**则 leaky 组只作参考，判决以 gelu 组为准**。
若要更平衡的两点对照，可把 `LEAKY_NEGATIVE_SLOPE` 改为 **0.1**（负能量占比约 1e-2，与 gelu 同量级），
但那**又是改一个实验设计变量**——**改与不改都要在跑前写死**。

---

## 11. 一句话总结

**代码侧已就绪且三重自检通过**（参数量一致 / 数值逐位一致 / 注入目标一致）；
**但判据侧有两个前提未检验**（跨协议基准 5.55 pp ≫ 阈值 1 pp；单 seed 噪声 2.2 pp ≫ 阈值 1 pp），
**两者都能在跑前修好，成本是 +1 次训练（≈0.5~1 h）和一段判据文字**。
建议按 §7.3 的 5 次方案执行；**不要**用 §5.5 的 4 条命令直接开跑。
