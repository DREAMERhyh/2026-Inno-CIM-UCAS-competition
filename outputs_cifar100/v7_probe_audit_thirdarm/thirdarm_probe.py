# -*- coding: utf-8 -*-
"""
C5 第三臂探针：**α 混合（可部署、α-无关）线性探针**
—— 补齐 v7_probe_audit 探针表的第三臂，回答"论文开放问题第 3 条的缺口里有多少只是'不知道 α'"。

状态：**未冒烟**（本机提交余量不足，2026-09-24 16:2x 测得可用提交 ≈1.5 GB，
而项目自己的事故记录是 4.9 GB 时炸过 1455）。首次执行**务必**先 `--smoke 256`。

⚠ 判据冻结说明：C1–C4 于 2026-09-24 16:30 前后冻结，**之后未改动**。
16:5x 的修改**只涉及实现**（import 路径避开 matplotlib、退出码、冒烟模式的保真豁免、导入前内存闸门），
不触碰任何判据文字与阈值。

退出码：0 = 正常完成；2 = S1 自检不过；3 = C2 保真不过；4 = 内存红线（导入前闸门或逐 batch 守卫）。
日志重定向需 `PYTHONIOENCODING=utf-8`（本机控制台 GBK）。

════════════════════════════════════════════════════════════════════════════
预登记判据（**跑前写死，跑后不改**；写定 2026-09-24 16:30 前后，作者 claude-e9）
════════════════════════════════════════════════════════════════════════════
记号：p_same(α) = 现有 oracle-α 探针（同条件训练）；p_clean(α) = 干净特征训练探针；
      p_mixed(α) = 本脚本的 α 混合探针；Δ_proc = 3.45 pp（论文自订第二级散布）。

C1（主判据）  Δ_α := p_same(+0.3) − p_mixed(+0.3) = 48.35 − p_mixed(+0.3)
    分支①  Δ_α ≥ 3.45 ⇒ "不知道 α"的代价**可分辨**：论文开放问题第 3 条应**改写**
           （缺口 = α-未知成分 Δ_α + 优化成分），**不得删除**；
    分支②  Δ_α <  3.45 ⇒ **不可分辨**：开放问题第 3 条应**删除**（该问在本文分辨力内不可裁决）。
    阈值唯一（Δ_proc = 3.45 pp）。不使用种子噪声：探针是单次确定性拟合（lbfgs 无随机性），
    本文可测的散布只有"过程自由度"这一级。
C2（管线保真）  用本脚本重跑 same-cond(+0.3)，必须复现 48.35。容差取**项目自己的 R0 容差 ±0.30 pp**
    （v7_probe_audit.py:251「headroom 继承 probe 的容差（±0.3 pp）」）。
    不通过 ⇒ 停止，不报第三臂（说明本脚本的提取/拟合与 M0c 不同源）。
C3（异常保护）  p_mixed(+0.3) 若落在 [18.10, 48.35] = [p_clean, p_same] 之外 ⇒
    **先查管线再报**，不把它当科学结论。
C4（描述性，无分支）  9 个 α 点上的三臂曲线；Δ_α / Δ_proc 比值。

════════════════════════════════════════════════════════════════════════════
口径（与 M0c 规范口径逐项对齐，除标注处外）
════════════════════════════════════════════════════════════════════════════
主干     : checkpoints_cifar100/simple_cnn/best_model.pth（clean SimpleCNN / CIFAR-100，冻结）
特征     : **失真后**的 fc 输入（GAP 后 128 维）。顺序与 v7_probe_audit.run_condition 一致：
           先注册失真钩子、再注册捕获钩子（pre-hook 按注册序执行，故捕获到的是失真后的张量）
探针     : StandardScaler + LogisticRegression(max_iter=1000, n_jobs=None) —— 与 PROBE_KW 逐字一致
n_train  : 8000（train split 前 8000 张，shuffle=False，与 M0c 同一批图）
n_test   : 10000（全量测试集）
α 采样   : **逐样本 i.i.d. ~ U[−0.3,+0.3]**，同一样本在**所有层**共用同一个 α
随机种子 : 20260924（写死）
设备     : **CPU**（M4 训练占用 GPU；本脚本不建 CUDA 上下文）
内存纪律 : n_jobs=None、num_workers=0、逐 batch 查可用提交；低于下限即中止并落盘部分结果

产物：ra_thirdarm.csv / thirdarm_meta.json（本目录）

用法：
    python thirdarm_probe.py --smoke 512      # 冒烟：自检 + 保真（2 个 batch），不写正式产物
    python thirdarm_probe.py                  # 正式跑（需内存余量充足）
冒烟规模取 512 而非 256：2 个 batch 才能同时走通"逐 batch 注册/摘除钩子 + RNG 递进"这条路径；
且 512 远小于 8000，保真值不可能被误读成正式结果（脚本会显式标注"冒烟值，无意义"）。
"""

import argparse
import csv
import ctypes
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# ----------------------------------------------------------------------
# 导入前内存闸门（**在任何重依赖 import 之前**）——红灯时连 torch 都不加载
# ----------------------------------------------------------------------
START_MIN_FREE_MB = 1500.0   # 启动闸门：我自己的占用（估 0.6~1.2 GB）+ 给并发任务留 >=0.5 GB
MIN_FREE_COMMIT_MB = 800.0   # 运行期守卫：低于此值立即中止（保护他人训练）


class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def commit_mb() -> float:
    """系统可用提交内存（MB）= 提交限 − 已提交，与 Task Manager 的"提交"余量同源。"""
    st = _MEMSTAT()
    st.dwLength = ctypes.sizeof(_MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return st.ullAvailPageFile / 1e6


_free0 = commit_mb()
print(f"[thirdarm] 导入前内存闸门：可用提交 {_free0:.0f} MB（闸门 {START_MIN_FREE_MB:.0f} MB）", flush=True)
if _free0 < START_MIN_FREE_MB:
    print(f"[abort] 可用提交 {_free0:.0f} MB < 启动闸门 {START_MIN_FREE_MB:.0f} MB："
          f"不加载 torch，直接退出（保护正在跑的训练）。语法自检请用 py_compile（不执行本文件）。",
          flush=True)
    raise SystemExit(4)

import numpy as np          # noqa: E402
import torch                # noqa: E402
import torch.nn as nn       # noqa: E402
from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402
from torch.utils.data import DataLoader               # noqa: E402
from torchvision import datasets, transforms          # noqa: E402

# 参考失真钩子：**按文件路径直载**，绕开 utils/__init__.py
# （后者会 import .trainer → matplotlib + tqdm + sklearn.metrics，白烧提交内存；
#  nonlinearity.py 自身只依赖 torch。这一步在 1.5 GB 余量的机器上是实打实的节省。）
_spec = importlib.util.spec_from_file_location("ref_nonlinearity", os.path.join(ROOT, "utils", "nonlinearity.py"))
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)
register_nonlinearity_hooks = _ref.register_nonlinearity_hooks
remove_hooks = _ref.remove_hooks

CKPT = os.path.join(ROOT, "checkpoints_cifar100", "simple_cnn", "best_model.pth")
DATA_ROOT = os.path.join(ROOT, "data")
OUT_CSV = os.path.join(HERE, "ra_thirdarm.csv")
OUT_META = os.path.join(HERE, "thirdarm_meta.json")
OUT_PARTIAL = os.path.join(HERE, "thirdarm_partial.json")

MEAN = (0.5071, 0.4865, 0.4409)   # 与 v7_probe_audit.CIFAR100_MEAN/STD 一致
STD = (0.2673, 0.2564, 0.2762)
PROBE_KW = dict(max_iter=1000, n_jobs=None)   # 与 v7_probe_audit.PROBE_KW 逐字一致
N_TRAIN = 8000
N_TEST = 10000
MIX_SEED = 20260924
ALPHA_MAX = 0.3
BATCH = 256
ALPHAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# （START_MIN_FREE_MB / MIN_FREE_COMMIT_MB / _MEMSTAT / commit_mb 已在文件头定义：
#   导入前闸门必须先于 torch import，故那些定义上移，此处不再重复。）


# ----------------------------------------------------------------------
# 内存守卫（运行期：逐 batch）
# ----------------------------------------------------------------------
_MEM_TICK = {"n": 0}


def mem_check(where: str, hard: bool = True, every: int = 8) -> float:
    _MEM_TICK["n"] += 1
    m = commit_mb()
    if _MEM_TICK["n"] % every == 0 or where.startswith(("启动", "阶段", "结束")):
        print(f"  [mem] {where}: 可用提交 {m:.0f} MB", flush=True)
    if hard and m < MIN_FREE_COMMIT_MB:
        print(f"[abort] 可用提交 {m:.0f} MB < 运行期红线 {MIN_FREE_COMMIT_MB:.0f} MB，"
              f"中止以保护并发训练（阶段：{where}）；已写的部分结果保留在 thirdarm_partial.json", flush=True)
        sys.exit(4)
    return m


# ----------------------------------------------------------------------
# 失真：逐样本 α（公式与 utils.nonlinearity.nonlinearity 逐字一致，只把标量换成 (B,) 向量）
# ----------------------------------------------------------------------
def distort_per_sample(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """y = a*x̂³ + (1−a)*x̂，x̂ = x / clamp(max|x|, 1e-12)（逐样本、按样本展平求 max）。
    参考实现在 |α|<1e-12 时短路返回 x；本脚本的 a 为连续采样（取到精确 0 的概率为 0），
    省去该分支只是免掉一次等价表达式，数值上与参考实现一致（由 S1 自检验证）。"""
    b = x.shape[0]
    shape = [b] + [1] * (x.dim() - 1)
    xf = x.reshape(b, -1)
    mx = torch.clamp(xf.abs().max(dim=1, keepdim=True).values, min=1e-12).reshape(*shape)
    a = a.reshape(*shape).to(x.dtype)
    xn = x / mx
    return (a * xn ** 3 + (1.0 - a) * xn) * mx


def register_per_sample_hooks(model, alpha_vec_getter):
    """逐样本 α 版本。返回的句柄列表交给 utils.nonlinearity.remove_hooks 统一移除。"""
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            def make(mod_name=name):
                def pre_hook(module, inputs):
                    if len(inputs) == 0:
                        return inputs
                    a = alpha_vec_getter(inputs[0])
                    return (distort_per_sample(inputs[0], a),) + tuple(inputs[1:])
                return pre_hook
            hooks.append(m.register_forward_pre_hook(make()))
    return hooks


# ----------------------------------------------------------------------
# 前向 + 捕获（顺序与 run_condition 一致：先失真钩子，后捕获钩子）
# ----------------------------------------------------------------------
def fc_input_under(net, x, make_hooks):
    hooks = make_hooks()
    store = {}
    hcap = net.fc.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0]))
    try:
        with torch.no_grad():
            net(x)
    finally:
        hcap.remove()
        if hooks:
            remove_hooks(hooks)
    return store["f"].detach()


def build():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    tr = datasets.CIFAR100(root=DATA_ROOT, train=True, download=False, transform=tf)
    te = datasets.CIFAR100(root=DATA_ROOT, train=False, download=False, transform=tf)
    return (DataLoader(tr, batch_size=BATCH, shuffle=False, num_workers=0),
            DataLoader(te, batch_size=BATCH, shuffle=False, num_workers=0))


def load_backbone():
    from models.simple_cnn import SimpleCNN
    net = SimpleCNN(num_classes=100)
    ck = torch.load(CKPT, map_location="cpu")
    net.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    net.eval()
    return net


@torch.no_grad()
def extract(net, loader, limit, alpha=None, mix_rng=None):
    """alpha 为标量（可为 0.0）⇒ 用项目参考钩子；给了 mix_rng ⇒ 逐样本混合 α。
    limit 只影响前向多少张，不改变取哪些样本（与 v7_probe_audit.extract 同语义）。"""
    feats, labels, n = [], [], 0
    for x, y in loader:
        if mix_rng is not None:
            a = torch.tensor(mix_rng.uniform(-ALPHA_MAX, ALPHA_MAX, size=x.shape[0]), dtype=torch.float32)
            f = fc_input_under(net, x, lambda _a=a: register_per_sample_hooks(net, lambda t: _a[: t.shape[0]]))
        else:
            f = fc_input_under(net, x, (lambda al=alpha: (register_nonlinearity_hooks(net, al) if al else [])))
        feats.append(f.cpu().numpy())
        labels.append(np.asarray(y))
        n += x.shape[0]
        mem_check(f"extract {n} 张")
        if limit and n >= limit:
            break
    return np.concatenate(feats)[:limit], np.concatenate(labels)[:limit]


def fit_probe(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(**PROBE_KW).fit(sc.transform(Xtr), ytr)
    return sc, clf


def acc(sc, clf, Xte, yte):
    return float(clf.score(sc.transform(Xte), yte)) * 100.0


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help=">0 时只跑自检+保真（该样本数），不写正式产物")
    args = ap.parse_args()
    n_train = args.smoke or N_TRAIN
    n_test = args.smoke or N_TEST

    print(f"[thirdarm] CPU-only 第三臂探针；n_train={n_train} n_test={n_test} mix_seed={MIX_SEED}",
          flush=True)
    m0 = mem_check("启动", hard=True)

    tr_ld, te_ld = build()
    net = load_backbone()

    # ---- S1 自检：逐样本钩子在常数 α 下必须与项目参考钩子数值一致 ----
    xb, _ = next(iter(tr_ld))
    ref = fc_input_under(net, xb, lambda: register_nonlinearity_hooks(net, 0.3))
    mine = fc_input_under(net, xb, lambda: register_per_sample_hooks(net, lambda t: torch.full((t.shape[0],), 0.3)))
    s1 = float((ref - mine).abs().max().item())
    print(f"[S1 自检] 常数 α=+0.3：参考钩子 vs 逐样本钩子 最大绝对差 = {s1:.3e}", flush=True)
    if s1 > 1e-3:
        print("[abort] S1 自检未过：逐样本钩子与参考实现不等价，停止", flush=True)
        sys.exit(2)

    # ---- C2 保真：重跑 same-cond(+0.3) ----
    Xtr_same, ytr_same = extract(net, tr_ld, n_train, alpha=0.3)
    Xte_03, yte = extract(net, te_ld, n_test, alpha=0.3)
    sc, clf = fit_probe(Xtr_same, ytr_same)
    p_same = acc(sc, clf, Xte_03, yte)
    d = abs(p_same - 48.35)
    tag = "  ← 冒烟模式：样本数不足，此值不作判据" if args.smoke else ""
    print(f"[C2 保真] same-cond(+0.3) 复现 = {p_same:.2f}（M0c 48.35，差 {d:.2f}，容差 0.30）{tag}",
          flush=True)
    if d > 0.30 and not args.smoke:
        print("[abort] C2 保真未过，停止（不报第三臂）", flush=True)
        sys.exit(3)

    # ---- 干净臂（对照，重算） ----
    Xtr_clean, ytr_clean = extract(net, tr_ld, n_train, alpha=0.0)
    sc_c, clf_c = fit_probe(Xtr_clean, ytr_clean)

    # ---- 第三臂：逐样本混合 α ----
    rng = np.random.default_rng(MIX_SEED)
    Xtr_mix, ytr_mix = extract(net, tr_ld, n_train, mix_rng=rng)
    mem_check("阶段 第三臂拟合前", hard=True)
    sc_m, clf_m = fit_probe(Xtr_mix, ytr_mix)
    print("[thirdarm] 混合臂拟合完成", flush=True)

    # ---- 评估：9 个条件 ----
    rows = []
    for a in ALPHAS:
        Xte, yte_a = (Xte_03, yte) if a == 0.3 else extract(net, te_ld, n_test, alpha=a)
        rows.append({
            "alpha": a,
            "probe_same_cond": round(acc(sc, clf, Xte, yte_a), 2),
            "probe_clean_train": round(acc(sc_c, clf_c, Xte, yte_a), 2),
            "probe_mixed": round(acc(sc_m, clf_m, Xte, yte_a), 2),
        })
        r = rows[-1]
        print(f"  α={a:+.1f}  same={r['probe_same_cond']:6.2f}  clean={r['probe_clean_train']:6.2f}  "
              f"mixed={r['probe_mixed']:6.2f}", flush=True)
        with open(OUT_PARTIAL, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "note": "partial"}, f, ensure_ascii=False, indent=1)

    p_mixed_03 = [r for r in rows if r["alpha"] == 0.3][0]["probe_mixed"]
    delta_alpha = round(48.35 - p_mixed_03, 2)
    branch = "① 可分辨（Δ_α ≥ 3.45）" if delta_alpha >= 3.45 else "② 不可分辨（Δ_α < 3.45）"
    tag = "  ← 冒烟值，无意义" if args.smoke else ""
    print(f"\n[C1 主判据] Δ_α = 48.35 − {p_mixed_03:.2f} = {delta_alpha:+.2f} ⇒ 分支{branch}{tag}",
          flush=True)
    if not (18.10 <= p_mixed_03 <= 48.35):
        print(f"[C3 异常] p_mixed={p_mixed_03:.2f} 落在 [18.10, 48.35] 之外，先查管线再报", flush=True)

    if args.smoke:
        print("[smoke] 冒烟模式：不写正式产物")
        return

    m_end = mem_check("结束", hard=False)
    meta = {
        "script": "thirdarm_probe.py", "argv": sys.argv, "device": "cpu",
        "ckpt": CKPT, "n_train": n_train, "n_test": n_test,
        "probe_kwargs": PROBE_KW, "mix_seed": MIX_SEED, "alpha_max": ALPHA_MAX,
        "alpha_sampling": "per-sample iid U[-0.3,0.3], shared across layers within a sample",
        "feature": "distorted fc input (capture hook registered AFTER distortion hooks)",
        "self_test_S1_max_abs_diff": s1, "fidelity_same_cond_0.3": round(p_same, 2),
        "fidelity_ref": 48.35, "fidelity_tol": 0.30,
        "delta_alpha": delta_alpha, "branch": branch,
        "commit_free_mb_start": round(m0, 1), "commit_free_mb_end": round(m_end, 1),
        "torch": torch.__version__, "numpy": np.__version__,
        "git_commit": subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    if os.path.exists(OUT_PARTIAL):
        os.remove(OUT_PARTIAL)
    print(f"[done] 写 {OUT_CSV} / {OUT_META}", flush=True)


if __name__ == "__main__":
    main()
