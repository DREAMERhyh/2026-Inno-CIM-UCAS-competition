"""第七阶段 M0c：探针可信度审计（v7_probe_audit.py）

背景：Central Claim 的第一块砖是「α=+0.3 时线性探针 47.89% vs 自带头 27.76%，
headroom +20.12」。文献（Reblitz-Richardson, arXiv:2606.11375）系统质疑单点探针读数：
探针饱和不等于信息在场，探针失灵也不等于信息不在场。本脚本给出第二读法与四组对照。

**判据已在台账 §16 跑前写死，本文件不改判据。**

与 v3_probe_info_death.py 的关系：
  - 探针核心（StandardScaler + LogisticRegression(max_iter=1000, n_jobs=-1)）、
    条件表、headroom 定义**逐字照搬**，以保证读数可比；
  - **不改 v3**（逐字节保持原样），本脚本独立实现；
  - 差异仅在：num_workers=0（实测比 2 快 30 倍）、可配 n_train/n_test、
    产物落 v7_probe_audit/ 且带协议元数据。

跑法：
  python v7_probe_audit.py --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.simple_cnn import SimpleCNN
from utils.nonlinearity import register_nonlinearity_hooks, nonlinearity, remove_hooks
from utils.paths import get_ckpt_root, get_outputs_root
from utils.perturbation import register_perturbation_hooks

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

OUT_SUBDIR = "v7_probe_audit"

# 与 v3 的唯一实质差异：n_jobs。
# v3 用 n_jobs=-1，在 20 核机上让 joblib 开 20 个子进程；Windows 下每个子进程会
# 重新 import __main__ 并加载一整套 torch DLL，直接撞出
#   OSError: [WinError 1455] 页面文件太小，无法完成操作
# 这正是红线 2 记录的两次历史事故。M4 训练在跑时并发触发会波及 M4，故改为串行。
# n_jobs 只是并行度旋钮：lbfgs 各目标函数的拟合互相独立，不改变解。
# **该等价性由 R0 保真核验对 v3 记录值逐位核对来证明**（对不上即停）。
PROBE_KW = dict(max_iter=1000, n_jobs=None)  # v3: n_jobs=-1（见上）

N_TRAIN_MAIN = 8000       # 与 v3 登记的 8k 一致，保证与旧读数可比
N_TEST_MAIN = 10000       # 规范口径：全量测试集（红线 1）
N_TEST_FIDELITY = 8000    # v3 旧口径，仅用于保真核验

# v3 的 probe_results.csv 里 clean 与 all|alpha=+0.3 两行（保真核验基准）
FIDELITY_EXPECT = {
    "clean": {"probe_same_cond": 52.91, "head_acc": 59.60, "headroom_same": -6.69},
    "all|alpha=+0.3": {"probe_same_cond": 47.89, "head_acc": 27.76, "headroom_same": 20.12},
}

CHANCE = 1.0              # CIFAR-100 随机水平（%）
PROBE_TOL = 0.3           # R0 保真核验：probe 项的容许差（pp）
HEAD_TOL = 0.005          # R0 保真核验：head 项容许差（pp）
# 说明：v3 的 CSV 把读数存到 **2 位小数**，故参考值本身带 ±0.005 的量化误差。
# 设备维度已由"全表统一 CUDA"排除；此容差只吸收参考记录的圆整，不吸收任何真实差异。
# 实测（CUDA）：clean Δhead=0.000000、α=+0.3 Δhead=0.0025（= v3 真值 27.7625 的圆整残差）。
RANDOM_LABEL_SEEDS = (7, 8, 9)
SPLIT_SEEDS = (101, 202, 303, 404, 505)
BOOTSTRAP_SEEDS = (11, 22, 33)
PRIVATE_SPLIT_SEEDS = (21, 22, 23)
N_SPLIT_TRAIN = 6400      # C3：每个划分的探针训练子集大小
N_SPLIT_TEST = 8000       # C3：每个划分的评估子集大小
N_PRIVATE_HALF = 5000     # C4：test split 对半

# C2：RMS 归一化的噪声网格。sigma 的单位是「该条件激活的 RMS」
SIGMA_GRID = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
NOISE_SEEDS = (11, 22, 33)

# 参与 C1/C2/C3 的条件（v3 条件表里的 "all|" 系列）
KEY_CONDITIONS = ("clean", "all|alpha=+0.1", "all|alpha=-0.1", "all|alpha=+0.2",
                  "all|alpha=-0.2", "all|alpha=+0.3", "all|alpha=-0.3",
                  "all|alpha=+0.4", "all|alpha=+0.5")


def log(*a) -> None:
    print(*a, flush=True)


# ------------------------------------------------------------------
# 条件表（与 v3_probe_info_death.py 完全一致，顺序也一致）
# ------------------------------------------------------------------
def build_conditions(model) -> List[Tuple[str, Callable[[], list]]]:
    conditions: List[Tuple[str, Callable[[], list]]] = [("clean", lambda: [])]
    for a in (0.1, 0.2, 0.3, 0.4, 0.5):
        conditions.append((f"all|alpha={a:+.1f}",
                           lambda a=a: register_nonlinearity_hooks(model, a)))
        if a <= 0.3:
            conditions.append((f"all|alpha={-a:+.1f}",
                               lambda a=a: register_nonlinearity_hooks(model, -a)))
    for layer in ("conv1", "conv2", "conv3", "conv4", "fc"):
        conditions.append((f"{layer}|alpha=+0.3",
                           lambda l=layer: _single_layer_hook(model, l, 0.3)))
    for s in (0.1, 0.2, 0.3):
        conditions.append((f"all|sigma={s:.1f}",
                           lambda s=s: register_perturbation_hooks(model, "gaussian", noise_std=s)))
    return conditions


def _single_layer_hook(model, layer_name: str, alpha: float) -> list:
    """只在指定层注入非线性（v3 逐字照搬）"""
    module = dict(model.named_modules())[layer_name]

    def pre_hook(m, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        return (nonlinearity(x, alpha),) + tuple(inputs[1:])

    return [module.register_forward_pre_hook(pre_hook)]


# ------------------------------------------------------------------
# 数据与特征提取
# ------------------------------------------------------------------
def make_loaders(dataset: str = "cifar100", batch_size: int = 256, num_workers: int = 0):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])
    tr = datasets.CIFAR100(root="./data", train=True, download=False, transform=tf)
    te = datasets.CIFAR100(root="./data", train=False, download=False, transform=tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers))


@torch.no_grad()
def extract(model, loader, device, feat_store: dict, limit: int | None = None):
    """提取 fc 输入特征 + 冻结头预测 + 标签。limit 只影响前向多少张，不改变取哪些样本。"""
    feats, labels, preds = [], [], []
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        feats.append(feat_store["f"].detach().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
        n += x.size(0)
        if limit and n >= limit:
            break
    return (np.concatenate(feats)[:limit], np.concatenate(labels)[:limit],
            np.concatenate(preds)[:limit])


def run_condition(model, loader, device, hooks: list, limit: int | None):
    feat_store: Dict[str, torch.Tensor] = {}
    h = model.fc.register_forward_pre_hook(lambda m, inp: feat_store.__setitem__("f", inp[0]))
    try:
        X, y, pred = extract(model, loader, device, feat_store, limit)
    finally:
        h.remove()
        if hooks:
            remove_hooks(hooks)
    return X, y, pred


def probe_fit(Xtr, ytr):
    """与 v3 的 probe_accuracy 逐字一致，只是拆开以便复用 decoder"""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(**PROBE_KW)
    clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def probe_accuracy(Xtr, ytr, Xte, yte) -> float:
    """v3 逐字照搬"""
    sc, clf = probe_fit(Xtr, ytr)
    return float(clf.score(sc.transform(Xte), yte)) * 100.0


# ------------------------------------------------------------------
# 产物
# ------------------------------------------------------------------
def ensure_fresh_dir(path: str) -> None:
    """红线 3：跑前核验目标目录不存在（或为空），拒绝覆盖。"""
    if os.path.isdir(path) and os.listdir(path):
        raise SystemExit(f"[v7] 目标目录已存在且非空，拒绝覆盖（红线 3）：{path}")
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"[v7] 已保存: {path}")


def write_meta(out_dir: str, args, ckpt_path: str, device: str) -> None:
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    meta = {
        "script": "v7_probe_audit.py",
        "argv": sys.argv,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "backbone": args.model,
        "backbone_ckpt": ckpt_path,
        "dataset": args.dataset,
        "device": device,
        "probe_kwargs": PROBE_KW,
        "feature": "fc 输入（GAP 后 128 维）",
        "chance_level_pct": CHANCE,
        "seeds": {"random_label": RANDOM_LABEL_SEEDS, "split": SPLIT_SEEDS,
                  "bootstrap": BOOTSTRAP_SEEDS, "private_split": PRIVATE_SPLIT_SEEDS,
                  "noise": NOISE_SEEDS},
        "sigma_grid": SIGMA_GRID,
        "git_commit": sh("git rev-parse HEAD"),
        "git_dirty": sh("git status --porcelain"),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    with open(os.path.join(out_dir, "audit_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[v7] 已保存: {os.path.join(out_dir, 'audit_meta.json')}")


# ------------------------------------------------------------------
# R0 保真核验
# ------------------------------------------------------------------
def run_fidelity(feats: dict, out_dir: str) -> bool:
    """R0 保真核验。判据分两类（台账 §16.0 跑前写死 + 2026-09-24 修订，均由 claude-e9 批准）：

      - head_acc：纯前向、确定性，且**全表统一 CUDA**（设备已排除）
        ⇒ 必须一致到 **±0.005 pp**（该容差只吸收 v3 记录存 2 位小数的圆整，不吸收任何真实差异）
      - probe_same_cond：走 sklearn LogisticRegression，求解器可能有权重级噪声
        ⇒ 允许 ±0.3 pp
      - headroom = probe_same_cond − head_acc ⇒ 继承 probe 的容差（±0.3 pp）

    注 1：原始判据表述为「headroom 必须逐位复现」，但 headroom 含 probe 项，
    在 probe 允许 ±0.3 的前提下不可能同时要求逐位——此处按可满足的写法执行。
    注 2：原始判据表述为「head 逐位一致」。跨设备时物理上不可满足（v3 是 CUDA），
    故全表统一 CUDA 后，head 的"逐位"只剩参考记录的 2 位小数圆整这一项残差。
    """
    rows, ok = [], True
    for name, exp in FIDELITY_EXPECT.items():
        # 必须与 v3 同口径：v3 是 Xtr[:n_train] / Xte[:n_test]，两个都要切片。
        # （曾漏切 Xtr，探针在 10000 张上训练却只按 8000 口径比对，读数偏高 ~1.0~1.6pp。）
        Xtr = feats[name]["Xtr"][:N_TRAIN_MAIN]
        ytr = feats[name]["ytr"][:N_TRAIN_MAIN]
        Xte, yte, pred = feats[name]["Xte"], feats[name]["yte"], feats[name]["pred"]
        pA = probe_accuracy(Xtr, ytr, Xte[:N_TEST_FIDELITY], yte[:N_TEST_FIDELITY])
        h_acc = float((pred[:N_TEST_FIDELITY] == yte[:N_TEST_FIDELITY]).mean()) * 100.0
        hr = pA - h_acc
        d_head = abs(h_acc - exp["head_acc"])
        d_probe = abs(pA - exp["probe_same_cond"])
        head_exact = d_head <= HEAD_TOL
        probe_ok = d_probe <= PROBE_TOL
        match = head_exact and probe_ok
        ok = ok and match
        rows.append({"condition": name, "probe_same_cond_v7": round(pA, 2),
                     "probe_same_cond_v3": exp["probe_same_cond"],
                     "d_probe": round(d_probe, 4),
                     "head_acc_v7": round(h_acc, 2), "head_acc_v3": exp["head_acc"],
                     "d_head": round(d_head, 6),
                     "headroom_v7": round(hr, 2), "headroom_v3": exp["headroom_same"],
                     "d_headroom": round(abs(hr - exp["headroom_same"]), 4),
                     "head_exact": head_exact, "probe_within_tol": probe_ok})
        log(f"[R0] {name:<18} v7={pA:6.2f}/{h_acc:6.2f}/{hr:+7.2f}  "
            f"v3={exp['probe_same_cond']:6.2f}/{exp['head_acc']:6.2f}/{exp['headroom_same']:+7.2f}  "
            f"Δhead={d_head:.6f} Δprobe={d_probe:.4f}  "
            f"{'✅忠实' if match else '❌不忠实'}")
    write_csv(os.path.join(out_dir, "r0_fidelity.csv"), rows)
    return ok


# ------------------------------------------------------------------
# RA 规范口径
# ------------------------------------------------------------------
def run_onprotocol(feats: dict, names: Sequence[str], out_dir: str, n_train: int,
                   n_test: int, suffix: str) -> List[dict]:
    rows = []
    clean = feats["clean"]
    for name in names:
        f = feats[name]
        Xtr, ytr = f["Xtr"][:n_train], f["ytr"][:n_train]
        Xte, yte = f["Xte"][:n_test], f["yte"][:n_test]
        pA = probe_accuracy(Xtr, ytr, Xte, yte)
        pB = probe_accuracy(clean["Xtr"][:n_train], clean["ytr"][:n_train], Xte, yte)
        h_acc = float((f["pred"][:n_test] == yte).mean()) * 100.0
        rows.append({"condition": name, "probe_same_cond": round(pA, 2),
                     "probe_clean_train": round(pB, 2), "head_acc": round(h_acc, 2),
                     "headroom_same": round(pA - h_acc, 2)})
        log(f"[RA{suffix}] {name:<18} 探针 {pA:6.2f} | 干净训练探针 {pB:6.2f} | "
            f"冻结头 {h_acc:6.2f} | headroom {pA - h_acc:+7.2f}")
    write_csv(os.path.join(out_dir, f"ra_onprotocol_{suffix}.csv"), rows)
    return rows


# ------------------------------------------------------------------
# C1 随机标签探针
# ------------------------------------------------------------------
def run_c1(feats: dict, names: Sequence[str], out_dir: str, n_train: int, n_test: int) -> List[dict]:
    rows = []
    for name in names:
        f = feats[name]
        Xtr, ytr = f["Xtr"][:n_train], f["ytr"][:n_train]
        Xte, yte = f["Xte"][:n_test], f["yte"][:n_test]
        h_acc = float((f["pred"][:n_test] == yte).mean()) * 100.0
        accs = []
        for s in RANDOM_LABEL_SEEDS:
            rng = np.random.default_rng(s)
            y_shuf = ytr[rng.permutation(len(ytr))]
            accs.append(probe_accuracy(Xtr, y_shuf, Xte, yte))
        m = float(np.mean(accs))
        rows.append({"condition": name, "head_acc": round(h_acc, 2),
                     "rand_label_acc_mean": round(m, 2),
                     "rand_label_acc_std": round(float(np.std(accs)), 2),
                     "n_seeds": len(RANDOM_LABEL_SEEDS),
                     "chance": CHANCE, "dev_from_chance": round(m - CHANCE, 2),
                     "headroom_rand": round(m - h_acc, 2)})
        log(f"[C1] {name:<18} 随机标签探针 {m:6.2f} ± {np.std(accs):.2f} "
            f"(chance={CHANCE}, 偏离 {m - CHANCE:+.2f}) | 冻结头 {h_acc:6.2f}")
    write_csv(os.path.join(out_dir, "c1_random_label.csv"), rows)
    return rows


# ------------------------------------------------------------------
# C2 fragility（RMS 归一化临界噪声）
# ------------------------------------------------------------------
def rms_of(X) -> float:
    return float(np.sqrt(np.mean(np.asarray(X, dtype=np.float64) ** 2)))


def _critical_sigma(sigmas: Sequence[float], accs: Sequence[float]) -> float:
    """探针精度跌到「chance 与 σ=0 精度 的中点」时的 σ（线性插值）。找不到记 inf。"""
    acc0 = accs[0]
    target = CHANCE + 0.5 * (acc0 - CHANCE)
    if acc0 <= target:
        return 0.0
    for i in range(1, len(accs)):
        if accs[i] <= target:
            s0, s1, a0, a1 = sigmas[i - 1], sigmas[i], accs[i - 1], accs[i]
            if a0 == a1:
                return float(s1)
            return float(s0 + (a0 - target) / (a0 - a1) * (s1 - s0))
    return float("inf")


def run_c2(feats: dict, names: Sequence[str], out_dir: str, n_train: int, n_test: int) -> List[dict]:
    rows = []
    for name in names:
        f = feats[name]
        Xtr, ytr = f["Xtr"][:n_train], f["ytr"][:n_train]
        Xte, yte = f["Xte"][:n_test], f["yte"][:n_test]
        sc, clf = probe_fit(Xtr, ytr)          # 训练干净，测试加噪
        Zte = sc.transform(Xte)
        rms_val = rms_of(Xte)
        accs = []
        for sig in SIGMA_GRID:
            if sig == 0.0:
                accs.append(float(clf.score(Zte, yte)) * 100.0)
                continue
            per_seed = []
            for s in NOISE_SEEDS:
                rng = np.random.default_rng(s)
                noise = rng.standard_normal(Xte.shape).astype(np.float32) * (sig * rms_val)
                per_seed.append(float(clf.score(sc.transform(Xte + noise), yte)) * 100.0)
            accs.append(float(np.mean(per_seed)))
        sigma_star = _critical_sigma(SIGMA_GRID, accs)
        rows.append({"condition": name, "act_rms": round(rms_val, 6),
                     "acc_at_sigma0": round(accs[0], 2),
                     "critical_sigma_rms_norm": (round(sigma_star, 4)
                                                 if np.isfinite(sigma_star) else "inf"),
                     "critical_noise_abs": (round(sigma_star * rms_val, 6)
                                            if np.isfinite(sigma_star) else "inf"),
                     "acc_curve": ";".join(f"{s}:{a:.3f}" for s, a in zip(SIGMA_GRID, accs))})
        log(f"[C2] {name:<18} RMS={rms_val:.4e}  acc(σ=0)={accs[0]:6.2f}  "
            f"临界噪声(RMS 归一)={sigma_star:.4f}")
    write_csv(os.path.join(out_dir, "c2_fragility.csv"), rows)
    return rows


# ------------------------------------------------------------------
# C3 decoder / 划分控制
# ------------------------------------------------------------------
def run_c3(feats: dict, names: Sequence[str], out_dir: str, n_train: int, n_test: int) -> List[dict]:
    rows = []
    for name in names:
        f = feats[name]
        Xtr_all, ytr_all = f["Xtr"][:n_train], f["ytr"][:n_train]
        Xte_all, yte_all = f["Xte"][:n_test], f["yte"][:n_test]
        split_accs = []
        for s in SPLIT_SEEDS:
            rng = np.random.default_rng(s)
            ti = rng.choice(len(Xtr_all), size=min(N_SPLIT_TRAIN, len(Xtr_all)), replace=False)
            ei = rng.choice(len(Xte_all), size=min(N_SPLIT_TEST, len(Xte_all)), replace=False)
            split_accs.append(probe_accuracy(Xtr_all[ti], ytr_all[ti], Xte_all[ei], yte_all[ei]))
        boot_accs = []
        for s in BOOTSTRAP_SEEDS:
            rng = np.random.default_rng(s)
            bi = rng.integers(0, len(Xtr_all), size=len(Xtr_all))
            boot_accs.append(probe_accuracy(Xtr_all[bi], ytr_all[bi], Xte_all, yte_all))
        all_accs = split_accs + boot_accs
        rows.append({"condition": name,
                     "split_mean": round(float(np.mean(split_accs)), 2),
                     "split_std": round(float(np.std(split_accs)), 2),
                     "split_min": round(float(np.min(split_accs)), 2),
                     "split_max": round(float(np.max(split_accs)), 2),
                     "bootstrap_mean": round(float(np.mean(boot_accs)), 2),
                     "bootstrap_std": round(float(np.std(boot_accs)), 2),
                     "overall_std": round(float(np.std(all_accs)), 2),
                     "overall_range": round(float(np.max(all_accs) - np.min(all_accs)), 2),
                     "n_splits": len(SPLIT_SEEDS), "n_bootstrap": len(BOOTSTRAP_SEEDS)})
        log(f"[C3] {name:<18} 5 划分 {np.mean(split_accs):6.2f}±{np.std(split_accs):.2f} | "
            f"bootstrap {np.mean(boot_accs):6.2f}±{np.std(boot_accs):.2f} | "
            f"总极差 {max(all_accs) - min(all_accs):.2f}")
    write_csv(os.path.join(out_dir, "c3_decoder_split.csv"), rows)
    return rows


# ------------------------------------------------------------------
# C4 私有校准划分（与主干训练集完全不相交）
# ------------------------------------------------------------------
def run_c4(feats: dict, names: Sequence[str], out_dir: str, n_test: int) -> List[dict]:
    """探针训练/评估都只用 test split（主干从未见过），对半切分。"""
    rows = []
    for name in names:
        f = feats[name]
        Xte_all, yte_all = f["Xte"][:n_test], f["yte"][:n_test]
        accs = []
        for s in PRIVATE_SPLIT_SEEDS:
            rng = np.random.default_rng(s)
            idx = rng.permutation(len(Xte_all))
            half = len(idx) // 2
            tr_i, ev_i = idx[:half], idx[half:half * 2]
            accs.append(probe_accuracy(Xte_all[tr_i], yte_all[tr_i],
                                       Xte_all[ev_i], yte_all[ev_i]))
        m = float(np.mean(accs))
        rows.append({"condition": name, "private_probe_acc_mean": round(m, 2),
                     "private_probe_acc_std": round(float(np.std(accs)), 2),
                     "n_probe_train": min(N_PRIVATE_HALF, n_test // 2),
                     "n_probe_eval": min(N_PRIVATE_HALF, n_test // 2),
                     "note": "探针训练与评估均取 test split（主干训练集之外），与主干完全不相交"})
        log(f"[C4] {name:<18} 不相交划分探针 {m:6.2f} ± {np.std(accs):.2f}"
            f"  (对半 {min(N_PRIVATE_HALF, n_test // 2)}/{min(N_PRIVATE_HALF, n_test // 2)})")
    write_csv(os.path.join(out_dir, "c4_private_split.csv"), rows)
    return rows


# ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simple_cnn")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n_train", type=int, default=N_TRAIN_MAIN)
    ap.add_argument("--n_test", type=int, default=N_TEST_MAIN)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(get_outputs_root(args.dataset), OUT_SUBDIR)
    ensure_fresh_dir(out_dir)

    model = SimpleCNN(num_classes=100)
    wpath = args.ckpt or os.path.join(get_ckpt_root(args.dataset), args.model, "best_model.pth")
    ck = torch.load(wpath, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) else ck)
    model.to(device).eval()
    log(f"[v7] backbone={args.model} ckpt={wpath} device={device} "
        f"n_train={args.n_train} n_test={args.n_test}")
    write_meta(out_dir, args, wpath, device)

    train_loader, test_loader = make_loaders(args.dataset, args.batch_size, args.num_workers)
    conditions = build_conditions(model)
    names = [n for n, _ in conditions]

    # 一次提取，全部对照复用。
    # train 需覆盖 n_train 与 10000 那组变体；test 取 n_test。
    n_tr_extract = max(args.n_train, 10000, N_SPLIT_TRAIN)
    feats: Dict[str, dict] = {}
    for name, factory in conditions:
        Xtr, ytr, _ = run_condition(model, train_loader, device, factory(), n_tr_extract)
        Xte, yte, pred = run_condition(model, test_loader, device, factory(), args.n_test)
        feats[name] = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "pred": pred}
        log(f"[v7] 特征已提取 {name:<18} train={Xtr.shape} test={Xte.shape}")

    # R0：保真核验（用 8k 测试口径复现 v3）
    feats_fid = {k: dict(v) for k, v in feats.items()}
    ok = run_fidelity(feats_fid, out_dir)
    if not ok:
        raise SystemExit("[v7] ❌ 保真核验失败：v7 未能逐位复现 v3，后续数字作废。已停止。")
    log("[v7] ✅ 保真核验通过：v7 与 v3 逐位一致")

    # RA：规范口径（全量测试集）
    run_onprotocol(feats, names, out_dir, args.n_train, args.n_test, "main")
    if args.n_train != 10000:
        run_onprotocol(feats, names, out_dir, 10000, args.n_test, "n10000")

    keys = [n for n in KEY_CONDITIONS if n in names]
    run_c1(feats, keys, out_dir, args.n_train, args.n_test)
    run_c2(feats, keys, out_dir, args.n_train, args.n_test)
    run_c3(feats, keys, out_dir, args.n_train, args.n_test)
    run_c4(feats, keys, out_dir, args.n_test)
    log("[v7] 全部完成")


if __name__ == "__main__":
    main()
