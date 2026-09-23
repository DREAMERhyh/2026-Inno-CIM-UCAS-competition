"""
论文配图生成器（零 GPU，只读 CSV，不修改任何既有产物）。

纪律：
  · 所有数字从 CSV 现读，**脚本内不硬编码任何精度数字**
  · 每张图末尾打印关键读数，便于与 论文v3.md / experiments_ledger.md 交叉核对
  · 配色用 Wong 8 色（色盲友好）；中文用 SimHei；输出 PDF（矢量）+ PNG（300dpi）

用法：PYTHONIOENCODING=utf-8 <pytorch_env>/python paper/figures/gen_figures.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))

# ---- 全局样式 ----
WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
POS, NEG, BASE = "#D55E00", "#0072B2", "#999999"   # 正侧/负侧/基线，全文固定
plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


def p(*a):
    print(*a)


def read_csv(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def alpha_curve(rel):
    """alpha_sensitivity.csv -> (alphas, accs)"""
    rows = read_csv(rel)
    if not rows:
        return None, None
    rows = sorted(rows, key=lambda r: float(r["alpha"]))
    return [float(r["alpha"]) for r in rows], [float(r["accuracy"]) for r in rows]


def readout_curve(rel, which):
    """readout_repair CSV -> (alphas, accs) for row `which` (a_original_fc / e_readout_NAT)"""
    rows = read_csv(rel)
    if not rows:
        return None, None
    row = next((r for r in rows if r["readout"] == which), None)
    if row is None:
        return None, None
    cols = [c for c in row if c != "readout"]
    cols = sorted(cols, key=float)
    return [float(c) for c in cols], [float(row[c]) for c in cols]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    p(f"  [saved] {name}.pdf / .png")


def mean7_from_curve(alphas, accs):
    """给定整数 7 点曲线，按台账口径算 mean7"""
    d = dict(zip([round(a, 4) for a in alphas], accs))
    pts = [d.get(a) for a in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)]
    return sum(pts) / 7 if all(v is not None for v in pts) else None


def seed_mean(curves):
    """多条 (alphas, accs) -> 逐点均值与总体标准差(ddof=0)"""
    base = curves[0][0]
    arr = np.array([c[1] for c in curves])
    return base, arr.mean(axis=0), arr.std(axis=0, ddof=0)


# =====================================================================
# F1 失真的方向性与容量依赖
# =====================================================================
def f01():
    p("\n[F1] 失真的方向性与容量依赖")
    series = [("SimpleCNN", "outputs_cifar100/task1_simplecnn/alpha_sensitivity.csv", WONG[6]),
              ("VGG-11", "outputs_cifar100/task1_vgg11/alpha_sensitivity.csv", WONG[5]),
              ("ResNet-18", "outputs_cifar100/task1_resnet18/alpha_sensitivity.csv", WONG[3])]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for name, rel, c in series:
        a, v = alpha_curve(rel)
        if a is None:
            p(f"  !! 缺 {rel}")
            continue
        ax.plot(a, v, "o-", color=c, lw=1.6, ms=4, label=name)
        p(f"  {name:10s} clean={v[a.index(0.0)]:.2f}  +0.3={v[a.index(0.3)]:.2f}")
    ax.set_xlabel(r"失真强度 $\alpha$")
    ax.set_ylabel("测试精度 (%)")
    ax.axvline(0, color=BASE, lw=0.8, ls=":")
    ax.legend(frameon=False)
    save(fig, "F01_direction")


# =====================================================================
# F2 灾难尾的瓶颈在读出
# =====================================================================
def f02():
    p("\n[F2] 探针 headroom：信息仍在线性可达范围内")
    rows = read_csv("outputs_cifar100/v3_probe_info_death/probe_results.csv")
    if not rows:
        p("  !! 缺 probe_results.csv")
        return
    idx = {-0.3: None, -0.2: None, -0.1: None, 0.0: None,
           0.1: None, 0.2: None, 0.3: None}
    for r in rows:
        c = r["condition"]
        if c == "clean":
            idx[0.0] = r
        elif c.startswith("all|alpha="):
            key = round(float(c.split("alpha=")[1]), 4)
            if key in idx:
                idx[key] = r
    xs = sorted(idx)
    probe = [float(idx[k]["probe_same_cond"]) for k in xs]
    head = [float(idx[k]["head_acc"]) for k in xs]
    hr = [float(idx[k]["headroom_same"]) for k in xs]

    x = np.arange(len(xs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.bar(x - w / 2, probe, w, label="线性探针（冻结主干特征）", color=WONG[5])
    ax.bar(x + w / 2, head, w, label="模型自带分类头", color=WONG[0], alpha=0.65)
    for xi, h, pv, hv in zip(x, hr, probe, head):
        if h > 0:
            ax.annotate(f"+{h:.1f}", (xi, max(pv, hv) + 1.2), ha="center",
                        fontsize=8, color=WONG[6], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:+.1f}" if k != 0.0 else "0.0" for k in xs])
    ax.set_xlabel(r"失真强度 $\alpha$")
    ax.set_ylabel("测试精度 (%)")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "F02_probe_headroom")
    i3 = xs.index(0.3)
    p(f"  α=+0.3: 探针 {probe[i3]:.2f} vs 头 {head[i3]:.2f} → headroom {hr[i3]:+.2f}")
    sig = [r for r in rows if r["condition"] == "all|sigma=0.3"]
    if sig:
        p(f"  对照(高斯 σ=0.3): 探针 {float(sig[0]['probe_same_cond']):.2f} vs 头"
          f" {float(sig[0]['head_acc']):.2f} → headroom {float(sig[0]['headroom_same']):+.2f}")


# =====================================================================
# F3 readout-NAT 跨架构（3 seed 均值）
# =====================================================================
def f03():
    p("\n[F3] readout-NAT 跨架构（3 seed 均值）")
    groups = {
        "SimpleCNN": [
            "outputs_cifar100/v3_readout_repair/readout_repair_c100_clean_ep24_alpha+0.30_n20000.csv",
            "outputs_cifar100/v3_readout_repair/readout_repair_c100_plain_s43_ep24_alpha+0.30_n20000_s43.csv",
            "outputs_cifar100/v3_readout_repair/readout_repair_c100_plain_s44_ep24_alpha+0.30_n20000_s44.csv"],
        "VGG-11": [f"outputs_cifar100/v3_readout_repair/readout_repair_c100_exp2_vgg11_s{s}_ep24_alpha+0.30_n20000{sfx}.csv"
                   for s, sfx in ((42, ""), (43, "_s43"), (44, "_s44"))],
        "ResNet-18": [f"outputs_cifar100/v3_readout_repair/readout_repair_c100_exp3_resnet18_s{s}_ep24_alpha+0.30_n20000{sfx}.csv"
                      for s, sfx in ((42, ""), (43, "_s43"), (44, "_s44"))],
    }
    colors = [WONG[6], WONG[5], WONG[3]]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for (name, files), c in zip(groups.items(), colors):
        orig = [readout_curve(f, "a_original_fc") for f in files]
        nat = [readout_curve(f, "e_readout_NAT") for f in files]
        orig = [c_ for c_ in orig if c_[0] is not None]
        nat = [c_ for c_ in nat if c_[0] is not None]
        if not orig or not nat:
            p(f"  !! {name} 缺文件")
            continue
        a0, m0, s0 = seed_mean(orig)
        a1, m1, s1 = seed_mean(nat)
        ax.plot(a0, m0, "o--", color=BASE, lw=1.2, ms=3.5, alpha=0.9)
        ax.plot(a1, m1, "o-", color=c, lw=1.8, ms=4, label=f"{name}·readout-NAT")
        i3 = a1.index(0.3)
        ax.annotate(f"{m1[i3]:.1f}±{s1[i3]:.2f}", (0.3, m1[i3]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7.5, color=c)
        p(f"  {name:10s} 原头@+0.3={m0[a0.index(0.3)]:.2f}  NAT@+0.3={m1[i3]:.2f}±{s1[i3]:.2f}"
          f"  NAT clean={m1[a1.index(0.0)]:.2f}±{s1[a1.index(0.0)]:.2f}")
    ax.plot([], [], "o--", color=BASE, lw=1.2, ms=3.5, label="原始分类头")
    ax.set_xlabel(r"失真强度 $\alpha$")
    ax.set_ylabel("测试精度 (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    save(fig, "F03_readout_nat")


# =====================================================================
# F4 训练端四条路径全部无效
# =====================================================================
def f04():
    p("\n[F4] 训练端的补救路径 vs 均匀采样基准")
    base_a, base_v = alpha_curve("outputs_cifar100/v3_methods/budget_uniform/alpha_sensitivity.csv")
    if base_a is None:
        p("  !! 缺基准")
        return
    base03 = base_v[base_a.index(0.3)]
    cands = [("预算·front", "budget_front"), ("预算·rear", "budget_rear"),
             ("预算·sens", "budget_sens"),
             ("α-对抗(PGD)", "adv_uniform"), ("课程式 α", "curriculum_uniform"),
             ("范围外推 ±0.5", "range_0.50")]
    labels, deltas = [], []
    for lab, d in cands:
        a, v = alpha_curve(f"outputs_cifar100/v3_methods/{d}/alpha_sensitivity.csv")
        if a is None:
            p(f"  !! 缺 {d}")
            continue
        labels.append(lab)
        deltas.append(v[a.index(0.3)] - base03)
        p(f"  {lab:14s} +0.3={v[a.index(0.3)]:.2f}  Δ={v[a.index(0.3)] - base03:+.2f}")
    p(f"  基准（uniform） +0.3={base03:.2f}")

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    y = np.arange(len(labels))
    cols = [WONG[5] if d < 0 else WONG[3] for d in deltas]
    ax.barh(y, deltas, color=cols, height=0.6)
    ax.axvline(0, color=BASE, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\alpha=+0.3$ 相对均匀采样基准的差值 (pp)")
    for yi, d in zip(y, deltas):
        ax.annotate(f"{d:+.2f}", (d, yi), xytext=(4 if d >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if d >= 0 else "right", fontsize=8)
    save(fig, "F04_training_paths")


# =====================================================================
# F5 有效秩的塌陷（归一化）
# =====================================================================
def f05():
    p("\n[F5] 有效秩 vs α（按各自 clean 归一化）")
    files = [("SimpleCNN (d=128)", "outputs_cifar100/v5_rank_collapse/rank_stats.csv", WONG[6]),
             ("VGG-11 (d=512)", "outputs_cifar100/v5_rank_collapse/rank_stats_exp2vgg11.csv", WONG[5]),
             ("ResNet-18 (d=512)", "outputs_cifar100/v5_rank_collapse/rank_stats_exp3resnet18.csv", WONG[3])]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for name, rel, c in files:
        rows = read_csv(rel)
        if not rows:
            p(f"  !! 缺 {rel}")
            continue
        rows = sorted(rows, key=lambda r: float(r["alpha"]))
        a = np.array([float(r["alpha"]) for r in rows])
        rk = np.array([float(r["rank_eff"]) for r in rows])
        base = rk[a == 0.0][0]
        norm = rk / base * 100
        ax.plot(a, norm, "o-", color=c, lw=1.6, ms=4, label=f"{name}")
        i3 = int(np.where(a == 0.3)[0][0])
        p(f"  {name:18s} 秩@0={base:.3f}  秩@+0.3={rk[i3]:.3f}"
          f"  → 损失 {(1 - norm[i3] / 100) * 100:.1f}%")
    ax.axhline(100, color=BASE, lw=0.8, ls=":")
    ax.set_xlabel(r"失真强度 $\alpha$")
    ax.set_ylabel("有效秩（相对 $\\alpha=0$ 的百分比，%）")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "F05_rank_collapse")


# =====================================================================
# F6 秩损失 → 读出可恢复量
# =====================================================================
def f06():
    p("\n[F6] 秩损失 vs 读出恢复率")
    specs = [
        ("SimpleCNN", "outputs_cifar100/v5_rank_collapse/rank_stats.csv",
         ["outputs_cifar100/v3_readout_repair/readout_repair_c100_clean_ep24_alpha+0.30_n20000.csv",
          "outputs_cifar100/v3_readout_repair/readout_repair_c100_plain_s43_ep24_alpha+0.30_n20000_s43.csv",
          "outputs_cifar100/v3_readout_repair/readout_repair_c100_plain_s44_ep24_alpha+0.30_n20000_s44.csv"]),
        ("VGG-11", "outputs_cifar100/v5_rank_collapse/rank_stats_exp2vgg11.csv",
         [f"outputs_cifar100/v3_readout_repair/readout_repair_c100_exp2_vgg11_s{s}_ep24_alpha+0.30_n20000{x}.csv"
          for s, x in ((42, ""), (43, "_s43"), (44, "_s44"))]),
        ("ResNet-18", "outputs_cifar100/v5_rank_collapse/rank_stats_exp3resnet18.csv",
         [f"outputs_cifar100/v3_readout_repair/readout_repair_c100_exp3_resnet18_s{s}_ep24_alpha+0.30_n20000{x}.csv"
          for s, x in ((42, ""), (43, "_s43"), (44, "_s44"))]),
    ]
    pts = []
    for name, rank_rel, rd in specs:
        rows = read_csv(rank_rel)
        if not rows:
            p(f"  !! 缺 {rank_rel}")
            continue
        a = np.array([float(r["alpha"]) for r in rows])
        rk = np.array([float(r["rank_eff"]) for r in rows])
        loss = (1 - rk[a == 0.3][0] / rk[a == 0.0][0]) * 100
        orig = [readout_curve(f, "a_original_fc") for f in rd]
        nat = [readout_curve(f, "e_readout_NAT") for f in rd]
        orig = [c for c in orig if c[0] is not None]
        nat = [c for c in nat if c[0] is not None]
        if not orig or not nat:
            p(f"  !! {name} 缺读出")
            continue
        clean = np.mean([c[1][c[0].index(0.0)] for c in orig])
        rec = np.mean([c[1][c[0].index(0.3)] for c in nat]) / clean * 100
        pts.append((loss, rec, name))
        p(f"  {name:10s} 秩损失={loss:.1f}%  恢复率={rec:.1f}%")

    if len(pts) >= 2:
        fig, ax = plt.subplots(figsize=(4.8, 3.4))
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        ax.plot(xs, ys, "o-", color=WONG[5], lw=1.6, ms=7)
        for x_, y_, n_ in pts:
            ax.annotate(n_, (x_, y_), xytext=(6, 4), textcoords="offset points", fontsize=8)
        r = np.corrcoef(xs, ys)[0, 1]
        ax.set_xlabel("有效秩损失 @ $\\alpha=+0.3$ (%)")
        ax.set_ylabel("读出恢复率 @ $\\alpha=+0.3$ (%)")
        ax.set_title(f"逐点 Pearson $r = {r:.3f}$（三点，不足以定曲线形状）", fontsize=9)
        save(fig, "F06_rank_vs_recovery")


# =====================================================================
# F7 池化密度杠杆
# =====================================================================
def f07():
    p("\n[F7] 池化密度 2→5（零训练成本）")
    pairs = [("2×MaxPool（异协议锚点）", "outputs_cifar100/task1_simplecnn/alpha_sensitivity.csv", BASE, "--"),
             ("5×MaxPool", "outputs_cifar100/v3_methods/simple_cnn_mp_clean_uniform/alpha_sensitivity.csv", WONG[6], "-")]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for lab, rel, c, ls in pairs:
        a, v = alpha_curve(rel)
        if a is None:
            p(f"  !! 缺 {rel}")
            continue
        ax.plot(a, v, "o" + ls, color=c, lw=1.7, ms=4, label=lab)
        p(f"  {lab:22s} clean={v[a.index(0.0)]:.2f}  +0.3={v[a.index(0.3)]:.2f}")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(r"失真强度 $\alpha$")
    ax.set_ylabel("测试精度 (%)")
    save(fig, "F07_maxpool_density")


# =====================================================================
# F8 门控红利 vs 鲁棒头 clean 代价（七点）
# =====================================================================
def recipe_metrics(rel):
    """(代价, 红利) —— 先对每个 alpha 取 seed 均值，再算 mean7 / clean 列。
    口径与台账 §13.11 / §14.2 一致（各 3 个 head seed）。"""
    rows = read_csv(rel)
    if not rows:
        return None
    keys = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
    by = {}
    for r in rows:
        by.setdefault(round(float(r["alpha"]), 4), []).append(r)
    if not all(k in by for k in keys):
        return None

    def at(k, col):
        return float(np.mean([float(x[col]) for x in by[k]]))

    m7 = lambda col: float(np.mean([at(k, col) for k in keys]))          # noqa: E731
    return dict(cost=at(0.0, "acc_robust_head") - at(0.0, "acc_clean_head"),
                benefit=m7("acc_softgate") - m7("acc_robust_head"))


def f08():
    p("\n[F8] 门控红利 vs 鲁棒头 clean 代价")
    pts = [("C100/mp", "outputs_cifar100/v5_full_recipe/recipe_simple_cnn_mp.csv", WONG[6]),
           ("C10/mp s42", "outputs/v5_full_recipe/recipe_simple_cnn_mp.csv", WONG[5]),
           ("C10/mp s43", "outputs/v5_full_recipe/recipe_simple_cnn_mp_bs43.csv", WONG[5]),
           ("C10/mp s44", "outputs/v5_full_recipe/recipe_simple_cnn_mp_bs44.csv", WONG[5]),
           ("C10/plain", "outputs/v5_full_recipe/recipe_simple_cnn.csv", WONG[1]),
           ("VGG-11/Exp2", "outputs_cifar100/v5_full_recipe/recipe_robust_vgg11_exp2vgg11.csv", WONG[3]),
           ("ResNet-18/Exp3", "outputs_cifar100/v5_full_recipe/recipe_robust_resnet18_exp3resnet18.csv", WONG[7])]
    data = []
    for lab, rel, c in pts:
        m = recipe_metrics(rel)
        if m is None:
            p(f"  !! 缺 {rel}")
            continue
        data.append((m["cost"], m["benefit"], lab, c))
        p(f"  {lab:16s} 代价={m['cost']:+.2f}  红利={m['benefit']:+.2f}")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.axhline(0, color=BASE, lw=1.0, ls="--")
    ax.axvline(0, color=BASE, lw=0.6, ls=":")
    for x_, y_, lab, c in data:
        ax.scatter(x_, y_, s=60, color=c, zorder=3, edgecolor="white", linewidth=0.6)
        ax.annotate(lab, (x_, y_), xytext=(6, 4), textcoords="offset points", fontsize=7.5)
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    ax.set_xlabel("鲁棒头的 clean 代价 (pp)")
    ax.set_ylabel("门控红利 mean7 (pp)")
    ax.set_title(f"七个点，逐点 Pearson $r = {np.corrcoef(xs, ys)[0, 1]:.3f}$", fontsize=9)
    save(fig, "F08_gating_benefit")


# =====================================================================
# F9 交叉评估：评估路径无辜，差异在权重侧
# =====================================================================
def f09():
    p("\n[F9] 交叉评估（8 权重 × 2 路径）")
    rows = read_csv("outputs_cifar100/v6_p4e_crosspath/crosspath.csv")
    if not rows:
        p("  !! 缺 crosspath.csv")
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ys = np.arange(len(rows))
    for i, r in enumerate(rows):
        v3, t2 = float(r["v3pos"]), float(r["t2pos"])
        ax.annotate("", xy=(v3, i), xytext=(t2, i),
                    arrowprops=dict(arrowstyle="-", color=BASE, lw=1.0))
        ax.scatter(v3, i, s=45, marker="o", color=WONG[5], zorder=3,
                   label="v3 评估路径" if i == 0 else None)
        ax.scatter(t2, i, s=45, marker="s", color=WONG[6], zorder=3,
                   label="task2 评估路径" if i == 0 else None)
        p(f"  {r['weight']:22s} v3={v3:.2f}  task2={t2:.2f}  差={v3 - t2:+.4f}")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["weight"] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\alpha=+0.3$ 测试精度 (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "F09_crosspath")


# =====================================================================
# F10 Dithering：bits 扫描
# =====================================================================
def f10():
    p("\n[F10] 4bit 量化在 VGG-11 上反转成增益")
    rows = read_csv("outputs_cifar100/extension5_dithering_vgg11/group_a_bits_scan.csv")
    if not rows:
        p("  !! 缺 group_a_bits_scan.csv")
        return
    b = [int(r["num_bits"]) for r in rows]
    acc = [float(r["accuracy"]) for r in rows]
    base = acc[b.index(8)]
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.plot(b, acc, "o-", color=WONG[5], lw=1.8, ms=5)
    ax.axhline(base, color=BASE, lw=1.0, ls="--", label=f"8bit 基线 {base:.2f}%")
    peak = max(range(len(acc)), key=lambda i: acc[i])
    ax.annotate(f"4bit: {acc[b.index(4)]:.2f}%\n(+{acc[b.index(4)] - base:.2f} pp)",
                (4, acc[b.index(4)]), xytext=(8, -26), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", color=WONG[6], lw=1.0), color=WONG[6])
    ax.set_xlabel("ADC 量化位宽 (bit)")
    ax.set_ylabel("测试精度 (%)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "F10_dithering")
    p(f"  8bit={base:.2f}  4bit={acc[b.index(4)]:.2f}  Δ={acc[b.index(4)] - base:+.2f}"
      f"  峰值 bit={b[peak]}")


# =====================================================================
# F11 外推边界：NAT 的鲁棒性是分布内属性
# =====================================================================
def f11():
    # 注意：这条结论来自第二阶段（Extension 4），数据集是 CIFAR-10
    # （clean α=0 为 84.90），不是 CIFAR-100。台账表 1-2 的 +0.77 / −10.77 出自这里。
    p("\n[F11] NAT 外推（CIFAR-10）：正侧归零后反转")
    rows = read_csv("outputs/extension4_alpha_wide/alpha_wide_scan_summary.csv")
    if not rows:
        p("  !! 缺 outputs/extension4_alpha_wide/alpha_wide_scan_summary.csv")
        return
    def curve(wt):
        sub = [r for r in rows if r["model"] == "simple_cnn" and r["weight_type"] == wt]
        sub = sorted(sub, key=lambda r: float(r["alpha"]))
        return {round(float(r["alpha"]), 4): float(r["accuracy"]) for r in sub}
    clean = curve("clean")
    if not clean:
        p("  !! 缺 simple_cnn/clean")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for wt, lab, c in [("nat_scratch", "NAT-scratch", WONG[6]),
                       ("nat_finetune", "NAT-finetune", WONG[5])]:
        cur = curve(wt)
        xs = sorted(set(cur) & set(clean))
        ax.plot(xs, [cur[x] - clean[x] for x in xs], "o-", color=c, lw=1.6, ms=4, label=lab)
        for x in xs:
            if abs(x) >= 0.4 and x > 0:
                p(f"  {lab:14s} α={x:+.2f}  Δ={cur[x] - clean[x]:+.2f}")
    ax.axhline(0, color=BASE, lw=1.0, ls="--")
    for x in (-0.3, 0.3):
        ax.axvline(x, color=BASE, lw=0.6, ls=":")
    ax.annotate("训练区间", (-0.3, ax.get_ylim()[0]), xytext=(4, 4),
                textcoords="offset points", fontsize=7.5, color=BASE)
    ax.annotate("外推区", (0.32, ax.get_ylim()[0]), xytext=(4, 4),
                textcoords="offset points", fontsize=7.5, color=BASE)
    ax.set_xlabel(r"失真强度 $\alpha$")
    ax.set_ylabel("相对 clean 的增益 (pp)")
    ax.set_title("SimpleCNN / CIFAR-10", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "F11_extrapolation")


if __name__ == "__main__":
    p("=" * 62)
    p("生成论文配图（只读 CSV；不修改任何既有产物）")
    p(f"仓库根：{ROOT}")
    p("=" * 62)
    for fn in (f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            p(f"  !! {fn.__name__} 失败: {type(e).__name__}: {e}")
    p("\n完成。全部图在 paper/figures/ 下（pdf + png）。")
