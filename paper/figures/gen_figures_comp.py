"""
竞赛版论文配图（第一 + 二阶段，CIFAR-10 为主）。零 GPU，只读 CSV。

与 gen_figures.py 同一套视觉规范（Wong 8 色 / SimHei / PDF 矢量），
但数据源全部指向 outputs/（CIFAR-10）一侧，因为竞赛阶段的实验以 CIFAR-10 为主。
脚本内不硬编码任何精度数字，末尾打印关键读数供交叉核对。
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))

WONG = ["#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
POS, NEG, BASE = "#D55E00", "#0072B2", "#999999"
plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 120, "savefig.bbox": "tight",
})


def p(*a):
    print(*a)


def read_csv(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def curve(rel, xk="alpha", yk="accuracy"):
    rows = read_csv(rel)
    if not rows:
        return None, None
    rows = sorted(rows, key=lambda r: float(r[xk]))
    return [float(r[xk]) for r in rows], [float(r[yk]) for r in rows]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    p(f"  [saved] {name}.pdf / .png")


# ---------------------------------------------------------------------
# C01 任务1：三架构的 α 敏感性（CIFAR-10）
# ---------------------------------------------------------------------
def c01():
    p("\n[C01] 任务1：三架构 α 敏感性（CIFAR-10）")
    series = [("SimpleCNN", "outputs/task1_simplecnn/alpha_sensitivity.csv", WONG[6]),
              ("VGG-11", "outputs/task1_vgg11/alpha_sensitivity.csv", WONG[5]),
              ("ResNet-18", "outputs/task1_resnet18/alpha_sensitivity.csv", WONG[3])]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for name, rel, c in series:
        a, v = curve(rel)
        if a is None:
            p(f"  !! 缺 {rel}")
            continue
        ax.plot(a, v, "o-", color=c, lw=1.6, ms=4, label=name)
        p(f"  {name:10s} clean={v[a.index(0.0)]:.2f}  +0.3={v[a.index(0.3)]:.2f}"
          f"  −0.3={v[a.index(-0.3)]:.2f}")
    ax.set_xlabel(r"失真强度 $\alpha$")
    ax.set_ylabel("测试精度 (%)")
    ax.legend(frameon=False)
    save(fig, "C01_task1_sensitivity")


# ---------------------------------------------------------------------
# C02 任务2：NAT 两种策略 vs 干净模型（CIFAR-10，三架构）
# ---------------------------------------------------------------------
def c02():
    p("\n[C02] 任务2：NAT scratch / finetune vs clean（CIFAR-10）")
    archs = [("SimpleCNN", "task2_simplecnn"), ("VGG-11", "task2_vgg11"),
             ("ResNet-18", "task2_resnet18")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    tag2tag = {"simplecnn": "simple_cnn", "vgg11": "vgg11", "resnet18": "resnet18"}
    for ax, (name, d) in zip(axes, archs):
        clean = f"outputs/task1_{d.split('_')[1]}/alpha_sensitivity.csv"
        a, v = curve(clean)
        if a is not None:
            ax.plot(a, v, "o--", color=BASE, lw=1.2, ms=3.5, label="clean")
        for mode, col in (("scratch", WONG[6]), ("finetune", WONG[5])):
            a2, v2 = curve(f"outputs/{d}/{mode}/alpha_sensitivity.csv")
            if a2 is None:
                continue
            ax.plot(a2, v2, "o-", color=col, lw=1.6, ms=3.5, label=f"NAT-{mode}")
            p(f"  {name:10s} {mode:9s} clean={v2[a2.index(0.0)]:.2f}  +0.3={v2[a2.index(0.3)]:.2f}")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(r"$\alpha$")
    axes[0].set_ylabel("测试精度 (%)")
    axes[0].legend(frameon=False, fontsize=7.5)
    save(fig, "C02_task2_nat")


# ---------------------------------------------------------------------
# C03 任务3：鲁棒增强的消融阶梯（CIFAR-10 / SimpleCNN）
# ---------------------------------------------------------------------
def c03():
    p("\n[C03] 任务3：Exp1/2/3 消融（CIFAR-10 / SimpleCNN）")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    a, v = curve("outputs/task1_simplecnn/alpha_sensitivity.csv")
    if a is not None:
        ax.plot(a, v, "o--", color=BASE, lw=1.2, ms=3.5, label="clean（无增强）")
        p(f"  clean         +0.3={v[a.index(0.3)]:.2f}")
    for exp, lab, col in (("Exp1_CalibOnly", "Exp1 仅校准模块", WONG[4]),
                          ("Exp2_Calib+Layerwise", "Exp2 +分层加权", WONG[6]),
                          ("Exp3_FullRobust", "Exp3 +非对称采样", WONG[3])):
        a2, v2 = curve(f"outputs/task3_simplecnn/{exp}/alpha_sensitivity.csv")
        if a2 is None:
            p(f"  !! 缺 {exp}")
            continue
        ax.plot(a2, v2, "o-", color=col, lw=1.6, ms=4, label=lab)
        p(f"  {lab:20s} clean={v2[a2.index(0.0)]:.2f}  +0.3={v2[a2.index(0.3)]:.2f}")
    ax.set_xlabel(r"失真强度 $\alpha$")
    ax.set_ylabel("测试精度 (%)")
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "C03_task3_ablation")


# ---------------------------------------------------------------------
# C04 拓展2：非线性失真 vs 高斯噪声（CIFAR-10，三架构）
# ---------------------------------------------------------------------
def c04():
    p("\n[C04] 拓展2：非线性 vs 高斯（CIFAR-10）")
    archs = [("SimpleCNN", "extension2_simplecnn/simple_cnn"),
             ("VGG-11", "extension2_vgg11/vgg11"),
             ("ResNet-18", "extension2_resnet18/resnet18")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    for ax, (name, d) in zip(axes, archs):
        ag, vg = curve(f"outputs/{d}/gaussian_sensitivity.csv", "noise_std")
        an, vn = curve(f"outputs/{d}/nonlinearity_sensitivity.csv", "alpha")
        if ag is not None:
            ax.plot(ag, vg, "s-", color=WONG[5], lw=1.5, ms=3.5, label=r"高斯噪声 $\sigma$")
            p(f"  {name:10s} 高斯 σ=0.3 → {vg[ag.index(0.3)] if 0.3 in ag else float('nan'):.2f}")
        if an is not None:
            ax.plot([abs(x) for x in an], vn, "o-", color=WONG[6], lw=1.5, ms=3.5,
                    label=r"非线性 $|\alpha|$")
            p(f"  {name:10s} 非线性 |α|=0.3 → {vn[an.index(0.3)]:.2f}")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(r"扰动强度")
    axes[0].set_ylabel("测试精度 (%)")
    axes[0].legend(frameon=False, fontsize=7.5)
    save(fig, "C04_task_ext2_gauss_vs_nonlin")


# ---------------------------------------------------------------------
# C05 拓展3：量化 × 非线性联合（CIFAR-10 / SimpleCNN / VGG-11）
# ---------------------------------------------------------------------
def c05():
    p("\n[C05] 拓展3：量化 × 非线性的加性裁决（CIFAR-10）")
    # 定义与台账 §3 一致（乘性独立预测）：
    #   独立预测(bit) = acc_nl(α=+0.3, 近无损bit) × acc_quant(bit) / acc_clean
    #   Δ = 实际 joint(α=+0.3, bit) − 独立预测(bit)
    #   Δ < −1 超加性（比独立更差）；Δ > +1 次加性（比独立更好）
    archs = [("SimpleCNN", "extension3_simplecnn", WONG[6]),
             ("VGG-11", "extension3_vgg11", WONG[5]),
             ("ResNet-18", "extension3_resnet18", WONG[3])]
    A = 0.3
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    for name, d, col in archs:
        rows = read_csv(f"outputs/{d}/joint_error_summary.csv")
        if not rows:
            p(f"  !! 缺 {d}")
            continue

        def first(et, alpha, bits=None):
            for r in rows:
                try:
                    if r["error_type"] != et or abs(float(r["alpha"]) - alpha) > 1e-6:
                        continue
                    if bits is not None and int(float(r["num_bits"])) != bits:
                        continue
                    return float(r["accuracy"])
                except (KeyError, ValueError):
                    continue
            return None

        nl_rows = [r for r in rows if r["error_type"] == "nonlinearity_only"]
        if not nl_rows:
            p(f"  !! {name} 无 nonlinearity_only 行")
            continue
        base_bits = max(int(float(r["num_bits"])) for r in nl_rows)   # 近无损位宽
        clean = first("nonlinearity_only", 0.0, base_bits)
        nl = first("nonlinearity_only", A, base_bits)
        if clean is None or nl is None:
            p(f"  !! {name} 缺 α=0/+0.3 基线（{base_bits}bit）")
            continue

        bits, deltas, seen = [], [], set()
        for r in rows:
            if r["error_type"] != "quantization_only":
                continue
            try:
                b = int(float(r["num_bits"]))
            except ValueError:
                continue
            if b in seen:          # 同一架构可能有重复的 quantization_only 行
                continue
            seen.add(b)
            q = first("quantization_only", 0.0, b)
            j = first("joint", A, b)
            if q is None or j is None:
                continue
            pred = nl * q / clean
            bits.append(b)
            deltas.append(j - pred)
        if not bits:
            continue
        order = np.argsort(bits)
        bits = [bits[i] for i in order]
        deltas = [deltas[i] for i in order]
        ax.plot(bits, deltas, "o-", color=col, lw=1.6, ms=4.5,
                label=f"{name}（{base_bits}bit 基线）")
        for b, dv in zip(bits, deltas):
            if b == 4:
                p(f"  {name:10s} 4bit Δ = {dv:+.2f}"
                  f"  （基线 clean={clean:.2f} / 非线性={nl:.2f}）")
        p(f"  {name:10s} 全 bits Δ: " + " ".join(f"{b}→{v:+.2f}" for b, v in zip(bits, deltas)))
    ax.axhline(0, color=BASE, lw=1.0, ls="--")
    ax.axhspan(-1, 1, color=BASE, alpha=0.10)
    ax.annotate(r"|$\Delta$|<1：≈独立", (max(bits), 0.25), fontsize=7.5, color=BASE, ha="right")
    ax.set_xlabel("ADC 量化位宽 (bit)")
    ax.set_ylabel(r"$\Delta$ = 实际联合 $-$ 独立预测 (pp)")
    ax.set_title(r"$\Delta<-1$ 超加性（相互放大）；$\Delta>+1$ 次加性（相互抵消）", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    save(fig, "C05_task_ext3_joint")


# ---------------------------------------------------------------------
# C06 拓展3/5：Dithering——4bit 在池化密集架构上反转（CIFAR-10）
# ---------------------------------------------------------------------
def c06():
    p("\n[C06] Dithering：4bit 增益的架构依赖（CIFAR-10）")
    rows = read_csv("outputs/extension5_dithering_vgg11/arch_dithering_summary.csv")
    if not rows:
        p("  !! 缺 arch_dithering_summary.csv")
        return
    names, mp, gain = [], [], []
    for r in rows:
        names.append(f"{r['model']}\n({r['maxpool_count']}×MaxPool)")
        mp.append(int(r["maxpool_count"]))
        gain.append(float(r["gain_4bit_vs_8bit"]))
        p(f"  {r['model']:12s} MaxPool={r['maxpool_count']}  4bit−8bit = "
          f"{float(r['gain_4bit_vs_8bit']):+.2f}")
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    cols = [WONG[3] if g > 0 else WONG[5] for g in gain]
    ax.bar(range(len(names)), gain, color=cols, width=0.55)
    ax.axhline(0, color=BASE, lw=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    for i, g in enumerate(gain):
        ax.annotate(f"{g:+.2f}", (i, g), xytext=(0, 4 if g >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=WONG[3] if g > 0 else WONG[5], fontweight="bold")
    ax.set_ylabel("4bit 相对 8bit 的精度差 (pp)")
    ax.set_title(r"$\alpha=+0.3$，CIFAR-10", fontsize=9)
    save(fig, "C06_dithering_arch")


# ---------------------------------------------------------------------
# C07 拓展6：深层迁移——Exp2 方案的三/四组权重对照（CIFAR-10）
# ---------------------------------------------------------------------
def c07():
    p("\n[C07] 拓展6：Exp2 方案迁移到深层主干（CIFAR-10，α=+0.3）")
    rows = read_csv("outputs/extension6_deep_robust/alpha_wide_scan_deep_robust.csv")
    if not rows:
        p("  !! 缺 ext6 CSV")
        return
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.2), sharey=False)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        for wt, col, lab in (("clean", BASE, "clean"),
                             ("nat_scratch", WONG[5], "NAT-scratch"),
                             ("nat_finetune", WONG[1], "NAT-finetune"),
                             ("exp2_robust", WONG[6], "Exp2（校准+分层）")):
            sub = sorted([r for r in rows if r["model"] == m and r["weight_type"] == wt],
                         key=lambda r: float(r["alpha"]))
            if not sub:
                continue
            ls = "--" if wt == "clean" else "-"
            ax.plot([float(r["alpha"]) for r in sub], [float(r["accuracy"]) for r in sub],
                    "o" + ls, color=col, lw=1.5, ms=3.5, label=lab)
            i3 = [i for i, r in enumerate(sub) if abs(float(r["alpha"]) - 0.3) < 1e-6]
            if i3:
                p(f"  {m:10s} {wt:14s} +0.3={float(sub[i3[0]]['accuracy']):.2f}")
        ax.set_title(m, fontsize=9)
        ax.set_xlabel(r"$\alpha$")
    axes[0].set_ylabel("测试精度 (%)")
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "C07_ext6_deep_transfer")


# ---------------------------------------------------------------------
# C08 第二阶段：NAT 外推边界（CIFAR-10 / SimpleCNN）
# ---------------------------------------------------------------------
def c08():
    p("\n[C08] 第二阶段：NAT 的外推边界（CIFAR-10 / SimpleCNN）")
    rows = read_csv("outputs/extension4_alpha_wide/alpha_wide_scan_summary.csv")
    if not rows:
        p("  !! 缺 ext4 CSV")
        return

    def cur(wt):
        sub = sorted([r for r in rows if r["model"] == "simple_cnn" and r["weight_type"] == wt],
                     key=lambda r: float(r["alpha"]))
        return {round(float(r["alpha"]), 4): float(r["accuracy"]) for r in sub}

    clean = cur("clean")
    if not clean:
        p("  !! 缺 simple_cnn/clean")
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for wt, lab, c in (("nat_scratch", "NAT-scratch", WONG[6]),
                       ("nat_finetune", "NAT-finetune", WONG[5]),
                       ("exp2_robust", "Exp2（校准+分层）", WONG[3])):
        d = cur(wt)
        xs = sorted(set(d) & set(clean))
        if not xs:
            continue
        ax.plot(xs, [d[x] - clean[x] for x in xs], "o-", color=c, lw=1.6, ms=4, label=lab)
        pos = [(x, d[x] - clean[x]) for x in xs if x >= 0.4]
        p("  " + lab + " 外推区: " + " ".join(f"{x:+.2f}={y:+.2f}" for x, y in pos))
    ax.axhline(0, color=BASE, lw=1.0, ls="--")
    for x in (-0.3, 0.3):
        ax.axvline(x, color=BASE, lw=0.6, ls=":")
    ax.set_xlabel(r"失真强度 $\alpha$")
    ax.set_ylabel("相对 clean 的增益 (pp)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "C08_ext4_extrapolation")


if __name__ == "__main__":
    p("=" * 62)
    p("竞赛版配图（第一 + 二阶段，CIFAR-10）")
    p("=" * 62)
    for fn in (c01, c02, c03, c04, c05, c06, c07, c08):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            p(f"  !! {fn.__name__} 失败: {type(e).__name__}: {e}")
    p("\n完成。")
