import argparse
import csv
import os
import sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

from utils.paths import get_outputs_root, get_num_classes


def parse_args():
    parser = argparse.ArgumentParser(
        description="跨架构汇总对比图（Extension 4 + Extension 6）"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100"],
        help="数据集，默认: cifar10",
    )
    parser.add_argument(
        "--ext4_csv", type=str, default=None,
        help="extension4 汇总 CSV，默认 {outputs_root}/extension4_alpha_wide/alpha_wide_scan_summary.csv",
    )
    parser.add_argument(
        "--ext6_csv", type=str, default=None,
        help="extension6 汇总 CSV，默认 {outputs_root}/extension6_deep_robust/alpha_wide_scan_deep_robust.csv",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="输出图片路径，默认 {outputs_root}/extension6_deep_robust/cross_architecture_summary.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    outputs_root = get_outputs_root(args.dataset)

    if args.ext4_csv is None:
        args.ext4_csv = os.path.join(outputs_root, "extension4_alpha_wide", "alpha_wide_scan_summary.csv")
    if args.ext6_csv is None:
        args.ext6_csv = os.path.join(outputs_root, "extension6_deep_robust", "alpha_wide_scan_deep_robust.csv")
    if args.out is None:
        args.out = os.path.join(outputs_root, "extension6_deep_robust", "cross_architecture_summary.png")

    has_ext4 = os.path.exists(args.ext4_csv)
    has_ext6 = os.path.exists(args.ext6_csv)
    if not has_ext4 and not has_ext6:
        print(f"[ERROR] 两个输入 CSV 均不存在:")
        print(f"  ext4_csv: {args.ext4_csv}")
        print(f"  ext6_csv: {args.ext6_csv}")
        print(f"请先运行:")
        print(f"  python task_extension4_alpha_wide_scan.py --dataset {args.dataset}")
        print(f"  python task_extension6_eval_deep_robust.py --dataset {args.dataset}")
        sys.exit(1)

    print(f"[Cross-Arch] 数据集: {args.dataset}, 类别数: {get_num_classes(args.dataset)}")
    print(f"[Cross-Arch] ext4_csv: {args.ext4_csv} ({'存在' if has_ext4 else '缺失'})")
    print(f"[Cross-Arch] ext6_csv: {args.ext6_csv} ({'存在' if has_ext6 else '缺失'})")
    print(f"[Cross-Arch] 输出: {args.out}")

    def load_csv(path):
        data = {}
        if not os.path.exists(path):
            return data
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                data.setdefault((row["model"], row["weight_type"]), {})[float(row["alpha"])] = float(row["accuracy"])
        return data

    curves = {}
    for d in (load_csv(args.ext4_csv), load_csv(args.ext6_csv)):
        curves.update(d)

    STYLE = {
        "clean": dict(color="#1f77b4", marker="o", ms=4, lw=1.6, label="Clean"),
        "nat_scratch": dict(color="#ff7f0e", marker="^", ms=4, lw=1.6, label="NAT Scratch"),
        "nat_finetune": dict(color="#2ca02c", marker="v", ms=4, lw=1.6, label="NAT Finetune"),
        "exp2_robust": dict(color="#d62728", marker="s", ms=6, lw=2.6, label="Exp2 Robust (Calib+NAT)"),
    }
    TITLES = {"simple_cnn": "SimpleCNN", "vgg11": "VGG-11", "resnet18": "ResNet-18"}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=120, sharey=True)
    for ax, model in zip(axes, ["simple_cnn", "vgg11", "resnet18"]):
        for wt in ["clean", "nat_scratch", "nat_finetune", "exp2_robust"]:
            if (model, wt) not in curves:
                continue
            pts = sorted(curves[(model, wt)].items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], **STYLE[wt])
        ax.axvline(-0.3, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.axvline(0.3, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_title(TITLES[model], fontsize=14, fontweight="bold")
        ax.set_xlabel("Nonlinearity Strength (alpha)", fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Test Accuracy (%)", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Exp2 Robust Scheme Transfer: Cross-Architecture Comparison ({args.dataset})",
                 fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
