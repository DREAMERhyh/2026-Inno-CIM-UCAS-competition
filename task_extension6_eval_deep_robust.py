"""
拓展研究6：Exp2 鲁棒方案迁移至深层网络 —— 评估脚本
task_extension6_eval_deep_robust.py

功能：
    对迁移训练得到的两个 Exp2 鲁棒 checkpoint（VGG-11 / ResNet-18）执行
    α 宽范围扫描 [-0.6, 0.6]，复用 task_extension4 的扫描逻辑（逐 α 注册
    非线性钩子 → 推理 → 移除钩子）。

    评估结果与 extension4 的 clean / nat_scratch / nat_finetune 曲线叠加对比，
    直观展示 Exp2 鲁棒方案在训练分布外的外推鲁棒性提升。

输入 checkpoint（纯推理，不训练）：
    vgg11   → ./checkpoints/Exp2_Calib+Layerwise_vgg11/best_model.pth
    resnet18→ ./checkpoints/Exp2_Calib+Layerwise_resnet18/best_model.pth

输出：
    alpha_wide_scan_deep_robust.csv
        列：model, weight_type(=exp2_robust), alpha, accuracy, loss, is_extrapolation
    deep_robust_comparison.png
        两子图（每模型一个），叠加 extension4 的 clean/nat_scratch/nat_finetune 基准曲线

运行命令：
    python task_extension6_eval_deep_robust.py
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_loader import get_dataloaders
from utils.nonlinearity import register_nonlinearity_hooks, remove_hooks


# ======================================================================
# 基础工具（复用 task_extension4 的实现模式）
# ======================================================================

def set_seed(seed: int = 42):
    """设置随机种子。"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    模型工厂，返回 RobustVGG11 / RobustResNet18（与训练脚本一致）。

    exp2_robust 权重来自迁移训练的 Robust 模型（含 FeatureCalibration），
    必须用对应的 Robust 类加载，否则 state_dict 键不匹配。
    """
    if model_name == "vgg11":
        from models.robust_vgg11 import RobustVGG11
        return RobustVGG11(num_classes=num_classes)
    elif model_name == "resnet18":
        from models.robust_resnet import RobustResNet18
        return RobustResNet18(num_classes=num_classes)
    else:
        raise ValueError(
            f"不支持的模型名称: '{model_name}'，当前支持 vgg11, resnet18"
        )


def load_checkpoint(checkpoint_path: str, device: str):
    """加载 checkpoint，兼容 dict（含 model_state_dict）与裸 state_dict 两种格式。"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        best_acc = ckpt.get("best_test_acc", None)
    else:
        state_dict = ckpt
        best_acc = None
    return state_dict, best_acc


@torch.no_grad()
def evaluate_on_test(model, test_loader, criterion, device):
    """在测试集上推理，返回 (accuracy %, avg_loss)。调用前确保钩子状态正确。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(logits, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, avg_loss


# ======================================================================
# 常量
# ======================================================================

DEFAULT_ALPHAS = [
    -0.6, -0.5, -0.4, -0.35, -0.3, -0.2, -0.1,
    0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6,
]

# 权重类型固定为 exp2_robust
WEIGHT_TYPE = "exp2_robust"

# 对比基准曲线配色（与 task_extension4 一致）
COLOR_MAP = {
    "clean": "#1f77b4",
    "nat_scratch": "#ff7f0e",
    "nat_finetune": "#2ca02c",
    "exp2_robust": "#d62728",
}
LABEL_MAP = {
    "clean": "Clean",
    "nat_scratch": "NAT Scratch",
    "nat_finetune": "NAT Finetune",
    "exp2_robust": "Exp2 Robust",
}


def load_ext4_baselines(ext4_csv_path: str, model_name: str) -> dict:
    """
    从 extension4 的 summary CSV 读取该模型的 clean/nat_scratch/nat_finetune 基准曲线。

    若 CSV 不存在则返回空 dict（对比图仅画 exp2_robust）。
    只读，不修改原 CSV。
    """
    baselines = {}
    if not os.path.exists(ext4_csv_path):
        print(f"[INFO] 未找到 extension4 基准 CSV: {ext4_csv_path}，对比图仅画 exp2_robust")
        return baselines
    try:
        with open(ext4_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["model"] != model_name:
                    continue
                wt = row["weight_type"]
                if wt not in ("clean", "nat_scratch", "nat_finetune"):
                    continue
                if wt not in baselines:
                    baselines[wt] = []
                baselines[wt].append((float(row["alpha"]), float(row["accuracy"])))
        for wt in baselines:
            baselines[wt].sort(key=lambda p: p[0])
        print(f"[INFO] 已加载 {model_name} 的 extension4 基准: {list(baselines.keys())}")
    except Exception as e:
        print(f"[WARNING] 读取 extension4 基准 CSV 失败: {e}")
    return baselines


# ======================================================================
# 绘图
# ======================================================================

def plot_deep_robust_comparison(
    all_results: dict,
    models: list,
    output_dir: str,
    ext4_csv_path: str,
):
    """
    绘制 deep_robust_comparison.png：每模型一个子图，
    叠加 extension4 的 clean/nat_scratch/nat_finetune 基准 + 本脚本 exp2_robust 曲线。
    用竖直虚线标出 α=±0.3 训练边界。
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), dpi=120, sharey=True)
    if n == 1:
        axes = [axes]

    # 全局 y 轴范围
    y_min, y_max = 100.0, 0.0
    for model_name in models:
        for r in all_results.get(model_name, []):
            y_min = min(y_min, r["accuracy"])
            y_max = max(y_max, r["accuracy"])
        for wt_data in load_ext4_baselines(ext4_csv_path, model_name).values():
            for _, acc in wt_data:
                y_min = min(y_min, acc)
                y_max = max(y_max, acc)
    y_margin = max(5.0, (y_max - y_min) * 0.1)
    y_lim = (max(0.0, y_min - y_margin), min(100.0, y_max + y_margin))

    wt_order = ["clean", "nat_scratch", "nat_finetune", "exp2_robust"]

    for idx, model_name in enumerate(models):
        ax = axes[idx]

        # 基准曲线（extension4）
        baselines = load_ext4_baselines(ext4_csv_path, model_name)
        for wt in ["clean", "nat_scratch", "nat_finetune"]:
            if wt not in baselines:
                continue
            data = baselines[wt]
            alphas = [a for a, _ in data]
            accs = [acc for _, acc in data]
            ax.plot(
                alphas, accs,
                marker="o", markersize=4, linewidth=1.6, alpha=0.8,
                color=COLOR_MAP[wt], label=LABEL_MAP[wt],
            )

        # exp2_robust（本脚本结果）
        if model_name in all_results:
            data = all_results[model_name]
            alphas = [r["alpha"] for r in data]
            accs = [r["accuracy"] for r in data]
            ax.plot(
                alphas, accs,
                marker="s", markersize=6, linewidth=2.4,
                color=COLOR_MAP["exp2_robust"], label=LABEL_MAP["exp2_robust"],
            )

        # 训练边界
        ax.axvline(x=-0.3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(x=0.3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_title(model_name, fontsize=13, fontweight="bold")
        ax.set_xlabel("Nonlinearity Strength (α)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax.set_ylim(y_lim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    png_path = os.path.join(output_dir, "deep_robust_comparison.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] 对比图已保存: {png_path}")


# ======================================================================
# 命令行参数
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extension 6 评估: 对 VGG-11/ResNet-18 的 Exp2 鲁棒 checkpoint 执行 α 宽范围扫描"
    )
    parser.add_argument(
        "--models", type=str, default="vgg11,resnet18",
        help="逗号分隔的模型列表（默认: vgg11,resnet18）",
    )
    parser.add_argument(
        "--alphas", type=str,
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        help="逗号分隔的 α 值（默认: 15 点覆盖 [-0.6, 0.6]）",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./outputs/extension6_deep_robust/",
        help="输出目录（默认: ./outputs/extension6_deep_robust/）",
    )
    parser.add_argument(
        "--checkpoint_root", type=str, default="./checkpoints",
        help="checkpoint 根目录（默认: ./checkpoints）",
    )
    parser.add_argument(
        "--ext4_csv", type=str,
        default="./outputs/extension4_alpha_wide/alpha_wide_scan_summary.csv",
        help="extension4 汇总 CSV 路径（对比基准，默认: ./outputs/extension4_alpha_wide/alpha_wide_scan_summary.csv）",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="推理 batch size（默认 128）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="推理设备（默认自动）")
    return parser.parse_args()


# ======================================================================
# 主函数
# ======================================================================

def main():
    args = parse_args()

    # Step 0：准备
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Extension6 Eval] 输出目录: {os.path.abspath(args.output_dir)}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    alpha_list = sorted([float(a.strip()) for a in args.alphas.split(",") if a.strip()])
    print(f"[Extension6 Eval] 模型列表: {models}")
    print(f"[Extension6 Eval] α 扫描点: {alpha_list}")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Extension6 Eval] 推理设备: {device}")

    # Step 1：数据
    print("\n[Extension6 Eval] 加载测试集 ...")
    _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    # Step 2：逐模型扫描
    csv_rows = []
    all_results = {}   # model_name -> [{"alpha":, "accuracy":, "loss":, "is_extrapolation":}, ...]

    for model_name in models:
        model_tag = model_name.replace("_", "")
        ckpt_path = os.path.join(
            args.checkpoint_root,
            f"Exp2_Calib+Layerwise_{model_tag}",
            "best_model.pth",
        )
        print(f"\n[Scan] model={model_name}, weight={WEIGHT_TYPE}")
        print(f"        checkpoint={ckpt_path}")

        if not os.path.exists(ckpt_path):
            print(f"[WARNING] Checkpoint 不存在: {ckpt_path}，跳过 {model_name}")
            continue

        # 创建模型 + 加载权重
        model = create_model(model_name, num_classes=10)
        state_dict, best_acc = load_checkpoint(ckpt_path, device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        if best_acc is not None:
            print(f"        loaded checkpoint, best_test_acc={best_acc:.2f}%")

        # 遍历 α
        model_results = []
        for alpha in alpha_list:
            hooks = register_nonlinearity_hooks(model, alpha)
            acc, loss = evaluate_on_test(model, test_loader, criterion, device)
            remove_hooks(hooks)

            is_extrapolation = abs(alpha) > 0.3
            record = {
                "alpha": alpha,
                "accuracy": round(acc, 4),
                "loss": round(loss, 6),
                "is_extrapolation": is_extrapolation,
            }
            model_results.append(record)
            csv_rows.append({
                "model": model_name,
                "weight_type": WEIGHT_TYPE,
                "alpha": alpha,
                "accuracy": round(acc, 4),
                "loss": round(loss, 6),
                "is_extrapolation": is_extrapolation,
            })

            extrap_mark = " [Extrapolation]" if is_extrapolation else ""
            print(f"    Alpha={alpha:>6.3f}, Accuracy={acc:.2f}%, Loss={loss:.4f}{extrap_mark}")

        all_results[model_name] = model_results

    if not csv_rows:
        print("[ERROR] 无可用结果（checkpoint 均不存在），退出。")
        sys.exit(1)

    # Step 3：保存汇总 CSV
    csv_path = os.path.join(args.output_dir, "alpha_wide_scan_deep_robust.csv")
    fieldnames = ["model", "weight_type", "alpha", "accuracy", "loss", "is_extrapolation"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"\n[Extension6 Eval] CSV 已保存: {csv_path}（共 {len(csv_rows)} 行）")

    # Step 4：对比图
    print("\n[Extension6 Eval] 绘制对比图 ...")
    plot_deep_robust_comparison(all_results, models, args.output_dir, args.ext4_csv)

    # Step 5：终端摘要
    print("\n" + "=" * 66)
    print("  [Extension6 Eval 摘要] α 宽范围扫描完成")
    print("=" * 66)
    print(f"  扫描模型     : {list(all_results.keys())}")
    print(f"  α 范围       : [{min(alpha_list):.2f}, {max(alpha_list):.2f}]")
    print(f"  训练区间点数 : {sum(1 for a in alpha_list if abs(a) <= 0.3)}")
    print(f"  外推区点数   : {sum(1 for a in alpha_list if abs(a) > 0.3)}")
    print()
    print("  各模型关键精度（exp2_robust）:")
    for model_name, data in all_results.items():
        for r in data:
            if abs(r["alpha"]) < 1e-9:
                clean_acc = r["accuracy"]
            if abs(r["alpha"] - 0.3) < 1e-9:
                alpha03_acc = r["accuracy"]
        print(f"    {model_name:>10}: α=0 → {clean_acc:.2f}%, α=+0.3 → {alpha03_acc:.2f}%")
    print()
    print("  输出文件:")
    print(f"    CSV : {csv_path}")
    print(f"    Plot: {os.path.join(args.output_dir, 'deep_robust_comparison.png')}")
    print("=" * 66)
    print("[Extension6 Eval Done]")


if __name__ == "__main__":
    main()
