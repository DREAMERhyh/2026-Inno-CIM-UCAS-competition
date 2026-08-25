"""
扩展研究4：α 宽范围扫描 —— 统一评估脚本

功能：
    将 α 扫描范围从训练区间 [-0.3, 0.3] 扩展至 [-0.6, 0.6]，
    对 10 组（模型 × 权重类型）组合执行纯推理扫描，评估 NAT 模型
    在训练分布外的外推鲁棒性。

实验矩阵：
    - α 扫描点（15 个）：[-0.6, -0.5, -0.4, -0.35, -0.3, -0.2, -0.1,
                          0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6]
      |α|>0.3 为外推区（加密步长 0.05-0.1），|α|≤0.3 为训练区间。
    - 权重矩阵（10 组）：simple_cnn(clean/nat_scratch/nat_finetune/exp2_robust)
                         + vgg11(clean/nat_scratch/nat_finetune)
                         + resnet18(clean/nat_scratch/nat_finetune)

输出：
    - alpha_wide_scan_summary.csv（6 列：model, weight_type, alpha, accuracy, loss, is_extrapolation）
    - {model}_alpha_wide.png（各模型分图）
    - all_models_alpha_wide.png（跨模型对比 3 子图）

运行命令：
    python task_extension4_alpha_wide_scan.py
"""

import argparse
import csv
import os
import random
import sys

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_loader import get_dataloaders
from utils.nonlinearity import (
    register_nonlinearity_hooks,
    remove_hooks,
)


# ======================================================================
# 基础工具函数（复用 task1 的实现模式）
# ======================================================================

def set_seed(seed: int = 42):
    """设置 Python / NumPy / PyTorch 随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_model(model_name: str, weight_type: str, num_classes: int = 10) -> nn.Module:
    """
    根据模型名和权重类型创建对应的模型实例。

    exp2_robust 权重来自 RobustCNN（FeatureCalibration 模块已启用），
    必须使用 RobustCNN 类加载，否则 state_dict 键不匹配会报错。
    其余 9 组权重使用标准模型工厂匹配。

    Args:
        model_name (str): simple_cnn / vgg11 / resnet18
        weight_type (str): clean / nat_scratch / nat_finetune / exp2_robust
        num_classes (int): 分类类别数，CIFAR-10 固定为 10

    Returns:
        nn.Module: 模型实例（未加载权重）
    """
    if weight_type == "exp2_robust":
        # exp2_robust 只对 simple_cnn 有效，此处由调用方保证
        from models.robust_cnn import RobustCNN
        return RobustCNN(num_classes=num_classes, use_calibration=True)

    # 标准模型工厂（与 task1_sensitivity_analysis.py 一致）
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "resnet18":
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes)
    elif model_name == "vgg11":
        from models.vgg11 import VGG11
        return VGG11(num_classes=num_classes)
    else:
        raise ValueError(
            f"不支持的模型名称: '{model_name}'，"
            f"当前已支持 simple_cnn, resnet18, vgg11"
        )


def load_checkpoint(checkpoint_path: str, device: str):
    """
    加载 checkpoint，兼容 dict（含 model_state_dict 键）与裸 state_dict 两种格式。

    Args:
        checkpoint_path (str): checkpoint 文件路径
        device (str): 加载设备

    Returns:
        dict: 模型 state_dict（可直接用于 model.load_state_dict）
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        if "best_test_acc" in ckpt:
            best_acc = ckpt["best_test_acc"]
        else:
            best_acc = None
    else:
        state_dict = ckpt
        best_acc = None
    return state_dict, best_acc


@torch.no_grad()
def evaluate_on_test(model, test_loader, criterion, device):
    """
    在给定测试集上对模型进行一次推理，返回总体准确率和平均 loss。

    调用前应确保非线性钩子已注册/移除到正确状态。
    """
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
    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return acc, avg_loss


# ======================================================================
# 权重矩阵构建
# ======================================================================

# 默认 α 扫描点（15 个，升序）
DEFAULT_ALPHAS = [
    -0.6, -0.5, -0.4, -0.35, -0.3, -0.2, -0.1,
    0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6,
]

# 权重类型 → checkpoint 子目录映射规则
# 占位符 {model} 为原始模型名（如 simple_cnn），{tag} 为 model.replace("_", "")
WEIGHT_TYPE_PATHS = {
    "clean": "{checkpoint_root}/{model}/best_model.pth",
    "nat_scratch": "{checkpoint_root}/nat_scratch_{tag}/best_model.pth",
    "nat_finetune": "{checkpoint_root}/nat_finetune_{tag}/best_model.pth",
    "exp2_robust": "{checkpoint_root}/Exp2_Calib+Layerwise_simplecnn/best_model.pth",
}


def build_weight_matrix(
    models: list,
    weight_types: list,
    checkpoint_root: str,
) -> list:
    """
    构建完整的权重扫描矩阵。

    每项包含 model, weight_type, checkpoint 路径三个字段。
    exp2_robust 仅对 simple_cnn 有效，其他模型自动跳过并打印警告。
    不存在的 checkpoint 自动跳过并打印警告。

    Args:
        models (list[str]): 模型名称列表
        weight_types (list[str]): 权重类型列表
        checkpoint_root (str): checkpoint 根目录

    Returns:
        list[dict]: 每项为 {"model": str, "weight_type": str, "checkpoint": str}
    """
    matrix = []
    for model_name in models:
        model_tag = model_name.replace("_", "")
        for wt in weight_types:
            # exp2_robust 仅对 simple_cnn 有效
            if wt == "exp2_robust" and model_name != "simple_cnn":
                print(
                    f"[WARNING] exp2_robust only available for simple_cnn, "
                    f"skipping ({model_name}, {wt})"
                )
                continue

            # 生成 checkpoint 路径
            ckpt_path = WEIGHT_TYPE_PATHS[wt].format(
                checkpoint_root=checkpoint_root,
                model=model_name,
                tag=model_tag,
            )

            if not os.path.exists(ckpt_path):
                print(f"[WARNING] Checkpoint not found: {ckpt_path}, skipping")
                continue

            matrix.append({
                "model": model_name,
                "weight_type": wt,
                "checkpoint": ckpt_path,
            })
    return matrix


# ======================================================================
# 绘图函数
# ======================================================================

# 各 weight_type 的配色方案（与项目现有图表风格一致）
COLOR_MAP = {
    "clean": "#1f77b4",
    "nat_scratch": "#ff7f0e",
    "nat_finetune": "#2ca02c",
    "exp2_robust": "#d62728",
}

# 各 weight_type 的显示标签（用于图例）
LABEL_MAP = {
    "clean": "Clean",
    "nat_scratch": "NAT Scratch",
    "nat_finetune": "NAT Finetune",
    "exp2_robust": "Exp2 Robust",
}


def plot_single_model(
    model_results: dict,
    model_name: str,
    output_dir: str,
):
    """
    绘制单个模型所有 weight_type 的 α-精度曲线。

    用竖直虚线标出 α=±0.3 训练边界，图例标注各 weight_type。

    Args:
        model_results (dict): weight_type → list[{"alpha": ..., "accuracy": ..., ...}]
        model_name (str): 模型名称（用于文件名和标题）
        output_dir (str): 输出目录
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # 按 weight_type 固定顺序绘制，保证图例一致性
    wt_order = ["clean", "nat_scratch", "nat_finetune", "exp2_robust"]
    for wt in wt_order:
        if wt not in model_results:
            continue
        data = model_results[wt]
        alphas = [r["alpha"] for r in data]
        accs = [r["accuracy"] for r in data]
        ax.plot(
            alphas, accs,
            marker="o", markersize=6, linewidth=2.2,
            color=COLOR_MAP.get(wt, "#333333"),
            label=LABEL_MAP.get(wt, wt),
        )

    # 竖直虚线标出训练边界 α=±0.3
    ax.axvline(x=-0.3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(x=0.3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    # 标注文字
    ax.text(-0.3, ax.get_ylim()[0] if ax.get_ylim()[0] < ax.get_ylim()[1] else 0,
            "α=-0.3", fontsize=8, color="gray", ha="center", va="bottom")
    ax.text(0.3, ax.get_ylim()[0] if ax.get_ylim()[0] < ax.get_ylim()[1] else 0,
            "α=0.3", fontsize=8, color="gray", ha="center", va="bottom")

    ax.set_xlabel("Nonlinearity Strength (α)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(f"{model_name}: Accuracy vs α", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    png_path = os.path.join(output_dir, f"{model_name}_alpha_wide.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Single model plot saved: {png_path}")


def plot_all_models(
    all_results: dict,
    models: list,
    output_dir: str,
):
    """
    跨模型对比图：3 个子图（每模型一个），共享 y 轴范围便于横向比较。

    Args:
        all_results (dict): model_name → {weight_type → [result_dict, ...]}
        models (list[str]): 模型名称列表（决定子图顺序）
        output_dir (str): 输出目录
    """
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), dpi=120, sharey=True)
    if n_models == 1:
        axes = [axes]

    # 计算全局 y 轴范围
    y_min, y_max = 100.0, 0.0
    for model_name in models:
        for wt_data in all_results.get(model_name, {}).values():
            for r in wt_data:
                acc = r["accuracy"]
                y_min = min(y_min, acc)
                y_max = max(y_max, acc)
    y_margin = max(5.0, (y_max - y_min) * 0.1)
    y_lim = (max(0.0, y_min - y_margin), min(100.0, y_max + y_margin))

    wt_order = ["clean", "nat_scratch", "nat_finetune", "exp2_robust"]

    for idx, model_name in enumerate(models):
        ax = axes[idx]
        model_data = all_results.get(model_name, {})

        for wt in wt_order:
            if wt not in model_data:
                continue
            data = model_data[wt]
            alphas = [r["alpha"] for r in data]
            accs = [r["accuracy"] for r in data]
            ax.plot(
                alphas, accs,
                marker="o", markersize=5, linewidth=2,
                color=COLOR_MAP.get(wt, "#333333"),
                label=LABEL_MAP.get(wt, wt),
            )

        # 训练边界虚线
        ax.axvline(x=-0.3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(x=0.3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

        ax.set_title(f"{model_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("α", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax.set_ylim(y_lim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    png_path = os.path.join(output_dir, "all_models_alpha_wide.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Cross-model comparison plot saved: {png_path}")


# ======================================================================
# 命令行参数
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extension 4: α 宽范围扫描 — 评估 NAT 模型在训练分布外的外推鲁棒性"
    )
    parser.add_argument(
        "--models", type=str, default="simple_cnn,vgg11,resnet18",
        help="逗号分隔的模型名称列表（默认: simple_cnn,vgg11,resnet18）",
    )
    parser.add_argument(
        "--weight_types", type=str,
        default="clean,nat_scratch,nat_finetune,exp2_robust",
        help="逗号分隔的权重类型列表（默认: clean,nat_scratch,nat_finetune,exp2_robust）",
    )
    parser.add_argument(
        "--alphas", type=str,
        default=",".join(str(a) for a in DEFAULT_ALPHAS),
        help="逗号分隔的 α 值列表（默认: 15 个扫描点，覆盖 [-0.6, 0.6]）",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./outputs/extension4_alpha_wide/",
        help="输出目录（默认: ./outputs/extension4_alpha_wide/）",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128,
        help="推理 batch size（默认: 128）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认: 42）",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="推理设备（默认自动选择）",
    )
    parser.add_argument(
        "--checkpoint_root", type=str, default="./checkpoints",
        help="checkpoint 根目录，允许重定位（默认: ./checkpoints）",
    )
    return parser.parse_args()


# ======================================================================
# 主函数
# ======================================================================

def main():
    args = parse_args()

    # ---------- Step 0: 随机种子 + 输出目录 ----------
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Extension4] 输出目录: {os.path.abspath(args.output_dir)}")

    # 解析参数列表
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    weight_types = [w.strip() for w in args.weight_types.split(",") if w.strip()]
    alpha_list = sorted([float(a.strip()) for a in args.alphas.split(",") if a.strip()])
    print(f"[Extension4] 模型列表: {models}")
    print(f"[Extension4] 权重类型: {weight_types}")
    print(f"[Extension4] α 扫描点: {alpha_list}")

    # 设备自动选择
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Extension4] 推理设备: {device}")

    # ---------- Step 1: 构建权重矩阵 ----------
    print("\n" + "-" * 60)
    print("[Extension4] 构建权重矩阵 ...")
    weight_matrix = build_weight_matrix(models, weight_types, args.checkpoint_root)
    if not weight_matrix:
        print("[ERROR] 权重矩阵为空，没有可用的 (model, weight_type) 组合。退出。")
        sys.exit(1)
    print(f"[Extension4] 共 {len(weight_matrix)} 个 (model, weight_type) 组合:")
    for entry in weight_matrix:
        print(f"    - {entry['model']:>12} / {entry['weight_type']:<14} → {entry['checkpoint']}")

    # ---------- Step 2: 加载测试集 ----------
    print("\n" + "-" * 60)
    print("[Extension4] 加载测试集 DataLoader ...")
    _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2)
    criterion = nn.CrossEntropyLoss()

    # ---------- Step 3: 执行全矩阵 α 扫描 ----------
    print("\n" + "-" * 60)
    print("[Extension4] 开始全矩阵 α 扫描 ...")

    # 汇总结果存储结构
    # all_results[model_name][weight_type] = [{"alpha": ..., "accuracy": ..., "loss": ..., "is_extrapolation": ...}, ...]
    all_results = {m: {} for m in models}

    # 总 CSV 行记录
    csv_rows = []

    for entry in weight_matrix:
        model_name = entry["model"]
        weight_type = entry["weight_type"]
        checkpoint_path = entry["checkpoint"]

        print(f"\n[Scan] model={model_name}, weight={weight_type}")
        print(f"        checkpoint={checkpoint_path}")

        # a. 创建模型实例（根据 weight_type 自动选择正确的模型类）
        model = create_model(model_name, weight_type, num_classes=10)

        # b. 加载权重
        state_dict, best_acc = load_checkpoint(checkpoint_path, device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        if best_acc is not None:
            print(f"        loaded checkpoint, best_test_acc={best_acc:.2f}%")

        # c. 遍历 α 扫描点
        model_results = []
        for alpha in alpha_list:
            # 注册非线性注入钩子
            hooks = register_nonlinearity_hooks(model, alpha)

            # 推理
            acc, loss = evaluate_on_test(model, test_loader, criterion, device)

            # 移除钩子（必须在下一轮 α 之前）
            remove_hooks(hooks)

            # 判断是否为外推区
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
                "weight_type": weight_type,
                "alpha": alpha,
                "accuracy": round(acc, 4),
                "loss": round(loss, 6),
                "is_extrapolation": is_extrapolation,
            })

            # 打印结果
            extrap_mark = " [Extrapolation]" if is_extrapolation else ""
            print(
                f"    Alpha={alpha:>6.3f}, Accuracy={acc:.2f}%, "
                f"Loss={loss:.4f}{extrap_mark}"
            )

        all_results[model_name][weight_type] = model_results

    # ---------- Step 4: 保存汇总 CSV ----------
    print("\n" + "-" * 60)
    print("[Extension4] 保存汇总 CSV ...")
    csv_path = os.path.join(args.output_dir, "alpha_wide_scan_summary.csv")
    fieldnames = ["model", "weight_type", "alpha", "accuracy", "loss", "is_extrapolation"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"[Extension4] CSV 已保存: {csv_path}")
    print(f"            共 {len(csv_rows)} 行")

    # ---------- Step 5: 分模型绘图 ----------
    print("\n" + "-" * 60)
    print("[Extension4] 绘制各模型曲线图 ...")
    for model_name in models:
        if model_name in all_results and all_results[model_name]:
            plot_single_model(all_results[model_name], model_name, args.output_dir)
        else:
            print(f"[WARNING] 模型 {model_name} 无数据，跳过绘图")

    # ---------- Step 6: 跨模型对比图 ----------
    print("\n" + "-" * 60)
    print("[Extension4] 绘制跨模型对比图 ...")
    plot_all_models(all_results, models, args.output_dir)

    # ---------- Step 7: 终端摘要 ----------
    print("\n" + "=" * 66)
    print("  [Extension4 摘要] α 宽范围扫描完成")
    print("=" * 66)
    print(f"  扫描 α 点数      : {len(alpha_list)}")
    print(f"  扫描 α 范围      : [{min(alpha_list):.2f}, {max(alpha_list):.2f}]")
    print(f"  训练区间          : |α| ≤ 0.3（{sum(1 for a in alpha_list if abs(a) <= 0.3)} 点）")
    print(f"  外推区            : |α| > 0.3（{sum(1 for a in alpha_list if abs(a) > 0.3)} 点）")
    print(f"  模型数            : {len(models)}")
    print(f"  权重组合数        : {len(weight_matrix)}")
    print(f"  总推理次数        : {len(csv_rows)}")
    print()
    print("  输出文件清单:")
    print(f"    CSV : {csv_path}")
    for model_name in models:
        print(f"    Plot: {os.path.join(args.output_dir, f'{model_name}_alpha_wide.png')}")
    print(f"    Plot: {os.path.join(args.output_dir, 'all_models_alpha_wide.png')}")
    print("=" * 66)
    print(f"[Extension4 Done] 所有结果保存在: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()