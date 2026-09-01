"""
拓展研究1：网络结构与参数量对非线性误差的影响 —— 对比分析脚本
task_extension1_network_comparison.py

核心目标：
    对比 SimpleCNN（~0.24M 参数，普通4层卷积+GAP+FC）与 ResNet-18（~11.17M 参数，
    4组残差块）在不同非线性强度 α 下的精度衰减模式，回答：
    1. 更大参数量的模型是否天然对非线性失真更鲁棒？
    2. 残差连接在缓解误差累积方面有何结构优势？
    3. "过参数化"的容错能力 vs "更深结构的误差累积放大" —— 哪个占主导？

脚本 6 Step 主流程（与需求文档对齐）：
    Step 0: 准备（解析参数 / 设种子 / 创目录 / 打印摘要）
    Step 1: 加载数据 (get_dataloaders → test_loader)
    Step 2: 定义模型加载函数 load_model_for_analysis()
    Step 3: 对每个模型执行 α 敏感性扫描（全量 10K 测试集）
    Step 4: 生成跨模型对比图（accuracy_comparison / accuracy_drop_comparison / robustness_vs_params）
    Step 5: 保存对比汇总表 network_comparison_summary.csv
    Step 6: 终端中文摘要输出

代码风格：
    - 所有函数 docstring 中文
    - 所有图表（标题/坐标轴/图例）英文
    - CSV 表头英文
    - 终端 print 输出中文（便于调试）
    - 所有钩子使用 try/finally 保证稳健移除
    - 每个 α 扫描前强制重载干净权重，避免状态污染
"""

import argparse
import os
import random
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_loader import get_dataloaders
from utils.paths import get_ckpt_root, get_outputs_root, get_num_classes
from utils.nonlinearity import (
    register_nonlinearity_hooks,
    remove_hooks,
)


# ------------------------------------------------------------------
# 基础工具：随机种子 / 模型工厂（复用 train.py/task1 风格）
# ------------------------------------------------------------------
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


def get_model(model_name: str, num_classes: int = 10):
    """统一的模型工厂，与 train.py / task1 保持一致。"""
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
        raise ValueError(f"不支持的模型名称 '{model_name}'，当前已支持 simple_cnn / resnet18 / vgg11")


# ------------------------------------------------------------------
# 基础评估：在全量 test_loader 上推理一次，返回 (acc, loss)
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_on_test(model, test_loader, criterion, device):
    """
    在给定测试集上对模型进行一次推理，返回总体准确率和平均 loss。

    注意：调用本函数前需确保非线性钩子已处于正确注册/移除状态。
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
    avg_loss = total_loss / total if total > 0 else 0.0
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc, avg_loss


# ------------------------------------------------------------------
# Step 2 模型加载：创建 + 加载 checkpoint + 打印参数量
# ------------------------------------------------------------------
def load_model_for_analysis(model_name: str, checkpoint_path: str, device: str, dataset: str = "cifar10"):
    """
    根据 model_name 创建模型实例，加载指定 checkpoint，打印总参数量。

    Args:
        model_name: "simple_cnn" / "resnet18"
        checkpoint_path: .pth 路径（支持纯 state_dict 或 Trainer 保存的 dict 格式）
        device: "cuda" / "cpu"

    Returns:
        (model, total_params)
            - model: 已 eval().to(device) 的模型实例
            - total_params: 模型总参数量（int，用于 summary CSV）
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")

    print(f"  [Load] 正在创建模型: {model_name}")
    model = get_model(model_name, num_classes=get_num_classes(dataset))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [Load] 总参数量: {total_params:,}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_dataset = ckpt.get("dataset", "cifar10")
        print(f"  [Load] checkpoint 数据集来源: {ckpt_dataset}")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "best_test_acc" in ckpt:
                print(f"  [Load] 权重文件记录的最佳 clean 测试准确率: {ckpt['best_test_acc']:.2f}%")
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    print(f"  [Load] 权重加载完成: {checkpoint_path}")
    return model, total_params


def load_nat_model(model_name, checkpoint_path, device, dataset="cifar10"):
    """
    加载 NAT 训练后的模型权重。
    """
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        model = SimpleCNN(num_classes=get_num_classes(dataset))
    elif model_name == "resnet18":
        from models.resnet import ResNet18
        model = ResNet18(num_classes=get_num_classes(dataset))
    elif model_name == "vgg11":
        from models.vgg11 import VGG11
        model = VGG11(num_classes=get_num_classes(dataset))
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    if not os.path.exists(checkpoint_path):
        print(f"[警告] NAT 权重文件不存在: {checkpoint_path}，跳过该模型")
        return None, 0

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_dataset = ckpt.get("dataset", "cifar10")
        print(f"  [NAT-{model_name}] checkpoint 数据集来源: {ckpt_dataset}")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            if "best_test_acc" in ckpt:
                print(f"  [NAT-{model_name}] 权重记录的最佳准确率: {ckpt['best_test_acc']:.2f}%")
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [NAT-{model_name}] 总参数量: {total_params:,}")
    return model, total_params


# ------------------------------------------------------------------
# Step 3：单模型 α 敏感性扫描（全量 10K 测试集）
# ------------------------------------------------------------------
def run_alpha_sensitivity_one_model(
    model_name, checkpoint_path, alpha_list,
    test_loader, criterion, device, output_dir,
    existing_model=None, existing_total_params=None,
    csv_suffix=None,
    dataset="cifar10",
):
    """
    对单个模型执行完整的 α 扫描，结果保存为 {output_dir}/{model_name}_{csv_suffix}_alpha_sensitivity.csv。

    每个 α 的流程：
        1) _reload_clean_weights() 重新加载干净权重（避免状态污染）
        2) register_nonlinearity_hooks(model, alpha=α)
        3) evaluate_on_test() 全量推理
        4) finally 中 remove_hooks()

    Args:
        model_name: str
        checkpoint_path: str（仅在 existing_model 为 None 时使用，用于 CSV 路径打印）
        alpha_list: List[float]
        test_loader: DataLoader（10,000 张全量测试集）
        criterion: CrossEntropyLoss
        device: str
        output_dir: 输出目录
        existing_model: 可选 —— 已加载好的模型（用于 NAT 模型，跳过内部 load）
        existing_total_params: 可选 —— 对应 existing_model 的参数量
        csv_suffix: 可选 —— CSV / 日志标签后缀（如 "nat"），用于 NAT 模型区分

    Returns:
        results: List[Tuple(alpha, accuracy, loss)] —— 与 task1_sensitivity_analysis 口径一致
        total_params: int（该模型总参数量，后续 summary CSV 用到）
    """
    # 第一次加载模型 & 保存 clean state_dict 作为基准，后续每个 α 都从这里 load 回去
    if existing_model is not None:
        model = existing_model
        total_params = existing_total_params
    else:
        model, total_params = load_model_for_analysis(model_name, checkpoint_path, device, dataset=dataset)
    clean_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    results = []
    display_label = f"{model_name}_{csv_suffix}" if csv_suffix else model_name

    print(f"\n  ====== [{display_label}] α 敏感性扫描（{len(alpha_list)} 个强度） ======")
    for alpha in tqdm(alpha_list, desc=f"Alpha Scan ({display_label})", leave=True):
        # 1. 重载干净权重
        model.load_state_dict(clean_state_dict)
        model.to(device)
        model.eval()

        # 2. 注册钩子 → 推理 → finally 移除钩子
        hooks = register_nonlinearity_hooks(model, alpha=float(alpha))
        try:
            acc, loss = evaluate_on_test(model, test_loader, criterion, device)
        finally:
            remove_hooks(hooks)

        results.append((float(alpha), float(acc), float(loss)))
        print(f"    [{display_label}] α={alpha:+.3f}  Acc={acc:.2f}%  Loss={loss:.4f}")

    # 保存 CSV（NAT 模型文件名带 _nat 后缀）
    if csv_suffix:
        csv_path = os.path.join(output_dir, f"{model_name}_nat_alpha_sensitivity.csv")
    else:
        csv_path = os.path.join(output_dir, f"{model_name}_alpha_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "accuracy", "loss"])
        for a, acc, l in results:
            writer.writerow([f"{a}", f"{acc:.4f}", f"{l:.6f}"])
    print(f"\n  [Save] {csv_path}")

    return results, total_params


# ------------------------------------------------------------------
# Step 4：跨模型对比图（3 张图）
# ------------------------------------------------------------------
def plot_network_comparison(all_results, all_params, output_dir,
                            nat_results=None, nat_params=None):
    """
    生成 3 张跨模型对比图（全英文标题/坐标轴/图例），支持额外叠加 NAT 模型曲线。

    Args:
        all_results: Dict[str, List[Tuple(alpha, accuracy, loss)]]
            key=clean_model_name，value=α 扫描结果列表
        all_params: Dict[str, int]
            key=clean_model_name，value=总参数量
        output_dir: str
            图片保存目录
        nat_results: Optional[Dict[str, List[Tuple(alpha, accuracy, loss)]]]
            key = display_name（如 "NAT-simple_cnn"），NAT 模型 α 扫描结果
        nat_params: Optional[Dict[str, int]]
            key = display_name，NAT 模型参数量；若未提供则复用同架构 clean 的
    """
    if nat_results is None:
        nat_results = {}
    if nat_params is None:
        nat_params = {}

    # 基础模型 → 颜色 / marker / linestyle 映射（clean：实线 + 圆圈/方块/三角）
    base_model_styles = {
        "simple_cnn": {"color": "#1f77b4", "marker": "o", "linestyle": "-",
                       "label": "SimpleCNN (0.24M params)"},
        "resnet18":   {"color": "#d62728", "marker": "s", "linestyle": "-",
                       "label": "ResNet-18 (11.17M params)"},
        "vgg11":      {"color": "#2ca02c", "marker": "^", "linestyle": "-",
                       "label": "VGG-11 (9.2M params)"},
    }
    # NAT 模型：同色 + 虚线 + 不同 marker，label 前缀 "NAT-"
    nat_marker = {"simple_cnn": "D", "resnet18": "v", "vgg11": "<"}

    def _get_style(model_name, is_nat):
        base = base_model_styles.get(model_name, {"color": "green", "marker": "*",
                                                   "linestyle": "-", "label": model_name})
        if not is_nat:
            return base
        return {
            "color": base["color"],
            "marker": nat_marker.get(model_name, "X"),
            "linestyle": "--",
            "label": f"NAT-{base['label']}",
        }

    # -------------------- 图1：accuracy_comparison.png --------------------
    print("\n  [Plot] 正在绘制 accuracy_comparison.png ...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    clean_baselines = {}   # model_name → (α=0 的 accuracy)

    # 先画 clean 模型
    for model_name, res_list in all_results.items():
        alphas = [r[0] for r in res_list]
        accs = [r[1] for r in res_list]
        s = _get_style(model_name, is_nat=False)
        ax.plot(alphas, accs, marker=s["marker"], linestyle=s["linestyle"],
                linewidth=2.2, color=s["color"], label=s["label"], markersize=7)
        for a, acc in zip(alphas, accs):
            if abs(a - 0.0) < 1e-9:
                clean_baselines[model_name] = acc

    # 再画 NAT 模型
    for display_name, res_list in nat_results.items():
        # 从 display_name 推断原始架构名（NAT-xxx → xxx）
        arch = display_name
        if display_name.startswith("NAT-"):
            arch = display_name[4:]
        alphas = [r[0] for r in res_list]
        accs = [r[1] for r in res_list]
        s = _get_style(arch, is_nat=True)
        ax.plot(alphas, accs, marker=s["marker"], linestyle=s["linestyle"],
                linewidth=2.0, color=s["color"], label=s["label"], markersize=7)
        for a, acc in zip(alphas, accs):
            if abs(a - 0.0) < 1e-9:
                clean_baselines[display_name] = acc

    # 红色星号标注 α=0 的干净基线
    i_star = 0
    for mn, clean_acc in clean_baselines.items():
        ax.scatter([0.0], [clean_acc], color="red", marker="*", s=200,
                   zorder=10 + i_star,
                   label=f"{mn} Clean (α=0) = {clean_acc:.2f}%")
        i_star += 1

    ax.axvline(x=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Nonlinearity Strength (α)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy vs Nonlinearity Strength (Clean + NAT)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best", ncol=2)
    all_alphas = sorted({a for res in all_results.values() for a, _, _ in res} |
                        {a for res in nat_results.values() for a, _, _ in res})
    ax.set_xticks(all_alphas)
    ax.set_xticklabels([f"{a:.1f}" for a in all_alphas])
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Save] {fig_path}")

    # -------------------- 图2：accuracy_drop_comparison.png --------------------
    print("  [Plot] 正在绘制 accuracy_drop_comparison.png ...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    def _plot_drop(results_dict, is_nat):
        for model_name, res_list in results_dict.items():
            clean_acc = None
            for a, acc, _ in res_list:
                if abs(a - 0.0) < 1e-9:
                    clean_acc = acc
                    break
            if clean_acc is None:
                print(f"    跳过 {model_name}：α=0 不在扫描范围内，无法计算 drop")
                continue
            xs = [a for a, _, _ in res_list if a > 1e-9]
            ys = [clean_acc - acc for a, acc, _ in res_list if a > 1e-9]
            arch = model_name[4:] if model_name.startswith("NAT-") else model_name
            s = _get_style(arch, is_nat=is_nat)
            ax.plot(xs, ys, marker=s["marker"], linestyle=s["linestyle"],
                    linewidth=2.2 if not is_nat else 2.0,
                    color=s["color"], label=s["label"], markersize=7)

    _plot_drop(all_results, is_nat=False)
    _plot_drop(nat_results, is_nat=True)

    ax.set_xlabel("Nonlinearity Strength (α > 0)", fontsize=12)
    ax.set_ylabel("Accuracy Drop from Clean Baseline (%)", fontsize=12)
    ax.set_title("Accuracy Drop Comparison (Clean vs NAT — Positive Distortion)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best", ncol=2)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "accuracy_drop_comparison.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Save] {fig_path}")

    # -------------------- 图3：robustness_vs_params.png（散点图） --------------------
    print("  [Plot] 正在绘制 robustness_vs_params.png ...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    def _get_acc_at_03(res_list):
        best_acc = None
        best_diff = None
        for a, acc, _ in res_list:
            diff = abs(a - 0.3)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_acc = acc
        return best_acc

    def _collect_points(results_dict, params_dict, is_nat):
        pts_x, pts_y, pts_labels, pts_colors = [], [], [], []
        for mn, res in results_dict.items():
            acc = _get_acc_at_03(res)
            if acc is None:
                print(f"    跳过 {mn}：α=+0.3 附近无数据")
                continue
            arch = mn[4:] if mn.startswith("NAT-") else mn
            if mn in params_dict:
                p = params_dict[mn]
            elif arch in all_params:
                p = all_params[arch]
            else:
                continue
            color = base_model_styles.get(arch, {"color": "green"})["color"]
            pts_x.append(p / 1_000_000.0)
            pts_y.append(acc)
            pts_labels.append(mn)
            pts_colors.append(color)
        return pts_x, pts_y, pts_labels, pts_colors

    xs_clean, ys_clean, lbl_clean, col_clean = _collect_points(all_results, all_params, False)
    xs_nat, ys_nat, lbl_nat, col_nat = _collect_points(nat_results, nat_params, True)

    # Clean：实心圆 + 黑边
    if xs_clean:
        ax.scatter(xs_clean, ys_clean, s=280, c=col_clean,
                   marker="o", zorder=5, edgecolors="black", linewidths=1.2)

    # NAT：空心圆 + 同色粗边
    if xs_nat:
        ax.scatter(xs_nat, ys_nat, s=320, facecolors="white",
                   edgecolors=col_nat, linewidths=2.2, marker="o", zorder=6)

    # 文字标注（clean：上方；NAT：下方避免重叠）
    for xi, yi, ln in zip(xs_clean, ys_clean, lbl_clean):
        ax.annotate(f"{ln}\n({xi:.2f}M, {yi:.2f}%)",
                    xy=(xi, yi), xytext=(12, 14), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0, alpha=0.7))
    for xi, yi, ln in zip(xs_nat, ys_nat, lbl_nat):
        ax.annotate(f"{ln}\n({xi:.2f}M, {yi:.2f}%)",
                    xy=(xi, yi), xytext=(12, -54), textcoords="offset points",
                    fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0, alpha=0.7))

    # 添加图例：空心 = NAT
    from matplotlib.lines import Line2D
    extra_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
               markeredgecolor="black", markersize=11, linewidth=0, label="Clean (filled)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="w",
               markeredgecolor="black", markersize=13, markeredgewidth=2.2,
               linewidth=0, label="NAT (hollow)"),
    ]
    ax.legend(handles=extra_handles, fontsize=10, loc="lower right")

    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("Model Parameters (M, Log Scale)", fontsize=12)
    ax.set_ylabel("Accuracy at α = +0.3 (%)", fontsize=12)
    ax.set_title("Robustness vs Model Size — Clean vs NAT (Strong Distortion)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "robustness_vs_params.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Save] {fig_path}")


# ------------------------------------------------------------------
# Step 5：保存对比汇总表 CSV
# ------------------------------------------------------------------
def save_summary_csv(all_results, all_params, output_dir,
                     nat_results=None, nat_params=None):
    """
    保存 network_comparison_summary.csv，列：
        model_name, total_params, clean_accuracy, acc_at_alpha_0.3, acc_drop_at_0.3

    若提供 nat_results / nat_params，则追加 NAT 行（model_name = display_name）。
    """
    if nat_results is None:
        nat_results = {}
    if nat_params is None:
        nat_params = {}

    rows = []

    def _append(iter_results, iter_params, suffix_tag):
        for model_name, res_list in iter_results.items():
            clean_acc = None
            for a, acc, _ in res_list:
                if abs(a - 0.0) < 1e-9:
                    clean_acc = acc
                    break
            best_acc_03 = None
            best_diff_03 = None
            for a, acc, _ in res_list:
                diff = abs(a - 0.3)
                if best_diff_03 is None or diff < best_diff_03:
                    best_diff_03 = diff
                    best_acc_03 = acc

            drop = 0.0 if (clean_acc is None or best_acc_03 is None) else (clean_acc - best_acc_03)
            total_p = iter_params.get(model_name, 0)
            if total_p == 0:
                arch = model_name[4:] if model_name.startswith("NAT-") else model_name
                total_p = all_params.get(arch, 0)

            rows.append({
                "model_name": model_name if suffix_tag == "" else model_name,
                "total_params": total_p,
                "clean_accuracy": round(float(clean_acc if clean_acc is not None else 0.0), 4),
                "acc_at_alpha_0.3": round(float(best_acc_03 if best_acc_03 is not None else 0.0), 4),
                "acc_drop_at_0.3": round(float(drop), 4),
            })

    _append(all_results, all_params, "")
    _append(nat_results, nat_params, "NAT")

    csv_path = os.path.join(output_dir, "network_comparison_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model_name", "total_params", "clean_accuracy",
                           "acc_at_alpha_0.3", "acc_drop_at_0.3"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\n  [Save] Summary CSV: {csv_path}")
    return rows


# ------------------------------------------------------------------
# argparse & main
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="拓展研究1：网络结构与参数量对非线性误差的影响 —— SimpleCNN vs VGG-11 vs ResNet-18 对比"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100"],
        help="数据集，默认: cifar10",
    )
    parser.add_argument(
        "--models", type=str, default="simple_cnn,resnet18",
        help="逗号分隔的对比模型列表（默认 simple_cnn,resnet18）",
    )
    parser.add_argument(
        "--checkpoints", type=str, default=None,
        help="逗号分隔的 checkpoint 路径，与 --models 一一对应，默认自动拼接",
    )
    parser.add_argument(
        "--nat_models", type=str, default="",
        help="（可选）逗号分隔的 NAT 模型名称列表，如 'simple_cnn,resnet18'",
    )
    parser.add_argument(
        "--nat_checkpoints", type=str, default="",
        help="（可选）逗号分隔的 NAT 权重路径列表，与 --nat_models 一一对应",
    )
    parser.add_argument(
        "--alpha_values", type=str,
        default="-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3",
        help="非线性失真 α 扫描范围（逗号分隔，默认与任务1保持一致：-0.3 到 +0.3 步长 0.1）",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="推理设备（默认自动选择：有 CUDA 用 CUDA）",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="结果输出目录，默认 {outputs_root}/extension1",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Step 0：准备 ----
    set_seed(args.seed)
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.checkpoints is None:
        ckpt_list = [os.path.join(get_ckpt_root(args.dataset), m, "best_model.pth") for m in model_list]
    else:
        ckpt_list = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    if len(model_list) != len(ckpt_list):
        raise ValueError(
            f"--models 和 --checkpoints 数量必须匹配，"
            f"当前 models={len(model_list)} 个，checkpoints={len(ckpt_list)} 个"
        )

    if args.output_dir is None:
        args.output_dir = os.path.join(get_outputs_root(args.dataset), "extension1")
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 解析 NAT 模型参数 ----
    nat_model_list = []
    nat_ckpt_list = []
    nat_source_info = {}
    if args.nat_models:
        nat_model_list = [m.strip() for m in args.nat_models.split(",") if m.strip()]
        nat_ckpt_list = [c.strip() for c in args.nat_checkpoints.split(",") if c.strip()]
        if len(nat_model_list) != len(nat_ckpt_list):
            print("[错误] --nat_models 和 --nat_checkpoints 数量不一致")
            sys.exit(1)

    alpha_list = sorted([float(s.strip()) for s in args.alpha_values.split(",") if s.strip() != ""])
    if 0.0 not in alpha_list:
        alpha_list = [0.0] + alpha_list
        alpha_list.sort()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("\n" + "=" * 72)
    print("【拓展研究1：网络结构与参数量对非线性误差的影响】启动")
    print(f"  数据集              : {args.dataset}")
    print(f"  对比模型            : {model_list}")
    print(f"  Checkpoint 路径     : {ckpt_list}")
    if nat_model_list:
        print(f"  NAT 模型            : {nat_model_list}")
        print(f"  NAT Checkpoint 路径 : {nat_ckpt_list}")
    print(f"  α 扫描范围          : {alpha_list}")
    print(f"  Batch Size          : {args.batch_size}")
    print(f"  Device              : {device}")
    print(f"  输出目录            : {args.output_dir}")
    print("=" * 72)

    # ---- Step 1：加载数据 ----
    print("\n[Step 1] 加载 " + args.dataset.upper() + " 测试集 DataLoader ...")
    _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2, dataset=args.dataset)
    criterion = nn.CrossEntropyLoss()

    # ---- Step 2：每个干净模型跑 α 扫描 ----
    all_results = {}
    all_params = {}
    for idx, (model_name, ckpt_path) in enumerate(zip(model_list, ckpt_list)):
        print(f"\n{'#' * 72}")
        print(f"### 模型 {idx + 1}/{len(model_list)}: {model_name}  checkpoint={ckpt_path}")
        print(f"{'#' * 72}")

        res, total_p = run_alpha_sensitivity_one_model(
            model_name=model_name,
            checkpoint_path=ckpt_path,
            alpha_list=alpha_list,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            output_dir=args.output_dir,
            dataset=args.dataset,
        )
        all_results[model_name] = res
        all_params[model_name] = total_p

    # ---- Step 2b：NAT 模型 α 扫描（可选） ----
    nat_results = {}
    nat_params = {}
    if nat_model_list:
        print(f"\n{'#' * 72}")
        print(f"### 开始处理 NAT 模型（共 {len(nat_model_list)} 个）")
        print(f"{'#' * 72}")
        for idx, (nat_model_name, nat_ckpt_path) in enumerate(zip(nat_model_list, nat_ckpt_list)):
            print(f"\n[NAT 模型 {idx + 1}/{len(nat_model_list)}] {nat_model_name} ← {nat_ckpt_path}")
            nat_model_obj, nat_total_p = load_nat_model(nat_model_name, nat_ckpt_path, device, dataset=args.dataset)
            if nat_model_obj is None:
                print(f"  → 跳过 {nat_model_name}（权重不存在或加载失败）")
                continue

            display_name = f"NAT-{nat_model_name}"
            nat_source_info[display_name] = nat_ckpt_path
            res_nat, _ = run_alpha_sensitivity_one_model(
                model_name=nat_model_name,
                checkpoint_path=nat_ckpt_path,
                alpha_list=alpha_list,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
                output_dir=args.output_dir,
                existing_model=nat_model_obj,
                existing_total_params=nat_total_p,
                csv_suffix="nat",
            )
            nat_results[display_name] = res_nat
            nat_params[display_name] = nat_total_p
            # 释放显存
            del nat_model_obj
            if device == "cuda":
                torch.cuda.empty_cache()

    # ---- Step 4：跨模型对比图（含 NAT 可选传入） ----
    print("\n[Step 4] 生成跨模型对比图 ...")
    plot_network_comparison(all_results, all_params, args.output_dir,
                            nat_results=nat_results if nat_results else None,
                            nat_params=nat_params if nat_params else None)

    # ---- Step 5：保存汇总表 ----
    print("\n[Step 5] 保存对比汇总表 ...")
    summary_rows = save_summary_csv(all_results, all_params, args.output_dir,
                                    nat_results=nat_results if nat_results else None,
                                    nat_params=nat_params if nat_params else None)

    # ---- Step 6：终端中文摘要 ----
    print("\n" + "=" * 80)
    print("【拓展研究1：终端中文摘要】")
    print("=" * 80)

    # 6.1 模型参数与精度对比表（包含 NAT）
    print("\n[1] 模型参数与精度对比表：")
    print("  {:<18} {:>14} {:>16} {:>16} {:>16}".format(
        "Model", "Total Params", "Clean Acc(α=0)", "Acc@α=0.3", "Drop@0.3"))
    print("  " + "-" * 86)
    for row in summary_rows:
        source_note = ""
        if row["model_name"] in nat_source_info:
            source_note = f"   (来源: {nat_source_info[row['model_name']]})"
        print("  {:<18} {:>14,} {:>14.2f}%  {:>14.2f}%  {:>14.2f}%{}".format(
            str(row["model_name"]),
            int(row["total_params"]),
            float(row["clean_accuracy"]),
            float(row["acc_at_alpha_0.3"]),
            float(row["acc_drop_at_0.3"]),
            source_note,
        ))

    # 6.2 各模型 α 扫描表（含 NAT）
    combined_scan_results = dict(all_results)
    combined_scan_results.update(nat_results)
    all_display_order = list(model_list) + [f"NAT-{m}" for m in nat_model_list]
    for display_mn in all_display_order:
        if display_mn not in combined_scan_results:
            continue
        res = combined_scan_results[display_mn]
        print(f"\n[2] {display_mn} 的 α 敏感性扫描表：")
        print("  {:>8} | {:>12} | {:>12} | {:>14}".format("α", "Accuracy", "Loss", "Drop vs Clean"))
        print("  " + "-" * 52)
        clean_a = None
        for a, acc, loss in res:
            if abs(a - 0.0) < 1e-9:
                clean_a = acc
                break
        for a, acc, loss in res:
            drop = 0.0 if clean_a is None else (clean_a - acc)
            print("  {:>+7.3f} | {:>10.2f}%   | {:>12.4f} | {:>+12.2f}%".format(
                a, acc, loss, drop))

    # 6.3 α=+0.3 跌幅对比 & 讨论（包含 NAT）
    print("\n[3] α=+0.3 时各模型精度跌幅对比：")
    drop_map = {r["model_name"]: float(r["acc_drop_at_0.3"]) for r in summary_rows}
    acc03_map = {r["model_name"]: float(r["acc_at_alpha_0.3"]) for r in summary_rows}
    for mn in summary_rows:
        dmn = mn["model_name"]
        print(f"  · {dmn:<18} : 跌幅 {drop_map.get(dmn, 0.0):.2f}%  (@α=0.3 精度 {acc03_map.get(dmn, 0.0):.2f}%)")

    # 6.4 NAT 模型相对同架构 clean 的 uplift 对比（若有）
    if nat_results:
        print("\n[4] NAT 模型相对同架构 Clean 模型 @α=+0.3 的 Uplift：")
        for display_name in nat_results:
            arch = display_name[4:] if display_name.startswith("NAT-") else display_name
            clean_drop = drop_map.get(arch, None)
            nat_drop = drop_map.get(display_name, None)
            clean_acc03 = acc03_map.get(arch, None)
            nat_acc03 = acc03_map.get(display_name, None)
            if clean_acc03 is not None and nat_acc03 is not None:
                uplift = nat_acc03 - clean_acc03
                arrow = "✓ 改善" if uplift > 0.5 else ("≈ 持平" if abs(uplift) <= 0.5 else "✗ 劣化")
                print(f"  · {display_name:<16} vs Clean-{arch:<10}:  ΔAcc = {uplift:+.2f}%  [{arrow}]")
                if clean_drop is not None and nat_drop is not None:
                    print(f"    Clean 跌幅 {clean_drop:.2f}%  →  NAT 跌幅 {nat_drop:.2f}%  (相对降低 {clean_drop - nat_drop:+.2f}%)")

    # 判断：参数量更大的模型是否更鲁棒（仅用 clean 模型对比）
    clean_rows = [r for r in summary_rows if not str(r["model_name"]).startswith("NAT-")]
    if len(clean_rows) >= 2:
        small_row = min(clean_rows, key=lambda r: r["total_params"])
        large_row = max(clean_rows, key=lambda r: r["total_params"])
        drop_diff = small_row["acc_drop_at_0.3"] - large_row["acc_drop_at_0.3"]
        acc_diff = large_row["acc_at_alpha_0.3"] - small_row["acc_at_alpha_0.3"]
        print(f"\n[5] 结论（参数量更小 {small_row['model_name']} vs 更大 {large_row['model_name']}）：")
        if drop_diff > 0.5:
            print(f"    ✓ 参数量更大的模型在 α=+0.3 下的精度跌幅更小（相差 {drop_diff:.2f}%），")
            print(f"      绝对准确率高出 {acc_diff:.2f}%，说明更大的模型（或残差结构）对非线性失真更鲁棒。")
            print(f"      可能原因：① 残差连接缓解了深层误差累积；② 更多参数提供了冗余自由度来容错。")
        elif drop_diff < -0.5:
            print(f"    ✗ 参数量更大的模型在 α=+0.3 下的精度跌幅反而更大（相差 {-drop_diff:.2f}%），")
            print(f"      表明更深的结构可能累积了更多非线性误差，过参数化的收益不足以抵消误差放大。")
        else:
            print(f"    ≈ 两者在 α=+0.3 下的跌幅差异较小（{abs(drop_diff):.2f}%），")
            print(f"      模型规模对该强度下的鲁棒性影响不显著。")

    print("\n所有输出文件位于:")
    for root, _, files in os.walk(args.output_dir):
        for fn in sorted(files):
            print(f"  · {os.path.join(os.path.abspath(root), fn)}")

    print("=" * 80)
    print("[Extension1 Done]")


if __name__ == "__main__":
    main()
