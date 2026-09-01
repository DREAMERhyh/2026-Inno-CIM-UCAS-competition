"""
拓展研究3：量化误差与非线性误差联合影响分析
task_extension3_joint_analysis.py

研究背景：
    存算一体芯片真实信号路径：输入激活 → 模拟域 MAC → 固有非线性失真
        → ADC 模数转换 → 量化-反量化误差 → 数字后续逻辑
    本拓展在推理时同时注入两种误差，分析：
        1) 量化误差单独影响（8bit → 2bit）
        2) 非线性 + 量化联合影响（是否存在超加性叠加？）
        3) 任务3 Exp2 最优鲁棒模型在联合误差下是否仍有增益

脚本主流程：
    Step 0: 准备（解析参数 + 设种子 + 创目录 + 打印配置）
    Step 1: 加载数据（get_dataloaders → test_loader 10K）
    Step 2: 定义模型加载 load_model_for_ext3()（simple_cnn / exp2）
    Step 3: 单独量化误差扫描（每个 bit，apply_nonlinearity=False）
    Step 4: 联合误差扫描（每个 α × 每个 bit，两开关都 True）
    Step 5: 可视化（4 类图：量化曲线 / 热力图 / 联合 vs 单独 / 鲁棒性对比）
    Step 6: 生成 joint_error_summary.csv（nonlinearity_only / quantization_only / joint）
    Step 7: 终端中文摘要 + 超加性效应判断 + Exp2 增益对比

代码风格：
    · 函数 docstring 中文；图表（标题/坐标轴/图例）英文；CSV 表头英文；终端 print 中文。
    · 所有钩子使用 try/finally 保证稳健释放。
    · 每个 (α, bit) 组合前强制从磁盘重载干净权重，杜绝状态污染。
"""

import argparse
import csv
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.data_loader import get_dataloaders
from utils.paths import get_ckpt_root, get_outputs_root, get_num_classes
from utils.quantization import (
    register_joint_error_hooks,
    remove_hooks,
)


# ------------------------------------------------------------------
# 基础工具：随机种子 + 模型工厂 + 评估
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
    """统一模型工厂，与 train.py / task1 保持一致。"""
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "exp2":
        # 任务3 Exp2：RobustCNN + use_calibration=True
        from models.robust_cnn import RobustCNN
        return RobustCNN(num_classes=num_classes, use_calibration=True)
    elif model_name == "resnet18":
        # （可选）拓展1 ResNet-18 也支持
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes)
    elif model_name == "vgg11":
        # CIFAR-10 轻量适配版 VGG-11，约 9.2M 参数，纯前馈深层对照
        from models.vgg11 import VGG11
        return VGG11(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型名称 '{model_name}'，当前已支持 simple_cnn, exp2, resnet18, vgg11")


def _model_checkpoint_path(model_name: str, dataset: str = "cifar10") -> str:
    ckpt_root = get_ckpt_root(dataset)
    mapping = {
        "simple_cnn": os.path.join(ckpt_root, "simple_cnn", "best_model.pth"),
        "exp2": os.path.join(ckpt_root, "Exp2_Calib+Layerwise_simplecnn", "best_model.pth"),
        "resnet18": os.path.join(ckpt_root, "resnet18", "best_model.pth"),
        "vgg11": os.path.join(ckpt_root, "vgg11", "best_model.pth"),
    }
    if model_name not in mapping:
        raise ValueError(f"未知模型 {model_name}，无法映射 checkpoint 路径")
    return mapping[model_name]


@torch.no_grad()
def evaluate_on_test(model, test_loader, criterion, device):
    """在全量 test_loader 上推理返回 (accuracy %, avg_loss)。"""
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
# Step 2：模型加载 + 干净权重快照（每个组合重载）
# ------------------------------------------------------------------
def load_model_for_ext3(model_name: str, device: str, dataset: str = "cifar10"):
    """
    根据 model_name 创建模型并加载默认 checkpoint；打印参数量。
    返回 (model.to(device).eval(), clean_state_dict_cpu_deepcopy, known_clean_acc)
    """
    ckpt_path = _model_checkpoint_path(model_name, dataset=dataset)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[{model_name}] checkpoint 不存在: {ckpt_path}")

    print(f"  [Load] 创建模型: {model_name}")
    model = get_model(model_name, num_classes=get_num_classes(dataset))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [Load] 总参数量: {total_params:,}")

    ckpt = torch.load(ckpt_path, map_location=device)
    known_clean_acc = None
    if isinstance(ckpt, dict):
        ckpt_dataset = ckpt.get("dataset", "cifar10")
        print(f"  [Load] checkpoint 数据集来源: {ckpt_dataset}")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "best_test_acc" in ckpt:
                known_clean_acc = float(ckpt["best_test_acc"])
                print(f"  [Load] checkpoint 记录的干净准确率: {known_clean_acc:.2f}%")
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    model.to(device).eval()
    clean_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return model, clean_state_dict, known_clean_acc


# ------------------------------------------------------------------
# Step 3：单独量化误差扫描
# ------------------------------------------------------------------
def run_quantization_only(model_name, device, num_bits_list, test_loader, criterion, output_dir,
                          clean_state_dict_snapshot=None, reload_model_each=True, dataset="cifar10"):
    """
    仅量化误差扫描：apply_nonlinearity=False, apply_quantization=True。
    对每个 num_bits_list 中的 bit 值 → 全量 10K 推理 → 写 CSV。
    """
    print(f"\n  ====== [{model_name}] 单独量化误差扫描（{len(num_bits_list)} 个 bit 值） ======")
    if clean_state_dict_snapshot is None or reload_model_each:
        # 重新加载得到 model + 干净权重快照
        model, clean_state_dict, _ = load_model_for_ext3(model_name, device, dataset=dataset)
    else:
        model = get_model(model_name, get_num_classes(dataset)).to(device).eval()
        clean_state_dict = clean_state_dict_snapshot
        model.load_state_dict(clean_state_dict)

    results = []
    for nb in tqdm(num_bits_list, desc=f"Quant Scan ({model_name})", leave=True):
        # 每个 bit 强制重载干净权重
        model.load_state_dict(clean_state_dict)
        model.to(device).eval()

        hooks = register_joint_error_hooks(
            model,
            alpha=0.0,
            num_bits=nb,
            apply_nonlinearity=False,
            apply_quantization=True,
        )
        try:
            acc, loss = evaluate_on_test(model, test_loader, criterion, device)
        finally:
            remove_hooks(hooks)

        results.append((int(nb), float(acc), float(loss)))
        print(f"    [{model_name}] bit={int(nb):>2}  Acc={acc:.2f}%  Loss={loss:.4f}")

    csv_path = os.path.join(output_dir, f"{model_name}_quantization_only.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["num_bits", "accuracy", "loss"])
        for nb, acc, loss in results:
            w.writerow([f"{nb}", f"{acc:.4f}", f"{loss:.6f}"])
    print(f"  [Save] {csv_path}")
    return results, model, clean_state_dict


# ------------------------------------------------------------------
# Step 4：联合误差扫描（α × bit 网格）
# ------------------------------------------------------------------
def run_joint_error(model_name, device, alpha_list, num_bits_list, test_loader, criterion, output_dir,
                    clean_state_dict_snapshot=None, dataset="cifar10"):
    """
    联合误差扫描：apply_nonlinearity=True, apply_quantization=True（两个开关都 True）
    网格：α ∈ alpha_list × bit ∈ num_bits_list。
    """
    print(f"\n  ====== [{model_name}] 联合误差扫描（{len(alpha_list)} α × {len(num_bits_list)} bit = "
          f"{len(alpha_list) * len(num_bits_list)} 组合） ======")
    if clean_state_dict_snapshot is None:
        model, clean_state_dict, _ = load_model_for_ext3(model_name, device, dataset=dataset)
    else:
        model = get_model(model_name, get_num_classes(dataset)).to(device).eval()
        clean_state_dict = clean_state_dict_snapshot
        model.load_state_dict(clean_state_dict)

    results = []
    for alpha in tqdm(alpha_list, desc=f"Joint α ({model_name})", leave=True):
        for nb in num_bits_list:
            # 每个组合强制重载干净权重
            model.load_state_dict(clean_state_dict)
            model.to(device).eval()

            hooks = register_joint_error_hooks(
                model,
                alpha=float(alpha),
                num_bits=nb,
                apply_nonlinearity=True,
                apply_quantization=True,
            )
            try:
                acc, loss = evaluate_on_test(model, test_loader, criterion, device)
            finally:
                remove_hooks(hooks)

            results.append((float(alpha), int(nb), float(acc), float(loss)))
            # 只打印 α=±0.3/0.0 或 bit=4 的标志性组合，避免刷屏
            if abs(alpha) in (0.0, 0.3) and nb in (8, 4, 2):
                print(f"    [{model_name}] α={alpha:+.2f} bit={nb:>2}  Acc={acc:.2f}%  Loss={loss:.4f}")

    csv_path = os.path.join(output_dir, f"{model_name}_joint_error.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "num_bits", "accuracy", "loss"])
        for a, nb, acc, loss in results:
            w.writerow([f"{a}", f"{nb}", f"{acc:.4f}", f"{loss:.6f}"])
    print(f"  [Save] {csv_path}")
    return results


# ------------------------------------------------------------------
# 工具：读取 CSV 成 list[dict]
# ------------------------------------------------------------------
def read_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: row[k] for k in row})
    return rows


# ------------------------------------------------------------------
# Step 5：可视化（4 类图）
# ------------------------------------------------------------------
def plot_quantization_only(model_list, output_dir):
    """
    图1：单独量化精度曲线 quantization_only_{model_name}.png
    横轴 Number of Bits（按从小到大 2→8 或 从 8→2 按真实顺序）
    纵轴 Test Accuracy (%)；并标注 32bit 无量化基线（从 joint_error.csv 或 单独推理）
    """
    for mn in model_list:
        rows = read_csv_rows(os.path.join(output_dir, f"{mn}_quantization_only.csv"))
        if not rows:
            print(f"  [Plot Skip] quantization_only for {mn}: CSV 为空")
            continue
        bits = [int(r["num_bits"]) for r in rows]
        accs = [float(r["accuracy"]) for r in rows]
        # 从 joint_error 中找 num_bits=32 或 从 clean_state_dict 推断；若都没有则用 bit=32 虚拟
        clean_acc_32bit = None
        jrows = read_csv_rows(os.path.join(output_dir, f"{mn}_joint_error.csv"))
        for jr in jrows:
            if float(jr["alpha"]) == 0.0 and int(jr["num_bits"]) == 32:
                clean_acc_32bit = float(jr["accuracy"])
                break
        if clean_acc_32bit is None:
            # 若没有 32bit 数据，则用 bit=8 数据作为参考（因 8bit 通常接近无损失）
            for b, a in zip(bits, accs):
                if b == 8:
                    clean_acc_32bit = a
                    break
        # 按 bit 升序绘制（便于观察 2→8 精度恢复）
        order = sorted(range(len(bits)), key=lambda i: bits[i])
        bits_sorted = [bits[i] for i in order]
        accs_sorted = [accs[i] for i in order]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        ax.plot(bits_sorted, accs_sorted, marker="o", linewidth=2.2, color="#1f77b4")
        if clean_acc_32bit is not None:
            ax.axhline(clean_acc_32bit, linestyle="--", color="#d62728", linewidth=1.8,
                       label=f"32-bit Baseline = {clean_acc_32bit:.2f}%")
            ax.scatter([32], [clean_acc_32bit], marker="*", s=220, color="#d62728", zorder=6,
                       label=f"32-bit (no quantization)")
        ax.set_xlabel("Number of Bits", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_title(f"Impact of Quantization on Accuracy ({mn})", fontsize=13, fontweight="bold")
        # X 轴刻度：所有扫描过的 bit + 32（如果没扫 32 就手动补一下参考）
        xticks = sorted(set(bits_sorted + ([32] if clean_acc_32bit is not None else [])))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])
        ax.grid(True, alpha=0.3)
        if clean_acc_32bit is not None:
            ax.legend(fontsize=10, loc="best")
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"quantization_only_{mn}.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Save] {fig_path}")


def plot_joint_heatmap(model_list, output_dir, alpha_list, num_bits_list):
    """
    图2：联合误差热力图 joint_error_heatmap_{model_name}.png
    X: Nonlinearity Strength (α)，Y: Number of Bits（按从上到下从大到小，习惯上高位在上方）
    颜色: Test Accuracy (%)，带每个单元格数值文字标注，colorbar 右侧显示。
    """
    cmap = plt.get_cmap("RdYlGn")
    for mn in model_list:
        jrows = read_csv_rows(os.path.join(output_dir, f"{mn}_joint_error.csv"))
        if not jrows:
            print(f"  [Plot Skip] heatmap for {mn}: CSV 为空")
            continue
        # 构建 dict: (alpha, bit) → accuracy
        acc_map = {}
        for jr in jrows:
            key = (float(jr["alpha"]), int(jr["num_bits"]))
            acc_map[key] = float(jr["accuracy"])
        # 只取实际扫描到的 alpha / num_bits（与 alpha_list/num_bits_list 对齐，若有）
        alphas_in = sorted({a for a, _ in acc_map.keys()})
        bits_in = sorted({b for _, b in acc_map.keys()})
        alphas_use = [a for a in alpha_list if a in alphas_in] or alphas_in
        bits_use = [b for b in num_bits_list if b in bits_in] or bits_in
        # 填充矩阵
        Z = np.zeros((len(bits_use), len(alphas_use)), dtype=np.float32)
        for j, b in enumerate(bits_use):
            for i, a in enumerate(alphas_use):
                Z[j, i] = acc_map.get((a, b), float("nan"))

        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
        # Y 轴从大 bit 到小 bit（从上向下退化），故翻转行顺序
        Z_plot = Z[::-1, :]
        bits_ylabels = list(reversed(bits_use))
        im = ax.imshow(Z_plot, cmap=cmap, aspect="auto", vmin=0.0, vmax=100.0)

        # 在每个单元格中写文字
        for j in range(len(bits_ylabels)):
            for i in range(len(alphas_use)):
                val = Z_plot[j, i]
                if not np.isnan(val):
                    ax.text(i, j, f"{val:.1f}", ha="center", va="center",
                            fontsize=9,
                            color="black" if (val > 60.0 and val < 95.0) or (val < 30.0) else "white",
                            fontweight="bold")

        ax.set_xticks(range(len(alphas_use)))
        ax.set_xticklabels([f"{a:.1f}" for a in alphas_use], fontsize=10)
        ax.set_yticks(range(len(bits_ylabels)))
        ax.set_yticklabels([str(b) for b in bits_ylabels], fontsize=10)
        ax.set_xlabel("Nonlinearity Strength (α)", fontsize=12)
        ax.set_ylabel("Number of Bits", fontsize=12)
        ax.set_title(f"Joint Impact of Nonlinearity & Quantization ({mn})",
                     fontsize=13, fontweight="bold")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Test Accuracy (%)", fontsize=11)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"joint_error_heatmap_{mn}.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Save] {fig_path}")


def plot_joint_vs_alone(model_list, output_dir, target_bit=4, dataset="cifar10"):
    """
    图3：联合 vs 单独对比曲线 joint_vs_alone_comparison.png
    对每个模型选取固定 bit=target_bit（默认 4，中等量化强度）
    三条曲线同轴：
        · 仅非线性：从 joint_error.csv 提取 num_bits=32（无量化）或
                     读取 outputs/task1/alpha_sensitivity.csv（有则优先）
        · 仅量化：从 quantization_only.csv 中提取 bit=target_bit 的精度，画水平虚线
        · 联合：   从 joint_error.csv 提取 num_bits=target_bit 的行
    """
    n_models = len(model_list)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), dpi=120,
                             sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax_idx, mn in enumerate(model_list):
        ax = axes[ax_idx]

        # ---- 曲线 A：仅非线性 ----
        nonlinear_rows = []
        # 优先读 task1 基线（对 simple_cnn 一定存在；exp2/resnet18 可能不存在）
        task1_csv = None
        if mn == "simple_cnn":
            task1_csv = os.path.join(get_outputs_root(dataset), "task1_simplecnn", "alpha_sensitivity.csv")
        elif mn == "exp2":
            task1_csv = os.path.join(get_outputs_root(dataset), "task3_simplecnn", "Exp2_Calib+Layerwise", "alpha_sensitivity.csv")
        if task1_csv and os.path.exists(task1_csv):
            for r in read_csv_rows(task1_csv):
                nonlinear_rows.append((float(r["alpha"]), float(r["accuracy"])))
        if not nonlinear_rows:
            # 回退：joint_error.csv 中 num_bits=32 行（或实际最大值）
            jrows = read_csv_rows(os.path.join(output_dir, f"{mn}_joint_error.csv"))
            bits_max = max((int(jr["num_bits"]) for jr in jrows), default=32)
            for jr in jrows:
                if int(jr["num_bits"]) == bits_max:
                    nonlinear_rows.append((float(jr["alpha"]), float(jr["accuracy"])))
        # 排序
        nonlinear_rows.sort(key=lambda t: t[0])

        # ---- 数值 B：仅量化 bit=target_bit 的水平虚线 ----
        quant_only_acc = None
        qrows = read_csv_rows(os.path.join(output_dir, f"{mn}_quantization_only.csv"))
        for qr in qrows:
            if int(qr["num_bits"]) == target_bit:
                quant_only_acc = float(qr["accuracy"])
                break

        # ---- 曲线 C：联合 bit=target_bit ----
        joint_rows = []
        jrows = read_csv_rows(os.path.join(output_dir, f"{mn}_joint_error.csv"))
        for jr in jrows:
            if int(jr["num_bits"]) == target_bit:
                joint_rows.append((float(jr["alpha"]), float(jr["accuracy"])))
        joint_rows.sort(key=lambda t: t[0])

        # ---- 绘图 ----
        if nonlinear_rows:
            ax.plot([t[0] for t in nonlinear_rows], [t[1] for t in nonlinear_rows],
                    marker="o", linewidth=2.0, color="#1f77b4",
                    label="Nonlinearity Only (32-bit)")
        if quant_only_acc is not None:
            xs_range = [min([t[0] for t in nonlinear_rows + joint_rows]),
                        max([t[0] for t in nonlinear_rows + joint_rows])] if (nonlinear_rows or joint_rows) else [-0.3, 0.3]
            ax.axhline(quant_only_acc, linestyle="--", linewidth=2.0, color="#2ca02c",
                       label=f"Quantization Only ({target_bit}-bit) = {quant_only_acc:.2f}%")
        if joint_rows:
            ax.plot([t[0] for t in joint_rows], [t[1] for t in joint_rows],
                    marker="s", linewidth=2.2, color="#d62728",
                    label=f"Joint Nonlin. + {target_bit}-bit Quant")

        ax.set_xlabel("Nonlinearity Strength (α)", fontsize=12)
        ax.set_ylabel("Test Accuracy (%)", fontsize=12)
        ax.set_title(f"Joint vs Individual Error Impact (bit={target_bit}, {mn})",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(f"Joint vs Individual Error Impact (bit={target_bit})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, "joint_vs_alone_comparison.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Save] {fig_path}")


def plot_robustness_under_joint(model_list, output_dir, target_bit=4):
    """
    图4：两模型在联合误差下的鲁棒性对比 robustness_under_joint_error.png
    固定 bit=target_bit；一条曲线一个模型；横轴 α，纵轴 Accuracy (%)。
    SimpleCNN → 蓝虚线圆圈；Exp2 → 红实线方块；α=0 标注干净基线红星。
    """
    # 颜色 & marker 映射
    style_map = {
        "simple_cnn": {"color": "#1f77b4", "marker": "o", "linestyle": "--",
                       "label": "SimpleCNN + Joint Error"},
        "exp2":       {"color": "#d62728", "marker": "s", "linestyle": "-",
                       "label": "Exp2 (Robust) + Joint Error"},
        "resnet18":   {"color": "#2ca02c", "marker": "^", "linestyle": "-.",
                       "label": "ResNet-18 + Joint Error"},
    }

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    clean_baselines = {}
    for mn in model_list:
        jrows = read_csv_rows(os.path.join(output_dir, f"{mn}_joint_error.csv"))
        points = []
        for jr in jrows:
            if int(jr["num_bits"]) == target_bit:
                points.append((float(jr["alpha"]), float(jr["accuracy"])))
        if not points:
            print(f"  [Plot Skip] robustness: {mn} 缺少 bit={target_bit} 联合误差数据")
            continue
        points.sort(key=lambda t: t[0])
        s = style_map.get(mn, {"color": "purple", "marker": "*", "linestyle": "-",
                               "label": f"{mn} + Joint Error"})
        ax.plot([t[0] for t in points], [t[1] for t in points],
                marker=s["marker"], linestyle=s["linestyle"],
                linewidth=2.2, color=s["color"], label=s["label"])
        # 记录 α=0 的干净基线（bit=4 量化后的 α=0 下的数值，用于红星参考）
        for a, acc in points:
            if abs(a - 0.0) < 1e-9:
                clean_baselines[mn] = acc
    # 画 α=0 红星标注
    i_star = 0
    for mn, acc in clean_baselines.items():
        color = style_map.get(mn, {}).get("color", "red")
        ax.scatter([0.0], [acc], marker="*", s=240, facecolor=color,
                   edgecolors="black", linewidths=1.0, zorder=8 + i_star,
                   label=f"{mn} baseline @ α=0 = {acc:.2f}%")
        i_star += 1

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Nonlinearity Strength (α)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(f"Model Robustness Under Joint Errors (Nonlinearity + Quantization bit={target_bit})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "robustness_under_joint_error.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Save] {fig_path}")


# ------------------------------------------------------------------
# Step 6：汇总表 joint_error_summary.csv
# ------------------------------------------------------------------
def build_summary_csv(model_list, output_dir, dataset="cifar10"):
    """
    汇总三类数据：
        1. nonlinearity_only → 从 outputs/task1/alpha_sensitivity.csv（simple_cnn）/
           task3/Exp2_Calib+Layerwise/alpha_sensitivity.csv 或 joint_error 的 num_bits=32
        2. quantization_only → quantization_only.csv
        3. joint → joint_error.csv
    写列：model_name, alpha, num_bits, accuracy, loss, error_type
    """
    all_rows = []
    for mn in model_list:
        # ---- 1. nonlinearity_only ----
        source1 = None
        if mn == "simple_cnn" and os.path.exists(os.path.join(get_outputs_root(dataset), "task1_simplecnn", "alpha_sensitivity.csv")):
            source1 = os.path.join(get_outputs_root(dataset), "task1_simplecnn", "alpha_sensitivity.csv")
        elif mn == "exp2" and os.path.exists(os.path.join(get_outputs_root(dataset), "task3_simplecnn", "Exp2_Calib+Layerwise", "alpha_sensitivity.csv")):
            source1 = os.path.join(get_outputs_root(dataset), "task3_simplecnn", "Exp2_Calib+Layerwise", "alpha_sensitivity.csv")
        if source1:
            for r in read_csv_rows(source1):
                all_rows.append({
                    "model_name": mn,
                    "alpha": r["alpha"],
                    "num_bits": "32",
                    "accuracy": r["accuracy"],
                    "loss": r.get("loss", "NaN"),
                    "error_type": "nonlinearity_only",
                })
        else:
            # 回退：从 joint_error.csv 取 num_bits=32（或最大）
            jrows = read_csv_rows(os.path.join(output_dir, f"{mn}_joint_error.csv"))
            bits_ = sorted({int(jr["num_bits"]) for jr in jrows})
            nb = 32 if (32 in bits_) else (bits_[-1] if bits_ else 32)
            for jr in jrows:
                if int(jr["num_bits"]) == nb:
                    all_rows.append({
                        "model_name": mn,
                        "alpha": jr["alpha"],
                        "num_bits": str(nb),
                        "accuracy": jr["accuracy"],
                        "loss": jr["loss"],
                        "error_type": "nonlinearity_only",
                    })

        # ---- 2. quantization_only ----
        for qr in read_csv_rows(os.path.join(output_dir, f"{mn}_quantization_only.csv")):
            all_rows.append({
                "model_name": mn,
                "alpha": "0.0",
                "num_bits": qr["num_bits"],
                "accuracy": qr["accuracy"],
                "loss": qr["loss"],
                "error_type": "quantization_only",
            })

        # ---- 3. joint ----
        for jr in read_csv_rows(os.path.join(output_dir, f"{mn}_joint_error.csv")):
            all_rows.append({
                "model_name": mn,
                "alpha": jr["alpha"],
                "num_bits": jr["num_bits"],
                "accuracy": jr["accuracy"],
                "loss": jr["loss"],
                "error_type": "joint",
            })

    csv_path = os.path.join(output_dir, "joint_error_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model_name", "alpha", "num_bits", "accuracy", "loss", "error_type"]
        )
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"\n  [Save] Summary CSV: {csv_path} (共 {len(all_rows)} 行)")
    return all_rows


# ------------------------------------------------------------------
# argparse & main
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="拓展研究3：量化误差与非线性误差联合影响分析"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100"],
        help="数据集，默认: cifar10",
    )
    parser.add_argument("--models", type=str, default="simple_cnn,exp2",
                        help="逗号分隔的模型列表（默认 simple_cnn,exp2）")
    parser.add_argument("--alpha_values", type=str,
                        default="-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3",
                        help="α 扫描范围（逗号分隔）")
    parser.add_argument("--num_bits_list", type=str, default="8,6,4,3,2",
                        help="量化比特数扫描范围（逗号分隔）")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，默认 {outputs_root}/extension3_simplecnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_bit", type=int, default=4,
                        help="图3/图4 固定对比的 bit 值（默认 4，中等强度量化）")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Step 0：准备 ----
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = os.path.join(get_outputs_root(args.dataset), "extension3_simplecnn")
    os.makedirs(args.output_dir, exist_ok=True)

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    alpha_list = sorted([float(s.strip()) for s in args.alpha_values.split(",") if s.strip()])
    num_bits_list = [int(s.strip()) for s in args.num_bits_list.split(",") if s.strip()]

    if 0.0 not in alpha_list:
        alpha_list = [0.0] + alpha_list
        alpha_list.sort()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("\n" + "=" * 80)
    print("【拓展研究3：量化误差与非线性误差联合影响分析】启动")
    print(f"  数据集              : {args.dataset}")
    print(f"  对比模型            : {model_list}")
    print(f"  α 扫描范围          : {alpha_list}")
    print(f"  量化 bit 范围       : {num_bits_list}")
    print(f"  联合扫描组合数/模型  : {len(alpha_list)} × {len(num_bits_list)} = "
          f"{len(alpha_list) * len(num_bits_list)}")
    print(f"  Batch Size          : {args.batch_size}")
    print(f"  对比 target_bit     : {args.target_bit}（用于图3/图4）")
    print(f"  Device              : {device}")
    print(f"  输出目录            : {args.output_dir}")
    print("=" * 80)

    # ---- Step 1：加载数据 ----
    print("\n[Step 1] 加载 CIFAR-10 测试集 DataLoader ...")
    _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2, dataset=args.dataset)
    criterion = nn.CrossEntropyLoss()

    # ---- 先为每个模型获取干净权重快照（一次磁盘读取，后续所有扫描从内存 snapshot 重载） ----
    clean_snapshots = {}
    for mn in model_list:
        _, cs, _ = load_model_for_ext3(mn, device, dataset=args.dataset)
        clean_snapshots[mn] = cs

    # ---- Step 3：单独量化 ----
    print("\n" + "#" * 80)
    print("# Step 3：单独量化误差扫描")
    print("#" * 80)
    quant_only_results = {}
    for mn in model_list:
        res, _, _ = run_quantization_only(
            model_name=mn, device=device, num_bits_list=num_bits_list,
            test_loader=test_loader, criterion=criterion, output_dir=args.output_dir,
            clean_state_dict_snapshot=clean_snapshots[mn],
            dataset=args.dataset,
        )
        quant_only_results[mn] = res

    # ---- Step 4：联合误差扫描 ----
    print("\n" + "#" * 80)
    print("# Step 4：联合误差扫描（α × bit 网格）")
    print("#" * 80)
    joint_results = {}
    for mn in model_list:
        res = run_joint_error(
            model_name=mn, device=device, alpha_list=alpha_list,
            num_bits_list=num_bits_list, test_loader=test_loader,
            criterion=criterion, output_dir=args.output_dir,
            clean_state_dict_snapshot=clean_snapshots[mn],
            dataset=args.dataset,
        )
        joint_results[mn] = res

    # ---- Step 5：可视化 ----
    print("\n[Step 5] 生成可视化图表 ...")
    plot_quantization_only(model_list, args.output_dir)
    plot_joint_heatmap(model_list, args.output_dir, alpha_list, num_bits_list)
    plot_joint_vs_alone(model_list, args.output_dir, target_bit=args.target_bit, dataset=args.dataset)
    plot_robustness_under_joint(model_list, args.output_dir, target_bit=args.target_bit)

    # ---- Step 6：汇总 CSV ----
    print("\n[Step 6] 生成 joint_error_summary.csv ...")
    build_summary_csv(model_list, args.output_dir, dataset=args.dataset)

    # ---- Step 7：终端中文摘要 ----
    print("\n" + "=" * 80)
    print("【拓展研究3 - 量化误差与非线性误差联合影响 - 最终摘要】")
    print("=" * 80)

    # 从已写入的 CSV 中再次读取，保证摘要口径与 CSV 完全一致
    qdict = {}   # qdict[model_name][num_bits] = (acc, loss)
    for mn in model_list:
        qdict[mn] = {}
        for r in read_csv_rows(os.path.join(args.output_dir, f"{mn}_quantization_only.csv")):
            qdict[mn][int(r["num_bits"])] = (float(r["accuracy"]), float(r["loss"]))
    jdict = {}   # jdict[model_name][(alpha, num_bits)] = (acc, loss)
    for mn in model_list:
        jdict[mn] = {}
        for r in read_csv_rows(os.path.join(args.output_dir, f"{mn}_joint_error.csv")):
            jdict[mn][(float(r["alpha"]), int(r["num_bits"]))] = (float(r["accuracy"]), float(r["loss"]))
    # clean baseline：从 joint_error 中取 (α=0, bit=8) 或 (α=0, bit_max)；若 quant_only 有 bit=32 也可参考
    clean_acc = {}
    for mn in model_list:
        # 优先 32bit（如果没扫，就取量化扫描中的最大 bit）
        acc_32 = None
        for nb in sorted(qdict.get(mn, {}).keys(), reverse=True):
            acc_32, _ = qdict[mn][nb]
            break
        acc_alpha0_maxbit = None
        for (a, nb), (acc, _) in jdict.get(mn, {}).items():
            if abs(a - 0.0) < 1e-9:
                if acc_alpha0_maxbit is None or nb > acc_alpha0_maxbit[0]:
                    acc_alpha0_maxbit = (nb, acc)
        clean_acc[mn] = acc_alpha0_maxbit[1] if acc_alpha0_maxbit else acc_32

    # ---- 摘要 1：单独量化误差影响 ----
    print("\n[1] 单独量化误差影响（apply_nonlinearity=False）：")
    header_bits = sorted(num_bits_list)
    print("  {:<12} {} ｜ {}".format(
        "Model",
        " ".join([f"{b:>6}-bit" for b in header_bits]),
        "32-bit (clean baseline)"
    ))
    print("  " + "-" * 14 * len(header_bits) + "-" * 30)
    for mn in model_list:
        cells = []
        for b in header_bits:
            acc, _ = qdict.get(mn, {}).get(b, (float("nan"), 0.0))
            drop = (0.0 if (clean_acc.get(mn) is None or np.isnan(acc))
                    else (clean_acc[mn] - acc))
            cells.append(f"{acc:>5.2f}%({drop:>+.2f})")
        print("  {:<12} {} ｜ {:>10.2f}%".format(
            mn, "  ".join(cells),
            clean_acc.get(mn, float("nan"))
        ))

    # ---- 摘要 2：联合误差（固定 α=+0.3, bit=4） ----
    ref_a, ref_b = 0.3, args.target_bit
    print(f"\n[2] 联合误差影响（固定 α={ref_a:+.2f}, bit={ref_b}）：")
    print("  {:<12} {:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "Nonlin-Only", "Quant-Only", "Sum-Drops", "Joint Actual", "Joint-Drop"
    ))
    print("  " + "-" * 78)
    super_additive_flags = {}
    for mn in model_list:
        n_acc, q_acc, j_acc = float("nan"), float("nan"), float("nan")
        # Nonlinearity-only accuracy：从 summary 推断（bit=32 α=0.3 或 任务1基线）
        if mn in jdict:
            nbits = sorted({b for (a, b) in jdict[mn].keys()})
            nb32 = 32 if 32 in nbits else (nbits[-1] if nbits else None)
            if nb32 and (ref_a, nb32) in jdict[mn]:
                n_acc, _ = jdict[mn][(ref_a, nb32)]
        # Quantization-only accuracy
        if mn in qdict and ref_b in qdict[mn]:
            q_acc, _ = qdict[mn][ref_b]
        # Joint actual accuracy
        if (ref_a, ref_b) in jdict.get(mn, {}):
            j_acc, _ = jdict[mn][(ref_a, ref_b)]

        base = clean_acc.get(mn, float("nan"))
        drop_n = (base - n_acc) if (base == base and n_acc == n_acc) else float("nan")
        drop_q = (base - q_acc) if (base == base and q_acc == q_acc) else float("nan")
        drop_j = (base - j_acc) if (base == base and j_acc == j_acc) else float("nan")
        sum_drops = (drop_n + drop_q) if (drop_n == drop_n and drop_q == drop_q) else float("nan")
        # 判断超加性：联合跌幅 > sum → 超加恶化；≈ → 线性；< → 亚线性抵消
        flag = "?"
        if drop_j == drop_j and sum_drops == sum_drops:
            if drop_j > sum_drops * 1.05:
                flag = "Super-Additive (恶化)"
                super_additive_flags[mn] = (True, "超加性叠加恶化")
            elif drop_j < sum_drops * 0.95:
                flag = "Sub-Additive (抵消)"
                super_additive_flags[mn] = (False, "亚线性抵消")
            else:
                flag = "Linear"
                super_additive_flags[mn] = (False, "近似线性叠加")
        print("  {:<12} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>11.2f}% [{}]".format(
            mn, n_acc, q_acc,
            (base - sum_drops if (sum_drops == sum_drops and base == base) else float("nan")),
            j_acc, drop_j, flag
        ))

    # ---- 摘要 3：超加性效应判断 ----
    print("\n[3] 超加性效应判断（α={:+.2f}, bit={}）：".format(ref_a, ref_b))
    for mn in model_list:
        msg = super_additive_flags.get(mn, (None, "数据不足无法判断"))
        print(f"  · {mn:<12}：{msg[1]}")

    # ---- 摘要 4：鲁棒模型在联合误差下的表现 ----
    print(f"\n[4] 鲁棒模型在联合误差下的表现（α={ref_a:+.2f}, bit={ref_b}）：")
    smp_acc, exp2_acc = None, None
    if "simple_cnn" in jdict and (ref_a, ref_b) in jdict["simple_cnn"]:
        smp_acc = jdict["simple_cnn"][(ref_a, ref_b)][0]
    if "exp2" in jdict and (ref_a, ref_b) in jdict["exp2"]:
        exp2_acc = jdict["exp2"][(ref_a, ref_b)][0]
    if smp_acc is not None:
        print(f"  · SimpleCNN 在同等条件下精度   : {smp_acc:.2f}%")
    if exp2_acc is not None:
        print(f"  · Exp2 在同等条件下精度        : {exp2_acc:.2f}%")
    if smp_acc is not None and exp2_acc is not None:
        diff = exp2_acc - smp_acc
        print(f"  · Exp2 相对 SimpleCNN 增益    : {diff:+.2f}%  "
              f"({'有鲁棒性增益' if diff > 0.5 else ('轻微退化' if diff < -0.5 else '基本相当')})")

    # ---- 摘要 5：输出文件列表 ----
    print(f"\n[5] 所有输出文件位于：{os.path.abspath(args.output_dir)}/")
    for root, _, files in os.walk(args.output_dir):
        for fn in sorted(files):
            print(f"  · {os.path.join(os.path.abspath(root), fn)}")

    print("=" * 80)
    print("[Extension3 Done]")


if __name__ == "__main__":
    main()
