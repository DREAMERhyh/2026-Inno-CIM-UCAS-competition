"""
拓展研究2：高斯噪声注入与非线性失真对模型鲁棒性的不同影响机制对比

功能：
    1. 整网精度扫描：分别对 nonlinearity (α ∈ [-0.3,0.3]) 与 gaussian (σ ∈ [0.05,0.30])
       在两个模型（SimpleCNN / Exp2_RobustCNN）的全量测试集上推理精度 & loss；
    2. 层输出分布偏移：固定样本 500 张，捕获 conv1~conv4 + fc 共 5 层输出，
       计算相对平均误差 (RME) 与余弦相似度，并绘制误差累积曲线；
    3. 跨扰动对比：将可比强度（精度相近）的 α vs σ 做同图精度曲线 & 同图误差累积曲线；
    4. 跨模型鲁棒性迁移：SimpleCNN vs Exp2（针对非线性训练的鲁棒模型）在两种扰动下
       的四条曲线同图对比，验证 Exp2 对高斯噪声是否也有收益。

运行命令示例：
    # 默认：simple_cnn + exp2 两个模型，两种扰动全扫描，500 张层分析样本
    python task_extension2_noise_vs_nonlinearity.py

    # 仅分析干净模型，自定义噪声范围
    python task_extension2_noise_vs_nonlinearity.py \
        --models simple_cnn --noise_levels "0.1,0.2,0.3,0.5"
"""

import argparse
import random
import os
import csv
import numpy as np
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_loader import get_dataloaders
from utils.paths import get_ckpt_root, get_outputs_root, get_num_classes
from utils.perturbation import (
    register_perturbation_hooks,
    remove_hooks,
)


# ------------------------------------------------------------------
# 基础工具：set_seed / 固定模型参数 / 观测层列表
# ------------------------------------------------------------------
def set_seed(seed=42):
    """设置随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] 随机种子已设置为 {seed}")


# SimpleCNN / RobustCNN 的 5 个观测层
OBSERVE_LAYERS_SIMPLECNN = [
    ("conv1", "conv1_output"),
    ("conv2", "conv2_output"),
    ("conv3", "conv3_output"),
    ("conv4", "conv4_output"),
    ("fc", "fc_output"),
]

# VGG-11 的 5 个观测层（索引定位 features[idx]）
OBSERVE_LAYERS_VGG11 = [
    (7,  "block2_output"),
    (14, "block3_output"),
    (21, "block4_output"),
    (28, "block5_output"),
    ("fc", "fc_output"),   # fc 由通用处理单独注册
]

# ResNet-18 的 5 个观测层
OBSERVE_LAYERS_RESNET18 = [
    ("layer1", "layer1_output"),
    ("layer2", "layer2_output"),
    ("layer3", "layer3_output"),
    ("layer4", "layer4_output"),
    ("fc", "fc_output"),
]

# 默认使用 SimpleCNN 的观测层（向后兼容）
OBSERVE_LAYERS = OBSERVE_LAYERS_SIMPLECNN
LAYER_ORDER_DISPLAY = [ln for _, ln in OBSERVE_LAYERS_SIMPLECNN]

def get_observe_layers_for_ext2(model_name: str):
    """根据模型名返回对应的观测层列表和显示名列表。"""
    if model_name in ("simple_cnn", "exp2"):
        layers = OBSERVE_LAYERS_SIMPLECNN
    elif model_name == "vgg11":
        layers = OBSERVE_LAYERS_VGG11
    elif model_name == "resnet18":
        layers = OBSERVE_LAYERS_RESNET18
    else:
        layers = OBSERVE_LAYERS_SIMPLECNN
    return layers, [ln for _, ln in layers]

# ------------------------------------------------------------------
# Step 2：模型 & 权重加载
# ------------------------------------------------------------------
def get_model_for_ext2(model_name: str, device: str, dataset: str = "cifar10"):
    """加载拓展2所需的模型 & 干净权重。"""
    num_classes = get_num_classes(dataset)
    ckpt_root = get_ckpt_root(dataset)
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        model = SimpleCNN(num_classes=num_classes)
        ckpt_path = os.path.join(ckpt_root, "simple_cnn", "best_model.pth")
        known_clean_acc = 84.90
    elif model_name == "exp2":
        from models.robust_cnn import RobustCNN
        model = RobustCNN(num_classes=num_classes, use_calibration=True)
        ckpt_path = os.path.join(ckpt_root, "Exp2_Calib+Layerwise_simplecnn", "best_model.pth")
        known_clean_acc = 87.37
    elif model_name == "vgg11":
        from models.vgg11 import VGG11
        model = VGG11(num_classes=num_classes)
        ckpt_path = os.path.join(ckpt_root, "vgg11", "best_model.pth")
        known_clean_acc = 89.8
    elif model_name == "resnet18":
        from models.resnet import ResNet18
        model = ResNet18(num_classes=num_classes)
        ckpt_path = os.path.join(ckpt_root, "resnet18", "best_model.pth")
        known_clean_acc = 92.09
    else:
        raise ValueError(f"未知模型 model_name='{model_name}'，期望 'simple_cnn'/'exp2'/'vgg11'")

    if dataset != "cifar10":
        print(f"[Warning] known_clean_acc 为 CIFAR-10 经验值，仅作 fallback，"
              f"当前数据集为 {dataset}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"权重不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_dataset = ckpt.get("dataset", "cifar10")
        print(f"[Model] checkpoint 数据集来源: {ckpt_dataset}")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "best_test_acc" in ckpt:
                known_clean_acc = float(ckpt["best_test_acc"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()
    return model, known_clean_acc


# ------------------------------------------------------------------
# Step 1：收集固定样本（所有模型 / 扰动共享）
# ------------------------------------------------------------------
def collect_fixed_samples(test_loader, num_samples=500, device="cpu"):
    """
    从 test_loader 中抽取前 num_samples 张图像作为固定分析样本。

    Returns:
        fixed_images (Tensor): shape (N, 3, 32, 32)
        fixed_labels (Tensor): shape (N,)
    """
    images_list = []
    labels_list = []
    remain = num_samples
    for imgs, lbls in test_loader:
        take = min(remain, imgs.size(0))
        images_list.append(imgs[:take].cpu())
        labels_list.append(lbls[:take].cpu())
        remain -= take
        if remain <= 0:
            break
    fixed_images = torch.cat(images_list, dim=0)
    fixed_labels = torch.cat(labels_list, dim=0)
    print(f"[Data] 已收集固定样本数: {fixed_images.size(0)}")
    return fixed_images.to(device), fixed_labels.to(device)


# ------------------------------------------------------------------
# 基础评估：带扰动钩子在全量 test_loader 上推理，返回 (acc, loss)
# ------------------------------------------------------------------
def evaluate_full(model, test_loader, criterion, device,
                  pert_type="nonlinearity", alpha=0.0, noise_std=0.0):
    """
    注册扰动钩子，在全量 test_loader 上跑一次推理；
    finally 中 remove_hooks 保证钩子清理。
    """
    hooks = register_perturbation_hooks(model, pert_type=pert_type,
                                         alpha=alpha, noise_std=noise_std)
    total_loss = 0.0
    correct = 0
    total = 0
    try:
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
    finally:
        remove_hooks(hooks)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    return accuracy, avg_loss


# ------------------------------------------------------------------
# 子实验 A：整网精度扫描
# ------------------------------------------------------------------
def run_global_sensitivity(model, test_loader, criterion, device,
                           model_name: str, output_subdir: str,
                           pert_type: str,
                           alpha_list=None, noise_list=None, dataset="cifar10"):
    """
    对指定扰动类型执行全量精度扫描并保存 CSV。

    Returns:
        list[tuple(param_value, accuracy, loss)] 扫描结果
    """
    if pert_type == "nonlinearity":
        params = alpha_list if alpha_list is not None else []
        csv_path = os.path.join(output_subdir, "nonlinearity_sensitivity.csv")
    elif pert_type == "gaussian":
        params = noise_list if noise_list is not None else []
        csv_path = os.path.join(output_subdir, "gaussian_sensitivity.csv")
    else:
        raise ValueError

    results = []
    for p in params:
        # 重新加载干净权重（从 model_name 对应的 checkpoint）
        _reload_model_weights(model, model_name, device, dataset=dataset)

        if pert_type == "nonlinearity":
            acc, loss = evaluate_full(model, test_loader, criterion, device,
                                      pert_type="nonlinearity", alpha=float(p))
            print(f"  [Global {model_name}|{pert_type}] α={p:+.3f}  Acc={acc:.2f}%  Loss={loss:.4f}")
        else:
            acc, loss = evaluate_full(model, test_loader, criterion, device,
                                      pert_type="gaussian", noise_std=float(p))
            print(f"  [Global {model_name}|{pert_type}] σ={p:.3f}  Acc={acc:.2f}%  Loss={loss:.4f}")
        results.append((float(p), acc, loss))

    os.makedirs(output_subdir, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if pert_type == "nonlinearity":
            writer.writerow(["alpha", "accuracy", "loss"])
        else:
            writer.writerow(["noise_std", "accuracy", "loss"])
        for p, acc, loss in results:
            writer.writerow([f"{p}", f"{acc:.4f}", f"{loss:.6f}"])
    print(f"  [Save] {csv_path}")
    return results


# 重载干净权重：保证每个扰动强度从同干净模型出发，避免状态污染
def _reload_model_weights(model, model_name, device, dataset="cifar10"):
    ckpt_root = get_ckpt_root(dataset)
    if model_name == "simple_cnn":
        path = os.path.join(ckpt_root, "simple_cnn", "best_model.pth")
    elif model_name == "exp2":
        path = os.path.join(ckpt_root, "Exp2_Calib+Layerwise_simplecnn", "best_model.pth")
    elif model_name == "vgg11":
        path = os.path.join(ckpt_root, "vgg11", "best_model.pth")
    elif model_name == "resnet18":
        path = os.path.join(ckpt_root, "resnet18", "best_model.pth")
    else:
        return
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_dataset = ckpt.get("dataset", "cifar10")
        print(f"[Ext2] checkpoint 数据集来源: {ckpt_dataset}")
        if ckpt_dataset != dataset:
            print(f"[Ext2] WARNING: checkpoint 数据集 ({ckpt_dataset}) 与当前 dataset "
                  f"({dataset}) 不一致，可能导致 load_state_dict 形状不匹配")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()


# ------------------------------------------------------------------
# 层捕获钩子：观测层输出捕获
# ------------------------------------------------------------------
def _register_layer_capture_hooks(model, model_name: str, layer_out_dict: dict):
    hooks = []
    observe_layers, _ = get_observe_layers_for_ext2(model_name)

    for layer_info in observe_layers:
        # VGG-11：通过 model.features[idx] 定位
        if model_name == "vgg11":
            if isinstance(layer_info, tuple) and len(layer_info) == 2:
                first, display_name = layer_info
                if isinstance(first, int):
                    # 整数索引 → model.features[idx]
                    if 0 <= first < len(model.features):
                        module = model.features[first]
                    else:
                        continue
                elif first == "fc":
                    # fc 单独处理
                    if hasattr(model, 'classifier'):
                        children = list(model.classifier.children())
                        module = children[-1] if children else None
                    else:
                        continue
                else:
                    continue
            else:
                continue
        # ResNet-18 / SimpleCNN / RobustCNN：通过 getattr 定位
        else:
            attr_name, display_name = layer_info
            if attr_name == "fc":
                module = getattr(model, "fc", None)
            else:
                module = getattr(model, attr_name, None)

        if module is None:
            continue

        def make_hook(ln=display_name):
            def capture_hook(m, inp, out):
                layer_out_dict[ln] = out.detach().cpu()
            return capture_hook
        hooks.append(module.register_forward_hook(make_hook()))
    return hooks


# RME & Cosine 相似度计算（展平后逐样本平均，与 task1_layer_analysis 保持一致
def compute_metrics_pair(clean_out: torch.Tensor, dist_out: torch.Tensor):
    """
    计算两个特征图之间的：
        RME = ‖mean(dist) - mean(clean)‖₂ / ‖mean(clean)‖₂
        cosine_similarity = 逐样本展平后 cos_sim 的均值
    """
    # mean: 全局平均 (对所有通道/空间位置)
    mc = clean_out.float().mean()
    md = dist_out.float().mean()
    norm_diff = float(torch.norm(md - mc, p=2))
    norm_clean = float(torch.norm(mc, p=2))
    RME = norm_diff / max(norm_clean, 1e-12)

    # 逐样本展平：
    B = clean_out.size(0)
    c_flat = clean_out.reshape(B, -1).float()
    d_flat = dist_out.reshape(B, -1).float()

    c_n = c_flat / torch.clamp(torch.norm(c_flat, dim=1, keepdim=True), min=1e-12)
    d_n = d_flat / torch.clamp(torch.norm(d_flat, dim=1, keepdim=True), min=1e-12)
    cos_per_sample = (c_n * d_n).sum(dim=1)
    cos_sim = float(cos_per_sample.mean().item())
    return RME, cos_sim


# ------------------------------------------------------------------
# 子实验 B：层输出分布偏移 & 误差累积曲线 & 直方图
# ------------------------------------------------------------------
def run_layer_shift(model, fixed_imgs, fixed_lbls, criterion, device,
                    model_name, output_subdir, pert_type,
                    alpha_list=None, noise_list=None, dataset="cifar10"):
    """
    对指定扰动类型的层偏移分析：
        - 先取无扰动 (α=0 或 σ=0) 作为 clean 基线层输出
        - 对每个强度值：注册扰动钩子 + 层捕获钩子 → 推理 → 计算 RME & cos → 记录
        - 保存 CSV：layer_shift_{nonlinearity|gaussian}.csv
        - 绘制误差累积曲线：error_accumulation_{nonlinearity|gaussian}.png
        - gaussian 场景下额外画 conv1/conv4 分布直方图（σ=0 vs σ=0.2）
    """
    os.makedirs(output_subdir, exist_ok=True)

    # 动态获取当前模型的观测层列表和显示名列表
    observe_layers, LAYER_ORDER_DISPLAY = get_observe_layers_for_ext2(model_name)

    # 参数列表和 CSV/文件名
    if pert_type == "nonlinearity":
        params = list(alpha_list)
        # 确保 0.0 在最前（或保证存在）
        if 0.0 not in params:
            params = [0.0] + params
        csv_path = os.path.join(output_subdir, "layer_shift_nonlinearity.csv")
        png_cum = os.path.join(output_subdir, "error_accumulation_nonlinearity.png")
        param_col_header = "alpha"
    elif pert_type == "gaussian":
        params = list(noise_list)
        # 高斯场景默认无 σ=0，显式追加在最前作为 clean 参考
        if 0.0 not in params:
            params = [0.0] + params
        csv_path = os.path.join(output_subdir, "layer_shift_gaussian.csv")
        png_cum = os.path.join(output_subdir, "error_accumulation_gaussian.png")
        param_col_header = "noise_std"
    else:
        raise ValueError

    # ---- Step B.0：一次 clean 参考推理 (无扰动) ----
    _reload_model_weights(model, model_name, device, dataset=dataset)
    clean_layer_outputs = {}
    cap_hooks = _register_layer_capture_hooks(model, model_name, clean_layer_outputs)
    try:
        with torch.no_grad():
            _ = model(fixed_imgs.to(device))
    finally:
        remove_hooks(cap_hooks)

    # 保存 clean 的浅层/深层输出（用于 histogram 对比，层名根据模型动态确定）
    # 先确定浅层和深层的显示名
    if model_name in ("simple_cnn", "exp2"):
        shallow_name, deep_name = "conv1_output", "conv4_output"
    elif model_name == "vgg11":
        shallow_name, deep_name = "block2_output", "block5_output"
    elif model_name == "resnet18":
        shallow_name, deep_name = "layer1_output", "layer4_output"
    else:
        shallow_name, deep_name = "conv1_output", "conv4_output"

    clean_shallow_hist = clean_layer_outputs.get(shallow_name, None)
    clean_deep_hist = clean_layer_outputs.get(deep_name, None)
    dist_shallow_hist = None
    dist_deep_hist = None

    # ---- Step B.1：遍历每个扰动强度，记录各层 RME/cos ----
    shift_rows = []  # CSV 行
    per_strength_cos = {}  # param_value -> dict(layer_name -> cos_sim)

    for p in params:
        _reload_model_weights(model, model_name, device, dataset=dataset)

        # 注册扰动钩子
        if pert_type == "nonlinearity":
            pert_hooks = register_perturbation_hooks(model, "nonlinearity",
                                                     alpha=float(p), noise_std=0.0)
        else:
            pert_hooks = register_perturbation_hooks(model, "gaussian",
                                                     alpha=0.0, noise_std=float(p))

        layer_out = {}
        cap_hooks = _register_layer_capture_hooks(model, model_name, layer_out)
        try:
            with torch.no_grad():
                _ = model(fixed_imgs.to(device))
        finally:
            remove_hooks(cap_hooks)
            remove_hooks(pert_hooks)

        # 记录 RME / cos
        cur_cos = {}
        for _, ln in observe_layers:
            c_out = clean_layer_outputs.get(ln)
            d_out = layer_out.get(ln)
            if c_out is None or d_out is None:
                RME, cos_sim = 0.0, 1.0
            else:
                RME, cos_sim = compute_metrics_pair(c_out, d_out)
            shift_rows.append({
                param_col_header: p,
                "layer_name": ln,
                "RME": float(RME),
                "cosine_similarity": float(cos_sim),
            })
            cur_cos[ln] = float(cos_sim)

        # 保存目标强度下的浅层/深层输出（直方图，σ=0.20）
        if pert_type == "gaussian" and abs(p - 0.20) < 1e-9:
            dist_shallow_hist = layer_out.get(shallow_name)
            dist_deep_hist = layer_out.get(deep_name)

        per_strength_cos[float(p)] = cur_cos
        print(f"  [Layer  {model_name}|{pert_type}] {param_col_header}={p}  已完成")

    # ---- Step B.2：保存 CSV ----
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[param_col_header, "layer_name", "RME", "cosine_similarity"])
        writer.writeheader()
        for r in shift_rows:
            writer.writerow({
                param_col_header: f"{r[param_col_header]}",
                "layer_name": r["layer_name"],
                "RME": f"{r['RME']:.8f}",
                "cosine_similarity": f"{r['cosine_similarity']:.8f}",
            })
    print(f"  [Save] {csv_path}")

    # ---- Step B.3：绘制误差累积曲线（每层 cos_sim 连线） ----
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    layer_idx = list(range(len(LAYER_ORDER_DISPLAY)))
    cmap = plt.get_cmap("viridis")
    sorted_params = sorted(per_strength_cos.keys())
    n_p = len(sorted_params)
    for i, p in enumerate(sorted_params):
        cos_series = [per_strength_cos[p].get(ln, 1.0) for ln in LAYER_ORDER_DISPLAY]
        label_str = f"{'α' if pert_type=='nonlinearity' else 'σ'}={p:+.3f}" if pert_type == "nonlinearity" else f"σ={p:.3f}"
        ax.plot(layer_idx, cos_series, marker="o", linewidth=2,
                color=cmap(i / max(n_p - 1, 1)), label=label_str)

    ax.set_xticks(layer_idx)
    ax.set_xticklabels(LAYER_ORDER_DISPLAY, rotation=20)
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Cosine Similarity with Clean Output", fontsize=11)
    title_png = (f"Error Accumulation Across Layers ({model_name}, "
                 f"{'Nonlinearity' if pert_type=='nonlinearity' else 'Gaussian Noise'})")
    ax.set_title(title_png, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(png_cum, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Save] {png_cum}")

    # ---- Step B.4：高斯噪声场景下额外画直方图（σ=0 vs σ=0.2） ----
    if pert_type == "gaussian":
        # 确定文件名
        if model_name in ("simple_cnn", "exp2"):
            shallow_file = "histogram_conv1_gaussian.png"
            deep_file = "histogram_conv4_gaussian.png"
        elif model_name == "vgg11":
            shallow_file = "histogram_block2_gaussian.png"
            deep_file = "histogram_block5_gaussian.png"
        elif model_name == "resnet18":
            shallow_file = "histogram_layer1_gaussian.png"
            deep_file = "histogram_layer4_gaussian.png"
        else:
            shallow_file = "histogram_shallow_gaussian.png"
            deep_file = "histogram_deep_gaussian.png"

        for (ln, clean_t, dist_t, fname) in [
            (shallow_name, clean_shallow_hist, dist_shallow_hist,
             os.path.join(output_subdir, shallow_file)),
            (deep_name, clean_deep_hist, dist_deep_hist,
             os.path.join(output_subdir, deep_file)),
        ]:
            if clean_t is None or dist_t is None:
                continue
            fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
            ax.hist(clean_t.numpy().reshape(-1), bins=80, alpha=0.5, label="Clean (σ=0.0)",
                    color="#1f77b4", density=True)
            ax.hist(dist_t.numpy().reshape(-1), bins=80, alpha=0.5, label="Noisy (σ=0.20)",
                    color="#d62728", density=True)
            ax.set_title(f"Output Distribution at {ln} (σ=0.0 vs σ=0.20)",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("Activation Value", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            print(f"  [Save] {fname}")

    return per_strength_cos  # 返回：param -> layer -> cos_sim，用于跨扰动对比


# ------------------------------------------------------------------
# Step 4：跨扰动类型对比图
# ------------------------------------------------------------------
def plot_accuracy_comparison_one_model(
    model_name,
    nonlinearity_results,
    gaussian_results,
    output_subdir,
):
    """
    图1：整网精度对比（同一模型下，非线性 α vs 高斯 σ 两条曲线）。
    - 横轴：扰动强度（统一 0~1 之间归一化刻度，分开显示；各自原始强度）
    - 为了更清晰，使用左右双横轴：
        下轴 (bottom): α ∈ [-0.3, 0.3]
        上轴 (top)   : σ ∈ [0.00, 0.30]
    - 两条曲线的纵轴同用 Accuracy (%)
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # 非线性失真（下横轴刻度）
    alphas = [r[0] for r in nonlinearity_results]
    accs_a = [r[1] for r in nonlinearity_results]
    ax.plot(alphas, accs_a, marker="o", linewidth=2.2,
            color="#d62728", label="Nonlinearity (α)")
    ax.set_xlabel("Nonlinearity Strength (α)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)

    # 高斯噪声：新增右侧独立 twinx 轴不合适；直接采用同一张图 + 上轴次刻度的方式
    # 这里实现：以扰动强度"绝对值范围归一化"的方式在同一 ax 叠加，
    # 并采用第二个 x 轴（顶部）显示噪声刻度。
    ax2 = ax.twiny()
    noises = [r[0] for r in gaussian_results]
    accs_n = [r[1] for r in gaussian_results]
    # 高斯 0~0.3 线性映射到 -0.3~0.3 的展示位置：x_g = noise*2 - 0.3
    mapped_x = [n * 2.0 - 0.3 for n in noises]
    ax.plot(mapped_x, accs_n, marker="s", linestyle="--", linewidth=2.2,
            color="#1f77b4", label="Gaussian Noise (σ)")
    # 顶部轴：显示 0.00, 0.06, 0.12, 0.18, 0.24, 0.30
    top_ticks_mapped = [-0.3, -0.18, -0.06, 0.06, 0.18, 0.30]
    top_ticks_noises = [0.00, 0.06, 0.12, 0.18, 0.24, 0.30]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(top_ticks_mapped)
    ax2.set_xticklabels([f"{v:.2f}" for v in top_ticks_noises])
    ax2.set_xlabel("Gaussian Noise STD (σ)", fontsize=11)

    # 标注 clean (α=0 / σ=0 公共起点)
    for i, a in enumerate(alphas):
        if abs(a - 0.0) < 1e-9:
            ax.scatter([0.0], [accs_a[i]], color="red", marker="*", s=150,
                       zorder=10, label=f"Clean Baseline = {accs_a[i]:.2f}%")
            break

    ax.set_title(f"Accuracy vs Perturbation Strength ({model_name})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    save_path = os.path.join(output_subdir, f"accuracy_comparison_{model_name}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {save_path}")


def plot_layer_accumulation_comparison_one_model(
    model_name,
    nl_cos_per_p,
    gs_cos_per_p,
    output_subdir,
    target_alpha=0.2,
    target_sigma=0.15,
):
    """
    图2：层误差累积对比（可比扰动强度，精度相近）。
    默认：非线性 α=0.2（SimpleCNN 精度约 73%）vs 高斯 σ=0.15（精度接近）
    """
    # 取距离目标值最近的参数
    def nearest(d, t):
        if not d:
            return None
        ks = list(d.keys())
        best = min(ks, key=lambda k: abs(k - t))
        return best

    p_a = nearest(nl_cos_per_p, target_alpha)
    p_n = nearest(gs_cos_per_p, target_sigma)

    layer_idx = list(range(len(LAYER_ORDER_DISPLAY)))
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    if p_a is not None:
        y_a = [nl_cos_per_p[p_a].get(ln, 1.0) for ln in LAYER_ORDER_DISPLAY]
        ax.plot(layer_idx, y_a, marker="o", linewidth=2.2,
                color="#d62728", label=f"Nonlinearity α={p_a:+.3f}")
    if p_n is not None:
        y_n = [gs_cos_per_p[p_n].get(ln, 1.0) for ln in LAYER_ORDER_DISPLAY]
        ax.plot(layer_idx, y_n, marker="s", linestyle="--", linewidth=2.2,
                color="#1f77b4", label=f"Gaussian σ={p_n:.3f}")

    ax.set_xticks(layer_idx)
    ax.set_xticklabels(LAYER_ORDER_DISPLAY, rotation=20)
    ax.set_xlabel("Layer Index", fontsize=11)
    ax.set_ylabel("Cosine Similarity with Clean Output", fontsize=11)
    ax.set_title(f"Error Accumulation: Nonlinearity vs Gaussian Noise ({model_name})",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    save_path = os.path.join(output_subdir, f"error_accumulation_comparison_{model_name}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {save_path}")


# ------------------------------------------------------------------
# Step 5：跨模型鲁棒性迁移对比（SimpleCNN vs Exp2 × 两种扰动 × 四条曲线）
# ------------------------------------------------------------------
def plot_robustness_transfer(
    all_global_results,
    output_root,
):
    """
    图3：robustness_transfer_comparison.png
    统一横轴采用扰动强度索引（对于两种类型，分别用 α 和 σ 绘制，
    但为了直观：将非线性和高斯分左右两个子图，或同图展示。
    需求指定的是四条曲线同图。
    实现：
        横轴使用 "perturbation index"（0,1,2,...,N-1）作为通用刻度。
        图例中展示对应 α / σ 值；或采用下方说明。
    更清晰做法：双列子图共享 y 轴；左列 Nonlinearity，右列 Gaussian。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120, sharey=True)

    # 左侧：非线性（横轴 α）
    ax = axes[0]
    for model_name, style in [("simple_cnn", ("--", "#1f77b4", "SimpleCNN + Nonlinearity")),
                              ("exp2",       ("-",  "#d62728", "Exp2 + Nonlinearity"))]:
        if model_name not in all_global_results:
            continue
        res = all_global_results[model_name]["nonlinearity"]
        xs = [r[0] for r in res]
        ys = [r[1] for r in res]
        ls, c, lb = style
        ax.plot(xs, ys, marker="o", linestyle=ls, linewidth=2.2, color=c, label=lb)
    ax.set_xlabel("Nonlinearity Strength (α)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.set_title("Nonlinearity Distortion", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 右侧：高斯（横轴 σ）
    ax = axes[1]
    for model_name, style in [("simple_cnn", ("--", "#1f77b4", "SimpleCNN + Gaussian")),
                              ("exp2",       ("-",  "#d62728", "Exp2 + Gaussian"))]:
        if model_name not in all_global_results:
            continue
        res = all_global_results[model_name]["gaussian"]
        xs = [r[0] for r in res]
        ys = [r[1] for r in res]
        ls, c, lb = style
        ax.plot(xs, ys, marker="s", linestyle=ls, linewidth=2.2, color=c, label=lb)
    ax.set_xlabel("Gaussian Noise STD (σ)", fontsize=11)
    ax.set_title("Gaussian Additive Noise", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("Robustness Transfer: Model-specific vs Perturbation Type",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(output_root, "robustness_transfer_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Save] {save_path}")


# ------------------------------------------------------------------
# argparse & main
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="拓展研究2：高斯噪声 vs 非线性失真 鲁棒性影响机制对比"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100"],
        help="数据集，默认: cifar10",
    )
    parser.add_argument(
        "--models", type=str, default="simple_cnn,exp2,resnet18",
        help="逗号分隔，可选 simple_cnn / exp2 / resnet18"
    )
    parser.add_argument(
        "--pert_types", type=str, default="nonlinearity,gaussian",
        help="逗号分隔，扰动类型，可选 nonlinearity / gaussian"
    )
    parser.add_argument(
        "--alpha_values", type=str,
        default="-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3",
        help="非线性 α 扫描范围，逗号分隔"
    )
    parser.add_argument(
        "--noise_levels", type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30",
        help="高斯噪声 σ 扫描范围（不强制包含 0，内部会自动添加 clean 参考），逗号分隔"
    )
    parser.add_argument("--num_samples", type=int, default=500,
                        help="层分析固定样本数（默认 500）")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，默认 {outputs_root}/extension2_{model_tag}")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device if args.device is not None else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 解析列表参数
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    pert_list = [p.strip() for p in args.pert_types.split(",") if p.strip()]
    alpha_list = [float(v) for v in args.alpha_values.split(",") if v.strip() != ""]
    noise_list = [float(v) for v in args.noise_levels.split(",") if v.strip() != ""]

    # 输出目录
    model_tag = model_list[0].replace("_", "") if model_list else "simplecnn"
    if args.output_dir is None:
        args.output_dir = os.path.join(get_outputs_root(args.dataset), f"extension2_{model_tag}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("【拓展研究2：高斯噪声 vs 非线性失真 鲁棒性影响机制对比】启动")
    print(f"  数据集         : {args.dataset}")
    print(f"  Models       : {model_list}")
    print(f"  Perturbations: {pert_list}")
    print(f"  α 范围       : {alpha_list}")
    print(f"  σ 范围       : {noise_list}")
    print(f"  层分析样本   : {args.num_samples}")
    print(f"  Device       : {device}")
    print(f"  Output dir   : {args.output_dir}")
    print("=" * 70)

    # Step 1：数据 & 固定样本
    _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2, dataset=args.dataset)
    fixed_imgs, fixed_lbls = collect_fixed_samples(
        test_loader, num_samples=args.num_samples, device=device
    )

    criterion = nn.CrossEntropyLoss()

    # 记录：所有模型 × 所有扰动类型 的 全局扫描结果 & 层 cos 结果
    all_global_results = {}   # model_name -> {"nonlinearity": [...], "gaussian": [...]}
    all_layer_cos = {}        # model_name -> {"nonlinearity": {...}, "gaussian": {...}}

    # Step 3：遍历模型 × 扰动
    for model_name in model_list:
        output_subdir = os.path.join(args.output_dir, model_name)
        os.makedirs(output_subdir, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"### 模型: {model_name}")
        print(f"{'#'*70}")

        # 先加载模型 & 权重
        model, known_clean = get_model_for_ext2(model_name, device, dataset=args.dataset)
        print(f"[Model] {model_name} 干净基线准确率 (已知): {known_clean:.2f}%")

        if model_name not in all_global_results:
            all_global_results[model_name] = {}
        if model_name not in all_layer_cos:
            all_layer_cos[model_name] = {}

        for pert_type in pert_list:
            print(f"\n---- [{model_name}|{pert_type}] 子实验 A：整网精度扫描 ----")
            if pert_type == "nonlinearity":
                res = run_global_sensitivity(
                    model, test_loader, criterion, device,
                    model_name=model_name, output_subdir=output_subdir,
                    pert_type="nonlinearity",
                    alpha_list=alpha_list,
                    dataset=args.dataset,
                )
                all_global_results[model_name]["nonlinearity"] = res
            else:
                res = run_global_sensitivity(
                    model, test_loader, criterion, device,
                    model_name=model_name, output_subdir=output_subdir,
                    pert_type="gaussian",
                    noise_list=noise_list,
                    dataset=args.dataset,  
                )
                all_global_results[model_name]["gaussian"] = res

            print(f"\n---- [{model_name}|{pert_type}] 子实验 B：层偏移分析 ----")
            if pert_type == "nonlinearity":
                cos_map = run_layer_shift(
                    model, fixed_imgs, fixed_lbls, criterion, device,
                    model_name=model_name, output_subdir=output_subdir,
                    pert_type="nonlinearity",
                    alpha_list=alpha_list,
                    dataset=args.dataset, 
                )
                all_layer_cos[model_name]["nonlinearity"] = cos_map
            else:
                cos_map = run_layer_shift(
                    model, fixed_imgs, fixed_lbls, criterion, device,
                    model_name=model_name, output_subdir=output_subdir,
                    pert_type="gaussian",
                    noise_list=noise_list,
                    dataset=args.dataset, 
                )
                all_layer_cos[model_name]["gaussian"] = cos_map

        # Step 4：跨扰动类型对比（该模型）
        if ("nonlinearity" in all_global_results[model_name]
                and "gaussian" in all_global_results[model_name]):
            plot_accuracy_comparison_one_model(
                model_name,
                all_global_results[model_name]["nonlinearity"],
                all_global_results[model_name]["gaussian"],
                output_subdir,
            )
        if ("nonlinearity" in all_layer_cos[model_name]
                and "gaussian" in all_layer_cos[model_name]):
            plot_layer_accumulation_comparison_one_model(
                model_name,
                all_layer_cos[model_name]["nonlinearity"],
                all_layer_cos[model_name]["gaussian"],
                output_subdir,
                target_alpha=0.2,
                target_sigma=0.15,
            )

    # Step 5：跨模型鲁棒性迁移对比（若同时有 simple_cnn & exp2）
    if len({"simple_cnn", "exp2"}.intersection(set(model_list))) >= 2:
        print("\n---- Step 5：跨模型鲁棒性迁移对比 ----")
        plot_robustness_transfer(all_global_results, args.output_dir)

    # Step 6：终端中文摘要
    print("\n" + "=" * 80)
    print("【拓展研究2：终端中文摘要】")
    print("=" * 80)

    for model_name in model_list:
        print(f"\n- 模型: {model_name}")
        known_clean = None
        try:
            _, kc = get_model_for_ext2(model_name, device, dataset=args.dataset)
            known_clean = kc
        except Exception:
            pass
        if known_clean is not None:
            print(f"  干净基线准确率 (已知): {known_clean:.2f}%")

        for pert_type in pert_list:
            if pert_type not in all_global_results.get(model_name, {}):
                continue
            res = all_global_results[model_name][pert_type]
            # 找 clean 点
            clean_acc = None
            for p, acc, _ in res:
                if abs(float(p) - 0.0) < 1e-9:
                    clean_acc = acc
                    break
            if clean_acc is None:
                clean_acc = res[0][1] if len(res) > 0 else None
            max_drop = 0.0
            max_drop_p = None
            for p, acc, _ in res:
                d = (clean_acc - acc) if clean_acc is not None else 0.0
                if d > max_drop:
                    max_drop = d
                    max_drop_p = p
            pt_label = "非线性失真" if pert_type == "nonlinearity" else "高斯噪声"
            print(f"  · 扰动: {pt_label}")
            if clean_acc is not None:
                print(f"    干净(扰动=0)准确率: {clean_acc:.2f}%")
            if max_drop_p is not None:
                pu = "α" if pert_type == "nonlinearity" else "σ"
                print(f"    最大精度跌幅: {max_drop:.2f}%  ({pu}={max_drop_p})")

            # 简单精度表（前 3 行 + 后 3 行）
            header = f"    {'α' if pert_type=='nonlinearity' else 'σ':>6} | {'Accuracy':>10} | {'Loss':>10}"
            print(header)
            print("    " + "-" * 36)
            shown = min(7, len(res))
            for p, acc, loss in res[:shown]:
                print(f"    {float(p):>+6.3f} | {acc:>9.2f}% | {loss:>10.4f}")

    # 层误差累积最敏感层
    for model_name in model_list:
        if model_name not in all_layer_cos:
            continue
        for pert_type in pert_list:
            if pert_type not in all_layer_cos[model_name]:
                continue
            cos_map = all_layer_cos[model_name][pert_type]
            # 取强度最大参数的 cos 值
            keys = sorted(cos_map.keys())
            if not keys:
                continue
            strongest = keys[-1]
            layer_cos = cos_map[strongest]
            # 找 cos 最小（最敏感）层
            most_sensitive = min(layer_cos, key=lambda ln: layer_cos[ln])
            least_sensitive = max(layer_cos, key=lambda ln: layer_cos[ln])
            pt_label = "非线性失真" if pert_type == "nonlinearity" else "高斯噪声"
            print(f"\n- [{model_name}|{pt_label}] 强度={strongest} 层敏感性对比:")
            print(f"    最敏感层 (最低余弦相似度): {most_sensitive} -> {layer_cos[most_sensitive]:.4f}")
            print(f"    最稳健层 (最高余弦相似度): {least_sensitive} -> {layer_cos[least_sensitive]:.4f}")

    # Exp2 迁移性（高斯噪声下 Exp2 vs SimpleCNN 的精度提升）
    if "simple_cnn" in all_global_results and "exp2" in all_global_results:
        if "gaussian" in all_global_results["simple_cnn"] and "gaussian" in all_global_results["exp2"]:
            print("\n- Exp2 鲁棒性迁移效果（高斯噪声）:")
            s_rows = all_global_results["simple_cnn"]["gaussian"]
            e_rows = all_global_results["exp2"]["gaussian"]
            s_dict = {p: acc for p, acc, _ in s_rows}
            e_dict = {p: acc for p, acc, _ in e_rows}
            common_p = sorted(set(s_dict.keys()) & set(e_dict.keys()))
            print(f"    {'σ':>6} | {'SimpleCNN':>11} | {'Exp2':>9} | {'Uplift':>8}")
            print("    " + "-" * 44)
            for p in common_p:
                uplift = e_dict[p] - s_dict[p]
                print(f"    {p:>6.3f} | {s_dict[p]:>10.2f}% | {e_dict[p]:>8.2f}% | {uplift:>+7.2f}%")

    print(f"\n所有输出文件位于: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
