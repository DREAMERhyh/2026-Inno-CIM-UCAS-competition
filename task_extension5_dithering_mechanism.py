"""
拓展研究5：Dithering效应定量机理解析
task_extension5_dithering_mechanism.py

背景：
    前期实验（拓展3）发现 VGG-11 干净模型在 α=+0.3 非线性失真下，
    4bit QDQ 量化精度（43.27%）显著高于 8bit（36.74%），即"Dithering效应"。
    该效应为 VGG-11 所独有，SimpleCNN 和 ResNet-18 均未观察到。

研究目标：
    1. 组(a) 复现校验：确认 36.74%/43.27% 锚点可精确复现
    2. 组(b) 可控噪声注入：检验 H1（经典 dither 去相关假设）
    3. 组(c) 纯噪声替代量化：区分随机性（H1）与量化特异性（H2）
    4. 样本级翻转统计：检验"边缘样本救回"假说
    5. H2 轻量验证：激活分布形态与量化精度的关联
    6. 组(d) 三架构 Dithering 对照：检验 H3（MaxPool 交互假设）

注入顺序（严格遵循物理信号路径）：
    模拟域 MAC 非线性 → 噪声注入（可选）→ ADC 量化

运行命令：
    python task_extension5_dithering_mechanism.py
"""

import argparse
import csv
import os
import random
import sys
from collections import OrderedDict

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.data_loader import get_dataloaders
from utils.paths import get_ckpt_root, get_outputs_root, get_num_classes
from utils.nonlinearity import nonlinearity, register_nonlinearity_hooks, remove_hooks
from utils.quantization import quantize_error


# ======================================================================
# 常量
# ======================================================================

# 默认 α 值（固定 +0.3）
DEFAULT_ALPHA = 0.3

# 组(a) bits 扫描范围
DEFAULT_BITS = [2, 3, 4, 5, 6, 8]

# 组(b) 噪声强度扫描范围
DEFAULT_NOISE_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

# 噪声类型
DEFAULT_NOISE_TYPES = ["gaussian", "uniform"]

# 组(d) 三架构 MaxPool 次数（用于图注）
MAXPOOL_COUNTS = {
    "simple_cnn": 2,
    "vgg11": 5,
    "resnet18": 0,
}

# 组(d) 可复用的 bits 集合（extension3 已有数据）
EXT3_AVAILABLE_BITS = [2, 3, 4, 6, 8]

# 颜色映射
COLOR_8BIT = "#1f77b4"
COLOR_4BIT = "#d62728"
COLOR_CLEAN = "#2ca02c"
COLOR_NOISE = "#ff7f0e"
COLOR_GAUSSIAN = "#9467bd"
COLOR_UNIFORM = "#8c564b"


# ======================================================================
# 1. 基础工具函数
# ======================================================================

def set_seed(seed: int = 42):
    """设置 Python / NumPy / PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def create_model(model_name: str, num_classes: int = 10):
    """模型工厂，支持 simple_cnn / vgg11 / resnet18。"""
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "vgg11":
        from models.vgg11 import VGG11
        return VGG11(num_classes=num_classes)
    elif model_name == "resnet18":
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型名称: '{model_name}'")


def load_checkpoint(checkpoint_path: str, device: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_dataset = ckpt.get("dataset", "cifar10")
        print(f"  [Load] checkpoint 数据集来源: {ckpt_dataset}")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            best_acc = ckpt.get("best_test_acc", None)
        else:
            state_dict = ckpt
            best_acc = None
    else:
        state_dict = ckpt
        best_acc = None
    return state_dict, best_acc


# ======================================================================
# 2. 剂量累加器（跨层全局池化）
# ======================================================================

class DoseAccumulator:
    """
    跨层剂量累加器。

    对每个注册了钩子的层，累加：
        total_perturbation += Σ|out - y|
        total_signal       += Σ|y|
    其中 y = NL(x) 为非线性后的参考信号，out 为最终输出（经噪声/量化后）。

    最终剂量 = total_perturbation / (total_signal + eps)
    """

    def __init__(self):
        self.total_perturbation = 0.0
        self.total_signal = 0.0
        self.layer_count = 0

    def add(self, y: torch.Tensor, perturbation: torch.Tensor):
        """累加一个层的剂量贡献。"""
        self.total_perturbation += perturbation.abs().sum().item()
        self.total_signal += y.abs().sum().item()
        self.layer_count += 1

    def dose(self) -> float:
        """返回全局剂量 = Σ|perturbation| / Σ|y|。"""
        if self.total_signal < 1e-12:
            return 0.0
        return self.total_perturbation / self.total_signal


# ======================================================================
# 3. 三段式 Dithering 钩子注册
# ======================================================================

def generate_relative_noise(y: torch.Tensor, noise_std: float, noise_type: str):
    """
    生成逐样本相对噪声：noise = noise_std * ||y||_∞(per-sample) * z

    归一化参考系与量化一致（逐样本 ∞-范数），确保噪声与量化步长在相同参考系下。

    Args:
        y: 输入张量（非线性后的激活值）
        noise_std: 噪声强度（gaussian 为标准差，uniform 为半宽度）
        noise_type: "gaussian" 或 "uniform"

    Returns:
        与 y 同形状的噪声张量
    """
    # 逐样本 ∞-范数（兼容所有 PyTorch 版本，不支持 tuple 作为 dim）
    y_abs = y.abs()
    # 展平除 batch 维度外的所有维度
    y_flat = y_abs.view(y_abs.size(0), -1)
    per_sample_max = y_flat.max(dim=1, keepdim=True)[0]
    # 扩展回原形状，以便进行逐元素广播乘法
    per_sample_max = per_sample_max.view(per_sample_max.size(0), *([1] * (y.dim() - 1)))
    per_sample_max = torch.clamp(per_sample_max, min=1e-12)


    if noise_type == "gaussian":
        z = torch.randn_like(y)
    elif noise_type == "uniform":
        # uniform 的半宽度为 noise_std，z ~ U(-1, 1)
        z = torch.rand_like(y) * 2 - 1
    else:
        raise ValueError(f"不支持的噪声类型: '{noise_type}'")

    return noise_std * per_sample_max * z


def register_dithering_hooks(
    model: nn.Module,
    alpha: float = 0.0,
    num_bits: int = 8,
    noise_std: float = 0.0,
    noise_type: str = None,
    dose_accumulator: DoseAccumulator = None,
):
    """
    注册三段式 Dithering 钩子链（对所有 Conv2d / Linear 的 forward_pre_hook）。

    注入顺序（严格遵循物理信号路径）：
        1. 非线性失真 (NL) : y = α·x̃³ + (1-α)·x̃
        2. 噪声注入 (可选)  : y' = y + noise（逐样本相对噪声）
        3. QDQ 量化 (可选)  : out = quantize_error(y', num_bits)

    若 dose_accumulator 不为 None，则在每个钩子中记录剂量数据。

    Args:
        model: 待注入的模型
        alpha: 非线性强度
        num_bits: 量化比特数（>=32 时不量化）
        noise_std: 噪声强度（0 时不注入噪声）
        noise_type: 噪声类型（"gaussian" / "uniform"）
        dose_accumulator: 可选剂量累加器

    Returns:
        list[RemovableHandle]: 钩子句柄列表
    """
    hooks = []
    apply_noise = (noise_std > 0 and noise_type is not None)
    apply_quant = (num_bits < 32)

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        # 闭包工厂：快照当前参数，避免 Python 延迟绑定
        def make_hook(
            alpha_val=alpha,
            nb_val=num_bits,
            ns_val=noise_std,
            nt_val=noise_type,
            an=apply_noise,
            aq=apply_quant,
            acc=dose_accumulator,
        ):
            def pre_hook_fn(m, inputs):
                if not inputs or inputs[0] is None:
                    return inputs
                x = inputs[0]

                # Step 1: 非线性失真
                y = nonlinearity(x, alpha=alpha_val)

                # Step 2: 噪声注入（相对噪声）
                out = y
                if an:
                    noise = generate_relative_noise(y, ns_val, nt_val)
                    out = out + noise

                # Step 3: QDQ 量化
                if aq:
                    out = quantize_error(out, num_bits=nb_val)

                # 剂量追踪
                if acc is not None:
                    perturbation = out - y
                    acc.add(y, perturbation)

                return (out,) + tuple(inputs[1:])
            return pre_hook_fn

        h = module.register_forward_pre_hook(make_hook())
        hooks.append(h)

    return hooks


# ======================================================================
# 4. 评估函数
# ======================================================================

@torch.no_grad()
def evaluate_on_test(model, test_loader, criterion, device):
    """标准测试集推理，返回 (accuracy %, avg_loss)。"""
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


@torch.no_grad()
def evaluate_with_logits(model, test_loader, device):
    """推理并返回每个样本的 logits、标签、预测类别。"""
    model.eval()
    all_logits = []
    all_labels = []
    all_preds = []
    for images, labels in tqdm(test_loader, desc="Eval w/ Logits", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        _, preds = torch.max(logits, dim=1)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_preds.append(preds.cpu())
    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_preds, dim=0),
    )


@torch.no_grad()
def evaluate_with_dose(
    model, test_loader, criterion, device,
    alpha, num_bits, noise_std=0.0, noise_type=None,
):
    """
    推理并同时计算剂量 dose。

    Returns:
        (accuracy, avg_loss, dose)
    """
    accumulator = DoseAccumulator()
    hooks = register_dithering_hooks(
        model, alpha=alpha, num_bits=num_bits,
        noise_std=noise_std, noise_type=noise_type,
        dose_accumulator=accumulator,
    )
    try:
        acc, loss = evaluate_on_test(model, test_loader, criterion, device)
    finally:
        remove_hooks(hooks)
    return acc, loss, accumulator.dose()


# ======================================================================
# 5. 组(a)：复现校验
# ======================================================================

def run_group_a(model, test_loader, criterion, device, alpha, bits, output_dir, dataset="cifar10"):
    """组(a) 复现校验：扫描 bits ∈ {2,3,4,5,6,8}，检查 36.74%/43.27% 锚点。"""
    print("\n" + "=" * 66)
    print("  组(a)：复现校验 — α=+0.3 下 bits 扫描")
    print("=" * 66)

    # 锚点数据（仅 CIFAR-10 有效，CIFAR-100 清空）
    if dataset == "cifar10":
        ext3_anchors = {2: 9.98, 3: 42.97, 4: 43.27, 6: 37.13, 8: 36.74}
    else:
        print(f"[组(a)] 当前数据集为 {dataset}，锚点数值仅适用于 CIFAR-10，"
              f"跳过锚点比对。")
        ext3_anchors = {}

    results = []
    for nb in bits:
        acc, loss, dose = evaluate_with_dose(
            model, test_loader, criterion, device,
            alpha=alpha, num_bits=nb, noise_std=0.0,
        )
        results.append({
            "num_bits": nb,
            "accuracy": round(acc, 4),
            "loss": round(loss, 6),
            "dose": round(dose, 6),
        })

        # 锚点校验
        check = ""
        if nb in ext3_anchors:
            diff = abs(acc - ext3_anchors[nb])
            if diff <= 0.1:
                check = " ✅ (match ext3, diff={:.4f})".format(diff)
            else:
                check = " ⚠️ (diff={:.4f} > 0.1, ext3={:.2f})".format(diff, ext3_anchors[nb])

        print(f"    bit={nb:>2}  Acc={acc:.2f}%  Loss={loss:.4f}  Dose={dose:.6f}{check}")

    # 保存 CSV
    csv_path = os.path.join(output_dir, "group_a_bits_scan.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["num_bits", "accuracy", "loss", "dose"])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"    [Save] {csv_path}")

    return results


# ======================================================================
# 6. 组(b)：可控噪声注入
# ======================================================================

def run_group_b(model, test_loader, criterion, device, alpha, noise_types, noise_levels, output_dir):
    """
    组(b) 可控噪声注入：α=+0.3 + 8bit QDQ + 量化前注入相对噪声。

    噪声类型：高斯 / 均匀
    噪声强度：σ ∈ {0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}
    """
    print("\n" + "=" * 66)
    print("  组(b)：可控噪声注入 — α=+0.3 + 8bit QDQ + 噪声")
    print("=" * 66)

    all_results = []
    for nt in noise_types:
        for sigma in noise_levels:
            acc, loss, dose = evaluate_with_dose(
                model, test_loader, criterion, device,
                alpha=alpha, num_bits=8,
                noise_std=sigma, noise_type=nt,
            )
            all_results.append({
                "noise_type": nt,
                "noise_std": sigma,
                "accuracy": round(acc, 4),
                "loss": round(loss, 6),
                "dose": round(dose, 6),
            })
            print(f"    {nt:>8} σ={sigma:>6.3f}  Acc={acc:.2f}%  Loss={loss:.4f}  Dose={dose:.6f}")

    # 保存 CSV
    csv_path = os.path.join(output_dir, "group_b_noise_injection.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["noise_type", "noise_std", "accuracy", "loss", "dose"])
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"    [Save] {csv_path}")

    return all_results


# ======================================================================
# 7. 组(c)：纯噪声替代量化
# ======================================================================

def run_group_c(model, test_loader, criterion, device, alpha, noise_types, group_a_results, output_dir):
    """
    组(c) 纯噪声替代量化：α=+0.3 + 纯相对噪声（无量化）。

    噪声强度与 4bit QDQ 等效剂量对齐。
    相对噪声 dose 对 σ 严格线性，先在 σ=0.1 下实测 dose，按比例换算。
    """
    print("\n" + "=" * 66)
    print("  组(c)：纯噪声替代量化")
    print("=" * 66)

    # 获取 4bit QDQ 的剂量
    dose_4bit = None
    for r in group_a_results:
        if r["num_bits"] == 4:
            dose_4bit = r["dose"]
            break
    if dose_4bit is None:
        print("  [ERROR] 未找到 4bit QDQ 的剂量数据，无法对齐")
        return [], None
    print(f"    4bit QDQ dose = {dose_4bit:.6f}")

    all_results = []
    for nt in noise_types:
        # 在 σ=0.1 下实测 dose
        test_sigma = 0.1
        _, _, test_dose = evaluate_with_dose(
            model, test_loader, criterion, device,
            alpha=alpha, num_bits=32,  # 无量化
            noise_std=test_sigma, noise_type=nt,
        )
        # 按比例换算：dose ∝ σ，所以 target_σ = dose_4bit / test_dose * test_sigma
        if test_dose > 1e-12:
            target_sigma = dose_4bit / test_dose * test_sigma
        else:
            target_sigma = 0.0

        # 在目标 σ 下评估
        acc, loss, actual_dose = evaluate_with_dose(
            model, test_loader, criterion, device,
            alpha=alpha, num_bits=32,  # 无量化
            noise_std=target_sigma, noise_type=nt,
        )
        all_results.append({
            "noise_type": nt,
            "noise_std": round(target_sigma, 6),
            "accuracy": round(acc, 4),
            "loss": round(loss, 6),
            "dose": round(actual_dose, 6),
            "target_dose": round(dose_4bit, 6),
        })
        print(f"    {nt:>8} σ_eq={target_sigma:.6f}  Acc={acc:.2f}%  Dose={actual_dose:.6f}  (target={dose_4bit:.6f})")

    # 保存 CSV
    csv_path = os.path.join(output_dir, "group_c_pure_noise.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["noise_type", "noise_std", "accuracy", "loss", "dose", "target_dose"])
        w.writeheader()
        for r in all_results:
            w.writerow(r)
    print(f"    [Save] {csv_path}")

    return all_results, dose_4bit


# ======================================================================
# 8. 样本级翻转统计
# ======================================================================

def compute_margin(logits: torch.Tensor, labels: torch.Tensor):
    """
    margin = logit(true_class) - max(logit(other_classes))

    margin > 0: 正确分类
    margin < 0: 错误分类（绝对值越大越确信地错）
    """
    batch_size = logits.size(0)
    true_logits = logits[range(batch_size), labels]
    # 屏蔽真实类
    masked = logits.clone()
    masked[range(batch_size), labels] = -float("inf")
    max_other, _ = torch.max(masked, dim=1)
    return true_logits - max_other


def run_sample_level_stats(
    model, test_loader, device, alpha,
    group_b_results, output_dir,
):
    """
    样本级翻转统计。

    参照系设计（核心修正）：
        - 主参照 = 8bit_QDQ（衡量各条件相对8bit的净翻转，即Dithering效应）
        - 附加参照 = clean/α=0（展示总损伤，不对主统计产生耦合）

    主统计锚点：4bit行的 saved - killed 应 ≈ +653（+6.53pp × 10000样本）
    noise行的 net_gain 即为噪声注入的净翻转（H1的样本级证据）。
    """
    print("\n" + "=" * 66)
    print("  样本级翻转统计")
    print("=" * 66)

    # ---- 第一步：获取干净预测（无任何失真，仅用于附加列） ----
    print("  [Step 1] 干净推理（α=0, 无噪声, 无量化）...")
    clean_logits, clean_labels, clean_preds = evaluate_with_logits(
        model, test_loader, device,
    )
    clean_acc = (clean_preds == clean_labels).float().mean().item() * 100
    print(f"            Clean Acc: {clean_acc:.2f}%")

    # ---- 第二步：定义要比较的条件 ----
    conditions = [
        {"name": "8bit_QDQ", "alpha": alpha, "num_bits": 8, "noise_std": 0.0, "noise_type": None},
        {"name": "4bit_QDQ", "alpha": alpha, "num_bits": 4, "noise_std": 0.0, "noise_type": None},
    ]

    # 从组(b)结果中找出最优 σ（精度最高的噪声组合）
    best_gain = {}
    for r in group_b_results:
        nt = r["noise_type"]
        if nt not in best_gain or r["accuracy"] > best_gain[nt]["accuracy"]:
            best_gain[nt] = r

    for nt, best in best_gain.items():
        conditions.append({
            "name": f"8bit+{nt}_opt",
            "alpha": alpha,
            "num_bits": 8,
            "noise_std": best["noise_std"],
            "noise_type": nt,
            "opt_acc": best["accuracy"],
        })

    # ---- 第三步：运行各条件推理 ----
    all_cond_data = {}
    for cond in conditions:
        print(f"  [Step 2] 条件: {cond['name']} ...")
        hooks = register_dithering_hooks(
            model, alpha=cond["alpha"], num_bits=cond["num_bits"],
            noise_std=cond["noise_std"], noise_type=cond.get("noise_type"),
        )
        try:
            logits, labels, preds = evaluate_with_logits(model, test_loader, device)
        finally:
            remove_hooks(hooks)

        margin = compute_margin(logits, labels)
        all_cond_data[cond["name"]] = {
            "logits": logits,
            "preds": preds,
            "margin": margin,
            "acc": (preds == labels).float().mean().item() * 100,
        }
        print(f"            Acc: {all_cond_data[cond['name']]['acc']:.2f}%")

    # ---- 第四步：翻转统计（主参照 = 8bit_QDQ） ----
    ref_cond = "8bit_QDQ"
    ref_data = all_cond_data.get(ref_cond)
    if ref_data is None:
        print("  [ERROR] 8bit_QDQ 数据缺失，无法计算翻转统计")
        return all_cond_data, clean_preds, clean_labels

    ref_preds = ref_data["preds"]
    ref_correct = (ref_preds == clean_labels)
    clean_correct = (clean_preds == clean_labels)

    flip_rows = []
    n_samples = len(clean_labels)

    for cond_name, cond_data in all_cond_data.items():
        if cond_name == ref_cond:
            continue

        cond_preds = cond_data["preds"]
        cond_correct = (cond_preds == clean_labels)

        # ---- 主统计：相对 8bit_QDQ ----
        saved_vs_ref = (~ref_correct & cond_correct).sum().item()
        killed_vs_ref = (ref_correct & ~cond_correct).sum().item()
        net_gain_vs_ref = saved_vs_ref - killed_vs_ref

        # ---- 附加统计：相对 clean（展示总损伤） ----
        saved_vs_clean = (~clean_correct & cond_correct).sum().item()
        killed_vs_clean = (clean_correct & ~cond_correct).sum().item()
        net_gain_vs_clean = saved_vs_clean - killed_vs_clean

        always_correct = (ref_correct & cond_correct).sum().item()
        always_wrong = (~ref_correct & ~cond_correct).sum().item()

        flip_rows.append({
            "condition": cond_name,
            "accuracy": round(cond_data["acc"], 4),
            # 主统计（相对 8bit_QDQ）
            "saved_vs_8bit": saved_vs_ref,
            "killed_vs_8bit": killed_vs_ref,
            "net_gain_vs_8bit": net_gain_vs_ref,
            # 附加统计（相对 clean）
            "saved_vs_clean": saved_vs_clean,
            "killed_vs_clean": killed_vs_clean,
            "net_gain_vs_clean": net_gain_vs_clean,
            # 公共
            "always_correct": always_correct,
            "always_wrong": always_wrong,
            "total_samples": n_samples,
        })
        print(f"  {cond_name:>20}: vs_8bit saved={saved_vs_ref:>4} killed={killed_vs_ref:>4} "
              f"net={net_gain_vs_ref:+d}  |  vs_clean net={net_gain_vs_clean:+d}")

    # 保存翻转统计 CSV
    csv_path = os.path.join(output_dir, "flip_statistics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "condition", "accuracy",
            "saved_vs_8bit", "killed_vs_8bit", "net_gain_vs_8bit",
            "saved_vs_clean", "killed_vs_clean", "net_gain_vs_clean",
            "always_correct", "always_wrong", "total_samples",
        ])
        w.writeheader()
        for r in flip_rows:
            w.writerow(r)
    print(f"  [Save] {csv_path}")

    # ---- 第五步：margin 分布分析（参照系 = 8bit_QDQ） ----
    # 分组 mask 基于 8bit 的预测正确/错误
    if "4bit_QDQ" in all_cond_data:
        fourbit_preds = all_cond_data["4bit_QDQ"]["preds"]
        ref_margin = ref_data["margin"]

        saved_mask = (~ref_correct) & (fourbit_preds == clean_labels)
        killed_mask = ref_correct & (fourbit_preds != clean_labels)
        always_wrong_mask = (~ref_correct) & (fourbit_preds != clean_labels)

        # 保存 margin 数据用于绘图
        margin_data = {
            "saved": ref_margin[saved_mask].numpy(),
            "killed": ref_margin[killed_mask].numpy(),
            "always_wrong": ref_margin[always_wrong_mask].numpy(),
        }
        np.savez(
            os.path.join(output_dir, "margin_data.npz"),
            **margin_data,
        )
        print(f"  [Save] margin_data.npz (saved={len(margin_data['saved'])}, "
              f"killed={len(margin_data['killed'])}, "
              f"always_wrong={len(margin_data['always_wrong'])})")

    return all_cond_data, clean_preds, clean_labels


# ======================================================================
# 9. H2：激活统计量
# ======================================================================

@torch.no_grad()
def run_h2_activation_stats(model, test_loader, device, alpha, bits, output_dir):
    """
    H2 轻量验证：统计 α=+0.3 下不同 bits 时的激活分布形态。

    双观测点：
        - 主观测点：分类器输入（VGG-11: avgpool 输出，512-dim 特征）
        - 次观测点：第一个卷积层输出

    统计量：均值 / 标准差 / 峰度（kurtosis）
    包含 α=0 干净参照行。
    """
    print("\n" + "=" * 66)
    print("  H2：激活统计量")
    print("=" * 66)

    # 注册 forward hook 捕获中间层输出
    activation_store = {}

    def make_forward_hook(name):
        def hook(m, inp, out):
            activation_store[name] = out.detach().cpu()
        return hook

    # 主观测点：分类器输入（avgpool 输出）
    if hasattr(model, "avgpool"):
        handle_avgpool = model.avgpool.register_forward_hook(make_forward_hook("classifier_input"))
    else:
        # 防御性回退：捕获 features 最后一层的输出（VGG-11 一定有 avgpool，此处为通用性）
        print("  [WARNING] model.avgpool 不存在，回退到 model.features[-1] 作为观测点")
        handle_avgpool = model.features[-1].register_forward_hook(make_forward_hook("classifier_input"))
    # 次观测点：第一个卷积层输出（VGG-11 的 features[0] 安全）
    handle_conv1 = model.features[0].register_forward_hook(make_forward_hook("conv1_output"))

    results = []
    # 包含 α=0 干净参照行
    scan_configs = [(0.0, 32, "clean")]
    for nb in bits:
        scan_configs.append((alpha, nb, f"bit={nb}"))

    for scan_alpha, scan_bits, label in scan_configs:
        activation_store.clear()
        hooks = register_dithering_hooks(
            model, alpha=scan_alpha, num_bits=scan_bits,
            noise_std=0.0, noise_type=None,
        )
        try:
            # 只需跑一个 batch 获取激活分布统计量
            for images, _ in test_loader:
                images = images.to(device)
                _ = model(images)
                break  # 只跑一个 batch
        finally:
            remove_hooks(hooks)

        # 提取统计量
        row = {"condition": label, "alpha": scan_alpha, "num_bits": scan_bits}

        for obs_name in ["conv1_output", "classifier_input"]:
            if obs_name not in activation_store:
                continue
            acts = activation_store[obs_name].float()
            # 展平所有样本和通道，计算全局统计量
            flat = acts.view(-1)
            row[f"{obs_name}_mean"] = round(flat.mean().item(), 6)
            row[f"{obs_name}_std"] = round(flat.std().item(), 6)
            # 峰度（Fisher 定义：正态分布峰度 = 0）
            if flat.numel() > 3:
                kurt = (flat - flat.mean()).pow(4).mean() / (flat.std().pow(4) + 1e-12) - 3
                row[f"{obs_name}_kurtosis"] = round(kurt.item(), 4)
            else:
                row[f"{obs_name}_kurtosis"] = float("nan")

        results.append(row)
        print(f"    {label:>12}: conv1_mean={row.get('conv1_output_mean', 'N/A'):>10}  "
              f"classif_mean={row.get('classifier_input_mean', 'N/A'):>10}")

    # 清理钩子
    handle_avgpool.remove()
    handle_conv1.remove()

    # 保存 CSV
    csv_path = os.path.join(output_dir, "activation_stats.csv")
    fieldnames = [
        "condition", "alpha", "num_bits",
        "conv1_output_mean", "conv1_output_std", "conv1_output_kurtosis",
        "classifier_input_mean", "classifier_input_std", "classifier_input_kurtosis",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"  [Save] {csv_path}")

    return results


# ======================================================================
# 10. 组(d)：三架构 Dithering 对照
# ======================================================================

def run_group_d(output_dir, ext3_root=None, dataset="cifar10"):
    """
    组(d) 三架构 Dithering 对照。

    从 extension3 现有 CSV 读取 α=+0.3 下的 bits 精度数据，
    绘制三架构叠加图并生成汇总表。
    """
    ext3_actual_root = ext3_root if ext3_root is not None else get_outputs_root(dataset)
    print("\n" + "=" * 66)
    print("  组(d)：三架构 Dithering 对照")
    print("=" * 66)

    arch_configs = [
        {
            "name": "simple_cnn",
            "display": "SimpleCNN",
            "maxpool": 2,
            "csv_path": os.path.join(ext3_actual_root, "extension3_simplecnn", "simple_cnn_joint_error.csv"),
        },
        {
            "name": "vgg11",
            "display": "VGG-11",
            "maxpool": 5,
            "csv_path": os.path.join(ext3_actual_root, "extension3_vgg11", "vgg11_joint_error.csv"),
        },
        {
            "name": "resnet18",
            "display": "ResNet-18",
            "maxpool": 0,
            "csv_path": os.path.join(ext3_actual_root, "extension3_resnet18", "resnet18_joint_error.csv"),
        },
    ]

    # 颜色映射
    arch_colors = {
        "simple_cnn": "#1f77b4",
        "vgg11": "#d62728",
        "resnet18": "#2ca02c",
    }
    arch_markers = {
        "simple_cnn": "o",
        "vgg11": "s",
        "resnet18": "^",
    }

    # 读取数据
    arch_data = {}
    for cfg in arch_configs:
        csv_path = cfg["csv_path"]
        if not os.path.exists(csv_path):
            print(f"  [WARNING] CSV 不存在: {csv_path}")
            continue

        # 读取 CSV 并过滤 α=+0.3
        points = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if abs(float(row["alpha"]) - 0.3) < 1e-9:
                    nb = int(row["num_bits"])
                    if nb in EXT3_AVAILABLE_BITS:
                        points.append({
                            "num_bits": nb,
                            "accuracy": float(row["accuracy"]),
                            "loss": float(row["loss"]),
                        })

        if not points:
            print(f"  [WARNING] {cfg['name']}: 无 α=+0.3 数据")
            continue

        # 按 bits 升序排列
        points.sort(key=lambda p: p["num_bits"])
        arch_data[cfg["name"]] = {
            "display": cfg["display"],
            "maxpool": cfg["maxpool"],
            "points": points,
            "color": arch_colors.get(cfg["name"], "#333"),
            "marker": arch_markers.get(cfg["name"], "o"),
        }
        print(f"  {cfg['display']:>12} (MaxPool={cfg['maxpool']}): "
              f"{[(p['num_bits'], p['accuracy']) for p in points]}")

    # 生成汇总表 CSV
    summary_rows = []
    for arch_name, data in arch_data.items():
        bits_map = {p["num_bits"]: p["accuracy"] for p in data["points"]}
        acc_8bit = bits_map.get(8, None)
        acc_4bit = bits_map.get(4, None)
        gain = (acc_4bit - acc_8bit) if (acc_8bit is not None and acc_4bit is not None) else None
        row = {
            "model": arch_name,
            "maxpool_count": data["maxpool"],
        }
        for b in EXT3_AVAILABLE_BITS:
            row[f"bits_{b}"] = round(bits_map.get(b, float("nan")), 4) if b in bits_map else float("nan")
        row["gain_4bit_vs_8bit"] = round(gain, 4) if gain is not None else float("nan")
        summary_rows.append(row)

    csv_path = os.path.join(output_dir, "arch_dithering_summary.csv")
    fieldnames = ["model", "maxpool_count"] + [f"bits_{b}" for b in EXT3_AVAILABLE_BITS] + ["gain_4bit_vs_8bit"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"  [Save] {csv_path}")

    return arch_data


# ======================================================================
# 11. 绘图函数
# ======================================================================

def plot_bits_accuracy(group_a_results, output_dir, alpha):
    """绘制组(a) bits-精度曲线，标注 36.74/43.27 锚点。"""
    bits = [r["num_bits"] for r in group_a_results]
    accs = [r["accuracy"] for r in group_a_results]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    ax.plot(bits, accs, marker="o", markersize=8, linewidth=2.5, color=COLOR_8BIT)
    ax.scatter(bits, accs, color=COLOR_8BIT, s=60, zorder=5)

    # 标注锚点
    for r in group_a_results:
        if r["num_bits"] in (4, 8):
            ax.annotate(
                f"{r['accuracy']:.2f}%",
                (r["num_bits"], r["accuracy"]),
                xytext=(0, 15 if r["num_bits"] == 4 else -15),
                textcoords="offset points",
                fontsize=11, ha="center",
                fontweight="bold",
                color=COLOR_4BIT if r["num_bits"] == 4 else COLOR_8BIT,
            )

    # 标注 Dithering 增益箭头
    acc_8 = next(r["accuracy"] for r in group_a_results if r["num_bits"] == 8)
    acc_4 = next(r["accuracy"] for r in group_a_results if r["num_bits"] == 4)
    ax.annotate(
        f"Dithering Gain\n+{acc_4 - acc_8:.2f}%",
        xy=(5, (acc_4 + acc_8) / 2),
        fontsize=10, color=COLOR_4BIT, fontweight="bold",
        ha="center",
    )

    ax.set_xlabel("Number of Bits", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(f"VGG-11: Accuracy vs Bits (α={alpha:+.2f})", fontsize=14, fontweight="bold")
    ax.set_xticks(bits)
    ax.set_xticklabels([str(b) for b in bits])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = os.path.join(output_dir, "bits_accuracy_curve.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] bits_accuracy_curve.png saved: {png_path}")


def plot_dose_accuracy(group_a_results, group_b_results, group_c_results, dose_4bit, output_dir):
    """
    绘制组(a/b/c) 同轴剂量-精度曲线。

    组(a)：不同 bits 的点（标注 bit 值）
    组(b)：高斯 / 均匀噪声的连续曲线
    组(c)：纯噪声（等效 4bit 剂量）的单点
    """
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)

    # ---- 组(a)：bits 点 ----
    for r in group_a_results:
        nb = r["num_bits"]
        marker_style = "o" if nb == 8 else ("s" if nb == 4 else "^")
        color = COLOR_4BIT if nb == 4 else (COLOR_8BIT if nb == 8 else "#7f7f7f")
        size = 120 if nb in (4, 8) else 60
        ax.scatter(
            r["dose"], r["accuracy"], marker=marker_style, s=size,
            color=color, zorder=6, edgecolors="black", linewidths=0.5,
        )
        ax.annotate(
            f"{nb}bit" if nb in (4, 8) else f"{nb}",
            (r["dose"], r["accuracy"]),
            xytext=(8, 5), textcoords="offset points", fontsize=9,
            color=color, fontweight="bold" if nb in (4, 8) else "normal",
        )

    # 标注组(a) 8bit 基线
    acc_8bit = next(r["accuracy"] for r in group_a_results if r["num_bits"] == 8)
    ax.axhline(acc_8bit, linestyle=":", color=COLOR_8BIT, alpha=0.5, linewidth=1.0)
    ax.text(ax.get_xlim()[1] * 0.98, acc_8bit, f"8bit baseline ({acc_8bit:.2f}%)",
            fontsize=9, color=COLOR_8BIT, ha="right", va="bottom")

    # ---- 组(b)：噪声曲线 ----
    for nt in set(r["noise_type"] for r in group_b_results):
        nt_data = [r for r in group_b_results if r["noise_type"] == nt]
        nt_data.sort(key=lambda r: r["dose"])
        doses = [r["dose"] for r in nt_data]
        accs = [r["accuracy"] for r in nt_data]
        color = COLOR_GAUSSIAN if nt == "gaussian" else COLOR_UNIFORM
        label = f"8bit + {nt} noise"
        ax.plot(doses, accs, marker=".", linewidth=1.8, color=color, label=label, alpha=0.8)

        # 标注最优 σ 点
        best_idx = max(range(len(accs)), key=lambda i: accs[i])
        ax.scatter(
            doses[best_idx], accs[best_idx], marker="*", s=180,
            color=color, zorder=7, edgecolors="black", linewidths=0.8,
        )
        ax.annotate(
            f"σ={nt_data[best_idx]['noise_std']:.3f}",
            (doses[best_idx], accs[best_idx]),
            xytext=(10, -15), textcoords="offset points", fontsize=8,
            color=color,
        )

    # ---- 组(c)：纯噪声点 ----
    if group_c_results:
        for r in group_c_results:
            nt = r["noise_type"]
            color = COLOR_GAUSSIAN if nt == "gaussian" else COLOR_UNIFORM
            ax.scatter(
                r["dose"], r["accuracy"], marker="D", s=130,
                color=color, zorder=7, edgecolors="black", linewidths=1.0,
            )
            ax.annotate(
                f"pure {nt} noise",
                (r["dose"], r["accuracy"]),
                xytext=(10, -15), textcoords="offset points", fontsize=8,
                color=color, fontweight="bold",
            )

    # 4bit 剂量参考竖线
    if dose_4bit is not None:
        ax.axvline(dose_4bit, linestyle="--", color=COLOR_4BIT, alpha=0.4, linewidth=1.0)
        ax.text(dose_4bit, ax.get_ylim()[0] + 5, f"4bit dose={dose_4bit:.4f}",
                fontsize=8, color=COLOR_4BIT, ha="center", rotation=90)

    ax.set_xlabel("Dose (E[|perturbation|] / E[|y|])", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Dose-Accuracy Curve: Dithering Effect Analysis", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    png_path = os.path.join(output_dir, "dose_accuracy_curve.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] dose_accuracy_curve.png saved: {png_path}")


def plot_margin_distribution(output_dir):
    """
    绘制三组样本的 margin 分布叠加图。

    从 margin_data.npz 读取数据。
    """
    npz_path = os.path.join(output_dir, "margin_data.npz")
    if not os.path.exists(npz_path):
        print(f"[WARNING] margin_data.npz 不存在，跳过 margin 分布图")
        return

    data = np.load(npz_path)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    if "saved" in data and len(data["saved"]) > 0:
        ax.hist(data["saved"], bins=50, alpha=0.6, color="#2ca02c",
                label=f"Saved (wrong→correct, n={len(data['saved'])})", density=True)
    if "killed" in data and len(data["killed"]) > 0:
        ax.hist(data["killed"], bins=50, alpha=0.6, color="#d62728",
                label=f"Killed (correct→wrong, n={len(data['killed'])})", density=True)
    if "always_wrong" in data and len(data["always_wrong"]) > 0:
        ax.hist(data["always_wrong"], bins=50, alpha=0.6, color="#7f7f7f",
                label=f"Always Wrong (n={len(data['always_wrong'])})", density=True)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Margin (8bit QDQ condition)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Margin Distribution under 8bit QDQ (α=+0.3)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    png_path = os.path.join(output_dir, "margin_distribution.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] margin_distribution.png saved: {png_path}")


def plot_arch_comparison(arch_data, output_dir):
    """
    绘制组(d) 三架构 bits-精度叠加图。

    图例标注各架构 MaxPool 次数。
    """
    if not arch_data:
        print("[WARNING] 无架构数据，跳过组(d) 绘图")
        return

    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)

    for arch_name, data in arch_data.items():
        points = sorted(data["points"], key=lambda p: p["num_bits"])
        bits = [p["num_bits"] for p in points]
        accs = [p["accuracy"] for p in points]
        ax.plot(
            bits, accs, marker=data["marker"], markersize=8,
            linewidth=2.2, color=data["color"],
            label=f"{data['display']} (MaxPool={data['maxpool']})",
        )

        # 标注 Dithering 增益
        acc_8 = next((p["accuracy"] for p in points if p["num_bits"] == 8), None)
        acc_4 = next((p["accuracy"] for p in points if p["num_bits"] == 4), None)
        if acc_8 is not None and acc_4 is not None:
            gain = acc_4 - acc_8
            ax.annotate(
                f"{gain:+.2f}%",
                xy=(4.5, (acc_4 + acc_8) / 2),
                fontsize=9, color=data["color"], fontweight="bold",
                ha="center",
            )

    ax.set_xlabel("Number of Bits", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Dithering Effect Across Architectures (α=+0.3)", fontsize=14, fontweight="bold")
    ax.set_xticks(EXT3_AVAILABLE_BITS)
    ax.set_xticklabels([str(b) for b in EXT3_AVAILABLE_BITS])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    # 图注说明 bits 集合
    fig.text(
        0.5, -0.02,
        f"Bits scanned: {EXT3_AVAILABLE_BITS} (data from Extension 3 CSVs). "
        f"ResNet-18 has no MaxPool; VGG-11 has the most MaxPool layers.",
        ha="center", fontsize=9, color="gray",
    )
    plt.tight_layout()
    png_path = os.path.join(output_dir, "arch_comparison_bits.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] arch_comparison_bits.png saved: {png_path}")


# ======================================================================
# 12. 命令行参数
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extension 5: Dithering 效应定量机理解析"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100"],
        help="数据集，默认: cifar10",
    )
    parser.add_argument(
        "--model", type=str, default="vgg11",
        help="主分析模型名称（默认: vgg11）",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="固定非线性强度 α（默认: 0.3）",
    )
    parser.add_argument(
        "--bits", type=str,
        default=",".join(str(b) for b in DEFAULT_BITS),
        help="逗号分隔的 bits 扫描范围（默认: 2,3,4,5,6,8）",
    )
    parser.add_argument(
        "--noise_types", type=str,
        default=",".join(DEFAULT_NOISE_TYPES),
        help="逗号分隔的噪声类型（默认: gaussian,uniform）",
    )
    parser.add_argument(
        "--noise_levels", type=str,
        default=",".join(str(s) for s in DEFAULT_NOISE_LEVELS),
        help="逗号分隔的噪声强度列表（默认: 0.005,0.01,...,0.5）",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录，默认 {outputs_root}/extension5_dithering_{model}/",
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
        "--checkpoint_root", type=str, default=None,
        help="checkpoint 根目录，默认 {ckpt_root}",
    )
    parser.add_argument(
        "--ext3_root", type=str, default=None,
        help="extension3 输出根目录，组(d) 读取用（默认: ./outputs）",
    )
    return parser.parse_args()


# ======================================================================
# 13. 主函数
# ======================================================================

def main():
    args = parse_args()

    # ---- Step 0: 准备 ----
    set_seed(args.seed)

    model_name = args.model
    model_tag = model_name.replace("_", "")
    if args.checkpoint_root is None:
        args.checkpoint_root = get_ckpt_root(args.dataset)
    if args.output_dir is None:
        args.output_dir = os.path.join(get_outputs_root(args.dataset), f"extension5_dithering_{model_tag}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Extension5] 数据集: {args.dataset}, 类别数: {get_num_classes(args.dataset)}")
    print(f"[Extension5] 输出目录: {os.path.abspath(args.output_dir)}")

    model_name = args.model
    alpha = args.alpha
    bits = sorted([int(b.strip()) for b in args.bits.split(",") if b.strip()])
    noise_types = [nt.strip() for nt in args.noise_types.split(",") if nt.strip()]
    noise_levels = sorted([float(s.strip()) for s in args.noise_levels.split(",") if s.strip()])

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[Extension5] 主分析模型: {model_name}")
    print(f"[Extension5] α = {alpha:+.2f}")
    print(f"[Extension5] bits = {bits}")
    print(f"[Extension5] 噪声类型 = {noise_types}")
    print(f"[Extension5] 噪声强度 = {noise_levels}")
    print(f"[Extension5] 设备 = {device}")

    # ---- Step 1: 加载数据 ----
    print("\n" + "-" * 60)
    print(f"[Extension5] 加载 {args.dataset.upper()} 测试集 ...")
    _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2, dataset=args.dataset)
    criterion = nn.CrossEntropyLoss()

    # ---- Step 2: 加载模型 ----
    print("\n" + "-" * 60)
    print(f"[Extension5] 加载模型: {model_name}")
    checkpoint_path = os.path.join(args.checkpoint_root, model_name, "best_model.pth")
    print(f"[Extension5] Checkpoint: {checkpoint_path}")
    model = create_model(model_name, num_classes=get_num_classes(args.dataset))
    state_dict, best_acc = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    if best_acc is not None:
        print(f"[Extension5] 加载完成，best_test_acc={best_acc:.2f}%")

    # ---- Step 3: 组(a) 复现校验 ----
    print("\n" + "#" * 70)
    print("# 组(a)：复现校验")
    print("#" * 70)
    group_a_results = run_group_a(
        model, test_loader, criterion, device, alpha, bits, args.output_dir,
        dataset=args.dataset,
    )

    # 锚点自检（36.74%/43.27% 为 CIFAR-10 锚点，仅 cifar10 时比对）
    if args.dataset == "cifar10":
        for r in group_a_results:
            if r["num_bits"] == 8:
                print(f"  [自检] 8bit anchor: {r['accuracy']:.2f}% (expected ≈36.74%)")
            if r["num_bits"] == 4:
                print(f"  [自检] 4bit anchor: {r['accuracy']:.2f}% (expected ≈43.27%)")
    else:
        print(f"  [自检] 当前数据集 {args.dataset}：CIFAR-10 锚点不适用，跳过自检")

    # ---- Step 4: 组(b) 可控噪声注入 ----
    print("\n" + "#" * 70)
    print("# 组(b)：可控噪声注入")
    print("#" * 70)
    group_b_results = run_group_b(
        model, test_loader, criterion, device, alpha, noise_types, noise_levels, args.output_dir,
    )

    # ---- Step 5: 组(c) 纯噪声替代量化 ----
    print("\n" + "#" * 70)
    print("# 组(c)：纯噪声替代量化")
    print("#" * 70)
    group_c_results, dose_4bit = run_group_c(
        model, test_loader, criterion, device, alpha, noise_types, group_a_results, args.output_dir,
    )

    # ---- Step 6: 样本级翻转统计 ----
    print("\n" + "#" * 70)
    print("# 样本级翻转统计")
    print("#" * 70)
    all_cond_data, clean_preds, clean_labels = run_sample_level_stats(
        model, test_loader, device, alpha, group_b_results, args.output_dir,
    )

    # ---- Step 7: H2 激活统计量 ----
    print("\n" + "#" * 70)
    print("# H2：激活统计量")
    print("#" * 70)
    h2_results = run_h2_activation_stats(
        model, test_loader, device, alpha, bits, args.output_dir,
    )

    # ---- Step 8: 组(d) 三架构对照 ----
    print("\n" + "#" * 70)
    print("# 组(d)：三架构 Dithering 对照")
    print("#" * 70)
    arch_data = run_group_d(args.output_dir, ext3_root=args.ext3_root, dataset=args.dataset)

    # ---- Step 9: 绘图 ----
    print("\n" + "-" * 60)
    print("[Extension5] 生成可视化图表 ...")
    plot_bits_accuracy(group_a_results, args.output_dir, alpha)
    plot_dose_accuracy(group_a_results, group_b_results, group_c_results, dose_4bit, args.output_dir)
    plot_margin_distribution(args.output_dir)
    plot_arch_comparison(arch_data, args.output_dir)

    # ---- Step 10: 终端摘要 ----
    print("\n" + "=" * 66)
    print("  [Extension5 摘要] Dithering 效应分析完成")
    print("=" * 66)
    print(f"  主分析模型: {model_name}")
    print(f"  α = {alpha:+.2f}")
    print(f"  bits 扫描: {bits}")
    print(f"  噪声类型: {noise_types}")
    print(f"  噪声强度: {noise_levels}")
    print()
    print("  组(a) 关键锚点:")
    for r in group_a_results:
        if r["num_bits"] in (4, 8):
            print(f"    bit={r['num_bits']}: Acc={r['accuracy']:.2f}%  Dose={r['dose']:.6f}")
    print()
    print("  组(b) 最优噪声:")
    for nt in noise_types:
        nt_data = [r for r in group_b_results if r["noise_type"] == nt]
        if nt_data:
            best = max(nt_data, key=lambda r: r["accuracy"])
            acc_8bit = next(r["accuracy"] for r in group_a_results if r["num_bits"] == 8)
            print(f"    {nt}: σ={best['noise_std']:.3f} → Acc={best['accuracy']:.2f}% "
                  f"(gain vs 8bit: {best['accuracy'] - acc_8bit:+.2f}%)")
    print()
    print("  组(c) 纯噪声等效对比 (dose aligned to 4bit QDQ):")
    if group_c_results:
        for r in group_c_results:
            acc_4bit = next(r2["accuracy"] for r2 in group_a_results if r2["num_bits"] == 4)
            print(f"    {r['noise_type']}: Acc={r['accuracy']:.2f}% vs 4bit QDQ={acc_4bit:.2f}%")
    print()
    print("  组(d) 三架构 Dithering 增益 (4bit vs 8bit):")
    for arch_name, data in arch_data.items():
        bits_map = {p["num_bits"]: p["accuracy"] for p in data["points"]}
        if 8 in bits_map and 4 in bits_map:
            gain = bits_map[4] - bits_map[8]
            print(f"    {data['display']:>12} (MaxPool={data['maxpool']}): "
                  f"{gain:+.2f}%  (8bit={bits_map[8]:.2f}%, 4bit={bits_map[4]:.2f}%)")
    print()
    print("  输出文件清单:")
    for fn in sorted(os.listdir(args.output_dir)):
        print(f"    {os.path.join(args.output_dir, fn)}")
    print("=" * 66)
    print("[Extension5 Done]")


if __name__ == "__main__":
    main()