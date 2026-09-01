"""
任务1：单层输出分布分析脚本 task1_layer_analysis.py

功能：
    针对 SimpleCNN 的每个 Conv2d/Linear 层输出，在多个不同 alpha 非线性强度下，
    分析层输出分布与干净基线 (alpha=0.0) 的偏移：
        - 相对平均误差 (RME, Relative Mean Error)
        - 余弦相似度 (Cosine Similarity)
    并绘制误差跨层累积曲线（Error Accumulation Across Layers），
    直观展示误差如何随网络深度逐级累积。
    可选地对 conv1 和 conv4 做直方图分布对比（alpha=0.0 vs alpha=0.3）。

运行命令示例：
    # 使用默认设置
    python task1_layer_analysis.py

    # 自定义 alpha 与样本数
    python task1_layer_analysis.py --alpha_values "-0.3,0.0,0.3" --num_samples 500
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

from utils.data_loader import get_dataloaders
from utils.paths import get_ckpt_root, get_outputs_root, get_num_classes
from utils.nonlinearity import (
    register_nonlinearity_hooks,
    remove_hooks,
)


# ------------------------------------------------------------------
# 随机种子
# ------------------------------------------------------------------
def set_seed(seed: int = 42):
    """设置随机种子以保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_model(model_name: str, num_classes: int = 10):
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)
    # 【拓展研究1 新增】CIFAR-10 适配版 ResNet-18（约 11.17M 参数）
    elif model_name == "resnet18":
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes)
    # 【VGG-11 新增】CIFAR-10 轻量适配版 VGG-11（约 9.2M 参数，纯前馈深层对照）
    elif model_name == "vgg11":
        from models.vgg11 import VGG11
        return VGG11(num_classes=num_classes)
    raise ValueError(f"不支持的模型名称: '{model_name}'，当前已支持 simple_cnn, resnet18, vgg11")


# ------------------------------------------------------------------
# 命令行参数
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Task1: 层分布偏移与误差累积分析")
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100"],
        help="数据集，默认: cifar10",
    )
    parser.add_argument(
        "--alpha_values", type=str,
        default="-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3",
        help="逗号分隔的 alpha 值列表"
    )
    parser.add_argument("--model", type=str, default="simple_cnn", help="模型名称")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型权重 checkpoint 路径，默认自动拼接"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录，默认 {outputs_root}/task1_{model_tag}"
    )
    parser.add_argument(
        "--num_samples", type=int, default=500,
        help="用于层分析的测试样本数量（取前 num_samples 张，所有 alpha 共享）")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ------------------------------------------------------------------
# 观测层：不同模型的 (属性名, 显示名) 映射列表
# ------------------------------------------------------------------
# SimpleCNN：conv1/conv2/conv3/conv4/fc 共 5 个观测点
OBSERVE_LAYERS_SIMPLECNN = [
    ("conv1", "conv1_output"),
    ("conv2", "conv2_output"),
    ("conv3", "conv3_output"),
    ("conv4", "conv4_output"),
    ("fc",    "fc_output"),
]

# 【拓展研究1 新增】ResNet-18 的 5 个观测点（与 SimpleCNN 数量保持一致，便于层间对比）
# layer1/layer2/layer3/layer4 都是 nn.Sequential 容器，
# register_forward_hook 可以直接挂在容器上捕获整个 stage 的输出
OBSERVE_LAYERS_RESNET18 = [
    ("layer1", "layer1_output"),   # 第1组残差块输出 (64通道, 32×32)
    ("layer2", "layer2_output"),   # 第2组残差块输出 (128通道, 16×16)
    ("layer3", "layer3_output"),   # 第3组残差块输出 (256通道, 8×8)
    ("layer4", "layer4_output"),   # 第4组残差块输出 (512通道, 4×4)
    ("fc",     "fc_output"),       # 全连接分类头输出
]

# 【VGG-11 新增】VGG-11 的 5 个关键观测点（与前两模型数量保持一致）
# 由于 VGG-11 使用 features = nn.Sequential，子模块只能通过整数索引访问。
# 格式：(int_features_idx, display_name)  （fc_output 在 register_* 中单独处理）
# 观测点选择：Block2/3/4/5 的最后一个 MaxPool 后，反映整个 stage 的特征
OBSERVE_LAYERS_VGG11 = [
    (7,  "block2_output"),   # features[7]  = 第2个 MaxPool2d（Block2 结束，16×16→8×8 前）
    (14, "block3_output"),   # features[14] = 第3个 MaxPool2d（Block3 结束）
    (21, "block4_output"),   # features[21] = 第4个 MaxPool2d（Block4 结束）
    (28, "block5_output"),   # features[28] = 第5个 MaxPool2d（Block5 结束）
    # fc_output 在注册函数中单独附加，避免在此混用 str/int
]

# ========== 新增：直方图对比层映射 ==========
# 每个模型取"最浅层"和"最深层"两个观测点做分布对比
HISTOGRAM_LAYERS = {
    "simple_cnn": [
        ("conv1_output", "histogram_conv1.png"),
        ("conv4_output", "histogram_conv4.png"),
    ],
    "resnet18": [
        ("layer1_output", "histogram_layer1.png"),
        ("layer4_output", "histogram_layer4.png"),
    ],
    "vgg11": [
        ("block2_output", "histogram_block2.png"),
        ("block5_output", "histogram_block5.png"),
    ],
}

def get_observe_layers(model_name: str):
    """根据模型名称字符串返回对应的观测层映射列表。"""
    if model_name == "simple_cnn":
        return OBSERVE_LAYERS_SIMPLECNN
    elif model_name == "resnet18":
        return OBSERVE_LAYERS_RESNET18
    elif model_name == "vgg11":
        return OBSERVE_LAYERS_VGG11
    else:
        raise ValueError(f"未知模型 {model_name} 未定义观测层映射，"
                         f"当前已支持 simple_cnn / resnet18 / vgg11")


def register_layer_capture_hooks(model: nn.Module, model_name: str):
    """
    注册 forward hook，捕获指定模型观测层的输出。

    Args:
        model (nn.Module): 待挂钩子的模型
        model_name (str): 模型名字符串，用于选择观测层列表

    Returns:
        layer_outputs (dict): key=层英文输出名，value=Tensor (CPU, detached)
        capture_hooks (list): 钩子句柄列表
    """
    layer_outputs = {}
    capture_hooks = []
    observe_layers = get_observe_layers(model_name)

    for layer_info in observe_layers:
        # 根据模型类型选择不同的子模块定位方式：
        #   - SimpleCNN / ResNet-18：(str_attr, out_name)，通过 getattr(model, attr_name) 定位
        #   - VGG-11：(int_idx, out_name)，通过 model.features[idx] 定位
        if model_name == "vgg11":
            idx, out_name = layer_info
            module = model.features[idx] if (0 <= idx < len(model.features)) else None
            if module is None:
                print(f"[Layer Analysis] 警告：VGG-11 features[{idx}] 越界，跳过观测 {out_name}")
                continue
        else:
            attr_name, out_name = layer_info
            module = getattr(model, attr_name, None)
            if module is None:
                print(f"[Layer Analysis] 警告：模型 {model_name} 中找不到子模块 {attr_name}，跳过观测 {out_name}")
                continue

        def _make_hook(name=out_name):
            def _hook(m, inp, out):
                layer_outputs[name] = out.detach().cpu()
            return _hook

        h = module.register_forward_hook(_make_hook())
        capture_hooks.append(h)

    # --------------------------------------------------------
    # 对所有模型，额外注册 fc_output（分类层输出）观测点
    # --------------------------------------------------------
    fc_module = None
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        # SimpleCNN / ResNet-18 风格：顶层 fc 属性
        fc_module = model.fc
    elif hasattr(model, 'classifier'):
        # VGG-11 风格：classifier = Sequential([Flatten, Linear(512, 10)])
        # 取最后一个子模块（Linear）挂 hook
        last_mod = list(model.classifier.children())[-1] if len(list(model.classifier.children())) > 0 else None
        if isinstance(last_mod, nn.Linear):
            fc_module = last_mod

    if fc_module is not None:
        def _make_fc_hook():
            def _hook(m, inp, out):
                layer_outputs["fc_output"] = out.detach().cpu()
            return _hook

        h = fc_module.register_forward_hook(_make_fc_hook())
        capture_hooks.append(h)
    else:
        print(f"[Layer Analysis] 警告：模型 {model_name} 未找到分类 Linear 层，fc_output 将缺失。")

    return layer_outputs, capture_hooks


# ------------------------------------------------------------------
# 从 test_loader 中取固定数量的测试样本，封装成一个 batch
# ------------------------------------------------------------------
def collect_fixed_samples(test_loader, num_samples: int, device: str):
    all_images = []
    all_labels = []
    count = 0
    for imgs, lbls in test_loader:
        need = num_samples - count
        take = min(need, imgs.size(0))
        all_images.append(imgs[:take])
        all_labels.append(lbls[:take])
        count += take
        if count >= num_samples:
            break
    images = torch.cat(all_images, dim=0).to(device)
    labels = torch.cat(all_labels, dim=0).to(device)
    return images, labels


# ------------------------------------------------------------------
# 计算 RME 和余弦相似度指标
# ------------------------------------------------------------------
def compute_metrics(clean_out: torch.Tensor, dist_out: torch.Tensor):
    """
    给定 clean / distorted 层输出（shape = (N, ...)），返回：(RME, cosine_similarity)。

    - RME = ||mean(distorted) - mean(clean)||_2  /  ||mean(clean)||_2
      其中 mean 是对 batch 维求平均后得到的 per-feature mean 向量，再求 L2 norm。
    - Cosine Similarity：展平后逐样本计算余弦相似度，然后取平均。
    """
    clean_flat = clean_out.view(clean_out.shape[0], -1)   # (N, D)
    dist_flat = dist_out.view(dist_out.shape[0], -1)      # (N, D)

    # ----- RME -----
    mean_clean = clean_flat.mean(dim=0)   # (D,)
    mean_dist = dist_flat.mean(dim=0)     # (D,)
    diff = mean_dist - mean_clean
    numerator = torch.norm(diff, p=2).item()
    denom = float(np.clip(float(torch.norm(mean_clean, p=2).item()), 1e-12, None))
    RME = float(numerator) / denom

    # ----- Cosine Similarity (逐样本平均) -----
    cossim_per_sample = torch.nn.functional.cosine_similarity(clean_flat, dist_flat, dim=1)
    cos_sim = float(cossim_per_sample.mean().item())

    return float(RME), cos_sim


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    model_tag = args.model.replace("-", "_")
    if args.checkpoint is None:
        args.checkpoint = os.path.join(get_ckpt_root(args.dataset), args.model, "best_model.pth")
    if args.output_dir is None:
        args.output_dir = os.path.join(get_outputs_root(args.dataset), f"task1_{model_tag}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 解析 alpha 列表
    alpha_list = sorted([float(s.strip()) for s in args.alpha_values.split(",") if s.strip() != ""])
    if 0.0 not in alpha_list:
        alpha_list = [0.0] + alpha_list
    alpha_list.sort()
    print(f"[Task1 Layer] alpha 列表: {alpha_list}")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # ---- Step1: 加载模型和权重
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint 不存在: {args.checkpoint}")
    model = get_model(args.model, num_classes=get_num_classes(args.dataset))
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_dataset = ckpt.get("dataset", "cifar10")
        print(f"[Task1 Layer] checkpoint 数据集来源: {ckpt_dataset}")
        if ckpt_dataset != args.dataset:
            print(f"[Task1 Layer] WARNING: checkpoint 数据集 ({ckpt_dataset}) 与当前 --dataset "
                  f"({args.dataset}) 不一致。")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[Task1 Layer] 模型与权重加载完成")

    # ---- Step2: 获取固定测试样本
    _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2, dataset=args.dataset)
    fixed_images, _ = collect_fixed_samples(test_loader, args.num_samples, device)
    print(f"[Task1 Layer] 固定分析样本数: {fixed_images.shape[0]}")

    # ---- Step4: 干净基线层输出 (alpha=0.0)
    print("\n" + "-" * 60)
    print("[Task1 Layer] 获取干净基线层输出 (alpha=0.0)")

    clean_outputs = {}
    model.load_state_dict(state_dict)
    model.eval()

    layer_out_clean, hooks_clean_capture = register_layer_capture_hooks(model, args.model)
    with torch.no_grad():
        _ = model(fixed_images)
    for ln in layer_out_clean:
        clean_outputs[ln] = layer_out_clean[ln].clone()
    remove_hooks(hooks_clean_capture)

    # 根据当前模型动态选择观测层顺序，兼容 SimpleCNN / ResNet-18
    observe_layers_current = get_observe_layers(args.model)
    OBSERVE_LAYERS_ORDER = [out_name for (_, out_name) in observe_layers_current if out_name in clean_outputs]
    print(f"[Task1 Layer] 当前模型: {args.model}, 观测层顺序: {OBSERVE_LAYERS_ORDER}")

    # ---- Step5: 遍历 alpha ≠ 0.0，计算层偏移
    print("\n" + "-" * 60)
    print("[Task1 Layer] 开始 alpha ≠ 0.0 层偏移计算 ...")

    rows_csv = []
    cosine_by_alpha = {}

    for alpha in alpha_list:
        if abs(alpha - 0.0) < 1e-12:
            cosine_by_alpha[0.0] = [1.0 for _ in OBSERVE_LAYERS_ORDER]
            continue

        # a. 重新加载干净权重
        model.load_state_dict(state_dict)
        model.eval()

        # b. 注册捕获 hooks + 非线性注入 hooks（传入 args.model 选择观测层）
        layer_out_dict, capture_hooks = register_layer_capture_hooks(model, args.model)
        nonlin_hooks = register_nonlinearity_hooks(model, alpha=alpha)

        # c. 推理
        with torch.no_grad():
            _ = model(fixed_images)
        remove_hooks(capture_hooks)
        remove_hooks(nonlin_hooks)

        # d. 每层计算 RME / cos_sim
        cos_list = []
        for ln in OBSERVE_LAYERS_ORDER:
            clean_t = clean_outputs[ln]
            dist_t = layer_out_dict[ln]
            RME, cs = compute_metrics(clean_t, dist_t)
            rows_csv.append({
                "alpha": alpha,
                "layer_name": ln,
                "RME": round(RME, 8),
                "cosine_similarity": round(cs, 8),
            })
            cos_list.append(cs)
        cosine_by_alpha[alpha] = cos_list
        print(f"    alpha = {alpha: .3f}  done")

    # ---- Step6: 保存层偏移汇总 CSV
    csv_path = os.path.join(args.output_dir, "layer_distribution_shift.csv")
    fieldnames = ["alpha", "layer_name", "RME", "cosine_similarity"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_csv:
            writer.writerow(row)
    print(f"\n[Task1 Layer] 层偏移 CSV 已保存: {csv_path}")

    # ---- Step7: 绘制误差累积曲线
    print("[Task1 Layer] 绘制 Error Accumulation Across Layers 图")

    layer_indices = list(range(len(OBSERVE_LAYERS_ORDER)))
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    cmap = plt.get_cmap("coolwarm")
    alpha_list_plot = sorted(list(cosine_by_alpha.keys()))
    n_alpha = len(alpha_list_plot)
    for i_alpha, alpha in enumerate(alpha_list_plot):
        y = cosine_by_alpha[alpha]
        color = cmap(i_alpha / max(1, n_alpha - 1))
        ax.plot(
            layer_indices, y,
            marker="o", linewidth=2, color=color,
            label=f"α = {alpha:.2f}"
        )

    ax.set_xticks(layer_indices)
    ax.set_xticklabels(OBSERVE_LAYERS_ORDER, rotation=20)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Cosine Similarity with Clean Output", fontsize=12)
    ax.set_title("Error Accumulation Across Layers", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    png_path = os.path.join(args.output_dir, "error_accumulation.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task1 Layer] 误差累积曲线已保存: {png_path}")

    # ---- Step8: 直方图对比 (conv1 / conv4, alpha=0.0 vs 0.3)
    print("\n" + "-" * 60)
    print("[Task1 Layer] 绘制直方图对比：conv1 / conv4 (alpha=0.0 vs alpha=0.3)")

    def plot_hist(layer_name, out_png_name):
        if layer_name not in clean_outputs:
            print(f"    跳过 {layer_name}：层输出不存在")
            return
        clean_vals = clean_outputs[layer_name].numpy().flatten()
        if 0.3 not in cosine_by_alpha:
            print(f"    跳过 {layer_name}：alpha=0.3 不在扫描列表")
            return
        # 重新获取 alpha=0.3 下的层输出
        model.load_state_dict(state_dict)
        model.eval()
        lod, capts = register_layer_capture_hooks(model, args.model)
        hooks_nl = register_nonlinearity_hooks(model, alpha=0.3)
        with torch.no_grad():
            _ = model(fixed_images)
        remove_hooks(hooks_nl)
        remove_hooks(capts)
        if layer_name not in lod:
            print(f"    跳过 {layer_name}：无法在 distorted 推理中获取")
            return
        dist_vals = lod[layer_name].numpy().flatten()

        fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
        ax.hist(clean_vals, bins=100, density=True, alpha=0.6,
                label=r"Clean ($\alpha=0.0$)", color="#1f77b4")
        ax.hist(dist_vals, bins=100, density=True, alpha=0.6,
                label=r"Distorted ($\alpha=0.3$)", color="#d62728")
        ax.set_xlabel("Output Value", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"Output Distribution at {layer_name} ($\\alpha=0.0$ vs $\\alpha=0.3$)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = os.path.join(args.output_dir, out_png_name)
        plt.savefig(png_path, bbox_inches="tight")
        plt.close(fig)
        print(f"    已保存: {png_path}")

    # ---- 根据模型类型动态选择直方图对比层 ----
    if args.model in HISTOGRAM_LAYERS:
        print(f"[Task1 Layer] 绘制直方图对比（模型: {args.model}）")
        for layer_name, out_png_name in HISTOGRAM_LAYERS[args.model]:
            plot_hist(layer_name, out_png_name)
    else:
        print(f"[Task1 Layer] 当前模型 {args.model} 未定义直方图对比层，跳过直方图绘制")

    # ---- Step9: 终端打印摘要（中文）
    print("\n" + "=" * 70)
    print("  [Task1 Layer 摘要] 层分布偏移与误差累积分析")
    print("=" * 70)

    avg_cos_per_layer = {ln: 0.0 for ln in OBSERVE_LAYERS_ORDER}
    count_alpha = 0
    for a in alpha_list:
        if abs(a) < 1e-12:
            continue
        if a in cosine_by_alpha:
            for i, ln in enumerate(OBSERVE_LAYERS_ORDER):
                avg_cos_per_layer[ln] += cosine_by_alpha[a][i]
            count_alpha += 1

    if count_alpha > 0:
        print(f"  各层平均余弦相似度（跨 alpha≠0 平均）：")
        pairs = [(ln, avg_cos_per_layer[ln] / count_alpha) for ln in OBSERVE_LAYERS_ORDER]
        sorted_layers = sorted(pairs, key=lambda x: x[1])
        for ln, v in pairs:
            print(f"    {ln:<16}：{v:.6f}")
        most_sensitive = sorted_layers[0]
        least_sensitive = sorted_layers[-1]
        print()
        print(f"  最敏感层（平均余弦最小）  : {most_sensitive[0]} (cos={most_sensitive[1]:.6f})")
        print(f"  最不敏感层（平均余弦最大）: {least_sensitive[0]} (cos={least_sensitive[1]:.6f})")

    print("=" * 70)
    print(f"[Task1 Layer Done] 所有结果保存在: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
