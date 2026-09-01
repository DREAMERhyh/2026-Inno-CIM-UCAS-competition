"""
任务1：非线性误差敏感性分析 —— 主分析脚本 task1_sensitivity_analysis.py

功能：
    模拟存算一体芯片中模拟域 MAC 的输入相关非线性失真模型：
        y = alpha * x^3 + (1 - alpha) * x
    将其注入到 SimpleCNN 的所有 Conv2d 与 Linear 输入端，
    扫描不同 alpha 值下的测试集准确率变化，绘制精度衰减曲线。

运行命令示例：
    # 使用默认 alpha 范围 (-0.3 ~ 0.3)
    python task1_sensitivity_analysis.py

    # 自定义 alpha 扫描范围
    python task1_sensitivity_analysis.py --alpha_values "-0.5,-0.3,0.0,0.3,0.5"

    # 指定 GPU 与权重路径
    python task1_sensitivity_analysis.py --model simple_cnn \
        --checkpoint ./checkpoints/simple_cnn/best_model.pth --device cuda
"""

import argparse
import os
import random
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
# 任务要求：所有图表使用英文，这里用 DejaVu Sans（matplotlib 默认英文字体）
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
# 随机种子（可复用 train.py 的 set_seed 逻辑）
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


# ------------------------------------------------------------------
# 模型工厂（与 train.py 保持一致）
# ------------------------------------------------------------------
def get_model(model_name: str, num_classes: int = 10):
    """根据模型名称字符串动态创建对应的模型实例。"""
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
    else:
        raise ValueError(f"不支持的模型名称: '{model_name}'，当前已支持 simple_cnn, resnet18, vgg11")


# ------------------------------------------------------------------
# 命令行参数
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Task1: 非线性误差 (Nonlinearity) 敏感性分析"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100"],
        help="数据集，默认: cifar10",
    )
    parser.add_argument(
        "--alpha_values", type=str,
        default="-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3",
        help="逗号分隔的 alpha 值列表（默认: -0.3,-0.2,-0.1,0.0,0.1,0.2,0.3）"
    )
    parser.add_argument("--model", type=str, default="simple_cnn", help="模型名称")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型权重 checkpoint 路径，默认自动拼接"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="推理 batch size")
    parser.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="推理设备（默认自动选择）"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="分析结果输出目录，默认 {outputs_root}/task1_{model_tag}"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


# ------------------------------------------------------------------
# 在测试集上执行一次完整推理，返回准确率和 loss
# ------------------------------------------------------------------
@torch.no_grad()
def evaluate_on_test(model, test_loader, criterion, device):
    """
    在给定测试集上对模型进行一次推理，返回总体准确率和平均 loss。

    注意：调用本函数前应确保非线性钩子已注册 / 移除到正确状态。
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


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def main():
    args = parse_args()

    # ---------- Step 0: 随机种子 + 输出目录 ----------
    set_seed(args.seed)
    model_tag = args.model.replace("-", "_")
    if args.checkpoint is None:
        args.checkpoint = os.path.join(get_ckpt_root(args.dataset), args.model, "best_model.pth")
    if args.output_dir is None:
        args.output_dir = os.path.join(get_outputs_root(args.dataset), f"task1_{model_tag}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Task1] 数据集: {args.dataset}, 类别数: {get_num_classes(args.dataset)}")
    print(f"[Task1] 输出目录: {os.path.abspath(args.output_dir)}")

    # 解析 alpha 值
    alpha_list = [float(s.strip()) for s in args.alpha_values.split(",") if s.strip() != ""]
    # 按升序排序，方便绘图
    alpha_list.sort()
    print(f"[Task1] 扫描 alpha 值列表: {alpha_list}")

    # 设备自动选择
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Task1] 推理设备: {device}")

    # ---------- Step 1: 加载干净模型和权重 ----------
    print("\n" + "-" * 60)
    print(f"[Task1] 加载模型: {args.model}")
    model = get_model(args.model, num_classes=get_num_classes(args.dataset))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Task1] 总参数量: {total_params:,}")

    # 检查 checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint 不存在: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_dataset = ckpt.get("dataset", "cifar10")
        print(f"[Task1] checkpoint 数据集来源: {ckpt_dataset}")
        if ckpt_dataset != args.dataset:
            print(f"[Task1] WARNING: checkpoint 数据集 ({ckpt_dataset}) 与当前 --dataset "
                  f"({args.dataset}) 不一致，按兼容模式继续加载。")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            if "best_test_acc" in ckpt:
                print(f"[Task1] 权重文件记录的原始最佳准确率: {ckpt['best_test_acc']:.2f}%")
        else:
            state_dict = ckpt  # dict 但无 model_state_dict 字段，假设即 state_dict
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    print(f"[Task1] 权重加载完成: {args.checkpoint}")

    # ---------- Step 2: 加载测试集 ----------
    print("\n" + "-" * 60)
    print("[Task1] 加载测试集 DataLoader ...")
    _, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=2, dataset=args.dataset)
    criterion = nn.CrossEntropyLoss()

    # ---------- Step 3: 干净基线推理 (alpha=0.0) ----------
    print("\n" + "-" * 60)
    print("[Task1] 干净基线推理 (alpha = 0.0) ...")
    clean_acc, clean_loss = evaluate_on_test(model, test_loader, criterion, device)
    print(f"[Task1] Clean baseline accuracy: {clean_acc:.2f}% , Loss: {clean_loss:.4f}")

    # ---------- Step 4: 非线性误差敏感性扫描 ----------
    print("\n" + "-" * 60)
    print("[Task1] 开始 alpha 扫描，每个 alpha 会重新加载干净权重以避免累积影响 ...")

    # 结果保存表：每项为 {alpha, accuracy, loss}
    results = []

    for alpha in tqdm(alpha_list, desc="Alpha Scan", leave=True):
        # a. 重新加载干净权重（每个 alpha 从同一基线出发）
        model.load_state_dict(state_dict)
        model.to(device)

        # b. 注册非线性注入钩子
        hooks = register_nonlinearity_hooks(model, alpha=alpha)

        # c. 推理
        acc, loss = evaluate_on_test(model, test_loader, criterion, device)

        # d. 移除钩子
        remove_hooks(hooks)

        # e. 记录
        results.append({
            "alpha": alpha,
            "accuracy": round(acc, 4),
            "loss": round(loss, 6),
        })

        # f. 打印
        print(
            f"Alpha={alpha:>6.3f}, Accuracy={acc:.2f}%, Loss={loss:.4f}, "
            f"Acc Drop={clean_acc - acc:.2f}%"
        )

    # 保存 CSV
    csv_path = os.path.join(args.output_dir, "alpha_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "accuracy", "loss"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\n[Task1] 敏感性扫描 CSV 已保存: {csv_path}")

    # ---------- Step 5: 绘制精度衰减曲线 ----------
    print("\n" + "-" * 60)
    print("[Task1] 绘制 Accuracy vs Alpha 曲线图 ...")

    alphas_plot = [r["alpha"] for r in results]
    accs_plot = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    ax.plot(
        alphas_plot, accs_plot,
        marker="o", markersize=7, linewidth=2.2,
        color="#1f77b4", label="Accuracy under Nonlinear Distortion",
    )
    # 标注 alpha=0 的基线点（如果列表中存在）
    if 0.0 in alphas_plot:
        idx0 = alphas_plot.index(0.0)
        ax.scatter(
            [0.0], [accs_plot[idx0]],
            color="red", s=160, marker="*", zorder=5,
            label=f"Clean Baseline (α=0) = {accs_plot[idx0]:.2f}%",
        )
        ax.axvline(x=0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Nonlinearity Strength (α)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy vs Nonlinearity Strength", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    # x 轴刻度使用所有扫描 alpha 值
    ax.set_xticks(alphas_plot)
    ax.set_xticklabels([f"{a:.1f}" for a in alphas_plot], rotation=0)
    plt.tight_layout()

    png_path = os.path.join(args.output_dir, "accuracy_vs_alpha.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Task1] 曲线图已保存: {png_path}")

    # ---------- Step 6: 打印任务 1 摘要（中文终端） ----------
    print("\n" + "=" * 66)
    print("  [Task1 摘要] 非线性误差敏感性分析结果")
    print("=" * 66)
    print(f"  Clean baseline accuracy (α=0.0)  : {clean_acc:.2f}%")
    # 计算最大精度跌幅
    all_drops = [clean_acc - r["accuracy"] for r in results]
    max_drop_idx = int(np.argmax(all_drops))
    max_drop = all_drops[max_drop_idx]
    max_drop_alpha = results[max_drop_idx]["alpha"]
    print(f"  Max accuracy drop                : {max_drop:.2f}%  (当 α = {max_drop_alpha})")
    print()
    print("  {:>10}  |  {:>12}  |  {:>14}".format("Alpha", "Accuracy", "Accuracy Drop"))
    print("  " + "-" * 48)
    for r in results:
        print(
            "  {:>10.3f}  |  {:>10.2f}%   |  {:>12.2f}%".format(
                r["alpha"], r["accuracy"], clean_acc - r["accuracy"]
            )
        )
    print("=" * 66)
    print(f"[Task1 Done] 所有结果保存在: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
