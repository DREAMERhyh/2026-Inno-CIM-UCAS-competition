"""
模型评估脚本 evaluate.py

功能：
    - 加载指定模型结构与训练好的权重文件（.pth）
    - 在 CIFAR-10 测试集上进行完整评估
    - 计算并打印：准确率、每类精确率 / 召回率 / F1-score 及宏平均
    - 可选择重新生成并保存混淆矩阵

使用示例：
    # 使用默认路径评估 simple_cnn 的最佳模型
    python evaluate.py --model simple_cnn --checkpoint ./checkpoints/simple_cnn/best_model.pth

    # 使用 CPU 评估
    python evaluate.py --model simple_cnn --checkpoint ./checkpoints/simple_cnn/best_model.pth --device cpu

    # 同时保存混淆矩阵到指定目录
    python evaluate.py --model simple_cnn --checkpoint <path> --save_output_dir ./outputs/simple_cnn
"""

import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from utils.data_loader import get_dataloaders


# CIFAR-10 固定 10 个类别名称
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ------------------------------------------------------------------
# 模型工厂：与 train.py 保持一致
# ------------------------------------------------------------------
def get_model(model_name: str, num_classes: int = 10):
    """
    根据模型名称字符串动态创建对应的模型实例，与 train.py 对应。

    Args:
        model_name (str): 模型名称，如 "simple_cnn"
        num_classes (int): 类别数，默认 10

    Returns:
        nn.Module: 实例化好的模型对象

    Raises:
        ValueError: 若模型名称未注册
    """
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)

    # 【拓展研究1 新增】CIFAR-10 适配版 ResNet-18
    elif model_name == "resnet18":
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes)

    # 【VGG-11 新增】CIFAR-10 轻量适配版 VGG-11（纯前馈深层对照）
    elif model_name == "vgg11":
        from models.vgg11 import VGG11
        return VGG11(num_classes=num_classes)

    # ---------- 扩展更多模型 ----------
    # elif model_name == "vgg16":
    #     from models.vgg import VGG16
    #     return VGG16(num_classes=num_classes)
    # --------------------------------------------------------

    else:
        raise ValueError(f"不支持的模型名称: '{model_name}'，当前已支持 simple_cnn, resnet18, vgg11")


# ------------------------------------------------------------------
# 命令行参数解析
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 模型评估脚本")
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        help="模型名称，默认: simple_cnn",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/simple_cnn/best_model.pth",
        help="模型权重文件路径 (.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="评估设备，默认自动选择",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="评估 batch size，默认: 128",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="数据加载线程数，默认: 2",
    )
    parser.add_argument(
        "--save_output_dir",
        type=str,
        default=None,
        help="（可选）评估结果输出目录，若指定则同时保存混淆矩阵与指标 JSON",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# 主评估流程
# ------------------------------------------------------------------
def main():
    args = parse_args()

    # 1) 设备选择
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Device] 使用评估设备: {device}")

    # 2) 加载测试集 DataLoader（不需要训练集，但 get_dataloaders 返回两个，train_loader 不会被使用
    print("\n[Data] 加载 CIFAR-10 测试集...")
    _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 3) 创建模型实例
    print(f"\n[Model] 创建模型: {args.model}")
    model = get_model(args.model, num_classes=10)

    # 4) 加载权重文件
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"权重文件不存在: {ckpt_path}")

    print(f"[Model] 加载权重: {ckpt_path}")
    # map_location 保证即使在仅 CPU 的机器上也能加载 GPU 训练得到的权重
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 兼容两种存储格式：
    #   a) Trainer 保存的字典：包含 "model_state_dict" 字段
    #   b) 直接保存的 state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if "best_test_acc" in checkpoint:
            print(f"[Model] 权重文件记录的最佳测试准确率: {checkpoint['best_test_acc']:.2f}% (Epoch {checkpoint.get('epoch', 'N/A')})")
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("[Model] 权重加载成功！")

    # 5) 在测试集上跑模型，收集所有真实标签与预测结果
    print("\n[Eval] 正在运行模型推理...")
    all_labels = []
    all_preds = []
    all_probs = []  # 可选，用于后续分析

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            _, predicted = torch.max(logits, 1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    # 6) 计算各项分类指标
    print("\n" + "=" * 70)
    print("[Metrics] 分类指标汇总表")
    print("=" * 70)

    # 总体准确率
    accuracy = accuracy_score(all_labels, all_preds) * 100.0
    print(f"\n总体准确率 (Accuracy): {accuracy:.2f}%")

    # 每类精确率 / 召回率 / F1-score
    precision_per_class = precision_score(
        all_labels, all_preds, labels=list(range(10)), average=None, zero_division=0
    ) * 100.0
    recall_per_class = recall_score(
        all_labels, all_preds, labels=list(range(10)), average=None, zero_division=0
    ) * 100.0
    f1_per_class = f1_score(
        all_labels, all_preds, labels=list(range(10)), average=None, zero_division=0
    ) * 100.0

    # 宏平均（Macro Average：对每类指标取算术平均，不考虑类别样本数差异）
    precision_macro = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    ) * 100.0
    recall_macro = recall_score(
        all_labels, all_preds, average="macro", zero_division=0
    ) * 100.0
    f1_macro = f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    ) * 100.0

    # 加权平均（Weighted Average：按各类样本数加权）
    precision_weighted = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    ) * 100.0
    recall_weighted = recall_score(
        all_labels, all_preds, average="weighted", zero_division=0
    ) * 100.0
    f1_weighted = f1_score(
        all_labels, all_preds, average="weighted", zero_division=0
    ) * 100.0

    # 打印每类指标表格
    print("\n{:<12} {:>10} {:>10} {:>10}".format("类别", "Precision", "Recall", "F1-score"))
    print("-" * 46)
    for i, cls_name in enumerate(CIFAR10_CLASSES):
        print(
            "{:<12} {:>9.2f}% {:>9.2f}% {:>9.2f}%".format(
                cls_name,
                precision_per_class[i],
                recall_per_class[i],
                f1_per_class[i],
            )
        )
    print("-" * 46)
    print(
        "{:<12} {:>9.2f}% {:>9.2f}% {:>9.2f}%".format(
            "Macro Avg", precision_macro, recall_macro, f1_macro
        )
    )
    print(
        "{:<12} {:>9.2f}% {:>9.2f}% {:>9.2f}%".format(
            "Weighted Avg", precision_weighted, recall_weighted, f1_weighted
        )
    )
    print("=" * 70)

    # 7) 若指定了输出目录，则保存混淆矩阵和指标 JSON
    if args.save_output_dir:
        os.makedirs(args.save_output_dir, exist_ok=True)

        # --- 混淆矩阵图
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
        fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASSES)
        disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
        ax.set_title(f"Confusion Matrix - Test Set ({args.model})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        cm_path = os.path.join(args.save_output_dir, "confusion_matrix_eval.png")
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[Output] 混淆矩阵已保存: {cm_path}")

        # --- 指标 JSON
        metrics = {
            "accuracy": round(accuracy, 4),
            "precision_macro": round(precision_macro, 4),
            "recall_macro": round(recall_macro, 4),
            "f1_macro": round(f1_macro, 4),
            "precision_weighted": round(precision_weighted, 4),
            "recall_weighted": round(recall_weighted, 4),
            "f1_weighted": round(f1_weighted, 4),
            "per_class": {
                cls_name: {
                    "precision": round(precision_per_class[i], 4),
                    "recall": round(recall_per_class[i], 4),
                    "f1": round(f1_per_class[i], 4),
                }
                for i, cls_name in enumerate(CIFAR10_CLASSES)
            },
        }
        metrics_path = os.path.join(args.save_output_dir, "metrics_eval.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"[Output] 评估指标已保存: {metrics_path}")

    print("\n[Done] 评估完成！")


if __name__ == "__main__":
    main()
