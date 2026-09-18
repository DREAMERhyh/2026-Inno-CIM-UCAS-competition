"""
任务2：非线性感知训练（Nonlinearity-Aware Training, NAT）主脚本

功能：
    - 在训练阶段将输入相关的非线性失真注入到所有 Conv2d/Linear 输入端，
      使得模型在带噪声的条件下学习，从而提升鲁棒性。
    - 支持两种训练模式：
      * finetune  : 从干净预训练权重 best_model.pth 继续训练
      * scratch   : 随机初始化，全程在非线性失真下训练
    - 支持两种 α 注入策略：
      * random    : 每个 batch 从 alpha_range 区间均匀随机采样 α
      * fixed     : 使用固定的 alpha_fixed
    - 每 batch 钩子注册 → 前向 → 反向 → 钩子移除，确保权重更新不受影响。
    - 测试/验证阶段使用干净数据（α=0），评估真实泛化能力。
    - 训练完成后自动执行 α 敏感性扫描，并与任务1基线（未感知训练）
      的 outputs/task1/alpha_sensitivity.csv 做对比绘图。

运行命令示例：
    # 微调模式（默认随机 α ∈ [-0.3, 0.3]，40 epochs，lr=1e-3
    python task2_nat.py --mode finetune --epochs 40 --lr 1e-3

    # 从头训练模式（默认 120 epochs，lr=0.01
    python task2_nat.py --mode scratch --epochs 120 --lr 0.01

    # 微调 + 固定 α=0.3
    python task2_nat.py --mode finetune --alpha_mode fixed --alpha_fixed 0.3 --epochs 40

    # 从头训练，自定义随机 α 范围
    python task2_nat.py --mode scratch --alpha_range "-0.5,0.5" --epochs 150
"""

import argparse
import random
import os
import json
import csv
import numpy as np
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils.data_loader import get_dataloaders
from utils.paths import get_ckpt_root, get_outputs_root, get_num_classes, CIFAR10_CLASSES
from utils.nonlinearity import (
    register_nonlinearity_hooks,
    remove_hooks,
)


# ------------------------------------------------------------------
# 复用 train.py 中的 set_seed 与 get_model（与原实现完全一致
# ------------------------------------------------------------------
def set_seed(seed: int = 42):
    """
    设置 Python / NumPy / PyTorch 的随机种子，以保证实验可复现。

    Args:
        seed (int): 随机种子数值，默认为 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] 随机种子已设置为 {seed}，确保实验可复现")


def get_model(model_name: str, num_classes: int = 10):
    """
    根据模型名称字符串动态创建对应的模型实例（与 train.py 中 get_model 保持一致。

    Args:
        model_name (str): 模型名称，如 "simple_cnn"
        num_classes (int): 分类类别数，默认 10 (CIFAR-10)

    Returns:
        nn.Module: 实例化好的模型对象

    Raises:
        ValueError: 若传入未支持的模型名称
    """
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
        raise ValueError(f"不支持的模型名称: '{model_name}'，当前已支持: simple_cnn, resnet18, vgg11")


# 与任务1完全一致的 α 扫描列表（保证对比的横轴一致）
TASK1_ALPHA_LIST = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]


# ------------------------------------------------------------------
# TrainerNAT：非线性感知训练器
# ------------------------------------------------------------------
class TrainerNAT:
    """
    Nonlinearity-Aware Training 训练器。

    与原 Trainer 类主要差异：
        - _train_one_epoch 中每个 batch：
            采样/确定 α → register_nonlinearity_hooks → forward + backward
            → remove_hooks → optimizer.step
          保证权重更新始终作用于真实参数，而非 hook 修改后的参数。
        - _evaluate 始终在干净数据上进行（不注册任何非线性钩子）。

    复用 Trainer 的结果保存逻辑：
        - 保存最佳 best_model.pth（epoch/model_state_dict/optimizer_state_dict/best_test_acc）
        - 训练曲线 training_curves.png
        - 混淆矩阵 confusion_matrix.png
        - 指标 metrics.json
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler=None,
        device=None,
        num_epochs: int = 50,
        model_name: str = "model",
        save_dir_checkpoint: str = "./checkpoints/nat",
        save_dir_output: str = "./outputs/task2",
        patience=None,
        # ============ NAT 专属参数 ============
        alpha_mode: str = "random",
        alpha_range=(-0.3, 0.3),
        alpha_fixed: float = 0.3,
        # ====== 双数据集支持参数 ======
        num_classes: int = 10,
        dataset: str = "cifar10",
    ):
        """
        初始化 TrainerNAT。

        Args:
            model, train_loader, test_loader, criterion, optimizer, scheduler,
            device, num_epochs, model_name, save_dir_checkpoint, save_dir_output,
            patience: 含义与 Trainer 完全一致
            alpha_mode (str): "random" 或 "fixed"
            alpha_range (tuple): (low, high)，当 alpha_mode="random" 时的采样区间
            alpha_fixed (float): 当 alpha_mode="fixed" 时使用的固定 α
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.model_name = model_name

        # 训练起始/最佳状态：NAT 训练始终从 epoch 1 开始，不支持续训（本脚本不提供续训入口
        self.start_epoch = 0
        self.best_test_acc = 0.0
        self.best_epoch = 0

        # 设备检测
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 目录创建
        self.save_dir_checkpoint = save_dir_checkpoint
        self.save_dir_output = save_dir_output
        os.makedirs(self.save_dir_checkpoint, exist_ok=True)
        os.makedirs(self.save_dir_output, exist_ok=True)

        # 早停
        self.patience = patience
        self.early_stop_counter = 0

        # 历史
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []

        # NAT 参数
        assert alpha_mode in ("random", "fixed")
        self.alpha_mode = alpha_mode
        self.alpha_range = tuple(alpha_range)
        self.alpha_fixed = alpha_fixed

        # 双数据集支持
        self.num_classes = num_classes
        self.dataset = dataset

        self.model.to(self.device)

        print(f"[TrainerNAT] 初始化完成，模式: {alpha_mode}")
        if alpha_mode == "random":
            print(f"[TrainerNAT] 随机 α 采样区间: {self.alpha_range}")
        else:
            print(f"[TrainerNAT] 固定 α = {self.alpha_fixed}")
        print(f"[TrainerNAT] 总 epochs = {num_epochs}，设备 = {self.device}")
        if self.patience is not None:
            print(f"[TrainerNAT] 早停已启用，patience = {self.patience}")

    # -------- 每个 batch 决定 α
    def _sample_alpha(self):
        """根据 alpha_mode 返回当前 batch 使用的 α。"""
        if self.alpha_mode == "random":
            lo, hi = self.alpha_range
            return float(random.uniform(lo, hi))
        else:
            return float(self.alpha_fixed)

    # -------- 训练一个 epoch（NAT 版：每 batch 动态注入/移除钩子 --------
    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}/{self.num_epochs}",
            leave=False,
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # ====== NAT 关键：每个 batch 注入 α 并前向传播 ======
            alpha = self._sample_alpha()
            hooks = register_nonlinearity_hooks(self.model, alpha)

            try:
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            finally:
                # 确保即使前向/反向完成后移除钩子，防止对后续 batch / optimizer.step 无影响
                remove_hooks(hooks)

            # 统计
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 进度条实时显示
            cur_loss = total_loss / total
            cur_acc = 100.0 * correct / total
            progress_bar.set_postfix({
                "loss": f"{cur_loss:.4f}",
                "acc": f"{cur_acc:.2f}%",
                "alpha": f"{alpha:.3f}",
            })

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # -------- 干净验证（不使用任何非线性钩子） --------
    def _evaluate(self):
        """
        在干净测试集上评估模型（与任务1中 clean baseline 一致）。
        不注册任何非线性钩子（α=0）。
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(predicted.cpu().numpy().tolist())

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy, all_labels, all_preds

    # -------- 保存最佳权重 --------
    def _save_best_checkpoint(self, epoch, test_acc):
        save_path = os.path.join(self.save_dir_checkpoint, "best_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_test_acc": test_acc,
            "dataset": self.dataset,
        }, save_path)
        return save_path

    # -------- 绘制训练曲线 --------
    def _plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), dpi=120)
        epochs_range = list(range(1, len(self.train_loss_history) + 1))

        # Loss
        ax1.plot(epochs_range, self.train_loss_history, label="Train Loss", color="#1f77b4", linewidth=2)
        ax1.plot(epochs_range, self.test_loss_history, label="Test Loss", color="#ff7f0e", linewidth=2)
        ax1.set_title(f"Loss Curves (NAT {self.model_name})", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs_range, self.train_acc_history, label="Train Acc", color="#1f77b4", linewidth=2)
        ax2.plot(epochs_range, self.test_acc_history, label="Test Acc", color="#ff7f0e", linewidth=2)

        # 最佳准确率星号标注
        if len(self.test_acc_history) > 0:
            best_idx_1based = self.best_epoch
            if 1 <= best_idx_1based <= len(self.test_acc_history):
                best_acc_value = self.test_acc_history[best_idx_1based - 1]
                ax2.scatter(
                    best_idx_1based, best_acc_value,
                    color="red", s=100, marker="*", zorder=5,
                    label=f"Best Test Acc = {best_acc_value:.2f}% (Epoch {self.best_epoch})",
                )

        ax2.set_title(f"Accuracy Curves (NAT {self.model_name})", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir_output, "training_curves.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[TrainerNAT] 训练曲线已保存: {save_path}")

    # -------- 绘制混淆矩阵 --------
    def _plot_confusion_matrix(self, all_labels, all_preds):
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.num_classes)))
        fig, ax = plt.subplots(figsize=(9, 8), dpi=120)
        if self.num_classes > 10:
            display_labels = None
        else:
            display_labels = CIFAR10_CLASSES
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
        if self.num_classes > 10:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"Confusion Matrix - Test Set (NAT {self.model_name})", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir_output, "confusion_matrix.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[TrainerNAT] 混淆矩阵已保存: {save_path}")

    # -------- 保存 metrics.json --------
    def _save_metrics(self, final_all_labels=None, final_all_preds=None):
        metrics = {
            "dataset": self.dataset,
            "best_test_acc": self.best_test_acc,
            "best_epoch": self.best_epoch,
            "total_epochs": self.num_epochs,
            "final_train_loss": self.train_loss_history[-1] if self.train_loss_history else None,
            "final_test_loss": self.test_loss_history[-1] if self.test_loss_history else None,
            "final_train_acc": self.train_acc_history[-1] if self.train_acc_history else None,
            "final_test_acc": self.test_acc_history[-1] if self.test_acc_history else None,
            "alpha_mode": self.alpha_mode,
            "alpha_range": list(self.alpha_range),
            "alpha_fixed": self.alpha_fixed,
            "mode_nat_train_mode": None,   # 主函数中另外写入
        }
        save_path = os.path.join(self.save_dir_output, "metrics.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"[TrainerNAT] 训练指标已保存: {save_path}")

    # -------- 主训练循环 --------
    def train(self):
        """执行完整的 NAT 训练流程。"""
        print("\n" + "=" * 60)
        print("[TrainerNAT] 开始训练...")
        print("=" * 60)

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch)
            test_loss, test_acc, all_labels, all_preds = self._evaluate()

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_acc)

            # scheduler.step（与干净数据（CosineAnnealingLR
            if self.scheduler is not None:
                self.scheduler.step()

            # 打印总结
            print(
                f"[Epoch {epoch:3d}/{self.num_epochs}] "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%"
            )

            # 最佳模型保存 & 早停
            improved = False
            if test_acc > self.best_test_acc:
                improved = True
                self.best_test_acc = test_acc
                self.best_epoch = epoch
                self.early_stop_counter = 0
                ckpt_path = self._save_best_checkpoint(epoch, test_acc)
                print(f"  >>> 新最佳！测试准确率 = {test_acc:.2f}%，已保存: {ckpt_path}")
            else:
                self.early_stop_counter += 1
                if self.patience is not None and self.early_stop_counter >= self.patience:
                    print(f"  早停触发：连续 {self.patience} 个 epoch 未提升，停止训练。")
                    break

        # 训练结束后：使用最佳权重加载后再画曲线/混淆矩阵，以保证曲线显示对应最佳
        best_ckpt = os.path.join(self.save_dir_checkpoint, "best_model.pth")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                print(f"[TrainerNAT] 已重新加载最佳权重 (Epoch {ckpt['epoch']}, Acc={ckpt['best_test_acc']:.2f}%)")

        # 用最佳权重再跑一次干净评估得到混淆矩阵
        _, _, final_labels, final_preds = self._evaluate()

        self._plot_training_curves()
        self._plot_confusion_matrix(final_labels, final_preds)
        self._save_metrics(final_labels, final_preds)

        print("\n" + "=" * 60)
        print(f"[TrainerNAT] 训练完成！最佳测试准确率: {self.best_test_acc:.2f}% (Epoch {self.best_epoch})")
        print("=" * 60)


# ------------------------------------------------------------------
# 训练后：α 敏感性扫描（与任务1扫描列表完全一致）
# ------------------------------------------------------------------
def evaluate_alpha_scan(
    model,
    test_loader,
    criterion,
    device,
    alpha_list,
    checkpoint_path,
    output_dir,
):
    """
    在已训练完成的 NAT 模型上执行 α 敏感性扫描。

    Args:
        model (nn.Module): 模型实例（已加载结构即可，内部会重新加载最佳权重）
        test_loader (DataLoader): 干净测试集
        criterion: 损失函数
        device (str): 设备
        alpha_list (list[float]): 扫描 α 列表
        checkpoint_path (str): 最佳权重路径（best_model.pth）
        output_dir (str): 保存 alpha_sensitivity.csv 的目录

    Returns:
        list[tuple]: [(alpha, accuracy, loss)，便于后续绘图
    """
    print("\n" + "-" * 60)
    print("[NAT 后 α 敏感性扫描] 开始执行...")
    print("-" * 60)

    # 加载最佳权重
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"NAT 训练权重不存在: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        ckpt_dataset = ckpt.get("dataset", "cifar10")
        print(f"[α 扫描] checkpoint 数据集来源: {ckpt_dataset}")
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[α 扫描] 已加载最佳权重: {checkpoint_path} (Epoch={ckpt['epoch']}, Acc={ckpt['best_test_acc']:.2f}%)")
        else:
            model.load_state_dict(ckpt)
            print(f"[α 扫描] 已加载权重 (直接 state_dict)")
    else:
        model.load_state_dict(ckpt)
        print(f"[α 扫描] 已加载权重 (非 dict)")

    model.to(device)
    model.eval()

    results = []

    for alpha in alpha_list:
        hooks = register_nonlinearity_hooks(model, alpha)
        total_loss = 0.0
        correct = 0
        total = 0
        try:
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

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        results.append((alpha, accuracy, avg_loss))
        print(f"[α 扫描] α = {alpha:+.2f} | Accuracy = {accuracy:.2f}% | Loss = {avg_loss:.4f}")

    # 保存 CSV
    csv_path = os.path.join(output_dir, "alpha_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "accuracy", "loss"])
        for alpha, acc, loss in results:
            writer.writerow([f"{alpha}", f"{acc:.4f}", f"{loss:.6f}"])
    print(f"[α 扫描] 敏感性 CSV 已保存: {csv_path}")
    return results


# ------------------------------------------------------------------
# 对比绘图：accuracy_vs_alpha_comparison.png
# ------------------------------------------------------------------
def plot_alpha_comparison(
    nat_results,
    output_dir,
    task1_csv_path="./outputs/task1_simplecnn/alpha_sensitivity.csv",
    mode_name="NAT Model",
):
    """
    绘制 α-精度对比曲线：
        曲线1：NAT 模型的 α 扫描结果
    曲线2：任务1干净基线（若文件存在则绘制，不存在则跳过）

    所有图中所有文字英文
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    nat_alphas = [r[0] for r in nat_results]
    nat_accs = [r[1] for r in nat_results]
    ax.plot(nat_alphas, nat_accs, marker="o", linewidth=2, label=mode_name, color="#d62728")
    ax.scatter(nat_alphas, nat_accs, color="#d62728", s=40, zorder=5)

    # 标注 α=0 基线点
    if 0.0 in nat_alphas:
        idx0 = nat_alphas.index(0.0)
        ax.scatter([0.0], [nat_accs[idx0]], color="red", marker="*", s=120,
                   zorder=10, label=f"{mode_name} Clean Baseline (α=0) = {nat_accs[idx0]:.2f}%")

    # 尝试读取任务1 CSV（若存在）
    task1_loaded = False
    if os.path.exists(task1_csv_path):
        try:
            t1_alphas = []
            t1_accs = []
            with open(task1_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t1_alphas.append(float(row["alpha"]))
                    t1_accs.append(float(row["accuracy"]))
            if len(t1_alphas) > 0 and len(t1_accs) > 0:
                ax.plot(t1_alphas, t1_accs, marker="s", linestyle="--", linewidth=2,
                        label="Task1 Baseline (Vanilla)", color="#1f77b4")
                task1_loaded = True
                print(f"[对比图] 已加载任务1基线，点数={len(t1_alphas)}")
        except Exception as e:
            print(f"[对比图] 任务1 CSV 加载失败（{e}），跳过对比")
    else:
        print(f"[对比图] 未找到任务1基线 CSV: {task1_csv_path}，跳过对比")

    ax.set_title("Model Accuracy vs Nonlinearity Strength (NAT vs Baseline)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Nonlinearity Strength (α)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "accuracy_vs_alpha_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[对比图] 已保存: {save_path}")


# ------------------------------------------------------------------
# 命令行参数
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="任务2：非线性感知训练（NAT）统一入口")

    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100"],
        help="数据集，默认: cifar10",
    )
    parser.add_argument(
        "--mode", type=str, required=True, choices=["finetune", "scratch"],
        help="训练模式：finetune=从干净预训练权重继续训练；scratch=随机初始化全程非线性感知训练",
    )
    parser.add_argument("--model", type=str, default="simple_cnn", help="模型名称，默认 simple_cnn")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="仅在 mode=finetune 时使用，默认自动拼接 {ckpt_root}/{model}/best_model.pth",
    )

    parser.add_argument(
        "--alpha_mode", type=str, default="random", choices=["random", "fixed"],
        help="α 注入策略：random=每 batch 随机采样；fixed=固定 α",
    )
    parser.add_argument(
        "--alpha_range", type=str, default="-0.3,0.3",
        help="random 模式下的 α 采样区间（闭区间，逗号分隔 low,high",
    )
    parser.add_argument("--alpha_fixed", type=float, default=0.3, help="fixed 模式下使用的固定 α")

    # epochs & lr 的默认值按 mode 自动赋值（在 main 中处理）
    parser.add_argument("--epochs", type=int, default=None, help="训练 epoch 数（finetune 默认 40，scratch 默认 120）")
    parser.add_argument("--lr", type=float, default=None, help="学习率（finetune 默认 1e-3，scratch 默认 0.01）")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="任务2输出根目录，默认 {outputs_root}/task2_{model_tag}/{mode}"
    )
    parser.add_argument("--patience", type=int, default=None, help="早停 patience（None 不启用)")

    parser.add_argument(
        "--tag", type=str, default="",
        help="产物目录后缀（第六阶段 P0 新增）。默认空 = 原行为不变；"
             "跑多种子时用 --tag _s43 之类避免覆盖既有产物。"
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def main():
    args = parse_args()

    # -------- Step0：准备工作 --------
    set_seed(args.seed)

    # 根据 mode 给 epochs / lr 默认值
    if args.epochs is None:
        args.epochs = 40 if args.mode == "finetune" else 120
    if args.lr is None:
        args.lr = 1e-3 if args.mode == "finetune" else 0.01

    # 设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 解析 alpha_range
    try:
        lo_s, hi_s = args.alpha_range.split(",")
        alpha_range = (float(lo_s.strip()), float(hi_s.strip()))
    except Exception as e:
        raise ValueError(f"--alpha_range 格式错误，应为 'low,high'，例如 '-0.3,0.3；收到 {args.alpha_range}") from e

    # 目录：根据 mode 创建子目录，自动包含模型名（simple_cnn → simplecnn，与目录重命名对齐）
    mode_tag = "finetune" if args.mode == "finetune" else "scratch"
    model_tag = args.model.replace("_", "")
    save_dir_checkpoint = os.path.join(get_ckpt_root(args.dataset), f"nat_{args.mode}_{model_tag}{args.tag}")
    if args.output_dir is not None:
        save_dir_output = args.output_dir
    else:
        save_dir_output = os.path.join(get_outputs_root(args.dataset), f"task2_{model_tag}", args.mode + args.tag)
    os.makedirs(save_dir_checkpoint, exist_ok=True)
    os.makedirs(save_dir_output, exist_ok=True)

    # 动态确定 task1 基线 CSV 路径
    task1_csv_path = os.path.join(get_outputs_root(args.dataset), f"task1_{model_tag}", "alpha_sensitivity.csv")
    print(f"[Task2] 数据集: {args.dataset}, 类别数: {get_num_classes(args.dataset)}")
    print(f"[Task2] 任务1基线 CSV 路径: {task1_csv_path}")

    # 打印完整配置摘要
    print("\n" + "=" * 70)
    print("[任务2：非线性感知训练（NAT）")
    print("=" * 70)
    print(f"  数据集 (dataset)         : {args.dataset}")
    print(f"  类别数 (num_classes)     : {get_num_classes(args.dataset)}")
    print(f"  训练模式 (mode)          : {args.mode}")
    print(f"  模型 (model)              : {args.model}")
    if args.mode == "finetune":
        print(f"  预训练权重 (checkpoint)    : {args.checkpoint}")
    print(f"  α 注入策略 (alpha_mode)    : {args.alpha_mode}")
    if args.alpha_mode == "random":
        print(f"  α 采样区间 (alpha_range)   : {alpha_range}")
    else:
        print(f"  固定 α (alpha_fixed)      : {args.alpha_fixed}")
    print(f"  训练 epochs                : {args.epochs}")
    print(f"  初始学习率 (lr)             : {args.lr}")
    print(f"  weight_decay              : {args.weight_decay}")
    print(f"  momentum                : {args.momentum}")
    print(f"  设备 (device)              : {device}")
    print(f"  权重保存目录             : {save_dir_checkpoint}")
    print(f"  输出保存目录             : {save_dir_output}")
    if args.patience is not None:
        print(f"  早停 patience           : {args.patience}")
    print("=" * 70)

    # -------- Step1：加载数据 --------
    print(f"\n[Step1] 加载 {args.dataset.upper()} 数据集...")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dataset=args.dataset,
    )

    # -------- Step2：创建模型（finetune 加载预训练权重） --------
    print(f"\n[Step2] 创建模型: {args.model}")
    model = get_model(args.model, num_classes=get_num_classes(args.dataset))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 总参数量: {total_params:,}，可训练: {trainable_params:,}")

    if args.mode == "finetune":
        if args.checkpoint is None:
            args.checkpoint = os.path.join(get_ckpt_root(args.dataset), args.model, "best_model.pth")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(
                f"finetune 模式下未找到预训练权重: {args.checkpoint}\n"
                f"请先运行: python train.py --model {args.model} --dataset {args.dataset}"
            )
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict):
            ckpt_dataset = ckpt.get("dataset", "cifar10")
            print(f"[Finetune] checkpoint 数据集来源: {ckpt_dataset}")
            if ckpt_dataset != args.dataset:
                print(f"[Finetune] WARNING: checkpoint 数据集 ({ckpt_dataset}) 与当前 --dataset "
                      f"({args.dataset}) 不一致，按兼容模式继续加载。")
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"[Finetune] 已加载干净预训练权重: {args.checkpoint}")
                if "best_test_acc" in ckpt:
                    print(f"[Finetune] 干净模型历史最佳准确率 = {float(ckpt['best_test_acc']):.2f}%")
            else:
                model.load_state_dict(ckpt)
                print(f"[Finetune] 已加载干净权重（直接 state_dict）: {args.checkpoint}")
        else:
            model.load_state_dict(ckpt)
            print(f"[Finetune] 已加载干净权重（直接 state_dict）: {args.checkpoint}")
    else:
        print("[Scratch] 保持随机初始化，从头开始非线性感知训练")

    # -------- Step3：损失 / 优化器 / 调度器 --------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0
    )

    # -------- Step4/5：TrainerNAT 训练 --------
    trainer_nat = TrainerNAT(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        model_name=f"{args.model}_{mode_tag}",
        save_dir_checkpoint=save_dir_checkpoint,
        save_dir_output=save_dir_output,
        patience=args.patience,
        alpha_mode=args.alpha_mode,
        alpha_range=alpha_range,
        alpha_fixed=args.alpha_fixed,
        num_classes=get_num_classes(args.dataset),
        dataset=args.dataset,
    )
    trainer_nat.train()

    # -------- Step6：α 敏感性扫描 --------
    best_ckpt_path = os.path.join(save_dir_checkpoint, "best_model.pth")
    nat_scan_results = evaluate_alpha_scan(
        model=get_model(args.model, num_classes=get_num_classes(args.dataset)),
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        alpha_list=TASK1_ALPHA_LIST,
        checkpoint_path=best_ckpt_path,
        output_dir=save_dir_output,
    )

    # -------- Step7：对比绘图 --------
    mode_display_name = "NAT Finetune" if args.mode == "finetune" else "NAT Scratch"
    plot_alpha_comparison(
        nat_results=nat_scan_results,
        output_dir=save_dir_output,
        task1_csv_path=task1_csv_path,
        mode_name=mode_display_name,
    )

    # 给 metrics.json 追加写回 train_mode
    metrics_path = os.path.join(save_dir_output, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        m["nat_train_mode"] = args.mode
        m["checkpoint_used"] = args.checkpoint if args.mode == "finetune" else None
        m["dataset"] = args.dataset
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=4, ensure_ascii=False)

    # -------- Step8：终端中文摘要 --------
    print("\n" + "=" * 70)
    print("【任务2 NAT 训练完成 - 最终摘要")
    print("=" * 70)
    print(f"训练模式              : {args.mode.upper()} ( {'微调自预训练权重' if args.mode == 'finetune' else '随机初始化从头训练'} )")
    if args.mode == "finetune":
        print(f"预训练权重路径        : {args.checkpoint}")
    print(f"干净测试集最佳准确率  : {trainer_nat.best_test_acc:.2f}% (Epoch {trainer_nat.best_epoch})")
    print(f"α 注入策略            : {args.alpha_mode}")
    if args.alpha_mode == "random":
        print(f"α 采样区间            : {alpha_range}")
    else:
        print(f"固定 α                : {args.alpha_fixed}")
    print()

    # 找 α=0 点（NAT clean baseline）
    clean_row = None
    for a, acc, loss in nat_scan_results:
        if abs(a - 0.0) < 1e-9:
            clean_row = (a, acc, loss)
            break
    if clean_row is not None:
        clean_acc_nat = clean_row[1]
        print(f"【NAT 模型 α 敏感性扫描结果】")
        print(f"{'Alpha':>8} | {'Accuracy':>10} | {'Acc Drop(vs α=0)':>16} | {'Loss':>10}")
        print("-" * 60)
        max_drop = 0.0
        max_drop_alpha = None
        for a, acc, loss in nat_scan_results:
            drop = clean_acc_nat - acc
            if drop > max_drop:
                max_drop = drop
                max_drop_alpha = a
            print(f"{a:>+8.2f} | {acc:>9.2f}% | {drop:>15.2f}% | {loss:>10.4f}")
        print()
        print(f"最大精度跌幅          : {max_drop:.2f}%  (α = {max_drop_alpha:+.2f})")

    # 与任务1基线对比（若文件存在）
    task1_csv = task1_csv_path
    if os.path.exists(task1_csv):
        try:
            t1_dict = {}
            with open(task1_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t1_dict[float(row["alpha"])] = float(row["accuracy"])
            # 对比 α=0.3（若都有）
            print()
            print("【与任务1 基线（未感知训练）对比】")
            headline = f"{'Alpha':>8} | {'Baseline Acc':>14} | {'NAT Acc':>10} | {'提升(%)':>10}"
            print(headline)
            print("-" * 60)
            for a, acc_nat, _ in nat_scan_results:
                if a in t1_dict:
                    acc_base = t1_dict[a]
                    uplift = acc_nat - acc_base
                    print(f"{a:>+8.2f} | {acc_base:>13.2f}% | {acc_nat:>9.2f}% | {uplift:>+9.2f}%")
        except Exception as e:
            print(f"[摘要] 任务1基线对比失败（{e}）")

    print()
    print(f"输出目录              : {save_dir_output}")
    print(f"最佳权重              : {best_ckpt_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
