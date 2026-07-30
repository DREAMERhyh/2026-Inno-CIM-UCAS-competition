"""
任务3：鲁棒性增强方法设计（主脚本）

功能：
    在 NAT-Scratch 的基础上叠加三种互补的鲁棒性增强方法，通过消融实验
    （ablation study）验证每种方法的独立贡献。

方法组合对应四个实验：
    Exp1 (--exp 1): 仅方案A（架构增强：残差校准模块）
    Exp2 (--exp 2): 方案A + 方案B（分层加权 α 采样）
    Exp3 (--exp 3): 方案A + 方案B + 方案C（非对称 70%正向/30%负向）
    All  (--exp all): 依次运行 Exp1 → Exp2 → Exp3，最后生成消融汇总。

运行命令示例：
    # 单独运行 Exp1
    python task3_robust.py --exp 1 --epochs 120
    # 单独运行 Exp3（完整方案）
    python task3_robust.py --exp 3 --epochs 120
    # 一键串联所有消融实验（推荐）
    python task3_robust.py --exp all
    # 自定义参数 + 早停
    python task3_robust.py --exp all --epochs 100 --lr 0.005 --patience 15
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
from utils.nonlinearity import (
    register_nonlinearity_hooks,
    remove_hooks,
)


# ------------------------------------------------------------------
# 基础工具：set_seed / get_model
# ------------------------------------------------------------------
def set_seed(seed: int = 42):
    """
    设置 Python / NumPy / PyTorch 随机种子，保证结果可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] 随机种子已设置为 {seed}")


def get_model(model_name: str, num_classes: int = 10, use_calibration: bool = True):
    """
    模型加载工厂。

    支持：
        - "simple_cnn" -> SimpleCNN（复用 models/simple_cnn.py
        - "robust_cnn" -> RobustCNN（任务3新增，支持 use_calibration 开关
    """
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)
    elif model_name == "robust_cnn":
        from models.robust_cnn import RobustCNN
        return RobustCNN(num_classes=num_classes, use_calibration=use_calibration)
    else:
        raise ValueError(f"不支持的模型名称: '{model_name}'")


# CIFAR-10 固定类别名（混淆矩阵绘制用）
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 与任务1、任务2 完全一致的 α 扫描列表，保证对比横轴一致
ALPHA_SCAN_LIST = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

# 任务2 NAT-Scratch 基线 CSV 路径（用于对比绘图）
NAT_SCRATCH_BASELINE_CSV = "./outputs/task2_simplecnn/scratch/alpha_sensitivity.csv"


# ------------------------------------------------------------------
# 实验配置：根据 --exp id 确定三个布尔开关 & 实验名称
# ------------------------------------------------------------------
def get_exp_config(exp_id: str):
    """
    将 --exp 字符串映射到消融实验配置。

    Returns:
        dict: 含 exp_id, exp_name, use_calibration, layerwise_alpha, asymmetric_sampling
    """
    if exp_id == "1":
        return {
            "exp_id": "1",
            "exp_name": "Exp1_CalibOnly",
            "use_calibration": True,
            "layerwise_alpha": False,
            "asymmetric_sampling": False,
        }
    elif exp_id == "2":
        return {
            "exp_id": "2",
            "exp_name": "Exp2_Calib+Layerwise",
            "use_calibration": True,
            "layerwise_alpha": True,
            "asymmetric_sampling": False,
        }
    elif exp_id == "3":
        return {
            "exp_id": "3",
            "exp_name": "Exp3_FullRobust",
            "use_calibration": True,
            "layerwise_alpha": True,
            "asymmetric_sampling": True,
        }
    else:
        raise ValueError(f"未知实验编号: exp_id='{exp_id}'，期望 '1'/'2'/'3'")


# ------------------------------------------------------------------
# TrainerRobust：任务3增强训练器（在 TrainerNAT 基础上加入分层 + 非对称采样
# ------------------------------------------------------------------
class TrainerRobust:
    """
    鲁棒增强版训练器。

    与任务2 TrainerNAT 的关键区别：
        1. 支持分层加权 α 范围（方案B）
            conv1/conv2: α ∈ [-0.15, 0.15]  浅层用轻失真
            conv3/conv4: α ∈ [-0.30, 0.30]  中层标准
            fc:          交替固定 ±0.3       深层强失真
        2. 支持非对称偏置采样（方案C）
            70% 概率从 [0, hi] 采样（正向强破坏场景更多）
            30% 概率从 [lo, 0) 采样（负向较少）
        3. _train_one_epoch 中对每个模块独立采样 α，挂独立的 pre-hook。
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
        num_epochs: int = 120,
        model_name: str = "robust_cnn",
        save_dir_checkpoint: str = "./checkpoints/Exp1_CalibOnly",
        save_dir_output: str = "./outputs/task3/Exp1_CalibOnly",
        patience=None,
        # ============ 任务3：三个增强开关 ============
        use_calibration: bool = True,
        layerwise_alpha: bool = False,
        asymmetric_sampling: bool = False,
        exp_name: str = "Exp1_CalibOnly",
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.model_name = model_name

        self.start_epoch = 0
        self.best_test_acc = 0.0
        self.best_epoch = 0

        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.save_dir_checkpoint = save_dir_checkpoint
        self.save_dir_output = save_dir_output
        os.makedirs(self.save_dir_checkpoint, exist_ok=True)
        os.makedirs(self.save_dir_output, exist_ok=True)

        self.patience = patience
        self.early_stop_counter = 0

        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []

        # 三个增强开关
        self.use_calibration = use_calibration
        self.layerwise_alpha = layerwise_alpha
        self.asymmetric_sampling = asymmetric_sampling
        self.exp_name = exp_name

        self.model.to(self.device)

        print(f"[TrainerRobust] 初始化完成，实验名称: {exp_name}")
        print(f"[TrainerRobust] use_calibration = {use_calibration}, "
              f"layerwise_alpha = {layerwise_alpha}, "
              f"asymmetric_sampling = {asymmetric_sampling}")
        print(f"[TrainerRobust] 总 epochs = {num_epochs}，设备 = {self.device}")
        if self.patience is not None:
            print(f"[TrainerRobust] 早停 patience = {self.patience}")

    # ============ 【方案B & C 核心】分层 α 采样 ============
    def _sample_alpha(self, module_name=None):
        """
        根据实验配置，为指定模块名采样 α。

        Args:
            module_name (str | None): 模块名（如 "conv1", "fc"；None 代表非分层默认范围）

        Returns:
            float: 采样得到的 α 值
        """
        # ---- 步骤1：确定 α 范围（方案B：分层加权） ----
        if self.layerwise_alpha and module_name is not None:
            if module_name in ("conv1", "conv2"):
                lo, hi = -0.15, 0.15                 # 浅层：轻失真
            elif module_name in ("conv3", "conv4"):
                lo, hi = -0.30, 0.30                 # 中层：标准
            elif module_name == "fc":
                # 深层：交替固定 ±0.3（每个 batch 随机选正或负
                return 0.3 if random.random() > 0.5 else -0.3
            else:
                lo, hi = -0.30, 0.30
        else:
            lo, hi = -0.30, 0.30                     # 非分层：统一范围

        # ---- 步骤2：在 [lo, hi] 内采样（方案C：非对称偏置） ----
        if self.asymmetric_sampling:
            # 70% 概率从 [0, hi] 采样正向（更常见的强破坏场景）
            # 30% 概率从 [lo, 0) 采样负向
            if random.random() < 0.7:
                alpha = random.uniform(0.0, hi)
            else:
                alpha = random.uniform(lo, 0.0)
        else:
            # 对称均匀采样
            alpha = random.uniform(lo, hi)
        return alpha

    # ============ 训练一个 epoch（每模块独立采样 α + try/finally 钩子安全） ============
    def _train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"[{self.exp_name}] Train {epoch}/{self.num_epochs}",
            leave=False,
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # ---- 为每个 Conv2d/Linear 模块 *独立* 采样 α，挂独立 pre-hook ----
            hooks = []
            try:
                for name, module in self.model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        alpha = self._sample_alpha(module_name=name)

                        # 闭包构造：捕获当前循环的 alpha / name，避免 Python 后期绑定坑
                        def make_hook(alpha_val=alpha):
                            def pre_hook_fn(m, inputs):
                                if not inputs:
                                    return inputs
                                x_d = inputs[0]
                                # 复用 utils.nonlinearity 中已有的逐样本归一化非线性函数
                                from utils.nonlinearity import nonlinearity
                                x_d = nonlinearity(x_d, alpha_val)
                                return (x_d,) + tuple(inputs[1:])
                            return pre_hook_fn

                        h = module.register_forward_pre_hook(make_hook())
                        hooks.append(h)

                # 标准前向 + 反向 + 更新
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            finally:
                # 保证即使 forward/backward 报错也移除所有钩子
                for h in hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass

            # 统计
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            cur_loss = total_loss / total
            cur_acc = 100.0 * correct / total
            progress_bar.set_postfix({
                "loss": f"{cur_loss:.4f}",
                "acc": f"{cur_acc:.2f}%",
            })

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ============ 干净验证（α=0，不挂任何非线性钩子） ============
    def _evaluate(self):
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

    # ============ 保存最佳权重 ============
    def _save_best_checkpoint(self, epoch, test_acc):
        save_path = os.path.join(self.save_dir_checkpoint, "best_model.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_test_acc": test_acc,
            "use_calibration": self.use_calibration,
            "layerwise_alpha": self.layerwise_alpha,
            "asymmetric_sampling": self.asymmetric_sampling,
            "exp_name": self.exp_name,
        }, save_path)
        return save_path

    # ============ 绘制训练曲线 ============
    def _plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), dpi=120)
        epochs_range = list(range(1, len(self.train_loss_history) + 1))

        ax1.plot(epochs_range, self.train_loss_history, label="Train Loss", color="#1f77b4", linewidth=2)
        ax1.plot(epochs_range, self.test_loss_history, label="Test Loss", color="#ff7f0e", linewidth=2)
        ax1.set_title(f"Loss Curves ({self.exp_name})", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, self.train_acc_history, label="Train Acc", color="#1f77b4", linewidth=2)
        ax2.plot(epochs_range, self.test_acc_history, label="Test Acc", color="#ff7f0e", linewidth=2)

        if len(self.test_acc_history) > 0:
            best_idx = self.best_epoch
            if 1 <= best_idx <= len(self.test_acc_history):
                best_acc_value = self.test_acc_history[best_idx - 1]
                ax2.scatter(
                    best_idx, best_acc_value,
                    color="red", s=100, marker="*", zorder=5,
                    label=f"Best Test Acc = {best_acc_value:.2f}% (Epoch {self.best_epoch})",
                )

        ax2.set_title(f"Accuracy Curves ({self.exp_name})", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir_output, "training_curves.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[TrainerRobust] 训练曲线已保存: {save_path}")

    # ============ 混淆矩阵 ============
    def _plot_confusion_matrix(self, all_labels, all_preds):
        cm = confusion_matrix(all_labels, all_preds)
        fig, ax = plt.subplots(figsize=(9, 8), dpi=120)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASSES)
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
        ax.set_title(f"Confusion Matrix - Test Set ({self.exp_name})", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = os.path.join(self.save_dir_output, "confusion_matrix.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[TrainerRobust] 混淆矩阵已保存: {save_path}")

    # ============ 保存 metrics.json（含任务3的三个开关） ============
    def _save_metrics(self):
        metrics = {
            "best_test_acc": self.best_test_acc,
            "best_epoch": self.best_epoch,
            "total_epochs": self.num_epochs,
            "final_train_loss": self.train_loss_history[-1] if self.train_loss_history else None,
            "final_test_loss": self.test_loss_history[-1] if self.test_loss_history else None,
            "final_train_acc": self.train_acc_history[-1] if self.train_acc_history else None,
            "final_test_acc": self.test_acc_history[-1] if self.test_acc_history else None,
            "use_calibration": self.use_calibration,
            "layerwise_alpha": self.layerwise_alpha,
            "asymmetric_sampling": self.asymmetric_sampling,
            "exp_name": self.exp_name,
        }
        save_path = os.path.join(self.save_dir_output, "metrics.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"[TrainerRobust] 训练指标 JSON 已保存: {save_path}")

    # ============ 主训练循环 ============
    def train(self):
        print("\n" + "=" * 70)
        print(f"[TrainerRobust] 开始训练实验: {self.exp_name}")
        print("=" * 70)

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch)
            test_loss, test_acc, all_labels, all_preds = self._evaluate()

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_acc)

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"[Epoch {epoch:3d}/{self.num_epochs}] "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%"
            )

            improved = False
            if test_acc > self.best_test_acc:
                improved = True
                self.best_test_acc = test_acc
                self.best_epoch = epoch
                self.early_stop_counter = 0
                ckpt_path = self._save_best_checkpoint(epoch, test_acc)
                print(f"  >>> 新最佳！Clean Test Acc = {test_acc:.2f}%，保存: {ckpt_path}")
            else:
                self.early_stop_counter += 1
                if self.patience is not None and self.early_stop_counter >= self.patience:
                    print(f"  早停触发：连续 {self.patience} epoch 未提升，停止训练。")
                    break

        # 训练结束：重新加载最佳权重再做一次评估，保证后续混淆矩阵/α 扫描用的是最佳模型
        best_ckpt = os.path.join(self.save_dir_checkpoint, "best_model.pth")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                print(f"[TrainerRobust] 已重新加载最佳权重 (Epoch {ckpt['epoch']}, Acc={ckpt['best_test_acc']:.2f}%)")

        _, _, final_labels, final_preds = self._evaluate()
        self._plot_training_curves()
        self._plot_confusion_matrix(final_labels, final_preds)
        self._save_metrics()

        print("\n" + "=" * 70)
        print(f"[TrainerRobust] 实验 {self.exp_name} 训练完成！")
        print(f"  最佳 Clean Test Acc: {self.best_test_acc:.2f}% (Epoch {self.best_epoch})")
        print("=" * 70)

        # 返回 clean test best acc，方便消融汇总
        return self.best_test_acc


# ------------------------------------------------------------------
# α 敏感性扫描（训练后）
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
    在训练后模型上执行 α 敏感性扫描，保存 alpha_sensitivity.csv。
    （逻辑与任务2 evaluate_alpha_scan 一致）
    """
    print(f"\n[α 扫描] 开始执行...（checkpoint={checkpoint_path}）")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重不存在: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[α 扫描] 已加载最佳权重 (Epoch={ckpt.get('epoch')}, Acc={ckpt.get('best_test_acc'):.2f}%)")
    else:
        model.load_state_dict(ckpt)

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
        print(f"[α 扫描] α={alpha:+.2f} | Acc={accuracy:.2f}% | Loss={avg_loss:.4f}")

    csv_path = os.path.join(output_dir, "alpha_sensitivity.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "accuracy", "loss"])
        for alpha, acc, loss in results:
            writer.writerow([f"{alpha}", f"{acc:.4f}", f"{loss:.6f}"])
    print(f"[α 扫描] 结果已保存: {csv_path}")
    return results


# ------------------------------------------------------------------
# 对比绘图：当前实验 vs NAT-Scratch 基线
# ------------------------------------------------------------------
def plot_comparison(
    current_results,
    output_dir,
    current_label: str,
    baseline_csv_path=NAT_SCRATCH_BASELINE_CSV,
    plot_filename: str = "accuracy_vs_alpha_comparison.png",
):
    """
    绘制 α-精度对比曲线（单实验 vs NAT-Scratch 基线）
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    cur_alphas = [r[0] for r in current_results]
    cur_accs = [r[1] for r in current_results]
    ax.plot(cur_alphas, cur_accs, marker="o", linewidth=2.2, label=current_label, color="#2ca02c")
    ax.scatter(cur_alphas, cur_accs, color="#2ca02c", s=45, zorder=5)

    # α=0 红点
    if 0.0 in cur_alphas:
        idx0 = cur_alphas.index(0.0)
        ax.scatter([0.0], [cur_accs[idx0]], color="red", marker="*", s=140, zorder=10,
                   label=f"{current_label} α=0 = {cur_accs[idx0]:.2f}%")

    # 加载 NAT-Scratch 基线（若存在）
    if os.path.exists(baseline_csv_path):
        try:
            t1_alphas = []
            t1_accs = []
            with open(baseline_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t1_alphas.append(float(row["alpha"]))
                    t1_accs.append(float(row["accuracy"]))
            if len(t1_alphas) > 0:
                ax.plot(t1_alphas, t1_accs, marker="s", linestyle="--", linewidth=2,
                        label="NAT-Scratch (Baseline)", color="#1f77b4")
                print(f"[对比图] 已加载 NAT-Scratch 基线: {baseline_csv_path}")
        except Exception as e:
            print(f"[对比图] 加载 NAT-Scratch 基线失败: {e}")
    else:
        print(f"[对比图] 未找到 NAT-Scratch 基线 CSV: {baseline_csv_path}，跳过基线绘制")

    ax.set_title(f"Accuracy vs Nonlinearity Strength ({current_label} vs NAT-Scratch)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Nonlinearity Strength (α)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    save_path = os.path.join(output_dir, plot_filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[对比图] 已保存: {save_path}")


# ------------------------------------------------------------------
# 消融汇总：仅 --exp all 时运行
# ------------------------------------------------------------------
def generate_ablation_summary(
    all_exp_configs,
    exp_results_dict,
    task3_output_root="./outputs/task3",
    baseline_csv_path=NAT_SCRATCH_BASELINE_CSV,
):
    """
    生成：
        1) ablation_summary.csv
        2) ablation_comparison.png（3 条消融 + 1 条 NAT-Scratch 基线）
        3) 终端中文摘要表格

    Args:
        all_exp_configs (list[dict]): [get_exp_config("1"), get_exp_config("2"), get_exp_config("3")]
        exp_results_dict (dict): exp_name -> {"clean_acc": float, "alpha_scan": list[(alpha, acc, loss)]}
    """
    os.makedirs(task3_output_root, exist_ok=True)

    # 1) 生成 ablation_summary.csv
    rows = []
    for cfg in all_exp_configs:
        exp_name = cfg["exp_name"]
        if exp_name not in exp_results_dict:
            continue
        clean_acc = exp_results_dict[exp_name]["clean_acc"]
        scan = exp_results_dict[exp_name]["alpha_scan"]
        acc_at_03 = None
        for a, acc, _ in scan:
            if abs(a - 0.3) < 1e-9:
                acc_at_03 = acc
                break
        acc_drop_at_03 = clean_acc - acc_at_03 if acc_at_03 is not None else None
        rows.append({
            "exp_name": exp_name,
            "use_calibration": cfg["use_calibration"],
            "layerwise_alpha": cfg["layerwise_alpha"],
            "asymmetric_sampling": cfg["asymmetric_sampling"],
            "clean_acc": round(clean_acc, 4) if clean_acc is not None else None,
            "acc_at_alpha_0.3": round(acc_at_03, 4) if acc_at_03 is not None else None,
            "acc_drop_at_0.3": round(acc_drop_at_03, 4) if acc_drop_at_03 is not None else None,
        })

    csv_path = os.path.join(task3_output_root, "ablation_summary.csv")
    fieldnames = [
        "exp_name", "use_calibration", "layerwise_alpha", "asymmetric_sampling",
        "clean_acc", "acc_at_alpha_0.3", "acc_drop_at_0.3",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n[消融汇总] CSV 已保存: {csv_path}")

    # 2) 绘制消融对比曲线（Exp1 / Exp2 / Exp3 + NAT-Scratch）
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    for i, cfg in enumerate(all_exp_configs):
        exp_name = cfg["exp_name"]
        if exp_name not in exp_results_dict:
            continue
        scan = exp_results_dict[exp_name]["alpha_scan"]
        alphas = [s[0] for s in scan]
        accs = [s[1] for s in scan]
        ax.plot(alphas, accs, marker=markers[i], linewidth=2, color=colors[i], label=exp_name)

    # NAT-Scratch 虚线基线
    if os.path.exists(baseline_csv_path):
        try:
            t1_alphas = []
            t1_accs = []
            with open(baseline_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t1_alphas.append(float(row["alpha"]))
                    t1_accs.append(float(row["accuracy"]))
            ax.plot(t1_alphas, t1_accs, linestyle="--", color="#d62728", linewidth=2.2,
                    label="NAT-Scratch (Baseline)")
        except Exception:
            pass

    ax.set_title("Ablation Study: Accuracy vs Nonlinearity Strength",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Nonlinearity Strength (α)", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    png_path = os.path.join(task3_output_root, "ablation_comparison.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[消融汇总] 对比图已保存: {png_path}")

    # 3) 终端中文摘要表格
    print("\n" + "=" * 90)
    print("【任务3 消融实验汇总】")
    print("=" * 90)
    header = (f"{'Experiment':<26} | {'Calib':<6} {'LayerW':<8} {'Asym':<6} | "
              f"{'CleanAcc':>9} | {'Acc@α=0.3':>10} | {'Drop@0.3':>9}")
    print(header)
    print("-" * 90)
    for row in rows:
        calib = "✓" if row["use_calibration"] else "✗"
        layer = "✓" if row["layerwise_alpha"] else "✗"
        asym = "✓" if row["asymmetric_sampling"] else "✗"
        clean = f"{row['clean_acc']:.2f}%" if row["clean_acc"] is not None else "N/A"
        a03 = f"{row['acc_at_alpha_0.3']:.2f}%" if row["acc_at_alpha_0.3"] is not None else "N/A"
        drop = f"{row['acc_drop_at_0.3']:.2f}%" if row["acc_drop_at_0.3"] is not None else "N/A"
        print(f"{row['exp_name']:<26} | {calib:<6} {layer:<8} {asym:<6} | "
              f"{clean:>9} | {a03:>10} | {drop:>9}")
    print("=" * 90)


# ------------------------------------------------------------------
# run_experiment(exp_id)：单个消融实验的完整流程
# ------------------------------------------------------------------
def run_experiment(exp_id: str, args) -> dict:
    """
    执行单个消融实验（Exp1 / Exp2 / Exp3）。

    Returns:
        dict: {"exp_name": str, "clean_acc": float, "alpha_scan": list[(alpha, acc, loss)]}
              供 --exp all 的消融汇总使用
    """
    cfg = get_exp_config(exp_id)
    exp_name = cfg["exp_name"]
    print("\n" + "#" * 80)
    print(f"### 开始执行实验: {exp_name} (use_calib={cfg['use_calibration']}, "
          f"layerwise={cfg['layerwise_alpha']}, asymmetric={cfg['asymmetric_sampling']})")
    print("#" * 80)

    # 目录：自动包含模型名（simple_cnn → simplecnn，与目录重命名对齐）
    model_tag = args.model.replace("_", "")
    save_dir_checkpoint = f"./checkpoints/{exp_name}_{model_tag}"
    output_dir = os.path.join(args.output_dir, f"{exp_name}_{model_tag}")
    checkpoint_dir = save_dir_checkpoint
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[路径] 输出目录: {output_dir}")
    print(f"[路径] 权重目录: {checkpoint_dir}")

    # 1) 数据
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 2) 模型
    model_name = args.model
    # 只有 robust_cnn 支持 use_calibration 开关；simple_cnn 忽略该开关
    if model_name == "robust_cnn":
        model = get_model(model_name, num_classes=10, use_calibration=cfg["use_calibration"])
    else:
        model = get_model(model_name, num_classes=10)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name} 总参数量: {total_params:,}，可训练: {trainable_params:,}")

    # 3) 损失 / 优化器 / 调度器（与任务2 NAT-Scratch 保持一致
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

    # 4) TrainerRobust 训练
    trainer = TrainerRobust(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs=args.epochs,
        model_name=model_name,
        save_dir_checkpoint=checkpoint_dir,
        save_dir_output=output_dir,
        patience=args.patience,
        use_calibration=cfg["use_calibration"],
        layerwise_alpha=cfg["layerwise_alpha"],
        asymmetric_sampling=cfg["asymmetric_sampling"],
        exp_name=exp_name,
    )
    clean_best_acc = trainer.train()

    # 5) α 敏感性扫描（用新的模型实例 + 加载 best 权重，避免 trainer 内 model 的状态干扰
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pth")
    if model_name == "robust_cnn":
        scan_model = get_model(model_name, num_classes=10, use_calibration=cfg["use_calibration"])
    else:
        scan_model = get_model(model_name, num_classes=10)
    alpha_scan_results = evaluate_alpha_scan(
        model=scan_model,
        test_loader=test_loader,
        criterion=criterion,
        device=trainer.device,
        alpha_list=ALPHA_SCAN_LIST,
        checkpoint_path=best_ckpt_path,
        output_dir=output_dir,
    )

    # 6) 单实验对比图（当前实验 vs NAT-Scratch 基线）
    plot_comparison(
        current_results=alpha_scan_results,
        output_dir=output_dir,
        current_label=exp_name,
    )

    # 7) 终端中文摘要（单实验）
    print("\n" + "=" * 80)
    print(f"【{exp_name} - 单实验摘要】")
    print("=" * 80)
    print(f"实验配置: use_calibration={cfg['use_calibration']}, "
          f"layerwise_alpha={cfg['layerwise_alpha']}, "
          f"asymmetric_sampling={cfg['asymmetric_sampling']}")
    print(f"Clean 测试集最佳准确率: {clean_best_acc:.2f}%")
    print()

    clean_row = None
    for a, acc, _ in alpha_scan_results:
        if abs(a - 0.0) < 1e-9:
            clean_row = (a, acc, _)
            break
    if clean_row is not None:
        clean_acc_alpha0 = clean_row[1]
        print(f"{'Alpha':>8} | {'Accuracy':>10} | {'Drop vs α=0':>13} | {'Loss':>10}")
        print("-" * 60)
        max_drop = 0.0
        max_drop_alpha = None
        for a, acc, loss in alpha_scan_results:
            drop = clean_acc_alpha0 - acc
            if drop > max_drop:
                max_drop = drop
                max_drop_alpha = a
            print(f"{a:>+8.2f} | {acc:>9.2f}% | {drop:>12.2f}% | {loss:>10.4f}")
        print()
        print(f"最大精度跌幅: {max_drop:.2f}%  (α = {max_drop_alpha:+.2f})")

    # 与 NAT-Scratch 对比（若基线 CSV 存在）
    if os.path.exists(NAT_SCRATCH_BASELINE_CSV):
        try:
            base_dict = {}
            with open(NAT_SCRATCH_BASELINE_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    base_dict[float(row["alpha"])] = float(row["accuracy"])
            print()
            print("【对比 NAT-Scratch 基线】")
            print(f"{'Alpha':>8} | {'NAT-Scratch':>13} | {exp_name+' Acc':>16} | {'Uplift':>9}")
            print("-" * 65)
            for a, acc_nat, _ in alpha_scan_results:
                if a in base_dict:
                    base_acc = base_dict[a]
                    uplift = acc_nat - base_acc
                    print(f"{a:>+8.2f} | {base_acc:>12.2f}% | {acc_nat:>15.2f}% | {uplift:>+8.2f}%")
        except Exception as e:
            print(f"[摘要] NAT-Scratch 对比失败: {e}")

    print()
    print(f"输出目录: {output_dir}")
    print(f"最佳权重: {best_ckpt_path}")
    print("=" * 80)

    return {
        "exp_name": exp_name,
        "clean_acc": clean_best_acc,
        "alpha_scan": alpha_scan_results,
    }


# ------------------------------------------------------------------
# argparse + main
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="任务3：鲁棒性增强方法设计（消融实验统一入口）")
    parser.add_argument(
        "--exp", type=str, default="all",
        choices=["1", "2", "3", "all"],
        help="选择实验：'1'=仅方案A，'2'=A+B，'3'=A+B+C 完整方案，'all'=依次运行 1→2→3 并生成消融汇总",
    )
    parser.add_argument("--model", type=str, default="robust_cnn",
                        help="模型名称，默认 robust_cnn（任务3默认使用新架构）")
    parser.add_argument("--epochs", type=int, default=120, help="训练 epoch 数，默认 120（与 NAT-Scratch 对齐）")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01, help="初始学习率，默认 0.01（与 NAT-Scratch 对齐）")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/task3_simplecnn", help="任务3输出根目录")
    parser.add_argument("--patience", type=int, default=None, help="早停 patience，None=不启用")
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 0：准备工作
    set_seed(args.seed)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("【任务3：鲁棒性增强方法设计 - 启动】")
    print(f"  --exp          : {args.exp}")
    print(f"  --model        : {args.model}")
    print(f"  --epochs       : {args.epochs}")
    print(f"  --lr           : {args.lr}")
    print(f"  --batch_size   : {args.batch_size}")
    print(f"  --device       : {args.device}")
    print(f"  --output_dir   : {args.output_dir}")
    if args.patience is not None:
        print(f"  --patience     : {args.patience}")
    print("=" * 80)

    if args.exp == "all":
        # 依次运行 Exp1 → Exp2 → Exp3，并收集结果用于消融汇总
        all_exp_configs = [get_exp_config(i) for i in ["1", "2", "3"]]
        results_dict = {}
        for i in ["1", "2", "3"]:
            res = run_experiment(i, args)
            results_dict[res["exp_name"]] = res

        # 生成消融汇总（CSV + 图 + 终端表格）
        generate_ablation_summary(
            all_exp_configs=all_exp_configs,
            exp_results_dict=results_dict,
            task3_output_root=args.output_dir,
        )
        print("\n" + "=" * 80)
        print("【任务3全部消融实验完成！】")
        print(f"  消融汇总 CSV: {os.path.join(args.output_dir, 'ablation_summary.csv')}")
        print(f"  消融汇总图  : {os.path.join(args.output_dir, 'ablation_comparison.png')}")
        print("=" * 80)
    else:
        # 单实验运行
        run_experiment(args.exp, args)


if __name__ == "__main__":
    main()
