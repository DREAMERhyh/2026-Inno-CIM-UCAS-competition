"""
拓展研究6：Exp2 鲁棒方案迁移至深层网络 —— 训练脚本
task_extension6_deep_robust_train.py

功能：
    将 SimpleCNN 上验证的 Exp2 方案（特征校准模块 + 逐层失真训练 NAT）迁移至
    VGG-11 和 ResNet-18 两个深层架构，从头训练（随机初始化，不加载预训练权重）。

训练配方（与 SimpleCNN Exp2 逐项一致，源自 task3_robust.py）：
    - epochs=120, lr=0.01, SGD(momentum=0.9, weight_decay=1e-4)
    - CosineAnnealingLR(T_max=epochs, eta_min=0.0)
    - batch_size=128, num_workers=2, seed=42
    - Exp2 配置：use_calibration=True, layerwise_alpha=True, asymmetric_sampling=False

NAT 逐层失真策略（迁移适配点 B）：
    α 数值范围与 SimpleCNN 逐项一致（浅 [-0.15,0.15]、中/默认 [-0.30,0.30]、fc 固定±0.3），
    但按各架构深度重新分组层名，保持"浅层轻、中层标准、深层强"语义。
    注入方式与 task3 一致：逐层独立采样 α + 每 batch 注册/移除 forward_pre_hook + try/finally。

评估：
    每个 epoch 同时评估 clean(α=0) 和 α=+0.3 的测试精度（task3 仅评估 clean，此处扩展）。

checkpoint 保存：
    vgg11   → ./checkpoints/Exp2_Calib+Layerwise_vgg11/best_model.pth
    resnet18→ ./checkpoints/Exp2_Calib+Layerwise_resnet18/best_model.pth
    格式与 task3 一致（dict 含 model_state_dict + best_test_acc + 三个开关 + exp_name）。

运行命令：
    python task_extension6_deep_robust_train.py --model vgg11
    python task_extension6_deep_robust_train.py --model resnet18
"""

import argparse
import os
import random
import time

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


# ======================================================================
# 基础工具
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
    print(f"[Seed] 随机种子已设置为 {seed}")


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    模型工厂，返回 RobustVGG11 / RobustResNet18（随机初始化）。

    训练流程为纯从头训练，此处不加载任何预训练权重。
    """
    if model_name == "vgg11":
        from models.robust_vgg11 import RobustVGG11
        return RobustVGG11(num_classes=num_classes)
    elif model_name == "resnet18":
        from models.robust_resnet import RobustResNet18
        return RobustResNet18(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型名称: '{model_name}'，当前支持 vgg11, resnet18")


def get_depth_group(model_name: str, module_name: str) -> str:
    """
    根据模型名与模块名返回深度分组：shallow / middle / deep / fc。

    迁移适配点 B：α 数值范围与 SimpleCNN 逐项一致，按各架构深度重新分组层名，
    保持"浅层轻、中层标准、深层强"语义。

    - 校准模块（含 "calibration" / "calib"）统一归 middle，与 SimpleCNN 的
      calibration 落入默认范围（[-0.30, 0.30]）一致。
    - VGG-11：block1-2 的 conv(features.0/4)→shallow；block3-5 的 conv→middle；
              classifier→fc。
    - ResNet-18：conv1+layer1→shallow；layer2/3→middle；layer4→deep；fc→fc。
      （deep 与 middle 数值同为 [-0.30,0.30]，仅语义区分。）
    """
    # 校准模块统一 middle（与 SimpleCNN 的 calibration 行为一致）
    if "calibration" in module_name or "calib" in module_name:
        return "middle"

    if model_name == "vgg11":
        if module_name.startswith("classifier"):
            return "fc"
        if module_name.startswith("features."):
            idx = int(module_name.split(".")[1])
            # 浅层：block1(features.0)、block2(features.4) 的卷积
            if idx in (0, 4):
                return "shallow"
            # block3-5 的卷积(8,11,15,18,22,25) → middle
            return "middle"
        return "middle"

    elif model_name == "resnet18":
        if module_name == "fc" or module_name.startswith("fc"):
            return "fc"
        if module_name == "conv1" or module_name.startswith("layer1."):
            return "shallow"
        if module_name.startswith("layer2.") or module_name.startswith("layer3."):
            return "middle"
        if module_name.startswith("layer4."):
            return "deep"
        return "middle"

    return "middle"


# ======================================================================
# 训练器：复刻 task3 的 TrainerRobust，适配深层网络 + 增加 α=0.3 评估
# ======================================================================

class TrainerDeepRobust:
    """
    深层网络鲁棒训练器（复刻 task3 TrainerRobust）。

    与 task3 TrainerRobust 的关键区别：
        1. _sample_alpha 用 get_depth_group 适配 VGG-11/ResNet-18 层名
        2. 新增 _evaluate_alpha(alpha) 方法，每 epoch 同时评估 α=+0.3 精度
        3. 训练完成后打印含训练时长的终端摘要
    其余（逐层 α 采样 + 钩子 try/finally、checkpoint 格式、超参数）与 task3 一致。
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
        model_name: str = "vgg11",
        save_dir_checkpoint: str = "./checkpoints/Exp2_Calib+Layerwise_vgg11",
        save_dir_output: str = "./outputs/extension6_deep_robust/vgg11",
        patience=None,
        use_calibration: bool = True,
        layerwise_alpha: bool = True,
        asymmetric_sampling: bool = False,
        exp_name: str = "Exp2_Calib+Layerwise",
        # ====== 双数据集支持参数 ======
        dataset: str = "cifar10",
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
        self.alpha03_acc_history = []   # 新增：α=+0.3 测试精度历史

        self.use_calibration = use_calibration
        self.layerwise_alpha = layerwise_alpha
        self.asymmetric_sampling = asymmetric_sampling
        self.exp_name = exp_name
        self.dataset = dataset

        self.model.to(self.device)

        print(f"[TrainerDeepRobust] 初始化完成，实验名称: {exp_name}")
        print(f"[TrainerDeepRobust] model={model_name}, use_calibration={use_calibration}, "
              f"layerwise_alpha={layerwise_alpha}, asymmetric_sampling={asymmetric_sampling}")
        print(f"[TrainerDeepRobust] 总 epochs={num_epochs}, 设备={self.device}")
        if self.patience is not None:
            print(f"[TrainerDeepRobust] 早停 patience={self.patience}")

    # ============ 【方案B 核心】分层 α 采样（适配深层网络层名） ============
    def _sample_alpha(self, module_name=None):
        """
        根据实验配置与模块深度分组，采样 α。

        α 数值范围与 SimpleCNN Exp2 逐项一致：
            shallow → [-0.15, 0.15]；middle/deep → [-0.30, 0.30]；fc → 固定 ±0.3
        asymmetric_sampling=False（Exp2）→ 对称均匀采样。
        """
        if self.layerwise_alpha and module_name is not None:
            group = get_depth_group(self.model_name, module_name)
            if group == "shallow":
                lo, hi = -0.15, 0.15
            elif group in ("middle", "deep"):
                lo, hi = -0.30, 0.30
            elif group == "fc":
                return 0.3 if random.random() > 0.5 else -0.3
            else:
                lo, hi = -0.30, 0.30
        else:
            lo, hi = -0.30, 0.30

        if self.asymmetric_sampling:
            if random.random() < 0.7:
                alpha = random.uniform(0.0, hi)
            else:
                alpha = random.uniform(lo, 0.0)
        else:
            alpha = random.uniform(lo, hi)
        return alpha

    # ============ 训练一个 epoch（逐层独立采样 α + try/finally 钩子安全） ============
    def _train_one_epoch(self, epoch: int):
        """与 task3 _train_one_epoch 逻辑一致：每个 Conv2d/Linear 独立采样 α 挂 pre_hook。"""
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

            hooks = []
            try:
                for name, module in self.model.named_modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        alpha = self._sample_alpha(module_name=name)

                        def make_hook(alpha_val=alpha):
                            def pre_hook_fn(m, inputs):
                                if not inputs:
                                    return inputs
                                x_d = inputs[0]
                                x_d = nonlinearity(x_d, alpha_val)
                                return (x_d,) + tuple(inputs[1:])
                            return pre_hook_fn

                        h = module.register_forward_pre_hook(make_hook())
                        hooks.append(h)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            finally:
                for h in hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass

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

    # ============ clean 验证（α=0，不挂任何非线性钩子） ============
    def _evaluate(self):
        """与 task3 _evaluate 一致：干净测试集评估。"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
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
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ============ 【新增】指定 α 下的验证（挂钩子推理后移除） ============
    def _evaluate_alpha(self, alpha: float):
        """
        在指定 α 非线性失真下评估测试精度（用 register_nonlinearity_hooks）。

        try/finally 保证钩子必被移除，不污染后续 epoch 训练。
        """
        self.model.eval()
        hooks = register_nonlinearity_hooks(self.model, alpha)
        total_loss = 0.0
        correct = 0
        total = 0
        try:
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
        finally:
            remove_hooks(hooks)
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ============ 保存最佳权重（格式与 task3 一致） ============
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
            "dataset": self.dataset,
        }, save_path)
        return save_path

    # ============ 绘制训练曲线（含 α=+0.3 鲁棒曲线） ============
    def _plot_training_curves(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), dpi=120)
        epochs_range = list(range(1, len(self.train_loss_history) + 1))

        ax1.plot(epochs_range, self.train_loss_history, label="Train Loss", color="#1f77b4", linewidth=2)
        ax1.plot(epochs_range, self.test_loss_history, label="Test Loss", color="#ff7f0e", linewidth=2)
        ax1.set_title(f"Loss Curves ({self.exp_name}, {self.model_name})", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, self.train_acc_history, label="Train Acc", color="#1f77b4", linewidth=2)
        ax2.plot(epochs_range, self.test_acc_history, label="Clean Test Acc", color="#ff7f0e", linewidth=2)
        if self.alpha03_acc_history:
            ax2.plot(epochs_range, self.alpha03_acc_history, label="Test Acc @ α=+0.3",
                     color="#d62728", linewidth=2, linestyle="--")

        if len(self.test_acc_history) > 0:
            best_idx = self.best_epoch
            if 1 <= best_idx <= len(self.test_acc_history):
                best_acc_value = self.test_acc_history[best_idx - 1]
                ax2.scatter(
                    best_idx, best_acc_value,
                    color="red", s=100, marker="*", zorder=5,
                    label=f"Best Clean Acc = {best_acc_value:.2f}% (Epoch {self.best_epoch})",
                )

        ax2.set_title(f"Accuracy Curves ({self.exp_name}, {self.model_name})", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir_output, "training_curves.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[TrainerDeepRobust] 训练曲线已保存: {save_path}")

    # ============ 保存 metrics.json ============
    def _save_metrics(self):
        metrics = {
            "dataset": self.dataset,
            "best_test_acc": round(self.best_test_acc, 4),
            "best_epoch": self.best_epoch,
            "total_epochs": self.num_epochs,
            "final_train_loss": self.train_loss_history[-1] if self.train_loss_history else None,
            "final_test_loss": self.test_loss_history[-1] if self.test_loss_history else None,
            "final_train_acc": self.train_acc_history[-1] if self.train_acc_history else None,
            "final_test_acc": self.test_acc_history[-1] if self.test_acc_history else None,
            "final_alpha03_acc": self.alpha03_acc_history[-1] if self.alpha03_acc_history else None,
            "use_calibration": self.use_calibration,
            "layerwise_alpha": self.layerwise_alpha,
            "asymmetric_sampling": self.asymmetric_sampling,
            "exp_name": self.exp_name,
            "model_name": self.model_name,
        }
        save_path = os.path.join(self.save_dir_output, "metrics.json")
        with open(save_path, "w", encoding="utf-8") as f:
            import json
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"[TrainerDeepRobust] 训练指标 JSON 已保存: {save_path}")

    # ============ 主训练循环 ============
    def train(self):
        start_time = time.time()
        print("\n" + "=" * 70)
        print(f"[TrainerDeepRobust] 开始训练: {self.exp_name} (model={self.model_name})")
        print("=" * 70)

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch)
            test_loss, test_acc = self._evaluate()                     # clean (α=0)
            alpha03_loss, alpha03_acc = self._evaluate_alpha(0.3)      # α=+0.3

            self.train_loss_history.append(train_loss)
            self.train_acc_history.append(train_acc)
            self.test_loss_history.append(test_loss)
            self.test_acc_history.append(test_acc)
            self.alpha03_acc_history.append(alpha03_acc)

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"[Epoch {epoch:3d}/{self.num_epochs}] "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
                f"Clean Loss={test_loss:.4f} Acc={test_acc:.2f}% | "
                f"α=+0.3 Acc={alpha03_acc:.2f}%"
            )

            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_epoch = epoch
                self.early_stop_counter = 0
                ckpt_path = self._save_best_checkpoint(epoch, test_acc)
                print(f"  >>> 新最佳！Clean Acc = {test_acc:.2f}%，保存: {ckpt_path}")
            else:
                self.early_stop_counter += 1
                if self.patience is not None and self.early_stop_counter >= self.patience:
                    print(f"  早停触发：连续 {self.patience} epoch 未提升，停止训练。")
                    break

        # 训练结束：重新加载最佳权重
        best_ckpt = os.path.join(self.save_dir_checkpoint, "best_model.pth")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=self.device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                print(f"[TrainerDeepRobust] 已重新加载最佳权重 "
                      f"(Epoch {ckpt.get('epoch')}, Acc={ckpt.get('best_test_acc'):.2f}%)")

        self._plot_training_curves()
        self._save_metrics()

        elapsed = time.time() - start_time
        final_alpha03 = self.alpha03_acc_history[-1] if self.alpha03_acc_history else None

        print("\n" + "=" * 70)
        print(f"[TrainerDeepRobust] 实验 {self.exp_name} (model={self.model_name}) 训练完成！")
        print("=" * 70)
        print(f"  训练时长          : {elapsed:.1f} 秒 ({elapsed / 60:.2f} 分钟)")
        print(f"  最佳 Clean Test Acc : {self.best_test_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"  最终 Clean Test Acc : {self.test_acc_history[-1]:.2f}%" if self.test_acc_history else "  最终 Clean Test Acc : N/A")
        if final_alpha03 is not None:
            print(f"  最终 α=+0.3 Test Acc: {final_alpha03:.2f}%")
        print(f"  权重保存目录       : {self.save_dir_checkpoint}")
        print(f"  输出目录           : {self.save_dir_output}")
        print("=" * 70)

        return self.best_test_acc


# ======================================================================
# argparse + main
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extension 6: Exp2 鲁棒方案迁移至深层网络（VGG-11 / ResNet-18）训练脚本"
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar10",
        choices=["cifar10", "cifar100"],
        help="数据集，默认: cifar10",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["vgg11", "resnet18"],
        help="训练模型：vgg11 或 resnet18",
    )
    parser.add_argument("--epochs", type=int, default=120, help="训练 epoch 数，默认 120（与 Exp2 一致）")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size，默认 128")
    parser.add_argument("--lr", type=float, default=0.01, help="初始学习率，默认 0.01（与 Exp2 一致）")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay，默认 1e-4")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum，默认 0.9")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载线程数，默认 2")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"], help="训练设备（默认自动）")
    parser.add_argument("--patience", type=int, default=None, help="早停 patience，None=不启用")
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 0：准备
    set_seed(args.seed)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = args.model
    model_tag = model_name.replace("_", "")
    exp_name = "Exp2_Calib+Layerwise"
    save_dir_checkpoint = os.path.join(get_ckpt_root(args.dataset), f"{exp_name}_{model_tag}")
    output_dir = os.path.join(get_outputs_root(args.dataset), f"extension6_deep_robust/{model_tag}")
    os.makedirs(save_dir_checkpoint, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("【Extension 6：Exp2 鲁棒方案迁移至深层网络 - 启动】")
    print(f"  --dataset      : {args.dataset}")
    print(f"  --model        : {args.model}")
    print(f"  --epochs       : {args.epochs}")
    print(f"  --lr           : {args.lr}")
    print(f"  --batch_size   : {args.batch_size}")
    print(f"  --weight_decay : {args.weight_decay}")
    print(f"  --momentum     : {args.momentum}")
    print(f"  --device       : {args.device}")
    print(f"  checkpoint 目录: {save_dir_checkpoint}")
    print(f"  输出目录        : {output_dir}")
    print("=" * 70)

    # Step 1：数据
    print(f"\n[Step 1] 加载 {args.dataset.upper()} 数据 ...")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        dataset=args.dataset,
    )

    # Step 2：模型（纯从头训练，随机初始化，不加载任何预训练权重）
    print(f"\n[Step 2] 创建模型 {model_name}（从头训练，随机初始化）...")
    model = get_model(model_name, num_classes=get_num_classes(args.dataset))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name} 总参数量: {total_params:,}，可训练: {trainable_params:,}")

    # Step 3：损失 / 优化器 / 调度器（与 SimpleCNN Exp2 一致）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.0,
    )

    # Step 4：训练
    trainer = TrainerDeepRobust(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        num_epochs=args.epochs,
        model_name=model_name,
        save_dir_checkpoint=save_dir_checkpoint,
        save_dir_output=output_dir,
        patience=args.patience,
        use_calibration=True,
        layerwise_alpha=True,
        asymmetric_sampling=False,
        exp_name=exp_name,
        dataset=args.dataset,
    )
    trainer.train()

    print("\n[Extension 6 Done]")


if __name__ == "__main__":
    main()
