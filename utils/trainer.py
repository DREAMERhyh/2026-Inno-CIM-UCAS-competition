"""
通用训练器模块 Trainer

功能：
    - 通用的模型训练与验证流程（支持任意 nn.Module 模型）
    - 自动记录每个 epoch 的 loss、准确率
    - 根据验证集（测试集）准确率保存最佳模型权重
    - 训练结束后绘制 loss/acc 曲线、混淆矩阵，保存 metrics.json
    - 支持可选早停 (Early Stopping)
    - 支持从指定 epoch 续训 (Resume Training)：
        * __init__ 新增 start_epoch、best_test_acc 参数
        * 训练循环从 start_epoch+1 开始，直至 num_epochs
        * 训练开始前自动将 scheduler.step() 调用 start_epoch 次，使学习率处于正确阶段
        * 保存最佳模型的逻辑继续基于传入的历史 best_test_acc 判断

CIFAR-10 类别：
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
"""

import os
import json
import numpy as np
import matplotlib
# ============ Task1 新增：图表全局字体为英文 DejaVu Sans ============
# 确保所有 trainer 生成的图表（曲线、混淆矩阵）中的标题、坐标轴、图例均为英文
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.paths import CIFAR10_CLASSES


class Trainer:
    """
    通用深度学习训练器，封装完整的训练、验证、评估与结果保存流程。

    用法示例（从头训练）：
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=50,
            model_name="simple_cnn",
            save_dir_checkpoint="./checkpoints/simple_cnn",
            save_dir_output="./outputs/simple_cnn",
            patience=None,
        )
        trainer.train()

    用法示例（续训，假设已训练到 epoch 50，best_test_acc=84.79）：
        trainer = Trainer(
            ...,
            num_epochs=100,          # 总目标 epoch
            start_epoch=50,          # 已训练完成的 epoch 数
            best_test_acc=84.79,     # 历史最佳测试准确率
        )
        # 实际训练将从 epoch 51 开始，一直到 epoch 100
        trainer.train()
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
        save_dir_checkpoint: str = "./checkpoints/model",
        save_dir_output: str = "./outputs/model",
        patience: int = None,
        # ====== 续训相关新增参数（Resume Training） ======
        start_epoch: int = 0,
        best_test_acc: float = 0.0,
        # ====== 双数据集支持新增参数 ======
        num_classes: int = 10,
        dataset: str = "cifar10",
    ):
        """
        初始化训练器。

        Args:
            model (nn.Module): 待训练的模型（任意继承 nn.Module 的类）
            train_loader (DataLoader): 训练集数据加载器
            test_loader (DataLoader): 测试集数据加载器（同时作为验证集使用）
            criterion: 损失函数，例如 nn.CrossEntropyLoss()
            optimizer: 优化器，例如 SGD、Adam 等
            scheduler: 学习率调度器（可选，如 CosineAnnealingLR）
            device: 训练设备，"cuda" 或 "cpu"，默认为自动判断
            num_epochs (int): 总训练轮数（含之前已完成的轮数，续训时应设置为总目标）
            model_name (str): 模型名称字符串，用于打印和记录
            save_dir_checkpoint (str): 模型权重保存目录
            save_dir_output (str): 训练曲线、混淆矩阵、指标保存目录
            patience (int | None): 早停耐心轮数，None 表示不启用早停
            start_epoch (int): 【续训新增】已完成训练的 epoch 数，默认为 0（从头训练）
                               实际第一个执行的 epoch 编号 = start_epoch + 1
            best_test_acc (float): 【续训新增】之前训练阶段的最佳测试准确率（0~100 百分比），
                                   默认为 0.0（从头训练）。续训模式下新的 epoch 准确率超过该
                                   值才会触发最佳模型保存。
            num_classes (int): 分类类别数（默认 10），用于混淆矩阵标签。
            dataset (str): 数据集名称（默认 "cifar10"），写入 checkpoint 与 metrics 的溯源字段。
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.model_name = model_name

        # ====== 续训参数赋值（新增） ======
        self.start_epoch = start_epoch
        # 历史最佳测试准确率：续训时由 train.py 从 checkpoint 中读取并传入
        # 从头训练时默认为 0.0，从头开始比较
        self.best_test_acc = best_test_acc
        # 最佳 epoch：从头训练设为 0；续训时暂时设为 start_epoch（若新 epoch 提升时会被覆盖）
        self.best_epoch = start_epoch if start_epoch > 0 else 0

        # 双数据集支持：类别数（用于混淆矩阵标签）与数据集名称（写入溯源字段）
        self.num_classes = num_classes
        self.dataset = dataset

        # 自动检测可用设备
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 保存目录
        self.save_dir_checkpoint = save_dir_checkpoint
        self.save_dir_output = save_dir_output
        # 自动创建目录，若不存在则递归创建
        os.makedirs(self.save_dir_checkpoint, exist_ok=True)
        os.makedirs(self.save_dir_output, exist_ok=True)

        # 早停设置
        self.patience = patience  # None 表示不启用早停
        self.early_stop_counter = 0  # 连续多少个 epoch 未提升

        # 历史记录：每个 epoch 对应一个值
        # 说明：当前实现不强制从 checkpoint 恢复历史 loss/acc 曲线，
        # 因此续训模式下曲线只会显示本次续训的增量部分；
        # 如需完整曲线，可在 train.py 中进一步从旧 metrics.json/history 加载并赋值。
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []

        # 将模型移动到指定设备
        self.model.to(self.device)

        # ====== 训练器初始化日志（增加续训信息打印） ======
        print(f"[Trainer] 初始化完成，模型: {model_name}, 设备: {self.device}")
        print(f"[Trainer] Epochs: 总目标 {num_epochs}，已完成 {start_epoch}，"
              f"本次训练区间 [{start_epoch + 1}, {num_epochs}]，"
              f"共 {max(0, num_epochs - start_epoch)} 个 epoch")
        if self.start_epoch > 0:
            print(f"[Trainer] 续训模式：历史最佳测试准确率 = {self.best_test_acc:.2f}%")
        if self.patience is not None:
            print(f"[Trainer] 早停已启用，patience = {self.patience}")

    # ------------------------------------------------------------------
    # 单 epoch 训练
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int):
        """
        在训练集上执行一个 epoch 的训练。

        Args:
            epoch (int): 当前 epoch 编号（用于进度条显示，范围：1~num_epochs）

        Returns:
            tuple: (avg_loss, accuracy)
                - avg_loss (float): 当前 epoch 的平均训练 loss
                - accuracy (float): 当前 epoch 的训练集准确率（百分比，0~100）
        """
        self.model.train()  # 切换到训练模式（启用 Dropout、BatchNorm 的训练统计等）

        total_loss = 0.0
        correct = 0
        total = 0

        # tqdm 进度条：显示当前 epoch / 总 epoch、batch 进度、loss 和 acc
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Train Epoch {epoch}/{self.num_epochs}",
            leave=False,  # epoch 结束后清除该进度条
        )

        for batch_idx, (images, labels) in enumerate(progress_bar):
            # 将数据移动到 GPU（如果可用）
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向传播：清零梯度 -> 计算 logits -> 计算 loss
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            # 反向传播 + 参数更新
            loss.backward()
            self.optimizer.step()

            # 统计 loss 与准确率
            total_loss += loss.item() * images.size(0)  # loss 是 batch 的均值，乘 batch size 求累计
            _, predicted = torch.max(logits, 1)  # 取每个样本概率最大的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条右侧的实时信息
            avg_loss_now = total_loss / total
            acc_now = 100.0 * correct / total
            progress_bar.set_postfix({
                "loss": f"{avg_loss_now:.4f}",
                "acc": f"{acc_now:.2f}%",
            })

        progress_bar.close()

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # 验证/测试（在整个测试集上评估）
    # ------------------------------------------------------------------
    def _evaluate(self):
        """
        在测试集（或验证集）上评估模型表现。

        Returns:
            tuple: (avg_loss, accuracy, all_labels, all_predictions)
                - avg_loss (float): 平均测试 loss
                - accuracy (float): 测试集准确率（百分比，0~100）
                - all_labels (np.ndarray): 所有样本的真实标签，形状 (N,)
                - all_predictions (np.ndarray): 所有样本的预测标签，形状 (N,)
        """
        self.model.eval()  # 切换到评估模式（关闭 Dropout，BatchNorm 使用 running stats）

        total_loss = 0.0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []

        # 禁用梯度计算，节省显存与计算
        with torch.no_grad():
            for images, labels in tqdm(
                self.test_loader,
                desc=f"Evaluate   ",
                leave=False,
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 保存所有真实标签和预测标签，用于后续混淆矩阵计算
                all_labels.append(labels.cpu().numpy())
                all_predictions.append(predicted.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total

        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)

        return avg_loss, accuracy, all_labels, all_predictions

    # ------------------------------------------------------------------
    # 主训练流程
    # ------------------------------------------------------------------
    def train(self):
        """
        执行完整的训练流程，包含：
            0. （续训时）手动将 scheduler.step() 调用 start_epoch 次，使学习率回到正确阶段
            1. 逐 epoch 训练 + 验证（从 start_epoch+1 到 num_epochs）
            2. 保存最佳模型权重（与历史 best_test_acc 比较）
            3. 可选早停
            4. 训练结束后绘制曲线、混淆矩阵、保存指标
        """
        print("=" * 70)
        if self.start_epoch == 0:
            print(f"[Trainer] 开始从头训练模型: {self.model_name}")
        else:
            print(f"[Trainer] 开始续训模型: {self.model_name} "
                  f"（从 Epoch {self.start_epoch + 1} 继续）")
        print("=" * 70)

        # ====== 续训关键步骤：恢复 scheduler 学习率阶段（新增） ======
        # 说明：train.py 创建的新 CosineAnnealingLR 默认处于第 0 步（lr 最大）。
        # 为了让调度器处于与当前 epoch 匹配的位置，需要手动调用 start_epoch 次 step()。
        # 这样即使优化器的 lr 被加载为 checkpoint 保存的值，后续 Cosine 衰减也是
        # "以当前 lr 为起点，按照 T_max=num_epochs 的曲线继续衰减" 的合理近似。
        if self.start_epoch > 0 and self.scheduler is not None:
            print(f"[Trainer] 恢复学习率调度器：手动调用 scheduler.step() x {self.start_epoch} 次")
            for _ in range(self.start_epoch):
                self.scheduler.step()
            # 打印恢复后的当前学习率，供用户确认
            current_lrs = [group["lr"] for group in self.optimizer.param_groups]
            print(f"[Trainer] 调度器恢复后当前学习率: {current_lrs}")

        # ====== 训练循环：从 start_epoch+1 开始，到 num_epochs（含）结束 ======
        # 原代码：range(1, self.num_epochs + 1)
        # 续训后：range(self.start_epoch + 1, self.num_epochs + 1)
        # 若 start_epoch >= num_epochs，则循环不会执行（提前结束并给出提示）
        epoch_loop_start = self.start_epoch + 1
        epoch_loop_end = self.num_epochs + 1  # range 上限不包含，因此 +1

        if epoch_loop_start >= epoch_loop_end:
            print(f"[Trainer] 警告：start_epoch={self.start_epoch} >= num_epochs={self.num_epochs}，"
                  f"无需训练，直接保存当前结果。")
        else:
            for epoch in range(epoch_loop_start, epoch_loop_end):
                # -------------------- 训练 --------------------
                train_loss, train_acc = self._train_one_epoch(epoch)

                # -------------------- 验证 --------------------
                test_loss, test_acc, all_labels, all_preds = self._evaluate()

                # -------------------- 记录历史 --------------------
                self.train_loss_history.append(train_loss)
                self.train_acc_history.append(train_acc)
                self.test_loss_history.append(test_loss)
                self.test_acc_history.append(test_acc)

                # -------------------- 打印 epoch 总结 --------------------
                print(
                    f"Epoch {epoch:3d}/{self.num_epochs:3d} | "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                    f"Test  Loss: {test_loss:.4f} Acc: {test_acc:.2f}%"
                )

                # -------------------- 保存最佳模型 --------------------
                # 逻辑：当前 epoch 的测试准确率超过历史最佳（包括续训前的历史）才保存
                if test_acc > self.best_test_acc:
                    self.best_test_acc = test_acc
                    self.best_epoch = epoch
                    self.early_stop_counter = 0  # 重置早停计数器

                    # 保存模型权重与优化器状态（方便断点续训）
                    ckpt_path = os.path.join(self.save_dir_checkpoint, "best_model.pth")
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_test_acc": self.best_test_acc,
                            "dataset": self.dataset,
                        },
                        ckpt_path,
                    )
                    print(f"  -> 保存最佳模型 (Test Acc: {test_acc:.2f}%) 到 {ckpt_path}")
                else:
                    self.early_stop_counter += 1

                # -------------------- 更新学习率 --------------------
                if self.scheduler is not None:
                    self.scheduler.step()

                # -------------------- 早停检查 --------------------
                if self.patience is not None and self.early_stop_counter >= self.patience:
                    print(f"\n[Trainer] 触发早停：连续 {self.patience} 个 epoch 测试准确率未提升")
                    print(f"[Trainer] 最佳模型在 Epoch {self.best_epoch}，Test Acc = {self.best_test_acc:.2f}%")
                    break

        # -------------------- 训练结束后：绘制曲线、保存指标、混淆矩阵 --------------------
        print("\n" + "=" * 70)
        print("[Trainer] 训练完成，正在保存结果...")
        print("=" * 70)

        # 重新加载最佳模型，以确保混淆矩阵是基于最佳权重计算的
        best_ckpt = os.path.join(self.save_dir_checkpoint, "best_model.pth")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # 溯源日志放在此分支内部
                print(f"[Trainer] checkpoint 数据集来源: {checkpoint.get('dataset', 'cifar10')}")
                self.model.load_state_dict(checkpoint["model_state_dict"])
                print(f"[Trainer] 已加载最佳模型权重 (Epoch {checkpoint['epoch']})")
                # 如果 best_test_acc 比 self.best_test_acc 更高（极端情况），以保存的为准
                self.best_test_acc = max(self.best_test_acc, checkpoint.get("best_test_acc", 0.0))
                self.best_epoch = checkpoint["epoch"]

        # 使用最佳模型重新跑一遍测试集获取标签和预测
        _, _, all_labels, all_preds = self._evaluate()

        # 1) 绘制并保存训练曲线
        self._plot_training_curves()

        # 2) 保存最终指标到 metrics.json
        #    注意：total_epochs = num_epochs（总目标 epoch 数），而不是增量，符合需求
        self._save_metrics()

        # 3) 绘制并保存混淆矩阵
        self._plot_confusion_matrix(all_labels, all_preds)

        print(f"\n[Trainer] 全部结果已保存到 {self.save_dir_output}/")
        print(f"[Trainer] 最佳测试准确率: {self.best_test_acc:.2f}% (Epoch {self.best_epoch})")

    # ------------------------------------------------------------------
    # 训练曲线绘制
    # ------------------------------------------------------------------
    def _plot_training_curves(self):
        """
        绘制训练曲线并保存为 training_curves.png。
        上下两个子图：
            - 上方：Loss 曲线（训练/测试）
            - 下方：Accuracy 曲线（训练/测试）
        同时在图上标注最佳准确率对应的点。

        说明：续训模式下曲线的 x 轴为增量 epoch 序号（从 1 到本次训练的 epoch 数），
        若需要完整曲线（包括前一次训练），可在 train.py 中从旧 metrics.json 加载并
        合并 history，本实现以简洁为主，只绘制本次增量部分。
        """
        epochs_range = range(1, len(self.train_loss_history) + 1)

        # 创建 2 行 1 列的子图，大小合理
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), dpi=120)

        # ---------- 上图：Loss ----------
        ax1.plot(epochs_range, self.train_loss_history, label="Train Loss", color="#1f77b4", linewidth=2)
        ax1.plot(epochs_range, self.test_loss_history, label="Test Loss", color="#ff7f0e", linewidth=2)
        ax1.set_title(f"Loss Curves ({self.model_name})", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch (Local Index in this Run)", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # ---------- 下图：Accuracy ----------
        ax2.plot(epochs_range, self.train_acc_history, label="Train Acc", color="#1f77b4", linewidth=2)
        ax2.plot(epochs_range, self.test_acc_history, label="Test Acc", color="#ff7f0e", linewidth=2)

        # 在最佳准确率处画一个醒目的标记点（仅当 best_epoch 落在本次训练范围内时）
        # best_epoch 是全局 epoch 编号，而横坐标是增量索引，因此需要计算偏移
        if len(self.test_acc_history) > 0:
            best_idx_1based = self.best_epoch - self.start_epoch
            if 1 <= best_idx_1based <= len(self.test_acc_history):
                best_acc_value = self.test_acc_history[best_idx_1based - 1]
                ax2.scatter(
                    best_idx_1based, best_acc_value,
                    color="red", s=100, marker="*", zorder=5,
                    label=f"Best Test Acc = {best_acc_value:.2f}% (Epoch {self.best_epoch})",
                )

        ax2.set_title(f"Accuracy Curves ({self.model_name})", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch (Local Index in this Run)", fontsize=11)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # 自动调整子图间距，避免标题/标签重叠
        plt.tight_layout()

        save_path = os.path.join(self.save_dir_output, "training_curves.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Trainer] 训练曲线已保存: {save_path}")

    # ------------------------------------------------------------------
    # 保存指标 JSON
    # ------------------------------------------------------------------
    def _save_metrics(self):
        """
        将最终训练指标保存为 metrics.json 文件，字段包括：
            - best_test_acc: 最佳测试准确率（跨续训全程，%）
            - final_train_loss: 最后一个 epoch 的训练 loss
            - final_test_loss:  最后一个 epoch 的测试 loss
            - final_train_acc:  最后一个 epoch 的训练准确率
            - final_test_acc:   最后一个 epoch 的测试准确率
            - total_epochs:     总目标 epoch 数（= num_epochs，不是增量）
            - start_epoch:      本次训练起始前的已完成 epoch（用于判断是否为续训）
        """
        # 如果本次没有执行任何训练（start_epoch >= num_epochs），给默认值避免索引报错
        if len(self.train_loss_history) == 0:
            final_train_loss = 0.0
            final_test_loss = 0.0
            final_train_acc = 0.0
            final_test_acc = 0.0
        else:
            final_train_loss = round(self.train_loss_history[-1], 6)
            final_test_loss = round(self.test_loss_history[-1], 6)
            final_train_acc = round(self.train_acc_history[-1], 4)
            final_test_acc = round(self.test_acc_history[-1], 4)

        metrics = {
            "best_test_acc": round(self.best_test_acc, 4),
            "final_train_loss": final_train_loss,
            "final_test_loss": final_test_loss,
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc,
            # 需求明确：total_epochs = 从头算起的总 epoch 数，即 num_epochs（不是续训增量）
            "total_epochs": self.num_epochs,
            "start_epoch": self.start_epoch,
            "resumed": self.start_epoch > 0,
            "best_epoch": self.best_epoch,
            "dataset": self.dataset,
        }

        save_path = os.path.join(self.save_dir_output, "metrics.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"[Trainer] 训练指标已保存: {save_path}")
        print(json.dumps(metrics, indent=4, ensure_ascii=False))

    # ------------------------------------------------------------------
    # 混淆矩阵绘制
    # ------------------------------------------------------------------
    def _plot_confusion_matrix(self, true_labels: np.ndarray, pred_labels: np.ndarray):
        """
        根据真实标签和预测标签计算混淆矩阵，并绘制热力图保存为 confusion_matrix.png。

        Args:
            true_labels (np.ndarray): 形状 (N,)，元素为 0~9 的整数类别索引
            pred_labels (np.ndarray): 形状 (N,)，元素为 0~9 的整数类别索引
        """
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(self.num_classes)))

        fig, ax = plt.subplots(figsize=(10, 9), dpi=120)
        # sklearn 的 ConfusionMatrixDisplay 可以直接绘制美观的混淆矩阵热力图
        # 类别数较多（如 CIFAR-100）时不显示类别刻度标签，避免刻度文字互相重叠
        if self.num_classes > 10:
            display_labels = None
        else:
            display_labels = CIFAR10_CLASSES
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
        if self.num_classes > 10:
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title(f"Confusion Matrix - Test Set ({self.model_name})", fontsize=13, fontweight="bold")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir_output, "confusion_matrix.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[Trainer] 混淆矩阵已保存: {save_path}")
