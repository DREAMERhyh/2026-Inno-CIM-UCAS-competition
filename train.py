"""
主训练脚本 train.py

功能：
    - 通过命令行参数灵活指定模型、超参数、训练设备等
    - 动态加载对应模型类（当前支持 simple_cnn，可扩展）
    - 设置 SGD + 动量 + 权重衰减 优化器，CosineAnnealingLR 调度器
    - 调用 Trainer 类完成训练与结果保存
    - 【新增】支持模型续训 (Resume Training)：
        * 新增 --resume 参数，指定 checkpoint 路径
        * 自动加载 model_state_dict、optimizer_state_dict、epoch、best_test_acc
        * 从 start_epoch+1 继续训练到 --epochs 指定的总 epoch 数
        * 学习率调度器会被手动 step() 相应次数，保持学习率阶段连续

使用示例：
    # 从头训练 50 个 epoch（默认参数）
    python train.py --model simple_cnn --epochs 50

    # 自定义学习率与 batch size
    python train.py --model simple_cnn --epochs 100 --lr 0.05 --batch_size 256

    # ====== 续训示例（新增） ======
    # 假设之前训练了 50 个 epoch 后中断，期望总训练到 100 个 epoch：
    python train.py --model simple_cnn --epochs 100 \
        --resume ./checkpoints/simple_cnn/best_model.pth
    # 效果：加载 epoch 50 的权重与优化器状态，从 epoch 51 继续训练至 epoch 100
"""

import argparse
import random
import os
import numpy as np
import torch
import torch.nn as nn

from utils.data_loader import get_dataloaders
from utils.trainer import Trainer


# ------------------------------------------------------------------
# 随机种子设置，保证结果可复现
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
    torch.cuda.manual_seed_all(seed)  # 多 GPU 情况下的种子
    # 关闭 CUDA 的某些非确定性算法（可能会稍微降低训练速度，提升可复现性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] 随机种子已设置为 {seed}，确保实验可复现")


# ------------------------------------------------------------------
# 模型加载工厂
# ------------------------------------------------------------------
def get_model(model_name: str, num_classes: int = 10):
    """
    根据模型名称字符串动态创建对应的模型实例。

    扩展新模型时：
        1. 在 models/ 下新建模型文件（例如 resnet.py）
        2. 在 models/__init__.py 中添加 import
        3. 在此函数中增加一个 elif 分支

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

    # ---------- 可在此处扩展更多模型 ----------
    # elif model_name == "vgg16":
    #     from models.vgg import VGG16
    #     return VGG16(num_classes=num_classes)
    # --------------------------------------------------------

    else:
        raise ValueError(
            f"不支持的模型名称: '{model_name}'\n"
            f"当前已支持: simple_cnn, resnet18, vgg11\n"
            f"如需添加新模型，请在 models/ 下新建文件并在 get_model() 中注册。"
        )


# ------------------------------------------------------------------
# 命令行参数解析
# ------------------------------------------------------------------
def parse_args():
    """
    解析命令行参数。所有参数都有合理的默认值，用户可按需覆盖。

    新增参数：
        --resume (str, 默认 None): checkpoint 文件路径，设置后启用续训模式

    Returns:
        argparse.Namespace: 解析后的参数字典对象
    """
    parser = argparse.ArgumentParser(description="CIFAR-10 图像分类训练脚本（支持续训）")

    # 模型相关
    parser.add_argument(
        "--model",
        type=str,
        default="simple_cnn",
        help="模型名称，默认: simple_cnn",
    )

    # ====== 【新增】续训参数 ======
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="（可选）checkpoint 文件路径，用于续训。"
             "例如 ./checkpoints/simple_cnn/best_model.pth。"
             "设置后将从该文件中恢复模型权重、优化器状态和训练进度。",
    )

    # 训练超参数
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="训练 batch size，默认: 128",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="【总目标】训练 epoch 数（续训时应设置为最终目标总 epoch）。默认: 50",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="初始学习率，默认: 0.01。续训时若 checkpoint 中有优化器状态，"
             "则优化器的 lr 会被 checkpoint 覆盖，从而继续衰减；"
             "若想在续训时强制重置 lr，请自行删除 checkpoint 中的 lr 或修改代码。",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="权重衰减（L2 正则化强度），默认: 1e-4",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="SGD 动量参数，默认: 0.9",
    )

    # 设备与系统
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="训练设备，默认自动选择 CUDA（若可用）否则使用 CPU",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader 加载线程数，默认: 2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认: 42",
    )

    # 早停（可选）
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="早停 patience（连续多少 epoch 不提升则停止），默认不启用",
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------
def main():
    """
    主流程：
        1. 解析参数，设置随机种子
        2. 加载数据
        3. 创建模型、损失函数、优化器、调度器
        4. 【可选续训】若指定 --resume：加载 checkpoint，恢复权重/优化器/进度
        5. 调用 Trainer（传入 start_epoch、best_test_acc）执行训练
        6. 打印最终最佳准确率
    """
    args = parse_args()

    # 1) 设置随机种子
    set_seed(args.seed)

    # 2) 自动选择训练设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Device] 使用训练设备: {device}")
    if device == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # 3) 加载数据
    print("\n" + "-" * 50)
    print("[Data] 正在加载 CIFAR-10 数据集...")
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # 4) 创建模型（无论是否续训，都先创建一个结构相同的模型实例）
    print("\n" + "-" * 50)
    print(f"[Model] 正在创建模型: {args.model}")
    model = get_model(args.model, num_classes=10)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 总参数量: {total_params:,}，可训练参数量: {trainable_params:,}")

    # 5) 定义损失函数、优化器、学习率调度器
    criterion = nn.CrossEntropyLoss()

    # 优化器先基于命令行 --lr 初始化（后续续训场景下会被 checkpoint 的 state 覆盖）
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # 余弦退火学习率调度：T_max = 总目标 epoch 数
    # 注意：续训场景下，scheduler 的内部 step 计数会在 Trainer.train() 内被手动推进到
    # start_epoch 次，从而保持余弦曲线的连续衰减。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0.0,
    )

    # ====== 【新增续训逻辑】6) 检查 --resume 参数，若设置则恢复 checkpoint ======
    # 续训相关变量：若未续训则保持从头训练的默认值
    start_epoch = 0
    best_test_acc = 0.0

    if args.resume is not None:
        resume_path = args.resume
        print("\n" + "-" * 50)
        print(f"[Resume] 检测到 --resume 参数，准备续训...")

        # 健壮性：检查文件是否存在
        if not os.path.exists(resume_path):
            raise FileNotFoundError(
                f"[Resume] 错误：指定的 checkpoint 文件不存在: {resume_path}\n"
                f"         请检查路径是否正确，或使用绝对路径。"
            )

        # 加载 checkpoint（map_location 保证即使 CPU 也能加载 GPU 训练的权重）
        print(f"[Resume] 正在加载 checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        # 解析 checkpoint 字段，兼容两种格式：
        #   a) Trainer 保存的字典（推荐）: {"epoch", "model_state_dict",
        #      "optimizer_state_dict", "best_test_acc"}
        #   b) 纯 state_dict：则仅恢复模型，其他信息按默认处理
        ckpt_is_dict = isinstance(checkpoint, dict)

        # ---- 恢复模型权重 ----
        if ckpt_is_dict and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[Resume] 模型权重加载成功 (来自 'model_state_dict' 字段)")
        else:
            # 回退：假设 checkpoint 直接就是 state_dict
            model.load_state_dict(checkpoint)
            print(f"[Resume] 模型权重加载成功 (直接 state_dict)")

        # ---- 恢复优化器状态（momentum buffer、权重衰减累计等） ----
        if ckpt_is_dict and "optimizer_state_dict" in checkpoint:
            # 先把模型移到正确设备，再加载 optimizer state（避免 device 不匹配）
            model.to(device)
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                # 关键：将优化器中每个参数组的设备一并移到当前 device
                # （否则 Adam/SGD 的动量缓冲可能留在 CPU 上）
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                print(f"[Resume] 优化器状态加载成功")
                # 打印加载后的学习率
                loaded_lrs = [g["lr"] for g in optimizer.param_groups]
                print(f"[Resume] 优化器当前学习率 (来自 checkpoint): {loaded_lrs}")
            except Exception as e:
                print(f"[Resume] 警告：优化器状态加载失败（{e}）。"
                      f"将使用命令行 --lr={args.lr} 作为新的起点。")
        else:
            print(f"[Resume] 警告：checkpoint 中未找到 'optimizer_state_dict'，"
                  f"优化器保持命令行初始化 (lr={args.lr})")

        # ---- 恢复 epoch 进度（已完成的 epoch 数） ----
        if ckpt_is_dict and "epoch" in checkpoint:
            start_epoch = int(checkpoint["epoch"])
            print(f"[Resume] 已完成训练的 epoch 数: start_epoch = {start_epoch}")
        else:
            start_epoch = 0
            print(f"[Resume] 警告：checkpoint 中未找到 'epoch' 字段，默认 start_epoch=0 "
                  f"（将从头训练）")

        # ---- 恢复历史最佳测试准确率 ----
        if ckpt_is_dict and "best_test_acc" in checkpoint:
            best_test_acc = float(checkpoint["best_test_acc"])
            print(f"[Resume] 历史最佳测试准确率: best_test_acc = {best_test_acc:.2f}%")
        else:
            best_test_acc = 0.0
            print(f"[Resume] 警告：checkpoint 中未找到 'best_test_acc' 字段，默认 best_test_acc=0.0% "
                  f"（续训期间可能重复保存一些已知不是最佳的权重）")

        # ---- 越界检查：如果 start_epoch >= epochs，给用户提示 ----
        if start_epoch >= args.epochs:
            print(f"\n[Resume] 警告：start_epoch ({start_epoch}) >= --epochs ({args.epochs})。"
                  f"若希望继续训练，请将 --epochs 调大，例如 --epochs {start_epoch + 50}")

        # ---- 打印清晰的续训总结 ----
        remaining = max(0, args.epochs - start_epoch)
        first_new_epoch = start_epoch + 1 if start_epoch < args.epochs else "已完成"
        print("\n" + "=" * 58)
        print(f"[Resume Summary] 续训信息总结：")
        print(f"  - Checkpoint 路径        : {resume_path}")
        print(f"  - 已完成训练 epoch 数     : {start_epoch}")
        print(f"  - 历史最佳测试准确率      : {best_test_acc:.2f}%")
        print(f"  - 本次总目标 epoch 数     : {args.epochs}")
        print(f"  - 即将开始的首个 epoch    : {first_new_epoch}")
        print(f"  - 本次新增训练 epoch 数   : {remaining}")
        print("=" * 58)
    else:
        print("\n" + "-" * 50)
        print("[Mode] 从头训练模式（未指定 --resume）")

    # 7) 保存路径：每个模型独立的子目录（与从头训练保持一致，不单独新建目录）
    save_dir_checkpoint = f"./checkpoints/{args.model}"
    save_dir_output = f"./outputs/{args.model}"
    print("\n" + "-" * 50)
    print(f"[Save] 模型权重保存目录: {save_dir_checkpoint}")
    print(f"[Save] 输出结果保存目录: {save_dir_output}")
    if args.resume is not None:
        print(f"[Save] 续训模式：将覆盖更新上述目录下的文件（best_model.pth、metrics.json 等）")

    # 8) 初始化 Trainer（新增续训参数 start_epoch、best_test_acc）并启动训练
    print("\n" + "-" * 50)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        model_name=args.model,
        save_dir_checkpoint=save_dir_checkpoint,
        save_dir_output=save_dir_output,
        patience=args.patience,
        # ====== 续训新增参数 ======
        start_epoch=start_epoch,
        best_test_acc=best_test_acc,
    )

    trainer.train()

    # 9) 训练完成，打印最终总结
    print("\n" + "=" * 70)
    if args.resume is not None:
        print(f"[Train Complete] 模型 {args.model} 续训结束！"
              f"（从 epoch {start_epoch + 1} 至 epoch {trainer.best_epoch}）")
    else:
        print(f"[Train Complete] 模型 {args.model} 训练结束！")
    print(f"[Train Complete] 最佳测试准确率: {trainer.best_test_acc:.2f}% (Epoch {trainer.best_epoch})")
    print("=" * 70)


if __name__ == "__main__":
    main()
