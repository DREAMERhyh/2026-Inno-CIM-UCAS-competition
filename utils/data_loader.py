"""
数据加载与预处理模块

功能：
    - 加载 CIFAR-10 数据集
    - 训练集数据增强：随机水平翻转、随机裁剪、归一化
    - 测试集：仅归一化
    - 返回 PyTorch DataLoader
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(batch_size: int = 128, num_workers: int = 2):
    """
    获取 CIFAR-10 的训练集和测试集 DataLoader。

    数据预处理策略：
        训练集：
            1. RandomCrop(32, padding=4)：先四周各填充 4 像素，再随机裁剪回 32x32
            2. RandomHorizontalFlip(p=0.5)：随机水平翻转
            3. ToTensor()：转为 Tensor，值域 [0, 1]
            4. Normalize：使用 CIFAR-10 数据集的全局均值和标准差
        测试集：
            1. ToTensor()
            2. Normalize：使用与训练集完全相同的均值和标准差

    Args:
        batch_size (int): 每个批次的样本数量，默认 128
        num_workers (int): 数据加载线程数，默认 2

    Returns:
        tuple: (train_loader, test_loader)
            - train_loader: 训练集 DataLoader（已打乱顺序）
            - test_loader: 测试集 DataLoader（不打乱顺序）
    """
    # CIFAR-10 数据集经过统计得出的全局均值和标准差（RGB 三个通道）
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    # -------------------- 训练集数据增强 --------------------
    # 组合多个变换，按顺序执行
    train_transform = transforms.Compose([
        # 先在四周各填充 4 个像素（填充值默认为 0），图像从 32x32 变为 40x40
        transforms.RandomCrop(32, padding=4),
        # 以 50% 的概率随机水平翻转
        transforms.RandomHorizontalFlip(p=0.5),
        # 将 PIL Image 或 numpy array (H, W, C) 转换为 Tensor (C, H, W)，值域 [0, 1]
        transforms.ToTensor(),
        # 逐通道进行标准化：output = (input - mean) / std
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    # -------------------- 测试集数据预处理 --------------------
    # 测试集不使用任何数据增强操作，仅进行归一化以保持与训练集一致的分布
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean, std=cifar_std),
    ])

    # -------------------- 加载数据集 --------------------
    # 使用 torchvision 内置的 CIFAR10 数据集类
    # download=False：因为数据已就位，无需重新下载
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=test_transform,
    )

    # -------------------- 创建 DataLoader --------------------
    # shuffle=True：训练集每个 epoch 重新打乱顺序，防止模型学习到样本顺序的偏差
    # drop_last=True：当最后一个批次样本数不足 batch_size 时，丢弃该批次
    #   （可选，主要是为了避免 BatchNorm 在批次样本数过少时统计不准的问题）
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # 如果使用 GPU，将数据锁页以加速传输
        drop_last=True,
    )

    # 测试集不打乱顺序，也不需要 drop_last
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"[DataLoader] 训练集样本数: {len(train_dataset)}, 测试集样本数: {len(test_dataset)}")
    print(f"[DataLoader] 训练批次数: {len(train_loader)}, 测试批次数: {len(test_loader)}")

    return train_loader, test_loader
