"""
数据集配置与路径管理模块

功能：
    - validate_dataset(dataset)：校验数据集名称，非法值抛 ValueError
    - get_ckpt_root(dataset)：返回 checkpoint 根目录（cifar10 → ./checkpoints，
      cifar100 → ./checkpoints_cifar100）
    - get_outputs_root(dataset)：返回输出根目录（cifar10 → ./outputs，
      cifar100 → ./outputs_cifar100）
    - get_num_classes(dataset)：返回类别数（cifar10 → 10，cifar100 → 100）
    - get_class_names(dataset)：返回类别名称列表
"""

# CIFAR-10 固定的 10 个类别名称
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-100 官方 100 个 fine-label 类别名称（顺序勿改）
CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]

# 各数据集的统一配置：checkpoint 根目录 / 输出根目录 / 类别数 / 类别名称
_DATASET_CONFIG = {
    "cifar10": {
        "ckpt_root": "./checkpoints",
        "outputs_root": "./outputs",
        "num_classes": 10,
        "class_names": CIFAR10_CLASSES,
    },
    "cifar100": {
        "ckpt_root": "./checkpoints_cifar100",
        "outputs_root": "./outputs_cifar100",
        "num_classes": 100,
        "class_names": CIFAR100_CLASSES,
    },
}


def validate_dataset(dataset: str) -> str:
    """
    校验数据集名称是否合法。

    Args:
        dataset (str): 数据集名称，仅支持 "cifar10" / "cifar100"

    Returns:
        str: 校验通过后原样返回数据集名称

    Raises:
        ValueError: 数据集名称不在支持列表中
    """
    if dataset not in _DATASET_CONFIG:
        raise ValueError(
            f"不支持的数据集: '{dataset}'，仅支持: cifar10 / cifar100"
        )
    return dataset


def get_ckpt_root(dataset: str) -> str:
    """
    返回指定数据集的 checkpoint 根目录。

    cifar10  → ./checkpoints
    cifar100 → ./checkpoints_cifar100
    """
    validate_dataset(dataset)
    return _DATASET_CONFIG[dataset]["ckpt_root"]


def get_outputs_root(dataset: str) -> str:
    """
    返回指定数据集的输出根目录。

    cifar10  → ./outputs
    cifar100 → ./outputs_cifar100
    """
    validate_dataset(dataset)
    return _DATASET_CONFIG[dataset]["outputs_root"]


def get_num_classes(dataset: str) -> int:
    """
    返回指定数据集的类别数。

    cifar10  → 10
    cifar100 → 100（fine label）
    """
    validate_dataset(dataset)
    return _DATASET_CONFIG[dataset]["num_classes"]


def get_class_names(dataset: str) -> list:
    """
    返回指定数据集的类别名称列表（顺序与官方标签一致）。
    """
    validate_dataset(dataset)
    return _DATASET_CONFIG[dataset]["class_names"]
