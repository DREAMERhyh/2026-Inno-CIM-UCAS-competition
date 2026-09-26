"""
第三阶段 §3.2 / §4.1 / §4.3 / §4.4：SimpleCNN 级方法实验统一入口

四个模式（--mode）：
  budget    : 固定总预算 Σ|α_l| = 1.5（均匀时=各层 0.3），扫 4 种分配轮廓
              uniform / front / rear / sens（sens = 按 |∂L/∂α_l| 敏感性排序分配）
  adv       : α-空间对抗训练 —— 每 batch 在逐层 α 向量上做 K 步 PGD 内最大化，
              再以最坏 α 做训练步（对手空间仅 5 维，内循环廉价）
  curriculum: 课程式 α —— 采样半宽从 ±0.05 线性涨到 ±0.3
  joint     : 联合扰动 —— α 与高斯 σ 同批注入（每 batch 同时抽逐层 α 与一个 σ）

协议（与 task2 NAT 严格一致，便于与 NAT-scratch 对照）：
  SimpleCNN / CIFAR-100 / 120 epochs / SGD(lr=0.01, momentum=0.9, wd=1e-4) /
  CosineAnnealingLR / batch 128 / seed 42 / 验证阶段干净（α=0）
评估（跑完自动执行）：
  载入 best 权重 → 7 点 α 扫描 + 6 点 σ 扫描（全量测试集）→
  写 metrics.json / alpha_sensitivity.csv / gaussian_sensitivity.csv

产物隔离：checkpoints_cifar100/v3_{mode}_{tag}/
          outputs_cifar100/v3_methods/{mode}_{tag}/
"""
import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.nonlinearity import nonlinearity, register_nonlinearity_hooks, remove_hooks
from utils.perturbation import register_perturbation_hooks
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

# 数据集统计量（与 v3_readout_repair.py 保持一致）
DATASET_STATS = {
    "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}

# 预算轮廓（Σ r_l = 1.5，允许集中轮廓的顶层略超 0.3，上限 0.45）
BUDGET_PROFILES = {
    "uniform": [0.30, 0.30, 0.30, 0.30, 0.30],
    "front":   [0.45, 0.35, 0.25, 0.25, 0.20],
    "rear":    [0.20, 0.25, 0.25, 0.35, 0.45],
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_loaders(batch_size, num_workers=2, dataset="cifar100"):
    mean, std = DATASET_STATS[dataset]
    D = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    tr = D(root="./data", train=True, download=False, transform=tf_train)
    te = D(root="./data", train=False, download=False, transform=tf_test)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(te, batch_size=256, shuffle=False, num_workers=num_workers))


class SimpleCNNMaxPool(nn.Module):
    """SimpleCNN + 额外 MaxPool，使 MaxPool 次数从 2 提升到 5（对齐 VGG-11）。

    用途（P1-5）：检验"MaxPool 密度"是否是 L3 架构依赖（VGG-11 独有高斯迁移）的成因。
    其余结构与 SimpleCNN 逐层一致（conv 通道/BN/ReLU/GAP/FC），仅插入额外下采样。
    32×32 → pool → 16 → pool → 8 → pool → 4 → pool → 2 → pool → 1
    """

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2)          # 新增
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2)          # 原有
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, 2)          # 新增
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, 2)          # 原有
        self.pool5 = nn.MaxPool2d(2, 2)          # 新增（密度对齐 VGG 的 5 次）
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool5(self.pool4(self.relu4(self.bn4(self.conv4(x)))))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class Top2MeanPool2d(nn.Module):
    """top-2 均值池化（2×2 窗口取最大两个元素的均值）——P0-3(c) 的算子替换对照。

    与 MaxPool 的差别：仍做降维（2×2→1），但对 f_α 不再"可交换"（低通而非取极值）。
    """

    def __init__(self, k: int = 2, stride: int = 2):
        super().__init__()
        self.k = k
        self.stride = stride
        self.kernel = 2

    def forward(self, x):
        B, C, H, W = x.shape
        # 用 unfold 切 2×2 窗口
        cols = torch.nn.functional.unfold(x, kernel_size=self.kernel, stride=self.stride)  # (B, C*4, L)
        cols = cols.view(B, C, self.kernel * self.kernel, -1)
        topk = torch.topk(cols, k=self.k, dim=2).values
        out = topk.mean(dim=2)                                # (B, C, L)
        L = out.shape[-1]
        Ho, Wo = (H - self.kernel) // self.stride + 1, (W - self.kernel) // self.stride + 1
        return out.view(B, C, Ho, Wo)


class SimpleCNNMaxPoolVariant(SimpleCNNMaxPool):
    """把 SimpleCNNMaxPool 的所有下采样替换为 avg 或 top-2 均值（P0-3(c) 用），其余结构不变。"""

    def __init__(self, num_classes: int = 100, pool_type: str = "avg"):
        super().__init__(num_classes=num_classes)
        def mk():
            if pool_type == "avg":
                return nn.AvgPool2d(2, 2)
            if pool_type == "top2":
                return Top2MeanPool2d(2, 2)
            raise ValueError(pool_type)
        for name in ("pool1", "pool2", "pool3", "pool4", "pool5"):
            setattr(self, name, mk())
        self.pool_type = pool_type


def build_arch(name, nc, act="relu"):
    if name == "vgg11":                       # 第六阶段 P4c：跨架构粒度对照用
        from models.vgg11 import VGG11
        return VGG11(num_classes=nc)
    if act != "relu":
        # M14b：激活参数化变体（models/simple_cnn_act.py，镜像实现）。
        # 参数命名与参数量与 ReLU 版逐字一致；act="relu" 不走这条路径 ⇒ 既有行为逐字不变。
        from models.simple_cnn_act import SimpleCNNAct, SimpleCNNMaxPoolAct
        if name == "simple_cnn_mp":
            return SimpleCNNMaxPoolAct(num_classes=nc, act=act)
        if name == "simple_cnn":
            return SimpleCNNAct(num_classes=nc, act=act)
        raise ValueError(f"--act={act} 只支持 simple_cnn / simple_cnn_mp，不支持 {name}")
    if name == "simple_cnn_mp":
        return SimpleCNNMaxPool(num_classes=nc)
    if name == "simple_cnn_mp_avg":
        return SimpleCNNMaxPoolVariant(num_classes=nc, pool_type="avg")
    if name == "simple_cnn_mp_top2":
        return SimpleCNNMaxPoolVariant(num_classes=nc, pool_type="top2")
    from models.simple_cnn import SimpleCNN
    return SimpleCNN(num_classes=nc)


def layer_names(model):
    return [n for n, m in model.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear)) and n != ""]


class Injector:
    """逐层可配的非线性注入：alpha_map = {layer_name: float}；若 alpha_map=None 则用标量 alpha"""

    def __init__(self, model, alpha_map):
        self.model = model
        self.alpha_map = alpha_map

    def __enter__(self):
        self.hooks = []
        mods = dict(self.model.named_modules())
        for name, a in self.alpha_map.items():
            m = mods[name]

            def mk(alpha=a):
                def pre_hook(mod, inputs):
                    if not inputs:
                        return inputs
                    return (nonlinearity(inputs[0], alpha),) + tuple(inputs[1:])
                return pre_hook

            self.hooks.append(m.register_forward_pre_hook(mk()))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        return False


def alpha_vector_hooks(model, names, alphas):
    """可微注入：alphas = list[Tensor(requires_grad)]，返回 hooks（供对抗训练反传）"""
    hooks = []
    mods = dict(model.named_modules())
    for name, a in zip(names, alphas):
        m = mods[name]

        def mk(alpha=a):
            def pre_hook(mod, inputs):
                if not inputs:
                    return inputs
                return (nonlinearity(inputs[0], alpha),) + tuple(inputs[1:])
            return pre_hook

        hooks.append(m.register_forward_pre_hook(mk()))
    return hooks


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    c = t = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(1) == y).sum().item()
        t += y.size(0)
    return 100.0 * c / t


def scan(model, loader, device, alphas):
    out = []
    for a in alphas:
        hooks = register_nonlinearity_hooks(model, a)
        try:
            out.append((a, evaluate(model, loader, device)))
        finally:
            remove_hooks(hooks)
    return out


def scan_gaussian(model, loader, device, sigmas):
    out = []
    for s in sigmas:
        hooks = register_perturbation_hooks(model, "gaussian", noise_std=s)
        try:
            out.append((s, evaluate(model, loader, device)))
        finally:
            remove_hooks(hooks)
    return out


def compute_sens_profile(model, loader, device, budget=1.5):
    """按 |∂L/∂α_l| 估计敏感性并分配预算（HAWQ 思想的廉价版）"""
    names = layer_names(model)
    grads = []
    crit = nn.CrossEntropyLoss()
    model.eval()
    for i, (x, y) in enumerate(loader):
        if i >= 5:
            break
        x, y = x.to(device), y.to(device)
        alphas = [torch.tensor(0.15, device=device, requires_grad=True) for _ in names]
        hooks = alpha_vector_hooks(model, names, alphas)
        try:
            loss = crit(model(x), y)
            loss.backward()
        finally:
            for h in hooks:
                h.remove()
        grads.append([abs(a.grad.item()) if a.grad is not None else 0.0 for a in alphas])
        model.zero_grad(set_to_none=True)
    g = np.mean(np.array(grads), axis=0)
    g = np.maximum(g, 1e-9)
    r = g / g.sum() * budget
    r = np.clip(r, 0.1, 0.45)
    r = r / r.sum() * budget
    return {n: float(v) for n, v in zip(names, r)}


def train(args, model, train_loader, test_loader, device, names):
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)

    # 预算轮廓（sens 需先估）
    if args.mode == "budget":
        if args.profile == "sens":
            prof = compute_sens_profile(model, train_loader, device)
            print(f"[budget-sens] 敏感性分配: { {k: round(v,3) for k,v in prof.items()} }")
        else:
            base = BUDGET_PROFILES[args.profile]
            prof = {n: base[i % len(base)] for i, n in enumerate(names)}
    else:
        prof = None

    best_acc, best_epoch = 0.0, 0
    hist = []
    final_a03 = None
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = tt = 0
        pbar = tqdm(train_loader, desc=f"[{args.mode}/{args.profile}] {ep}/{args.epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            # ---- 采样本轮注入参数 ----
            if args.mode == "adv":
                # 注意：不能初始化为 0 —— utils.nonlinearity 在 |α|<1e-12 时直接 return x，
                # 会让梯度无法回传到 α（PGD 空转）。用随机重启（对抗训练标准做法）。
                alphas = [torch.tensor(random.uniform(-0.3, 0.3), device=device,
                                       requires_grad=True) for _ in names]
                for _ in range(args.adv_steps):
                    hooks = alpha_vector_hooks(model, names, alphas)
                    try:
                        loss = crit(model(x), y)
                        gs = torch.autograd.grad(loss, alphas, retain_graph=False, allow_unused=True)
                    finally:
                        for h in hooks:
                            h.remove()
                    with torch.no_grad():
                        for a, g in zip(alphas, gs):
                            if g is not None:
                                a.add_(args.adv_lr * g.sign())
                            a.clamp_(-0.3, 0.3)
                opt.zero_grad()
                hooks = alpha_vector_hooks(model, names, [a.detach() for a in alphas])
                try:
                    loss = crit(model(x), y)
                    loss.backward()
                finally:
                    for h in hooks:
                        h.remove()
                opt.step()
                alpha_now = float(np.mean([a.detach().item() for a in alphas]))
            else:
                if args.mode == "curriculum":
                    half = 0.05 + (0.30 - 0.05) * (ep - 1) / max(1, args.epochs - 1)
                    amap = {n: random.uniform(-half, half) for n in names}
                elif args.mode == "budget":
                    amap = {n: random.uniform(-prof[n], prof[n]) for n in names}
                elif args.mode == "range":
                    r = args.alpha_max
                    amap = {n: random.uniform(-r, r) for n in names}
                elif args.mode == "clean":
                    amap = None  # 完全无注入（干净训练基线；P1-5 的对照组）
                elif args.mode == "gauss":
                    amap = None  # 纯高斯训练（无 α 注入），用于 2×2 设计的 σ-only 格
                else:  # joint / plain
                    # 消耗数开关（第六阶段 P4f）：alpha_consume=1 → 每 batch 只抽 1 个随机数
                    # （语义上等于"全层共享一个 α"）；=5 → 现状（逐层各抽一个）。
                    # 默认 5 时本分支与改造前逐字等价。
                    ndraw = 1 if args.alpha_consume == 1 else len(names)
                    amap = {n: random.uniform(-0.3, 0.3) for n in names[:ndraw]}
                    if ndraw < len(names):
                        a0 = next(iter(amap.values()))
                        amap = {n: a0 for n in names}
                # 粒度开关（第六阶段 P4a）：shared = 全层共用一个 α。
                # 实现上取"第一个抽到的值"广播给所有层 —— 与 per-layer 消耗同样多的随机数，
                # 所以默认路径（per-layer）的 RNG 流逐位不变，本分支根本不会执行。
                if amap is not None and args.alpha_granularity == "shared":
                    amap = {n: next(iter(amap.values())) for n in amap}
                extra_hooks = []
                if args.mode in ("joint", "gauss"):
                    sigma = random.uniform(0.0, 0.3)
                    extra_hooks = register_perturbation_hooks(model, "gaussian", noise_std=sigma)
                opt.zero_grad()
                if amap is None:
                    loss = crit(model(x), y)
                    loss.backward()
                else:
                    with Injector(model, amap):
                        loss = crit(model(x), y)
                        loss.backward()
                for h in extra_hooks:
                    h.remove()
                opt.step()
                if amap is None:
                    alpha_now = -1.0  # 纯高斯训练无 α
                else:
                    alpha_now = float(np.mean([abs(v) for v in amap.values()]))

            tl += loss.item() * x.size(0)
            tt += x.size(0)
            pbar.set_postfix({"loss": f"{tl/tt:.3f}", "|a|": (f"{alpha_now:.3f}" if alpha_now >= 0 else "n/a")})
        sched.step()
        acc = evaluate(model, test_loader, device)
        hooks = register_nonlinearity_hooks(model, 0.3)
        try:
            a03 = evaluate(model, test_loader, device)
        finally:
            remove_hooks(hooks)
        final_a03 = a03
        hist.append({"epoch": ep, "train_loss": tl / tt, "clean_acc": acc, "alpha03_acc": a03})
        print(f"[{args.mode}/{args.profile}] ep{ep:3d} loss={tl/tt:.4f} clean={acc:.2f}% a03={a03:.2f}%")
        if acc > best_acc:
            best_acc, best_epoch = acc, ep
            torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                        "best_test_acc": acc, "mode": args.mode, "profile": args.profile,
                        "dataset": args.dataset},
                       os.path.join(args.ckpt_dir, "best_model.pth"))
    return best_acc, best_epoch, final_a03, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["budget", "adv", "curriculum", "joint", "range", "gauss",
                             "plain", "clean"])
    ap.add_argument("--arch", default="simple_cnn",
                    choices=["simple_cnn", "vgg11", "simple_cnn_mp", "simple_cnn_mp_avg", "simple_cnn_mp_top2"],
                    help="主干架构；simple_cnn_mp = MaxPool 密度 2→5（P1-5 用）")
    ap.add_argument("--profile", default="uniform", choices=["uniform", "front", "rear", "sens"])
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha_max", type=float, default=0.3, help="range 模式的采样半宽（§4.7 外推边界测绘用）")
    ap.add_argument("--eval_wide", action="store_true", help="最终评估用 15 点宽扫描（默认 7 点）")
    ap.add_argument("--adv_steps", type=int, default=2, help="PGD 内层步数")
    ap.add_argument("--adv_lr", type=float, default=0.1, help="PGD 内层步长")
    ap.add_argument("--alpha-granularity", default="per-layer",
                    choices=["per-layer", "shared"],
                    help="α 采样粒度（第六阶段 P4a 新增）。per-layer=逐层独立（默认，原行为）；"
                         "shared=每 batch 采一个 α 共享全层。默认值下新增代码不执行。")
    ap.add_argument("--alpha-consume", type=int, default=5, choices=[1, 5],
                    help="每 batch 消耗的随机数个数（第六阶段 P4f）。5=现状（逐层各抽一个）；"
                         "1=只抽一个并广播全层。默认 5，改造前行为逐字不变。")
    ap.add_argument("--act", default="relu", choices=["relu", "gelu", "leaky"],
                    help="M14b：激活函数。默认 relu = 既有行为逐字不变；"
                         "gelu/leaky 走 models/simple_cnn_act.py 的镜像实现")
    ap.add_argument("--tag", default="")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader 工作进程数（内存受限时保持 0）")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    nc = get_num_classes(args.dataset)
    model = build_arch(args.arch, nc, args.act).to(device)
    names = layer_names(model)

    name = f"{args.mode}_{args.profile}{args.tag}"
    if args.arch != "simple_cnn":
        name = f"{args.arch}_{name}"
    if args.act != "relu":
        name = f"{name}_{args.act}"          # M14b：防覆盖既有 ReLU 产物
    if args.mode == "range":
        name = f"range_{args.alpha_max:.2f}{args.tag}"
    args.ckpt_dir = os.path.join(get_ckpt_root(args.dataset), f"v3_{name}")
    args.out_dir = os.path.join(get_outputs_root(args.dataset), "v3_methods", name)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[v3-methods] mode={args.mode} profile={args.profile} epochs={args.epochs} device={device}")
    print(f"[v3-methods] layers={names}")
    print(f"[v3-methods] ckpt={args.ckpt_dir}\n[v3-methods] out={args.out_dir}")

    train_loader, test_loader = build_loaders(args.batch_size, args.num_workers, dataset=args.dataset)
    t0 = time.time()
    best_acc, best_epoch, final_a03, hist = train(args, model, train_loader, test_loader, device, names)

    # ---- 载入 best 权重做最终评估 ----
    ck = torch.load(os.path.join(args.ckpt_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    eval_alphas = ([-0.6, -0.5, -0.4, -0.35, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6]
                   if args.eval_wide else [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    a7 = scan(model, test_loader, device, eval_alphas)
    s6 = scan_gaussian(model, test_loader, device, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3])

    with open(os.path.join(args.out_dir, "alpha_sensitivity.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "accuracy", "loss"])
        for a, acc in a7:
            w.writerow([f"{a}", f"{acc:.4f}", ""])
    with open(os.path.join(args.out_dir, "gaussian_sensitivity.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["noise_std", "accuracy", "loss"])
        for s, acc in s6:
            w.writerow([f"{s}", f"{acc:.4f}", ""])
    metrics = {
        "dataset": args.dataset, "mode": args.mode, "profile": args.profile,
        "best_test_acc": round(best_acc, 4), "best_epoch": best_epoch,
        "total_epochs": args.epochs, "final_alpha03_acc": round(final_a03, 4),
        "seed": args.seed, "elapsed_sec": round(time.time() - t0, 1),
        "alpha_scan": {str(a): round(acc, 4) for a, acc in a7},
        "gaussian_scan": {str(s): round(acc, 4) for s, acc in s6},
        "history_tail": hist[-5:],
        "config": {"adv_steps": args.adv_steps, "adv_lr": args.adv_lr, "lr": args.lr},
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"\n[v3-methods] 完成 {name}: best_clean={best_acc:.2f}% (ep{best_epoch}) "
          f"a03={final_a03:.2f}% | 7α={[round(v,2) for _,v in a7]}")
    print(f"[v3-methods] 用时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
