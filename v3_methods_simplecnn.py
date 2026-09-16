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


def build_loaders(batch_size, num_workers=2):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])
    tr = datasets.CIFAR100(root="./data", train=True, download=False, transform=tf_train)
    te = datasets.CIFAR100(root="./data", train=False, download=False, transform=tf_test)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),
            DataLoader(te, batch_size=256, shuffle=False, num_workers=num_workers))


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
                alphas = [torch.zeros((), device=device, requires_grad=True) for _ in names]
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
                else:  # joint / plain
                    amap = {n: random.uniform(-0.3, 0.3) for n in names}
                extra_hooks = []
                if args.mode == "joint":
                    sigma = random.uniform(0.0, 0.3)
                    extra_hooks = register_perturbation_hooks(model, "gaussian", noise_std=sigma)
                opt.zero_grad()
                with Injector(model, amap):
                    loss = crit(model(x), y)
                    loss.backward()
                for h in extra_hooks:
                    h.remove()
                opt.step()
                alpha_now = float(np.mean([abs(v) for v in amap.values()]))

            tl += loss.item() * x.size(0)
            tt += x.size(0)
            pbar.set_postfix({"loss": f"{tl/tt:.3f}", "|a|": f"{alpha_now:.3f}"})
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
    ap.add_argument("--mode", required=True, choices=["budget", "adv", "curriculum", "joint"])
    ap.add_argument("--profile", default="uniform", choices=["uniform", "front", "rear", "sens"])
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adv_steps", type=int, default=2, help="PGD 内层步数")
    ap.add_argument("--adv_lr", type=float, default=0.1, help="PGD 内层步长")
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    nc = get_num_classes(args.dataset)
    from models.simple_cnn import SimpleCNN
    model = SimpleCNN(num_classes=nc).to(device)
    names = layer_names(model)

    name = f"{args.mode}_{args.profile}{args.tag}"
    args.ckpt_dir = os.path.join(get_ckpt_root(args.dataset), f"v3_{name}")
    args.out_dir = os.path.join(get_outputs_root(args.dataset), "v3_methods", name)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[v3-methods] mode={args.mode} profile={args.profile} epochs={args.epochs} device={device}")
    print(f"[v3-methods] layers={names}")
    print(f"[v3-methods] ckpt={args.ckpt_dir}\n[v3-methods] out={args.out_dir}")

    train_loader, test_loader = build_loaders(args.batch_size)
    t0 = time.time()
    best_acc, best_epoch, final_a03, hist = train(args, model, train_loader, test_loader, device, names)

    # ---- 载入 best 权重做最终评估 ----
    ck = torch.load(os.path.join(args.ckpt_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    a7 = scan(model, test_loader, device, [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
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
