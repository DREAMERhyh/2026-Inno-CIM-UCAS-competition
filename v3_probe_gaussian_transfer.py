"""
第三阶段 §6 旗标④：深层**校准模型**的跨扰动迁移检验（L3 的普适性检验）

背景：L3 称"非线性（方向通路）与高斯（幅度通路）两套损伤机制，跨扰动迁移为零"，
      但该结论在 CIFAR-100 上只被 SimpleCNN 家族验证过（ext2 的模型注册表只有 exp2=校准版 SimpleCNN）。
      ext6 已产出深层校准权重（Exp2_Calib+Layerwise_{vgg11,resnet18}），本脚本补上深层检验。

设计：对 {vgg11, resnet18} 各取 clean 与 Exp2 校准两套权重，分别扫
      α ∈ [-0.3..+0.3]（7 点，用于与 ext6 产物对拍）与 σ ∈ {0.05..0.3}（6 点）。
判据（沿用 L3 口径）：若各 σ 上 |acc_calibrated − acc_clean| ≤ 1.2 pp → 零迁移成立；
      若显著 > 1.2 pp → L3 在深层不成立，需修订。

用法：
  python v3_probe_gaussian_transfer.py --dataset cifar100
输出：
  outputs_cifar100/v3_probe_gaussian_transfer/transfer_{model}.csv
"""
import argparse
import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.perturbation import register_perturbation_hooks
from utils.nonlinearity import remove_hooks
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

PAIRS = {
    "vgg11": ("models.vgg11", "VGG11", "models.robust_vgg11", "RobustVGG11"),
    "resnet18": ("models.resnet", "ResNet18", "models.robust_resnet", "RobustResNet18"),
}


def build(model_name, num_classes):
    plain_mod, plain_cls, rob_mod, rob_cls = PAIRS[model_name]
    import importlib
    P = getattr(importlib.import_module(plain_mod), plain_cls)
    R = getattr(importlib.import_module(rob_mod), rob_cls)
    return P(num_classes=num_classes), R(num_classes=num_classes)


def load_into(model, path, device):
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    return model.to(device).eval()


def make_loader(dataset):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)])
    ds = datasets.CIFAR100(root="./data", train=False, download=False, transform=tf)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate(model, loader, device, pert_type, alpha=0.0, noise_std=0.0):
    hooks = register_perturbation_hooks(model, pert_type, alpha=alpha, noise_std=noise_std)
    correct = total = 0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    finally:
        remove_hooks(hooks)
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    nc = get_num_classes(args.dataset)
    loader = make_loader(args.dataset)
    ckpt_root = get_ckpt_root(args.dataset)
    out_dir = os.path.join(get_outputs_root(args.dataset), "v3_probe_gaussian_transfer")
    os.makedirs(out_dir, exist_ok=True)

    alphas = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    sigmas = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    for name in PAIRS:
        tag = name.replace("_", "")
        plain, robust = build(name, nc)
        p_path = os.path.join(ckpt_root, name, "best_model.pth")
        r_path = os.path.join(ckpt_root, f"Exp2_Calib+Layerwise_{tag}", "best_model.pth")
        if not os.path.exists(r_path):
            print(f"[skip] {name}: 校准权重不存在 {r_path}")
            continue
        load_into(plain, p_path, device)
        load_into(robust, r_path, device)
        print(f"\n=== {name} ===")
        print(f"  clean 权重: {p_path}")
        print(f"  校准 权重: {r_path}")
        rows = []
        print(f"{'pert':>10} {'x':>6} {'clean':>8} {'calibr':>8} {'Δ':>8}")
        for a in alphas:
            c = evaluate(plain, loader, device, "nonlinearity", alpha=a)
            r = evaluate(robust, loader, device, "nonlinearity", alpha=a)
            rows.append({"pert": "nonlinearity", "x": a, "acc_clean": round(c, 2),
                         "acc_calibrated": round(r, 2), "delta": round(r - c, 2)})
            print(f"{'alpha':>10} {a:>+6.1f} {c:>8.2f} {r:>8.2f} {r-c:>+8.2f}")
        for s in sigmas:
            c = evaluate(plain, loader, device, "gaussian", noise_std=s)
            r = evaluate(robust, loader, device, "gaussian", noise_std=s)
            rows.append({"pert": "gaussian", "x": s, "acc_clean": round(c, 2),
                         "acc_calibrated": round(r, 2), "delta": round(r - c, 2)})
            print(f"{'sigma':>10} {s:>6.2f} {c:>8.2f} {r:>8.2f} {r-c:>+8.2f}")
        gsig = [abs(x["delta"]) for x in rows if x["pert"] == "gaussian"]
        print(f"  → 高斯最大 |Δ| = {max(gsig):.2f} pp "
              f"({'零迁移成立(≤1.2)' if max(gsig) <= 1.2 else '⚠️ 迁移非零'})")
        p = os.path.join(out_dir, f"transfer_{name}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["pert", "x", "acc_clean", "acc_calibrated", "delta"])
            w.writeheader()
            w.writerows(rows)
        print(f"  [saved] {p}")


if __name__ == "__main__":
    main()
