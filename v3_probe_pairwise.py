"""
第三阶段 §2.2 后续：跨层损伤是"成对"还是"全局"？（层组合探针）

背景：信息探针（§3.1）发现单层 α=+0.3 几乎无害（−1~−3 pp），全层同时注入却跌 −31.84 pp
→ 强超加性。本脚本回答：超加性在**两两组合**时是否已出现，还是必须全层？

设计（冻结 clean SimpleCNN，纯推理）：
  - 单层：{conv1, conv2, conv3, conv4, fc} 各 α=+0.3
  - 全对：C(5,2)=10 个两层组合，各 α=+0.3
  - 全层：5 层同时 α=+0.3
  记录精度；定义"加性预期" = clean − Σ(单层跌幅)，超加性 = 加性预期 − 实测。

用法：python v3_probe_pairwise.py --dataset cifar100
产物：outputs_cifar100/v3_pairwise/pairwise.csv
"""
import argparse
import csv
import itertools
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.nonlinearity import nonlinearity
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
ALPHA = 0.3


def make_loader():
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    ds = datasets.CIFAR100(root="./data", train=False, download=False, transform=tf)
    return DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)


def inject_hooks(model, layer_names, alpha):
    mods = dict(model.named_modules())
    hooks = []
    for name in layer_names:
        m = mods[name]

        def mk(a=alpha):
            def pre_hook(mod, inputs):
                if not inputs:
                    return inputs
                return (nonlinearity(inputs[0], a),) + tuple(inputs[1:])
            return pre_hook

        hooks.append(m.register_forward_pre_hook(mk()))
    return hooks


@torch.no_grad()
def acc_with(model, loader, device, layer_names, alpha):
    hooks = inject_hooks(model, layer_names, alpha) if layer_names else []
    c = t = 0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item()
            t += y.size(0)
    finally:
        for h in hooks:
            h.remove()
    return 100.0 * c / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--model", default="simple_cnn")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    nc = get_num_classes(args.dataset)

    from models.simple_cnn import SimpleCNN
    model = SimpleCNN(num_classes=nc)
    ck = torch.load(os.path.join(get_ckpt_root(args.dataset), args.model, "best_model.pth"),
                    map_location=device)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    model.to(device).eval()

    loader = make_loader()
    layers = ["conv1", "conv2", "conv3", "conv4", "fc"]

    clean = acc_with(model, loader, device, [], 0.0)
    print(f"[pairwise] clean = {clean:.2f}%")

    singles = {}
    for l in layers:
        a = acc_with(model, loader, device, [l], ALPHA)
        singles[l] = clean - a
        print(f"  单层 {l:<6} α=+{ALPHA} → {a:6.2f}%  (跌 {clean-a:5.2f} pp)")

    rows = [{"combo": "clean", "size": 0, "acc": round(clean, 2), "drop": 0.0,
             "additive_expect": "", "super_additivity": ""}]
    for l in layers:
        rows.append({"combo": l, "size": 1, "acc": "", "drop": round(singles[l], 2),
                     "additive_expect": "", "super_additivity": ""})

    print(f"\n{'组合':<22} {'实测':>8} {'加性预期':>10} {'超加性':>8}")
    for pair in itertools.combinations(layers, 2):
        a = acc_with(model, loader, device, list(pair), ALPHA)
        drop = clean - a
        add_exp = singles[pair[0]] + singles[pair[1]]
        super_add = add_exp - drop
        rows.append({"combo": "+".join(pair), "size": 2, "acc": round(a, 2), "drop": round(drop, 2),
                     "additive_expect": round(add_exp, 2), "super_additivity": round(super_add, 2)})
        print(f"{'+'.join(pair):<22} {a:>8.2f} {clean-add_exp:>10.2f} {super_add:>+8.2f}")

    a_all = acc_with(model, loader, device, layers, ALPHA)
    add_all = sum(singles.values())
    rows.append({"combo": "ALL5", "size": 5, "acc": round(a_all, 2), "drop": round(clean - a_all, 2),
                 "additive_expect": round(add_all, 2), "super_additivity": round(add_all - (clean - a_all), 2)})
    print(f"{'ALL5':<22} {a_all:>8.2f} {clean-add_all:>10.2f} {add_all-(clean-a_all):>+8.2f}")

    out_dir = os.path.join(get_outputs_root(args.dataset), "v3_pairwise")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "pairwise.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["combo", "size", "acc", "drop",
                                          "additive_expect", "super_additivity"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[pairwise] 已保存: {p}")


if __name__ == "__main__":
    main()
