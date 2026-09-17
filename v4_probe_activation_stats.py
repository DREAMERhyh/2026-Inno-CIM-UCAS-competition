"""
第四阶段 P1-4：逐层激活统计（纯推理）

验证假说：
  +α（增益饱和型）为**无界放大器**（激活幅度随深度几何级放大）；
  −α（压缩型）为**有界压缩器**（幅度被压、不放大）。
并与 L5（MaxPool 敏感性）对照：看 MaxPool 层是否改变了放大斜率。

设计：冻结 clean SimpleCNN；对每个注入条件，在**每个 Conv2d/Linear 的输入**上挂捕获 hook，
记录逐样本 L2 范数均值与全体元素 |x| 的 p99 与 max。全程无梯度。

用法：python v4_probe_activation_stats.py --dataset cifar100 --model simple_cnn
产物：outputs_cifar100/v4_activation_stats/act_stats.csv
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.nonlinearity import register_nonlinearity_hooks, remove_hooks
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}


def make_loader(dataset):
    mean, std = STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    D = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    ds = D(root="./data", train=False, download=False, transform=tf)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)


@torch.no_grad()
def collect(model, loader, device, alpha):
    """返回 {layer_name: (l2_mean, abs_p99, abs_max)}，统计对象是**层输入**"""
    acc = {}
    hooks = []

    def mk(name):
        def pre_hook(m, inputs):
            if not inputs:
                return inputs
            x = inputs[0].detach()
            flat = x.reshape(x.size(0), -1)
            l2 = flat.norm(dim=1).mean().item()
            a = x.abs()
            # 分块收集 p99/max（避免全量展开占内存）
            acc.setdefault(name, {"l2": [], "p99": [], "mx": []})
            acc[name]["l2"].append(l2)
            acc[name]["p99"].append(torch.quantile(a.flatten()[:2_000_000].float(), 0.99).item())
            acc[name]["mx"].append(a.max().item())
            return inputs
        return pre_hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_pre_hook(mk(name)))
    pert = register_nonlinearity_hooks(model, alpha) if alpha != 0.0 else []
    try:
        for x, _ in loader:
            model(x.to(device))
    finally:
        for h in hooks:
            h.remove()
        if pert:
            remove_hooks(pert)
    return {k: (float(np.mean(v["l2"])), float(np.max(v["p99"])), float(np.max(v["mx"])))
            for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--model", default="simple_cnn")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_batches", type=int, default=40, help="只用前 N 个 batch（默认 40 ≈ 10k 张）")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    nc = get_num_classes(args.dataset)

    from models.simple_cnn import SimpleCNN
    model = SimpleCNN(num_classes=nc)
    ck = torch.load(os.path.join(get_ckpt_root(args.dataset), args.model, "best_model.pth"),
                    map_location=device)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    model.to(device).eval()

    loader = make_loader(args.dataset)
    alphas = [0.0, 0.3, -0.3, 0.5, -0.5]
    res = {}
    for a in alphas:
        res[a] = collect(model, loader, device, a)
        print(f"[act] α={a:+.1f} 完成")

    layers = list(res[0.0].keys())
    rows = []
    print(f"\n{'layer':<8} " + " ".join(f"{'a=%+.1f L2' % a:>10}" for a in alphas)
          + "   " + " ".join(f"{'p99%+.1f' % a:>10}" for a in (0.0, 0.3, -0.3)))
    for lname in layers:
        vals = [res[a][lname][0] for a in alphas]
        p99 = [res[a][lname][1] for a in alphas]
        print(f"{lname:<8} " + " ".join(f"{v:>10.3f}" for v in vals)
              + "   " + " ".join(f"{p99[alphas.index(a)]:>10.4f}" for a in (0.0, 0.3, -0.3)))
        rows.append({"layer": lname,
                     **{f"l2_alpha{a:+.1f}": round(res[a][lname][0], 4) for a in alphas},
                     **{f"p99_alpha{a:+.1f}": round(res[a][lname][1], 6) for a in alphas},
                     **{f"max_alpha{a:+.1f}": round(res[a][lname][2], 6) for a in alphas}})

    # 层间放大比（相对上一层）
    print("\n层间放大比（L2_mean 相对上一层）:")
    print(f"{'layer':<8} {'clean':>10} {'+0.3':>10} {'-0.3':>10}")
    for i, lname in enumerate(layers):
        if i == 0:
            continue
        prev = layers[i - 1]
        r = [res[a][lname][0] / max(res[a][prev][0], 1e-9) for a in (0.0, 0.3, -0.3)]
        print(f"{lname:<8} " + " ".join(f"{v:>10.3f}" for v in r))

    out_dir = os.path.join(get_outputs_root(args.dataset), "v4_activation_stats")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "act_stats.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[act] 已保存: {p}")


if __name__ == "__main__":
    main()
