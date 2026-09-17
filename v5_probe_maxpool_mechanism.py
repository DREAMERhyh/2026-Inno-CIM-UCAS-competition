"""
第五阶段 P0-3(a)(b)：MaxPool 保护的机制钉死

(a) 交换律数值验证：maxpool(f_α(h)) == f_α(maxpool(h))？
    理论：归一化三次映射 ŷ = αx̂³ + (1−α)x̂ 在 x̂∈[−1,1] 上
          g'_α(u) = 3αu² + (1−α) ≥ 1−|α|(3·1−1)/... → 对 |α|≤0.3 严格为正（α≥0 时 ≥0.7；α<0 时 ≥0.4）
          ⇒ 逐元素严格单调 ⇒ 与 max 交换；
          且 maxpool 保持全局 max 不变 ⇒ 两条路径的归一化尺度 M 相同 ⇒ 恒等式应达机器精度。
    测法：对每层激活 h，直接比较 maxpool(f_α(h)) 与 f_α(maxpool(h))（用模型自带的 pool 模块）。

(b) 不变区集中假说：统计各层/各 pool 输出在归一化域 x̂=x/max|x| 上的分布，
    计算"失真易感度" S = E[|x̂|(1−x̂²)]（f_α 偏离恒等的强度因子）与不变区占比 P(|x̂|≥0.9)。
    对比 2-pool（原 SimpleCNN）与 5-pool（simple_cnn_mp）的同层统计。

用法：python v5_probe_maxpool_mechanism.py --dataset cifar100
产物：outputs_cifar100/v5_maxpool_mechanism/{commutativity.csv, invariant_frac.csv}
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.nonlinearity import nonlinearity
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}
ALPHAS = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3]


def make_loader(dataset, n=2048):
    mean, std = STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    D = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    ds = D(root="./data", train=False, download=False, transform=tf)
    return DataLoader(ds, batch_size=n, shuffle=False, num_workers=0)


def build_arch(name, nc):
    if name == "simple_cnn_mp":
        from v3_methods_simplecnn import SimpleCNNMaxPool
        return SimpleCNNMaxPool(num_classes=nc)
    from models.simple_cnn import SimpleCNN
    return SimpleCNN(num_classes=nc)


def collect_activations(model, x, names):
    """返回 {层名: 该层输出}（forward hook）"""
    out = {}
    hooks = []
    for nm in names:
        mod = dict(model.named_modules())[nm]

        def mk(k=nm):
            def h(m, i, o):
                out[k] = o.detach()
            return h
        hooks.append(mod.register_forward_hook(mk()))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return out


def collect_pool_inputs(model, x):
    """返回 {池化层名: 其输入张量}（forward_pre_hook）——网络中即 ReLU 之后，全非负"""
    out = {}
    hooks = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.MaxPool2d):
            def mk(k=nm):
                def h(mod, inp):
                    if inp:
                        out[k] = inp[0].detach()
                return h
            hooks.append(m.register_forward_pre_hook(mk()))
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    nc = get_num_classes(args.dataset)
    loader = make_loader(args.dataset, args.n)
    x, _ = next(iter(loader))
    x = x.to(dev)

    out_dir = os.path.join(get_outputs_root(args.dataset), "v5_maxpool_mechanism")
    os.makedirs(out_dir, exist_ok=True)

    pool2 = nn.MaxPool2d(2, 2).to(dev)
    rows_a, rows_b = [], []

    for arch in ("simple_cnn", "simple_cnn_mp"):
        tag = arch.replace("_", "")
        model = build_arch(arch, nc)
        ckpt = (os.path.join(get_ckpt_root(args.dataset), "simple_cnn", "best_model.pth")
                if arch == "simple_cnn"
                else os.path.join(get_ckpt_root(args.dataset), f"v3_{arch}_clean_uniform", "best_model.pth"))
        if not os.path.exists(ckpt):
            print(f"[skip] {arch}: 权重不存在 {ckpt}")
            continue
        sd = torch.load(ckpt, map_location=dev)
        model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
        model.to(dev).eval()

        names = [n_ for n_, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        acts = collect_activations(model, x, names)

        # ---------- (a) 交换律：在**池化层输入**上验证（＝网络中 ReLU 之后，全非负） ----------
        pool_in = collect_pool_inputs(model, x)
        for nm, h in pool_in.items():
            if h.shape[-1] < 2:
                continue
            for a in ALPHAS:
                A = pool2(nonlinearity(h, a))         # maxpool(f_α(h))
                B = nonlinearity(pool2(h), a)         # f_α(maxpool(h))
                diff = (A - B).abs().max().item()
                scale = A.abs().max().item() + 1e-12
                rows_a.append({"arch": arch, "layer": f"poolin:{nm}", "alpha": a,
                               "max_abs_diff": diff, "rel_diff": diff / scale,
                               "note": "post-ReLU(非负)"})
        # 对照：在有符号的 conv 原始输出上（abs-max 与有符号 max 不一致 → 预期破缺）
        for nm in names:
            h = acts[nm]
            if h.shape[-1] < 2:
                continue
            for a in (0.3,):
                A = pool2(nonlinearity(h, a))
                B = nonlinearity(pool2(h), a)
                diff = (A - B).abs().max().item()
                scale = A.abs().max().item() + 1e-12
                rows_a.append({"arch": arch, "layer": f"convout:{nm}", "alpha": a,
                               "max_abs_diff": diff, "rel_diff": diff / scale,
                               "note": "有符号(含负值)→尺度不匹配"})
        # ---------- (b) 不变区统计 ----------
        for nm in names:
            h = acts[nm]
            B_, C_, H_, W_ = h.shape
            flat = h.view(B_, -1)
            M = flat.abs().max(dim=1, keepdim=True).values.clamp(min=1e-12)
            xn = (h / M.view(B_, 1, 1, 1)).abs()
            # 失真易感度 S = E[|x̂|(1−x̂²)]（f_α 相对恒等的强度因子，|α| 单独乘）
            S = (xn * (1 - xn ** 2)).mean().item()
            rows_b.append({"arch": arch, "layer": nm,
                           "abs_xhat_mean": xn.mean().item(),
                           "susceptibility_S": S,
                           "frac_ge_0.9": (xn >= 0.9).float().mean().item(),
                           "frac_le_0.3": (xn <= 0.3).float().mean().item(),
                           "spatial_elems": H_ * W_})

    p_a = os.path.join(out_dir, "commutativity.csv")
    with open(p_a, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "layer", "alpha", "max_abs_diff", "rel_diff", "note"])
        w.writeheader()
        w.writerows(rows_a)
    print(f"\n=== (a) 交换律验证：maxpool(f_α(h)) vs f_α(maxpool(h)) ===")
    print(f"{'arch':<14} {'位置':<18} {'a':>6} {'max|Δ|':>12} {'相对':>10}  备注")
    for r in rows_a:
        print(f"{r['arch']:<14} {r['layer']:<18} {r['alpha']:>+6.1f} {r['max_abs_diff']:>12.3e} "
              f"{r['rel_diff']:>10.2e}  {r['note']}")
    print(f"[saved] {p_a}")

    p_b = os.path.join(out_dir, "invariant_frac.csv")
    with open(p_b, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "layer", "abs_xhat_mean", "susceptibility_S",
                                          "frac_ge_0.9", "frac_le_0.3", "spatial_elems"])
        w.writeheader()
        w.writerows(rows_b)
    print(f"\n=== (b) 归一化域分布与易感度 S = E[|x̂|(1−x̂²)] ===")
    print(f"{'arch':<14} {'layer':<7} {'E|x̂|':>8} {'S':>8} {'P(|x̂|≥.9)':>10} {'P(|x̂|≤.3)':>10} {'H×W':>6}")
    for r in rows_b:
        print(f"{r['arch']:<14} {r['layer']:<7} {r['abs_xhat_mean']:>8.4f} {r['susceptibility_S']:>8.4f} "
              f"{r['frac_ge_0.9']:>10.4f} {r['frac_le_0.3']:>10.4f} {r['spatial_elems']:>6}")
    print(f"[saved] {p_b}")


if __name__ == "__main__":
    main()
