"""
第五阶段 P2-6(a)：正侧不可训练区的"秩/条件数塌陷"机制（L4）

假说：α>0（压缩器，见 P1-4）会压低深层表征的有效秩与病态度，
      使"重新训练一个读出"的信息容量不足 → 正侧尾部不可训练。
测法：冻结 clean 主干，在 α ∈ {0, ±0.1, ±0.2, ±0.3, ±0.4, ±0.5} 上提取 fc 输入特征（N=10000），
      计算 ① 有效秩 rank_eff = exp(H(σ_i²/Σσ²))（谱熵指数）；② 条件数 κ = σ₁/σ_r
      （r 取解释 99% 方差的截断）；③ 在特征上做**线性分类的可分性代理**：类间/类内散布比。

用法：python v5_probe_rank_collapse.py --dataset cifar100
产物：outputs_cifar100/v5_rank_collapse/rank_stats.csv
"""
import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.nonlinearity import register_nonlinearity_hooks, remove_hooks
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}
ALPHAS = [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5]


@torch.no_grad()
def features(model, loader, device, alpha, readout, limit):
    store = {}
    h = readout.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0]))
    pert = register_nonlinearity_hooks(model, alpha) if alpha != 0.0 else []
    F, Y = [], []
    n = 0
    try:
        for x, y in loader:
            x = x.to(device)
            model(x)
            F.append(store["f"].detach().cpu().numpy())
            Y.append(y.numpy())
            n += x.size(0)
            if n >= limit:
                break
    finally:
        h.remove()
        if pert:
            remove_hooks(pert)
    return np.concatenate(F)[:limit], np.concatenate(Y)[:limit]


def rank_stats(X, Y):
    """有效秩、条件数、类间/类内散布比"""
    Xc = X - X.mean(0, keepdims=True)
    # SVD（经济型）
    s = np.linalg.svd(Xc, compute_uv=False)
    p = (s ** 2) / np.sum(s ** 2)
    p = p[p > 0]
    rank_eff = float(np.exp(-np.sum(p * np.log(p))))          # 谱熵指数（effective rank）
    # 条件数：截断到 99% 方差
    cum = np.cumsum(p)
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    kappa = float(s[0] / max(s[r99 - 1], 1e-12))
    # 类间/类内散布（Fisher 判据的全局版本）
    classes = np.unique(Y)
    mu_all = X.mean(0)
    Sb = 0.0
    Sw = 0.0
    for c in classes:
        Xc_ = X[Y == c]
        mu_c = Xc_.mean(0)
        Sb += len(Xc_) * np.sum((mu_c - mu_all) ** 2)
        Sw += np.sum((Xc_ - mu_c) ** 2)
    fisher = float(Sb / max(Sw, 1e-12))
    return rank_eff, kappa, fisher, s.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--arch", default="simple_cnn")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    nc = get_num_classes(args.dataset)
    mean, std = STATS[args.dataset]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    D = datasets.CIFAR100 if args.dataset == "cifar100" else datasets.CIFAR10
    ds = D(root="./data", train=False, download=False, transform=tf)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

    from models.simple_cnn import SimpleCNN
    model = SimpleCNN(num_classes=nc)
    ck = os.path.join(get_ckpt_root(args.dataset), args.arch, "best_model.pth")
    sd = torch.load(ck, map_location=dev)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    model.to(dev).eval()

    rows = []
    print(f"{'α':>6} {'有效秩':>10} {'条件数κ(99%)':>14} {'类间/类内散布':>14} {'维度':>6}")
    for a in ALPHAS:
        X, Y = features(model, loader, dev, a, model.fc, args.n)
        re_, ka, fi, dim = rank_stats(X, Y)
        rows.append({"alpha": a, "rank_eff": round(re_, 3), "kappa99": round(ka, 2),
                     "fisher": round(fi, 5), "dim": dim, "n": len(X)})
        print(f"{a:>+6.2f} {re_:>10.2f} {ka:>14.2f} {fi:>14.5f} {dim:>6}")

    out = os.path.join(get_outputs_root(args.dataset), "v5_rank_collapse")
    os.makedirs(out, exist_ok=True)
    fp = os.path.join(out, "rank_stats.csv")
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] {fp}")


if __name__ == "__main__":
    main()
