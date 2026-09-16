"""
第三阶段 §4（自提方向 4.7b）：读出修复（Readout Repair）概念验证

动机（来自 §3.1 信息死亡探针）：α=+0.3 时同条件线性探针 47.89% vs 冻结头 27.76%，
即"信息仍在表征里，只是分类头解码不了"。若成立，则可推出一个**极廉价的新方法**：
不改主干、只重训/替换最后的分类读出（部署期适配）。

设计（同一冻结主干，比较 4 种读出）：
  (a) original  : 原始 fc（基线）
  (b) clean-fit : 用**干净**特征重训 fc（对照：说明"重训"本身不带来鲁棒性）
  (c) dist-fit  : 用**失真**特征（α=+0.3）重训 fc（本方法最小实现）
  (d) dist-MLP  : 用失真特征训 2 层小 MLP（容量对照）
评估：分别在 α ∈ {0, ±0.1, ±0.2, ±0.3} 上测全量测试集精度。

判据：
  若 (c) 在 α=+0.3 上显著高于 (a)（阈值：+5 pp）→ 读出修复成立（新方法候选）；
  若 (b)≈(a) 而 (c)≫(a) → 提升确由"按失真分布适配读出"带来，而非单纯重训。

产物：outputs_cifar100/v3_readout_repair/readout_repair.csv
用法：python v3_readout_repair.py --model simple_cnn --alpha 0.3 --dataset cifar100
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

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)


def build_loaders(batch_size=512):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    tr = datasets.CIFAR100(root="./data", train=True, download=False, transform=tf)
    te = datasets.CIFAR100(root="./data", train=False, download=False, transform=tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=False, num_workers=0),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0))


@torch.no_grad()
def extract_features(model, loader, alpha, device, limit=None):
    """返回 (feats, labels)；feats = fc 的输入（GAP 后）"""
    store = {}
    h = model.fc.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0]))
    hooks = register_nonlinearity_hooks(model, alpha) if alpha != 0.0 else []
    F, Y = [], []
    n = 0
    try:
        for x, y in loader:
            x = x.to(device)
            model(x)
            F.append(store["f"].detach().cpu().numpy())
            Y.append(y.numpy())
            n += x.size(0)
            if limit and n >= limit:
                break
    finally:
        h.remove()
        if hooks:
            remove_hooks(hooks)
    return np.concatenate(F)[:limit], np.concatenate(Y)[:limit]


@torch.no_grad()
def eval_with_readout(model, loader, device, alpha, readout):
    hooks = register_nonlinearity_hooks(model, alpha) if alpha != 0.0 else []
    correct = total = 0
    try:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            store = {}
            h = model.fc.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0]))
            model(x)
            h.remove()
            logits = readout(store["f"])
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    finally:
        if hooks:
            remove_hooks(hooks)
    return 100.0 * correct / total


def train_linear(X, Y, nc, device, epochs=200, lr=0.05, wd=1e-4):
    lin = nn.Linear(X.shape[1], nc).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(X).float().to(device)
    Yt = torch.from_numpy(Y).long().to(device)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(lin(Xt), Yt)
        loss.backward()
        opt.step()
    lin.eval()
    return lin


def train_readout_nat(model, loader, nc, device, epochs=6, lr=0.01, wd=1e-4, alpha_max=0.3, seed=42):
    """Readout-NAT：冻结主干，在每个 batch 上以**新的随机 α** 提取特征并更新读出。

    与 full-NAT 的区别：只训练最后的线性读出（主干冻结），成本 ≈ full-NAT 的 1/120。
    与 §3.1 固定条件读出的区别：读出覆盖整个 α 分布（不是单点特化）。
    """
    import random as _r
    rng = _r.Random(seed)
    lin = nn.Linear(model.fc.in_features, nc).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    model.eval()
    for ep in range(epochs):
        tot = cor = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            alpha = rng.uniform(-alpha_max, alpha_max)
            with torch.no_grad():
                store = {}
                h = model.fc.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0]))
                hooks = register_nonlinearity_hooks(model, alpha) if alpha != 0.0 else []
                try:
                    model(x)
                finally:
                    h.remove()
                    if hooks:
                        remove_hooks(hooks)
                feats = store["f"].detach()
            logits = lin(feats)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += y.size(0)
            cor += (logits.argmax(1) == y).sum().item()
        print(f"  [readout-NAT] ep{ep+1}/{epochs} (α~U[±{alpha_max}]) 训练精度 {100*cor/tot:.2f}%")
    lin.eval()
    return lin


def train_mlp(X, Y, nc, device, epochs=200, lr=0.01, wd=1e-4, hidden=512):
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, nc)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(X).float().to(device)
    Yt = torch.from_numpy(Y).long().to(device)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(net(Xt), Yt)
        loss.backward()
        opt.step()
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simple_cnn")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--alpha", type=float, default=0.3, help="读出适配所用的失真强度")
    ap.add_argument("--epochs_nat", type=int, default=6, help="readout-NAT 的轮数")
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
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[readout-repair] 冻结主干 {args.model}（clean 权重），适配失真 α={args.alpha:+.2f}")

    train_loader, test_loader = build_loaders()
    print("[readout-repair] 提取特征（干净 + 失真）...")
    Xc_tr, Y_tr = extract_features(model, train_loader, 0.0, device)
    Xd_tr, _ = extract_features(model, train_loader, args.alpha, device)
    print(f"  训练特征: clean {Xc_tr.shape} / dist {Xd_tr.shape}")

    print("[readout-repair] 训练三种固定条件读出 ...")
    lin_clean = train_linear(Xc_tr, Y_tr, nc, device)
    lin_dist = train_linear(Xd_tr, Y_tr, nc, device)
    mlp_dist = train_mlp(Xd_tr, Y_tr, nc, device)

    print("[readout-repair] 训练 readout-NAT（α~U[±0.3] 分布上的读出）...")
    lin_nat = train_readout_nat(model, train_loader, nc, device, epochs=args.epochs_nat)

    readouts = {
        "a_original_fc": lambda f: model.fc(f),
        "b_clean_fit_linear": lambda f: lin_clean(f),
        "c_dist_fit_linear": lambda f: lin_dist(f),
        "d_dist_fit_mlp": lambda f: mlp_dist(f),
        "e_readout_NAT": lambda f: lin_nat(f),
    }
    alphas = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    rows = []
    print(f"\n{'readout':<20} " + " ".join(f"{a:>+7.2f}" for a in alphas))
    for name, fn in readouts.items():
        accs = []
        for a in alphas:
            accs.append(eval_with_readout(model, test_loader, device, a, fn))
        rows.append({"readout": name, **{str(a): round(v, 2) for a, v in zip(alphas, accs)}})
        print(f"{name:<20} " + " ".join(f"{v:>7.2f}" for v in accs))

    out_dir = os.path.join(get_outputs_root(args.dataset), "v3_readout_repair")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"readout_repair_alpha{args.alpha:+.2f}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["readout"] + [str(a) for a in alphas])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[readout-repair] 已保存: {p}")
    base = [r for r in rows if r["readout"] == "a_original_fc"][0]
    cand = [r for r in rows if r["readout"] == "c_dist_fit_linear"][0]
    nat = [r for r in rows if r["readout"] == "e_readout_NAT"][0]
    k = str(args.alpha)
    print(f"[readout-repair] 固定条件读出判据：α={k} 处 Δ(c−a) = {cand[k] - base[k]:+.2f} pp "
          f"（但 clean 处 {cand['0.0'] - base['0.0']:+.2f} pp → 再分配性质）")
    print(f"[readout-repair] readout-NAT：clean {nat['0.0']:.2f} (Δ{nat['0.0']-base['0.0']:+.2f}) | "
          f"α+0.3 {nat[k]:.2f} (Δ{nat[k]-base[k]:+.2f}) | α-0.3 {nat['-0.3']:.2f} (Δ{nat['-0.3']-base['-0.3']:+.2f})")


if __name__ == "__main__":
    main()
