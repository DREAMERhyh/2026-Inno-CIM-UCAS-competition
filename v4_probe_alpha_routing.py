"""
第四阶段 P2-6（探索）：在线 α 估计 + 双头路由

问题：readout-NAT 的鲁棒头在 α=0 处比原始头差（SimpleCNN：56.80 vs 59.99），
      而原始头在尾部差得多。**部署时 α 未知**，能否从特征在线估计 α 并据此路由？

设计（冻结 clean 主干，零额外训练）：
  1. 在 α ∈ {0,±0.05,…,±0.3} 上提取训练/测试特征（各 10k）；
  2. 用岭回归拟合 α̂ = w·f + b（在**训练特征**上，α 已知）；
  3. 测试时按 |α̂| 路由到「原始头」或「readout-NAT 鲁棒头」；
  4. 对比四种策略：always-clean / always-robust / **oracle（每 α 取较优）** / **learned router**。

判据：learned router 的 7 点均值是否 > always-robust，且逼近 oracle（差距 ≤0.5pp）。

用法：python v4_probe_alpha_routing.py --dataset cifar100 --arch simple_cnn
产物：outputs_cifar100/v4_alpha_routing/routing_{arch}.csv
"""
import argparse
import csv
import os

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.nonlinearity import register_nonlinearity_hooks, remove_hooks
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}
GRID = [0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15, 0.2, -0.2, 0.25, -0.25, 0.3, -0.3]
EVAL_ALPHAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
TAU_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
TAU_DEFAULT = 0.15


def make_loader(dataset, train, n_limit):
    mean, std = STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    D = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    ds = D(root="./data", train=train, download=False, transform=tf)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=0), ds


@torch.no_grad()
def extract(model, loader, device, alpha, limit, readout):
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
            if limit and n >= limit:
                break
    finally:
        h.remove()
        if pert:
            remove_hooks(pert)
    return np.concatenate(F)[:limit], np.concatenate(Y)[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--arch", default="simple_cnn")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=10000)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    nc = get_num_classes(args.dataset)

    from models.simple_cnn import SimpleCNN
    model = SimpleCNN(num_classes=nc)
    ck = torch.load(os.path.join(get_ckpt_root(args.dataset), args.arch, "best_model.pth"),
                    map_location=device)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    model.to(device).eval()
    readout = model.fc

    # ---- 读出头：原始头权重（always-clean）与 readout-NAT 鲁棒头权重（若存在） ----
    # 鲁棒头由既有实验产出；这里直接从磁盘加载其指标并在同一特征上复现「鲁棒头」的可用性：
    # 说明：为避免权重序列化，本脚本用**在 α~U[±0.3] 上现训的线性头**作为鲁棒头（与 §9 同协议，8ep）。
    tr_loader, _ = make_loader(args.dataset, True, args.n_train)
    te_loader, _ = make_loader(args.dataset, False, args.n_test)

    # 1) 训练集：多 α 特征 + 标签（用于拟合 α̂ 与训练鲁棒头）
    X_list, Y_list, A_list = [], [], []
    for a in GRID:
        Xa, Ya = extract(model, tr_loader, device, a, args.n_train, readout)
        X_list.append(Xa)
        Y_list.append(Ya)
        A_list.append(np.full(Xa.shape[0], a))
        print(f"[routing] 训练特征 α={a:+.2f} 完成 {Xa.shape}")
    Xtr = np.concatenate(X_list)
    Ytr = np.concatenate(Y_list)
    Atr = np.concatenate(A_list)

    # 2) α 估计器：岭回归（标准化后）
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = (Xtr - mu) / sd
    ridge = Ridge(alpha=1.0).fit(Ztr, Atr)
    ahat_tr = ridge.predict(Ztr)
    print(f"[routing] α 估计器 训练集 R² = {ridge.score(Ztr, Atr):.4f}，"
          f"MAE = {np.abs(ahat_tr - Atr).mean():.4f}")

    # 3) 鲁棒头：在 α~U[±0.3] 特征上训练线性头（8 epoch，同 §9 协议）
    import random as _r
    rng = _r.Random(42)
    head_clean = torch.nn.Linear(Xtr.shape[1], nc).to(device)
    with torch.no_grad():
        head_clean.weight.copy_(readout.weight)
        head_clean.bias.copy_(readout.bias)
    head_rob = torch.nn.Linear(Xtr.shape[1], nc).to(device)
    opt = torch.optim.Adam(head_rob.parameters(), lr=0.01, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).float().to(device)
    Yt = torch.from_numpy(Ytr).long().to(device)
    idx = np.arange(len(Xtr))
    for ep in range(8):
        rng.shuffle(idx)
        for i in range(0, len(idx), 512):
            b = idx[i:i + 512]
            opt.zero_grad()
            loss = crit(head_rob(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
    head_rob.eval()
    print("[routing] 鲁棒头训练完成（8 epoch，α 混合分布）")

    # 4) 测试：逐 α 评估四种策略
    rows = []
    for a in EVAL_ALPHAS:
        Xte, Yte = extract(model, te_loader, device, a, args.n_test, readout)
        Zte = (Xte - mu) / sd
        with torch.no_grad():
            lg_clean = head_clean(torch.from_numpy(Xte).float().to(device))
            lg_rob = head_rob(torch.from_numpy(Xte).float().to(device))
        acc_clean = (lg_clean.argmax(1).cpu().numpy() == Yte).mean() * 100
        acc_rob = (lg_rob.argmax(1).cpu().numpy() == Yte).mean() * 100
        ahat = ridge.predict(Zte)
        # 路由策略：|α̂| > τ 用鲁棒头，否则原始头（扫描多个 τ）
        pc, pr = lg_clean.argmax(1).cpu().numpy(), lg_rob.argmax(1).cpu().numpy()
        acc_by_tau = {}
        for tau in TAU_GRID:
            use_rob = np.abs(ahat) > tau
            pred = np.where(use_rob, pr, pc)
            acc_by_tau[tau] = round((pred == Yte).mean() * 100, 2)
        acc_route = acc_by_tau[TAU_DEFAULT]
        rows.append({
            "alpha": a, "acc_clean_head": round(acc_clean, 2), "acc_robust_head": round(acc_rob, 2),
            "acc_router": acc_route, "oracle": round(max(acc_clean, acc_rob), 2),
            "ahat_mean": round(float(ahat.mean()), 4),
            **{f"router_tau{t}": acc_by_tau[t] for t in TAU_GRID},
        })
        print(f"[routing] α={a:+.2f} | clean头 {acc_clean:6.2f} | 鲁棒头 {acc_rob:6.2f} | "
              f"**路由 {acc_route:6.2f}** | oracle {rows[-1]['oracle']:6.2f} | "
              f"α̂={ahat.mean():+.3f} | 各τ: " + " ".join(f"{t}:{acc_by_tau[t]:.2f}" for t in TAU_GRID))

    def mean_of(k):
        return round(float(np.mean([r[k] for r in rows])), 2)

    print(f"\n[routing] 7 点均值：clean头 {mean_of('acc_clean_head')} | 鲁棒头 {mean_of('acc_robust_head')} | "
          f"**路由(τ=0.15) {mean_of('acc_router')}** | oracle {mean_of('oracle')}")
    print("[routing] 各 τ 的 7 点均值: " + " | ".join(
        f"τ={t}: {round(float(np.mean([r[f'router_tau{t}'] for r in rows])), 2)}" for t in TAU_GRID))

    out_dir = os.path.join(get_outputs_root(args.dataset), "v4_alpha_routing")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"routing_{args.arch}.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[routing] 已保存: {p}")


if __name__ == "__main__":
    main()
