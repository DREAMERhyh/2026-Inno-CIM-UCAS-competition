"""
第五阶段 P0-2：完整配方端到端实测（确立/证伪 "≈54.5"）

配方 = 冻结主干 + 双线性头（原始 + 24ep α-混合头）+ 岭回归 α 估计器（13 点网格）+ τ 路由。
与 v4 版的三处升级：① 24 epoch 头（新标准协议）；② τ 在**验证集**上选（不再用测试集）；
③ 3 seeds；④ 报告 α 估计器 R²/MAE。

预登记判据：mean7 > 54.0 且 clean > 60.5 且 +0.3 > 48.5 同时成立 → "部署配方确立"。

用法：python v5_full_recipe.py --arch simple_cnn_mp --dataset cifar100 --seeds 42,43,44
产物：outputs_cifar100/v5_full_recipe/recipe_{arch}.csv（逐 seed × 逐 α）
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils.nonlinearity import register_nonlinearity_hooks, remove_hooks
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
}
GRID = [0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15, 0.2, -0.2, 0.25, -0.25, 0.3, -0.3]
EVAL = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
TAUS = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25]
N_TRAIN, N_VAL, N_TEST = 20000, 5000, 10000


def build_arch(name, nc):
    if name.startswith("simple_cnn_mp"):
        from v3_methods_simplecnn import build_arch as ba
        return ba(name, nc)
    from models.simple_cnn import SimpleCNN
    return SimpleCNN(num_classes=nc)


def make_loaders(dataset):
    mean, std = STATS[dataset]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    D = datasets.CIFAR100 if dataset == "cifar100" else datasets.CIFAR10
    tr = D(root="./data", train=True, download=False, transform=tf)
    te = D(root="./data", train=False, download=False, transform=tf)
    return tr, te


@torch.no_grad()
def extract(model, ds, idx, device, alpha, readout, bs=512):
    """在给定索引子集上提取 readout 层输入特征"""
    store = {}
    h = readout.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0]))
    pert = register_nonlinearity_hooks(model, alpha) if alpha != 0.0 else []
    F, Y = [], []
    try:
        for i in range(0, len(idx), bs):
            sel = idx[i:i + bs]
            X = torch.stack([ds[j][0] for j in sel]).to(device)
            y = torch.tensor([ds[j][1] for j in sel])
            model(X)
            F.append(store["f"].detach().cpu().numpy())
            Y.append(y.numpy())
    finally:
        h.remove()
        if pert:
            remove_hooks(pert)
    return np.concatenate(F), np.concatenate(Y)


def train_head(X, Y, nc, seed, epochs=24, lr=0.01):
    """α-混合鲁棒头：24 epoch（新标准协议）"""
    torch.manual_seed(seed)
    lin = nn.Linear(X.shape[1], nc)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    Xt, Yt = torch.from_numpy(X).float(), torch.from_numpy(Y).long()
    rng = np.random.RandomState(seed)
    order = np.arange(len(Xt))
    for ep in range(epochs):
        rng.shuffle(order)
        for i in range(0, len(order), 512):
            b = order[i:i + 512]
            opt.zero_grad()
            loss = crit(lin(Xt[b]), Yt[b])
            loss.backward()
            opt.step()
    lin.eval()
    return lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="simple_cnn_mp")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--backbone_tag", default="",
                    help="主干 checkpoint 后缀；空=按现有 s42 惯例取 v3_{arch}_clean_uniform（P0-5 新增）")
    ap.add_argument("--tag", default="", help="输出文件名后缀，避免覆盖既有 recipe（P0-5 新增）")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    nc = get_num_classes(args.dataset)
    seeds = [int(s) for s in args.seeds.split(",")]

    model = build_arch(args.arch, nc)
    ck = os.path.join(get_ckpt_root(args.dataset),
                      f"v3_{args.arch}_clean_uniform{args.backbone_tag}", "best_model.pth")
    if not os.path.exists(ck):
        ck = os.path.join(get_ckpt_root(args.dataset), "simple_cnn", "best_model.pth")
    sd = torch.load(ck, map_location=dev)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    readout = model.fc
    print(f"[P0-2] 主干={args.arch} 权重={ck}")

    ds_tr, ds_te = make_loaders(args.dataset)
    all_idx = np.arange(len(ds_tr))
    rng = np.random.RandomState(0)
    rng.shuffle(all_idx)
    tr_idx, val_idx = all_idx[:N_TRAIN], all_idx[N_TRAIN:N_TRAIN + N_VAL]
    te_idx = np.arange(N_TEST)
    print(f"[P0-2] 划分: train {len(tr_idx)} / val {len(val_idx)} / test {len(te_idx)}")

    rows = []
    for seed in seeds:
        # ---- 训练集多 α 特征（拟合 α 估计器 + 训练鲁棒头）----
        Xl, Yl = [], []
        for a in GRID:
            Xa, Ya = extract(model, ds_tr, tr_idx, dev, a, readout)
            Xl.append(Xa); Yl.append(Ya)
        Xtr, Ytr = np.concatenate(Xl), np.concatenate(Yl)
        Atr = np.concatenate([np.full(len(tr_idx), a) for a in GRID])

        mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-8
        Ztr = (Xtr - mu) / sd_
        ridge = Ridge(alpha=1.0).fit(Ztr, Atr)
        r2 = ridge.score(Ztr, Atr)
        mae = float(np.abs(ridge.predict(Ztr) - Atr).mean())
        print(f"\n[seed {seed}] α 估计器: R²={r2:.4f}  MAE={mae:.4f}（在 {args.arch} 特征上重训）")

        head_rob = train_head(Xtr, Ytr, nc, seed)
        head_cln = nn.Linear(Xtr.shape[1], nc)
        with torch.no_grad():
            head_cln.weight.copy_(readout.weight); head_cln.bias.copy_(readout.bias)
        head_cln.eval()

        # ---- 验证集：逐 α 评估两头的精度 → 选 τ ----
        def eval_split(idx, ds):
            res = {}
            for a in EVAL:
                Xa, Ya = extract(model, ds, idx, dev, a, readout)
                Za = (Xa - mu) / sd_
                with torch.no_grad():
                    lg_c = head_cln(torch.from_numpy(Xa).float()).numpy()
                    lg_r = head_rob(torch.from_numpy(Xa).float()).numpy()
                lc = lg_c.argmax(1)
                lr_ = lg_r.argmax(1)
                ah = ridge.predict(Za)
                res[a] = (lc, lr_, ah, Ya, lg_c, lg_r)
            return res

        val = eval_split(val_idx, ds_tr)
        best_tau, best_val = None, -1
        for tau in TAUS:
            tot = cor = 0
            for a, (lc, lr_, ah, Ya, _gc, _gr) in val.items():
                pred = np.where(np.abs(ah) > tau, lr_, lc)
                cor += (pred == Ya).sum(); tot += len(Ya)
            acc = 100 * cor / tot
            if acc > best_val:
                best_val, best_tau = acc, tau

        test = eval_split(te_idx, ds_te)
        for a in EVAL:
            lc, lr_, ah, Ya, lg_c, lg_r = test[a]
            acc_c = 100 * (lc == Ya).mean()
            acc_r = 100 * (lr_ == Ya).mean()
            pred = np.where(np.abs(ah) > best_tau, lr_, lc)
            acc_rt = 100 * (pred == Ya).mean()
            # 软门控：w = clip(|α̂|/0.3, 0, 1)，在两头 logits 上线性插值
            w = np.clip(np.abs(ah) / 0.3, 0.0, 1.0)[:, None]
            lg_soft = (1 - w) * lg_c + w * lg_r
            acc_soft = 100 * (lg_soft.argmax(1) == Ya).mean()
            rows.append({"seed": seed, "alpha": a, "acc_clean_head": round(acc_c, 2),
                         "acc_robust_head": round(acc_r, 2), "acc_router": round(acc_rt, 2),
                         "acc_softgate": round(acc_soft, 2),
                         "oracle": round(max(acc_c, acc_r), 2), "tau": best_tau,
                         "ahat_mean": round(float(ah.mean()), 4),
                         "ridge_R2": round(r2, 4), "ridge_MAE": round(mae, 4)})

    out = os.path.join(get_outputs_root(args.dataset), "v5_full_recipe")
    os.makedirs(out, exist_ok=True)
    fp = os.path.join(out, f"recipe_{args.arch}{args.tag}.csv")
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(f"\n=== 完整配方（{args.arch}，{len(seeds)} seeds）===")
    print(f"{'seed':>5} {'clean':>7} {'+0.3':>7} {'+0.2':>7} {'mean7':>7} {'soft7':>7} {'τ':>6} {'R²':>7} {'MAE':>7}")
    summ = []
    for seed in seeds:
        rs = [r for r in rows if r["seed"] == seed]
        d = {r["alpha"]: r for r in rs}
        clean = d[0.0]["acc_router"]; p3 = d[0.3]["acc_router"]; p2 = d[0.2]["acc_router"]
        mean7 = float(np.mean([d[a]["acc_router"] for a in EVAL]))
        soft7 = float(np.mean([d[a]["acc_softgate"] for a in EVAL]))
        summ.append((clean, p3, mean7, soft7))
        print(f"{seed:>5} {clean:>7.2f} {p3:>7.2f} {p2:>7.2f} {mean7:>7.2f} {soft7:>7.2f} "
              f"{d[0.0]['tau']:>6.3f} {d[0.0]['ridge_R2']:>7.4f} {d[0.0]['ridge_MAE']:>7.4f}")
    m = np.array(summ)
    print(f"{'均值':>5} {m[:,0].mean():>7.2f} {m[:,1].mean():>7.2f} "
          f"{'—':>7} {m[:,2].mean():>7.2f}   ±std: clean {m[:,0].std():.2f}, mean7 {m[:,2].std():.2f}")
    ok = (m[:, 2].mean() > 54.0) and (m[:, 0].mean() > 60.5) and (m[:, 1].mean() > 48.5)
    if args.dataset == "cifar100":
        print(f"\n[判据] mean7>54.0 且 clean>60.5 且 +0.3>48.5 → "
              f"{'✅ 部署配方确立' if ok else '❌ 未达标，需逐组件消融'}")
    else:
        print(f"\n[判据] {args.dataset} 无内置门槛（C1–C5 由预登记脚本 v5_p0c_criteria.py 单独评估，见台账 §13.7）")
    print(f"[saved] {fp}")


if __name__ == "__main__":
    main()
