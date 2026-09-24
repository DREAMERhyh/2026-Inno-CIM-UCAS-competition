"""M8：读出的容量阶梯 —— 线性是不是天花板？（台账 §16.6）

六臂：logreg_C0.01 / linear（跨优化器参考臂）/ torch_linear（公平锚）
      / knn_k20 / mlp_bneck_32 / mlp_2layer_256

设计要点（详见 §16.6）：
- 容量问题的公平锚必须与 MLP 同族（torch_linear）——否则会同时动"架构"与"优化器"两个变量，
  而 §16.4 刚测出优化器/轮数一项就值 3.45 pp。
- torch 臂轮数跑 {50,150}，主比较取每臂较优者（容量问的是"这个架构最好能到多少"）。
- 判据门槛同时用 §16.4 的 Δ_proc（这条规矩第一次被使用）。

判据已在 §16.6 跑前写死（经 claude-e9 批准并补入分支 ③′）。
0 GPU / CPU / n_jobs=None / 4 位小数 / 新目录零覆盖。

跑法：python v7_readout_capacity.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import v7_probe_audit as V
from utils.nonlinearity import register_nonlinearity_hooks

ALPHAS = (0.0, 0.3, 0.5)
N_TRAIN, N_TEST = 8000, 10000
EPOCHS = (50, 150)
SEEDS = (0, 1, 2)
BATCH, LR = 256, 1e-3
KNN_K = 20
MARGIN_PP = 1.0
TORCH_ARMS = ("torch_linear", "mlp_bneck_32", "mlp_2layer_256")
NONLINEAR_ARMS = ("knn_k20", "mlp_bneck_32", "mlp_2layer_256")
HIDDEN = {"torch_linear": None, "mlp_bneck_32": 32, "mlp_2layer_256": 256}
PROC_CSV = os.path.join("outputs_cifar100", "v7_etf_readout", "proc_sensitivity.csv")
OUT_DIRNAME = "v7_readout_capacity"
ND = 4


def log(*a) -> None:
    print(*a, flush=True)


def read_delta_proc(repo: str) -> float:
    """从 §16.4 的产物读 α=+0.3 上的 Δ_proc（门槛 2 的取值来源）"""
    with open(os.path.join(repo, PROC_CSV), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if abs(float(r["alpha"]) - 0.3) < 1e-9:
                return float(r["delta_proc"])
    raise SystemExit("[m8] 读不到 §16.4 的 Δ_proc")


def build_head(d: int, c: int, hidden: int | None) -> nn.Module:
    if hidden is None:
        return nn.Linear(d, c)
    return nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, c))


def train_torch(ztr, ytr, c: int, hidden: int | None, epochs: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    m = build_head(ztr.shape[1], c, hidden)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lf = nn.CrossEntropyLoss()
    X = torch.as_tensor(ztr, dtype=torch.float32)
    Y = torch.as_tensor(ytr, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lf(m(X[idx]), Y[idx]).backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def acc_torch(m: nn.Module, zte, yte) -> float:
    out = m(torch.as_tensor(zte, dtype=torch.float32))
    return float((out.argmax(1).numpy() == yte).mean()) * 100.0


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(V.get_outputs_root("cifar100"), OUT_DIRNAME)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"[m8] 目标目录已存在且非空，拒绝覆盖（红线 3）：{out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    delta_proc = read_delta_proc(repo)
    log(f"[m8] 门槛 2 取自 §16.4 的 Δ_proc = {delta_proc} pp")

    device = "cpu"
    model = V.SimpleCNN(num_classes=100)
    wpath = os.path.join(V.get_ckpt_root("cifar100"), "simple_cnn", "best_model.pth")
    ck = torch.load(wpath, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) else ck)
    model.to(device).eval()

    tr_loader, te_loader = V.make_loaders("cifar100", 256, 0)
    per_alpha: Dict[float, Dict[str, dict]] = {}
    rows: List[dict] = []
    for a in ALPHAS:
        factory = (lambda: []) if a == 0.0 else (lambda a=a: register_nonlinearity_hooks(model, a))
        Xtr, ytr, _ = V.run_condition(model, tr_loader, device, factory(), N_TRAIN)
        Xte, yte, _ = V.run_condition(model, te_loader, device, factory(), N_TEST)
        sc = StandardScaler().fit(Xtr)
        ztr, zte = sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)
        log(f"[m8] alpha={a:+.1f} 特征就绪 {ztr.shape} / {zte.shape}")

        acc: Dict[str, float] = {}
        acc["logreg_C0.01"] = float(LogisticRegression(
            C=0.01, max_iter=1000, n_jobs=None).fit(ztr, ytr).score(zte, yte)) * 100.0
        acc["linear"] = float(LogisticRegression(
            max_iter=1000, n_jobs=None).fit(ztr, ytr).score(zte, yte)) * 100.0
        acc["knn_k20"] = float(KNeighborsClassifier(
            n_neighbors=KNN_K, n_jobs=None).fit(ztr, ytr).score(zte, yte)) * 100.0

        chosen: Dict[str, dict] = {}
        for arm in TORCH_ARMS:
            best = None
            for ep in EPOCHS:
                seeds = [acc_torch(train_torch(ztr, ytr, 100, HIDDEN[arm], ep, s), zte, yte)
                         for s in SEEDS]
                mean, std = float(np.mean(seeds)), float(np.std(seeds))
                log(f"[m8]   alpha={a:+.1f} {arm:<14} ep={ep:>3}  {mean:6.2f} ± {std:.2f}")
                if best is None or mean > best["mean"]:
                    best = {"mean": mean, "std": std, "epochs": ep, "seeds": seeds}
            acc[arm] = best["mean"]
            chosen[arm] = best
        per_alpha[a] = {"acc": acc, "chosen": chosen}
        rows.append({"alpha": float(a), **{k: round(v, ND) for k, v in acc.items()},
                     **{f"{k}_epochs": chosen[k]["epochs"] for k in TORCH_ARMS},
                     **{f"{k}_seed_std": round(chosen[k]["std"], ND) for k in TORCH_ARMS}})
        log(f"[m8] alpha={a:+.1f}  " + " | ".join(f"{k} {v:6.2f}" for k, v in acc.items()))

    # ---- 判决：α=+0.3，锚 = torch_linear ----
    t = per_alpha[0.3]["acc"]
    tcl = per_alpha[0.3]["chosen"]["torch_linear"]
    anchor = t["torch_linear"]
    clean = per_alpha[0.0]["acc"]
    hits, mid, low = [], [], []
    for arm in NONLINEAR_ARMS:
        gain = t[arm] - anchor
        # kNN 零训练、无随机性 ⇒ seed std = 0（构造性，非实测），如实注明
        sig = per_alpha[0.3]["chosen"].get(arm, {}).get("std", 0.0)
        clean_cost = clean["torch_linear"] - clean[arm]
        rec = (arm, gain, sig, clean_cost)
        if gain >= MARGIN_PP and gain > 2 * sig:
            (hits if gain > delta_proc else mid).append(rec)
        else:
            low.append(rec)
    def fmt(rs):
        return "; ".join(f"{a}: +0.3 增益 {g:+.2f}pp (seed std ±{s:.2f}), clean 代价 {c:+.2f}pp"
                         for a, g, s, c in rs) or "无"
    log(f"[m8] 锚 torch_linear @+0.3 = {anchor:.2f}（取 epochs={tcl['epochs']}）")
    log(f"[m8] 过三门槛: {fmt(hits)}")
    log(f"[m8] 过①③不过②（落 ③′）: {fmt(mid)}")
    log(f"[m8] 未过 1pp/2σ（落 ③）: {fmt(low)}")
    if hits and all(c <= MARGIN_PP for _, _, _, c in hits):
        br, detail = "①", fmt(hits) + " ⇒ 线性不是天花板"
    elif hits:
        br, detail = "②", fmt(hits) + " ⇒ 有得有失（clean 代价 >1pp）"
    elif mid:
        br, detail = "③′", fmt(mid) + f" ⇒ 有统计真实增益但 ≤ Δ_proc({delta_proc}pp)"
    else:
        br, detail = "③", fmt(low) + " ⇒ 线性已是本设置的容量天花板"
    log(f"[m8] 判决：分支 {br} —— {detail}")

    with open(os.path.join(out_dir, "capacity_ladder.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "capacity_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"script": "v7_readout_capacity.py", "argv": sys.argv, "device": device,
                   "backbone_ckpt": wpath, "alphas": list(ALPHAS),
                   "n_train": N_TRAIN, "n_test": N_TEST,
                   "torch_arms": list(TORCH_ARMS), "epochs_grid": list(EPOCHS),
                   "seeds": list(SEEDS), "knn_k": KNN_K,
                   "anchor": "torch_linear（公平锚，与 MLP 同族）",
                   "delta_proc_source": PROC_CSV, "delta_proc_pp": delta_proc,
                   "chosen_epochs": {f"{a:+.1f}": {k: per_alpha[a]["chosen"][k]["epochs"]
                                                   for k in TORCH_ARMS} for a in ALPHAS},
                   "decimals": ND, "verdict_branch": br, "verdict_detail": detail,
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                                capture_output=True, text=True,
                                                cwd=repo).stdout.strip(),
                   "torch": torch.__version__, "numpy": np.__version__},
                  fh, ensure_ascii=False, indent=2)
    log(f"[m8] 已保存: {os.path.join(out_dir, 'capacity_ladder.csv')}")


if __name__ == "__main__":
    main()
