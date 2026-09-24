"""§16.4：读出对**训练过程**的敏感度 —— 一个最小系统的标定

在同一批冻结特征上只更换**拟合过程**，测读数散布，作为后续一切"读出类"效应量的
可分辨门槛参照（§16.3 已指出过程自由度可能达 1.5~2.1 pp，而 ETF 效应只有 +0.02/+0.30）。

判据已在 §16.4 跑前写死（经 claude-e9 一字不改批准）。
0 GPU / CPU / n_jobs=None / 4 位小数 / 新文件零覆盖。

跑法：python v7_proc_sensitivity.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import v7_probe_audit as V
from utils.nonlinearity import register_nonlinearity_hooks

ALPHAS = (0.0, 0.3)
N_TRAIN, N_TEST = 8000, 10000
TORCH_SEED = 0
BATCH = 256
OUT_DIRNAME = "v7_etf_readout"
OUT_CSV = "proc_sensitivity.csv"
ND = 4
ETF_EFFECT = 0.30          # §16.3 在 +0.3 上的最大 ETF 效应（pp），用于算比值


def log(*a) -> None:
    print(*a, flush=True)


class Head(nn.Module):
    def __init__(self, d: int, c: int):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, x):
        return self.fc(x)


def train_torch(ztr, ytr, c: int, opt_name: str, epochs: int, seed: int = TORCH_SEED) -> Head:
    torch.manual_seed(seed)
    m = Head(ztr.shape[1], c)
    if opt_name == "adam":
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    elif opt_name == "sgd_mom":
        opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    else:
        raise ValueError(opt_name)
    lossf = nn.CrossEntropyLoss()
    X = torch.as_tensor(ztr, dtype=torch.float32)
    Y = torch.as_tensor(ytr, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(m(X[idx]), Y[idx]).backward()
            opt.step()
    m.eval()
    return m


@torch.no_grad()
def acc_torch(m: Head, zte, yte) -> float:
    return float((m(torch.as_tensor(zte, dtype=torch.float32)).argmax(1).numpy()
                  == yte).mean()) * 100.0


def acc_sk(ztr, ytr, zte, yte, solver: str, max_iter: int) -> float:
    clf = LogisticRegression(solver=solver, max_iter=max_iter, n_jobs=None)
    clf.fit(ztr, ytr)
    return float(clf.score(zte, yte)) * 100.0


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(V.get_outputs_root("cifar100"), OUT_DIRNAME)
    csv_path = os.path.join(out_dir, OUT_CSV)
    if os.path.exists(csv_path):
        raise SystemExit(f"[16.4] 拒绝覆盖已存在的文件（红线 3）：{csv_path}")

    device = "cpu"
    model = V.SimpleCNN(num_classes=100)
    wpath = os.path.join(V.get_ckpt_root("cifar100"), "simple_cnn", "best_model.pth")
    ck = torch.load(wpath, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) else ck)
    model.to(device).eval()
    log(f"[16.4] backbone=simple_cnn device={device} ckpt={wpath}")

    tr_loader, te_loader = V.make_loaders("cifar100", 256, 0)
    rows: List[dict] = []
    for a in ALPHAS:
        factory = (lambda: []) if a == 0.0 else (lambda a=a: register_nonlinearity_hooks(model, a))
        Xtr, ytr, _ = V.run_condition(model, tr_loader, device, factory(), N_TRAIN)
        Xte, yte, _ = V.run_condition(model, te_loader, device, factory(), N_TEST)
        sc = StandardScaler().fit(Xtr)
        ztr = sc.transform(Xtr).astype(np.float32)
        zte = sc.transform(Xte).astype(np.float32)
        log(f"[16.4] alpha={a:+.1f} 特征就绪 train={ztr.shape} test={zte.shape}")

        accs: Dict[str, float] = {}
        accs["sk_lbfgs"] = acc_sk(ztr, ytr, zte, yte, "lbfgs", 1000)
        accs["sk_saga"] = acc_sk(ztr, ytr, zte, yte, "saga", 100)
        for ep in (50, 150, 400):
            accs[f"adam_{ep}"] = acc_torch(train_torch(ztr, ytr, 100, "adam", ep), zte, yte)
        accs["sgd_mom_150"] = acc_torch(train_torch(ztr, ytr, 100, "sgd_mom", 150), zte, yte)

        for k, v in accs.items():
            log(f"[16.4] alpha={a:+.1f}  {k:<12} {v:6.2f}")
        delta = max(accs.values()) - min(accs.values())
        row = {"alpha": float(a), "n_procedures": len(accs),
               "delta_proc": round(delta, ND),
               "argmax_proc": max(accs, key=accs.get), "argmin_proc": min(accs, key=accs.get),
               "delta_over_etf_effect": round(delta / ETF_EFFECT, ND) if a == 0.3 else ""}
        row.update({k: round(v, ND) for k, v in accs.items()})
        rows.append(row)
        log(f"[16.4] alpha={a:+.1f}  Δ_proc = {delta:.4f} pp  "
            f"(max {row['argmax_proc']}, min {row['argmin_proc']})")

    brs = {r["alpha"]: ("①" if r["delta_proc"] >= 1.0 else "②") for r in rows}
    detail = "; ".join(f"α={a:+.1f} Δ_proc={r['delta_proc']:.2f}pp ⇒ 分支 {brs[a]}"
                       for a, r in ((r["alpha"], r) for r in rows))
    log(f"[16.4] 判决：{detail}")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "proc_sensitivity_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"script": "v7_proc_sensitivity.py", "argv": sys.argv, "device": device,
                   "backbone_ckpt": wpath, "alphas": list(ALPHAS),
                   "n_train": N_TRAIN, "n_test": N_TEST,
                   "procedures": {"sk_lbfgs": "LogisticRegression(lbfgs, max_iter=1000)",
                                  "sk_saga": "LogisticRegression(saga, max_iter=100)",
                                  "adam_50/150/400": "Adam lr=1e-3, batch 256",
                                  "sgd_mom_150": "SGD lr=0.1 momentum=0.9, 150 ep"},
                   "margin_pp": 1.0, "etf_effect_pp": ETF_EFFECT, "decimals": ND,
                   "verdict_branch": brs, "verdict_detail": detail,
                   "scope_note": "不是 5.55pp 之谜的实例（优化器/超参本来就不同）；"
                                 "用途是给 §16.3 的阴性结论提供「增益 < 训练过程自由度」的旁证",
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                                capture_output=True, text=True,
                                                cwd=repo).stdout.strip()},
                  fh, ensure_ascii=False, indent=2)
    log(f"[16.4] 已保存: {csv_path}")


if __name__ == "__main__":
    main()
