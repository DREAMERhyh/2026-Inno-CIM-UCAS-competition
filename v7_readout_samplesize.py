"""M8d：读出训练样本量的冗余曲线（台账 §16.5）

问题：现场校准到底需要多少标注样本？deploy_notes.md 现在完全没有这一项。
本项目读出适配一直用 20k 样本，从未扫过样本量。

判据已在 §16.5 跑前写死。拟合过程刻意钉死为 sklearn lbfgs
（§16.4 指出训练过程自由度可能达 1.5~2.1 pp，不能让它淹没样本量效应）。
0 GPU / CPU / n_jobs=None / 4 位小数 / 新目录零覆盖。

跑法：python v7_readout_samplesize.py
"""
from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

import v7_probe_audit as V
from utils.nonlinearity import register_nonlinearity_hooks

ALPHAS = (0.0, 0.3)
SIZES = (500, 1000, 2000, 4000, 8000, 16000, 20000)
SEEDS = (11, 22, 33)
N_TRAIN_POOL, N_TEST = 20000, 10000
CHANCE = 1.0
THRESH = 0.95
OUT_DIRNAME = "v7_readout_samplesize"
ND = 4


def log(*a) -> None:
    print(*a, flush=True)


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(V.get_outputs_root("cifar100"), OUT_DIRNAME)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"[m8d] 目标目录已存在且非空，拒绝覆盖（红线 3）：{out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cpu"
    model = V.SimpleCNN(num_classes=100)
    wpath = os.path.join(V.get_ckpt_root("cifar100"), "simple_cnn", "best_model.pth")
    ck = torch.load(wpath, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) else ck)
    model.to(device).eval()
    log(f"[m8d] backbone=simple_cnn device={device} ckpt={wpath} "
        f"sizes={SIZES} seeds={SEEDS}")

    tr_loader, te_loader = V.make_loaders("cifar100", 256, 0)
    feats: Dict[float, dict] = {}
    for a in ALPHAS:
        factory = (lambda: []) if a == 0.0 else (lambda a=a: register_nonlinearity_hooks(model, a))
        Xtr, ytr, _ = V.run_condition(model, tr_loader, device, factory(), N_TRAIN_POOL)
        Xte, yte, pred = V.run_condition(model, te_loader, device, factory(), N_TEST)
        feats[a] = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "pred": pred}
        log(f"[m8d] 特征已提取 alpha={a:+.1f}  train={Xtr.shape} test={Xte.shape}")

    rows: List[dict] = []
    summary: Dict[float, List[dict]] = {}
    for a in ALPHAS:
        f = feats[a]
        Xtr, ytr, Xte, yte = f["Xtr"], f["ytr"], f["Xte"], f["yte"]
        head = float((f["pred"] == yte).mean()) * 100.0
        per_n = []
        for n in SIZES:
            accs = []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                idx = rng.choice(len(Xtr), size=n, replace=False)
                accs.append(V.probe_accuracy(Xtr[idx], ytr[idx], Xte, yte))
            row = {"alpha": float(a), "n": n, "n_per_class": round(n / 100.0, 2),
                   "acc_mean": round(float(np.mean(accs)), ND),
                   "acc_std": round(float(np.std(accs)), ND),
                   "acc_min": round(float(np.min(accs)), ND),
                   "acc_max": round(float(np.max(accs)), ND),
                   "n_seeds": len(SEEDS), "head_acc": round(head, ND)}
            rows.append(row)
            per_n.append(row)
            log(f"[m8d] alpha={a:+.1f}  n={n:>6}  acc={row['acc_mean']:6.2f} "
                f"± {row['acc_std']:.2f}  (每类 {row['n_per_class']:.0f})")
        summary[a] = per_n

    verdicts: Dict[str, dict] = {}
    for a in ALPHAS:
        per_n = summary[a]
        a_max = per_n[-1]["acc_mean"]
        thr = CHANCE + THRESH * (a_max - CHANCE)
        n_star = next((r["n"] for r in per_n if r["acc_mean"] >= thr), None)
        if n_star is not None and n_star <= 2000:
            br = "①"
        elif n_star is not None and n_star <= 8000:
            br = "②"
        else:
            br = "③"
        verdicts[f"{a:+.1f}"] = {"A_max": a_max, "threshold_95pct": round(thr, ND),
                                "n_star": n_star, "branch": br,
                                "head_acc": per_n[0]["head_acc"]}
        log(f"[m8d] alpha={a:+.1f}  A_max={a_max:.2f}  95%阈值={thr:.2f}  "
            f"n*={n_star}  ⇒ 分支 {br}")

    with open(os.path.join(out_dir, "samplesize_curve.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "samplesize_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"script": "v7_readout_samplesize.py", "argv": sys.argv, "device": device,
                   "backbone_ckpt": wpath, "alphas": list(ALPHAS), "sizes": list(SIZES),
                   "seeds": list(SEEDS), "n_train_pool": N_TRAIN_POOL, "n_test": N_TEST,
                   "chance": CHANCE, "threshold_fraction": THRESH, "decimals": ND,
                   "fit_procedure": "sklearn LogisticRegression(lbfgs, max_iter=1000, n_jobs=None)",
                   "proc_fixed_note": "拟合过程刻意钉死；§16.4 指出过程自由度可达 1.5~2.1pp",
                   "literature_note": "arXiv 2310.03843 在 LITERATURE_NOTES §二 标 ⚠ 未核实，未作依据",
                   "verdicts": verdicts,
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                                capture_output=True, text=True,
                                                cwd=repo).stdout.strip(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "python": platform.python_version(), "platform": platform.platform()},
                  fh, ensure_ascii=False, indent=2)
    log(f"[m8d] 已保存: {os.path.join(out_dir, 'samplesize_curve.csv')}")


if __name__ == "__main__":
    main()
