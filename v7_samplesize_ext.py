"""§16.5a：读出样本量 —— 把 n 推到 50000，把 n* 钉死

§16.5 的两条限定同源（A(20000) 不是渐近值；α=0 的 n* 裕度只有 0.023σ），根源都是样本量没推到头。
本节把 n 推到 train split 全量 50000，用**等宽对数斜率**判饱和（两段都是 ln 2.5）。

判据已在 §16.5a 跑前写死（经 claude-e9 一字不改批准）。
0 GPU / CPU / n_jobs=None / 4 位小数 / 新文件零覆盖。

跑法：python v7_samplesize_ext.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
from typing import Dict, List

import numpy as np
import torch

import v7_probe_audit as V
from utils.nonlinearity import register_nonlinearity_hooks

ALPHAS = (0.0, 0.3)
SIZES = (20000, 30000, 40000, 50000)
SEEDS = (11, 22, 33)          # 与 §16.5 相同
N_TRAIN_POOL, N_TEST = 50000, 10000
CHANCE, THRESH = 1.0, 0.95
REF_CSV = "samplesize_curve.csv"   # §16.5 的表，供 s_early 取数
OUT_CSV = "samplesize_ext.csv"
ND = 4


def log(*a) -> None:
    print(*a, flush=True)


def read_ref(out_dir: str) -> Dict[str, Dict[int, dict]]:
    """从 §16.5 的表读各 (alpha, n) 的 acc_mean 与 acc_std（供 s_early 与合并曲线用）"""
    ref: Dict[str, Dict[int, dict]] = {}
    with open(os.path.join(out_dir, REF_CSV), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            a = f"{float(r['alpha']):+.1f}"
            ref.setdefault(a, {})[int(r["n"])] = {"acc_mean": float(r["acc_mean"]),
                                                  "acc_std": float(r["acc_std"])}
    return ref


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(V.get_outputs_root("cifar100"), "v7_readout_samplesize")
    csv_path = os.path.join(out_dir, OUT_CSV)
    if os.path.exists(csv_path):
        raise SystemExit(f"[16.5a] 拒绝覆盖已存在的文件（红线 3）：{csv_path}")

    ref = read_ref(out_dir)
    log(f"[16.5a] 参考（§16.5 表）: " + "; ".join(
        f"α={a} A(8000)={v.get(8000)} A(20000)={v.get(20000)}" for a, v in ref.items()))

    device = "cpu"
    model = V.SimpleCNN(num_classes=100)
    wpath = os.path.join(V.get_ckpt_root("cifar100"), "simple_cnn", "best_model.pth")
    ck = torch.load(wpath, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) else ck)
    model.to(device).eval()
    log(f"[16.5a] backbone=simple_cnn device={device} ckpt={wpath}")

    tr_loader, te_loader = V.make_loaders("cifar100", 256, 0)
    rows: List[dict] = []
    for a in ALPHAS:
        factory = (lambda: []) if a == 0.0 else (lambda a=a: register_nonlinearity_hooks(model, a))
        Xtr, ytr, _ = V.run_condition(model, tr_loader, device, factory(), N_TRAIN_POOL)
        Xte, yte, _ = V.run_condition(model, te_loader, device, factory(), N_TEST)
        log(f"[16.5a] 特征已提取 alpha={a:+.1f} train={Xtr.shape} test={Xte.shape}")
        for n in SIZES:
            accs = []
            for s in SEEDS:
                rng = np.random.default_rng(s)
                idx = rng.choice(len(Xtr), size=n, replace=False)
                accs.append(V.probe_accuracy(Xtr[idx], ytr[idx], Xte, yte))
            rows.append({"alpha": float(a), "n": n, "n_per_class": round(n / 100.0, 2),
                         "acc_mean": round(float(np.mean(accs)), ND),
                         "acc_std": round(float(np.std(accs)), ND),
                         "acc_min": round(float(np.min(accs)), ND),
                         "acc_max": round(float(np.max(accs)), ND), "n_seeds": len(SEEDS)})
            log(f"[16.5a] alpha={a:+.1f} n={n:>6} acc={rows[-1]['acc_mean']:6.2f} "
                f"± {rows[-1]['acc_std']:.2f}")

    # ---- 饱和判据：等宽对数斜率 ----
    verdicts: Dict[str, dict] = {}
    for a in ALPHAS:
        key = f"{a:+.1f}"
        rs = [r for r in rows if abs(r["alpha"] - a) < 1e-9]
        a20 = next(r for r in rs if r["n"] == 20000)["acc_mean"]
        a50 = next(r for r in rs if r["n"] == 50000)["acc_mean"]
        a8 = ref[key][8000]["acc_mean"]
        a20_ref = ref[key][20000]["acc_mean"]
        w = math.log(2.5)
        s_early = (a20_ref - a8) / w            # 按预登记：取自 §16.5 表
        s_late = (a50 - a20) / w
        r_ratio = s_late / s_early if s_early else float("inf")
        br = "①" if r_ratio <= 0.5 else "②"

        thr = CHANCE + THRESH * (a50 - CHANCE)
        # n* 与裕度：在合并曲线（§16.5 的 n<=20000 + 本节 n>20000）上找
        merged = {n: dict(v) for n, v in ref[key].items()}
        merged.update({r["n"]: {"acc_mean": r["acc_mean"], "acc_std": r["acc_std"]}
                       for r in rs})
        n_star = next((n for n in sorted(merged) if merged[n]["acc_mean"] >= thr), None)
        margin = round(merged[n_star]["acc_mean"] - thr, ND) if n_star else None
        sig = merged[n_star]["acc_std"] if n_star else 0.0
        margin_sigma = round(margin / sig, 3) if (margin is not None and sig > 0) else None
        verdicts[key] = {"s_early": round(s_early, ND), "s_late": round(s_late, ND),
                         "r_ratio": round(r_ratio, ND), "branch": br,
                         "A_50000": a50, "threshold_95pct": round(thr, ND),
                         "n_star": n_star, "margin_pp": margin,
                         "n_star_seed_std": sig, "margin_sigma": margin_sigma}
        log(f"[16.5a] alpha={a:+.1f}  s_early={s_early:.4f}  s_late={s_late:.4f}  "
            f"r={r_ratio:.4f} ⇒ 分支 {br}")
        log(f"[16.5a]   A(50000)={a50:.2f}  T'={thr:.4f}  n*={n_star}  "
            f"裕度={margin} pp = {margin_sigma} σ")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w_.writeheader(); w_.writerows(rows)
    with open(os.path.join(out_dir, "samplesize_ext_meta.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"script": "v7_samplesize_ext.py", "argv": sys.argv, "device": device,
                   "backbone_ckpt": wpath, "alphas": list(ALPHAS), "sizes": list(SIZES),
                   "seeds": list(SEEDS), "n_train_pool": N_TRAIN_POOL, "n_test": N_TEST,
                   "slope_window": "ln(2.5) 两段等宽", "ref_csv": REF_CSV,
                   "fit_procedure": "sklearn LogisticRegression(lbfgs, max_iter=1000, n_jobs=None)",
                   "decimals": ND, "verdicts": verdicts,
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                                capture_output=True, text=True,
                                                cwd=repo).stdout.strip(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "python": platform.python_version(), "platform": platform.platform()},
                  fh, ensure_ascii=False, indent=2)
    log(f"[16.5a] 已保存: {csv_path}")


if __name__ == "__main__":
    main()
