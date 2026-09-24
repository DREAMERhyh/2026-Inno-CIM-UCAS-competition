"""M9b：失真是否让网络更接近神经坍缩（NC）几何？（台账 §16.2）

背景：本项目有一个未解释的异常——正侧失真下有效秩掉 20.6%（§13.6），
而 Fisher 判别比在 +0.3 (0.45669) 反而**高于** −0.3 (0.39061)，尽管 +0.3 的精度远低。
NC 的判别特征恰是「类均值收敛到等角等模的单纯形框架（ETF）+ Fisher 最大化」这个组合。

本脚本在**与 §13.6 完全同源的口径**上算 NC 的两个量：
  A(α) = 类均值两两夹角余弦与 ETF 理想值 −1/(C−1) 的平均绝对偏离
  V(α) = 去均值类均值的模长变异系数
并**交叉校验** v5_rank_collapse/rank_stats.csv（同一批特征应复现 rank_eff 与 fisher）。

判据已在台账 §16.2 跑前写死，本文件不改判据。
0 GPU / CPU / num_workers=0 / 产物记 4 位小数。

跑法：python v7_nc_geometry.py
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
from torchvision import datasets, transforms

from models.simple_cnn import SimpleCNN
from utils.nonlinearity import register_nonlinearity_hooks, remove_hooks
from utils.paths import get_ckpt_root, get_outputs_root

MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

ALPHAS = (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5)
N_TEST = 10000
OUT_DIRNAME = "v7_nc_geometry"
REF_CSV = os.path.join("outputs_cifar100", "v5_rank_collapse", "rank_stats.csv")
REF_TOL = 0.02          # 交叉校验：相对偏差上限（CPU/GPU 浮点差异之下仍应远小于此）
ND = 4                  # 小数位（§16.2 要求）


def log(*a) -> None:
    print(*a, flush=True)


# ------------------------------------------------------------------
def build_loaders(batch_size: int = 256):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    te = datasets.CIFAR100(root="./data", train=False, download=False, transform=tf)
    from torch.utils.data import DataLoader
    return DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=0)


@torch.no_grad()
def extract(model, loader, device, readout, alpha: float, limit: int):
    """与 v5_probe_rank_collapse.features 同口径：取 readout 的输入"""
    store: Dict[str, torch.Tensor] = {}
    h = readout.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0]))
    hooks = register_nonlinearity_hooks(model, alpha) if alpha != 0.0 else []
    F, Y, n = [], [], 0
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
        if hooks:
            remove_hooks(hooks)
    return np.concatenate(F)[:limit], np.concatenate(Y)[:limit]


def rank_stats(X, Y) -> Tuple[float, float, float]:
    """与 v5_probe_rank_collapse.rank_stats 逐字一致（用于交叉校验）"""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = (s ** 2) / np.sum(s ** 2)
    p = p[p > 0]
    rank_eff = float(np.exp(-np.sum(p * np.log(p))))
    cum = np.cumsum(p)
    r99 = int(np.searchsorted(cum, 0.99) + 1)
    kappa = float(s[0] / max(s[r99 - 1], 1e-12))
    Sb = Sw = 0.0
    mu_all = X.mean(0)
    for c in np.unique(Y):
        Xc_ = X[Y == c]
        mu_c = Xc_.mean(0)
        Sb += len(Xc_) * np.sum((mu_c - mu_all) ** 2)
        Sw += np.sum((Xc_ - mu_c) ** 2)
    return rank_eff, kappa, float(Sb / max(Sw, 1e-12))


def nc_metrics(X, Y) -> dict:
    """NC 的两个被测量（定义见台账 §16.2）"""
    classes = np.unique(Y)
    C = len(classes)
    mus = np.stack([X[Y == c].mean(0) for c in classes])
    mu_bar = X.mean(0)
    ideal = -1.0 / (C - 1)

    def geom(M):
        nrm = np.linalg.norm(M, axis=1)
        U = M / np.maximum(nrm[:, None], 1e-12)
        G = U @ U.T
        iu = np.triu_indices(C, k=1)
        cos = G[iu]
        return (float(np.mean(np.abs(cos - ideal))),
                float(np.std(nrm) / max(float(np.mean(nrm)), 1e-12)),
                cos, nrm)

    A, V, cos_c, nrm_c = geom(mus - mu_bar)      # 去全局均值（NC 的定义）
    A_u, V_u, cos_u, nrm_u = geom(mus)           # 未去均值（补充记录）
    return {"C": C, "ideal_cos": ideal,
            "A_centered": A, "V_centered": V,
            "A_uncentered": A_u, "V_uncentered": V_u,
            "cos_mean": float(np.mean(cos_c)), "cos_std": float(np.std(cos_c)),
            "cos_min": float(np.min(cos_c)), "cos_max": float(np.max(cos_c)),
            "norm_cv_centered": V, "norm_mean_centered": float(np.mean(nrm_c)),
            # 供复算用（§16.2 要求：没有 mu_bar 就重建不出 mu_tilde_c）
            "C": int(C), "mu_bar_norm": float(np.linalg.norm(mu_bar)),
            "mus": mus, "mu_bar": mu_bar}


def verdict(rows: Sequence[dict]) -> Tuple[str, str]:
    """按 §16.2 预登记判据判决（跑后不改）"""
    base = next(r for r in rows if r["alpha"] == 0.0)
    pos = [r for r in rows if r["alpha"] > 0]
    a_down = sum(1 for r in pos if r["A_centered"] < base["A_centered"])
    v_down = sum(1 for r in pos if r["V_centered"] < base["V_centered"])
    a_up = sum(1 for r in pos if r["A_centered"] > base["A_centered"])
    v_up = sum(1 for r in pos if r["V_centered"] > base["V_centered"])
    n = len(pos)
    detail = (f"A 下降 {a_down}/{n}、V 下降 {v_down}/{n}；"
              f"A 上升 {a_up}/{n}、V 上升 {v_up}/{n}")
    if a_down == n and v_down == n:
        return "①", detail + " ⇒ 病理性 NC 化"
    if a_up == n or v_up == n:
        return "②", detail + " ⇒ Fisher 升高不能用 NC 解释"
    return "③", detail + " ⇒ 不分离"


# ------------------------------------------------------------------
def cross_check(rows: Sequence[dict], repo: str) -> bool:
    path = os.path.join(repo, REF_CSV)
    if not os.path.exists(path):
        log(f"[nc] ⚠ 找不到参考 {REF_CSV}，跳过交叉校验")
        return True
    with open(path, encoding="utf-8") as fh:
        ref = {float(r["alpha"]): r for r in csv.DictReader(fh)}
    ok = True
    log("[nc] 交叉校验（对 v5_rank_collapse/rank_stats.csv，同源特征）")
    for r in rows:
        rr = ref.get(r["alpha"])
        if rr is None:
            continue
        for key, col in (("rank_eff", "rank_eff"), ("fisher", "fisher"), ("kappa99", "kappa99")):
            a, b = r[key], float(rr[col])
            rel = abs(a - b) / max(abs(b), 1e-12)
            good = rel <= REF_TOL
            ok = ok and good
            flag = "✅" if good else "❌"
            log(f"[nc]   α={r['alpha']:+.1f} {key:<9} 本脚本 {a:10.5f} vs v5 {b:10.5f} "
                f"相对差 {rel:.2e} {flag}")
    return ok


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(get_outputs_root("cifar100"), OUT_DIRNAME)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"[nc] 目标目录已存在且非空，拒绝覆盖（红线 3）：{out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cpu"          # §16.2：0 GPU
    model = SimpleCNN(num_classes=100)
    wpath = os.path.join(get_ckpt_root("cifar100"), "simple_cnn", "best_model.pth")
    ck = torch.load(wpath, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) else ck)
    model.to(device).eval()
    from v3_readout_repair import get_readout_module
    readout = get_readout_module(model)
    log(f"[nc] backbone=simple_cnn device={device} 读出层={type(readout).__name__} "
        f"(in_features={readout.in_features}) ckpt={wpath}")

    loader = build_loaders()
    rows: List[dict] = []
    means_store: Dict[str, np.ndarray] = {}
    for a in ALPHAS:
        X, Y = extract(model, loader, device, readout, a, N_TEST)
        m = nc_metrics(X, Y)
        rank_eff, kappa, fisher = rank_stats(X, Y)
        means_store[f"mus_{a:+.1f}"] = m["mus"]
        means_store[f"mu_bar_{a:+.1f}"] = m["mu_bar"]
        row = {"alpha": float(a), "n": int(len(Y)), "C": m["C"],
               "A_centered": round(m["A_centered"], ND), "V_centered": round(m["V_centered"], ND),
               "A_uncentered": round(m["A_uncentered"], ND),
               "V_uncentered": round(m["V_uncentered"], ND),
               "mu_bar_norm": round(m["mu_bar_norm"], ND),
               "cos_mean": round(m["cos_mean"], ND), "cos_std": round(m["cos_std"], ND),
               "cos_min": round(m["cos_min"], ND), "cos_max": round(m["cos_max"], ND),
               "ideal_cos": round(m["ideal_cos"], ND),
               "rank_eff": round(rank_eff, 5), "kappa99": round(kappa, 5),
               "fisher": round(fisher, 5)}
        rows.append(row)
        log(f"[nc] α={a:+.1f}  A={row['A_centered']:.4f} V={row['V_centered']:.4f} "
            f"| cos 均值 {row['cos_mean']:+.4f} 理想 {row['ideal_cos']:+.4f} "
            f"| rank_eff {row['rank_eff']:.3f} fisher {row['fisher']:.5f}")

    if not cross_check(rows, repo):
        raise SystemExit("[nc] ❌ 交叉校验失败：特征口径与 v5 不一致，已停止，未落盘。")

    br, detail = verdict(rows)
    log(f"[nc] 判决：分支 {br} —— {detail}")

    csv_path = os.path.join(out_dir, "nc_geometry.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    npz_path = os.path.join(out_dir, "class_means.npz")
    np.savez_compressed(npz_path, alpha=np.array(ALPHAS), **means_store)
    log(f"[nc] 已保存: {npz_path}（{len(ALPHAS)} 个 α 的类均值与全局均值，可据此重建 mu_tilde_c）")

    with open(os.path.join(out_dir, "nc_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"script": "v7_nc_geometry.py", "argv": sys.argv, "device": device,
                   "backbone_ckpt": wpath, "alphas": list(ALPHAS), "n_test": N_TEST,
                   "C": rows[0]["C"], "class_mean_shape": [rows[0]["C"], 128],
                   "mu_bar_norm_per_alpha": {f"{r['alpha']:+.1f}": r["mu_bar_norm"] for r in rows},
                   "feature": "readout 输入（SimpleCNN fc 输入，128 维 GAP）",
                   "decimals": ND, "ref_csv": REF_CSV, "ref_tol": REF_TOL,
                   "verdict_branch": br, "verdict_detail": detail,
                   "literature_note": "NC 判据为概念性描述；LITERATURE_NOTES §二 那条 ICML 摘要仍标 ⚠ 未核实，未作依据",
                   "git_commit": subprocess.run(
                       ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                       cwd=repo).stdout.strip(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "python": platform.python_version(), "platform": platform.platform()},
                  fh, ensure_ascii=False, indent=2)
    log(f"[nc] 已保存: {csv_path}")


if __name__ == "__main__":
    main()
