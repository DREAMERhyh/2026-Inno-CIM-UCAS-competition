"""§16.2a Fisher 正负不对称的对数分解（事后追加对照，经 claude-e9 批准）

问题：同 |alpha| 下为什么 Fisher(+alpha) > Fisher(-alpha)？（NC 已在 §16.2 排除）

方法（纯代数恒等式，无拟合）：
    log[Fisher(+a)/Fisher(-a)] = log[Sb(+a)/Sb(-a)] - log[Sw(+a)/Sw(-a)]
                              =:  d_logSb          -   d_logSw

判据已在台账 §16.2a 跑前写死。自检：两项之差须与 log Fisher 之差一致到 1e-9。
0 GPU / CPU / 4 位小数 / 产物新文件（零覆盖）。

跑法：python v7_fisher_decompose.py
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

import v7_nc_geometry as G
from utils.nonlinearity import register_nonlinearity_hooks

ABS_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5)
N_TEST = 10000
TOL = 1e-9
ND = 4


def log(*a) -> None:
    print(*a, flush=True)


def sb_sw(X, Y) -> Tuple[float, float]:
    """类间散布 Sb 与类内散布 Sw（与 v5/v7 的 fisher 定义同口径）"""
    mu_all = X.mean(0)
    sb = sw = 0.0
    for c in np.unique(Y):
        xc = X[Y == c]
        mu_c = xc.mean(0)
        sb += len(xc) * np.sum((mu_c - mu_all) ** 2)
        sw += np.sum((xc - mu_c) ** 2)
    return float(sb), float(sw)


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(G.get_outputs_root("cifar100"), G.OUT_DIRNAME)
    csv_path = os.path.join(out_dir, "fisher_decompose.csv")
    if os.path.exists(csv_path):
        raise SystemExit(f"[16.2a] 拒绝覆盖已存在的文件（红线 3）：{csv_path}")

    device = "cpu"
    from models.simple_cnn import SimpleCNN
    from v3_readout_repair import get_readout_module
    model = SimpleCNN(num_classes=100)
    wpath = os.path.join(G.get_ckpt_root("cifar100"), "simple_cnn", "best_model.pth")
    ck = torch.load(wpath, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) else ck)
    model.to(device).eval()
    readout = get_readout_module(model)
    log(f"[16.2a] backbone=simple_cnn device={device} ckpt={wpath}")

    loader = G.build_loaders()
    stats: Dict[float, Tuple[float, float]] = {}
    for a in (0.0,) + tuple(x for v in ABS_ALPHAS for x in (v, -v)):
        X, Y = G.extract(model, loader, device, readout, a, N_TEST)
        sb, sw = sb_sw(X, Y)
        stats[a] = (sb, sw)
        log(f"[16.2a] alpha={a:+.1f}  Sb={sb:.6e}  Sw={sw:.6e}  fisher={sb/sw:.5f}")

    rows: List[dict] = []
    ok = True
    for a in ABS_ALPHAS:
        sbp, swp = stats[a]
        sbm, swm = stats[-a]
        d_sb = float(np.log(sbp / sbm))
        d_sw = float(np.log(swp / swm))
        lhs = d_sb - d_sw
        rhs = float(np.log((sbp / swp) / (sbm / swm)))
        good = abs(lhs - rhs) <= TOL
        ok = ok and good
        rows.append({"abs_alpha": a,
                     "Sb_pos": round(sbp, 2), "Sb_neg": round(sbm, 2),
                     "Sw_pos": round(swp, 2), "Sw_neg": round(swm, 2),
                     "d_logSb": round(d_sb, ND), "d_logSw": round(d_sw, ND),
                     "fisher_ratio_log": round(rhs, ND),
                     "identity_residual": f"{abs(lhs - rhs):.3e}",
                     "identity_ok": good,
                     "dominant": ("Sb" if abs(d_sb) > abs(d_sw) else "Sw")})
        log(f"[16.2a] |a|={a:.1f}  dlogSb={d_sb:+.4f}  dlogSw={d_sw:+.4f}  "
            f"log(Fisher 比)={rhs:+.4f}  残差={abs(lhs - rhs):.2e}  "
            f"主导={'Sb' if abs(d_sb) > abs(d_sw) else 'Sw'}")

    if not ok:
        raise SystemExit("[16.2a] ❌ 恒等自检失败，已停止，未落盘。")

    sb_dom = sum(1 for r in rows if r["dominant"] == "Sb" and r["d_logSb"] > 0)
    sw_dom = sum(1 for r in rows if r["dominant"] == "Sw" and r["d_logSw"] < 0)
    n = len(rows)
    if sb_dom == n:
        br, detail = "①", f"5/5 由类间散布主导且 dlogSb>0（{sb_dom}/{n}）⇒ Fisher 不对称由类间散布主导"
    elif sw_dom == n:
        br, detail = "②", f"5/5 由类内散布主导且 dlogSw<0（{sw_dom}/{n}）⇒ Fisher 不对称由类内散布主导"
    else:
        br, detail = "③", (f"不分离：Sb 主导且正 {sb_dom}/{n}、Sw 主导且负 {sw_dom}/{n} "
                           f"⇒ 两项量级相当，不给单一归因")
    log(f"[16.2a] 判决：分支 {br} —— {detail}")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "fisher_decompose_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"script": "v7_fisher_decompose.py", "argv": sys.argv, "device": device,
                   "backbone_ckpt": wpath, "abs_alphas": list(ABS_ALPHAS), "n_test": N_TEST,
                   "identity_tol": TOL, "decimals": ND,
                   "verdict_branch": br, "verdict_detail": detail,
                   "scope_note": "只描述 Fisher 不对称的来源；不修改 §13.6 的维度丢失叙述，也不改 §16.2 分支②",
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                                capture_output=True, text=True,
                                                cwd=repo).stdout.strip()},
                  fh, ensure_ascii=False, indent=2)
    log(f"[16.2a] 已保存: {csv_path}")


if __name__ == "__main__":
    main()
