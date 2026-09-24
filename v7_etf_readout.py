"""M8b：ETF / 原型读出是否更抗输入扰动？（台账 §16.3）

设计要点（详见 §16.3）：主干冻结时，「自由学的映射 + 固定 ETF 原型」与线性头**等价**
（W = A^T E），故不可测。本节改为五臂：
  L_sk  sklearn 线性头（与 §16.0 同口径的参考臂）
  L_0   torch 线性头 λ=0（与 ETF 臂共用优化器/轮数 ⇒ 公平基线）
  NCM   最近类均值（零训练）
  ETF_λ torch 线性头 + ETF 正则（λ ∈ {0.1, 1.0}）
  RND   固定随机正交投影 + 固定随机 ETF + 固定分配（零训练；floor，不作判决）

跨优化器红线：主比较一律在 torch 族内部；L_sk 只作与 §16.0 的对接参考。
判据已在 §16.3 跑前写死。0 GPU / CPU / 4 位小数 / n_jobs=None。

跑法：python v7_etf_readout.py
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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import v7_probe_audit as V
from utils.nonlinearity import register_nonlinearity_hooks

ALPHAS = (0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5)
LAMBDAS = (0.1, 1.0)
N_TRAIN, N_TEST = 8000, 10000
EPOCHS, BATCH, LR = 150, 256, 1e-3
TORCH_SEED = 0
SEED_CHECK_ALPHA = 0.3
SEED_CHECK_SEEDS = (0, 1, 2)
OUT_DIRNAME = "v7_etf_readout"
ND = 4
MARGIN_PP = 1.0            # 判据门槛（pp）


def log(*a) -> None:
    print(*a, flush=True)


# ------------------------------------------------------------------
class LinearHead(nn.Module):
    def __init__(self, d: int, c: int):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, x):
        return self.fc(x)


def etf_penalty(w: torch.Tensor, c: int) -> torch.Tensor:
    """ETF 正则：夹角项 + 等模项（定义见台账 §16.3）"""
    ideal = -1.0 / (c - 1)
    wn = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    g = wn @ wn.t()
    iu = torch.triu_indices(c, c, offset=1)
    angle = ((g[iu[0], iu[1]] - ideal) ** 2).mean()
    nrm = w.norm(dim=1)
    norm_pen = (nrm.std() / nrm.mean().clamp_min(1e-12)) ** 2
    return angle + norm_pen


def train_head(Xtr, ytr, n_cls: int, lam: float, seed: int = TORCH_SEED) -> LinearHead:
    torch.manual_seed(seed)
    model = LinearHead(Xtr.shape[1], n_cls)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    X = torch.as_tensor(Xtr, dtype=torch.float32)
    Y = torch.as_tensor(ytr, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(X[idx]), Y[idx])
            if lam > 0:
                loss = loss + lam * etf_penalty(model.fc.weight, n_cls)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def head_acc(model: LinearHead, Xte, yte) -> float:
    out = model(torch.as_tensor(Xte, dtype=torch.float32))
    return float((out.argmax(1).numpy() == yte).mean()) * 100.0


def ncm_acc(Xtr, ytr, Xte, yte, n_cls: int, chunk: int = 1024) -> float:
    """最近类均值（欧氏）。零训练。分块算距离以控内存。"""
    mus = np.stack([Xtr[ytr == c].mean(0) for c in range(n_cls)])
    correct = 0
    for i in range(0, len(Xte), chunk):
        xb = Xte[i:i + chunk]
        d = ((xb[:, None, :] - mus[None, :, :]) ** 2).sum(-1)
        correct += int((d.argmin(1) == yte[i:i + chunk]).sum())
    return 100.0 * correct / len(Xte)


def build_fixed_etf_head(n_cls: int, d: int, seed: int = 0) -> np.ndarray:
    """固定随机 ETF 结构读出：全固定、零训练。返回 (C, d) 权重。"""
    rng = np.random.default_rng(seed)
    m = np.sqrt(n_cls / (n_cls - 1)) * (np.eye(n_cls) - np.ones((n_cls, n_cls)) / n_cls)
    q, _ = np.linalg.qr(rng.standard_normal((d, n_cls)))     # d x C，正交列
    w = m @ q.T                                              # C x d
    return w[rng.permutation(n_cls)]                         # 固定随机类别↔顶点分配


def scale_arm(Xtr, Xte):
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr).astype(np.float32), sc.transform(Xte).astype(np.float32)


# ------------------------------------------------------------------
def run_arm(feats: Dict[str, dict], n_cls: int) -> List[dict]:
    rows: List[dict] = []
    w_rnd = build_fixed_etf_head(n_cls, feats["clean"]["Xtr"].shape[1])
    for a in ALPHAS:
        name = f"all|alpha={a:+.1f}" if a != 0.0 else "clean"
        f = feats[name]
        Xtr, ytr = f["Xtr"][:N_TRAIN], f["ytr"][:N_TRAIN]
        Xte, yte = f["Xte"][:N_TEST], f["yte"][:N_TEST]
        Ztr, Zte = scale_arm(Xtr, Xte)

        row = {"alpha": float(a), "n_train": len(Ztr), "n_test": len(Zte),
               "head_acc": round(float((f["pred"][:N_TEST] == yte).mean()) * 100.0, ND),
               "L_sk": round(V.probe_accuracy(Xtr, ytr, Xte, yte), ND),
               "NCM": round(ncm_acc(Ztr, ytr, Zte, yte, n_cls), ND),
               "RND_fixed": round(float(((Zte @ w_rnd.T).argmax(1) == yte).mean()) * 100.0, ND),
               "L_0": round(head_acc(train_head(Ztr, ytr, n_cls, 0.0), Zte, yte), ND)}
        for lam in LAMBDAS:
            m = train_head(Ztr, ytr, n_cls, lam)
            row[f"ETF_{lam}"] = round(head_acc(m, Zte, yte), ND)
        rows.append(row)
        log(f"[m8b] α={a:+.1f}  冻结头 {row['head_acc']:6.2f} | L_sk {row['L_sk']:6.2f} | "
            f"L_0 {row['L_0']:6.2f} | NCM {row['NCM']:6.2f} | "
            + " | ".join(f"ETF_{l} {row[f'ETF_{l}']:6.2f}" for l in LAMBDAS)
            + f" | RND {row['RND_fixed']:5.2f}")
    return rows


def run_seed_check(feats: Dict[str, dict], n_cls: int) -> List[dict]:
    name = f"all|alpha={SEED_CHECK_ALPHA:+.1f}"
    f = feats[name]
    ztr, zte = scale_arm(f["Xtr"][:N_TRAIN], f["Xte"][:N_TEST])
    ytr, yte = f["ytr"][:N_TRAIN], f["yte"][:N_TEST]
    out = []
    for arm, lam in (("L_0", 0.0), (f"ETF_{LAMBDAS[0]}", LAMBDAS[0])):
        accs = [head_acc(train_head(ztr, ytr, n_cls, lam, seed=s), zte, yte)
                for s in SEED_CHECK_SEEDS]
        out.append({"arm": arm, "alpha": SEED_CHECK_ALPHA, "seeds": list(SEED_CHECK_SEEDS),
                    "mean": round(float(np.mean(accs)), ND),
                    "std": round(float(np.std(accs)), ND),
                    "min": round(float(np.min(accs)), ND),
                    "max": round(float(np.max(accs)), ND)})
        log(f"[m8b] seed 检查 {arm:<10} α=+0.3  {np.mean(accs):6.2f} ± {np.std(accs):.2f} "
            f"(3 seed)")
    return out


def verdict(rows: Sequence[dict], seed_rows: Sequence[dict]) -> Tuple[str, str]:
    """按 §16.3 预登记判据判决"""
    base = next(r for r in rows if r["alpha"] == 0.0)
    tgt = next(r for r in rows if r["alpha"] == SEED_CHECK_ALPHA)
    sig = {r["arm"]: r["std"] for r in seed_rows}

    cands = ["NCM"] + [f"ETF_{l}" for l in LAMBDAS]
    hits = []
    for arm in cands:
        gain = tgt[arm] - tgt["L_0"]
        s = sig.get(arm, sig.get("L_0", 0.0))
        clean_cost = base["L_0"] - base[arm]
        if gain >= MARGIN_PP and gain > 2 * s:
            hits.append((arm, gain, clean_cost))
    detail = "; ".join(
        f"{a}: +0.3 增益 {g:+.2f}pp, clean 代价 {c:+.2f}pp" for a, g, c in hits) or \
        "无臂达到门槛（≥1pp 且 >2σ）"
    if hits:
        if all(c <= MARGIN_PP for _, _, c in hits):
            return "①", detail + " ⇒ 尾部占优且 clean 代价小"
        return "②", detail + " ⇒ 有得有失（clean 代价 >1pp）"
    ncm_gain = tgt["NCM"] - tgt["L_0"]
    return "③", (f"{detail}；NCM 在 +0.3 的增益 {ncm_gain:+.2f}pp，"
                 f"L_0 的 seed 噪声 ±{sig.get('L_0', 0):.2f}pp ⇒ ETF/原型读出在本场景无增益")


# ------------------------------------------------------------------
def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(V.get_outputs_root("cifar100"), OUT_DIRNAME)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"[m8b] 目标目录已存在且非空，拒绝覆盖（红线 3）：{out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cpu"          # §16.3：0 GPU
    model = V.SimpleCNN(num_classes=100)
    wpath = os.path.join(V.get_ckpt_root("cifar100"), "simple_cnn", "best_model.pth")
    ck = torch.load(wpath, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) else ck)
    model.to(device).eval()
    n_cls = 100
    log(f"[m8b] backbone=simple_cnn device={device} ckpt={wpath} "
        f"epochs={EPOCHS} batch={BATCH} lr={LR}")

    tr_loader, te_loader = V.make_loaders("cifar100", 256, 0)
    # 本地构造条件表：v3/v7 的条件表只到 ±0.3（+0.4/+0.5），本节需要完整的 ±0.1…±0.5
    conds = {"clean": lambda: []}
    for a in ALPHAS:
        if a != 0.0:
            conds[f"all|alpha={a:+.1f}"] = (lambda a=a: register_nonlinearity_hooks(model, a))
    feats: Dict[str, dict] = {}
    for a in ALPHAS:
        name = f"all|alpha={a:+.1f}" if a != 0.0 else "clean"
        factory = conds[name]
        Xtr, ytr, _ = V.run_condition(model, tr_loader, device, factory(), N_TRAIN)
        Xte, yte, pred = V.run_condition(model, te_loader, device, factory(), N_TEST)
        feats[name] = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "pred": pred}
        log(f"[m8b] 特征已提取 {name}")

    rows = run_arm(feats, n_cls)
    seed_rows = run_seed_check(feats, n_cls)
    br, detail = verdict(rows, seed_rows)
    log(f"[m8b] 判决：分支 {br} —— {detail}")

    with open(os.path.join(out_dir, "etf_readout.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "etf_seed_check.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(seed_rows[0].keys()))
        w.writeheader(); w.writerows(seed_rows)
    with open(os.path.join(out_dir, "etf_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"script": "v7_etf_readout.py", "argv": sys.argv, "device": device,
                   "backbone_ckpt": wpath, "alphas": list(ALPHAS), "lambdas": list(LAMBDAS),
                   "n_train": N_TRAIN, "n_test": N_TEST,
                   "torch_hparams": {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                                     "seed": TORCH_SEED, "weight_decay": 0},
                   "seed_check": {"alpha": SEED_CHECK_ALPHA, "seeds": list(SEED_CHECK_SEEDS)},
                   "margin_pp": MARGIN_PP, "decimals": ND,
                   "cross_optimizer_note": "主比较在 torch 族内部；L_sk 仅作与 §16.0 对接参考",
                   "verdict_branch": br, "verdict_detail": detail,
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                                capture_output=True, text=True,
                                                cwd=repo).stdout.strip(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "python": platform.python_version(), "platform": platform.platform()},
                  fh, ensure_ascii=False, indent=2)
    log(f"[m8b] 已保存: {os.path.join(out_dir, 'etf_readout.csv')}")


if __name__ == "__main__":
    main()
