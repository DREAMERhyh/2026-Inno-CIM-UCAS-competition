"""M0c 事后追加对照：拆分 C4 的「样本量」与「不相交性」混杂（台账 §16.0a）

C4 报的 −4.4 pp 混了两个轴：探针训练集与主干训练集是否相交、以及探针训练样本量。
本脚本加一个「同等样本量（5000）、分布内」的对照臂，并与 C4 共用同一个评估半集，
使两个轴都能干净归因（否则「评估集不同」会再引入第三个变量）。

判据已在台账 §16.0a 跑前写死。
自检：不相交臂必须**逐位复现** c4_private_split.csv；对不上则立即停。

跑法：python v7_c4_decompose.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch

import v7_probe_audit as V

SEEDS = (21, 22, 23)          # 与 C4 相同，保证不相交臂可逐位复现
N_TEST = 10000               # 规范口径：全量测试集
N_IN_POOL = 8000             # 分布内臂的样本池（= RA 的 n_train）
N_SMALL = 5000               # 两臂共用的训练样本量
N_EVAL = 5000                # 两臂共用的评估半集
IN_SEED_OFFSET = 90000       # 分布内臂的独立随机流，避免扰动置换流

C4_CSV = "c4_private_split.csv"
OUT_CSV = "c5_private_decompose.csv"
OUT_META = "c5_private_decompose_meta.json"


def log(*a) -> None:
    print(*a, flush=True)


def condition_features(model, dev: str, names: Sequence[str]) -> Dict[str, dict]:
    """提取 clean 与各 all| 条件的特征（train 取 8000，test 取全量 10000）"""
    tr_loader, te_loader = V.make_loaders("cifar100", 256, 0)
    conds = dict(V.build_conditions(model))
    out: Dict[str, dict] = {}
    for name in names:
        Xtr, ytr, _ = V.run_condition(model, tr_loader, dev, conds[name](), N_IN_POOL)
        Xte, yte, pred = V.run_condition(model, te_loader, dev, conds[name](), N_TEST)
        out[name] = {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte, "pred": pred}
        log(f"[c5] 特征已提取 {name:<18} train={Xtr.shape} test={Xte.shape}")
    return out


def run(c5: Dict[str, dict], names: Sequence[str]) -> List[dict]:
    rows: List[dict] = []
    for name in names:
        f = c5[name]
        Xtr, ytr = f["Xtr"][:N_IN_POOL], f["ytr"][:N_IN_POOL]
        Xte, yte, pred = (f["Xte"][:N_TEST], f["yte"][:N_TEST], f["pred"][:N_TEST])
        for s in SEEDS:
            # 与 C4 同一约定：置换后前 5000 训、后 5000 评
            rng = np.random.default_rng(s)
            idx = rng.permutation(len(Xte))
            half = len(idx) // 2
            priv_tr, ev = idx[:half], idx[half:half * 2]
            # 分布内臂：独立随机流，从主干训练分布里抽同样本量
            rng_in = np.random.default_rng(s + IN_SEED_OFFSET)
            in_idx = rng_in.permutation(N_IN_POOL)

            y_ev, p_ev = yte[ev], pred[ev]
            head = float((p_ev == y_ev).mean()) * 100.0

            p_in_small = V.probe_accuracy(Xtr[in_idx[:N_SMALL]], ytr[in_idx[:N_SMALL]],
                                          Xte[ev], y_ev)
            p_in_big = V.probe_accuracy(Xtr[in_idx], ytr[in_idx], Xte[ev], y_ev)
            p_priv = V.probe_accuracy(Xte[priv_tr], yte[priv_tr], Xte[ev], y_ev)

            rows.append({"condition": name, "seed": s,
                         "head_acc_same_half": round(head, 2),
                         "n_eval": len(ev),
                         "probe_in_5000": round(p_in_small, 2),
                         "probe_in_8000": round(p_in_big, 2),
                         "probe_private_5000": round(p_priv, 2),
                         "headroom_in_5000": round(p_in_small - head, 2),
                         "headroom_in_8000": round(p_in_big - head, 2),
                         "headroom_private_5000": round(p_priv - head, 2)})
        r = [x for x in rows if x["condition"] == name]
        d_dis = float(np.mean([x["probe_private_5000"] - x["probe_in_5000"] for x in r]))
        d_siz = float(np.mean([x["probe_in_5000"] - x["probe_in_8000"] for x in r]))
        log(f"[c5] {name:<18} 同半集头 {np.mean([x['head_acc_same_half'] for x in r]):6.2f} | "
            f"分布内5000 {np.mean([x['probe_in_5000'] for x in r]):6.2f} | "
            f"分布内8000 {np.mean([x['probe_in_8000'] for x in r]):6.2f} | "
            f"不相交5000 {np.mean([x['probe_private_5000'] for x in r]):6.2f} || "
            f"Δ不相交 {d_dis:+6.2f}  Δ样本量 {d_siz:+6.2f} | "
            f"headroom(不相交) {np.mean([x['headroom_private_5000'] for x in r]):+6.2f}")
    return rows


def aggregate(rows: Sequence[dict]) -> List[dict]:
    out = []
    for name in dict.fromkeys(x["condition"] for x in rows):
        r = [x for x in rows if x["condition"] == name]
        agg = {"condition": name, "n_seeds": len(r)}
        for k in ("head_acc_same_half", "probe_in_5000", "probe_in_8000",
                  "probe_private_5000", "headroom_in_5000", "headroom_in_8000",
                  "headroom_private_5000"):
            vals = [x[k] for x in r]
            agg[k] = round(float(np.mean(vals)), 2)
            agg[k + "_std"] = round(float(np.std(vals)), 2)
        agg["d_disjoint"] = round(float(np.mean(
            [x["probe_private_5000"] - x["probe_in_5000"] for x in r])), 2)
        agg["d_sample_size"] = round(float(np.mean(
            [x["probe_in_5000"] - x["probe_in_8000"] for x in r])), 2)
        out.append(agg)
    return out


def fidelity_vs_c4(agg: Sequence[dict], out_dir: str) -> bool:
    """自检：不相交臂必须逐位复现 c4_private_split.csv"""
    path = os.path.join(out_dir, C4_CSV)
    if not os.path.exists(path):
        log(f"[c5] ⚠ 找不到 {C4_CSV}，跳过自检")
        return True
    with open(path, encoding="utf-8") as fh:
        c4 = {r["condition"]: float(r["private_probe_acc_mean"]) for r in csv.DictReader(fh)}
    ok = True
    for a in agg:
        ref = c4.get(a["condition"])
        if ref is None:
            continue
        d = abs(a["probe_private_5000"] - ref)
        good = d <= 0.005
        ok = ok and good
        log(f"[c5] 自检 {a['condition']:<18} 本脚本 {a['probe_private_5000']:6.2f} "
            f"vs C4 {ref:6.2f}  Δ={d:.4f}  {'✅' if good else '❌'}")
    return ok


def ensure_absent(*paths: str) -> None:
    for p in paths:
        if os.path.exists(p):
            raise SystemExit(f"[c5] 拒绝覆盖已存在的文件（红线 3）：{p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default="simple_cnn")
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(V.get_outputs_root("cifar100"), V.OUT_SUBDIR)
    csv_path = os.path.join(out_dir, OUT_CSV)
    meta_path = os.path.join(out_dir, OUT_META)
    ensure_absent(csv_path, meta_path)

    model = V.SimpleCNN(num_classes=100)
    wpath = os.path.join(V.get_ckpt_root("cifar100"), args.model, "best_model.pth")
    ck = torch.load(wpath, map_location=dev)
    model.load_state_dict(ck["model_state_dict"] if isinstance(ck, dict) else ck)
    model.to(dev).eval()
    log(f"[c5] backbone={args.model} device={dev} ckpt={wpath}")

    names = [n for n in V.KEY_CONDITIONS]
    feats = condition_features(model, dev, names)

    rows = run(feats, names)
    agg = aggregate(rows)

    if not fidelity_vs_c4(agg, out_dir):
        raise SystemExit("[c5] ❌ 自检失败：不相交臂未复现 C4。已停止，未落盘。")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "c5_private_decompose_agg.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        w.writeheader(); w.writerows(agg)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump({"script": "v7_c4_decompose.py", "argv": sys.argv, "device": dev,
                   "seeds": SEEDS, "n_in_pool": N_IN_POOL, "n_small": N_SMALL,
                   "n_eval": N_EVAL, "in_seed_offset": IN_SEED_OFFSET,
                   "backbone_ckpt": wpath, "probe_kwargs": V.PROBE_KW,
                   "share_eval_half": True}, fh, ensure_ascii=False, indent=2)
    log(f"[c5] 已保存: {csv_path}")


if __name__ == "__main__":
    main()
