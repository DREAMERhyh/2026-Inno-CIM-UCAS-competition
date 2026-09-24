"""M12：门控为什么能超过 oracle？—— 直接测两头错误的互补性（台账 §16.7）

背景：软门控在 C100 上 3/7 个点超过逐点选优的 oracle、C10 上 1/7。
当时的解释「插值即集成」跨数据集未复现后撤回，**且从未被直接测量**。

本节精确复现 §13.3b 的设置（直接 import v5_full_recipe 复用其函数以保忠实），
唯一改动是**把逐样本预测存下来**；并加两条硬自检证明成本优化不改结果。

判据已在台账 §16.7 跑前写死（经 claude-e9 批准并加前置）。
0 GPU / CPU / n_jobs=None / 4 位小数 / 新目录零覆盖。

跑法：python v7_gate_mechanism.py
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
from sklearn.linear_model import Ridge

import v5_full_recipe as R
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

DATASETS = ("cifar100", "cifar10")
SEEDS = (42, 43, 44)
GRID = R.GRID                     # 13 点 α 池（与 §13.3b 一致）
EVAL = R.EVAL                     # 7 点评估 α
N_TRAIN, N_TEST = R.N_TRAIN, R.N_TEST
ARCH = "simple_cnn_mp"
SELFCHECK_ALPHAS = (0.0, 0.3)     # 自检 ② 抽样
OUT_DIRNAME = "v7_gate_mechanism"
ND = 4
EXPECT_COUNT = {"cifar100": 3, "cifar10": 1}


def log(*a) -> None:
    print(*a, flush=True)


# ------------------------------------------------------------------
def selfcheck_extract(model, ds, idx, dev, readout, alphas) -> Tuple[bool, bool, list]:
    """自检②：两次独立 extract 必须逐位相同；且抽取不改模型 state_dict。"""
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rows, ok_extract, ok_state = [], True, True
    for a in alphas:
        X1, _ = R.extract(model, ds, idx[:20000], dev, a, readout)
        X2, _ = R.extract(model, ds, idx[:20000], dev, a, readout)
        same = np.array_equal(X1, X2)
        ok_extract = ok_extract and same
        rows.append({"alpha": float(a), "shape": list(X1.shape),
                     "array_equal": bool(same),
                     "max_abs_diff": float(np.abs(X1 - X2).max()) if not same else 0.0})
        log(f"[16.7] 自检② α={a:+.1f} extract×2 array_equal={same}")
    after = model.state_dict()
    ok_state = all(torch.equal(before[k], after[k]) for k in before)
    log(f"[16.7] 自检② 模型 state_dict 抽取前后逐位不变 = {ok_state}")
    return ok_extract, ok_state, rows


def selfcheck_train_head_mutation(Xtr, Ytr, nc) -> bool:
    """自检③：train_head 不得原地修改输入特征"""
    X0 = Xtr.copy()
    _ = R.train_head(Xtr, Ytr, nc, SEEDS[0])
    same = np.array_equal(X0, Xtr)
    log(f"[16.7] 自检③ train_head 未修改输入特征 = {same}")
    return bool(same)


def run_dataset(ds_name: str, out_dir: str) -> Tuple[List[dict], dict]:
    dev = "cpu"
    nc = get_num_classes(ds_name)
    model = R.build_arch(ARCH, nc)
    ck = os.path.join(get_ckpt_root(ds_name), f"v3_{ARCH}_clean_uniform", "best_model.pth")
    sd = torch.load(ck, map_location=dev)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    from v3_readout_repair import get_readout_module
    readout = get_readout_module(model)
    log(f"[16.7] {ds_name}: ckpt={ck} 读出层 in_features={readout.in_features}")

    ds_tr, ds_te = R.make_loaders(ds_name)
    all_idx = np.arange(len(ds_tr))
    rng = np.random.RandomState(0)
    rng.shuffle(all_idx)
    tr_idx = all_idx[:N_TRAIN]
    te_idx = np.arange(N_TEST)

    # ---- 自检② ----
    ok_ex, ok_st, sc2 = selfcheck_extract(model, ds_tr, tr_idx, dev, readout, SELFCHECK_ALPHAS)

    # ---- 训练特征：每个数据集只提一次（成本优化）----
    Xl, Yl = [], []
    for a in GRID:
        Xa, Ya = R.extract(model, ds_tr, tr_idx, dev, a, readout)
        Xl.append(Xa); Yl.append(Ya)
    Xtr, Ytr = np.concatenate(Xl), np.concatenate(Yl)
    Atr = np.concatenate([np.full(N_TRAIN, a) for a in GRID])
    log(f"[16.7] {ds_name}: 训练特征 {Xtr.shape}（13 α × {N_TRAIN}）")

    # ---- 自检③ ----
    ok_mut = selfcheck_train_head_mutation(Xtr, Ytr, nc)

    mu, sd_ = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = (Xtr - mu) / sd_
    ridge = Ridge(alpha=1.0).fit(Ztr, Atr)
    log(f"[16.7] {ds_name}: α 估计器 R²={ridge.score(Ztr, Atr):.4f}")

    rows: List[dict] = []
    persample: Dict[str, np.ndarray] = {}
    for seed in SEEDS:
        head_rob = R.train_head(Xtr, Ytr, nc, seed)
        head_cln = nn.Linear(Xtr.shape[1], nc)
        with torch.no_grad():
            head_cln.weight.copy_(readout.weight)
            head_cln.bias.copy_(readout.bias)
        head_cln.eval()
        for a in EVAL:
            Xa, Ya = R.extract(model, ds_te, te_idx, dev, a, readout)
            Za = (Xa - mu) / sd_
            with torch.no_grad():
                lg_c = head_cln(torch.from_numpy(Xa).float()).numpy()
                lg_r = head_rob(torch.from_numpy(Xa).float()).numpy()
            lc, lr_ = lg_c.argmax(1), lg_r.argmax(1)
            ah = ridge.predict(Za)
            w = np.clip(np.abs(ah) / 0.3, 0.0, 1.0)[:, None]
            ls = ((1 - w) * lg_c + w * lg_r).argmax(1)

            acc_c, acc_r = float((lc == Ya).mean()), float((lr_ == Ya).mean())
            acc_s = float((ls == Ya).mean())
            # oracle = 逐 α 取较优头（与 §13.3b 一致）
            oracle = lc if acc_c >= acc_r else lr_
            key = f"{ds_name}_s{seed}_a{a:+.1f}"
            persample[f"{key}_y"] = Ya.astype(np.int16)
            persample[f"{key}_lc"] = lc.astype(np.int16)
            persample[f"{key}_lr"] = lr_.astype(np.int16)
            persample[f"{key}_ls"] = ls.astype(np.int16)

            ec, er = (lc != Ya), (lr_ != Ya)
            df = float((ec & er).mean())
            pc, pr = float(ec.mean()), float(er.mean())
            denom = pc * pr
            dfr = float(df / denom) if denom > 1e-12 else float("nan")
            std_e = float(np.sqrt(pc * (1 - pc) * pr * (1 - pr)))
            phi = float((df - denom) / std_e) if std_e > 1e-12 else float("nan")
            rows.append({"dataset": ds_name, "seed": seed, "alpha": float(a),
                         "acc_clean": round(acc_c * 100, ND), "acc_robust": round(acc_r * 100, ND),
                         "acc_soft": round(acc_s * 100, ND),
                         "oracle": round(max(acc_c, acc_r) * 100, ND),
                         "soft_minus_oracle": round((acc_s - max(acc_c, acc_r)) * 100, ND),
                         "p_err_clean": round(pc, ND), "p_err_robust": round(pr, ND),
                         "double_fault": round(df, ND), "df_ratio": round(dfr, ND),
                         "err_phi": round(phi, ND),
                         "rescue": round(float(((ls == Ya) & (oracle != Ya)).mean()), ND),
                         "hurt": round(float(((ls != Ya) & (oracle == Ya)).mean()), ND)})

    return rows, {"selfcheck_extract": ok_ex, "selfcheck_state": ok_st,
                  "selfcheck_no_mutation": ok_mut, "selfcheck_extract_rows": sc2,
                  "ckpt": ck, "persample": persample}


def main() -> None:
    repo = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(get_outputs_root("cifar100"), OUT_DIRNAME)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise SystemExit(f"[16.7] 目标目录已存在且非空，拒绝覆盖（红线 3）：{out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[dict] = []
    meta: Dict[str, dict] = {}
    for ds in DATASETS:
        rows, m = run_dataset(ds, out_dir)
        all_rows.extend(rows)
        meta[ds] = {k: v for k, v in m.items() if k != "persample"}
        np.savez_compressed(os.path.join(out_dir, f"gate_persample_{ds}.npz"), **m["persample"])
        log(f"[16.7] {ds}: 逐样本预测已存")

    # ---- 自检①：复现 3/7 与 1/7 ----
    counts = {}
    for ds in DATASETS:
        rs = [r for r in all_rows if r["dataset"] == ds]
        n_beat = 0
        for a in EVAL:
            sub = [r for r in rs if abs(r["alpha"] - a) < 1e-9]
            if np.mean([r["acc_soft"] for r in sub]) > np.mean([r["oracle"] for r in sub]):
                n_beat += 1
        counts[ds] = n_beat
        log(f"[16.7] 自检① {ds}: soft>oracle 的 α 数 = {n_beat}/7（预期 {EXPECT_COUNT[ds]}）")

    ok1 = all(counts[ds] == EXPECT_COUNT[ds] for ds in DATASETS)
    ok2 = all(meta[ds]["selfcheck_extract"] and meta[ds]["selfcheck_state"] for ds in DATASETS)
    ok3 = all(meta[ds]["selfcheck_no_mutation"] for ds in DATASETS)
    log(f"[16.7] 自检汇总: ①复现计数={ok1}  ②抽取逐位={ok2}  ③不复用污染={ok3}")
    if not (ok1 and ok2 and ok3):
        raise SystemExit("[16.7] ❌ 自检未过，已停止，未写判决。")

    # ---- 跨数据集汇总（7 个 α 的中位数，3 seed 均值）----
    summ = {}
    for ds in DATASETS:
        rs = [r for r in all_rows if r["dataset"] == ds]
        agg = {}
        for a in EVAL:
            sub = [r for r in rs if abs(r["alpha"] - a) < 1e-9]
            agg[a] = {k: float(np.mean([r[k] for r in sub]))
                      for k in ("soft_minus_oracle", "df_ratio", "err_phi", "double_fault")}
        summ[ds] = {k: float(np.median([agg[a][k] for a in EVAL]))
                    for k in ("soft_minus_oracle", "df_ratio", "err_phi", "double_fault")}
        log(f"[16.7] {ds} 中位数: soft−oracle={summ[ds]['soft_minus_oracle']:+.4f}  "
            f"df_ratio={summ[ds]['df_ratio']:.4f}  phi={summ[ds]['err_phi']:+.4f}")

    d_phi = summ["cifar100"]["err_phi"] - summ["cifar10"]["err_phi"]
    d_dfr = summ["cifar100"]["df_ratio"] - summ["cifar10"]["df_ratio"]
    d_soft = summ["cifar100"]["soft_minus_oracle"] - summ["cifar10"]["soft_minus_oracle"]
    consistent = (d_phi < 0) and (d_dfr < 0) and (d_soft > 0)
    opposite = ((d_phi > 0) and (d_dfr > 0) and (d_soft < 0))
    if consistent:
        br, detail = "①", (f"C100 的 phi 更负({d_phi:+.4f})、df_ratio 更小({d_dfr:+.4f})、"
                           f"soft−oracle 更高({d_soft:+.4f}) ⇒ 机制成立，有条件恢复「插值即集成」")
    elif opposite:
        br, detail = "②", f"三项方向与机制预测相反 ⇒ 机制不成立，永久关闭「插值即集成」"
    else:
        br, detail = "③", (f"不分离：Δphi={d_phi:+.4f} Δdf_ratio={d_dfr:+.4f} "
                           f"Δ(soft−oracle)={d_soft:+.4f} ⇒ 不恢复主张")
    log(f"[16.7] 判决：分支 {br} —— {detail}")

    with open(os.path.join(out_dir, "gate_errors.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    with open(os.path.join(out_dir, "gate_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"script": "v7_gate_mechanism.py", "argv": sys.argv, "device": "cpu",
                   "arch": ARCH, "seeds": list(SEEDS), "grid": GRID, "eval": EVAL,
                   "n_train": N_TRAIN, "n_test": N_TEST,
                   "reproduction": "import v5_full_recipe 复用 build_arch/make_loaders/extract/train_head",
                   "wording": "鲁棒头未存盘、需重训；train_head 由 manual_seed 完全确定 ⇒ 同 seed 确定性复现，非重新实验",
                   "optimization": "训练特征 seed 无关 ⇒ 每数据集只提一次供 3 seed 复用；已由自检②③逐位核验",
                   "selfcheck": {"reproduced_counts": counts, "expect": EXPECT_COUNT,
                                 "extract_bitwise": meta, "all_pass": True},
                   "cross_dataset_medians": summ,
                   "deltas": {"phi": d_phi, "df_ratio": d_dfr, "soft_minus_oracle": d_soft},
                   "decimals": ND, "verdict_branch": br, "verdict_detail": detail,
                   "scope_note": "不修改 §13.3b/§13.7 任何既有判决；即使判①恢复的也只是有条件的主张",
                   "git_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                                capture_output=True, text=True,
                                                cwd=repo).stdout.strip()},
                  fh, ensure_ascii=False, indent=2)
    log(f"[16.7] 已保存: {os.path.join(out_dir, 'gate_errors.csv')}")


if __name__ == "__main__":
    main()
