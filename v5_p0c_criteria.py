"""
第五阶段 P0-C：CIFAR-10「mp 主干 + 双头 + 软门控」复现的预登记判据评估

判据（跑前写死于台账 §13.7，本脚本只做读取与裁决，不含任何调参）：
  C1 mean7(soft) > max(mean7(clean 头), mean7(robust 头))
  C2 clean(soft) >= clean(clean 头) - 0.5
  C3 +0.3(soft) > +0.3(clean 头) 且 +0.3(soft) >= +0.3(robust 头) - 1.0
  C4 soft 严格超过"两头较优者"的 α 点数 >= 2
  C5 mean7(mp·soft) >= mean7(plain·clean 头) + 2.0

用法：python v5_p0c_criteria.py --dataset cifar10
输入：{outputs_root}/v5_full_recipe/recipe_{simple_cnn_mp,simple_cnn}.csv
"""
import argparse
import csv
import os

import numpy as np

from utils.paths import get_outputs_root

EVAL = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
CIFAR10_PLAIN_MEAN7 = 76.49  # simple_cnn clean 基线（seed 42, 7 点），台账 §13.7


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def by_alpha(rows, seed, col):
    return np.array([float(r[col]) for r in rows if int(r["seed"]) == seed])


def summarize(rows, seeds):
    """→ {col: (mean7, per-α 均值向量)}"""
    out = {}
    for col in ("acc_clean_head", "acc_robust_head", "acc_router", "acc_softgate", "oracle"):
        vecs = [by_alpha(rows, s, col) for s in seeds]
        out[col] = np.array(vecs).mean(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--tag", default="", help="recipe 文件名后缀（P2-7 用，如 _bs43）；默认空=原行为")
    args = ap.parse_args()
    root = os.path.join(get_outputs_root(args.dataset), "v5_full_recipe")

    mp = load(os.path.join(root, f"recipe_simple_cnn_mp{args.tag}.csv"))
    plain_path = os.path.join(root, f"recipe_simple_cnn{args.tag}.csv")
    plain = load(plain_path) if os.path.exists(plain_path) else None
    seeds = sorted({int(r["seed"]) for r in mp})
    S = summarize(mp, seeds)

    def m7(col):
        return float(S[col].mean())

    def at(col, a):
        return float(S[col][EVAL.index(a)])

    print(f"\n=== P0-C 复现评估（{args.dataset} · simple_cnn_mp · {len(seeds)} seeds）===")
    print(f"{'α':>6} {'clean头':>9} {'robust头':>9} {'硬路由':>9} {'软门控':>9} {'oracle':>9}")
    for i, a in enumerate(EVAL):
        print(f"{a:>+6.1f} " + " ".join(f"{S[c][i]:>9.2f}" for c in
              ("acc_clean_head", "acc_robust_head", "acc_router", "acc_softgate", "oracle")))
    print(f"{'mean7':>6} " + " ".join(f"{m7(c):>9.2f}" for c in
          ("acc_clean_head", "acc_robust_head", "acc_router", "acc_softgate", "oracle")))

    base = max(m7("acc_clean_head"), m7("acc_robust_head"))
    n_better = int(sum(S["acc_softgate"][i] > max(S["acc_clean_head"][i], S["acc_robust_head"][i])
                       for i in range(len(EVAL))))

    c1 = m7("acc_softgate") > base
    c2 = at("acc_softgate", 0.0) >= at("acc_clean_head", 0.0) - 0.5
    c3 = (at("acc_softgate", 0.3) > at("acc_clean_head", 0.3)
          and at("acc_softgate", 0.3) >= at("acc_robust_head", 0.3) - 1.0)
    c4 = n_better >= 2
    c5, p_base = None, None
    if plain is not None:
        pseeds = sorted({int(r["seed"]) for r in plain})
        P = summarize(plain, pseeds)
        p_base = float(P["acc_clean_head"].mean()) if args.dataset == "cifar100" else CIFAR10_PLAIN_MEAN7
        c5 = m7("acc_softgate") >= p_base + 2.0
        print(f"\n[对照] plain 主干 clean 头 mean7 = {p_base:.2f}"
              f"（{'脚本实测' if args.dataset == 'cifar100' else '台账锚点'}）")

    print("\n--- 预登记判据裁决 ---")
    print(f"  C1 mean7(soft) > max(两头)          : {m7('acc_softgate'):.2f} > {base:.2f}  → {'成立' if c1 else '不成立'}")
    print(f"  C2 clean(soft) >= clean(clean头)-0.5: {at('acc_softgate',0.0):.2f} >= "
          f"{at('acc_clean_head',0.0)-0.5:.2f}  → {'成立' if c2 else '不成立'}")
    print(f"  C3 +0.3(soft) > +0.3(clean头) 且 ≥ robust-1.0: {at('acc_softgate',0.3):.2f} > "
          f"{at('acc_clean_head',0.3):.2f} 且 ≥ {at('acc_robust_head',0.3)-1.0:.2f}  → {'成立' if c3 else '不成立'}")
    print(f"  C4 soft 超两头较优者的点数 ≥ 2       : {n_better}/{len(EVAL)}  → {'成立' if c4 else '不成立'}")
    print(f"  C5 mean7(mp·soft) >= plain+2.0      : "
          + (f"{m7('acc_softgate'):.2f} >= {p_base+2.0:.2f}  → {'成立' if c5 else '不成立'}" if c5 is not None
             else "（缺 plain 对照，未评）"))
    if c1 and c2 and c3:
        print("\n  => 跨数据集复现成立（C1∧C2∧C3）：'插值即集成'判为数据集无关效应")
    elif not c1:
        print("\n  => C1 不成立：该效应为 CIFAR-100 特有，软门控维持'事后变体'标签")


if __name__ == "__main__":
    main()
