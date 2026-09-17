"""
第五阶段 P1-5：路由在 α 分布漂移下的压力测试（零 GPU 成本，基于已有逐 α 结果重加权）

背景：训练时 α~U[±0.3]，但部署环境的失真分布可能不同（偏斜/双峰）。
问题：τ 路由（在 U[±0.3] 的验证集上选）在漂移分布下是否仍优于 always-robust / always-clean？

方法：从 v5_full_recipe 的逐 α 结果（每 α 上各策略的精度）出发，
      按**新的 α 分布**对 7 个 α 点重加权，得到各策略在漂移分布下的期望精度。
      注意：这是**重加权近似**（τ 未重选）——正是"漂移"的定义（部署时不能重选超参）。

用法：python v5_probe_alpha_drift.py
产物：outputs_cifar100/v5_alpha_drift/drift_stress.csv
"""
import csv
import os

import numpy as np

EV = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
SRC = "outputs_cifar100/v5_full_recipe/recipe_{arch}.csv"
OUT = "outputs_cifar100/v5_alpha_drift"


def mixture_weights(comps, ev=EV):
    """comps = [(mu, sigma, weight), ...] → 在 ev 上的归一化离散权重"""
    w = np.zeros(len(ev))
    for mu, sig, pi in comps:
        w += pi * np.exp(-0.5 * ((np.array(ev) - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
    return w / w.sum()


DISTS = {
    "① 训练分布 U[−0.3,+0.3]（参照）": None,      # 等权
    "② 偏斜 U[0,+0.3]（仅正向）": ("uniform_pos", None),
    "③ 偏斜 U[−0.3,0]（仅负向）": ("uniform_neg", None),
    "④ 双峰混合 0.5N(−0.2,0.05)+0.5N(+0.2,0.05)": [( -0.2, 0.05, 0.5), (0.2, 0.05, 0.5)],
    "⑤ 尾部集中 0.5·δ(−0.3)+0.5·δ(+0.3)": [(-0.3, 0.02, 0.5), (0.3, 0.02, 0.5)],
    "⑥ 单峰正向 N(+0.3,0.05)": [(0.3, 0.05, 1.0)],
}


def weights_for(name, ev=EV):
    spec = DISTS[name]
    if spec is None:
        return np.ones(len(ev)) / len(ev)
    if spec[0] == "uniform_pos":
        m = np.array([a >= 0 for a in ev], dtype=float)
        return m / m.sum()
    if spec[0] == "uniform_neg":
        m = np.array([a <= 0 for a in ev], dtype=float)
        return m / m.sum()
    return mixture_weights(spec, ev)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for arch in ("simple_cnn_mp", "simple_cnn"):
        fp = SRC.format(arch=arch)
        if not os.path.exists(fp):
            continue
        data = list(csv.DictReader(open(fp, encoding="utf-8")))
        # 每 α 上各策略的 3-seed 均值
        per = {}
        for a in EV:
            rs = [r for r in data if float(r["alpha"]) == a]
            per[a] = {k: float(np.mean([float(r[k]) for r in rs]))
                      for k in ("acc_clean_head", "acc_robust_head", "acc_router", "acc_softgate", "oracle")}
        print(f"\n=== {arch} ===")
        print(f"{'测试分布':<44} {'clean头':>8} {'robust头':>9} {'硬路由':>8} {'软门控':>8} {'oracle':>8} {'软−rob':>8}")
        for name in DISTS:
            w = weights_for(name)
            acc = {k: float(np.sum([w[i] * per[a][k] for i, a in enumerate(EV)]))
                   for k in per[EV[0]]}
            rows.append({"arch": arch, "dist": name, **{f"w_{a}": round(w[i], 4) for i, a in enumerate(EV)},
                         **{k: round(v, 2) for k, v in acc.items()},
                         "soft_minus_robust": round(acc["acc_softgate"] - acc["acc_robust_head"], 2),
                         "soft_minus_clean": round(acc["acc_softgate"] - acc["acc_clean_head"], 2)})
            print(f"{name:<44} {acc['acc_clean_head']:>8.2f} {acc['acc_robust_head']:>9.2f} "
                  f"{acc['acc_router']:>8.2f} {acc['acc_softgate']:>8.2f} {acc['oracle']:>8.2f} "
                  f"{acc['acc_softgate']-acc['acc_robust_head']:>+8.2f}")
    fp = os.path.join(OUT, "drift_stress.csv")
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] {fp}")


if __name__ == "__main__":
    main()
