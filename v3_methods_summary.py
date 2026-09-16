"""
第三阶段：方法实验结果汇总器（只读，v3_methods 下所有 run → 对比表）

输出：控制台表格 + outputs_cifar100/v3_methods/_summary.csv
用法：python v3_methods_summary.py
"""
import csv
import glob
import json
import os

ROOT = "outputs_cifar100/v3_methods"
OUT = os.path.join(ROOT, "_summary.csv")

# 历史参照（同协议，作为对照锚点）
REFS = {
    "clean 模型":            (59.99, 28.09, 42.91),
    "task2 NAT-scratch":     (60.70, 21.41, 46.20),
    "task2 NAT-finetune":    (60.16, 24.81, 45.79),
    "task3 Exp1(+校准)":     (60.87, 21.87, 40.58),
    "task3 Exp2(+校准+分层)": (61.22, 25.16, 43.10),
    "task3 Exp3(全研究最优)": (60.82, 29.89, 39.11),
    "readout-NAT(50k)":      (56.80, 44.09, 48.21),
}


def main():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/*")):
        if not os.path.isdir(d):
            continue
        mp = os.path.join(d, "metrics.json")
        scan = os.path.join(d, "alpha_sensitivity.csv")
        if not (os.path.exists(mp) and os.path.exists(scan)):
            continue
        mj = json.load(open(mp, encoding="utf-8"))
        amap = {}
        for r in csv.DictReader(open(scan, encoding="utf-8")):
            try:
                amap[round(float(r["alpha"]), 4)] = float(r["accuracy"])
            except (KeyError, ValueError):
                pass
        if not amap:
            continue
        clean = amap.get(0.0, "")
        p3 = amap.get(0.3, "")
        n3 = amap.get(-0.3, "")
        mid = [amap.get(a) for a in (-0.2, -0.1, 0.1, 0.2)]
        mid = round(sum(mid) / 4, 2) if all(v is not None for v in mid) else ""
        vals = [amap.get(a) for a in (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)]
        mean7 = round(sum(v for v in vals if v is not None) / max(1, sum(v is not None for v in vals)), 2)
        rows.append({
            "run": os.path.basename(d), "mode": mj.get("mode", ""), "profile": mj.get("profile", ""),
            "clean": clean, "neg0.3": n3, "pos0.3": p3, "mid_band": mid, "mean7": mean7,
            "best_epoch": mj.get("best_epoch", ""), "epochs": mj.get("total_epochs", ""),
            "elapsed_min": round(mj.get("elapsed_sec", 0) / 60, 1),
        })

    print(f"{'run':<28} {'clean':>7} {'-0.3':>7} {'+0.3':>7} {'mid':>7} {'mean7':>7} {'ep':>4} {'min':>6}")
    print("-" * 82)
    for r in rows:
        print(f"{r['run']:<28} {r['clean']:>7} {r['neg0.3']:>7} {r['pos0.3']:>7} "
              f"{r['mid_band']:>7} {r['mean7']:>7} {r['best_epoch']:>4} {r['elapsed_min']:>6}")
    print("-" * 82)
    print("历史参照：")
    for k, (c, p, n) in REFS.items():
        print(f"{k:<28} {c:>7} {n:>7} {p:>7}")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["run", "mode", "profile", "clean", "neg0.3", "pos0.3",
                            "mid_band", "mean7", "best_epoch", "epochs", "elapsed_min"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n已保存: {OUT}  （{len(rows)} 个 run）")


if __name__ == "__main__":
    main()
