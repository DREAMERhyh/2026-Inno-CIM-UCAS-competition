"""
P4e-1 交叉评估：把 task2 的权重量到 v3 的评估路径下，反之亦然。

只做前向，不训练。两条路径都从各自脚本里**原样**导入评估函数，
不复制、不改写，保证"只换脚本跑"。
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.paths import get_ckpt_root, get_num_classes

ALPHAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
OUT = "outputs_cifar100/v6_p4e_crosspath"


def build(nc):
    from models.simple_cnn import SimpleCNN
    return SimpleCNN(num_classes=nc)


def load(model, path, dev):
    sd = torch.load(path, map_location=dev)
    model.load_state_dict(sd["model_state_dict"] if "model_state_dict" in sd else sd)
    model.to(dev).eval()
    return model


def run_v3_path(ckpt, dataset, dev):
    """走 v3_methods_simplecnn 的 scan()（原样导入）。"""
    import v3_methods_simplecnn as V
    model = load(build(get_num_classes(dataset)), ckpt, dev)
    _, test_loader = V.build_loaders(256, 0, dataset=dataset)
    res = V.scan(model, test_loader, dev, ALPHAS)
    return [a for a, _ in res], [round(acc, 4) for _, acc in res]


def run_task2_path(ckpt, dataset, dev):
    """走 task2_nat 的 evaluate_alpha_scan（原样导入），落到临时目录再读回。"""
    import task2_nat as T
    import torch.nn as nn
    from utils.data_loader import get_dataloaders
    model = load(build(get_num_classes(dataset)), ckpt, dev)
    _, test_loader = get_dataloaders(batch_size=256, num_workers=0, dataset=dataset)
    tmp = os.path.join(OUT, "_tmp_task2path")
    os.makedirs(tmp, exist_ok=True)
    T.evaluate_alpha_scan(model, test_loader, nn.CrossEntropyLoss(), dev,
                          ALPHAS, ckpt, tmp)
    rows = list(csv.DictReader(open(os.path.join(tmp, "alpha_sensitivity.csv"), encoding="utf-8")))
    return [float(r["alpha"]) for r in rows], [round(float(r["accuracy"]), 4) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = get_ckpt_root(args.dataset)
    os.makedirs(OUT, exist_ok=True)

    jobs = []
    for tag in ("", "_s43", "_s44"):
        jobs.append((f"task2-shared{tag or '_s42'}", f"{root}/nat_scratch_simplecnn{tag}/best_model.pth"))
    for tag in ("", "_s43", "_s44"):
        jobs.append((f"v3-perlayer{tag or '_s42'}", f"{root}/v3_plain_uniform{tag}/best_model.pth"))
    for tag in ("_sh_s42", "_sh_s43"):
        jobs.append((f"v3-shared{tag}", f"{root}/v3_plain_uniform{tag}/best_model.pth"))

    rows = []
    print(f"{'权重':<22} {'v3路径 +0.3':>12} {'task2路径 +0.3':>15} {'差(v3−t2)':>11} | {'v3 clean':>9} {'t2 clean':>9}")
    for name, ck in jobs:
        if not os.path.exists(ck):
            print(f"{name:<22} 缺权重，跳过")
            continue
        a1, v1 = run_v3_path(ck, args.dataset, dev)
        a2, v2 = run_task2_path(ck, args.dataset, dev)
        i3 = a1.index(0.3); i0 = a1.index(0.0)
        d = v1[i3] - v2[i3]
        rows.append(dict(weight=name, v3pos=v1[i3], t2pos=v2[i3], diff=round(d, 4),
                         v3clean=v1[i0], t2clean=v2[i0]))
        print(f"{name:<22} {v1[i3]:>12.2f} {v2[i3]:>15.2f} {d:>+11.2f} | {v1[i0]:>9.2f} {v2[i0]:>9.2f}")

    fp = os.path.join(OUT, "crosspath.csv")
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\n[saved] {fp}")

    print("\n=== 汇总 ===")
    for grp in ("task2-shared", "v3-perlayer", "v3-shared"):
        sub = [r for r in rows if r["weight"].startswith(grp)]
        if not sub: continue
        v3p = np.mean([r["v3pos"] for r in sub]); t2p = np.mean([r["t2pos"] for r in sub])
        print(f"  {grp:<14} n={len(sub)}  v3路径 +0.3 = {v3p:.2f}±{np.std([r['v3pos'] for r in sub]):.2f}"
              f"   task2路径 +0.3 = {t2p:.2f}±{np.std([r['t2pos'] for r in sub]):.2f}")


if __name__ == "__main__":
    main()
