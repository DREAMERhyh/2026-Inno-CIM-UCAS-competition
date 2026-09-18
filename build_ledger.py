"""
第三阶段：全局实验台账生成器（只读扫描磁盘，不修改任何既有产物）

输出：results_master.csv
字段：run_id, date, dataset, model, weight_type, seed, epochs, alpha_protocol,
      clean, drop_at_0.3, mid_band_mean, neg_0.3, pos_0.3, params,
      pairing_check, artifact_path, verdict

口径（与本阶段方案一致）：
  clean        = α=0 精度
  drop_at_0.3  = clean − acc(+0.3)
  mid_band_mean= mean(acc @ α ∈ {-0.2,-0.1,+0.1,+0.2})
  pairing_check= |csv α=0 − 同 run metrics.best_test_acc|（无 metrics 记 n/a）
"""
import csv
import json
import os
import datetime

ROOT = "."
OUT_CSV = "results_master.csv"

# 模型参数量（取自 ablation/network_comparison 产物，硬编码仅用于台账展示）
PARAMS = {
    "simple_cnn": 254084,
    "vgg11": 9277284,
    "resnet18": 11220132,
    "robust_cnn": 270660,
    "exp2": 270660,
}

FIELDS = [
    "run_id", "date", "dataset", "model", "weight_type", "seed", "epochs",
    "alpha_protocol", "clean", "drop_at_0.3", "mid_band_mean", "neg_0.3", "pos_0.3",
    "params", "pairing_check", "artifact_path", "verdict",
]


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def alpha_map(rows, alpha_key="alpha", acc_key="accuracy"):
    """→ {float(alpha): float(acc)}"""
    m = {}
    for r in rows:
        try:
            m[round(float(r[alpha_key]), 4)] = float(r[acc_key])
        except (KeyError, ValueError):
            continue
    return m


def metrics_from_scan(amap):
    """从 α→acc 映射算 clean/drop/mid_band/±0.3"""
    clean = amap.get(0.0)
    pos03 = amap.get(0.3)
    neg03 = amap.get(-0.3)
    mid_pts = [amap.get(a) for a in (-0.2, -0.1, 0.1, 0.2)]
    mid = round(sum(mid_pts) / len(mid_pts), 4) if all(v is not None for v in mid_pts) else ""
    drop = round(clean - pos03, 4) if (clean is not None and pos03 is not None) else ""
    return {
        "clean": clean if clean is not None else "",
        "drop_at_0.3": drop,
        "mid_band_mean": mid,
        "neg_0.3": neg03 if neg03 is not None else "",
        "pos_0.3": pos03 if pos03 is not None else "",
    }


def mtime(path):
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except OSError:
        return ""


def add(rows, dataset, model, weight_type, protocol, scan_csv, metrics_path,
        artifact_path, verdict=""):
    amap = alpha_map(read_csv_rows(scan_csv))
    if not amap:
        return
    m = metrics_from_scan(amap)
    mj = read_json(metrics_path)
    best = mj.get("best_test_acc")
    if best is not None and m["clean"] != "":
        pc = round(abs(m["clean"] - float(best)), 4)
        pc = f"{pc:.4f} {'PASS' if pc <= 0.01 else 'FAIL'}"
    else:
        pc = "n/a"
    rows.append({
        "run_id": f"{dataset}|{model}|{weight_type}|{protocol}",
        "date": mtime(scan_csv) or mtime(metrics_path),
        "dataset": dataset,
        "model": model,
        "weight_type": weight_type,
        "seed": mj.get("seed", 42),
        "epochs": mj.get("total_epochs", ""),
        "alpha_protocol": protocol,
        "clean": m["clean"],
        "drop_at_0.3": m["drop_at_0.3"],
        "mid_band_mean": m["mid_band_mean"],
        "neg_0.3": m["neg_0.3"],
        "pos_0.3": m["pos_0.3"],
        "params": PARAMS.get(model, ""),
        "pairing_check": pc,
        "artifact_path": artifact_path.replace("\\", "/"),
        "verdict": verdict,
    })


def harvest_dataset(rows, root, dataset):
    """root = outputs/ 或 outputs_cifar100/"""
    # ---- clean（train.py 产物 + task1 的 7 点扫描）----
    for model in ("simple_cnn", "vgg11", "resnet18"):
        tag = model.replace("_", "")
        add(rows, dataset, model, "clean", "7pt",
            f"{root}/task1_{tag}/alpha_sensitivity.csv",
            f"{root}/{model}/metrics.json",
            f"{root}/task1_{tag}/",
            "train.py clean + task1 敏感性扫描")
    # ---- NAT（task2）----
    for tag in ("simplecnn", "vgg11", "resnet18"):
        for mode in ("scratch", "finetune"):
            d = f"{root}/task2_{tag}/{mode}"
            add(rows, dataset, tag.replace("simplecnn", "simple_cnn"), f"nat_{mode}", "7pt",
                f"{d}/alpha_sensitivity.csv", f"{d}/metrics.json", f"{d}/",
                f"NAT-{mode}")
    # ---- task3 鲁棒变体（仅 simplecnn 家族）----
    for exp in ("Exp1_CalibOnly", "Exp2_Calib+Layerwise", "Exp3_FullRobust"):
        d = f"{root}/task3_simplecnn/{exp}"
        if os.path.isdir(d):
            add(rows, dataset, "simple_cnn", f"task3_{exp}", "7pt",
                f"{d}/alpha_sensitivity.csv", f"{d}/metrics.json", f"{d}/",
                "task3 消融（robust_cnn 协议）")
    # ---- ext4 宽扫（15 点，10 组权重）----
    ext4 = f"{root}/extension4_alpha_wide/alpha_wide_scan_summary.csv"
    if os.path.exists(ext4):
        allrows = read_csv_rows(ext4)
        keys = sorted({(r["model"], r["weight_type"]) for r in allrows})
        for model, wt in keys:
            sub = [r for r in allrows if r["model"] == model and r["weight_type"] == wt]
            tmp = f"{root}/extension4_alpha_wide/_tmp_{model}_{wt}.csv"
            amap = alpha_map(sub)
            m = metrics_from_scan(amap)
            rows.append({
                "run_id": f"{dataset}|{model}|{wt}|15pt",
                "date": mtime(ext4), "dataset": dataset, "model": model,
                "weight_type": wt, "seed": 42, "epochs": "",
                "alpha_protocol": "15pt",
                "clean": m["clean"], "drop_at_0.3": m["drop_at_0.3"],
                "mid_band_mean": m["mid_band_mean"], "neg_0.3": m["neg_0.3"],
                "pos_0.3": m["pos_0.3"], "params": PARAMS.get(model, ""),
                "pairing_check": "n/a(扫描复用权重)",
                "artifact_path": f"{ext4} [rows={len(sub)}]",
                "verdict": "ext4 宽范围外推扫描",
            })
    # ---- ext6 深层 Exp2/Exp3 迁移 ----
    ext6_root = f"{root}/extension6_deep_robust"
    scan6 = f"{ext6_root}/alpha_wide_scan_deep_robust.csv"
    if os.path.exists(scan6):
        allrows = read_csv_rows(scan6)
        for model in sorted({r["model"] for r in allrows}):
            sub = [r for r in allrows if r["model"] == model]
            amap = alpha_map(sub)
            m = metrics_from_scan(amap)
            mp = f"{ext6_root}/{model}/metrics.json"
            mj = read_json(mp)
            best = mj.get("best_test_acc")
            pc = "n/a"
            if best is not None and m["clean"] != "":
                d = round(abs(m["clean"] - float(best)), 4)
                pc = f"{d:.4f} {'PASS' if d <= 0.01 else 'FAIL'}"
            rows.append({
                "run_id": f"{dataset}|{model}|exp2_robust|15pt",
                "date": mtime(mp) or mtime(scan6), "dataset": dataset, "model": model,
                "weight_type": "exp2_robust", "seed": 42,
                "epochs": mj.get("total_epochs", ""), "alpha_protocol": "15pt",
                "clean": m["clean"], "drop_at_0.3": m["drop_at_0.3"],
                "mid_band_mean": m["mid_band_mean"], "neg_0.3": m["neg_0.3"],
                "pos_0.3": m["pos_0.3"], "params": PARAMS.get(model, ""),
                "pairing_check": pc,
                "artifact_path": f"{ext6_root}/{model}/",
                "verdict": "ext6 Exp2 深层迁移",
            })
    # ---- ext6 Exp3 变体（第三阶段新增；扫描结果在 eval_exp3/ 下）----
    exp3_scan = f"{ext6_root}/eval_exp3/alpha_wide_scan_deep_robust_exp3.csv"
    exp3_rows = read_csv_rows(exp3_scan) if os.path.exists(exp3_scan) else []
    for model in ("resnet18", "vgg11"):
        d = f"{ext6_root}/{model}_exp3"
        scan = f"{d}/alpha_wide_scan_exp3.csv"   # 兼容旧路径
        mp = f"{d}/metrics.json"
        if os.path.exists(mp):
            mj = read_json(mp)
            sub = [r for r in exp3_rows if r.get("model") == model]
            amap = alpha_map(sub) if sub else (alpha_map(read_csv_rows(scan)) if os.path.exists(scan) else {})
            if amap:
                m = metrics_from_scan(amap)
                clean, drop, mid = m["clean"], m["drop_at_0.3"], m["mid_band_mean"]
                n3, p3 = m["neg_0.3"], m["pos_0.3"]
                pc = "n/a"
                if mj.get("best_test_acc") is not None and clean != "":
                    dd = round(abs(clean - float(mj["best_test_acc"])), 4)
                    pc = f"{dd:.4f} {'PASS' if dd <= 0.01 else 'FAIL'}"
            else:
                clean = drop = mid = n3 = p3 = ""
                pc = "待 eval 扫描"
            rows.append({
                "run_id": f"{dataset}|{model}|exp3_robust|15pt",
                "date": mtime(mp), "dataset": dataset, "model": model,
                "weight_type": "task3_Exp3_FullRobust", "seed": 42,
                "epochs": mj.get("total_epochs", ""), "alpha_protocol": "15pt",
                "clean": clean, "drop_at_0.3": drop, "mid_band_mean": mid,
                "neg_0.3": n3, "pos_0.3": p3, "params": PARAMS.get(model, ""),
                "pairing_check": pc,
                "artifact_path": f"{d}/",
                "verdict": "v3 新增：深层 Exp3（非对称采样）",
            })


def harvest_v3(rows):
    """第三阶段新增产物：v3_methods（训练类）与 v3_readout_repair（读出适配类）"""
    # --- v3_methods：每个 mode_profile 一个 run（两个数据集都扫）---
    import glob
    for root, ds in (("outputs_cifar100", "cifar100"), ("outputs", "cifar10")):
        for d in sorted(glob.glob(f"{root}/v3_methods/*")):
            if not os.path.isdir(d):
                continue
            name = os.path.basename(d)
            mp = os.path.join(d, "metrics.json")
            scan = os.path.join(d, "alpha_sensitivity.csv")
            if not os.path.exists(scan):
                continue
            mj = read_json(mp)
            amap = alpha_map(read_csv_rows(scan))
            if not amap:
                continue
            m = metrics_from_scan(amap)
            best = mj.get("best_test_acc")
            pc = "n/a"
            if best is not None and m["clean"] != "":
                dd = round(abs(m["clean"] - float(best)), 4)
                pc = f"{dd:.4f} {'PASS' if dd <= 0.01 else 'FAIL'}"
            rows.append({
                "run_id": f"{ds}|simple_cnn|v3_{name}|7pt(or wide)",
                "date": mtime(mp), "dataset": ds, "model": "simple_cnn",
                "weight_type": f"v3_{mj.get('mode','?')}_{mj.get('profile','?')}",
                "seed": mj.get("seed", 42), "epochs": mj.get("total_epochs", ""),
                "alpha_protocol": "v3", "clean": m["clean"], "drop_at_0.3": m["drop_at_0.3"],
                "mid_band_mean": m["mid_band_mean"], "neg_0.3": m["neg_0.3"],
                "pos_0.3": m["pos_0.3"], "params": PARAMS.get("simple_cnn", ""),
                "pairing_check": pc, "artifact_path": d.replace("\\", "/"),
                "verdict": mj.get("mode", "") + "/" + str(mj.get("profile", "")),
            })
    # --- v3_readout_repair：每个 (模型, 配置) 取 e_readout_NAT 与 a_original_fc 两行 ---
    for f in sorted(glob.glob("outputs_cifar100/v3_readout_repair/*.csv")):
        base = os.path.basename(f).replace(".csv", "")
        rr = read_csv_rows(f)
        if not rr:
            continue
        cols = [c for c in rr[0].keys() if c != "readout"]
        for target in ("a_original_fc", "e_readout_NAT"):
            row = next((r for r in rr if r["readout"] == target), None)
            if row is None:
                continue
            amap = {float(c): float(row[c]) for c in cols}
            m = metrics_from_scan(amap)
            rows.append({
                "run_id": f"cifar100|{base}|{target}|7pt",
                "date": mtime(f), "dataset": "cifar100", "model": base.split("_alpha")[0].replace("readout_repair_", ""),
                "weight_type": target, "seed": 42, "epochs": "",
                "alpha_protocol": "7pt(读出适配)", "clean": m["clean"],
                "drop_at_0.3": m["drop_at_0.3"], "mid_band_mean": m["mid_band_mean"],
                "neg_0.3": m["neg_0.3"], "pos_0.3": m["pos_0.3"],
                "params": "", "pairing_check": "n/a(读出适配)",
                "artifact_path": f.replace("\\", "/"),
                "verdict": "readout-NAT 实验",
            })


def main():
    rows = []
    if os.path.isdir("outputs"):
        harvest_dataset(rows, "outputs", "cifar10")
    if os.path.isdir("outputs_cifar100"):
        harvest_dataset(rows, "outputs_cifar100", "cifar100")
    if os.path.isdir("outputs_cifar100/v3_methods") or os.path.isdir("outputs_cifar100/v3_readout_repair"):
        harvest_v3(rows)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ledger] {len(rows)} rows -> {OUT_CSV}")
    fails = [r for r in rows if isinstance(r["pairing_check"], str) and "FAIL" in r["pairing_check"]]
    print(f"[ledger] pairing FAIL = {len(fails)}")
    for r in fails:
        print("   FAIL:", r["run_id"], r["pairing_check"])


if __name__ == "__main__":
    main()
