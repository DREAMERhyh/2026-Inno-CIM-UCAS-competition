"""
第三阶段 1.2：四块未分析数据的收割分析（只读，输出到 stdout）

覆盖：
  A. ext2_vgg11 / ext2_resnet18 —— 高斯 vs 非线性 的架构依赖机制
  B. ext3 × {simple_cnn, vgg11, resnet18} —— 量化×非线性 的加性裁决
  C. ext5_dithering_vgg11 —— H1/H2/H3 三假设在 CIFAR-100 上的复判
  D. ext2_simplecnn —— 校准模型的跨扰动迁移（简单核对）
"""
import csv
import os

ROOT = "outputs_cifar100"


def rows(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def acc_at(rs, key, val, col="accuracy"):
    for r in rs:
        if abs(float(r[key]) - float(val)) < 1e-9:
            return float(r[col])
    return None


def sec(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ---------------------------------------------------------------- A
sec("A. ext2：高斯 vs 非线性，三架构对比（CIFAR-100）")
models = {
    "simple_cnn": f"{ROOT}/extension2_simplecnn/simple_cnn",
    "exp2": f"{ROOT}/extension2_simplecnn/exp2",
    "vgg11": f"{ROOT}/extension2_vgg11/vgg11",
    "resnet18": f"{ROOT}/extension2_resnet18/resnet18",
}
print(f"{'model':<12} {'clean':>7} {'nl@+0.3':>8} {'nl@-0.3':>8} {'σ=0.3':>7} {'σ=0.15':>7} "
      f"{'dropNL':>7} {'dropσ':>7} {'谁更致命':>10}")
for name, d in models.items():
    nl = rows(f"{d}/nonlinearity_sensitivity.csv")
    gs = rows(f"{d}/gaussian_sensitivity.csv")
    c = acc_at(nl, "alpha", 0.0)
    p = acc_at(nl, "alpha", 0.3)
    n = acc_at(nl, "alpha", -0.3)
    s3 = acc_at(gs, "noise_std", 0.3)
    s15 = acc_at(gs, "noise_std", 0.15)
    dn, dg = c - p, c - s3
    who = "高斯" if dg > dn else "非线性"
    print(f"{name:<12} {c:>7.2f} {p:>8.2f} {n:>8.2f} {s3:>7.2f} {s15:>7.2f} "
          f"{dn:>7.2f} {dg:>7.2f} {who:>10}")

sec("A2. 迁移检验：exp2(校准) vs simple_cnn(plain) 在各 σ 上的差")
d1 = f"{ROOT}/extension2_simplecnn/simple_cnn"
d2 = f"{ROOT}/extension2_simplecnn/exp2"
g1, g2 = rows(f"{d1}/gaussian_sensitivity.csv"), rows(f"{d2}/gaussian_sensitivity.csv")
print(f"{'σ':>6} {'plain':>8} {'exp2':>8} {'Δ(exp2-plain)':>14}")
mx = 0.0
for r in g1:
    s = r["noise_std"]
    a, b = float(r["accuracy"]), acc_at(g2, "noise_std", s)
    mx = max(mx, abs(b - a))
    print(f"{s:>6} {a:>8.2f} {b:>8.2f} {b-a:>+14.2f}")
print(f"最大绝对差 = {mx:.2f} pp  →  {'零迁移（≤1.2pp）' if mx <= 1.2 else '有迁移'}")

# ---------------------------------------------------------------- B
sec("B. ext3：量化 × 非线性 加性裁决（α=+0.3 主判据）")
for model in ("simplecnn", "vgg11", "resnet18"):
    rs = rows(f"{ROOT}/extension3_{model}/joint_error_summary.csv")
    key = {"simplecnn": "simple_cnn"}.get(model, model)
    nl = [r for r in rs if r["error_type"] == "nonlinearity_only"]
    qo = [r for r in rs if r["error_type"] == "quantization_only"]
    jt = [r for r in rs if r["error_type"] == "joint"]
    c0 = acc_at(qo, "num_bits", 8)  # α=0, 8bit（近无损基线）
    # nonlinearity_only 的 bit 标记因模型而异（simple_cnn=32, vgg/resnet=8）→ 取其最大值行
    nl_bits_max = max(int(r["num_bits"]) for r in nl)
    nl03 = acc_at([r for r in nl if int(r["num_bits"]) == nl_bits_max], "alpha", 0.3)
    print(f"\n--- {key} ---  clean(α=0,8bit)={c0:.2f}  nl_only(α=+0.3,{nl_bits_max}bit)={nl03:.2f}")
    print(f"{'bit':>4} {'quant_only':>11} {'joint@α+0.3':>12} {'独立预测':>9} {'Δ(实际−独立)':>14} {'裁决':>10}")
    for bit in (8, 6, 4, 3, 2):
        q = acc_at(qo, "num_bits", bit)
        j = acc_at([r for r in jt if int(r["num_bits"]) == bit], "alpha", 0.3)
        if q is None or j is None:
            continue
        pred = nl03 * q / c0
        d = j - pred
        if abs(d) < 1.0:
            verdict = "≈独立"
        elif d < 0:
            verdict = "超加性(更差)"
        else:
            verdict = "次加性(更好)"
        print(f"{bit:>4} {q:>11.2f} {j:>12.2f} {pred:>9.2f} {d:>+14.2f} {verdict:>10}")

# ---------------------------------------------------------------- C
sec("C. ext5：Dithering 三假设在 CIFAR-100 的复判（VGG-11, α=+0.3）")
d5 = f"{ROOT}/extension5_dithering_vgg11"
ga = rows(f"{d5}/group_a_bits_scan.csv")
print("组(a) bits 扫描：")
print(f"{'bits':>5} {'acc':>7} {'dose':>10}")
for r in ga:
    print(f"{int(r['num_bits']):>5} {float(r['accuracy']):>7.2f} {float(r['dose']):>10.4f}")
a8 = acc_at([r for r in ga], "num_bits", 8)
a4 = acc_at([r for r in ga], "num_bits", 4)
print(f"→ 4bit 增益 vs 8bit = {a4 - a8:+.2f} pp   （CIFAR-10 同口径 = +6.53）")

gb = rows(f"{d5}/group_b_noise_injection.csv")
print("\n组(b) 8bit + 噪声（H1 剂量匹配）：最优增益")
for nt in ("gaussian", "uniform"):
    sub = [r for r in gb if r["noise_type"] == nt]
    best = max(sub, key=lambda r: float(r["accuracy"]))
    print(f"  {nt:<9} 最优 σ={best['noise_std']:<7} acc={float(best['accuracy']):.2f} "
          f"→ vs 8bit {float(best['accuracy']) - a8:+.2f} pp")
gc = rows(f"{d5}/group_c_pure_noise.csv")
print("组(c) 等剂量纯噪声替代 4bit（target_dose≈0.2576）：")
for r in gc:
    print(f"  {r['noise_type']:<9} σ={r['noise_std']:<10} acc={float(r['accuracy']):.2f} "
          f"→ vs 8bit {float(r['accuracy']) - a8:+.2f} pp")

fl = rows(f"{d5}/flip_statistics.csv")
print("\n翻转统计（vs 8bit）：")
for r in fl:
    print(f"  {r['condition']:<20} saved={r['saved_vs_8bit']:>5} killed={r['killed_vs_8bit']:>5} "
          f"net={r['net_gain_vs_8bit']:>6}")

st = rows(f"{d5}/activation_stats.csv")
print("\n激活统计（H2）：分类器输入峰度 / 标准差 / 精度")
for r in st:
    print(f"  {r['condition']:<10} α={r['alpha']:<4} bits={r['num_bits']:>2} "
          f"kurt={float(r['classifier_input_kurtosis']):>7.4f} "
          f"std={float(r['classifier_input_std']):>7.4f}")
ac = rows(f"{d5}/arch_dithering_summary.csv")
print("\n组(d) 三架构（H3）：4bit 增益 vs maxpool 数")
for r in ac:
    print(f"  {r['model']:<12} maxpool={r['maxpool_count']:>2} 8bit={r['bits_8']:>6} "
          f"4bit={r['bits_4']:>6} gain={r['gain_4bit_vs_8bit']:>7}")

# ---------------------------------------------------------------- D
sec("D. 汇总：三定律在 CIFAR-100 台账上的复核")
import json
print("（clean 与 NAT/鲁棒变体的 clean、drop@0.3、中带均值，见 results_master.csv）")
