import csv, os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

EXT4 = "./outputs/extension4_alpha_wide/alpha_wide_scan_summary.csv"
EXT6 = "./outputs/extension6_deep_robust/alpha_wide_scan_deep_robust.csv"
OUT  = "./outputs/extension6_deep_robust/cross_architecture_summary.png"

def load(paths):
    data = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                data.setdefault((row["model"], row["weight_type"]), {})[float(row["alpha"])] = float(row["accuracy"])
    return data

curves = load([EXT4, EXT6])
STYLE = {
    "clean":       dict(color="#1f77b4", marker="o", ms=4, lw=1.6, label="Clean"),
    "nat_scratch": dict(color="#ff7f0e", marker="^", ms=4, lw=1.6, label="NAT Scratch"),
    "nat_finetune":dict(color="#2ca02c", marker="v", ms=4, lw=1.6, label="NAT Finetune"),
    "exp2_robust": dict(color="#d62728", marker="s", ms=6, lw=2.6, label="Exp2 Robust (Calib+NAT)"),
}
TITLES = {"simple_cnn": "SimpleCNN", "vgg11": "VGG-11", "resnet18": "ResNet-18"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=120, sharey=True)
for ax, model in zip(axes, ["simple_cnn", "vgg11", "resnet18"]):
    for wt in ["clean", "nat_scratch", "nat_finetune", "exp2_robust"]:
        if (model, wt) not in curves:
            continue
        pts = sorted(curves[(model, wt)].items())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], **STYLE[wt])
    ax.axvline(-0.3, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0.3, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.set_title(TITLES[model], fontsize=14, fontweight="bold")
    ax.set_xlabel("Nonlinearity Strength (alpha)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("Test Accuracy (%)", fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Exp2 Robust Scheme Transfer: Cross-Architecture Comparison", fontsize=15, fontweight="bold", y=1.00)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.savefig(OUT, bbox_inches="tight")
print(f"Saved: {OUT}")
