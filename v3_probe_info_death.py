"""
第三阶段 3.1：信息死亡探针（Information-Death Probe）

问题：灾难尾（α=±0.3 附近）"训练不可达"的根因，是**信息被摧毁**（→ 信息论下界）
      还是**信息幸存、只是分类头脆**（→ 特征修复分支可救）？

设计：
  冻结 clean SimpleCNN；对同一批样本，在指定注入条件下提取 **fc 的输入特征**（GAP 后 128 维）；
  用该特征训练线性探针（logistic regression）预测**真标签**，在测试集上评估。
  对照三组数：
    clean_probe       : 干净特征 + 干净标签（信息上限参考）
    dist_probe        : 失真特征 + 干净标签（信息是否仍线性可分）
    model_acc         : 冻结分类头在该条件下的精度（现有可达性）
  判据：
    若 dist_probe ≈ 随机(1%)           → 信息被摧毁（尾部无解有信息论下界）
    若 dist_probe 显著 > model_acc     → 信息幸存、瓶颈在头部（可引出特征修复分支）

附带：同探针在 gaussian σ 上跑一遍，检验 L3「方向通路 vs 幅度通路」在信息层面是否分层。

用法：
  python v3_probe_info_death.py --model simple_cnn --dataset cifar100
输出：
  outputs_cifar100/v3_probe_info_death/probe_results.csv
  outputs_cifar100/v3_probe_info_death/probe_summary.md（终端同步打印）
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils.nonlinearity import register_nonlinearity_hooks, remove_hooks, nonlinearity
from utils.perturbation import register_perturbation_hooks
from utils.paths import get_ckpt_root, get_num_classes, get_outputs_root

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def build_model(model_name: str, num_classes: int):
    """主干工厂（与 readout 脚本一致；v4 增加 robust 版以便探测 NAT/Exp2 主干）"""
    if model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        return SimpleCNN(num_classes=num_classes)
    if model_name == "robust_cnn":
        from models.robust_cnn import RobustCNN
        return RobustCNN(num_classes=num_classes, use_calibration=True)
    if model_name == "vgg11":
        from models.vgg11 import VGG11
        return VGG11(num_classes=num_classes)
    if model_name == "resnet18":
        from models.resnet import ResNet18
        return ResNet18(num_classes=num_classes)
    raise ValueError(f"本探针支持 simple_cnn / robust_cnn / vgg11 / resnet18，收到 {model_name}")


def load_clean_weights(model, model_name, dataset, device, ckpt=None):
    path = ckpt or os.path.join(get_ckpt_root(dataset), model_name, "best_model.pth")
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd)
    model.to(device).eval()
    return path


def make_loaders(dataset, batch_size=256):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
    ])
    tr = datasets.CIFAR100(root="./data", train=True, download=False, transform=tf)
    te = datasets.CIFAR100(root="./data", train=False, download=False, transform=tf)
    return (DataLoader(tr, batch_size=batch_size, shuffle=False, num_workers=2),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=2))


def single_layer_hook(model, layer_name, alpha):
    """只在指定层注入非线性（用于逐层归因）"""
    module = dict(model.named_modules())[layer_name]

    def pre_hook(m, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        return (nonlinearity(x, alpha),) + tuple(inputs[1:])

    return [module.register_forward_pre_hook(pre_hook)]


@torch.no_grad()
def extract(model, loader, device, feat_store, limit=None):
    """提取 fc 输入特征 + 冻结头预测 + 标签"""
    feats, labels, preds = [], [], []
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        feats.append(feat_store["f"].detach().cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
        n += x.size(0)
        if limit and n >= limit:
            break
    return (np.concatenate(feats)[:limit], np.concatenate(labels)[:limit],
            np.concatenate(preds)[:limit])


def run_condition(model, loader, device, cond_name, hooks):
    feat_store = {}
    h = model.fc.register_forward_pre_hook(
        lambda m, inp: feat_store.__setitem__("f", inp[0])
    )
    try:
        X, y, pred = extract(model, loader, device, feat_store)
    finally:
        h.remove()
        if hooks:
            remove_hooks(hooks)
    return cond_name, X, y, pred


def probe_accuracy(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xte), yte)) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="simple_cnn")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt", default=None, help="主干权重路径（默认 = clean 权重）")
    ap.add_argument("--backbone_tag", default="", help="主干标签（用于输出命名）")
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_test", type=int, default=10000)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = get_num_classes(args.dataset)
    model = build_model(args.model, num_classes)
    wpath = load_clean_weights(model, args.model, args.dataset, device, ckpt=args.ckpt)
    print(f"[probe] 模型={args.model} 权重={wpath} device={device}")

    train_loader, test_loader = make_loaders(args.dataset)
    out_dir = os.path.join(get_outputs_root(args.dataset), "v3_probe_info_death")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 条件表：(名称, 注册 hooks 的工厂) ----
    conditions = [("clean", lambda: [])]
    for a in (0.1, 0.2, 0.3, 0.4, 0.5):
        conditions.append((f"all|alpha={a:+.1f}", lambda a=a: register_nonlinearity_hooks(model, a)))
        if a <= 0.3:
            conditions.append((f"all|alpha={-a:+.1f}", lambda a=a: register_nonlinearity_hooks(model, -a)))
    for layer in ("conv1", "conv2", "conv3", "conv4", "fc"):
        conditions.append((f"{layer}|alpha=+0.3", lambda l=layer: single_layer_hook(model, l, 0.3)))
    for s in (0.1, 0.2, 0.3):
        conditions.append((f"all|sigma={s:.1f}", lambda s=s: register_perturbation_hooks(model, "gaussian", noise_std=s)))

    rows = []
    clean_Xtr = clean_ytr = None
    clean_Xte = clean_yte = None
    for name, hook_factory in conditions:
        # 训练侧与测试侧**各自独立注册同一条件**的 hooks（提取完即移除，避免状态串扰）
        _, Xtr, ytr, _ = run_condition(model, train_loader, device, name, hook_factory())
        _, Xte, yte, pred = run_condition(model, test_loader, device, name, hook_factory())
        Xtr, ytr = Xtr[:args.n_train], ytr[:args.n_train]
        Xte, yte, pred = Xte[:args.n_test], yte[:args.n_test], pred[:args.n_test]
        if name == "clean":
            clean_Xtr, clean_ytr, clean_Xte, clean_yte = Xtr, ytr, Xte, yte

        # 探针 A（主判据）：同条件训练 → 同条件测试 = 「失真表征中标签信息的可得性」
        pA = probe_accuracy(Xtr, ytr, Xte, yte)
        # 探针 B（次判据）：干净特征训练 → 失真特征测试 = 「表征对齐/迁移」
        pB = probe_accuracy(clean_Xtr, clean_ytr, Xte, yte)
        h_acc = (pred == yte).mean() * 100.0
        rows.append({
            "condition": name,
            "probe_same_cond": round(pA, 2),
            "probe_clean_train": round(pB, 2),
            "head_acc": round(h_acc, 2),
            "headroom_same": round(pA - h_acc, 2),
        })
        print(f"[probe] {name:<18} 同条件探针 {pA:6.2f}% | 干净训练探针 {pB:6.2f}% | "
              f"冻结头 {h_acc:6.2f}% | headroom {pA-h_acc:+7.2f}")

    bb = args.backbone_tag or args.model
    csv_path = os.path.join(out_dir, f"probe_results_{bb}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["condition", "probe_same_cond", "probe_clean_train", "head_acc", "headroom_same"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[probe] 已保存: {csv_path}")


if __name__ == "__main__":
    main()
