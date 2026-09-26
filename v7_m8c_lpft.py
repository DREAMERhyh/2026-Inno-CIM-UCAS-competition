"""M8c：LP-FT —— 先线性探针（LP）、再解冻主干微调（FT），在失真场景下是否优于纯读出适配？

## 问题

文献（Kumar et al., arXiv:2202.10054，摘要已逐字核实）说 LP-FT "combines the benefits of both"
（比全量微调 1% better ID / 10% better OOD）—— **但那是 OOD 分布偏移场景；失真场景下没人测过。**

而本项目现有结论是「训练端做的事越少越好」（L4）。LP-FT 说"两者结合更好"。
若 LP-FT 在失真下也更好，则「训练端无效」这句要**限定**为"直接训练无效，但以读出初始化后再微调有效"。

## 预登记判据（跑前写死，跑后不改）

LP-FT 在 **α=+0.3** 的精度，相对**纯读出适配**（同主干、同协议、同读出种子）：

| 分支 | 条件 | 判决 |
|---|---|---|
| ① | **≥ +3 pp** | **成立** —— 论文「训练端无效」的表述必须加限定；写台账 §16.x |
| ② | **< +3 pp** | **阴性** —— 记"失真场景下 LP-FT 不优于纯 LP"，并说明与 OOD 场景的差异 |

**3 pp 这个门槛的来历**：它是本项目自订的种子噪声分辨门槛（§14.x：<3 pp 的差值在方差内无法裁决）。

## 基线（纯读出适配 `e_readout_NAT`，α=+0.3，3 个读出种子均值）

SimpleCNN 43.31 / VGG-11 47.79 / ResNet-18 26.82    （论文 tab:nat）

## 协议（与既有读数严格对齐，否则不可比）

- 数据加载 / 失真注入 / 评估：**直接 import `v3_readout_repair`**，不重写（口径必须同源）
- α 训练分布：U[±0.3]（与 readout-NAT 一致）
- 读出侧固定 **seed 42**（与 M4 的读出适配同口径）
- **阶段 1（LP）**：24 ep，Adam lr=0.01 —— 与既有读出适配一致
- **阶段 2（FT）**：30 ep，Adam lr=1e-3 —— ⚠ **lr 与 ep 是执行者的选择、不是预登记原文**
  （LP-FT 的要点是"小 lr"；此处取从头训练用的 0.01 的 1/10）

## 产物

`outputs_cifar100/v7_m8c_lpft/lpft_c100_{backbone_tag}_alpha+0.30_n20000.csv`
三行：`a_original_fc`（原始头）/ `e_readout_NAT`（LP 阶段后）/ `f_lpft`（FT 阶段后）

用法：
  PYTHONIOENCODING=utf-8 <python> v7_m8c_lpft.py --arch robust_vgg11 \
    --ckpt checkpoints_cifar100/Exp2_Calib+Layerwise_vgg11/best_model.pth \
    --backbone_tag exp2_vgg11 --dataset cifar100
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.paths import get_outputs_root  # noqa: E402
from utils.nonlinearity import register_nonlinearity_hooks, remove_hooks  # noqa: E402
import v3_readout_repair as rr  # noqa: E402  —— 数据/评估/失真全部复用，保证口径同源

ALPHAS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]


def finetune_joint(model, head, train_loader, nc, device, epochs, lr,
                   alpha_max, seed, limit=None):
    """LP-FT 的第二阶段：**解冻主干**，与已初始化好的头联合微调。

    与阶段 1（冻结主干、只训头）的唯一区别是主干参与梯度更新。
    失真注入与阶段 1 完全一致（每个 batch 一个新的随机 α）。

    ⚠ 为什么需要这一步（LP-FT 的要点）：直接全量微调会让**随机初始化的头**
    在早期产生大梯度、把预训练特征破坏掉；先把头训好（LP）再解冻，特征得以保留。
    """
    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    rng = random.Random(seed + 1)   # 与 LP 阶段用不同的流，避免同一批 α 序列
    model.train()
    for ep in range(epochs):
        tot = cor = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            alpha = rng.uniform(-alpha_max, alpha_max)
            hooks = register_nonlinearity_hooks(model, alpha) if alpha != 0.0 else []
            # 抓读出层的输入特征（与 train_readout_nat 同一手法）
            store = {}
            h = rr.READOUT.register_forward_pre_hook(
                lambda m, inp: store.__setitem__("f", inp[0]))
            try:
                model(x)
                logits = head(store["f"])
                loss = crit(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
            finally:
                h.remove()
                if hooks:
                    remove_hooks(hooks)
            tot += y.size(0)
            cor += (logits.argmax(1) == y).sum().item()
            if limit and tot >= limit:
                break
        print(f"  [LP-FT] ep{ep+1}/{epochs} (α~U[±{alpha_max}], lr={lr:g}) 训练精度 {100*cor/tot:.2f}%")
    model.eval()
    head.eval()
    return head


def main() -> int:
    ap = argparse.ArgumentParser(description="M8c：LP-FT vs 纯读出适配")
    ap.add_argument("--arch", default="robust_vgg11",
                    choices=["simple_cnn", "vgg11", "resnet18", "simple_cnn_mp",
                             "robust_vgg11", "robust_resnet18", "robust_cnn"])
    ap.add_argument("--ckpt", required=True, help="主干权重路径")
    ap.add_argument("--backbone_tag", required=True, help="主干标签（用于产物命名与记录）")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--alpha", type=float, default=0.3, help="α 采样上限（U[±alpha]）")
    ap.add_argument("--lp_epochs", type=int, default=24, help="阶段 1（LP）轮数")
    ap.add_argument("--ft_epochs", type=int, default=30, help="阶段 2（FT）轮数")
    ap.add_argument("--ft_lr", type=float, default=1e-3, help="阶段 2 学习率")
    ap.add_argument("--n_train", type=int, default=20000)
    ap.add_argument("--ft_batch", type=int, default=128,
                    help="FT 阶段 batch size（LP/评估仍用 512）。"
                         "512 是为『冻结主干前向 / 从头训练』选的；做**全模型反传**时它让"
                         "ResNet-18 慢到 >13 min/epoch（实测，比从头训练慢 7 倍）。"
                         "改小是**修正配置**，不是调参找结果——lr 与 ep 保持预登记值。")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    nc = rr.get_num_classes(args.dataset)
    model = rr.build_backbone(args.arch, nc)
    print(f"[M8c] 主干架构={args.arch}")
    print(f"[M8c] 主干权重={args.ckpt}  (tag={args.backbone_tag})")
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    rr.READOUT = rr.get_readout_module(model)
    print(f"[M8c] 读出层: {type(rr.READOUT).__name__} (in_features={rr.READOUT.in_features})")

    train_loader, test_loader = rr.build_loaders(dataset=args.dataset)

    # ---- 阶段 1：LP（冻结主干，只训头）----
    print(f"[M8c] === 阶段 1/2：LP（冻结主干，{args.lp_epochs} ep）===")
    head = rr.train_readout_nat(model, train_loader, nc, device, epochs=args.lp_epochs,
                                limit=args.n_train, seed=args.seed)

    def acc_at(alpha: float, fn) -> float:
        """在给定 α 上用某个读出函数测精度（复用 v3 的评估路径）。"""
        return rr.eval_with_readout(model, test_loader, device, alpha, fn)

    original_fc = lambda f: torch.nn.functional.linear(f, rr.READOUT.weight, rr.READOUT.bias)  # noqa: E731

    rows = {}
    for name, fn in (("a_original_fc", original_fc), ("e_readout_NAT", lambda f: head(f))):
        rows[name] = [acc_at(a, fn) for a in ALPHAS]
        print(f"  {name:<18} " + " ".join(f"{v:>7.2f}" for v in rows[name]))

    # ---- 阶段 2：FT（解冻主干，联合微调）----
    print(f"[M8c] === 阶段 2/2：FT（解冻主干，{args.ft_epochs} ep, lr={args.ft_lr:g}, "
          f"batch={args.ft_batch}）===")
    ft_loader = train_loader
    if args.ft_batch != 512:
        ft_loader, _ = rr.build_loaders(batch_size=args.ft_batch, dataset=args.dataset)
    head = finetune_joint(model, head, ft_loader, nc, device,
                          epochs=args.ft_epochs, lr=args.ft_lr,
                          alpha_max=args.alpha, seed=args.seed, limit=args.n_train)
    rows["f_lpft"] = [acc_at(a, lambda f: head(f)) for a in ALPHAS]
    print(f"  {'f_lpft':<18} " + " ".join(f"{v:>7.2f}" for v in rows["f_lpft"]))

    # ---- 判决 ----
    i03 = ALPHAS.index(0.3)
    base = rows["e_readout_NAT"][i03]
    lpft = rows["f_lpft"][i03]
    delta = lpft - base
    print("\n" + "=" * 64)
    print(f"判决：α=+0.3 处  纯读出 {base:.2f}  →  LP-FT {lpft:.2f}   提升 {delta:+.2f} pp")
    print(f"判据：≥ +3.00 pp 为成立。本结果 = "
          f"{'① 成立（论文表述需加限定）' if delta >= 3.0 else '② 阴性（LP-FT 不优于纯 LP）'}")
    print("=" * 64)

    # ---- 产物 ----
    out_dir = os.path.join(get_outputs_root(args.dataset), "v7_m8c_lpft")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"lpft_c100_{args.backbone_tag}_alpha+0.30_n{args.n_train}.csv")
    if os.path.exists(path):
        print(f"!! 产物已存在，拒绝覆盖：{path}")
        return 1
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["readout"] + [f"{a:.1f}" for a in ALPHAS])
        for name, vals in rows.items():
            w.writerow([name] + [round(v, 2) for v in vals])
    print(f"产物：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
