# paper/ —— 论文与归档产物

> 本目录由两个论文写作工作流产出：`rewrite-scientific-writing`（骨架）与 `paper-form-writing`（皮肤）。
> 骨架重建走的是**模式 0（项目考古）**；形式走的是模式 1（结构）→ 2（摘要）→ 4（图表）→ 5（格式）→ 6（体检）。

---

## 一、三份稿件

| 目录 | 覆盖范围 | 定位 | 页数 | 状态 |
|---|---|---|---|---|
| [`competition/`](competition/) | **第一 + 第二阶段**（任务 1–3、拓展 1–3、评审三项优化） | 竞赛/课程提交 | **13** | ✅ 编译通过 |
| [`journal/`](journal/) | **全六阶段** | 期刊/会议投稿底稿 | **15** | ✅ 编译通过 |
| [`technical-report/`](technical-report/) | **全六阶段 + 研究历程 + 工程落地 + 复现手册** | 技术报告 / 内部归档 | **18** | ✅ 编译通过 |

三份的**科学结论是同一套**（同一 central claim），差别在**覆盖范围与保留的材料**：

- **竞赛版**以 CIFAR-10 为主线（第一阶段的实际口径），保留探索过程与消融细节。
- **期刊版**严格问题驱动，Results 的每个小节标题都写成 **Answer**（不写"我们做了什么"）；
  含一节方法学内容（方差纪律与跨实现不可移植）。
- **技术报告版**额外保留三类材料：**三次"先归因、后被自己否证"的完整链条**、
  **工程落地三档判定**、**可复现性手册与方法论制度**。它是唯一包含停车场与下一阶段入场券的版本。

### 编译

```bash
cd paper/<版本名>
xelatex -interaction=nonstopmode main.tex   # 连续跑两次以解析交叉引用
```

需要 XeLaTeX 与 `ctex`（中文支持）。本机工具链在 `/d/texlive/2025/bin/windows/`。

---

## 二、工作文件（骨架与形式）

| 文件 | 产出者 | 内容 |
|---|---|---|
| [`storyline.md`](storyline.md) | REWRITE 模式 0 | **Central Claim** + Question–Experiment–Finding 映射表（17 条）+ Results 大纲 + Discussion 骨架 + Answerability Check |
| [`results-outline.md`](results-outline.md) | REWRITE | 每个结果小节的 Question / Experiment / Data / Finding / 1-hop 五件套 |
| [`discussion-notes.md`](discussion-notes.md) | REWRITE | 2-hop 综合、可迁移的一般原理、向外展开的问题空间、六条局限与对应未来工作 |
| [`figure-plan.md`](figure-plan.md) | paper-form 模式 4 | 每张图的 claim / 数据来源 / 类型 / 三份稿件的用图分配 / 配色与规格 |
| [`health-check-report.md`](health-check-report.md) | paper-form 模式 6 | 全文体检：事实性缺失 / 形式层 / 论证层三层报告 + **未能核查的项** |
| [`format-checklist.md`](format-checklist.md) | paper-form 模式 5 | 格式检查清单（**目标载体未定，通用版**；含抓 guidelines 的标准流程） |

---

## 三、图表

```
figures/
├── gen_figures.py         # 期刊版 / 技术报告版：F01–F11
├── gen_figures_comp.py    # 竞赛版：C01–C08
├── F01…F11 .pdf / .png    # 11 张
└── C01…C08 .pdf / .png    # 8 张
```

**两条硬性纪律**：

1. **脚本内不硬编码任何精度数字** —— 全部从仓库 CSV 现读，末尾打印关键读数供交叉核对。
2. **零 GPU** —— 两个脚本都只读 CSV，不训练、不推理、不修改任何既有产物。

重新出图：

```bash
<python> paper/figures/gen_figures.py
<python> paper/figures/gen_figures_comp.py
```

（`<python>` 为本机的 conda `pytorch_env` 解释器。）

三份稿件共用的视觉规范：Wong 8 色（色盲友好）、中文 SimHei、正/负侧固定配色映射、
输出 PDF 矢量 + PNG 300 dpi。

---

## 四、数字的出处纪律

本目录下的任何数字都必须能指向下面三者之一：

1. `results_master.csv` 的一行（含 `pairing_check` 列的配对自检结果）；
2. `experiments_ledger.md` 的某个小节；
3. 某个 `outputs*/` 下的产物文件（图注中给出路径）。

**无法追溯的陈述一律不写入。** 文献引用同理：本目录只使用仓库里已核实的 8 篇，
未编造任何参考文献（缺口见 `health-check-report.md` §1.1）。

---

## 五、已知的两处待人工处理

1. **补参考文献**。三份稿件都只用了仓库既有的 8 篇；期刊版需要的
   "对抗鲁棒性 / 表征可解码性 / 读出适配"三条文献线目前为空，需作者补齐。
2. **确定目标载体后替换格式**。`format-checklist.md` 的 B 段全部是占位项；
   若目标为 Springer LNCS，仓库里已有 `llncs.cls` 全套模板可用。

详见 `health-check-report.md` 的"未能核查的项"。
