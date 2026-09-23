# 格式检查清单

> ⚠ **目标载体未定。** 本清单是**通用中文期刊/会议**的检查项，
> **不是**任何具体期刊或会议的合规结论。
> `paper-form-writing` 模式的硬性要求是：格式结论必须现查目标期刊的 author guidelines，
> 查不到就明说查不到，**不得用记忆里的期刊规范填补空白**。
> 本机没有获得任何目标载体的 guidelines，因此本文件只列**与载体无关**的通用项，
> 以及**必须按具体载体替换**的占位项。
>
> 来源标注：每条标注出处。标 `[通用写作实践]` 的不得当作权威规则引用。

---

## A. 与载体无关的通用项（三份稿件均适用）

| # | 检查项 | 状态 | 证据 | 出处 |
|---|---|---|---|---|
| A1 | 摘要能独立读懂，不依赖正文 | ✅ | 三份摘要均含背景、方法、**带具体数值**的结果、结论 | `[通用写作实践]` |
| A2 | 标题有信息量，不空洞 | ✅ | 期刊版："逐层非线性失真下神经网络精度退化的读出瓶颈与双杠杆修复" | `[通用写作实践]` |
| A3 | 关键词 3–8 个 | ✅ | 三份均为 6 个 | `[通用写作实践]` |
| A4 | 每个图表在正文中都被引用 | ✅ | 体检脚本核对：三份稿件 0 处未引用图表 | `[通用写作实践]` |
| A5 | 图表能独立读懂（含来源、n、单双 seed） | ✅ | 图注四段式：在说什么 / 数据来源 / n 与 seed / 怎么读 | `[来源: Stanford WITS module 5.1]` |
| A6 | 一个图一个 claim | ✅ | `paper/figure-plan.md` 每条都有"claim"列 | `[来源: Stanford WITS module 5.1]` |
| A7 | 三线表（无竖线） | ✅ | 全部用 `booktabs` | `[通用写作实践]` |
| A8 | 数字可追溯到产物文件 | ✅ | 出图脚本不硬编码数字；图注给出文件路径 | `[来源: 本项目 §13.13 制度]` |
| A9 | 缩写首次出现时给出全称 | ✅ | CiM / NAT / MAC / ADC 等均已展开 | `[通用写作实践]` |
| A10 | 结论不超出证据 | ✅ | 分级词汇（初步 / 提示性 / 不可分辨 / 已知未知）逐字保留 | `[来源: 本项目措辞纪律]` |

---

## B. 必须按目标载体替换的项（**当前全部为占位**）

| # | 检查项 | 当前状态 | 需要做什么 |
|---|---|---|---|
| B1 | 文档类与版式 | 三份均用 `ctexart` + A4 + 2.4cm 页边距 | 若目标载体提供模板（如 LNCS、IEEEtran、中文期刊模板），替换 `\documentclass` 与 `geometry`。**注意：仓库里已有 `Springer_Lecture_Notes_in_Computer_Science/llncs.cls`，若目标是 LNCS 应改用它** |
| B2 | 页数上限 | 13 / 15 / 18 页 | 查目标载体规定，超限则压缩（优先级：删附录表 $>$ 合并图 $>$ 删讨论小节） |
| B3 | 摘要字数上限 | 未统计 | 中文期刊常见 200–300 字；**需现查** |
| B4 | 是否需要英文摘要 | 当前**无** | 按目标载体要求补 `\begin{abstract}` 的英文版 |
| B5 | 参考文献样式 | 手写 `thebibliography`，中文式（作者. 题名[C]//会议. 地点: 出版者, 年: 页） | 按目标载体改为对应样式（GB/T 7714 / IEEE / Springer 等）；若要求 `.bib`，需把 8 条转成 BibTeX 条目 |
| B6 | 作者与单位格式 | 姓名 + 单位 + 邮箱 | 按目标载体要求调整（是否要通讯作者标注、基金号、ORCID） |
| B7 | 图的分辨率与格式要求 | PDF 矢量 + PNG 300dpi | 多数期刊要求 300–600 dpi 或矢量；当前满足，但**需现查**是否有线宽/字号下限 |
| B8 | 图表编号样式 | "图 1 / 表 1" | 部分期刊要求 "Fig. 1 / Table 1" 或中文"图 1"；已有 `\captionsetup` 可一处改 |
| B9 | 是否需要 Highlights / Graphical Abstract | 无 | 部分期刊要求，需现查 |
| B10 | 投稿系统元数据 | 无 | 标题与摘要在部分系统中只接受 ASCII，非 ASCII 需转义 —— `[来源: arXiv 提交要求]` |

---

## C. 抓取目标期刊 guidelines 的标准流程（供后续使用）

本机 `WebFetch` 不可用（会被安全策略拦截）。用下面两条路：

**路线 A（首选，轻量）**：curl 抓静态网页
```bash
curl -sL --max-time 25 -A "Mozilla/5.0" "<guidelines-url>" \
  | sed 's/<script[^>]*>.*<\/script>//g; s/<[^>]*>/ /g' \
  | tr -s ' \n' ' \n' > guidelines_raw.txt
grep -iE "word limit|abstract|figure|reference style|title" guidelines_raw.txt | head -20
```

**路线 B（JS 渲染页面）**：
```bash
playwright-cli open --browser=msedge --persistent --headed "<url>"
playwright-cli snapshot
playwright-cli close
```
注意：本机只有 Edge，没有 Chrome。

**规则**：先 curl，不行再上浏览器；**抓到什么写什么，抓不到就说抓不到**。
把结果追加到本文件的 B 段表格里，并标注来源 URL 与抓取日期。

---

## D. 建议的替换路径

用户已说明的可能去向是**期刊/会议投稿**。两条主流路径：

1. **先 arXiv 预印本占位，再按期刊精修** —— 标准做法，不是退而求其次。
   arXiv 要求宽松（TeX/LaTeX 或 PDF、title 与 abstract 必填、元数据只接受 ASCII）。
2. **直接定目标期刊** —— 若如此，需先确定刊名，再走 §C 的流程。

**若目标确定为 Springer LNCS**：仓库里已有完整模板
（`Springer_Lecture_Notes_in_Computer_Science/` 下的 `llncs.cls`、`samplepaper.tex`、`splncs04.bst`），
可直接把三份稿件的正文迁入。注意 LNCS 是英文模板，中文需另配 `ctex`，属非标准用法。
