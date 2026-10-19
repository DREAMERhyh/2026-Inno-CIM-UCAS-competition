# archive/ —— 归档区（2026-10-19）

| 目录 | 是什么 | 注意 |
|---|---|---|
| `competition_2026/` | **真实的竞赛提交材料**（原始 PDF/DOCX/PPTX + 演示视频） | 视频与口述草稿被 `.gitignore` 忽略；其余 6 个文件在 git 中 |
| `paper_competition_rewrite/` | **赛后重写的竞赛版**，不是真实竞赛提交版 | `main.tex` 的 `\graphicspath` 指向 `paper/figures/`，归档后不能直接重编；当前 `main.pdf` 是冻结交付物，详见 `BUILD_NOTE.md` |

## 课程报告

- `ucas-cod-lab-report/`：课程报告，29 MB。2026-10-19 已移入归档区（目录整体 rename 被文件锁挡住，改为逐项移动内容后删除空目录）。
  - 若要重编：32 处 `\includegraphics{../outputs/...}` 需改为 `../../outputs/...`。
  - 该目录被根 `.gitignore` 的 `ucas-cod-lab-report/` 规则忽略，不进 git。
  - 逐项移动的唯一差异：原目录有一个零字节临时锁文件 `main.synctex(busy)`，无内容价值，未迁移。

## 历史引用

`docs/verify/*`、`docs/INDEPENDENT_REVIEW.md` 等历史审计文档里的 `paper/competition/`、`论文v3.md:NNN`、`第五阶段报告.md:NNN` 是**当时的快照**。本次整理没有批量改这些历史记录；路径变了，但行号仍对应原文件。需要查原件时，到 `archive/`、`docs/reports/` 或根目录找。
