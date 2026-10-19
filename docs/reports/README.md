# docs/reports/ —— 历史报告归档（2026-10-19）

本目录从仓库根目录移入：

- `deploy_notes.md`
- `第三阶段报告.md` / `第四阶段报告.md` / `第五阶段报告.md`
- `第三阶段优化方案.md`
- `二阶段优化+实施记录+结果分析.md`
- `原始题目+题目解读+研究思路+研究发现.md`

`论文v3.md` **仍在仓库根目录**：它是 `README.md` 的第一入口，且 `paper/technical-report/` 的交付 PDF 正文引用了它；移动会打断 89 处引用。

## 历史审计文档怎么处理

`docs/verify/*`、`docs/INDEPENDENT_REVIEW.md`、`docs/PROJECT_RETROSPECTIVE.md` 等文档里的 `论文v3.md:NNN`、`第五阶段报告.md:NNN`、`deploy_notes.md` 是**当时的快照**。本次整理没有批量改这些历史记录——路径变了，但行号仍对应原文件。需要查原件时：

- `论文v3.md`：仓库根目录
- 其余报告：本目录

## 可执行命令注意

`paper/coverage-audit.md` 的 grep 路径与 `docs/HANDOFF.md` 的 `tail` 路径已在 2026-10-19 更新为 `docs/reports/*_报告.md` 与 `logs/legacy/logs_v7_m4_readout.log`。
