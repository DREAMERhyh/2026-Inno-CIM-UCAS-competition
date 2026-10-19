# 归档说明（2026-10-19）

这是**赛后重写的竞赛版**，不是真实竞赛提交版；真实材料在 `../competition_2026/`。

- 当前 `main.pdf` 是归档时的交付物（冻结）。
- `main.tex` 的 `\graphicspath{{../figures/}}` 指向 `paper/figures/`；归档后相对路径变成 `archive/figures`，重编会找不到图。
- 若要重编：把 `\graphicspath` 改为 `{{../../paper/figures/}}`，再编译。
- 本目录的 `*.aux`、`*.out`、`*.toc`、`*.synctex.gz` 已被根 `.gitignore` 忽略。
