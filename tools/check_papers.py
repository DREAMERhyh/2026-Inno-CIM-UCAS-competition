"""
三份论文的提交前检查器。

**为什么存在**：本项目在 LaTeX 源里混写 markdown 语法踩过三次（均由 claude-e9 犯），
且其中两次**不报编译错误、静默渲染成错误的东西**——
- 反引号：LaTeX 里是开引号，成对出现时不报错，会把内容渲染成引号
- `**加粗**`：`*` 在文本模式只是普通字符，会渲染成字面的星号

手动检查记不住，所以做成脚本。**提交 paper/ 之前必须跑它，退出码非 0 就不要提交。**

2026-09-25 外部体检后**加了两条判据**，原因是这两类问题当时让所有闸门全绿却成品有问题：
- **缺字形**（`Missing character`）：字体里没有的字符被 xelatex **静默丢掉**，成品里一个不剩；
- **排版溢出 > 右边距**：超出的内容被**裁到纸外**（不是"不好看"，是信息丢失）。

用法：
    PYTHONIOENCODING=utf-8 <python> tools/check_papers.py

退出码：0 = 全部通过；1 = 有问题。
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from static_tex_check import run as static_run  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPERS = ["journal", "technical-report", "competition"]
TEXBIN = "/d/texlive/2025/bin/windows"
# 三份稿的 geometry 都是 left=2.4cm,right=2.4cm ⇒ 右边距 2.4cm = 68.29pt。
# 排版溢出超过这个值的部分**会被裁到纸外**（不是"不好看"，是内容丢失）。
RIGHT_MARGIN_PT = 68.2


def sh(cmd, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, shell=True,
                          capture_output=True, text=True, errors="replace")


def check_source(tex_path):
    """静态检查：markdown 残留。"""
    problems = []
    src = open(tex_path, encoding="utf-8").read()
    # 逐行，跳过注释行
    for i, line in enumerate(src.split("\n"), 1):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue
        if "**" in line:
            problems.append(f"    行 {i}: markdown 粗体 `**` —— 会静默渲染成字面星号")
        if "`" in line:
            problems.append(f"    行 {i}: 反引号 —— 会渲染成引号（成对时不报错）")
        # markdown 引用块：行首的 "> " —— LaTeX 里会渲染成字面的 > 加空格
        # （用 \begin{quote} 才是 LaTeX 的引用块）
        if stripped.startswith(">"):
            problems.append(f"    行 {i}: markdown 引用块 `>` —— 会渲染成字面的大于号，"
                            f"应改用 quote 环境")
    return problems


def compile_paper(name):
    """编译两遍，返回 (页数, 错误数, 未定义引用数, 缺字形数, 最大 overfull pt)。

    后两项是 2026-09-25 外部体检之后加的。它们**都不报编译错误、也不影响退出码**，
    但都会让成品 PDF 出问题：缺字形是字符**静默消失**，超宽是内容**被裁到纸外**。
    """
    d = os.path.join(ROOT, "paper", name)
    env = dict(os.environ)
    env["PATH"] = TEXBIN + os.pathsep + env.get("PATH", "")
    for _ in range(2):
        sh("xelatex -interaction=nonstopmode main.tex", cwd=d, env=env)
    log = os.path.join(d, "main.log")
    if not os.path.exists(log):
        return None, -1, -1, -1, -1.0
    txt = open(log, encoding="utf-8", errors="replace").read()
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", txt)
    over = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", txt)]
    return (int(pages.group(1)) if pages else None,
            len(re.findall(r"^!", txt, re.M)),
            txt.count("undefined"),
            txt.count("Missing character"),
            max(over) if over else 0.0)


def main():
    print("=" * 64)
    print("论文提交前检查")
    print("=" * 64)
    bad = 0
    for name in PAPERS:
        tex = os.path.join(ROOT, "paper", name, "main.tex")
        if not os.path.exists(tex):
            print(f"\n[{name}] 缺 main.tex，跳过")
            continue
        print(f"\n[{name}]")
        probs = check_source(tex)
        # 扩展静态检查（不编译）：引用/引文/环境配平/花括号/行内 $ 闭合。
        # **放在编译之前**：像"多一个 $"这种错会让整份稿编译不过，
        # 而它本该在编译前就被拦下（2026-09-24 实际漏检过一次，见 tools/static_tex_check.py 的说明）
        serr, swarn = static_run(tex)
        n_serr = sum(len(m) for _, m in serr)
        n_swarn = sum(len(m) for _, m in swarn)
        if n_serr:
            print(f"  ✗ 静态检查（不编译）：{n_serr} 处")
            for cname, msgs in serr:
                for m in msgs:
                    print(f"    [{cname}] {m}")
            bad += 1
        else:
            print(f"  ✓ 静态检查（不编译）：引用 / 引文 / 环境配平 / 花括号 / 行内 $ 全部通过"
                  + (f"（{n_swarn} 条提示）" if n_swarn else ""))
        pages, errs, undef, missing, overmax = compile_paper(name)
        if probs:
            print(f"  ✗ 源码静态检查：{len(probs)} 处 markdown 残留")
            for p in probs:
                print(p)
            bad += 1
        else:
            print("  ✓ 源码静态检查：无 markdown 残留")
        if errs == 0 and undef == 0 and pages:
            print(f"  ✓ 编译：{pages} 页 / 0 错误 / 0 未定义引用")
        else:
            print(f"  ✗ 编译：{pages} 页 / {errs} 错误 / {undef} 未定义引用")
            bad += 1
        # 缺字形：xelatex 把字体里没有的字符**静默丢掉** —— 不报错、不影响退出码，
        # 但成品 PDF 里那些字符**一个都没有**。2026-09-25 外部体检查出 46 处
        # （圈码 ①-⑥ 与 ⇒ 全丢），把编号句和交叉引用整句打穿，而所有闸门当时全绿。
        if missing:
            print(f"  ✗ 缺字形：{missing} 处 —— 这些字符在成品 PDF 里**静默消失**"
                  f"（编译不报错；查 main.log 的 'Missing character'）")
            bad += 1
        else:
            print("  ✓ 缺字形：0 处")
        # 排版溢出：超过右边距的部分**会被裁到纸外**（内容丢失，不是"不好看"）。
        if overmax > RIGHT_MARGIN_PT:
            print(f"  ✗ 排版溢出：最大 {overmax:.1f}pt > {RIGHT_MARGIN_PT}pt（= 2.4cm 右边距）"
                  f" —— 内容会被裁到纸外")
            bad += 1
        elif overmax:
            print(f"  ✓ 排版溢出：最大 {overmax:.1f}pt ≤ {RIGHT_MARGIN_PT}pt（右边距之内）")
        else:
            print("  ✓ 排版溢出：无")
    print("\n" + "=" * 64)
    if bad:
        print(f"结果：{bad} 份有问题 —— **不要提交**")
        return 1
    print("结果：全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
