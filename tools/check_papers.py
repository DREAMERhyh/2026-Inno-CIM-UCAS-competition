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
PAPERS = ["journal", "technical-report", "competition", "advisor-summary"]
TEXBIN = "/d/texlive/2025/bin/windows"
# 前三份稿的 geometry 都是 left=2.4cm,right=2.4cm ⇒ 右边距 2.4cm = 68.29pt。
# 排版溢出超过这个值的部分**会被裁到纸外**（不是"不好看"，是内容丢失）。
# ⚠ advisor-summary 的边距是 1.9cm（右边距 53.9pt），比这个阈值更紧；
#    它当前 0 处溢出，故沿用同一个保守阈值（用更宽的阈值只会漏报，不会误报）。
RIGHT_MARGIN_PT = 68.2


def sh(cmd, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, shell=True,
                          capture_output=True, text=True, errors="replace")


def _is_cjk(ch):
    """是不是中日韩字符 / 全角标点。

    用途：把「中文后面的漏写 `%`」与「ASCII 的吃换行惯用法」分开。
    2026-09-26 复核用对抗例实测：不区分时 `\\foo%` / `{%` / `$x$%` **全部误报**。
    """
    return ("㐀" <= ch <= "鿿"      # CJK 扩展 A + 基本区
            or "　" <= ch <= "〿"    # CJK 标点
            or "＀" <= ch <= "￯")   # 全角形式


def _braced_inner(s):
    """若 s 以 `}` 结尾，返回与它配对的那个 `{` 之间的内容；配不上则 None。"""
    if not s.endswith("}"):
        return None
    depth = 0
    for k in range(len(s) - 1, -1, -1):
        if s[k] == "}":
            depth += 1
        elif s[k] == "{":
            depth -= 1
            if depth == 0:
                return s[k + 1:-1]
    return None


def _is_numeric(s):
    """花括号内容是不是【纯数字】（允许小数点、千分位空格与 LaTeX 细空格 `\\,`）。

    用途：区分 `\\textbf{20.26}%`（漏写 `\\%`）与
    `\\ref{sec:v7}%` / `\\cite{fragility2026}%`（都是吃换行的惯用法）。
    后两者以数字**结尾**但不是纯数字 —— 只看"前一个字符是数字"会两者全报。
    """
    return bool(re.fullmatch(r"[0-9][0-9.,\\ ]*", s.strip()))


def check_source(tex_path):
    """静态检查：markdown 残留 + 裸 % + texttt 未转义下划线。"""
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
        # 裸 %（本该写 \%）。
        # 三条同时满足才报：① 不在行首（行首 % 是正常注释）② 紧邻前面**没有空白**
        # （正常注释习惯上会留空格）③ 后面到行尾**只有空白**（说明它不是注释，
        # 而是一个被静默吃掉的百分号）。
        # 危害：行尾的 % 会把换行"吃掉"——**不报错、不影响退出码**，
        # 但若本意是百分号，成品里就少一个 %。
        # 2026-09-26 外部体检查出 3 处，其中 TR 的 "headroom \textbf{20.26}%"
        # 成品里只剩 "20.26"（用 -bbox 坐标核过，那个位置没有任何 % 字形）。
        body = re.sub(r"\\%", "", line)          # 先去掉合法的转义 \%
        # 形态一：**行尾、紧贴前一个字符、且那个字符是中日韩字符**。
        # 收得这么紧是因为：`\foo%` / `{%` / `$x$%` 都是 **LaTeX 吃换行的惯用法**，
        # 一刀切会全误报（2026-09-26 复核时用对抗例实测到 5 处误报，故收紧）。
        # 中文稿里真正的漏写几乎都长成 `……个百分点%` 这个样子。
        for m in re.finditer(r"%(?=\s*$)", body):
            head = body[:m.start()]
            if (head and _is_cjk(head[-1])
                    and not head.lstrip().startswith("%")):
                problems.append(f"    行 {i}: 裸 `%`（行尾、紧贴中文字符）——"
                                f"会被 LaTeX 当成注释符；若本意是百分号，成品里会少一个 `%`。"
                                f"应写 `\\%`")
                break
        # 形态三：**行尾、`%` 前有空白、再前面是数字** —— 如 `headroom 20.26 %`。
        # 只吞换行、危害最小，但它同样可能是漏写的 `\%`（成品里会少一个 %）。
        # 判据收得很紧（必须"空白 + 数字"），故 `\clearpage  % 注释` 这类不会被误伤
        # ——后者 `%` 后面还有正文，本来就不匹配 `%\s*$`。
        for m in re.finditer(r"%(?=\s*$)", body):
            head = body[:m.start()]
            stripped = head.rstrip()
            if (stripped and stripped[-1].isdigit() and len(stripped) < len(head)
                    and not head.lstrip().startswith("%")):
                problems.append(f"    行 {i}: 裸 `%`（行尾、前面是空格、再前面是数字）——"
                                f"若本意是百分号，成品里会少一个 `%`。应写 `\\%`")
                break
        # 下面那条覆盖**行中**形态；它比行尾那种更危险——会吞掉**后面的正文**
        # （行尾那种只吞换行）。2026-09-26 复核时由顾问指出漏了这一形态，总管复验成立。
        # 判据取"紧邻前面是数字；或前面是 `}` 而**花括号里是纯数字**"：
        #   · `精度 20.26% 个百分点，见下文`    ⇒ 报（后面那句会消失）
        #   · `\textbf{20.26}%`                ⇒ 报（花括号里是 20.26）
        #   · `\ref{sec:v7}%`                  ⇒ **不报** —— 它以数字**结尾**但不是纯数字
        #   · `\cite{fragility2026}%`          ⇒ **不报**（同理）
        #   · `\newcommand{\x}{abc}%`          ⇒ **不报**
        # ⇒ 2026-09-26 第二重复核：顾问用对抗例实测出 5 处误报（`\ref{sec:v7}%` 那条
        #    用的还是本项目真实存在的 label），正是靠"纯数字"这个切分修掉的。
        for m in re.finditer(r"%", body):
            head = body[:m.start()]
            if not head or head.lstrip().startswith("%"):
                continue
            prev = head[-1]
            if prev.isdigit():
                pass                             # `20.26%` ⇒ 是漏写
            elif prev == "}":
                # `\textbf{20.26}%` ⇒ 是漏写；`\cite{fragility2026}%` / `\ref{sec:v7}%`
                # ⇒ **不是**（它们是吃换行的惯用法）。切分靠"花括号里是不是纯数字"。
                inner = _braced_inner(head)      # head 本身就以 `}` 结尾
                if inner is None or not _is_numeric(inner):
                    continue
            else:
                continue
            problems.append(f"    行 {i}: 裸 `%` 紧跟在数字后面 —— "
                            f"会把这行**后面剩下的正文整段吞掉**（比行尾那种更危险）。"
                            f"应写 `\\%`")
            break
        # \texttt{} 里的**未转义下划线**。
        # ⚠ 对比记忆：**`\path{}`（url 宏包）天生吃下划线，不用转义**；
        # 只有 `\texttt{}` 不行——`_` 在 text mode 是数学下标，报 "Missing $ inserted."。
        # 2026-09-26 自曝：把 `\texttt{ridge\_R2}` 写成 `\texttt{ridge_R2}`，
        # 触发 **43 个连锁编译错误**（错误位置还会指到别处，像本项目的 `\cref` 那次）。
        for m in re.finditer(r"\\texttt\{([^}]*)\}", line):
            if re.search(r"(?<!\\)_", m.group(1)):
                problems.append(f"    行 {i}: `\\texttt{{}}` 里有未转义的下划线 —— "
                                f"`_` 在 text mode 会报 `Missing $ inserted.`（且引发连锁错误）。"
                                f"应写 `\\_`；或用 `\\path{{}}`（它不需要转义）")
                break
    return problems


def _cleanup_check_artifacts(d):
    """删掉 `-jobname=_check` 留下的临时产物 —— 让工具自己保证"跑完不留痕"。"""
    for ext in (".pdf", ".log", ".aux", ".out", ".toc", ".synctex.gz"):
        p = os.path.join(d, "_check" + ext)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def compile_paper(name, emit=False):
    """编译两遍，返回 (页数, 错误数, 未定义引用数, 缺字形数, 最大 overfull pt)。

    后两项是 2026-09-25 外部体检之后加的。它们**都不报编译错误、也不影响退出码**，
    但都会让成品 PDF 出问题：缺字形是字符**静默消失**，超宽是内容**被裁到纸外**。

    ⚠ **默认用 `-jobname=_check` 编到同目录**（2026-09-26 起）：
    这样检查器**不碰交付物** `paper/<name>/main.pdf`，跑完不留痕。
    此前是就地编 —— 每跑一次就把四个**被跟踪的** `main.pdf` 变成 `M`
    （实测只差 1~9 字节，纯 PDF 内部时间戳，内容没变），逼得人手工
    `git checkout -- paper/`。**而那是个哑雷**：它分不清"检查器弄脏的元数据"
    与"我真正要提交的新 PDF"，会把真实更改一起丢掉。

    **为什么是"同目录换输出名"而不是"拷进临时目录"**：同目录 ⇒
    `\\graphicspath{{../figures/}}` 与所有相对路径**照常工作**，
    不存在"静默少编进图"的风险；拷目录是在手工复刻一个本来就工作的环境，**复刻就会漏东西**。
    前提已验证：四份稿 `\\include` 全 0（它会写**固定名** aux，`-jobname` 管不住）。

    `emit=True` 时才真正产出交付物 PDF（对应命令行 `--emit`）。
    """
    d = os.path.join(ROOT, "paper", name)
    env = dict(os.environ)
    env["PATH"] = TEXBIN + os.pathsep + env.get("PATH", "")
    job = "main" if emit else "_check"
    for _ in range(2):
        sh(f"xelatex -jobname={job} -interaction=nonstopmode main.tex", cwd=d, env=env)
    log = os.path.join(d, f"{job}.log")
    if not os.path.exists(log):
        return None, -1, -1, -1, -1.0
    try:
        txt = open(log, encoding="utf-8", errors="replace").read()
        # ⚠ "pages" 必须写成 "pages?"：xelatex 对**单页**文档写的是 "(1 page"（单数）。
        # 漏掉这个 s 会让任何 1 页的稿子被误报成"编译失败"（2026-09-26 加 advisor-summary 时踩到）。
        pages = re.search(rf"Output written on {re.escape(job)}\.pdf \((\d+) pages?", txt)
        over = [float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", txt)]
        result = (int(pages.group(1)) if pages else None,
                  len(re.findall(r"^!", txt, re.M)),
                  txt.count("undefined"),
                  txt.count("Missing character"),
                  max(over) if over else 0.0)
    finally:
        if not emit:
            _cleanup_check_artifacts(d)
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="论文提交前检查。默认用 -jobname=_check 编到同目录，**不碰交付物 PDF**。",
    )
    ap.add_argument("--emit", action="store_true",
                    help="**真正产出交付物** paper/<name>/main.pdf。"
                         "只在你要提交新 PDF 时用；平时跑检查器不要加这个参数。")
    args = ap.parse_args()

    print("=" * 64)
    print("论文提交前检查" + ("（--emit：就地编，会更新交付物 PDF）" if args.emit else ""))
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
        pages, errs, undef, missing, overmax = compile_paper(name, emit=args.emit)
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
