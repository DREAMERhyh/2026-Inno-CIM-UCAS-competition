"""
.tex 源码静态检查（不编译）。

**为什么存在**：本项目踩过 12 次"markdown 写进 .tex"，其中 `**加粗**` 与反引号
**不报编译错误、静默渲染成字面字符**——只能靠扫描抓。
2026-09-24 又加一次教训：写者用手打的命令逐条扫，**两次之间漏掉了一条**，
于是"多一个 `$`"这种会**让整份稿编译不过**的错，直到别人编译才发现。

⇒ **把手打的检查清单固化成脚本**：清单不再靠记忆，跑法只有一条命令。

**检查项**（全部零内存开销，纯文本扫描）：
1. markdown 残留：`**` / 反引号 / 行首 `>`
2. 引用完整性：`\\ref` 指向的 label 是否存在；label 是否被引用（警告）
3. 引文完整性：`\\cite` 的键是否有 `\\bibitem`；bibitem 是否被引用（警告）
4. 环境配平：`\\begin{env}` 与 `\\end{env}` 按环境名逐一对齐
5. 行内 `$` 奇偶：同一行内未转义 `$` 的个数为奇数 ⇒ 数学模式没闭合
6. 花括号配平：整文件未转义 `{` 与 `}`

用法：
    PYTHONIOENCODING=utf-8 python tools/static_tex_check.py            # 查三份稿件
    PYTHONIOENCODING=utf-8 python tools/static_tex_check.py 路径.tex …  # 查指定文件

退出码：0 = 全部通过；1 = 有 ERROR（警告不影响退出码）。
"""
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPERS = ["journal", "technical-report", "competition"]


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def strip_comment(line: str) -> str:
    """去掉未转义的 % 之后的注释部分。"""
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def count_unescaped(text: str, ch: str) -> int:
    """数未被反斜杠转义的字符个数。"""
    n, i = 0, 0
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == ch:
            n += 1
        i += 1
    return n


# ----------------------------------------------------------------------
# 各项检查：返回 (errors, warnings)，元素为字符串
# ----------------------------------------------------------------------
def check_markdown(lines):
    errs = []
    for i, raw in enumerate(lines, 1):
        line = strip_comment(raw)
        if not line.strip():
            continue
        if "**" in line:
            errs.append(f"    行 {i}: markdown 粗体 `**` —— 会静默渲染成字面星号")
        if "`" in line:
            errs.append(f"    行 {i}: 反引号 —— 会渲染成引号（成对时不报错）")
        if line.lstrip().startswith(">"):
            errs.append(f"    行 {i}: markdown 引用块 `>` —— 会渲染成字面的大于号")
    return errs, []


def check_refs(src):
    labels = set(re.findall(r"\\label\{([^}]*)\}", src))
    refs = set(re.findall(r"\\(?:c|C)?ref\{([^}]*)\}", src))
    errs = [f"    \\ref 指向不存在的 label: {k}" for k in sorted(refs - labels)]
    warns = [f"    label 定义了但没被引用: {k}" for k in sorted(labels - refs)]
    return errs, warns


def check_cites(src):
    items = set(re.findall(r"\\bibitem\{([^}]*)\}", src))
    keys = set()
    for grp in re.findall(r"\\cite\{([^}]*)\}", src):
        keys.update(k.strip() for k in grp.split(",") if k.strip())
    errs = [f"    \\cite 的键没有对应 \\bibitem: {k}" for k in sorted(keys - items)]
    warns = [f"    bibitem 从未被引用: {k}" for k in sorted(items - keys)]
    return errs, warns


def check_envs(src):
    begins = re.findall(r"\\begin\{([^}]*)\}", src)
    ends = re.findall(r"\\end\{([^}]*)\}", src)
    errs = []
    for name in sorted(set(begins) | set(ends)):
        b, e = begins.count(name), ends.count(name)
        if b != e:
            errs.append(f"    环境 {name}: \\begin {b} 次 / \\end {e} 次")
    return errs, []


def check_dollar(lines):
    """同一行内未转义 $ 的个数为奇数 ⇒ 数学模式没闭合。"""
    errs = []
    for i, raw in enumerate(lines, 1):
        line = strip_comment(raw)
        if "\\[" in line or "\\]" in line or "$$" in line:
            continue  # 跨行公式，逐行奇偶不适用
        n = count_unescaped(line, "$")
        if n % 2:
            errs.append(f"    行 {i}: 未转义 $ 的个数为奇数（{n}）—— 数学模式没闭合\n"
                        f"        {raw.strip()[:100]}")
    return errs, []


def check_braces(src):
    n_open = count_unescaped(src, "{")
    n_close = count_unescaped(src, "}")
    if n_open != n_close:
        return [f"    未转义花括号不配平：{{ × {n_open} / }} × {n_close}"], []
    return [], []


CHECKS = [
    ("markdown 残留", check_markdown),
    ("引用完整性", check_refs),
    ("引文完整性", check_cites),
    ("环境配平", check_envs),
    ("行内 $ 闭合", check_dollar),
    ("花括号配平", check_braces),
]


# ----------------------------------------------------------------------
def run(path):
    """非打印 API（供 check_papers.py 复用）。

    返回 (errors, warnings)，各为 [(检查名, [消息, ...]), ...]。
    """
    src_raw = open(path, encoding="utf-8").read()
    lines = src_raw.split("\n")
    src = "\n".join(strip_comment(ln) for ln in lines)  # 检查一律跳过注释

    errors, warnings = [], []
    for name, fn in CHECKS:
        # 逐行检查收 lines，整文件检查收 src
        errs, warns = fn(lines) if fn in (check_markdown, check_dollar) else fn(src)
        if errs:
            errors.append((name, errs))
        if warns:
            warnings.append((name, warns))
    return errors, warnings


def check_file(path):
    errors, warnings = run(path)
    total_err = sum(len(msgs) for _, msgs in errors)
    total_warn = sum(len(msgs) for _, msgs in warnings)
    for name, errs in errors:
        print(f"  ✗ {name}：{len(errs)} 处")
        for e in errs:
            print(e)
    for name, warns in warnings:
        if not any(n == name for n, _ in errors):
            print(f"  ✓ {name}：通过（{len(warns)} 条提示）")
            for w in warns:
                print(w)
    for name, _ in CHECKS:
        if not any(n == name for n, _ in errors) and not any(n == name for n, _ in warnings):
            print(f"  ✓ {name}：通过")
    return total_err, total_warn


def main():
    try:                                   # GBK 控制台下也能直接跑
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if len(sys.argv) > 1:
        targets = sys.argv[1:]
    else:
        targets = [os.path.join(ROOT, "paper", p, "main.tex") for p in PAPERS]

    print("=" * 64)
    print(".tex 源码静态检查（不编译）")
    print("=" * 64)

    bad, warns = 0, 0
    for path in targets:
        name = os.path.basename(os.path.dirname(path)) or path
        if not os.path.exists(path):
            print(f"\n[{name}] 缺 {path}，跳过")
            continue
        print(f"\n[{name}]")
        e, w = check_file(path)
        bad += e
        warns += w
    print("\n" + "=" * 64)
    if bad:
        print(f"结果：{bad} 处 ERROR、{warns} 处提示 —— **先修再编译**")
        return 1
    print(f"结果：全部通过 ✓（{warns} 条提示，不影响退出码）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
