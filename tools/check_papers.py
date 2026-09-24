"""
三份论文的提交前检查器。

**为什么存在**：本项目在 LaTeX 源里混写 markdown 语法踩过三次（均由 claude-e9 犯），
且其中两次**不报编译错误、静默渲染成错误的东西**——
- 反引号：LaTeX 里是开引号，成对出现时不报错，会把内容渲染成引号
- `**加粗**`：`*` 在文本模式只是普通字符，会渲染成字面的星号

手动检查记不住，所以做成脚本。**提交 paper/ 之前必须跑它，退出码非 0 就不要提交。**

用法：
    PYTHONIOENCODING=utf-8 <python> tools/check_papers.py

退出码：0 = 全部通过；1 = 有问题。
"""
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPERS = ["journal", "technical-report", "competition"]
TEXBIN = "/d/texlive/2025/bin/windows"


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
    return problems


def compile_paper(name):
    """编译两遍，返回 (页数, 错误数, 未定义引用数)。"""
    d = os.path.join(ROOT, "paper", name)
    env = dict(os.environ)
    env["PATH"] = TEXBIN + os.pathsep + env.get("PATH", "")
    for _ in range(2):
        sh("xelatex -interaction=nonstopmode main.tex", cwd=d, env=env)
    log = os.path.join(d, "main.log")
    if not os.path.exists(log):
        return None, -1, -1
    txt = open(log, encoding="utf-8", errors="replace").read()
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", txt)
    return (int(pages.group(1)) if pages else None,
            len(re.findall(r"^!", txt, re.M)),
            txt.count("undefined"))


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
        pages, errs, undef = compile_paper(name)
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
    print("\n" + "=" * 64)
    if bad:
        print(f"结果：{bad} 份有问题 —— **不要提交**")
        return 1
    print("结果：全部通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
