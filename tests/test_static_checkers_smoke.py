"""静态检查器自身 smoke 测试 — 复杂分析器需防回归 (评审建议).

check_dead_code.py / check_imports_deep.py 是自制静态分析器,
复杂度高但此前没有自身测试. 本文件验证: 运行不崩溃 + 输出格式正常.
(不断言"0 候选" — 未来出现真死代码时检查器报出是正确行为.)
"""
import os
import subprocess
import sys
import tempfile

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, timeout=120):
    return subprocess.run(
        [sys.executable] + args,
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout)


def test_dead_code_checker_runs_on_fem2d():
    r = _run(["scripts/check_dead_code.py"])
    assert r.returncode == 0, r.stderr
    assert "共 " in r.stdout and "候选" in r.stdout, r.stdout[-400:]
    assert "保护名单" in r.stdout, r.stdout[-400:]


def test_dead_code_checker_runs_on_scripts():
    r = _run(["scripts/check_dead_code.py", "scripts"])
    assert r.returncode == 0, r.stderr
    assert "共 " in r.stdout, r.stdout[-400:]


def test_imports_deep_checker_runs():
    r = _run(["scripts/check_imports_deep.py"])
    assert r.returncode == 0, r.stderr
    assert "import" in r.stdout.lower(), r.stdout[-400:]


def test_imports_deep_reports_broken_import_in_custom_dir():
    """check_imports_deep 必须真正分析命令行目录 — 断裂导入必须被报出.

    曾硬编码 ["fem2d","scripts"] 静默忽略传入目录 (评审发现测试写空);
    此测试用真实断裂场景验证脚本确实扫描了传入的临时目录.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, "a.py"), "w",
                  encoding="utf-8") as fh:
            fh.write("def foo():\n    pass\n")
        with open(os.path.join(folder, "b.py"), "w",
                  encoding="utf-8") as fh:
            fh.write("from a import bar\nprint(bar)\n")
        r = _run(["scripts/check_imports_deep.py", folder])
        assert r.returncode == 1, f"断裂导入未被报出: {r.stdout[-300:]}"
        assert "断裂" in r.stdout, r.stdout[-400:]


def test_imports_deep_does_not_crash_on_top_level_with():
    """check_imports_deep 曾因模块顶层 ast.With 无 orelse 属性崩溃 (评审).

    传入目录必须被分析 (输出含文件数), 且顶层 with 不崩溃.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        with open(os.path.join(folder, "w1.py"), "w",
                  encoding="utf-8") as fh:
            fh.write("with open('x.txt') as f:\n    data = f.read()\n")
        with open(os.path.join(folder, "__init__.py"), "w",
                  encoding="utf-8") as fh:
            fh.write("")
        r = _run(["scripts/check_imports_deep.py", folder])
        assert r.returncode == 0, r.stderr[-300:]
        assert "个文件" in r.stdout, (
            f"目录未被分析 (输出无文件数): {r.stdout[-300:]}")


def test_all_python_files_compile():
    """全项目 (含 tests/) ast 编译检查.

    默认 pytest 只收集 test_*.py / *_test.py — 非 test 前缀的脚本
    (如 tests/verify_plane.py) 若语法错误, "测试全绿" 而脚本一运行
    即 SyntaxError (曾因薄壳 f-string 转义写坏而实际发生).
    """
    import ast
    for root in ("fem2d", "scripts", "tests"):
        base = os.path.join(PROJECT_ROOT, root)
        for dirpath, _, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    ast.parse(fh.read(), filename=p)
    for f in ("run.py", "run_demo.py"):
        p = os.path.join(PROJECT_ROOT, f)
        with open(p, encoding="utf-8") as fh:
            ast.parse(fh.read(), filename=p)


def _fstring_expression_backslashes(path):
    """3.12+ tokenize 语义下, 返回 f-string 表达式段含反斜杠的行号.

    PEP 701 (3.12) 使 f-string 拆成 FSTRING_START/MIDDLE/END token —
    表达式段反斜杠 (3.11 为 SyntaxError) 只出现在表达式段的 STRING
    token 内; 字面文本 (FSTRING_MIDDLE) 的反斜杠 3.9-3.11 合法, 不报。
    <3.12 解释器无 FSTRING_* token, 返回 [] (由调用方走真语法检查)。
    """
    import io
    import tokenize
    if not hasattr(tokenize, "FSTRING_START"):
        return []
    bad = []
    with open(path, "rb") as fh:
        in_fstring = False
        expr_depth = 0
        for tok in tokenize.tokenize(fh.readline):
            if tok.type == tokenize.FSTRING_START:
                in_fstring = True
            elif tok.type == tokenize.FSTRING_END:
                in_fstring, expr_depth = False, 0
            elif in_fstring and tok.type == tokenize.OP:
                if tok.string == "{":
                    expr_depth += 1
                elif tok.string == "}":
                    expr_depth = max(0, expr_depth - 1)
            elif expr_depth and tok.type == tokenize.STRING:
                if "\\" in tok.string:
                    bad.append(tok.start[0])
    return bad


def _compile_311(path):
    """3.11 语义编译门: 返回违规行描述列表 (PEP 701 等 3.12+ 语法).

    >=3.12 解释器: ast.parse(feature_version=(3,11)) 抓 grammar 级差异
    (feature_version 不切换 tokenizer, PEP 701 f-string 反斜杠须用
    tokenize 表达式段检测补齐); <=3.11 解释器: 普通 ast.parse 即真
    3.11 语法, tokenize 检测自动跳过。
    """
    import ast
    with open(path, encoding="utf-8", errors="ignore") as fh:
        source = fh.read()
    try:
        ast.parse(source, filename=path, feature_version=(3, 11))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: {exc.msg}"]
    return [f"{path}:{line}: f-string 表达式内反斜杠 (PEP 701)"
            for line in _fstring_expression_backslashes(path)]


def test_no_backslash_in_fstrings():
    """全项目必须能在 3.11 语法下编译 (requires-python 声明 >=3.9).

    曾用 tokenize 扫 f-string 整 token — <3.12 把 f-string 当单一
    STRING token, 字面文本反斜杠 (3.9-3.11 合法, 全项目 42 处) 全被
    误报; >=3.12 拆成 FSTRING_* token 扫描完全失明。_compile_311
    组合 ast.parse(feature_version) + 表达式段 tokenize 检测, 版本无关,
    只拒真 3.12+ 语法。
    """
    bad = []
    for root in ("fem2d", "scripts", "tests"):
        base = os.path.join(PROJECT_ROOT, root)
        for dirpath, _, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                bad.extend(_compile_311(os.path.join(dirpath, f)))
    for f in ("run.py", "run_demo.py"):
        bad.extend(_compile_311(os.path.join(PROJECT_ROOT, f)))
    assert not bad, f"3.11 语法编译失败 (PEP 701 等 3.12+ 语法): {bad}"


def test_fstring_expression_backslash_rejected_under_311():
    """判别性: 真 PEP 701 用例 (表达式内反斜杠, 3.11 语法错误) 必须被
    语法门报出, 合法字面文本反斜杠不得误报 — 放回旧 tokenize 整 token
    扫描: 真用例在 >=3.12 上失明 (不报 → 本测试失败), 字面用例在
    <3.12 上误报 (主测试失败)."""
    with tempfile.TemporaryDirectory() as folder:
        pep701 = os.path.join(folder, "pep701.py")
        with open(pep701, "w", encoding="utf-8") as fh:
            fh.write("x = f\"{'\\n'}\"\n")     # 表达式内反斜杠 → 必须报
        assert _compile_311(pep701), "PEP 701 反斜杠未被语法门报出"

        literal = os.path.join(folder, "literal.py")
        with open(literal, "w", encoding="utf-8") as fh:
            fh.write("x = f'a\\nb'\n")         # 字面文本反斜杠 → 不得误报
        assert not _compile_311(literal), "合法字面文本反斜杠被误报"
