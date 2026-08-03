"""静态检查器自身 smoke 测试 — 复杂分析器需防回归 (评审建议).

check_dead_code.py / check_imports_deep.py 是自制静态分析器,
复杂度高但此前没有自身测试. 本文件验证: 运行不崩溃 + 输出格式正常.
(不断言"0 候选" — 未来出现真死代码时检查器报出是正确行为.)
"""
import os
import subprocess
import sys

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


def test_no_backslash_in_fstrings():
    """f-string 字面量内禁止反斜杠 — Python < 3.12 编译失败 (PEP 701)."""
    import tokenize
    bad = []
    for root in ("fem2d", "scripts", "tests"):
        base = os.path.join(PROJECT_ROOT, root)
        for dirpath, _, files in os.walk(base):
            if "__pycache__" in dirpath:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                p = os.path.join(dirpath, f)
                with open(p, "rb") as fh:
                    for tok in tokenize.tokenize(fh.readline):
                        if (tok.type == tokenize.STRING
                                and tok.string[:1].lower() == "f"
                                and "\\" in tok.string):
                            bad.append(f"{p}:{tok.start[0]}")
    for f in ("run.py", "run_demo.py"):
        p = os.path.join(PROJECT_ROOT, f)
        with open(p, "rb") as fh:
            for tok in tokenize.tokenize(fh.readline):
                if (tok.type == tokenize.STRING
                        and tok.string[:1].lower() == "f"
                        and "\\" in tok.string):
                    bad.append(f"{p}:{tok.start[0]}")
    assert not bad, f"f-string 内反斜杠 (3.12- 编译失败): {bad}"
