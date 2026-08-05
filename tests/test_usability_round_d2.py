"""D2 判别性测试 — 输入入口引导 (心智负担收敛).

判别性 (回滚改动必须红):
  - run.py --help 尾部追加入口选择指南 (5 条入口各一行)
  - .inp 报错含 "入口" 引导关键词; .inp 内容是 .geo 脚本时给针对性引导
  - 未知扩展名报错含 "入口" 引导关键词
  - docs/input_entries.md 对照表存在且覆盖 5 条入口
"""
import contextlib
import io
import os

import pytest

from fem2d.errors import CliError
from fem2d.input_source import resolve_input_file


# ── run.py --help 入口指南 ──

def test_run_py_help_appends_entry_guide():
    """--help: argparse 帮助后追加 5 条入口指南, 退出码 0."""
    import run as run_entry

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        with pytest.raises(SystemExit) as exc_info:
            run_entry.print_help_with_entry_guide(["--help"])
    assert exc_info.value.code == 0
    text = out.getvalue()
    assert "入口选择指南" in text
    for keyword in ("run.py", "run_demo.py", "fem2d", ".spec", "gmsh"):
        assert keyword in text, f"--help 入口指南缺 {keyword}"


def test_run_py_help_short_flag():
    import run as run_entry

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        with pytest.raises(SystemExit) as exc_info:
            run_entry.print_help_with_entry_guide(["-h"])
    assert exc_info.value.code == 0
    assert "入口选择指南" in out.getvalue()


def test_run_py_no_help_returns_silently():
    """无 -h/--help: 不拦截, 正常进入 main."""
    import run as run_entry

    assert run_entry.print_help_with_entry_guide(["model.geo"]) is None


# ── input_source 报错入口引导 ──

def test_inp_error_contains_entry_guidance(tmp_path):
    """.inp 拒绝报错含 "入口" 引导关键词 (判别性核心)."""
    inp = tmp_path / "m.inp"
    inp.write_text("*NODE\n", encoding="ascii")
    with pytest.raises(CliError) as exc_info:
        resolve_input_file(str(inp), None)
    message = str(exc_info.value)
    assert ".inp" in message          # 既有文案保留
    assert "入口" in message          # 新增引导关键词
    assert "run.py" in message


def test_inp_with_geo_script_content_gives_targeted_hint(tmp_path):
    """.geo 被误当 .inp 传: 内容识别 → 提示改用 run.py 或 gmsh 路径."""
    inp = tmp_path / "model.inp"
    inp.write_text(
        'SetFactory("OpenCASCADE");\nRectangle(1) = {0, 0, 0, 3, 2};\n'
        "lc = 0.1;\n", encoding="utf-8")
    with pytest.raises(CliError) as exc_info:
        resolve_input_file(str(inp), None)
    message = str(exc_info.value)
    assert "入口" in message
    assert ".geo" in message
    assert "run.py" in message
    assert "gmsh" in message


def test_inp_plain_content_no_geo_hint(tmp_path):
    """非 .geo 内容的 .inp: 不给误导性脚本提示 (只给通用入口引导)."""
    inp = tmp_path / "m.inp"
    inp.write_text("*NODE\n1,0,0\n*ELEMENT\n", encoding="ascii")
    with pytest.raises(CliError) as exc_info:
        resolve_input_file(str(inp), None)
    message = str(exc_info.value)
    assert "入口" in message
    assert "Gmsh .geo 脚本" not in message


def test_unsupported_extension_contains_entry_guidance(tmp_path):
    """未知扩展名: 报错含 "入口" 引导关键词."""
    other = tmp_path / "m.dat"
    other.write_text("data\n", encoding="utf-8")
    with pytest.raises(CliError) as exc_info:
        resolve_input_file(str(other), None)
    message = str(exc_info.value)
    assert "不支持的输入" in message   # 既有文案保留
    assert "入口" in message
    assert "run.py" in message


# ── docs 对照表 ──

def test_docs_input_entries_table_covers_five_entries():
    """docs/input_entries.md 对照表覆盖 5 条入口."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc_path = os.path.join(repo_root, "docs", "input_entries.md")
    with open(doc_path, encoding="utf-8") as fh:
        text = fh.read()
    assert "输入入口对照表" in text
    for keyword in ("run.py", "run_demo.py", "fem2d", ".spec", "gmsh"):
        assert keyword in text, f"入口对照表缺 {keyword}"
