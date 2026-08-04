""".geo 危险指令拦截测试 (审查修复包第 5 项).

旧清洗逻辑只剥离 Save/Mesh 等少数指令 — Gmsh 支持 SystemCall 执行任意
系统命令, 第三方 .geo 即 RCE 面。判别性: 放回旧实现 (无黑名单) 必须失败。
"""
import contextlib
import io

import pytest

from fem2d.errors import GeoScriptRejected
from scripts.gmsh_runner import (
    _geometry_without_explicit_save, sanitize_geo_source)


def test_systemcall_rejected_with_clear_error():
    """含 SystemCall 的 .geo → 拒绝, 报错指明指令与行号."""
    source = 'lc = 0.1;\nSystemCall "calc.exe";\n'
    with pytest.raises(GeoScriptRejected, match="SystemCall"):
        sanitize_geo_source(source)


def test_systemcall_case_and_whitespace_variants():
    """大小写/缩进变体同样拦截 (Gmsh 关键字不区分大小写)."""
    for source in ('  systemcall "x";\n', 'SYSTEMCALL"x";\n',
                   '\tSystemCall "x";\n'):
        with pytest.raises(GeoScriptRejected):
            sanitize_geo_source(source)


def test_commented_systemcall_not_rejected():
    """注释里的 SystemCall 不触发 — Gmsh 解析器忽略注释 (避免误拒)."""
    source = '// SystemCall "x" is dangerous\nPoint(1) = {0,0,0,1};\n'
    assert sanitize_geo_source(source) == source


def test_normal_geo_behavior_unchanged():
    """无危险指令的 .geo 行为零变化 — 原有剥离规则照旧."""
    source = "Mesh 2;\nMesh.Format = 39;\nPoint(1) = {0,0,0,1};\n"
    sanitized = sanitize_geo_source(source)
    assert "removed" in sanitized  # 原有 Save/Mesh 剥离仍生效
    assert "Point(1) = {0,0,0,1};" in sanitized
    assert "SystemCall" not in sanitized


def test_subprocess_path_rejects_systemcall(tmp_path):
    """子进程路径 (_geometry_without_explicit_save) 同样拒绝."""
    geo = tmp_path / "evil.geo"
    geo.write_text('SystemCall "whoami";\n', encoding="utf-8")
    with pytest.raises(GeoScriptRejected):
        _geometry_without_explicit_save(str(geo))


def test_cli_geo_with_systemcall_exits_one(monkeypatch, tmp_path):
    """CLI .geo 路径含 SystemCall → 退出码 1 (用户错误; 曾泄漏 2)."""
    import fem2d.input_source as input_source
    from fem2d.runner import main

    geo = tmp_path / "evil.geo"
    geo.write_text('SystemCall "whoami";\n', encoding="utf-8")
    monkeypatch.setattr(
        input_source, "generate_geo_with_topology",
        lambda *a, **k: (_ for _ in ()).throw(
            GeoScriptRejected(
                '.geo 第 1 行含被禁止的 SystemCall 指令 — 只应运行'
                "自己编写的文件")))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            code = main([str(geo), "--no-plot"])
        except SystemExit as exc:  # 迁移前形态 — 归一为退出码
            code = exc.code if isinstance(exc.code, int) else 1
    assert code == 1
    assert "SystemCall" in buf.getvalue()
