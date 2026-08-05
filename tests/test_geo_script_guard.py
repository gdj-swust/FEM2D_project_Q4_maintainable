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


# ─────────────────────────────────────────────────────────────────────
# 审查轮 6b — SystemCall 任意位置拦截 + Include 递归扫描
#
# 判别性说明: 下方"同行多语句 / 块注释后紧跟"绕过用例在旧行首正则
# (r"^\s*SystemCall\b", MULTILINE) 下必须通过 (旧实现不拦截, 审查实测
# 被接受) — 临时退回行首正则, 此节绕过用例必须红; 恢复新实现必须绿。
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("source", [
    'x = 1; SystemCall "whoami";',
    '/* comment */ SystemCall "whoami";',
    'SystemCall  "whoami";',   # 多空格 (行首正则已覆盖, 保留)
])
def test_systemcall_any_position_rejected(source):
    """行首正则可绕过用例 — 退回行首正则此用例必须红 (旧实现放行)."""
    with pytest.raises(GeoScriptRejected, match="SystemCall"):
        sanitize_geo_source(source)


def test_systemcall_line_comment_not_rejected():
    """// 注释内 SystemCall 不触发 (行为保持) — 先剥离注释再匹配."""
    source = '// SystemCall "whoami";\nx = 1;\n'
    assert sanitize_geo_source(source) == source


def test_systemcall_block_comment_not_rejected():
    """/* */ 块注释内 SystemCall 不触发."""
    source = '/* SystemCall "whoami"; */\nx = 1;\n'
    assert sanitize_geo_source(source) == source


def test_systemcall_in_string_conservatively_rejected():
    """字符串参数含 SystemCall 字样 → 保守拒绝 (安全优先: 无法区分
    "字符串参数"与真正调用, 误拒纯字符串场景可接受 — 见
    _mask_geo_comments 的"为什么")."""
    with pytest.raises(GeoScriptRejected, match="SystemCall"):
        sanitize_geo_source('echo "SystemCall";')


def test_systemcall_inside_string_slashslash_not_comment():
    """字符串内的 // 是普通字符不是注释 — SystemCall 仍命中不逃逸."""
    with pytest.raises(GeoScriptRejected, match="SystemCall"):
        sanitize_geo_source('Print("a // b SystemCall");')


def test_systemcall_line_number_across_comments():
    """行号定位: 前有行注释/字符串/跨行块注释时, SystemCall 行号仍准."""
    source = ('// header\n'
              'x = "s";\n'
              '/* block\n'
              '   comment */\n'
              'SystemCall "whoami";\n')
    with pytest.raises(GeoScriptRejected, match="第 5 行"):
        sanitize_geo_source(source)


# ── Include 递归扫描 ──

def _write_geo(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_include_child_systemcall_rejected(tmp_path):
    """Include 子文件含 SystemCall → 拒绝, 错误信息含引用链定位.

    退回旧实现 (无 Include 扫描) 此用例必须红."""
    _write_geo(tmp_path, "part.geo", 'SystemCall "whoami";\n')
    main = _write_geo(tmp_path, "main.geo", 'Include "part.geo";\n')
    with pytest.raises(GeoScriptRejected,
                       match=r"part\.geo.*第 1 行.*SystemCall"):
        sanitize_geo_source("", geo_path=str(main))


def test_include_relative_path_different_dir(tmp_path):
    """相对 Include 基于被引文件所在目录解析 (子目录引用)."""
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_geo(sub, "part.geo", 'SystemCall "whoami";\n')
    main = _write_geo(tmp_path, "main.geo", 'Include "sub/part.geo";\n')
    with pytest.raises(GeoScriptRejected, match="part.geo"):
        sanitize_geo_source("", geo_path=str(main))


def test_include_absolute_path_scanned(tmp_path):
    """绝对路径 Include (跨目录) 同样递归扫描."""
    other = tmp_path / "other"
    other.mkdir()
    part = _write_geo(other, "part.geo", 'SystemCall "whoami";\n')
    main = _write_geo(tmp_path, "main.geo",
                      f'Include "{part.as_posix()}";\n')
    with pytest.raises(GeoScriptRejected, match="part.geo"):
        sanitize_geo_source("", geo_path=str(main))


@pytest.mark.parametrize("chain_def, loop_re", [
    (("a.geo", 'Include "b.geo";\n', "b.geo", 'Include "a.geo";\n'),
     r"循环引用.*a\.geo → b\.geo → a\.geo"),
    (("a.geo", 'Include "b.geo";\n',
      "b.geo", 'Include "c.geo";\n',
      "c.geo", 'Include "a.geo";\n'),
     r"循环引用.*a\.geo → b\.geo → c\.geo → a\.geo"),
])
def test_include_cycle_rejected_clearly(tmp_path, chain_def, loop_re):
    """父子双向/三节点循环 include → 清晰拒绝, 不挂死."""
    for name, text in zip(chain_def[::2], chain_def[1::2]):
        _write_geo(tmp_path, name, text)
    with pytest.raises(GeoScriptRejected, match=loop_re):
        sanitize_geo_source("", geo_path=str(tmp_path / "a.geo"))


def test_include_self_reference_rejected(tmp_path):
    """自引用 Include (a → a) 同样视为循环."""
    _write_geo(tmp_path, "a.geo", 'Include "a.geo";\n')
    with pytest.raises(GeoScriptRejected, match=r"循环引用.*a\.geo → a\.geo"):
        sanitize_geo_source("", geo_path=str(tmp_path / "a.geo"))


def test_include_diamond_no_false_cycle(tmp_path):
    """钻石形共享引用 (b→d, c→d) 不是循环 — 不误拒."""
    _write_geo(tmp_path, "d.geo", "lc = 1;\n")
    _write_geo(tmp_path, "b.geo", 'Include "d.geo";\n')
    _write_geo(tmp_path, "c.geo", 'Include "d.geo";\n')
    _write_geo(tmp_path, "a.geo",
               'Include "b.geo";\nInclude "c.geo";\n')
    assert sanitize_geo_source("", geo_path=str(tmp_path / "a.geo")) == ""


def test_include_diamond_child_systemcall_still_caught(tmp_path):
    """钻石共享引用中共享子文件含 SystemCall — 仍必须拒绝."""
    _write_geo(tmp_path, "d.geo", 'SystemCall "whoami";\n')
    _write_geo(tmp_path, "b.geo", 'Include "d.geo";\n')
    _write_geo(tmp_path, "c.geo", 'Include "d.geo";\n')
    _write_geo(tmp_path, "a.geo",
               'Include "b.geo";\nInclude "c.geo";\n')
    with pytest.raises(GeoScriptRejected, match="d.geo"):
        sanitize_geo_source("", geo_path=str(tmp_path / "a.geo"))


def test_include_in_comment_not_scanned(tmp_path):
    """注释里的 Include 不是指令 — 目标存在且含 SystemCall 也不扫."""
    _write_geo(tmp_path, "evil.geo", 'SystemCall "whoami";\n')
    _write_geo(tmp_path, "main.geo",
               '// Include "evil.geo";\nlc = 1;\n')
    assert sanitize_geo_source("", geo_path=str(tmp_path / "main.geo")) == ""


def test_include_in_string_not_scanned(tmp_path):
    """字符串内容里的 Include 字样不是指令."""
    _write_geo(tmp_path, "evil.geo", 'SystemCall "whoami";\n')
    _write_geo(tmp_path, "main.geo",
               'Print("see Include evil.geo");\n')
    assert sanitize_geo_source("", geo_path=str(tmp_path / "main.geo")) == ""


def test_include_missing_file_not_blocked(tmp_path):
    """Include 目标不存在 → 扫描跳过不报错 (现有行为保持: Gmsh 解析时
    报错; test_output_dir_policy 的缺失-Include 用例依赖此行为)."""
    main = _write_geo(tmp_path, "main.geo", 'Include "nope.geo";\n')
    assert sanitize_geo_source("", geo_path=str(main)) == ""


def test_include_child_comment_systemcall_passes(tmp_path):
    """子文件注释内的 SystemCall 不触发 (注释剥离对 Include 子树同样生效)."""
    _write_geo(tmp_path, "part.geo", '// SystemCall "whoami";\nlc = 1;\n')
    _write_geo(tmp_path, "main.geo", 'Include "part.geo";\n')
    assert sanitize_geo_source("", geo_path=str(tmp_path / "main.geo")) == ""


def test_include_chain_error_locates_file(tmp_path):
    """三级 Include 链 — 错误信息含完整链 (a → b → c) 定位违规文件."""
    _write_geo(tmp_path, "c.geo", 'SystemCall "whoami";\n')
    _write_geo(tmp_path, "b.geo", 'Include "c.geo";\n')
    _write_geo(tmp_path, "a.geo", 'Include "b.geo";\n')
    with pytest.raises(GeoScriptRejected,
                       match=r"a\.geo → b\.geo → c\.geo.*SystemCall"):
        sanitize_geo_source("", geo_path=str(tmp_path / "a.geo"))


def test_subprocess_path_rejects_include_child_systemcall(tmp_path):
    """子进程路径 (_geometry_without_explicit_save) 同样递归扫描 Include."""
    _write_geo(tmp_path, "part.geo", 'SystemCall "whoami";\n')
    main = _write_geo(tmp_path, "main.geo", 'Include "part.geo";\n')
    with pytest.raises(GeoScriptRejected, match="part.geo"):
        _geometry_without_explicit_save(str(main))
