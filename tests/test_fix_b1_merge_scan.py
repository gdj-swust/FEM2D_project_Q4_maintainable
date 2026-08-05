"""[R-β B1] Merge "*.geo" 纳入 SystemCall 递归扫描 (绕过回归锁).

Gmsh 的 ``Merge "file.geo"`` 与 Include 同为"解析并执行被引用 .geo 脚本"
的执行面 — 审查实证: main.geo 写 ``Merge "evil.geo";`` 时, evil.geo 内的
SystemCall 被 gmsh 真实调用 (日志输出 ``Info : Calling 'cmd /c echo ...'``)。
旧实现递归扫描只沿 Include 引用树展开 → 一行 Merge 击穿拦截。

判别性: 放回旧实现 (``_iter_geo_includes`` 只识别 Include), 本节所有
含 ``Merge`` 的用例必须红。

设计: 扫描而非剥离 — Merge "model.step" 等 CAD 导入是合法用法, 剥离会
破坏模型; 只有 .geo 目标会被当作脚本执行 (Gmsh 按扩展名分发, 一个 Merge
语句一个文件), 故只追 .geo 目标, 避免把 STEP 等文本 CAD 文件里的
"SystemCall" 字样误拒。
"""
import pytest

from fem2d.errors import GeoScriptRejected
from scripts.gmsh_runner import (
    _geometry_without_explicit_save, sanitize_geo_source, temp_copy_dir)


def _write_geo(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_merge_geo_child_systemcall_rejected(tmp_path):
    """顶层 .geo 用 Merge 引用 evil.geo → 其内 SystemCall 必须拦截.

    判别性: 旧实现 (只追 Include) 此用例通过 — 必须红."""
    _write_geo(tmp_path, "evil.geo", 'SystemCall "whoami";\n')
    main = _write_geo(tmp_path, "main.geo", 'Merge "evil.geo";\n')
    with pytest.raises(GeoScriptRejected,
                       match=r"evil\.geo.*SystemCall"):
        sanitize_geo_source("", geo_path=str(main))


def test_merge_subprocess_path_rejects_systemcall(tmp_path):
    """子进程路径 (_geometry_without_explicit_save) 同样拦截 Merge 目标 —
    双路径共用清洗器, 改一处双路生效."""
    _write_geo(tmp_path, "evil.geo", 'SystemCall "whoami";\n')
    main = _write_geo(tmp_path, "main.geo", 'Merge "evil.geo";\n')
    with pytest.raises(GeoScriptRejected, match="evil.geo"):
        _geometry_without_explicit_save(str(main))


def test_merge_relative_path_different_dir(tmp_path):
    """相对 Merge 基于被引文件所在目录解析 (子目录引用)."""
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_geo(sub, "part.geo", 'SystemCall "whoami";\n')
    main = _write_geo(tmp_path, "main.geo", 'Merge "sub/part.geo";\n')
    with pytest.raises(GeoScriptRejected, match="part.geo"):
        sanitize_geo_source("", geo_path=str(main))


def test_merge_absolute_path_scanned(tmp_path):
    """绝对路径 Merge (跨目录) 同样递归扫描."""
    other = tmp_path / "other"
    other.mkdir()
    part = _write_geo(other, "part.geo", 'SystemCall "whoami";\n')
    main = _write_geo(tmp_path, "main.geo",
                      f'Merge "{part.as_posix()}";\n')
    with pytest.raises(GeoScriptRejected, match="part.geo"):
        sanitize_geo_source("", geo_path=str(main))


@pytest.mark.parametrize("cycle, loop_re", [
    (("a.geo", 'Merge "b.geo";\n', "b.geo", 'Merge "a.geo";\n'),
     r"循环引用.*a\.geo → b\.geo → a\.geo"),
    (("a.geo", 'Include "b.geo";\n', "b.geo", 'Merge "a.geo";\n'),
     r"循环引用.*a\.geo → b\.geo → a\.geo"),
])
def test_merge_cycle_rejected_clearly(tmp_path, cycle, loop_re):
    """Merge 循环 / Include-Merge 混合循环 → 清晰拒绝, 不挂死."""
    for name, text in zip(cycle[::2], cycle[1::2]):
        _write_geo(tmp_path, name, text)
    with pytest.raises(GeoScriptRejected, match=loop_re):
        sanitize_geo_source("", geo_path=str(tmp_path / "a.geo"))


def test_merge_self_reference_rejected(tmp_path):
    """自引用 Merge (a → a) 同样视为循环."""
    _write_geo(tmp_path, "a.geo", 'Merge "a.geo";\n')
    with pytest.raises(GeoScriptRejected,
                       match=r"循环引用.*a\.geo → a\.geo"):
        sanitize_geo_source("", geo_path=str(tmp_path / "a.geo"))


def test_merge_diamond_no_false_cycle(tmp_path):
    """钻石形共享 Merge (b→d, c→d) 不是循环 — 不误拒."""
    _write_geo(tmp_path, "d.geo", "lc = 1;\n")
    _write_geo(tmp_path, "b.geo", 'Merge "d.geo";\n')
    _write_geo(tmp_path, "c.geo", 'Merge "d.geo";\n')
    _write_geo(tmp_path, "a.geo",
               'Include "b.geo";\nMerge "c.geo";\n')
    assert sanitize_geo_source("", geo_path=str(tmp_path / "a.geo")) == ""


def test_merge_diamond_child_systemcall_still_caught(tmp_path):
    """钻石形共享 Merge 中共享子文件含 SystemCall — 仍必须拒绝."""
    _write_geo(tmp_path, "d.geo", 'SystemCall "whoami";\n')
    _write_geo(tmp_path, "b.geo", 'Merge "d.geo";\n')
    _write_geo(tmp_path, "c.geo", 'Merge "d.geo";\n')
    _write_geo(tmp_path, "a.geo",
               'Include "b.geo";\nMerge "c.geo";\n')
    with pytest.raises(GeoScriptRejected, match="d.geo"):
        sanitize_geo_source("", geo_path=str(tmp_path / "a.geo"))


def test_merge_in_comment_not_scanned(tmp_path):
    """注释里的 Merge 不是指令 — 目标含 SystemCall 也不扫."""
    _write_geo(tmp_path, "evil.geo", 'SystemCall "whoami";\n')
    _write_geo(tmp_path, "main.geo",
               '// Merge "evil.geo";\nlc = 1;\n')
    assert sanitize_geo_source("", geo_path=str(tmp_path / "main.geo")) == ""


def test_merge_in_string_not_scanned(tmp_path):
    """字符串内容里的 Merge 字样不是指令."""
    _write_geo(tmp_path, "evil.geo", 'SystemCall "whoami";\n')
    _write_geo(tmp_path, "main.geo",
               'Print("see Merge evil.geo");\n')
    assert sanitize_geo_source("", geo_path=str(tmp_path / "main.geo")) == ""


def test_merge_non_geo_target_not_scanned(tmp_path):
    """Merge 非 .geo 目标 (CAD 导入) 不执行脚本 — 其内容含 SystemCall
    字样不得误拒 (扩展名分发与 Gmsh 一致)."""
    step = tmp_path / "model.step"
    step.write_text(
        "#1 = PRODUCT('SystemCall \"whoami\"', 'x', 'y');\n",
        encoding="utf-8")
    main = _write_geo(tmp_path, "main.geo", 'Merge "model.step";\n')
    assert sanitize_geo_source("", geo_path=str(main)) == ""


def test_merge_missing_file_not_blocked(tmp_path):
    """Merge 目标不存在 → 扫描跳过不报错 (Gmsh 解析时报错, 与缺失
    Include 行为一致)."""
    main = _write_geo(tmp_path, "main.geo", 'Merge "nope.geo";\n')
    assert sanitize_geo_source("", geo_path=str(main)) == ""


def test_merge_directive_kept_for_clean_child(tmp_path):
    """合法 Merge (目标无 SystemCall) 指令原样保留 — 扫描而非剥离."""
    _write_geo(tmp_path, "clean.geo", "lc = 1;\n")
    main = _write_geo(tmp_path, "main.geo", 'Merge "clean.geo";\n')
    sanitized = sanitize_geo_source('Merge "clean.geo";\n',
                                    geo_path=str(main))
    assert 'Merge "clean.geo";' in sanitized


def test_merge_chain_error_locates_file(tmp_path):
    """Merge 链 — 错误信息含完整链 (a → b → c) 定位违规文件."""
    _write_geo(tmp_path, "c.geo", 'SystemCall "whoami";\n')
    _write_geo(tmp_path, "b.geo", 'Merge "c.geo";\n')
    _write_geo(tmp_path, "a.geo", 'Merge "b.geo";\n')
    with pytest.raises(GeoScriptRejected,
                       match=r"a\.geo → b\.geo → c\.geo.*SystemCall"):
        sanitize_geo_source("", geo_path=str(tmp_path / "a.geo"))


def test_relative_merge_forces_source_dir(tmp_path, capsys):
    """含相对 Merge 的 .geo 临时副本必须留在源目录 (相对引用以所在目录
    解析) — 否则 Merge 目标断裂或指向副本目录里的未扫描同名文件."""
    geo = tmp_path / "m.geo"
    geo.write_text('Merge "sub/part.geo";\n', encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = temp_copy_dir(str(geo), str(out_dir))

    assert result == str(tmp_path)
    assert "Merge" in capsys.readouterr().out  # 必须 WARN 说明原因


def test_absolute_merge_allows_output_dir(tmp_path):
    """绝对 Merge (盘符/POSIX 路径) 不受影响 — 正常进入 --output-dir."""
    import os
    geo = tmp_path / "m.geo"
    geo.write_text(f'Merge "{os.path.abspath(tmp_path)}/part.geo";\n',
                   encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    assert temp_copy_dir(str(geo), str(out_dir)) == str(out_dir)
