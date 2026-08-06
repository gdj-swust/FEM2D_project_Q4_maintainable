"""[R-β B2] Mesh.Format / Mesh.MshFileVersion 剥离正则去行首锚点.

旧剥离正则带 ``^\\s*`` 行首锚点 — 同行前导语句 (``x = 1; Mesh.Format =
39;``) 完全绕过; Save/Mesh 2 同用 ``\\b`` 无锚点 — 同类指令处理不一致。
绕过后果: MshFileVersion=2 保留 → MSH 2.2 输出 → 生成物标记注入失败 +
覆盖保护失效 (静默降级); Format=39 保留 → Abaqus 格式 → import 失败 (响亮)。

判别性: 退回 ``^\\s*`` 锚点版本, 本节同行写法用例必须红 (旧实现原样保留)。
"""
import os

import pytest

from scripts.gmsh_runner import _geometry_without_explicit_save, sanitize_geo_source


@pytest.mark.parametrize("source", [
    "Mesh.Format = 39;\n",             # 行首 (既有行为回归锁)
    "  Mesh.Format = 39;\n",           # 行首带缩进
    "\tMesh.Format = 39;\n",
    "x = 1; Mesh.Format = 39;",        # 同行前导语句 — 旧锚点绕过
    "Mesh.Format=39;",                 # 无空格
    "Mesh . Format = 39;",             # 点两侧空格
    "mesh.format = 39;",               # 大小写 (Gmsh 关键字不区分)
    "lc = 0.1; Mesh.Format = 1 + 2;",  # 表达式值
])
def test_mesh_format_variants_stripped(source):
    sanitized = sanitize_geo_source(source)
    assert "= 39;" not in sanitized
    assert "removed" in sanitized


@pytest.mark.parametrize("source", [
    "Mesh.MshFileVersion = 2;\n",      # 行首 (既有行为回归锁)
    'SetFactory("OpenCASCADE"); Mesh.MshFileVersion = 2;',  # 同行前导
    "x = 1; mesh.mshfileversion = 2;",  # 大小写变体
])
def test_msh_file_version_variants_stripped(source):
    sanitized = sanitize_geo_source(source)
    assert "= 2;" not in sanitized
    assert "removed" in sanitized


def test_mesh_format_in_string_not_corrupted():
    """字符串内含 'Mesh.Format =' 字样 → 不破坏脚本: 值部分排除引号,
    否则 'Mesh.Format =");' 会把闭合引号吞进 [^;]* 造成引号失衡."""
    source = 'Print("Mesh.Format =");\n'
    assert sanitize_geo_source(source) == source


def test_mesh_format_in_string_conservatively_stripped():
    """字符串内完整 'Mesh.Format = 39;' 字样 → 保守替换 (与 Save 同策略:
    脚本结构保持可解析, 打印内容改变可接受)."""
    sanitized = sanitize_geo_source('Print("Mesh.Format = 39;");\n')
    assert sanitized.count('"') % 2 == 0  # 引号仍配对 — 脚本可解析
    assert "removed" in sanitized


def test_subprocess_path_strips_same_line_format(tmp_path):
    """子进程路径 (临时副本) 同样剥离同行写法 — 双路径共用清洗器."""
    geo = tmp_path / "m.geo"
    geo.write_text('x = 1; Mesh.Format = 39;\n'
                   'Mesh.MshFileVersion = 2;\n', encoding="utf-8")
    copy_path, _cleanup = _geometry_without_explicit_save(str(geo))
    try:
        with open(copy_path, encoding="utf-8") as stream:
            text = stream.read()
    finally:
        os.unlink(copy_path)
    assert "= 39;" not in text
    assert "= 2;" not in text
