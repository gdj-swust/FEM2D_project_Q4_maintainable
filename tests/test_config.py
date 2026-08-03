"""AnalysisConfig 与配置链路的回归锁定 — 维护性承诺的测试兑现.

覆盖:
  1. AnalysisConfig 校验 (8 类非法值拒绝)
  2. to_dict/from_dict 序列化往返
  3. from_args (CLI → config, None 保持默认)
  4. .spec 通用字段映射 (resolve_spec_overrides)
  5. main(argv) 阶段化编排可重入 (返回退出码)
"""
import contextlib
import io
import os
import tempfile

import numpy as np
import pytest

from fem2d.cli import parse_args
from fem2d.config import AnalysisConfig
from fem2d.errors import CliError
from fem2d.input_source import resolve_spec_overrides

# ── 1. 校验 ──

@pytest.mark.parametrize("kwargs", [
    {"E": -1.0},
    {"E": 0.0},
    {"nu": 0.6},
    {"nu": -2.0},
    {"thickness": 0.0},
    {"lc": -0.1},
    {"plane": "3d"},
    {"linear_solver": "bogus"},
    {"error_method": "nope"},
    {"band_tag": "xx"},
    # 评审补齐: jump_ref 非有限/非正全拒绝 (曾漏测, 手工复现通过)
    {"jump_ref": float("nan")},
    {"jump_ref": float("inf")},
    {"jump_ref": 0.0},
    {"jump_ref": -1.0},
    # 评审补齐: band_step 超出带宽区间 (即使 --no-plot 也提前拒绝)
    {"band_min": 0.0, "band_max": 1.0, "band_step": 2.0},
    {"band_min": 0.0, "band_max": 1.0, "band_step": 0.0},
    {"band_min": 2.0, "band_max": 1.0, "band_step": 0.5},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        AnalysisConfig(**kwargs)


def test_negative_poisson_valid():
    """负泊松比 (auxetic) 是合法材料 — 校验范围必须允许."""
    config = AnalysisConfig(E=3e7, nu=-0.3)
    assert config.nu == -0.3


# ── 2. 序列化 ──

def test_config_roundtrip():
    config = AnalysisConfig(E=3e7, nu=0.3, thickness=1.0, plane="strain",
                             linear_solver="ilu", error_method="l2")
    data = config.to_dict()
    restored = AnalysisConfig.from_dict(data)
    assert restored.to_dict() == data


def test_from_dict_ignores_unknown_keys():
    config = AnalysisConfig(E=3e7)
    data = {**config.to_dict(), "future_key": 1}
    assert AnalysisConfig.from_dict(data).to_dict() == config.to_dict()


# ── 3. from_args ──

def test_from_args_none_keeps_defaults():
    ns = parse_args(["dummy.inp"])  # 未指定任何覆盖参数
    config = AnalysisConfig.from_args(ns)
    assert config.E == 2.10e11       # 程序默认
    assert config.nu == 0.3
    assert config.plane is None      # 待网格判型
    assert config.linear_solver == "auto"


def test_from_args_overrides():
    ns = parse_args(["dummy.inp", "--E", "3e7", "--plane", "strain",
                     "--linear-solver", "ilu"])
    config = AnalysisConfig.from_args(ns)
    assert config.E == 3e7
    assert config.plane == "strain"
    assert config.linear_solver == "ilu"


# ── 4. .spec 通用字段映射 ──

def _write_spec(text):
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".spec", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return handle.name


def test_spec_overrides_apply_and_cli_wins():
    # 真实存在的 mesh 文件 (绝对路径) — resolve_spec_overrides 校验存在性
    mesh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".inp", delete=False, encoding="utf-8")
    mesh.write("*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n"
               "*ELEMENT,TYPE=CPS4\n1,1,2,3,4\n")
    mesh.close()
    spec = _write_spec(
        f"mesh = {mesh.name}\nE = 3e7\nnu = 0.25\nt = 0.02\n"
        "plane = strain\nfix = left\ntraction = right:1e6,0\n"
        "body = 0,-78000\nno_plot = true\n")
    try:
        config = AnalysisConfig()
        with contextlib.redirect_stdout(io.StringIO()):
            assert resolve_spec_overrides(spec, config) == mesh.name
        assert config.E == 3e7
        assert config.nu == 0.25
        assert config.thickness == 0.02
        assert config.plane == "strain"
        assert config.fix == "left"
        assert config.traction == "right:1e6,0"
        assert config.body == "0,-78000"
        assert config.no_plot is True

        # CLI 显式优先: .spec 不覆盖
        config2 = AnalysisConfig(E=5e7, plane="stress")
        with contextlib.redirect_stdout(io.StringIO()):
            resolve_spec_overrides(spec, config2)
        assert config2.E == 5e7
        assert config2.plane == "stress"
        assert config2.thickness == 0.02  # 未显式指定 → .spec 覆盖
    finally:
        os.unlink(spec)
        os.unlink(mesh.name)


def test_spec_invalid_plane_fatal():
    spec = _write_spec("mesh = m.inp\nplane = 3d\n")
    try:
        config = AnalysisConfig()
        with pytest.raises(CliError), contextlib.redirect_stdout(io.StringIO()):
            resolve_spec_overrides(spec, config)
    finally:
        os.unlink(spec)


def test_spec_invalid_values_revalidated():
    """.spec 合并在构造后 — 非法值必须重新校验 (曾绕过进入求解)."""
    mesh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".inp", delete=False, encoding="utf-8")
    mesh.write("*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n"
               "*ELEMENT,TYPE=CPS4\n1,1,2,3,4\n")
    mesh.close()
    try:
        for bad_line in ("E = -5e11", "nu = 2.0", "t = 0"):
            spec = _write_spec(f"mesh = {mesh.name}\n{bad_line}\n")
            try:
                config = AnalysisConfig()
                with pytest.raises(ValueError):
                    with contextlib.redirect_stdout(io.StringIO()):
                        resolve_spec_overrides(spec, config)
            finally:
                os.unlink(spec)
    finally:
        os.unlink(mesh.name)


def test_spec_bom_first_line_accepted(tmp_path):
    """.spec UTF-8 BOM 曾把首行键变成 '﻿mesh' → mesh 键丢失 FATAL (审计
    2026-08-03). 必须正常解析. 直接测 preprocess.parse_spec_config (规避
    文件存在性校验)."""
    from fem2d.preprocess import parse_spec_config
    p = tmp_path / "bom.spec"
    p.write_bytes(b"\xef\xbb\xbf" + b"mesh = m.geo\nE = 2.1e11\n")
    spec = parse_spec_config(str(p))
    assert spec["mesh"] == "m.geo"
    assert spec["E"] == "2.1e11"


def test_resolve_geo_lc_batch_mode_skips_ask(tmp_path):
    """.geo lc 交互判定必须与 is_batch_mode 统一 — 仅 --fix-ux (无 lc)
    时必须视为批处理不提问; 曾手写条件漏掉 fix_ux/fix_uy, 批处理下
    提问挂起 (审计 2026-08-03 输入端整改)."""
    from fem2d.input_source import _resolve_geo_lc
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.5;\nPoint(1) = {0, 0, 0, lc};\n", encoding="utf-8")

    def _ask(prompt):
        raise AssertionError(f"批处理不应交互提问: {prompt!r}")

    config = AnalysisConfig(fix_ux="left")
    fp, tmp = _resolve_geo_lc(str(geo), config, ask=_ask)
    assert fp == str(geo) and tmp is None


def test_resolve_geo_lc_missing_lc_no_spurious_warn_in_batch(capsys, tmp_path):
    """.geo 无 lc 行 + 仅 --fix-ux 批处理: 不得 WARN (曾无条件提示)."""
    from fem2d.input_source import _resolve_geo_lc
    geo = tmp_path / "m.geo"
    geo.write_text("Point(1) = {0, 0, 0, 0.5};\n", encoding="utf-8")

    def _ask(prompt):
        raise AssertionError(f"批处理不应交互提问: {prompt!r}")

    config = AnalysisConfig(fix_ux="left")
    fp, tmp = _resolve_geo_lc(str(geo), config, ask=_ask)
    out = capsys.readouterr().out
    assert "WARN" not in out, f"批处理下不应提示 lc 缺失: {out!r}"
    assert fp == str(geo) and tmp is None


def test_spec_float_conversion_reports_key_name(tmp_path):
    """.spec float 值转换失败必须带键名 — 曾裸 ValueError 无上下文
   ."""
    from fem2d.input_source import resolve_spec_overrides
    from fem2d.config import AnalysisConfig
    spec = _write_spec("mesh = m.geo\nE = 2,1e11\n")
    try:
        config = AnalysisConfig()
        with pytest.raises(ValueError, match="'E'"):
            with contextlib.redirect_stdout(io.StringIO()):
                resolve_spec_overrides(spec, config)
    finally:
        os.unlink(spec)


def test_spec_does_not_override_cli_from_args():
    """.spec 合并前 config 来自 from_args — CLI 显式 (non-None) 字段保持优先."""
    mesh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".inp", delete=False, encoding="utf-8")
    mesh.write("*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n"
               "*ELEMENT,TYPE=CPS4\n1,1,2,3,4\n")
    mesh.close()
    spec = _write_spec(f"mesh = {mesh.name}\nE = 3e7\nnu = 0.25\n")
    try:
        ns = parse_args(["dummy.inp", "--E", "5e7"])  # CLI 显式 E
        config = AnalysisConfig.from_args(ns)
        assert config.E == 5e7
        with contextlib.redirect_stdout(io.StringIO()):
            resolve_spec_overrides(spec, config)
        assert config.E == 5e7       # CLI 优先
        assert config.nu == 0.25     # 未指定 → .spec 覆盖
    finally:
        os.unlink(spec)
        os.unlink(mesh.name)


def test_spec_applies_when_args_not_explicit():
    """.spec 的 fix/traction/save/no_plot 必须应用 — 曾因 argparse 空字符串/
    False 被误标记为 CLI 显式而跳过 (生产路径: from_args(parse_args()))."""
    mesh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".inp", delete=False, encoding="utf-8")
    mesh.write("*NODE\n1,0,0\n2,1,0\n3,1,1\n4,0,1\n"
               "*ELEMENT,TYPE=CPS4\n1,1,2,3,4\n")
    mesh.close()
    spec = _write_spec(f"mesh = {mesh.name}\nfix = left\n"
                       "body = 0,-1000\nno_plot = true\nsave = out.png\n")
    try:
        # 生产路径: 未指定任何覆盖参数
        ns = parse_args([mesh.name])
        config = AnalysisConfig.from_args(ns)
        assert "fix" not in config._explicit
        with contextlib.redirect_stdout(io.StringIO()):
            resolve_spec_overrides(spec, config)
        assert config.fix == "left"
        assert config.no_plot is True
        assert config.save == "out.png"
        assert config.body == "0,-1000"
    finally:
        os.unlink(spec)
        os.unlink(mesh.name)


# ── 4b. replace_elements 公共 API 锁定 (零测试覆盖 → 补) ──

def test_replace_elements_validates_and_rebuilds():
    """replace_elements: 校验 (非空/重复) + 缓存重建 + 只读保持."""
    from fem2d import Mesh

    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.],
                      [2., 0.], [2., 1.]])
    elems = np.array([[0, 1, 2, 3]])
    mesh = Mesh(nodes=nodes, elements=elems, E=3e7, nu=0.3,
                thickness=1.0, elem_type="CPS4")
    mesh.build_connectivity()
    a1 = mesh.areas.sum()

    # 正常替换: 第二单元加入 → 面积翻倍
    mesh.replace_elements(np.array([[0, 1, 2, 3], [1, 4, 5, 2]]))
    mesh.build_connectivity()
    assert abs(mesh.areas.sum() - 2 * a1) < 1e-12
    # 只读保持
    with pytest.raises(ValueError):
        mesh.elements[0, 0] = 5
    # 重复单元拒绝
    with pytest.raises(ValueError, match="重复单元"):
        mesh.replace_elements(np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))
    # 空单元集拒绝
    with pytest.raises(ValueError, match="不能为空"):
        mesh.replace_elements(np.empty((0, 4), dtype=int))
    # 越界索引拒绝
    with pytest.raises(ValueError, match="越界"):
        mesh.replace_elements(np.array([[0, 1, 2, 99]]))


# ── 5. main(argv) 阶段化编排可重入 ──

def test_main_returns_zero_for_batch(monkeypatch):
    """main(argv) 直接调用 (不经 run.py) — 阶段化编排可重入.

    网格生成 mock 掉 — 不依赖真实 Gmsh (评审: 缺 Gmsh 环境的可移植性).
    """
    from fem2d import input_source

    class _FakeImport:
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        elements = np.array([[0, 1, 2, 3]])
        elem_type = "CPS4"
        node_tag_to_index = {0: 0, 1: 1, 2: 2, 3: 3}
        element_tag_to_index = {0: 0}
        regions = None

    def fake_generate(geo_path, *, quad=False, output_path=None,
                      plane_type="stress"):
        return "dummy.msh", _FakeImport()

    monkeypatch.setattr(input_source, "generate_geo_with_topology",
                        fake_generate)
    from fem2d.runner import main
    with contextlib.redirect_stdout(io.StringIO()):
        code = main(["models/plate_q4.geo", "--fix", "left",
                     "--traction", "right:1e6,0", "--no-plot"])
    assert code == 0


def test_msh_input_passes_plane_type(monkeypatch):
    """.msh 直接输入必须把 --plane 传递给 import_msh (曾默认 stress 冲突)."""
    import tempfile

    from fem2d import gmsh_adapter
    from fem2d.runner import main

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".msh", delete=False, encoding="utf-8")
    handle.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
    handle.close()
    seen = {}

    class _Fake:
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        elements = np.array([[0, 1, 2, 3]])
        elem_type = "CPE4"
        node_tag_to_index = {0: 0, 1: 1, 2: 2, 3: 3}
        element_tag_to_index = {0: 0}
        regions = None

    def fake_import_msh(fp, *, require_quads=False, plane_type="stress"):
        seen["plane_type"] = plane_type
        return _Fake()

    monkeypatch.setattr(gmsh_adapter, "import_msh", fake_import_msh)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([handle.name, "--fix", "left", "--no-plot",
                         "--plane", "strain"])
        assert code == 0
        assert seen.get("plane_type") == "strain", (
            f"--plane strain 未传给 import_msh: {seen}")
    finally:
        os.unlink(handle.name)


def test_main_missing_file_returns_one():
    from fem2d.runner import main
    with contextlib.redirect_stdout(io.StringIO()):
        code = main(["nonexistent.inp", "--no-plot"])
    assert code == 1


def test_main_rejects_inp_input(monkeypatch):
    """Abaqus .inp 输入口已移除 — main 必须明确拒绝 (返回 2)."""
    import tempfile

    from fem2d import input_source
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".inp", delete=False, encoding="utf-8")
    handle.write("*NODE\n1,0,0\n")
    handle.close()
    monkeypatch.setattr(
        input_source, "generate_geo_with_topology",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应调用")))
    from fem2d.runner import main
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([handle.name, "--no-plot"])
        assert code == 2
    finally:
        os.unlink(handle.name)


def test_main_accepts_msh_input(monkeypatch):
    """.msh 直接输入必须可解析 (无需重新网格化) — mock import_msh 防真实 Gmsh."""

    class _FakeMshImport:
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        elements = np.array([[0, 1, 2, 3]])
        elem_type = "CPS4"
        node_tag_to_index = {0: 0, 1: 1, 2: 2, 3: 3}
        element_tag_to_index = {0: 0}
        regions = None

    import tempfile
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".msh", delete=False, encoding="utf-8")
    handle.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
    handle.close()
    from fem2d import gmsh_adapter
    monkeypatch.setattr(
        gmsh_adapter, "import_msh",
        lambda fp, **kwargs: _FakeMshImport())
    from fem2d.runner import main
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            code = main([handle.name, "--fix", "left",
                         "--traction", "right:1e6,0", "--no-plot"])
        assert code == 0
    finally:
        os.unlink(handle.name)


def test_resolve_txt_preserves_handwritten_geo(tmp_path, monkeypatch):
    """.txt 输入遇到同名手写 .geo 必须生成临时副本, 原始文件不碰 —
    曾静默覆盖导致手写几何永久丢失."""
    import shutil
    from fem2d import input_source
    from fem2d.input_source import resolve_txt
    txt = tmp_path / "m.txt"
    txt.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n边界 左 固定\n",
                   encoding="utf-8")
    geo = tmp_path / "m.geo"
    geo.write_text("// 手写几何\nPoint(1) = {0, 0, 0, 0.5};\n", encoding="utf-8")
    original = geo.read_text(encoding="utf-8")
    monkeypatch.setattr(
        input_source, "generate_geo_with_topology",
        lambda *a, **k: ("dummy.msh", None))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        resolve_txt(str(txt), AnalysisConfig())
    out = buf.getvalue()
    assert "临时副本" in out, f"未提示临时副本: {out!r}"
    assert geo.read_text(encoding="utf-8") == original, \
        "手写 .geo 被覆盖!"
    # 生成的临时副本含 @FEM 注解 (边界语义来源)
    tmp_geos = [f for f in os.listdir(tmp_path) if f.startswith(".fem2d-txt-")]
    assert tmp_geos, "未生成临时 .geo 副本"
    tmp_text = (tmp_path / tmp_geos[0]).read_text(encoding="utf-8")
    assert "@FEM:fix=" in tmp_text, "临时副本缺 @FEM 注解"
    shutil.rmtree(tmp_path)
