"""复查轮审计 2b — 四入口 ( .txt / .geo / .spec / .msh ) 端到端集成测试.

resolve_input_file 的四个分支此前只有 .spec 有 e2e (test_spec_input_end_to_end_resolve);
.geo/.txt/.msh 分支只有单函数测试 — 正是 .spec 扩展名回归的教训
(单函数测试通过但集成路径无测试)。全部 monkeypatch 隔离 gmsh。
"""
import contextlib
import io

import pytest

from fem2d.config import AnalysisConfig
from fem2d.errors import CliError
from fem2d.input_source import resolve_input_file


def _run(fp, config=None):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        resolved = resolve_input_file(fp, config or AnalysisConfig())
    return resolved, out.getvalue()


# ── .geo 入口 ──

def test_geo_entry_end_to_end(tmp_path, monkeypatch):
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.1;\n", encoding="utf-8")
    called = {}

    def fake_resolve_geo(fp, config, ask=None):
        called["fp"] = fp
        return str(tmp_path / "m.msh"), "gmsh_import_fake", str(geo)

    monkeypatch.setattr("fem2d.input_source.resolve_geo", fake_resolve_geo)
    resolved, _ = _run(str(geo))
    assert called["fp"] == str(geo)
    assert resolved.fp.endswith("m.msh")
    assert resolved.gmsh_import == "gmsh_import_fake"
    assert resolved.source_geo_path == str(geo)
    assert resolved.quad_applied is False
    assert resolved.geo_config_applied is True  # @FEM 已在 resolve_geo 内合并


# ── .txt 入口 ──

def test_txt_entry_end_to_end(tmp_path, monkeypatch):
    txt = tmp_path / "m.txt"
    txt.write_text("矩形 长=1 高=1 网格=0.1\n", encoding="utf-8")
    called = {}

    def fake_resolve_txt(fp, config):
        called["fp"] = fp
        return str(tmp_path / "m.msh"), "gmsh_import_fake", str(tmp_path / "m.geo")

    monkeypatch.setattr("fem2d.input_source.resolve_txt", fake_resolve_txt)
    resolved, _ = _run(str(txt))
    assert called["fp"] == str(txt)
    assert resolved.fp.endswith("m.msh")
    assert resolved.gmsh_import == "gmsh_import_fake"
    assert resolved.quad_applied is False


# ── .msh 入口 ──

def test_msh_entry_end_to_end(tmp_path, monkeypatch):
    msh = tmp_path / "m.msh"
    msh.write_text("$MeshFormat\n", encoding="ascii")
    called = {}

    def fake_import_msh(fp, **kwargs):
        called["fp"] = fp
        called["plane"] = kwargs.get("plane_type")
        return "gmsh_import_fake"

    monkeypatch.setattr("fem2d.gmsh_adapter.import_msh", fake_import_msh)
    resolved, _ = _run(str(msh))
    assert called["fp"] == str(msh)
    assert called["plane"] == "stress"  # 默认平面态必须显式传入
    assert resolved.fp == str(msh)
    assert resolved.gmsh_import == "gmsh_import_fake"
    assert resolved.source_geo_path is None


def test_msh_entry_plane_strain_propagated(tmp_path, monkeypatch):
    msh = tmp_path / "m.msh"
    msh.write_text("$MeshFormat\n", encoding="ascii")
    called = {}

    def fake_import_msh(fp, **kwargs):
        called["plane"] = kwargs.get("plane_type")
        return object()

    monkeypatch.setattr("fem2d.gmsh_adapter.import_msh", fake_import_msh)
    resolve_input_file(str(msh), AnalysisConfig(plane="strain"))
    assert called["plane"] == "strain"


def test_msh_entry_uppercase_extension(tmp_path, monkeypatch):
    # 扩展名大小写不敏感 — Windows .MSH 曾直接被拒
    msh = tmp_path / "m.MSH"
    msh.write_text("$MeshFormat\n", encoding="ascii")
    monkeypatch.setattr(
        "fem2d.gmsh_adapter.import_msh",
        lambda fp, **kw: "gmsh_import_fake")
    resolved, _ = _run(str(msh))
    assert resolved.fp == str(msh)


def test_msh_entry_import_failure_clierror(tmp_path, monkeypatch):
    msh = tmp_path / "m.msh"
    msh.write_text("$MeshFormat\n", encoding="ascii")
    monkeypatch.setattr(
        "fem2d.gmsh_adapter.import_msh", lambda fp, **kw: None)
    with pytest.raises(CliError, match="导入失败"):
        _run(str(msh))


# ── .spec 入口 (判别: 解析后必须重算扩展名进入对应分支) ──

def test_spec_entry_resolves_to_geo_branch(tmp_path, monkeypatch):
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.1;\n", encoding="utf-8")
    spec = tmp_path / "m.spec"
    spec.write_text(f"mesh = {geo.name}\n", encoding="utf-8")
    called = {}

    def fake_resolve_geo(fp, config, ask=None):
        called["fp"] = fp
        return str(tmp_path / "m.msh"), None, None

    monkeypatch.setattr("fem2d.input_source.resolve_geo", fake_resolve_geo)
    resolved, _ = _run(str(spec))
    assert called["fp"].endswith(".geo"), \
        f".spec 解析后未进入 .geo 分支: {called.get('fp')!r}"
    assert resolved.fp.endswith("m.msh")


# ── 拒绝路径 ──

def test_inp_entry_rejected(tmp_path):
    inp = tmp_path / "m.inp"
    inp.write_text("*NODE\n", encoding="ascii")
    with pytest.raises(CliError) as exc:
        _run(str(inp))
    assert exc.value.exit_code == 2
    assert ".inp" in str(exc.value)


def test_unknown_extension_rejected(tmp_path):
    bad = tmp_path / "m.xyz"
    bad.write_text("x", encoding="ascii")
    with pytest.raises(CliError) as exc:
        _run(str(bad))
    assert exc.value.exit_code == 2


def test_final_path_not_msh_rejected(tmp_path, monkeypatch):
    # .geo 分支返回非 .msh → "最终需要 .msh" (曾静默接受错误产物)
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.1;\n", encoding="utf-8")

    def fake_resolve_geo(fp, config, ask=None):
        return str(tmp_path / "m.out"), None, None

    monkeypatch.setattr("fem2d.input_source.resolve_geo", fake_resolve_geo)
    with pytest.raises(CliError) as exc:
        _run(str(geo))
    assert exc.value.exit_code == 1
    assert ".msh" in str(exc.value)
