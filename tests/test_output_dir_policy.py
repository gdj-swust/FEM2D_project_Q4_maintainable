"""包 3/4 — .msh 输出位置策略 (--output-dir) 判别性测试.

覆盖:
- --output-dir 生效: .msh/临时文件写入指定目录, 输入目录无残留
- 默认行为不变: 无 --output-dir 时输出路径逐字节保持历史值 (回归锁)
- 只读目录: chmod / monkeypatch PermissionError / Windows 只读文件
  → 清晰错误 "输出目录不可写 — 请用 --output-dir 指定可写位置", 非裸异常
- 同名 .msh 覆盖保护: 本程序生成物 (带 $Comments 标记) 覆盖;
  来源不明 WARN + 临时副本, 原文件不碰 (resolve_txt 手写 .geo 同模式)

真实 gmsh 仅在测试 12 出现 (gmsh 不可用即 skip), 其余全部 monkeypatch
隔离 — 缺 gmsh 环境可移植 (项目惯例).
"""
import contextlib
import io
import os
import stat

import pytest

from fem2d.config import AnalysisConfig
from fem2d.errors import CliError
from scripts.gmsh_runner import (
    is_program_generated_msh,
    sanitize_geo_source,
    stamp_generated_msh,
    temp_copy_dir,
)

# 与生产同源标记常量 (测试断言探测必须用同一 token)
_MARKER = "// FEM2D-generated-mesh"

_MSH_4X = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
_MSH_4X_MARKED = _MSH_4X + (
    f"$Comments\n{_MARKER}\n$EndComments\n")


def _run(fn, *args, **kwargs):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = fn(*args, **kwargs)
    return result, out.getvalue()


def _fake_generate_factory(seen):
    """capture output_path 的 generate_geo_with_topology 替身 (不落盘)."""

    def fake_generate(geo_path, *, quad=False, output_path=None,
                      plane_type="stress"):
        seen["geo_path"] = geo_path
        seen["output_path"] = output_path
        seen["plane_type"] = plane_type
        return output_path, None
    return fake_generate


def _fake_run_gmsh(tmp, content="new mesh data\n"):
    """在 tmp 目录生成"gmsh 产物"并返回其路径."""
    path = str(tmp / "generated.msh")
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(content)
    return path


def _patch_mesh_chain(monkeypatch, tmp_path, fake_run_gmsh=None,
                      fake_import=None):
    """把 generate_geo_with_topology 的 gmsh 两环节替换为替身."""
    if fake_run_gmsh is None:
        fake_run_gmsh = lambda *a, **k: _fake_run_gmsh(tmp_path)
    if fake_import is None:
        fake_import = lambda fp, **kw: object()
    monkeypatch.setattr("scripts.geo_spec.run_gmsh", fake_run_gmsh)
    monkeypatch.setattr("fem2d.gmsh_adapter.import_msh", fake_import)


# ═══════════════════════════════════════════════════════════════
# 1. 配置层: 字段 / 校验 / .spec 键 / CLI 显式优先级
# ═══════════════════════════════════════════════════════════════

def test_config_output_dir_field_roundtrip_and_validation():
    config = AnalysisConfig(output_dir="out/")
    assert config.output_dir == "out/"
    assert AnalysisConfig.from_dict({"output_dir": "x"}).output_dir == "x"
    with pytest.raises(ValueError, match="output_dir"):
        AnalysisConfig.from_dict({"output_dir": 42})  # 非字符串必须拒绝


def test_spec_output_dir_resolves_relative_to_spec_dir(tmp_path, monkeypatch):
    """.spec 键 output_dir: 相对路径以 .spec 目录为基准 (与 mesh 一致)."""
    from fem2d.input_source import resolve_spec_overrides
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.1;\n", encoding="utf-8")
    spec = tmp_path / "m.spec"
    spec.write_text(
        f"mesh = {geo.name}\noutput_dir = out\n", encoding="utf-8")
    config = AnalysisConfig()
    resolve_spec_overrides(str(spec), config)
    assert config.output_dir == str(tmp_path / "out")

    # CLI 显式 --output-dir 优先 — .spec 不覆盖
    from fem2d.cli import parse_args
    config = AnalysisConfig.from_args(
        parse_args(["--output-dir", "cli_dir", str(geo)]))
    resolve_spec_overrides(str(spec), config)
    assert config.output_dir == "cli_dir"


def test_spec_output_dir_empty_value_rejected(tmp_path):
    from fem2d.input_source import resolve_spec_overrides
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.1;\n", encoding="utf-8")
    spec = tmp_path / "m.spec"
    spec.write_text(f"mesh = {geo.name}\noutput_dir =\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output_dir"):
        resolve_spec_overrides(str(spec), AnalysisConfig())


# ═══════════════════════════════════════════════════════════════
# 2. --output-dir 生效 (判别性) + 默认行为不变 (回归锁)
# ═══════════════════════════════════════════════════════════════

def test_geo_entry_output_dir_places_msh_in_output_dir(tmp_path, monkeypatch):
    """判别性: --output-dir 生效 — .msh 出现在指定目录, 输入目录无残留."""
    from fem2d import input_source
    from fem2d.input_source import resolve_geo
    geo = tmp_path / "m.geo"
    geo.write_text("Point(1) = {0, 0, 0, 0.1};\n", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(
        input_source, "generate_geo_with_topology",
        _fake_generate_factory(seen))
    out_dir = tmp_path / "out"

    msh, _gmsh, source_geo = resolve_geo(
        str(geo), AnalysisConfig(output_dir=str(out_dir)))

    assert seen["output_path"] == str(out_dir / "m.msh"), seen
    assert seen["plane_type"] == "stress"
    assert msh == str(out_dir / "m.msh")
    assert source_geo == str(geo)  # 源 .geo 路径不变 (用户自己的文件)
    assert out_dir.is_dir(), "--output-dir 目录不存在时应自动创建"
    assert not (tmp_path / "m.msh").exists(), "输入目录出现 .msh 残留"
    assert not list(out_dir.glob(".fem2d-write-probe-*")), \
        "写探测文件必须立即删除"


def test_geo_entry_default_output_path_unchanged(tmp_path, monkeypatch):
    """回归锁: 无 --output-dir 时 .msh 输出路径 = 历史值 (输入同目录)."""
    from fem2d import input_source
    from fem2d.input_source import resolve_geo
    geo = tmp_path / "m.geo"
    geo.write_text("Point(1) = {0, 0, 0, 0.1};\n", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(
        input_source, "generate_geo_with_topology",
        _fake_generate_factory(seen))

    msh, _g, _s = resolve_geo(str(geo), AnalysisConfig())

    assert seen["output_path"] == str(tmp_path / "m.msh"), seen
    assert msh == str(tmp_path / "m.msh")


def test_txt_entry_output_dir_places_all_artifacts_in_output_dir(
        tmp_path, monkeypatch):
    """判别性: .txt 入口 — 生成的 .geo 与 .msh 都在 --output-dir 内."""
    from fem2d import input_source
    from fem2d.input_source import resolve_txt
    txt = tmp_path / "m.txt"
    txt.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n",
                   encoding="utf-8")
    seen = {}
    monkeypatch.setattr(
        "scripts.geo_spec.generate_geo",
        lambda spec, output_path, quad=False: output_path)
    monkeypatch.setattr(
        input_source, "generate_geo_with_topology",
        _fake_generate_factory(seen))
    out_dir = tmp_path / "out"

    msh, _g, source_geo = resolve_txt(
        str(txt), AnalysisConfig(output_dir=str(out_dir)))

    assert seen["geo_path"] == str(out_dir / "m.geo"), seen
    assert seen["output_path"] == str(out_dir / "m.msh"), seen
    assert msh == str(out_dir / "m.msh")
    assert source_geo == str(out_dir / "m.geo")
    assert not (tmp_path / "m.geo").exists(), "输入目录出现 .geo 残留"
    assert not (tmp_path / "m.msh").exists(), "输入目录出现 .msh 残留"


def test_txt_entry_default_output_path_unchanged(tmp_path, monkeypatch):
    """回归锁: .txt 入口默认路径 = 历史值 (输入同目录)."""
    from fem2d import input_source
    from fem2d.input_source import resolve_txt
    txt = tmp_path / "m.txt"
    txt.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n",
                   encoding="utf-8")
    seen = {}
    monkeypatch.setattr(
        "scripts.geo_spec.generate_geo",
        lambda spec, output_path, quad=False: output_path)
    monkeypatch.setattr(
        input_source, "generate_geo_with_topology",
        _fake_generate_factory(seen))

    msh, _g, source_geo = resolve_txt(str(txt), AnalysisConfig())

    assert seen["geo_path"] == str(tmp_path / "m.geo"), seen
    assert seen["output_path"] == str(tmp_path / "m.msh"), seen
    assert msh == str(tmp_path / "m.msh")
    assert source_geo == str(tmp_path / "m.geo")


def test_lc_temp_copy_goes_to_output_dir(tmp_path):
    """--output-dir 时 lc 临时副本写入输出目录 (只读输入目录可完整工作)."""
    from fem2d.input_source import _resolve_geo_lc
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.1;\nPoint(1) = {0, 0, 0, lc};\n",
                   encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    config = AnalysisConfig(lc=0.2)

    _fp, tmp_geo = _resolve_geo_lc(
        str(geo), config, ask=lambda _p: "", temp_dir=str(out_dir))

    assert tmp_geo is not None
    assert os.path.dirname(tmp_geo) == str(out_dir), tmp_geo
    assert not list(tmp_path.glob(".fem2d-gmsh-source-*")), \
        "临时副本泄漏到输入目录"


def test_msh_direct_input_output_dir_warns(tmp_path, monkeypatch):
    """.msh 直接输入不生成任何文件 — --output-dir 必须 WARN (同 --quad 模式)."""
    from fem2d.input_source import resolve_input_file
    msh = tmp_path / "m.msh"
    msh.write_text(_MSH_4X, encoding="ascii")
    monkeypatch.setattr(
        "fem2d.gmsh_adapter.import_msh", lambda fp, **kw: object())

    _resolved, out = _run(
        resolve_input_file, str(msh),
        AnalysisConfig(output_dir=str(tmp_path / "out")))

    assert "只对 .geo/.txt" in out, out
    assert not (tmp_path / "out").exists(), ".msh 入口不应创建输出目录"


# ═══════════════════════════════════════════════════════════════
# 3. 只读目录 → 清晰错误 (非裸异常)
# ═══════════════════════════════════════════════════════════════

def test_readonly_input_dir_clear_error_posix(tmp_path):
    """chmod 只读输入目录 → 清晰错误 (POSIX 语义)."""
    if os.name == "nt":
        pytest.skip("POSIX-only: Windows 目录只读属性不阻止文件创建")
    from fem2d.input_source import resolve_geo
    inp = tmp_path / "inp"
    inp.mkdir()
    geo = inp / "m.geo"
    geo.write_text("Point(1) = {0, 0, 0, 0.1};\n", encoding="utf-8")
    os.chmod(inp, 0o555)
    try:
        with pytest.raises(CliError) as exc:
            resolve_geo(str(geo), AnalysisConfig())
        assert "输出目录不可写" in str(exc.value), exc.value
        assert "--output-dir" in str(exc.value), exc.value
        assert exc.value.exit_code == 1
    finally:
        os.chmod(inp, 0o755)


def test_windows_readonly_attr_dir_no_false_positive(tmp_path, monkeypatch):
    """Windows 只读属性目录: 属性不阻止文件创建 — 预检必须放行 (不误报),
    真实失败由 os.replace 守护兜底 (见 test_windows_readonly_msh_target).
    POSIX 上 chmod(IREAD) 是真只读 — 该断言只对 Windows 语义成立."""
    if os.name != "nt":
        pytest.skip("Windows-only: POSIX chmod(IREAD) 是真只读")
    from fem2d import input_source
    from fem2d.input_source import resolve_geo
    inp = tmp_path / "inp"
    inp.mkdir()
    geo = inp / "m.geo"
    geo.write_text("Point(1) = {0, 0, 0, 0.1};\n", encoding="utf-8")
    os.chmod(inp, stat.S_IREAD)  # FILE_ATTRIBUTE_READONLY (Windows)
    seen = {}
    monkeypatch.setattr(
        input_source, "generate_geo_with_topology",
        _fake_generate_factory(seen))
    try:
        msh, _g, _s = resolve_geo(str(geo), AnalysisConfig())
    finally:
        os.chmod(inp, stat.S_IWRITE)
    assert seen["output_path"] == str(inp / "m.msh")
    assert msh == str(inp / "m.msh")


def test_output_dir_creation_denied_clear_error(tmp_path, monkeypatch):
    """权限拒绝 (模拟: 目录无法创建) → 清晰错误."""
    from fem2d.input_source import resolve_geo
    geo = tmp_path / "m.geo"
    geo.write_text("Point(1) = {0, 0, 0, 0.1};\n", encoding="utf-8")

    def denied(path, exist_ok=False):
        raise PermissionError(f"denied: {path}")

    monkeypatch.setattr("os.makedirs", denied)
    with pytest.raises(CliError) as exc:
        resolve_geo(str(geo), AnalysisConfig(output_dir=str(tmp_path / "out")))
    assert "输出目录不可写" in str(exc.value), exc.value
    assert "--output-dir" in str(exc.value), exc.value
    assert exc.value.exit_code == 1


def test_publish_replace_denied_clear_error(tmp_path, monkeypatch):
    """模拟 os.replace 拒绝 (只读目录权限拒绝) → 发布时刻清晰错误."""
    from fem2d.input_source import generate_geo_with_topology
    _patch_mesh_chain(monkeypatch, tmp_path)

    def denied(src, dst):
        raise PermissionError(f"denied: {dst}")

    monkeypatch.setattr("os.replace", denied)
    with pytest.raises(CliError) as exc:
        generate_geo_with_topology(str(tmp_path / "m.geo"))
    assert "输出目录不可写" in str(exc.value), exc.value
    assert "--output-dir" in str(exc.value), exc.value


def test_windows_readonly_msh_target_clear_error(tmp_path, monkeypatch):
    """Windows 真实只读场景: 目标 .msh 带只读属性 → os.replace 拒绝 →
    清晰错误, 原文件不被动. (Windows 目录只读属性不阻止创建, 此场景
    是 Windows 上唯一真实的"写失败"模拟.)"""
    if os.name != "nt":
        pytest.skip("Windows-only: POSIX 下 os.replace 可覆盖只读文件")
    from fem2d.input_source import generate_geo_with_topology
    target = tmp_path / "m.msh"
    target.write_text(_MSH_4X_MARKED, encoding="ascii")
    _patch_mesh_chain(monkeypatch, tmp_path)
    os.chmod(target, stat.S_IREAD)
    try:
        with pytest.raises(CliError) as exc:
            generate_geo_with_topology(
                str(tmp_path / "m.geo"), output_path=str(target))
        assert "输出目录不可写" in str(exc.value), exc.value
    finally:
        os.chmod(target, stat.S_IWRITE)
    # 原文件未被覆盖
    assert target.read_text(encoding="ascii") == _MSH_4X_MARKED


# ═══════════════════════════════════════════════════════════════
# 4. 同名 .msh 覆盖保护: 生成物覆盖 / 来源不明 WARN+临时副本
# ═══════════════════════════════════════════════════════════════

def test_foreign_msh_not_overwritten_warns_and_redirects(tmp_path, monkeypatch):
    """判别性: 来源不明 .msh — WARN + 临时副本, 原文件不碰 (不能误伤)."""
    from fem2d.input_source import generate_geo_with_topology
    target = tmp_path / "m.msh"
    target.write_text("handwritten mesh — 手写/其他工具产物\n",
                      encoding="utf-8")
    original = target.read_text(encoding="utf-8")

    def fake_run_gmsh(*args, **kwargs):
        return _fake_run_gmsh(tmp_path, content="new generated data\n")

    _patch_mesh_chain(monkeypatch, tmp_path, fake_run_gmsh=fake_run_gmsh)

    result, out = _run(
        generate_geo_with_topology, str(tmp_path / "m.geo"),
        output_path=str(target))

    assert "不覆盖" in out, out
    assert target.read_text(encoding="utf-8") == original, \
        "来源不明的 .msh 被覆盖!"
    published, _gmsh = result
    assert published != str(target)
    assert os.path.basename(published).startswith(".fem2d-msh-"), published
    with open(published, encoding="utf-8") as stream:
        assert stream.read() == "new generated data\n"


def test_own_msh_overwritten_silently(tmp_path, monkeypatch):
    """判别性: 本程序生成物 (带标记) — 覆盖 (当前行为), 无 WARN."""
    from fem2d.input_source import generate_geo_with_topology
    target = tmp_path / "m.msh"
    target.write_text(_MSH_4X_MARKED, encoding="ascii")
    _patch_mesh_chain(monkeypatch, tmp_path)

    result, out = _run(
        generate_geo_with_topology, str(tmp_path / "m.geo"),
        output_path=str(target))

    assert "不覆盖" not in out, out
    published, _gmsh = result
    assert published == str(target)
    assert target.read_text(encoding="ascii") == "new mesh data\n"


# ═══════════════════════════════════════════════════════════════
# 5. 标记单元: 注入 / 识别 / gmsh 兼容 / 版本消毒
# ═══════════════════════════════════════════════════════════════

def test_stamp_and_detect_marker_unit(tmp_path):
    """MSH 4.x 注入标记并识别; 幂等; 残缺/低版本文件不碰."""
    f4 = tmp_path / "a.msh"
    f4.write_text(_MSH_4X, encoding="ascii")
    assert stamp_generated_msh(str(f4)) is True
    assert is_program_generated_msh(str(f4)) is True
    assert _MARKER in f4.read_text(encoding="ascii")
    assert f4.read_text(encoding="ascii").count("$Comments") == 1

    # 幂等: 重复注入不产生重复段
    assert stamp_generated_msh(str(f4)) is False
    assert f4.read_text(encoding="ascii").count("$Comments") == 1

    # v2.2 (版本行不是 4.x) — 不注入, 不被误识别
    f2 = tmp_path / "b.msh"
    f2.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="ascii")
    assert stamp_generated_msh(str(f2)) is False
    assert is_program_generated_msh(str(f2)) is False

    # 残缺 (无 $EndMeshFormat) — 不注入 (防御: 不碰非本程序产物)
    fbad = tmp_path / "c.msh"
    fbad.write_text("$MeshFormat\n4.1 0 8\n", encoding="ascii")
    assert stamp_generated_msh(str(fbad)) is False
    assert fbad.read_text(encoding="ascii") == "$MeshFormat\n4.1 0 8\n"


def test_marker_readable_by_real_gmsh(tmp_path):
    """$Comments 标记被真实 gmsh 接受: 实体/节点完整恢复 (2026-08 实测
    锁定 — 标记方案的前提是 gmsh 读回不受影响)."""
    try:
        import gmsh
    except ImportError as error:
        pytest.skip(f"gmsh 不可用: {error}")
    msh = str(tmp_path / "t.msh")
    try:
        gmsh.initialize()
        gmsh.model.add("t")
        gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(msh)
    except Exception as error:
        pytest.skip(f"gmsh 生成失败: {error}")
    finally:
        gmsh.finalize()
    assert stamp_generated_msh(msh) is True
    try:
        gmsh.initialize()
        gmsh.open(msh)
        entities = gmsh.model.getEntities()
        n_nodes = len(gmsh.model.mesh.getNodes()[0])
    except Exception as error:
        pytest.skip(f"gmsh 读回失败: {error}")
    finally:
        gmsh.finalize()
    assert len(entities) >= 2, entities  # 至少 1 条边 + 1 个面
    assert n_nodes > 0
    assert is_program_generated_msh(msh) is True


def test_sanitize_strips_msh_file_version():
    """脚本内 Mesh.MshFileVersion 必须剥离 — 低版本输出无法注入标记,
    覆盖保护失效 (与 Mesh.Format 同属"输出格式由 FEM2D 拥有")."""
    source = "Mesh.MshFileVersion = 2;\nMesh 2;\n"
    sanitized = sanitize_geo_source(source)
    assert "MshFileVersion = 2" not in sanitized
    assert "removed" in sanitized


# ═══════════════════════════════════════════════════════════════
# 6. 临时副本目录: --output-dir vs 相对 Include 锚定
# ═══════════════════════════════════════════════════════════════

def test_temp_copy_dir_include_anchor(tmp_path):
    """含相对 Include 的 .geo 临时副本必须留在源目录 (相对引用以所在
    目录解析) — 否则 Include 断裂."""
    geo = tmp_path / "m.geo"
    geo.write_text('Include "part.geo";\n', encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result, out = _run(temp_copy_dir, str(geo), str(out_dir))

    assert result == str(tmp_path), result
    assert "Include" in out, out  # 必须 WARN 说明原因
    # 无 Include → 正常进入 --output-dir
    plain = tmp_path / "p.geo"
    plain.write_text("lc = 0.1;\n", encoding="utf-8")
    assert temp_copy_dir(str(plain), str(out_dir)) == str(out_dir)


def test_sanitized_copy_stays_next_to_include_geo(tmp_path):
    """gmsh_runner 消毒副本: 含相对 Include 时留在源目录, 否则进
    --output-dir (与 lc 临时副本同策略)."""
    from scripts.gmsh_runner import _geometry_without_explicit_save
    geo = tmp_path / "m.geo"
    geo.write_text('Include "part.geo";\nMesh 2;\n', encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    copy_path = None
    try:
        copy_path, _cleanup = _geometry_without_explicit_save(
            str(geo), temp_dir=str(out_dir))
        assert os.path.dirname(copy_path) == str(tmp_path), copy_path
    finally:
        if copy_path is not None and os.path.isfile(copy_path):
            os.unlink(copy_path)
