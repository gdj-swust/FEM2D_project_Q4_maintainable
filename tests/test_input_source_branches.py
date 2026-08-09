"""input_source.py 防御分支补测 (覆盖率 76% → ≥80%) — 包 2 覆盖率任务.

未覆盖行集中的路径: 模块路径守卫、generate_geo_with_topology 的 quad
验证重试/耗尽、physical_point_from_geo 的 Gmsh 会话内部 (fake 模块,
不依赖真实 gmsh)、lc 覆盖临时副本、resolve_txt 手写 .geo 保护、
resolve_input_file 的 .txt/.msh 参数 WARN。

判别性: 每条测试断言具体返回元组/异常类型/输出文本/文件状态。
"""
import importlib
import os
import sys

import numpy as np
import pytest

import fem2d.gmsh_adapter as gmsh_adapter_mod
import fem2d.input_source as isrc
import fem2d.stress as stress_mod
from fem2d.config import AnalysisConfig
from fem2d.errors import CliError
from fem2d.gmsh_adapter import GmshTopologyError
from fem2d.mesh import Mesh


def _square_mesh():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        elem_type="CPS4")


# ═══════════════════════════════════════════════════════════════
# 模块路径守卫 (sys.path 注入)
# ═══════════════════════════════════════════════════════════════

def test_module_reload_does_not_restore_project_root(monkeypatch):
    """reload 不得重新注入项目根 (pkg11 A16 新契约).

    曾模块顶层 sys.path.insert 把项目根写进库用户进程 — reload 后
    守卫"恢复"是旧行为的判别标记; 现注入归 _import_scripts 惰性化,
    模块顶层零副作用, 首次使用 scripts 工具层时才注入。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(isrc.__file__)))
    monkeypatch.setattr(sys, "path",
                        [p for p in sys.path if p != root])
    importlib.reload(isrc)
    assert root not in sys.path
    isrc._import_scripts("geo_spec")
    assert root in sys.path


# ═══════════════════════════════════════════════════════════════
# generate_geo_with_topology: quad 验证重试
# ═══════════════════════════════════════════════════════════════

def _patch_run_gmsh(monkeypatch, tmp_path):
    """run_gmsh 每次调用都写出一个真实临时 .msh (os.replace 需要源存在)."""
    import scripts.geo_spec as geo_spec_mod
    generated = tmp_path / "gen.msh"
    def _run(*a, **k):
        generated.write_text("x", encoding="utf-8")
        return str(generated)
    monkeypatch.setattr(geo_spec_mod, "run_gmsh", _run)
    return generated


def test_generate_geo_quad_retry_then_publish(tmp_path, monkeypatch, capsys):
    """拓扑验证失败 → 清理临时网格重试, 成功后才原子发布."""
    import scripts.geo_spec as geo_spec_mod  # noqa: F401 (patch 目标)
    geo = tmp_path / "m.geo"
    geo.write_text('SetFactory("OpenCASCADE");\n', encoding="utf-8")
    generated = _patch_run_gmsh(monkeypatch, tmp_path)
    calls = {"n": 0}
    def _import(msh_path, require_quads=False, plane_type="stress"):
        calls["n"] += 1
        if calls["n"] == 1:
            raise GmshTopologyError("混合三角/四边")
        return "imported"
    monkeypatch.setattr(gmsh_adapter_mod, "import_msh", _import)
    out_path = tmp_path / "final.msh"
    final, imp = isrc.generate_geo_with_topology(
        str(geo), quad=True, output_path=str(out_path))
    assert calls["n"] == 2
    assert final == str(out_path) and imp == "imported"
    assert "[Gmsh] quad 重组验证失败 (第 1 次) — 重试..." in \
        capsys.readouterr().out
    assert out_path.exists() and not generated.exists()   # 已发布


def test_generate_geo_quad_retry_exhausted_raises(tmp_path, monkeypatch):
    """连续验证失败到重试上限 → 抛原始异常 (旧文件保留)."""
    geo = tmp_path / "m.geo"
    geo.write_text('SetFactory("OpenCASCADE");\n', encoding="utf-8")
    _patch_run_gmsh(monkeypatch, tmp_path)
    boom = GmshTopologyError("始终失败")
    monkeypatch.setattr(
        gmsh_adapter_mod, "import_msh",
        lambda *a, **k: (_ for _ in ()).throw(boom))
    with pytest.raises(GmshTopologyError, match="始终失败"):
        isrc.generate_geo_with_topology(
            str(geo), quad=True, output_path=str(tmp_path / "f.msh"))


# ═══════════════════════════════════════════════════════════════
# physical_point_from_geo: fake Gmsh 会话 (不依赖真实 gmsh)
# ═══════════════════════════════════════════════════════════════

class _FakeModel:
    def __init__(self, groups, names, entities, coords):
        self.groups = groups                 # [(dim, tag), ...]
        self.names = names                   # {(dim, tag): name}
        self.entities = entities             # {(dim, tag): [entity, ...]}
        self.coords = coords                 # {entity: (x, y, z) | raise}

    def getPhysicalGroups(self):
        return self.groups

    def getPhysicalName(self, dim, tag):
        return self.names.get((int(dim), int(tag)), "")

    def getEntitiesForPhysicalGroup(self, dim, tag):
        return self.entities.get((int(dim), int(tag)), [])

    def getValue(self, dim, entity, params):
        entry = self.coords[int(entity)]
        if isinstance(entry, Exception):
            raise entry
        return entry

    class geo:
        @staticmethod
        def synchronize():
            pass


class _FakeGmshModule:
    def __init__(self, model):
        self.model = model
        self.initialized = False
        self.finalized = False
        self.opened = None

    def isInitialized(self):
        return self.initialized

    def initialize(self):
        self.initialized = True

    def finalize(self):
        self.finalized = True

    def open(self, path):
        self.opened = path


def _install_fake_gmsh(monkeypatch, model):
    module = _FakeGmshModule(model)
    monkeypatch.setattr(gmsh_adapter_mod, "_load_gmsh_module",
                        lambda: module)
    return module


def _geo_file(tmp_path, extra=""):
    geo = tmp_path / "pt.geo"
    geo.write_text('SetFactory("OpenCASCADE");\n' + extra, encoding="utf-8")
    return str(geo)


def test_physical_point_gmsh_unavailable(monkeypatch):
    """gmsh API 导入失败 → gmsh_unavailable (曾误报其他原因)."""
    monkeypatch.setattr(
        gmsh_adapter_mod, "_load_gmsh_module",
        lambda: (_ for _ in ()).throw(ImportError("no gmsh")))
    result = isrc.physical_point_from_geo("some.geo", "load", _square_mesh())
    assert result[3] == "gmsh_unavailable"
    assert result[:3] == (None, None, None)


def test_physical_point_fake_session_success(tmp_path, monkeypatch):
    """fake gmsh 会话: 点命中网格节点, 会话自启自终."""
    model = _FakeModel(
        groups=[(0, 1)],
        names={(0, 1): "load"},
        entities={(0, 1): [7]},
        coords={7: (0.5, 0.5, 0.0)})
    module = _install_fake_gmsh(monkeypatch, model)
    nid, label, dist, reason = isrc.physical_point_from_geo(
        _geo_file(tmp_path), "load", _square_mesh())
    assert reason is None
    assert nid in (0, 1, 2, 3)
    assert dist == pytest.approx(np.sqrt(0.5), rel=1e-12)
    assert module.initialized and module.finalized   # owns_session 自清理
    assert module.opened is not None


def test_physical_point_fake_session_reuses_existing(monkeypatch, tmp_path):
    """gmsh 会话已初始化 → 不重复 initialize/finalize."""
    model = _FakeModel(
        groups=[(0, 1)],
        names={(0, 1): "load"},
        entities={(0, 1): [7]},
        coords={7: (0.5, 0.5, 0.0)})
    module = _install_fake_gmsh(monkeypatch, model)
    module.initialized = True
    nid, label, dist, reason = isrc.physical_point_from_geo(
        _geo_file(tmp_path), "load", _square_mesh())
    assert reason is None and nid is not None
    assert not module.finalized   # 外部会话不归本函数管


def test_physical_point_bad_entity_skipped(tmp_path, monkeypatch):
    """坏实体坐标读取失败 → 跳过继续 (不整段崩溃)."""
    model = _FakeModel(
        groups=[(0, 1)],
        names={(0, 1): "load"},
        entities={(0, 1): [7, 8]},
        coords={7: RuntimeError("bad entity"), 8: (0.5, 0.5, 0.0)})
    _install_fake_gmsh(monkeypatch, model)
    nid, label, dist, reason = isrc.physical_point_from_geo(
        _geo_file(tmp_path), "load", _square_mesh())
    assert reason is None and nid is not None   # 跳过 7, 命中 8


def test_physical_point_all_bad_entities_not_found(tmp_path, monkeypatch):
    """全部实体损坏 → 无命中 → not_found."""
    model = _FakeModel(
        groups=[(0, 1)],
        names={(0, 1): "load"},
        entities={(0, 1): [7]},
        coords={7: RuntimeError("bad")})
    _install_fake_gmsh(monkeypatch, model)
    result = isrc.physical_point_from_geo(
        _geo_file(tmp_path), "load", _square_mesh())
    assert result[3] == "not_found"


def test_physical_point_ambiguous_via_fake(tmp_path, monkeypatch):
    """多个同标签 Physical Point → ambiguous."""
    model = _FakeModel(
        groups=[(0, 1)],
        names={(0, 1): "dup"},
        entities={(0, 1): [7, 8]},
        coords={7: (0.5, 0.5, 0.0), 8: (0.2, 0.8, 0.0)})
    _install_fake_gmsh(monkeypatch, model)
    result = isrc.physical_point_from_geo(
        _geo_file(tmp_path), "dup", _square_mesh())
    assert result[3] == "ambiguous"


def test_physical_point_skips_nonzero_dimension_groups(tmp_path,
                                                       monkeypatch):
    """非 0 维物理组 (曲线/曲面) → 跳过, 只取 0 维点."""
    model = _FakeModel(
        groups=[(1, 5), (0, 1)],       # 曲线组先出现
        names={(1, 5): "load", (0, 1): "load"},
        entities={(1, 5): [3], (0, 1): [7]},
        coords={3: (9.0, 9.0, 0.0), 7: (0.5, 0.5, 0.0)})
    _install_fake_gmsh(monkeypatch, model)
    nid, label, dist, reason = isrc.physical_point_from_geo(
        _geo_file(tmp_path), "load", _square_mesh())
    assert reason is None and nid in (0, 1, 2, 3)


def test_physical_point_session_failure_returns_unavailable(tmp_path,
                                                            monkeypatch):
    """gmsh 会话读取失败 → gmsh_unavailable (曾宽 except 吞内部逻辑错误)."""
    class _BoomModel:
        def getPhysicalGroups(self):
            raise RuntimeError("session dead")
        class geo:
            @staticmethod
            def synchronize():
                pass
    _install_fake_gmsh(monkeypatch, _BoomModel())
    result = isrc.physical_point_from_geo(
        _geo_file(tmp_path), "load", _square_mesh())
    assert result[3] == "gmsh_unavailable"


def test_physical_point_cleanup_unlink_failure_swallowed(tmp_path,
                                                         monkeypatch):
    """临时副本删除失败 (OSError) → 吞掉, 不影响结果."""
    model = _FakeModel(
        groups=[(0, 1)],
        names={(0, 1): "load"},
        entities={(0, 1): [7]},
        coords={7: (0.5, 0.5, 0.0)})
    _install_fake_gmsh(monkeypatch, model)
    real_unlink = os.unlink
    def _failing_unlink(path, *a, **k):
        if ".fem2d-gmsh-api-source-" in str(path):
            raise OSError("locked")
        return real_unlink(path, *a, **k)
    monkeypatch.setattr(os, "unlink", _failing_unlink)
    geo = _geo_file(tmp_path, extra='Save "x.inp";\n')
    result = isrc.physical_point_from_geo(geo, "load", _square_mesh())
    assert result[3] is None   # 清理失败不掩盖成功结果


def test_physical_point_cleans_temporary_geo(tmp_path, monkeypatch):
    """消毒副本 (.geo 含 Save 命令) → 会话结束后删除."""
    model = _FakeModel(
        groups=[(0, 1)],
        names={(0, 1): "load"},
        entities={(0, 1): [7]},
        coords={7: (0.5, 0.5, 0.0)})
    _install_fake_gmsh(monkeypatch, model)
    geo = _geo_file(tmp_path, extra='Save "x.inp";\n')
    result = isrc.physical_point_from_geo(geo, "load", _square_mesh())
    assert result[3] is None
    leftovers = [p for p in tmp_path.iterdir()
                 if p.name.startswith(".fem2d-gmsh-api-source")]
    assert leftovers == []   # 临时副本已清理


def test_physical_point_outside_element_rejected(tmp_path, monkeypatch):
    """AABB 内但不在任何单元内 (凹域/孔洞 construction point) → 域外."""
    tri_mesh = Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
        elements=np.array([[0, 1, 2]], dtype=int), elem_type="CPS3")
    model = _FakeModel(
        groups=[(0, 1)],
        names={(0, 1): "load"},
        entities={(0, 1): [7]},
        coords={7: (0.8, 0.8, 0.0)})
    _install_fake_gmsh(monkeypatch, model)
    result = isrc.physical_point_from_geo(
        _geo_file(tmp_path), "load", tri_mesh)
    assert result[3] == "outside_domain"


def test_physical_point_too_far_rejected(monkeypatch, tmp_path):
    """最近节点超 3× 特征尺寸 → too_far (距离阈值三级过滤)."""
    model = _FakeModel(
        groups=[(0, 1)],
        names={(0, 1): "load"},
        entities={(0, 1): [7]},
        coords={7: (0.5, 0.5, 0.0)})
    _install_fake_gmsh(monkeypatch, model)
    # 单元内判定放行, 距离膨胀到 100 → 必须被 too_far 拒绝
    monkeypatch.setattr(stress_mod, "point_in_element", lambda *a, **k: 0)
    monkeypatch.setattr(np.linalg, "norm",
                        lambda *a, **k: np.full(len(a[0]), 100.0))
    result = isrc.physical_point_from_geo(
        _geo_file(tmp_path), "load", _square_mesh())
    assert result[3] == "too_far"


# ═══════════════════════════════════════════════════════════════
# resolve_spec_overrides
# ═══════════════════════════════════════════════════════════════

def test_resolve_spec_missing_mesh_file_fatal(tmp_path):
    """.spec 指定不存在的网格 → CliError(1) 带目录上下文."""
    spec = tmp_path / "m.spec"
    spec.write_text("mesh = nope.msh\n", encoding="utf-8")
    with pytest.raises(CliError) as exc:
        isrc.resolve_spec_overrides(str(spec), AnalysisConfig())
    assert exc.value.exit_code == 1
    assert "指定的网格文件不存在: " in str(exc.value)
    assert ".spec 目录" in str(exc.value)


def test_resolve_spec_absolute_mesh_path(tmp_path):
    """绝对路径 mesh → 直接采用 (不以 .spec 目录拼相对)."""
    target = tmp_path / "real.msh"
    target.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n",
                      encoding="utf-8")
    spec = tmp_path / "m.spec"
    spec.write_text(f"mesh = {target}\n", encoding="utf-8")
    fp = isrc.resolve_spec_overrides(str(spec), AnalysisConfig())
    assert fp == str(target)


def test_resolve_spec_unknown_key_warns(capsys, tmp_path):
    """未知键 → WARN 列出可用键 (曾静默忽略导致载荷不生效)."""
    target = tmp_path / "real.msh"
    target.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n",
                      encoding="utf-8")
    spec = tmp_path / "m.spec"
    spec.write_text(f"mesh = {target}\nbogus_key = 1\n", encoding="utf-8")
    isrc.resolve_spec_overrides(str(spec), AnalysisConfig())
    out = capsys.readouterr().out
    assert "[WARN] .spec 键 'bogus_key' 不被识别" in out
    assert "'fix'" in out and "'mesh'" in out   # 可用键列表


# ═══════════════════════════════════════════════════════════════
# _resolve_geo_lc: lc 覆盖与临时副本
# ═══════════════════════════════════════════════════════════════

def test_resolve_geo_lc_no_lc_line_warns(monkeypatch, tmp_path, capsys):
    """.geo 无 lc 赋值 + 请求覆盖 → WARN (曾静默假装修改)."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    geo = tmp_path / "m.geo"
    geo.write_text('SetFactory("OpenCASCADE");\n', encoding="utf-8")
    fp, tmp = isrc._resolve_geo_lc(str(geo), AnalysisConfig(lc=0.5), None)
    assert fp == str(geo) and tmp is None
    assert "未找到 'lc' 赋值行" in capsys.readouterr().out


def test_resolve_geo_lc_no_lc_silent_in_batch(monkeypatch, tmp_path, capsys):
    """无 lc 且批处理不请求覆盖 → 静默 (无意义警告)."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    geo = tmp_path / "m.geo"
    geo.write_text('SetFactory("OpenCASCADE");\n', encoding="utf-8")
    fp, tmp = isrc._resolve_geo_lc(str(geo), AnalysisConfig(), None)
    assert fp == str(geo) and tmp is None
    assert "WARN" not in capsys.readouterr().out


def test_resolve_geo_lc_replace_first_only(tmp_path, capsys):
    """多个 lc 赋值 → 只替换第一个, 其余保留 + WARN."""
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.5;\nlc = 0.1;\n", encoding="utf-8")
    fp, tmp = isrc._resolve_geo_lc(str(geo), AnalysisConfig(lc=0.2), None)
    assert fp == tmp and tmp is not None
    text = open(tmp, encoding="utf-8").read()
    assert "lc = 0.2;" in text and "lc = 0.1;" in text
    assert "2 个 'lc' 赋值" in capsys.readouterr().out
    assert geo.read_text(encoding="utf-8") == "lc = 0.5;\nlc = 0.1;\n"
    # 原始文件未被修改; 临时副本 atexit 清理


def test_resolve_geo_lc_unchanged_returns_original(tmp_path):
    """精确相等视为未更改 → 返回原路径无副本 (曾 1e-15 容差吞微尺度)."""
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.5;\n", encoding="utf-8")
    fp, tmp = isrc._resolve_geo_lc(str(geo), AnalysisConfig(lc=0.5), None)
    assert fp == str(geo) and tmp is None


def test_resolve_geo_lc_interactive_empty_keeps_current(monkeypatch,
                                                        tmp_path):
    """交互回车 (空输入) → 保持当前 lc, 不创建副本."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.5;\n", encoding="utf-8")
    fp, tmp = isrc._resolve_geo_lc(str(geo), AnalysisConfig(),
                                   lambda prompt: "")
    assert fp == str(geo) and tmp is None


# ═══════════════════════════════════════════════════════════════
# resolve_geo
# ═══════════════════════════════════════════════════════════════

def test_resolve_geo_default_ask_and_publish(tmp_path, monkeypatch):
    """ask 缺省 → cli 默认; 批处理跳过提问; 返回 (msh, import, 源 geo)."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.5;\n", encoding="utf-8")
    monkeypatch.setattr(
        isrc, "generate_geo_with_topology",
        lambda *a, **k: (str(tmp_path / "m.msh"), "imported"))
    msh, imp, src = isrc.resolve_geo(str(geo), AnalysisConfig())
    assert msh.endswith("m.msh") and imp == "imported"
    assert src == str(geo)


def test_resolve_geo_temp_copy_cleaned_on_failure(tmp_path, monkeypatch):
    """lc 覆盖生成临时副本 → 生成失败时 finally 清理副本."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.5;\n", encoding="utf-8")
    monkeypatch.setattr(isrc, "generate_geo_with_topology",
                        lambda *a, **k: (None, None))
    with pytest.raises(CliError, match="Gmsh 网格生成失败"):
        isrc.resolve_geo(str(geo), AnalysisConfig(lc=0.2))
    leftovers = [p for p in tmp_path.iterdir() if p.name != "m.geo"]
    assert leftovers == []   # 临时副本已删除


def test_resolve_geo_cleanup_unlink_failure_swallowed(tmp_path,
                                                      monkeypatch):
    """finally 清理副本失败 (OSError) → 吞掉, CliError 正常冒出."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    geo = tmp_path / "m.geo"
    geo.write_text("lc = 0.5;\n", encoding="utf-8")
    monkeypatch.setattr(isrc, "generate_geo_with_topology",
                        lambda *a, **k: (None, None))
    real_unlink = os.unlink
    def _failing_unlink(path, *a, **k):
        if ".geo" in str(path) and "m.geo" not in str(path):
            raise OSError("locked")
        return real_unlink(path, *a, **k)
    monkeypatch.setattr(os, "unlink", _failing_unlink)
    with pytest.raises(CliError, match="Gmsh 网格生成失败"):
        isrc.resolve_geo(str(geo), AnalysisConfig(lc=0.2))


# ═══════════════════════════════════════════════════════════════
# resolve_txt: 手写 .geo 保护 / 失败路径
# ═══════════════════════════════════════════════════════════════

def _patch_txt_chain(monkeypatch, tmp_path, geo_spec_mod=None):
    """接管 scripts.geo_spec 的 parse_spec/generate_geo 与生成链路."""
    if geo_spec_mod is None:
        import scripts.geo_spec as geo_spec_mod
    monkeypatch.setattr(geo_spec_mod, "parse_spec", lambda fp: {})
    monkeypatch.setattr(geo_spec_mod, "generate_geo",
                        lambda spec, path, quad=False: None)
    return geo_spec_mod


def test_resolve_txt_handwritten_geo_preserved(tmp_path, monkeypatch,
                                               capsys):
    """同名 .geo 无生成标记 → 生成到临时副本, 原始文件不碰."""
    txt = tmp_path / "m.txt"
    txt.write_text("x", encoding="utf-8")
    geo = tmp_path / "m.geo"
    geo.write_text("// 手写几何\nlc = 0.5;\n", encoding="utf-8")
    _patch_txt_chain(monkeypatch, tmp_path)
    monkeypatch.setattr(
        isrc, "generate_geo_with_topology",
        lambda *a, **k: (str(tmp_path / "m.msh"), "imported"))
    msh, imp, src = isrc.resolve_txt(str(txt), AnalysisConfig())
    out = capsys.readouterr().out
    assert "[INFO]" in out and "手写 .geo" in out
    assert geo.read_text(encoding="utf-8").startswith("// 手写几何")
    assert src != str(geo)      # 源指向临时副本
    assert ".fem2d-txt-" in os.path.basename(src)
    assert msh.endswith("m.msh") and imp == "imported"


def test_resolve_txt_unreadable_geo_treated_as_handwritten(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    """生成标记读取失败 (OSError) → 按手写处理, 不崩溃."""
    import builtins
    txt = tmp_path / "m.txt"
    txt.write_text("x", encoding="utf-8")
    geo = tmp_path / "m.geo"
    geo.write_text("// Auto-generated by geo_spec.py\nlc = 0.1;\n",
                   encoding="utf-8")
    _patch_txt_chain(monkeypatch, tmp_path)
    monkeypatch.setattr(
        isrc, "generate_geo_with_topology",
        lambda *a, **k: (str(tmp_path / "m.msh"), "imported"))
    real_open = builtins.open
    def _locked_open(path, *a, **k):
        mode = a[0] if a else "r"
        if str(path) == str(geo) and "r" in mode and "b" not in mode:
            raise OSError("locked")
        return real_open(path, *a, **k)
    monkeypatch.setattr(builtins, "open", _locked_open)
    msh, imp, src = isrc.resolve_txt(str(txt), AnalysisConfig())
    out = capsys.readouterr().out
    assert "[INFO]" in out and "手写 .geo" in out
    assert src != str(geo) and msh.endswith("m.msh") and imp == "imported"


def test_resolve_txt_generated_geo_overwrite_warns(tmp_path, monkeypatch,
                                                   capsys):
    """同名 .geo 是生成物 → 覆盖并 WARN (无真实损失)."""
    txt = tmp_path / "m.txt"
    txt.write_text("x", encoding="utf-8")
    geo = tmp_path / "m.geo"
    geo.write_text("// Auto-generated by geo_spec.py\nlc = 0.1;\n",
                   encoding="utf-8")
    _patch_txt_chain(monkeypatch, tmp_path)
    monkeypatch.setattr(
        isrc, "generate_geo_with_topology",
        lambda *a, **k: (str(tmp_path / "m.msh"), "imported"))
    msh, imp, src = isrc.resolve_txt(str(txt), AnalysisConfig())
    assert "[WARN] 覆盖已生成的" in capsys.readouterr().out
    assert src == str(geo)      # 直接覆盖生成物, 不建临时副本


def test_resolve_txt_generate_failure_raises_clierror(tmp_path,
                                                      monkeypatch):
    """generate_geo 抛 ValueError → CliError 带几何生成上下文."""
    txt = tmp_path / "m.txt"
    txt.write_text("x", encoding="utf-8")
    import scripts.geo_spec as geo_spec_mod
    monkeypatch.setattr(geo_spec_mod, "parse_spec", lambda fp: {})
    monkeypatch.setattr(
        geo_spec_mod, "generate_geo",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("几何错误")))
    with pytest.raises(CliError) as exc:
        isrc.resolve_txt(str(txt), AnalysisConfig())
    assert "几何生成失败: 几何错误" in str(exc.value)


def test_resolve_txt_gmsh_failure_raises_clierror(tmp_path, monkeypatch):
    """生成物缺失 (run_gmsh 失败) → CliError 退出 1."""
    txt = tmp_path / "m.txt"
    txt.write_text("x", encoding="utf-8")
    _patch_txt_chain(monkeypatch, tmp_path)
    monkeypatch.setattr(isrc, "generate_geo_with_topology",
                        lambda *a, **k: (None, None))
    with pytest.raises(CliError, match="Gmsh 网格生成失败"):
        isrc.resolve_txt(str(txt), AnalysisConfig())


# ═══════════════════════════════════════════════════════════════
# resolve_input_file: 参数 WARN 与导入失败
# ═══════════════════════════════════════════════════════════════

def test_resolve_input_txt_lc_ignored_warns(tmp_path, monkeypatch, capsys):
    """.txt 输入 + --lc → WARN 并说明 (.txt 用自身网格行)."""
    txt = tmp_path / "m.txt"
    txt.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        isrc, "resolve_txt",
        lambda fp, config: (str(tmp_path / "m.msh"), "imp", str(tmp_path)))
    r = isrc.resolve_input_file(str(txt), AnalysisConfig(lc=0.1))
    assert r.quad_applied is False and r.geo_config_applied is False
    assert "--lc 只对 .geo 输入生效" in capsys.readouterr().out
    assert r.fp.endswith("m.msh")


def test_resolve_input_msh_quad_and_lc_warn(tmp_path, monkeypatch, capsys):
    """.msh 直接输入 + --quad/--lc → 两个 WARN 不静默."""
    msh = tmp_path / "m.msh"
    msh.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    monkeypatch.setattr(gmsh_adapter_mod, "import_msh",
                        lambda fp, plane_type="stress": "imported")
    r = isrc.resolve_input_file(str(msh), AnalysisConfig(quad=True, lc=0.1))
    out = capsys.readouterr().out
    assert "--quad 只对 .geo/.txt 网格生成生效" in out
    assert "--lc 只对 .geo 输入生效, .msh 直接输入时忽略" in out
    assert r.gmsh_import == "imported" and r.quad_applied is False


def test_resolve_input_msh_import_none_raises(tmp_path, monkeypatch):
    """.msh 导入返回空 → CliError(1) 带文件路径."""
    msh = tmp_path / "m.msh"
    msh.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    monkeypatch.setattr(gmsh_adapter_mod, "import_msh",
                        lambda fp, plane_type="stress": None)
    with pytest.raises(CliError) as exc:
        isrc.resolve_input_file(str(msh), AnalysisConfig())
    assert exc.value.exit_code == 1
    assert ".msh 导入失败" in str(exc.value)


# ═══════════════════════════════════════════════════════════════
# resolve_geo: .msh 复用 (geo 未修改时跳过 gmsh 网格化)
# ═══════════════════════════════════════════════════════════════

def _patch_generate(monkeypatch):
    """mock generate_geo_with_topology, 记录调用次数."""
    calls = {"n": 0}
    def _fake_gen(geo_path, *, quad=False, output_path=None,
                  plane_type="stress"):
        calls["n"] += 1
        return "gen.msh", "gen-import"
    monkeypatch.setattr(isrc, "generate_geo_with_topology", _fake_gen)
    return calls


def test_resolve_geo_reuses_fresh_msh(tmp_path, monkeypatch, capsys):
    """geo 未修改且已有 .msh → 复用跳过网格化 (交付识别提速).

    判别: generate_geo_with_topology 调用 0 次, 返回复用路径 + 打印标记.
    """
    geo = tmp_path / "m.geo"
    msh = tmp_path / "m.msh"
    geo.write_text('SetFactory("OpenCASCADE");\n', encoding="utf-8")
    msh.write_text("x", encoding="utf-8")
    t = os.path.getmtime(str(msh)) - 10
    os.utime(str(geo), (t, t))               # geo 旧于 msh
    calls = _patch_generate(monkeypatch)
    seen = {}
    def _fake_import(msh_path, require_quads=False, plane_type="stress"):
        seen["path"] = msh_path
        return "reused-import"
    monkeypatch.setattr(gmsh_adapter_mod, "import_msh", _fake_import)
    msh_path, imp, src = isrc.resolve_geo(str(geo), AnalysisConfig())
    assert calls["n"] == 0, "复用命中时不应调用 generate_geo_with_topology"
    assert msh_path == str(msh) and imp == "reused-import"
    assert seen["path"] == str(msh)
    assert "复用已有网格" in capsys.readouterr().out


def test_resolve_geo_regenerates_when_geo_newer(tmp_path, monkeypatch, capsys):
    """geo 新于 msh → 网格可能过期, 必须重新网格化."""
    geo = tmp_path / "m.geo"
    msh = tmp_path / "m.msh"
    geo.write_text('SetFactory("OpenCASCADE");\n', encoding="utf-8")
    msh.write_text("x", encoding="utf-8")
    t = os.path.getmtime(str(geo)) - 10
    os.utime(str(msh), (t, t))               # msh 旧于 geo
    calls = _patch_generate(monkeypatch)
    boom = AssertionError("复用不应命中")
    monkeypatch.setattr(gmsh_adapter_mod, "import_msh",
                        lambda *a, **k: (_ for _ in ()).throw(boom))
    msh_path, imp, _ = isrc.resolve_geo(str(geo), AnalysisConfig())
    assert calls["n"] == 1
    assert msh_path == "gen.msh" and imp == "gen-import"
    assert "复用已有网格" not in capsys.readouterr().out


def test_resolve_geo_reuses_foreign_msh(tmp_path, monkeypatch, capsys):
    """交付包 msh 均为外来无标记文件 — 无 lc 覆盖时必须同样复用."""
    geo = tmp_path / "m.geo"
    msh = tmp_path / "m.msh"
    geo.write_text('lc = 0.1;\nSetFactory("OpenCASCADE");\n',
                   encoding="utf-8")
    msh.write_text("x", encoding="utf-8")
    t = os.path.getmtime(str(msh)) - 10
    os.utime(str(geo), (t, t))
    calls = _patch_generate(monkeypatch)
    monkeypatch.setattr(gmsh_adapter_mod, "import_msh",
                        lambda *a, **k: "reused-import")
    msh_path, imp, _ = isrc.resolve_geo(str(geo), AnalysisConfig())
    assert calls["n"] == 0 and msh_path == str(msh)
    capsys.readouterr().out


def test_resolve_geo_lc_override_skips_reuse(tmp_path, monkeypatch, capsys):
    """显式 --lc 覆盖密度 → 旧 msh 不复用 (否则结果用错网格密度)."""
    geo = tmp_path / "m.geo"
    msh = tmp_path / "m.msh"
    geo.write_text('lc = 0.1;\nSetFactory("OpenCASCADE");\n',
                   encoding="utf-8")
    msh.write_text("x", encoding="utf-8")
    t = os.path.getmtime(str(msh)) - 10
    os.utime(str(geo), (t, t))
    calls = _patch_generate(monkeypatch)
    boom = AssertionError("复用不应命中")
    monkeypatch.setattr(gmsh_adapter_mod, "import_msh",
                        lambda *a, **k: (_ for _ in ()).throw(boom))
    msh_path, imp, _ = isrc.resolve_geo(str(geo), AnalysisConfig(lc=0.5))
    assert calls["n"] == 1 and msh_path == "gen.msh"
    assert "复用已有网格" not in capsys.readouterr().out


def test_resolve_geo_fallback_when_reuse_fails(tmp_path, monkeypatch, capsys):
    """msh 损坏/读回失败 → WARN + fallback 重新网格化."""
    geo = tmp_path / "m.geo"
    msh = tmp_path / "m.msh"
    geo.write_text('SetFactory("OpenCASCADE");\n', encoding="utf-8")
    msh.write_text("x", encoding="utf-8")
    t = os.path.getmtime(str(msh)) - 10
    os.utime(str(geo), (t, t))
    calls = _patch_generate(monkeypatch)
    monkeypatch.setattr(gmsh_adapter_mod, "import_msh",
                        lambda *a, **k: (_ for _ in ()).throw(
                            ValueError("无法打开")))
    msh_path, imp, _ = isrc.resolve_geo(str(geo), AnalysisConfig())
    assert calls["n"] == 1 and msh_path == "gen.msh"
    out = capsys.readouterr().out
    assert "复用" in out and "重新网格化" in out
