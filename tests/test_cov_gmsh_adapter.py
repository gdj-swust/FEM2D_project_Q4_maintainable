"""覆盖轮 C1 — gmsh_adapter.py 缺口行 (48 行).

策略: 绝大多数缺口是防御/错误分支, 用轻量假 gmsh API 对象触发 —
CI 无 gmsh 安装也全绿; 仅 2 个真实 API 路径测试走子进程 (coverage 与
gmsh C 扩展在 Windows 上互斥 — 主进程永不 import gmsh), 无 gmsh
环境 skip 而非失败.
"""
import importlib
import io
import contextlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import fem2d.gmsh_adapter as GA
from fem2d.gmsh_adapter import (
    GmshTopologyError, GmshUnavailableError,
    _load_gmsh_module, _safe_geo_source, read_geo_curve_groups,
    _entity_type, _element_properties, _map_node_tags,
    _physical_node_ids, _surface_elements_for_entity, _physical_name,
    _fallback_point_node_ids, _extract_point_region,
    _surface_boundary_occurrences, _extract_mesh,
    _file_declares_physical_names, import_msh, _write_atomic,
)


def _run_gmsh_subprocess(body: str) -> dict:
    """子进程执行真实 gmsh 脚本 — 返回 JSON 结果 dict.

    coverage 运行时主进程 import gmsh 会触发 numpy 重复加载崩溃
    (gmsh.py 自加载 C 扩展), 故真实 API 路径全部在子进程执行;
    无 gmsh 环境打印哨兵 → 主进程 skip.
    """
    code = (
        "import json, sys\n"
        "try:\n"
        "    import gmsh\n"
        "except (ImportError, OSError):\n"
        "    print('__GMSH_UNAVAILABLE__')\n"
        "    sys.exit(0)\n"
        + body + "\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        timeout=300, cwd=".")
    if "__GMSH_UNAVAILABLE__" in proc.stdout:
        pytest.skip("gmsh Python API not available")
    marker = "__JSON__"
    if marker in proc.stdout:
        return json.loads(proc.stdout.split(marker, 1)[1])
    # gmsh 进度 Info 输出会污染 stdout, 脚本必须用 __JSON__ 标记
    assert proc.returncode == 0, proc.stdout + proc.stderr
    raise AssertionError(f"no {marker} marker in subprocess output")


def _props(name="TRI3", dim=2, order=1, n_nodes=3, primary=3):
    """构造 getElementProperties 兼容返回值 (gmsh 属性元组)."""
    return (name, dim, order, n_nodes, primary, primary)


class _ElemProps:
    def __init__(self, props):
        self._props = props

    def getElementProperties(self, etype):
        return self._props


# ── 模块加载 / 临时文件防御分支 ─────────────────────────────────────────────

def test_load_gmsh_module_import_failure(monkeypatch):
    """importlib 抛 ImportError → GmshUnavailableError (非裸异常)."""
    def boom(name):
        raise ImportError("no gmsh")
    monkeypatch.setattr(GA.importlib, "import_module", boom)
    with pytest.raises(GmshUnavailableError, match="not available"):
        _load_gmsh_module()


def test_safe_geo_source_write_failure_cleans(monkeypatch, tmp_path):
    """源文件写入失败 → 清理半成品临时文件后原异常冒出."""
    geo = tmp_path / "g.geo"
    geo.write_text("Point(1) = {0,0,0,1};\n", encoding="utf-8")

    # 强制 sanitize 改写 → 走 mkstemp 临时文件路径 (原样内容提前返回)
    monkeypatch.setattr(
        "scripts.gmsh_runner.sanitize_geo_source",
        lambda src, geo_path=None: src + "// rewritten\n")

    # write 阶段失败: with 上下文会关闭 fd, unlink 清理分支才可达
    # (fdopen/__enter__ 阶段失败在 Windows 上 fd 未关, unlink 被锁)
    orig_fdopen = os.fdopen
    def fake_fdopen(fd, mode, **kw):
        stream = orig_fdopen(fd, mode, **kw)
        def boom_write(data):
            raise OSError("disk full")
        stream.write = boom_write
        return stream
    monkeypatch.setattr(GA.os, "fdopen", fake_fdopen)
    with pytest.raises(OSError, match="disk full"):
        _safe_geo_source(str(geo))
    # mkstemp 产物已被清理
    assert not list(tmp_path.glob(".fem2d-gmsh-api-source-*"))


def test_read_geo_curve_groups_missing_file():
    """路径不存在 → None (调用方走文本解析回退, 不报错)."""
    assert read_geo_curve_groups("definitely_missing.geo") is None


def test_safe_geo_source_rejects_non_path():
    """非 str 路径 → TypeError — open(int) 会把 int 当 fd 打开并
    关闭 stdout (修复: 误传节点编号等 int 不再静默破坏调用进程)."""
    for bad in (123, 0, None, np.float64(1.5), b"x.geo"):
        with pytest.raises(TypeError, match="str/os.PathLike"):
            _safe_geo_source(bad)
    # 判别性: 修复前 open(123) 读取 fd 123 或关闭 fd — stdout 存活
    assert sys.stdout is not None and sys.stdout.fileno() >= 0


# ── 假 API 对象上的辅助函数 (真实 gmsh 无关) ────────────────────────────────

class _GetTypeOnly:
    def getType(self, dim, tag):
        return "Circle"


class _GetEntityType:
    def getEntityType(self, dim, tag):
        return "Line"


def test_entity_type_gettype_fallback():
    """老版 gmsh API 无 getEntityType → getType 兜底."""
    assert _entity_type(_GetTypeOnly(), 1, 7) == "Circle"
    assert _entity_type(_GetEntityType(), 1, 7) == "Line"


def test_element_properties_incomplete_raises():
    """属性元组 <6 项 (gmsh 契约) → GmshTopologyError."""
    mesh_api = _ElemProps(("TRI3", 2, 1, 3))
    with pytest.raises(GmshTopologyError, match="incomplete"):
        _element_properties(mesh_api, 2)


def test_element_properties_ok():
    """正常属性 → 类型化解析 (str/int 转换)."""
    mesh_api = _ElemProps(("TRI3", 2, 1, 3, 3, 3))
    assert _element_properties(mesh_api, 2) == ("TRI3", 2, 1, 3, 3)


def test_map_node_tags_missing_raises():
    """节点 tag 缺映射 (构造节点被剔出位移网格) → 亮错误."""
    with pytest.raises(GmshTopologyError, match="absent"):
        _map_node_tags(np.array([0, 5]), {0: 0, 1: 1}, "Elem")


def test_map_node_tags_ok():
    assert _map_node_tags(np.array([2, 0]), {2: 9, 0: 3}, "Elem") == [9, 3]


def test_physical_node_ids_no_getter_returns_empty():
    """老版 API 无 getNodesForPhysicalGroup → 空集 (不能崩)."""
    assert _physical_node_ids(_ElemProps(()), 1, 1, {}) == []


def test_surface_elements_missing_raises():
    """曲面元素 tag 不在位移网格映射 → GmshTopologyError."""
    class _Api:
        def getElements(self, dim, tag):
            return None, [np.array([7, 8])], None
    with pytest.raises(GmshTopologyError, match="absent"):
        _surface_elements_for_entity(_Api(), 1, {7: 0})


def test_surface_elements_ok():
    class _Api:
        def getElements(self, dim, tag):
            return None, [np.array([2, 5])], None
    assert _surface_elements_for_entity(_Api(), 1, {2: 7, 5: 9}) == [7, 9]


def test_physical_name_default_fallback():
    """无名物理组 → physical_{dim}_{tag} 派生名."""
    class _M:
        def getPhysicalName(self, dim, tag):
            return "   "
    assert _physical_name(_M(), 0, 3) == "physical_0_3"


# ── 物理点映射回退 / 区域提取 ───────────────────────────────────────────────

class _PointModel:
    def __init__(self, value):
        self._value = value

    def getValue(self, dim, tag, params):
        return self._value


def test_fallback_point_outside_aabb_rejected(capsys):
    """域外 Physical Point → WARN 拒绝 (返回空, 下游报错)."""
    coords = np.array([[0., 0.], [1., 0.], [0., 1.]])
    out = _fallback_point_node_ids(
        _PointModel([5.0, 5.0, 0.0]), coords, None, "CPS3",
        [1], "p_out")
    assert out == ()
    assert "拒绝映射" in capsys.readouterr().out


def test_fallback_point_inside_returns_nearest(capsys):
    """域内 Point → 回退最近位移节点 + WARN."""
    coords = np.array([[0., 0.], [1., 0.], [0., 1.]])
    out = _fallback_point_node_ids(
        _PointModel([0.9, 0.1, 0.0]), coords, None, "CPS3",
        [1], "p_in")
    assert out == (1,)                      # 最近为节点 1 (1.0, 0.0)
    assert "回退到最近节点" in capsys.readouterr().out


def test_extract_point_region_fallback_branch():
    """构造节点为空 → 走最近节点回退 (line 424 分支)."""
    coords = np.array([[0., 0.], [1., 0.], [0., 1.]])
    region = _extract_point_region(
        _PointModel([0.0, 0.0, 0.0]), 1, (2,), "p1",
        (), coords, None, "CPS3")
    assert region.node_ids == (0,)
    assert region.name == "p1" and region.physical_tag == 1


def test_surface_boundary_occurrences_skip_no_elements():
    """活动曲面无位移元素 → 跳过 (不产生幽灵边界)."""
    class _Api:
        def getElements(self, dim, tag):
            return None, [], None
    occ = _surface_boundary_occurrences(
        _Api(), _PointModel([]), {}, {1, 2})
    assert occ == {}


# ── _extract_mesh 错误路径 (假 mesh_api) ───────────────────────────────────

def _fake_module(node_tags, coords, elements, props_map):
    class _Mesh:
        def getNodes(self):
            return node_tags, coords, []
        def getElements(self, dim, tag):
            et, tb, nb = elements
            return et, tb, nb
        def getElementProperties(self, etype):
            return props_map[etype]
    class _Model:
        mesh = _Mesh()
    class _Mod:
        model = _Model()
    return _Mod()


def test_extract_mesh_no_nodes():
    m = _fake_module([], np.empty((0, 3)), None, {})
    with pytest.raises(GmshTopologyError, match="no mesh nodes"):
        _extract_mesh(m)


def test_extract_mesh_inconsistent_lengths():
    m = _fake_module([1], np.empty((0, 3)), None, {})
    with pytest.raises(GmshTopologyError, match="inconsistent lengths"):
        _extract_mesh(m)


def test_extract_mesh_unsupported_element():
    """二阶单元 (order=2) → 拒绝 (只支持一阶 TRI3/QUAD4)."""
    m = _fake_module(
        [1, 2, 3], np.zeros((3, 3)), ([1], [np.array([1])],
                                      [np.zeros((1, 6), int)]),
        {1: _props("TRI6", 2, 2, 6, 3)})
    with pytest.raises(GmshTopologyError, match="Unsupported"):
        _extract_mesh(m)


def test_extract_mesh_tag_connectivity_mismatch():
    """元素 tag 块与连接块长度不一致 → 抛."""
    m = _fake_module(
        [1, 2, 3], np.zeros((3, 3)),
        ([2], [np.array([1])], [np.zeros((2, 3), int)]),
        {2: _props("TRI3", 2, 1, 3, 3)})
    with pytest.raises(GmshTopologyError, match="inconsistent tag"):
        _extract_mesh(m)


def test_extract_mesh_only_1d_elements():
    """只有一维 (边界) 元素 → 无位移单元 → 抛."""
    m = _fake_module(
        [1, 2], np.zeros((2, 3)),
        ([1], [np.array([1])], [np.zeros((1, 2), int)]),
        {1: _props("LINE2", 1, 1, 2, 2)})
    with pytest.raises(GmshTopologyError, match="no supported"):
        _extract_mesh(m)


def test_extract_mesh_mixed_topology():
    """三角+四边混合 → 抛 (FEM2D 要求单一同质拓扑)."""
    m = _fake_module(
        [1, 2, 3, 4], np.zeros((4, 3)),
        ([2, 3], [np.array([1]), np.array([1])],
         [np.array([[1, 2, 3]]), np.array([[1, 2, 3, 4]])]),
        {2: _props("TRI3", 2, 1, 3, 3), 3: _props("QUAD4", 2, 1, 4, 4)})
    with pytest.raises(GmshTopologyError, match="mixed triangle/quad"):
        _extract_mesh(m)


def test_extract_mesh_quad_required_but_triangle():
    """quad 模式要求下仍为三角 → 抛."""
    m = _fake_module(
        [1, 2, 3], np.zeros((3, 3)),
        ([2], [np.array([1])], [np.array([[1, 2, 3]])]),
        {2: _props("TRI3", 2, 1, 3, 3)})
    with pytest.raises(GmshTopologyError, match="Quad mode requested"):
        _extract_mesh(m, require_quads=True)


# ── _file_declares_physical_names (纯文件 IO, 无 gmsh) ─────────────────────

def test_file_declares_physical_names_true(tmp_path):
    msh = tmp_path / "p.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
                   "$PhysicalNames\n1\n0 1 point\n$EndPhysicalNames\n",
                   encoding="ascii")
    assert _file_declares_physical_names(str(msh)) is True


def test_file_declares_physical_names_none(tmp_path):
    msh = tmp_path / "p.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n", encoding="ascii")
    assert _file_declares_physical_names(str(msh)) is False


def test_file_declares_physical_names_oserror(monkeypatch, tmp_path):
    """读取失败 (IO 错误) → False 而非崩溃 (调用方仅告警)."""
    def boom(path, mode, **kw):
        raise OSError("locked")
    monkeypatch.setattr("builtins.open", boom)
    assert _file_declares_physical_names(str(tmp_path / "x.msh")) is False


# ── _write_atomic (假 gmsh 模块, 验证原子写语义) ────────────────────────────

class _FakeGmshModule:
    def __init__(self, creates_file=True):
        self.option = _Opt()
        self._creates = creates_file

    def write(self, path):
        if self._creates:
            with open(path, "w", encoding="ascii") as f:
                f.write("$MeshFormat\n")


class _Opt:
    def __init__(self):
        self.set_number = None

    def setNumber(self, key, val):
        self.set_number = (key, val)


def test_write_atomic_success(tmp_path):
    """原子写: 临时文件 → os.replace → 最终路径, 无临时残留."""
    mod = _FakeGmshModule(creates_file=True)
    out = _write_atomic(mod, str(tmp_path / "mesh.msh"))
    assert out == os.path.abspath(str(tmp_path / "mesh.msh"))
    assert os.path.isfile(out)
    assert mod.option.set_number == ("Mesh.SaveAll", 1)
    assert not list(tmp_path.glob(".fem2d-gmsh-api-*"))


def test_write_atomic_no_output_file(tmp_path):
    """gmsh 未产生非空输出 → GmshTopologyError (防静默)."""
    mod = _FakeGmshModule(creates_file=False)
    with pytest.raises(GmshTopologyError, match="non-empty"):
        _write_atomic(mod, str(tmp_path / "mesh.msh"))


# ── 真实 gmsh API 路径 (无 gmsh 环境 skip) ─────────────────────────────────

def test_read_geo_curve_groups_real_gmsh(tmp_path):
    """真实 gmsh: .geo 读取 + Physical Curve 组 + 临时文件清理."""
    geo = tmp_path / "g.geo"
    geo.write_text(
        "Point(1) = {0,0,0,1.0}; Point(2) = {1,0,0,1.0};\n"
        'Line(1) = {1,2};\nPhysical Curve("bottom", 1) = {1};\n',
        encoding="utf-8")
    result = _run_gmsh_subprocess(f"""
import glob, os, fem2d.gmsh_adapter as GA
groups = GA.read_geo_curve_groups({json.dumps(str(geo))})
tmp_left = bool(glob.glob(os.path.join(
    {json.dumps(str(tmp_path))}, '.fem2d-gmsh-api-source-*')))
print("__JSON__" + json.dumps({{"groups": groups, "tmp_left": tmp_left}}))""")
    assert result["groups"] is not None and "bottom" in result["groups"]
    # 临时源文件已清理 (read 成功路径 line 139)
    assert result["tmp_left"] is False


def test_import_msh_warns_lost_physical_names(tmp_path):
    """提取成功但物理组全丢 + 文件声明 $PhysicalNames → 防静默 WARN."""
    msh = tmp_path / "p.msh"
    msh.write_text("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
                   "$PhysicalNames\n1\n0 1 pt\n$EndPhysicalNames\n",
                   encoding="ascii")
    result = _run_gmsh_subprocess(f"""
import io, contextlib, fem2d.gmsh_adapter as GA
class _Res:
    regions = type("R", (), {{"curves": [], "surfaces": []}})()
    output_path = None
GA._extract_mesh = lambda *a, **k: _Res()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    GA.import_msh({json.dumps(str(msh))})
print("__JSON__" + json.dumps({{"has_warn": "WARN" in buf.getvalue()}}))""")
    assert result["has_warn"] is True


def test_import_msh_missing_file():
    """缺失 .msh → FileNotFoundError (前置检查, 无需 gmsh)."""
    with pytest.raises(FileNotFoundError):
        import_msh("definitely_missing.msh")
