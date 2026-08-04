"""契约表-代码一致性探针 (复查轮可复用审计工具).

逐条实测 docs/api_contract.md 各行的"应有错误行为"列:
  probe(name, fn, expect) — expect 为异常类型或 None (不应抛)。
输出 PASS/FAIL, 任何 FAIL → 退出码 1。

覆盖对照 (docs/api_contract.md 契约行 ↔ 探针数, 2026-08-04):
  A0/A1/A2  Mesh 构造器/节点 API/结构 API   41 (全部行)
  B         求解与 BC                      19 (全部行)
  C         载荷                           10 (全部行)
  D         应力与误差                     15 (全部行)
  E         输入链 (无 gmsh 依赖路径)      12 (resolve_geo/resolve_txt/
                                            generate_geo_with_topology 需
                                            真实 Gmsh, 不在本探针覆盖内)
  F         材料与单元注册                  9 (全部行)
  G         边界                           10 (契约 6 行全覆盖)
  G2        识别器注册表与插件接口         7 (阶段 2/3 新增契约)
  H         配置与质量                     11 (全部行)
  I         装配                            2 (契约 2 行全覆盖)
  合计: 136 项探针 (可用 AST 统计 probe() 调用数核对)。
每组的"全部行"以契约表行数为准; 行内无具体误用错误声明的
(如 I 组"非对称内核 → RuntimeError") 以合法输入不抛为探针内容。
"""
import os
import sys
import tempfile

import numpy as np

# 脚本位于 scripts/ 下 — 审计必须针对本项目代码。editable install 指向
# 其他 worktree 时 sys.path 无 cwd, `python scripts/xxx.py` 会 import 到
# 外部 fem2d 副本 (曾静默测到旧实现, 数据失真)。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import fem2d as F
from fem2d.bc import apply_elimination, apply_penalty
from fem2d.boundary import (
    build_boundary_segments,
    default_registry,
    register_detector,
)
from fem2d.boundary.detectors import Detector
from fem2d.boundary.plugins.circle_label import (
    NativeCircleLabelDetector,
)
from fem2d.config import AnalysisConfig
from fem2d.error_est import (
    element_refinement_indicator,
    estimate as estimate_error,
)
from fem2d.errors import CliError
from fem2d.gmsh_adapter import generate_from_geo, import_msh
from fem2d.input_source import resolve_input_file, resolve_spec_overrides
from fem2d.loads_core import parse_traction, parse_vec2
from fem2d.material import D_matrix, von_mises
from fem2d.mesh import Mesh
from fem2d.patch_test import run_patch_test
from fem2d.preprocess import (
    MeshValidationError,
    parse_geo_fem_config,
    parse_spec_config,
    read_geo_groups,
    validate_mesh,
)
from fem2d.regions import RegionRegistry
from fem2d.solver import estimate_condition, solve
from fem2d.spr import spr_recovery
from fem2d.stress import (
    compute_stresses,
    nodal_L2_projection,
    nodal_average,
    nodal_weighted,
    point_in_element,
    principal_stresses,
    stress_at_point,
)

FAILS = []


def _expect_label(expect):
    """期望异常的可读名 — 元组 (IndexError, ValueError) 曾用
    expect.__name__ 直接崩溃 (AttributeError), 审计工具自身不能炸."""
    if expect is None:
        return "no error"
    if isinstance(expect, tuple):
        return "/".join(e.__name__ for e in expect)
    return expect.__name__


def probe(name, fn, expect):
    try:
        result = fn()
        if expect is None:
            print(f"  PASS {name}")
            return result
        FAILS.append(f"{name}: NO ERROR (expected {_expect_label(expect)})")
        print(f"  FAIL {name}: NO ERROR (expected {_expect_label(expect)})")
    except Exception as exc:  # noqa: BLE001 — 审计工具
        if expect is not None and isinstance(exc, expect):
            print(f"  PASS {name}: {type(exc).__name__}")
        else:
            FAILS.append(f"{name}: {type(exc).__name__} (expected "
                         f"{_expect_label(expect)}): {exc}")
            print(f"  FAIL {name}: {type(exc).__name__} (expected "
                  f"{_expect_label(expect)}): {str(exc)[:80]}")
    return None


NODES = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])
TRI_NODES = np.array([[0., 0.], [1., 0.], [0., 1.]])


def _mesh():
    return Mesh(NODES, ELEMS, E=1e6, nu=0.3, thickness=1.0)


def _tri():
    return Mesh(TRI_NODES, np.array([[0, 1, 2]]), E=1e6, nu=0.3, thickness=1.0)


def _solved():
    m = _mesh()
    for i in range(4):
        m.fix_node(i, "both", 0.0)
    m.add_force(2, 1.0, 0.0)
    return m, solve(m, verbose=False)


# E 组误用探针需要真实坏文件 (解析器按行报错). 系统临时目录, 审计脚本
# 运行频率低, 留待 OS 清理 (Windows 上 read_geo_groups 的 Gmsh 句柄
# 会锁文件, 主动删除反而抛错).
_PROBE_TMP = tempfile.mkdtemp(prefix="fem2d_probe_")
_BAD_SPEC = os.path.join(_PROBE_TMP, "bad.spec")
with open(_BAD_SPEC, "w", encoding="utf-8") as _fh:
    _fh.write("E = 210e9\nmesh =\n")
_BAD_GEO = os.path.join(_PROBE_TMP, "bad.geo")
with open(_BAD_GEO, "w", encoding="utf-8") as _fh:
    _fh.write("@FEM:pressure=\n")

# G 组共用: 边界段缓存 + 歧义模糊名探针 (4 条直边都含"直边"标签).
_SEGS = build_boundary_segments(_mesh())


print("== A0 构造器 ==")
probe("nodes scalar", lambda: Mesh(5, ELEMS), ValueError)
probe("nodes 1-D", lambda: Mesh(np.zeros(4), ELEMS), ValueError)
probe("nodes NaN", lambda: Mesh(np.full((2, 2), np.nan), ELEMS), ValueError)
probe("nodes empty", lambda: Mesh(np.empty((0, 2)), ELEMS), ValueError)
probe("elements scalar", lambda: Mesh(NODES, 5), ValueError)
probe("elements shape", lambda: Mesh(NODES, np.ones((2, 5))), ValueError)
probe("elements float idx", lambda: Mesh(NODES, np.array([[0.5, 1, 2]])), ValueError)
probe("elements neg idx", lambda: Mesh(NODES, np.array([[-1, 1, 2]])), ValueError)
probe("elements dup", lambda: Mesh(NODES, np.array([[0, 1, 3], [0, 1, 3]])), ValueError)
probe("E non-numeric", lambda: Mesh(NODES, ELEMS, E="abc"), TypeError)
probe("E NaN", lambda: Mesh(NODES, ELEMS, E=float("nan")), ValueError)
probe("nu bounds", lambda: Mesh(NODES, ELEMS, nu=0.6).validate_state(), ValueError)
probe("t zero", lambda: Mesh(NODES, ELEMS, thickness=0.0), ValueError)
probe("fixed bool mask", lambda: Mesh(NODES, ELEMS, fixed_dofs=np.array([True] * 8)), TypeError)
probe("fixed float nonint", lambda: Mesh(NODES, ELEMS, fixed_dofs=[0.5]), ValueError)
probe("fixed range", lambda: Mesh(NODES, ELEMS, fixed_dofs=[8]), ValueError)
probe("prescribed key not fixed", lambda: Mesh(NODES, ELEMS, prescribed_vals={0: 0.1}), ValueError)
probe("body 3 comp (validate_state)", lambda: Mesh(NODES, ELEMS, body_force=(1., 2., 3.)).validate_state(), ValueError)
probe("body scalar (validate_state)", lambda: Mesh(NODES, ELEMS, body_force=5.0).validate_state(), ValueError)
probe("elem_type unknown", lambda: Mesh(NODES, ELEMS, elem_type="BOGUS"), ValueError)

print("== A1 节点 API ==")
probe("fix_node bool nid", lambda: _mesh().fix_node(True, "both"), TypeError)
probe("fix_node range", lambda: _mesh().fix_node(9, "both"), ValueError)
probe("fix_node dof bad", lambda: _mesh().fix_node(0, "z"), ValueError)
probe("fix_node value str", lambda: _mesh().fix_node(0, "both", "abc"), TypeError)
probe("fix_node value nan", lambda: _mesh().fix_node(0, "both", float("nan")), ValueError)
probe("fix_nodes_func scalar", lambda: _mesh().fix_nodes_func(5, 0.0), ValueError)
probe("fix_nodes_func range", lambda: _mesh().fix_nodes_func([9], 0.0), ValueError)
probe("fix_nodes_func 3comp", lambda: _mesh().fix_nodes_func([0], lambda x, y: (1., 2., 3.)), ValueError)
probe("fix_nodes_func nan", lambda: _mesh().fix_nodes_func([0], lambda x, y: (float("nan"), 0.)), ValueError)
probe("add_force range", lambda: _mesh().add_force(-1), ValueError)
probe("add_force nan", lambda: _mesh().add_force(0, float("nan"), 0.0), ValueError)
probe("add_force str", lambda: _mesh().add_force(0, "abc", 0.0), TypeError)
probe("add_traction interior", lambda: _mesh().add_traction(0, 3, 1e6, 0.0), ValueError)
probe("add_pressure nan", lambda: _mesh().add_pressure(0, 1, float("nan")), ValueError)
probe("outward_normal internal", lambda: _mesh().boundary_outward_normal(0, 3), ValueError)

print("== A2 结构 API ==")
m = _mesh()
probe("replace_nodes shape", lambda: m.replace_nodes(NODES[:3]), ValueError)
probe("replace_nodes nan", lambda: m.replace_nodes(np.full((4, 2), np.nan)), ValueError)
probe("replace_elements shape", lambda: m.replace_elements(np.ones((3, 3))), ValueError)
probe("replace_elements float", lambda: m.replace_elements(np.array([[0.5, 1., 3.]])), ValueError)
probe("nodes_on_edge axis", lambda: m.nodes_on_edge("z", "min"), ValueError)
probe("nodes_on_edge tol", lambda: m.nodes_on_edge("x", "min", tol=-1.0), ValueError)

print("== B 求解与 BC ==")
probe("solve method bogus", lambda: solve(_solved()[0], method="bogus", verbose=False), ValueError)
probe("solve ls bogus", lambda: solve(_solved()[0], linear_solver="bogus", verbose=False), ValueError)
probe("solve non-mesh", lambda: solve({}), TypeError)
probe("solve underconstrained", lambda: solve(_tri(), verbose=False), RuntimeError)
probe("estimate_condition ok", lambda: estimate_condition(np.diag([1., 2., 3.])), None)
probe("estimate_condition K tuple", lambda: estimate_condition((1.0, 2.0)), ValueError)
probe("estimate_condition K scalar", lambda: estimate_condition(5.0), ValueError)
probe("estimate_condition K nonsquare", lambda: estimate_condition(np.ones((2, 3))), ValueError)
probe("estimate_condition method bogus", lambda: estimate_condition(np.eye(3), method="bogus"), ValueError)
probe("apply_elim F shape", lambda: apply_elimination(np.eye(4), np.zeros(3), [0, 1, 2], [3], [0.]), ValueError)
probe("apply_elim free bool", lambda: apply_elimination(np.eye(4), np.zeros(4), np.array([True] * 4), [0], [0.]), ValueError)
probe("apply_elim overlap", lambda: apply_elimination(np.eye(4), np.zeros(4), [0, 1], [1, 2, 3], [0., 0., 0.]), ValueError)
probe("apply_elim gap", lambda: apply_elimination(np.eye(4), np.zeros(4), [0, 2], [1], [0.]), ValueError)
probe("apply_elim dup free", lambda: apply_elimination(np.eye(4), np.zeros(4), [0, 0, 2], [1, 3], [0., 0.]), ValueError)
probe("apply_penalty nan penalty", lambda: apply_penalty(np.eye(4), np.zeros(4), [0], penalty=float("nan")), ValueError)
probe("apply_penalty bool", lambda: apply_penalty(np.eye(4), np.zeros(4), np.array([True] * 4)), ValueError)
probe("apply_penalty presc len", lambda: apply_penalty(np.eye(4), np.zeros(4), [0, 1], [0.0]), ValueError)
probe("estimate method bogus", lambda: estimate_error(_mesh(), {"stress": np.ones((2, 3))}, method="bogus"), ValueError)
probe("estimate missing key", lambda: estimate_error(_mesh(), {"u": np.zeros(8)}), ValueError)

print("== C 载荷 ==")
m2, res = _solved()
# 曾期望 (IndexError, ValueError) — 9.19.0 已修 n_dof 裸 IndexError
# (集中力越界写), 收紧为只接受 ValueError (契约: 带上下文的领域错误).
probe("assemble_loads n_dof mismatch", lambda: F.assemble_loads(m2, 3), ValueError)
probe("parse_vec2 1comp", lambda: parse_vec2("1e6"), ValueError)
probe("parse_vec2 3comp", lambda: parse_vec2("1,2,3"), ValueError)
probe("parse_vec2 nan", lambda: parse_vec2("nan,0"), ValueError)
probe("parse_vec2 inf", lambda: parse_vec2("1e999,0"), ValueError)
probe("parse_vec2 syntax", lambda: parse_vec2("1e6x,0"), ValueError)
probe("parse_vec2 nonstr", lambda: parse_vec2(None), ValueError)
probe("parse_traction nonstr", lambda: parse_traction(5), ValueError)
probe("parse_traction profile", lambda: parse_traction("r:1,2:q"), ValueError)
probe("parse_traction 4seg", lambda: parse_traction("a:1:2:3"), ValueError)

print("== D 应力与误差 ==")
probe("compute_stresses shape", lambda: compute_stresses(m2, np.zeros(3)), ValueError)
probe("compute_stresses nan", lambda: compute_stresses(m2, np.array([float("nan")] * 8)), ValueError)
probe("nodal_average shape", lambda: nodal_average(m2, np.ones((5, 3))), ValueError)
probe("nodal_weighted nan", lambda: nodal_weighted(m2, np.array([[np.nan, 1., 1.], [1., 1., 1.]])), ValueError)
probe("nodal_L2 ndim", lambda: nodal_L2_projection(m2, np.ones(4)), ValueError)
probe("principal shape", lambda: principal_stresses(np.ones((2, 2))), ValueError)
probe("principal nan", lambda: principal_stresses(np.array([[np.nan, 1., 0.]])), ValueError)
probe("stress_at_point mode", lambda: stress_at_point(m2, res, 0.5, 0.5, mode="bogus"), ValueError)
probe("stress_at_point missing key", lambda: stress_at_point(m2, {"u": np.zeros(8)}, 0.5, 0.5), ValueError)
probe("stress_at_point outside", lambda: stress_at_point(m2, res, 50.0, 50.0), ValueError)
probe("point_in_element outside", lambda: point_in_element(m2, 50.0, 50.0), None)
probe("spr_recovery shape", lambda: spr_recovery(m2, np.ones((5, 3))), ValueError)
probe("spr_recovery nan", lambda: spr_recovery(m2, np.array([[np.nan, 1., 1.], [1., 1., 1.]])), ValueError)
probe("refinement missing key", lambda: element_refinement_indicator(m2, {"u": np.zeros(8)}), ValueError)
probe("estimate non-dict", lambda: estimate_error(m2, 5), ValueError)

print("== F 材料 ==")
probe("D_matrix E neg", lambda: D_matrix(-1.0, 0.3), ValueError)
probe("D_matrix nu bound", lambda: D_matrix(1.0, 0.5), ValueError)
probe("D_matrix plane", lambda: D_matrix(1.0, 0.3, "bogus"), ValueError)
probe("D_matrix E str", lambda: D_matrix("abc", 0.3), TypeError)
probe("von_mises shape", lambda: von_mises(np.ones((2, 2))), ValueError)
probe("von_mises nan", lambda: von_mises(np.array([[np.nan, 1., 0.]])), ValueError)
probe("von_mises plane", lambda: von_mises(np.ones((2, 3)), plane_type="bogus"), ValueError)
probe("get_element_kernel unknown", lambda: F.get_element_kernel("BOGUS"), ValueError)
probe("register_element non-instance", lambda: F.register_element(object()), TypeError)

print("== H 配置 ==")
probe("config E neg", lambda: AnalysisConfig(E=-1.0), ValueError)
probe("config nu bound", lambda: AnalysisConfig(nu=0.9), ValueError)
probe("config plane", lambda: AnalysisConfig(plane="bogus"), ValueError)
probe("config solver", lambda: AnalysisConfig(linear_solver="bogus"), ValueError)
probe("config band trio", lambda: AnalysisConfig(band_min=0.0), ValueError)
probe("config band step", lambda: AnalysisConfig(band_min=0., band_max=1., band_step=0.), ValueError)
probe("config band nonint", lambda: AnalysisConfig(band_min=0., band_max=1., band_step=0.3), ValueError)
# H 组其余契约行: 网格质量无输入误用面 (空网格不可构造) → 合法调用不抛;
# patch test E/plane/elem_type 非法 → 领域错误.
probe("evaluate_mesh_quality ok", lambda: F.evaluate_mesh_quality(_mesh()), None)
probe("run_patch_test E neg", lambda: run_patch_test(E=-1.0, verbose=False), ValueError)
probe("run_patch_test plane bogus", lambda: run_patch_test(plane="bogus", verbose=False), ValueError)
probe("run_patch_test elem bogus", lambda: run_patch_test(elem_type="BOGUS", verbose=False), ValueError)

print("== E 输入链 (无 gmsh 依赖路径) ==")
probe("import_msh missing", lambda: import_msh("C:/no_such_dir_xyz/x.msh"), FileNotFoundError)
from fem2d.input_source import physical_point_from_geo
probe("physical_point no_geo", lambda: physical_point_from_geo(None, "p", _mesh())[3], None)
# .inp/.xyz 在扩展名分派即拒, 不触 ask 交互 (探针非交互运行安全).
probe("resolve_input_file .inp", lambda: resolve_input_file("x.inp", AnalysisConfig()), CliError)
probe("resolve_input_file .xyz", lambda: resolve_input_file("x.xyz", AnalysisConfig()), CliError)
probe("resolve_spec_overrides badfmt", lambda: resolve_spec_overrides(_BAD_SPEC, AnalysisConfig()), ValueError)
probe("generate_from_geo missing", lambda: generate_from_geo("C:/no_such_dir_xyz/x.geo"), FileNotFoundError)
probe("parse_spec_config badfmt", lambda: parse_spec_config(_BAD_SPEC), ValueError)
probe("parse_geo_fem_config empty", lambda: parse_geo_fem_config(_BAD_GEO), ValueError)
# 文件不存在 → None 是 read_geo_groups 的设计行为 (契约 E 行).
probe("read_geo_groups missing", lambda: read_geo_groups("C:/no_such_dir_xyz/x.geo"), None)
probe("validate_mesh neg idx", lambda: validate_mesh(NODES, np.array([[-1, 1, 2]])), MeshValidationError)
probe("validate_mesh empty", lambda: validate_mesh(NODES, np.empty((0, 3))), MeshValidationError)
probe("validate_mesh ok", lambda: validate_mesh(NODES, ELEMS), None)

print("== G 边界 ==")
probe("detect_boundaries ok", lambda: F.detect_boundaries(_mesh()), None)
probe("build_boundary_segments ok", lambda: build_boundary_segments(_mesh()), None)
# 诊断型 API: 返回诊断对象而非抛异常 (契约 G 行).
probe("validate_boundary_segments ok", lambda: F.validate_boundary_segments(_mesh(), _SEGS), None)
probe("describe_geometry ok", lambda: F.describe_geometry(_SEGS), None)
probe("print_segments ok", lambda: F.print_segments(_SEGS), None)
# 模糊匹配歧义 (4 条直边均含"直边"标签) → 解析器 ValueError.
probe("parse_edge_name ambiguous", lambda: F.parse_edge_name("~直", _SEGS), ValueError)
# 未映射标签 → 空/诊断 (契约 G 行): 无标签与不存在的标签均不抛.
probe("physical_curves None", lambda: F.segments_from_physical_curves(_mesh(), None), None)
probe("physical_curves unmapped", lambda: F.segments_from_physical_curves(_mesh(), {(0, 1): "no_such_curve"}), None)
probe("region_registry empty", lambda: F.segments_from_region_registry(_mesh(), RegionRegistry()), None)
probe("semantic_coverage ok", lambda: F.semantic_coverage(_mesh(), _SEGS), None)

print("== G2 识别器注册表与插件接口 (阶段 2/3 新增) ==")


def _probe_default_order():
    names = [d.name for d in default_registry().detectors()]
    assert names == ["line", "circle", "ellipse", "general"], names
probe("default registry order", _probe_default_order, None)


def _probe_detector_interface():
    # 基类未实现 detect → NotImplementedError; 段类型枚举受控
    Detector().detect(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        scale=1.0, is_outer=True, closed=False)
probe("detector base detect", _probe_detector_interface, NotImplementedError)


def _probe_registry_classify():
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    seg_type, label, info = default_registry().classify(line, 2.0, True)
    assert seg_type == "line" and info["axis"] == "y"
probe("registry classify line", _probe_registry_classify, None)


def _probe_plugin_interface():
    plugin = NativeCircleLabelDetector()
    assert plugin.name == "native_circle_label"
    angle = 2.0 * np.pi * np.arange(32) / 32
    chain = np.column_stack([1.5 * np.cos(angle), 1.5 * np.sin(angle)])
    detection = plugin.detect(
        chain, scale=2.0, is_outer=True, closed=True,
        native_entities=("Circle",))
    assert detection is not None
    assert detection.type == "arc"
    assert detection.params["native_circle"] is True
    assert "[Gmsh 原生圆]" in detection.label
    # 让位: 无原生实体 → None
    assert plugin.detect(
        chain, scale=2.0, is_outer=True, closed=True) is None
probe("plugin detector interface", _probe_plugin_interface, None)


def _probe_register_lifecycle():
    detector = NativeCircleLabelDetector()
    register_detector(detector)
    default_registry().remove(detector.name)
    assert detector.name not in [
        d.name for d in default_registry().detectors()]
probe("register_detector lifecycle", _probe_register_lifecycle, None)


def _probe_register_duplicate():
    detector = NativeCircleLabelDetector()
    register_detector(detector)
    try:
        register_detector(detector)
    finally:
        default_registry().remove(detector.name)
probe("register_detector duplicate", _probe_register_duplicate, ValueError)

probe("register_detector non-detector", lambda: register_detector("x"), TypeError)

print("== I 装配 ==")
probe("assemble_sparse ok", lambda: F.assemble_sparse(_mesh()), None)
probe("assemble_sparse_vectorized ok", lambda: F.assemble_sparse_vectorized(_mesh()), None)

if __name__ == "__main__":
    print(f"\n{'='*40}")
    print(f"FAILS: {len(FAILS)}")
    for f in FAILS:
        print("  -", f)
    sys.exit(1 if FAILS else 0)
