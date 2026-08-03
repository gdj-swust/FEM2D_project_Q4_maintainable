"""契约表-代码一致性探针 (复查轮可复用审计工具).

逐条实测 docs/api_contract.md 各行的"应有错误行为"列:
  probe(name, fn, expect) — expect 为异常类型或 None (不应抛)。
输出 PASS/FAIL, 任何 FAIL → 退出码 1。
"""
import sys

import numpy as np

import fem2d as F
from fem2d.bc import apply_elimination, apply_penalty
from fem2d.config import AnalysisConfig
from fem2d.error_est import (
    element_refinement_indicator,
    estimate as estimate_error,
)
from fem2d.gmsh_adapter import import_msh
from fem2d.loads_core import parse_traction, parse_vec2
from fem2d.material import D_matrix, von_mises
from fem2d.mesh import Mesh
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


def probe(name, fn, expect):
    try:
        result = fn()
        if expect is None:
            print(f"  PASS {name}")
            return result
        FAILS.append(f"{name}: NO ERROR (expected {expect.__name__})")
        print(f"  FAIL {name}: NO ERROR (expected {expect.__name__})")
    except Exception as exc:  # noqa: BLE001 — 审计工具
        if expect is not None and isinstance(exc, expect):
            print(f"  PASS {name}: {type(exc).__name__}")
        else:
            FAILS.append(f"{name}: {type(exc).__name__} (expected "
                         f"{expect.__name__ if expect else 'no error'}): {exc}")
            print(f"  FAIL {name}: {type(exc).__name__} (expected "
                  f"{expect.__name__ if expect else 'no error'}): "
                  f"{str(exc)[:80]}")
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
probe("assemble_loads n_dof mismatch", lambda: F.assemble_loads(m2, 3), (IndexError, ValueError))
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

print("== F 材料 ==")
probe("D_matrix E neg", lambda: D_matrix(-1.0, 0.3), ValueError)
probe("D_matrix nu bound", lambda: D_matrix(1.0, 0.5), ValueError)
probe("D_matrix plane", lambda: D_matrix(1.0, 0.3, "bogus"), ValueError)
probe("D_matrix E str", lambda: D_matrix("abc", 0.3), TypeError)
probe("von_mises shape", lambda: von_mises(np.ones((2, 2))), ValueError)
probe("von_mises nan", lambda: von_mises(np.array([[np.nan, 1., 0.]])), ValueError)
probe("von_mises plane", lambda: von_mises(np.ones((2, 3)), plane_type="bogus"), ValueError)
probe("get_element_kernel unknown", lambda: F.get_element_kernel("BOGUS"), ValueError)

print("== H 配置 ==")
probe("config E neg", lambda: AnalysisConfig(E=-1.0), ValueError)
probe("config nu bound", lambda: AnalysisConfig(nu=0.9), ValueError)
probe("config plane", lambda: AnalysisConfig(plane="bogus"), ValueError)
probe("config solver", lambda: AnalysisConfig(linear_solver="bogus"), ValueError)
probe("config band trio", lambda: AnalysisConfig(band_min=0.0), ValueError)
probe("config band step", lambda: AnalysisConfig(band_min=0., band_max=1., band_step=0.), ValueError)
probe("config band nonint", lambda: AnalysisConfig(band_min=0., band_max=1., band_step=0.3), ValueError)

print("== E 输入链 (无 gmsh 依赖路径) ==")
probe("import_msh missing", lambda: import_msh("C:/no_such_dir_xyz/x.msh"), FileNotFoundError)
from fem2d.input_source import physical_point_from_geo
probe("physical_point no_geo", lambda: physical_point_from_geo(None, "p", _mesh())[3], None)

if __name__ == "__main__":
    print(f"\n{'='*40}")
    print(f"FAILS: {len(FAILS)}")
    for f in FAILS:
        print("  -", f)
    sys.exit(1 if FAILS else 0)
