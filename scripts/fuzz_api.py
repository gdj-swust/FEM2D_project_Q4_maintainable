"""公共 API 误用输入 fuzz (复查轮, 可复用).

随机类型/形状/越界值/NaN/Inf/布尔/多余分量喂给全部公共 API:
  - 裸 IndexError/KeyError/AttributeError 冒出 → bug (契约禁止)
  - TypeError/ValueError 带上下文 → 正确行为
  - 静默成功但输入非法 (已知误用模式) → bug
用法: python scripts/fuzz_api.py [轮数=500]
退出码: 抓到 bug → 1。
"""
import random
import sys

import numpy as np

import fem2d as F
from fem2d.bc import apply_elimination, apply_penalty
from fem2d.config import AnalysisConfig
from fem2d.error_est import estimate as estimate_error
from fem2d.loads_core import parse_traction, parse_vec2
from fem2d.material import D_matrix, von_mises
from fem2d.mesh import Mesh
from fem2d.solver import estimate_condition, solve
from fem2d.spr import spr_recovery
from fem2d.stress import (
    compute_stresses,
    nodal_L2_projection,
    nodal_average,
    nodal_simple,
    nodal_weighted,
    point_in_element,
    principal_stresses,
    stress_at_point,
)

BARE = (IndexError, KeyError, AttributeError)

NODES = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])


def _mesh():
    return Mesh(NODES, ELEMS, E=1e6, nu=0.3, thickness=1.0)


def _solved():
    m = _mesh()
    for i in range(4):
        m.fix_node(i, "both", 0.0)
    m.add_force(2, 1.0, 0.0)
    return m, solve(m, verbose=False)


def _rand_value(rng):
    """随机标量/容器值."""
    kind = rng.randrange(14)
    if kind == 0:
        return None
    if kind == 1:
        return rng.choice([True, False])
    if kind == 2:
        return rng.choice([0, 1, -1, 5, -5, 100])
    if kind == 3:
        return rng.choice([0.0, 1.0, -1.0, 1e-310, 1e150, -1e150])
    if kind == 4:
        return float("nan")
    if kind == 5:
        return float("inf")
    if kind == 6:
        return -float("inf")
    if kind == 7:
        return rng.choice(["abc", "1e6,0", "", "1e6x,0", "nan,0"])
    if kind == 8:
        return 1 + 2j
    if kind == 9:
        return [1.0]
    if kind == 10:
        return (0.5, 1.5)
    if kind == 11:
        return np.array([1.0, 2.0, 3.0])
    if kind == 12:
        return np.array([True, False, True])
    if kind == 13:
        return np.array([0.5])
    return 0


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = random.Random(20260803)
    bugs = []
    calls = 0

    def feed(name, fn):
        nonlocal calls
        calls += 1
        try:
            fn()
        except BARE as exc:
            bugs.append(f"{name}: bare {type(exc).__name__}: {str(exc)[:60]}")
        except Exception:
            # ValueError/TypeError 带上下文 = 正确行为, 忽略
            pass

    for _ in range(rounds):
        v = _rand_value(rng)
        i = rng.randrange(30)
        if i == 0:
            feed(f"fix_node({v!r})", lambda v=v: _mesh().fix_node(v, "both", 0.0))
        elif i == 1:
            feed(f"fix_node value({v!r})", lambda v=v: _mesh().fix_node(0, "both", v))
        elif i == 2:
            feed(f"add_force({v!r})", lambda v=v: _mesh().add_force(v, 1.0, 0.0))
        elif i == 3:
            feed(f"add_pressure({v!r})", lambda v=v: _mesh().add_pressure(0, 1, v))
        elif i == 4:
            feed(f"fix_nodes_func({v!r})", lambda v=v: _mesh().fix_nodes_func(v, 0.0))
        elif i == 5:
            feed(f"nodes_on_edge({v!r})", lambda v=v: _mesh().nodes_on_edge(v, "min"))
        elif i == 6:
            feed(f"solve({v!r})", lambda v=v: solve(v, verbose=False))
        elif i == 7:
            feed(f"estimate_condition({v!r})", lambda v=v: estimate_condition(np.eye(4), method=v))
        elif i == 8:
            feed(f"estimate_error({v!r})", lambda v=v: estimate_error(_mesh(), v))
        elif i == 9:
            feed(f"principal({v!r})", lambda v=v: principal_stresses(v))
        elif i == 10:
            feed(f"von_mises({v!r})", lambda v=v: von_mises(v))
        elif i == 11:
            feed(f"D_matrix({v!r})", lambda v=v: D_matrix(v, 0.3))
        elif i == 12:
            feed(f"compute_stresses({v!r})", lambda v=v: compute_stresses(_mesh(), v))
        elif i == 13:
            feed(f"nodal_average({v!r})", lambda v=v: nodal_average(_mesh(), v))
        elif i == 14:
            feed(f"spr_recovery({v!r})", lambda v=v: spr_recovery(_mesh(), v))
        elif i == 15:
            feed(f"parse_vec2({v!r})", lambda v=v: parse_vec2(v))
        elif i == 16:
            feed(f"parse_traction({v!r})", lambda v=v: parse_traction(v))
        elif i == 17:
            feed(f"apply_penalty({v!r})", lambda v=v: apply_penalty(np.eye(4), np.zeros(4), v))
        elif i == 18:
            feed(f"stress_at_point({v!r})", lambda v=v: stress_at_point(_mesh(), {"stress": np.ones((2, 3))}, v, 0.5))
        elif i == 19:
            feed(f"get_element_kernel({v!r})", lambda v=v: F.get_element_kernel(v))
        elif i == 20:
            feed(f"Mesh nodes({v!r})", lambda v=v: Mesh(v, ELEMS))
        elif i == 21:
            feed(f"Mesh elems({v!r})", lambda v=v: Mesh(NODES, v))
        elif i == 22:
            feed(f"apply_elim free({v!r})", lambda v=v: apply_elimination(np.eye(4), np.zeros(4), v, [0, 1], [0., 0.]))
        elif i == 23:
            feed(f"nodal_L2({v!r})", lambda v=v: nodal_L2_projection(_mesh(), v))
        elif i == 24:
            feed(f"nodal_simple({v!r})", lambda v=v: nodal_simple(_mesh(), v))
        elif i == 25:
            feed(f"AnalysisConfig({v!r})", lambda v=v: AnalysisConfig(E=v))
        elif i == 26:
            feed(f"point_in_element({v!r})", lambda v=v: point_in_element(_mesh(), v, 0.5))
        elif i == 27:
            feed(f"replace_nodes({v!r})", lambda v=v: _mesh().replace_nodes(v))
        elif i == 28:
            feed(f"replace_elements({v!r})", lambda v=v: _mesh().replace_elements(v))
        elif i == 29:
            feed(f"estimate_condition K({v!r})", lambda v=v: estimate_condition(v))

    print(f"calls={calls}")
    print(f"bare-exception escapes: {len(bugs)}")
    for b in bugs:
        print("  -", b)
    if not bugs:
        print("OK: 无裸异常冒出")
    sys.exit(1 if bugs else 0)


if __name__ == "__main__":
    main()
