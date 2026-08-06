"""A6 (P3, R-α): 同一边界边重复 add_traction/add_pressure 必须响亮警告.

审查现象 (2026-08-05): 同边两次 add_traction 无警告无去重 → F/位移
精确 ×2 (审查实测比值 2.0000)。交互路径 (bc_apply 段选择 dict.fromkeys)
已去重, API 路径无防护。

修复: add_traction/add_pressure 统一经 _append_traction — 检测同
(nodes 无序对, is_pressure) 重复 → 响亮 UserWarning (完全相同载荷:
"将翻倍 ×2"; 不同载荷: "线性叠加")。**保持累加语义** — 载荷拆分是
既有锁定契约 (test_regressions_round4 的 load-splitting 不变性),
去重会静默吞掉拆分载荷, 故选择"警告须响亮"分支。

判别性: 回滚本提交 → 双次 add_traction 无警告 (×2.0000 静默回归)。
"""
import warnings

import numpy as np
import pytest

from fem2d.mesh import Mesh
from fem2d.solver import solve

NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])


def _cantilever():
    m = Mesh(NODES, ELEMS, E=1e6, nu=0.3, thickness=1.0)
    for n in (0, 2):
        m.fix_node(n, "both", 0.0)
    return m


def _solve_u(m):
    return solve(m, verbose=False)["u"]


def test_duplicate_traction_warns_loudly():
    # 回滚 → 无警告, ×2.0000 静默回归
    m = _cantilever()
    with pytest.warns(UserWarning, match="翻倍"):
        m.add_traction(1, 3, 1e6, 0.0)
        m.add_traction(1, 3, 1e6, 0.0)


def test_duplicate_traction_still_cumulative_with_warning():
    # 累加语义保持 (载荷拆分契约) — 警告出现的同时位移仍精确 ×2
    m = _cantilever()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.add_traction(1, 3, 1e6, 0.0)
        m.add_traction(1, 3, 1e6, 0.0)
    assert len(caught) == 1
    single = _cantilever()
    single.add_traction(1, 3, 1e6, 0.0)
    u_single = _solve_u(single)
    u_double = _solve_u(m)
    mask = np.abs(u_single) > 0
    np.testing.assert_allclose(u_double[mask] / u_single[mask],
                               np.full(mask.sum(), 2.0), rtol=1e-12,
                               atol=0.0)


def test_duplicate_traction_reversed_order_warns():
    # (3,1) 与 (1,3) 判为同一载荷 — 无序节点对归一
    m = _cantilever()
    with pytest.warns(UserWarning, match="翻倍"):
        m.add_traction(3, 1, 1e6, 0.0)
        m.add_traction(1, 3, 1e6, 0.0)


def test_duplicate_pressure_warns():
    m = _cantilever()
    with pytest.warns(UserWarning, match="翻倍"):
        m.add_pressure(1, 3, 1e6)
        m.add_pressure(3, 1, 1e6)


def test_different_traction_on_same_edge_warns_superposition():
    # 同边不同载荷是合法叠加 — 响亮警告告知叠加
    m = _cantilever()
    with pytest.warns(UserWarning, match="线性叠加"):
        m.add_traction(1, 3, 1e6, 0.0)
        m.add_traction(1, 3, 0.0, 1e6)


def test_different_pressure_warns_superposition():
    m = _cantilever()
    with pytest.warns(UserWarning, match="线性叠加"):
        m.add_pressure(1, 3, 1e6)
        m.add_pressure(1, 3, 2e6)


def test_callable_traction_same_object_warns():
    f = lambda x, y: (1e6, 0.0)
    m = _cantilever()
    with pytest.warns(UserWarning, match="翻倍"):
        m.add_traction(1, 3, f, 0.0)
        m.add_traction(1, 3, f, 0.0)


def test_traction_and_pressure_on_same_edge_no_warning():
    # 面力 vs 压力标志不同 → 不判重 (两种载荷机制, 无警告)
    m = _cantilever()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.add_traction(1, 3, 1e6, 0.0)
        m.add_pressure(1, 3, 1e6)
    assert len(caught) == 0


def test_single_traction_no_warning():
    m = _cantilever()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.add_traction(1, 3, 1e6, 0.0)
    assert len(caught) == 0
