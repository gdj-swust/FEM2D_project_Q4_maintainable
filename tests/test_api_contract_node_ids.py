"""契约清账阶段 2 — 节点索引校验收敛判别性测试.

旧实现: fix_nodes_func 传单个节点号 → 裸 TypeError ('int' not iterable);
func 返回 3+ 分量 → 静默忽略; 载荷记录节点校验与 _validate_node_id 双实现。
"""
import numpy as np
import pytest

from fem2d.mesh import Mesh


def _mesh():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 3], [0, 3, 2]])
    return Mesh(nodes, elements)


# ── fix_nodes_func 节点列表契约 ──

def test_fix_nodes_func_scalar_node_rejected():
    m = _mesh()
    # 单个节点号曾 for-in 迭代抛裸 TypeError — 明示需要列表
    with pytest.raises(ValueError, match="节点索引列表"):
        m.fix_nodes_func(2, 0.0)


def test_fix_nodes_func_extra_components_rejected():
    m = _mesh()
    # func 返回 3 分量曾静默忽略第 3 个 — 载荷静默错误
    with pytest.raises(ValueError, match="2 个"):
        m.fix_nodes_func([0], lambda x, y: (1.0, 2.0, 3.0))


def test_fix_nodes_func_non_iterable_result_rejected():
    m = _mesh()
    with pytest.raises(TypeError, match="返回值类型非法"):
        m.fix_nodes_func([0], lambda x, y: 1 + 2j)  # 复数: 非 Real 且不可迭代


def test_fix_nodes_func_nan_result_rejected():
    m = _mesh()
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.fix_nodes_func([0], lambda x, y: (float("nan"), 0.0))


def test_fix_nodes_func_numpy_scalar_result_ok():
    m = _mesh()
    # np.float64 标量返回 (旧 isinstance(int, float) 未覆盖 np 标量)
    m.fix_nodes_func([0], lambda x, y: np.float64(1e-4))
    assert m.prescribed_vals[0] == pytest.approx(1e-4)


def test_fix_nodes_func_bool_node_rejected():
    m = _mesh()
    with pytest.raises(TypeError, match="node id must be an integer"):
        m.fix_nodes_func([True], 0.0)


# ── 载荷记录节点 (构造函数直传) — 与 _validate_node_id 同契约 ──

def test_concentrated_force_node_bool_rejected():
    m = _mesh()
    with pytest.raises(TypeError, match="node id must be an integer"):
        m.concentrated_forces.append({"node": True, "force": (1.0, 0.0)})
        m.validate_state()


def test_concentrated_force_node_float_noninteger_rejected():
    m = _mesh()
    m.concentrated_forces.append({"node": 1.5, "force": (1.0, 0.0)})
    with pytest.raises(TypeError, match="node id must be an integer"):
        m.validate_state()


def test_concentrated_force_node_float_integer_normalized():
    m = _mesh()
    m.concentrated_forces.append({"node": 2.0, "force": (1.0, 0.0)})
    m.validate_state()
    assert m.concentrated_forces[0]["node"] == 2


def test_surface_traction_node_bool_rejected():
    m = _mesh()
    m.surface_tractions.append({
        "nodes": (True, 1), "traction": (1e6, 0.0)})
    with pytest.raises(TypeError, match="node id must be an integer"):
        m.validate_state()


def test_surface_traction_node_negative_rejected():
    m = _mesh()
    m.surface_tractions.append({
        "nodes": (-1, 1), "traction": (1e6, 0.0)})
    with pytest.raises(ValueError, match="out of range"):
        m.validate_state()
