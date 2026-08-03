"""Mesh 构造器/约束接口的裸 IndexError 守卫 (外部审查 P2).

标量 nodes/elements 曾 0 维数组 shape[0] 裸 IndexError; fix_nodes_func
越界 nid 曾先索引后校验裸 IndexError; 负 nid 曾静默约束最后一个节点.
判别性: 旧实现抛 IndexError, 修复后必须 ValueError 带上下文.
"""
import numpy as np
import pytest

from fem2d.mesh import Mesh

_GOOD_NODES = np.array([[0., 0.], [1., 0.], [0., 1.]])
_GOOD_ELEMS = np.array([[0, 1, 2]])


def test_scalar_nodes_rejected_with_valueerror():
    with pytest.raises(ValueError, match="nodes must be a 2-D array"):
        Mesh(nodes=1.0, elements=_GOOD_ELEMS)


def test_scalar_elements_rejected_with_valueerror():
    with pytest.raises(ValueError, match="elements must be a 2-D array"):
        Mesh(nodes=_GOOD_NODES, elements=1.0)


def test_fix_nodes_func_out_of_range_valueerror():
    m = Mesh(nodes=_GOOD_NODES, elements=_GOOD_ELEMS)
    with pytest.raises(ValueError, match="out of range"):
        m.fix_nodes_func([3], lambda x, y: (0.0, 0.0))


def test_fix_nodes_func_negative_rejected():
    """负 nid 曾静默约束最后一个节点 (arr[-1] 语义)."""
    m = Mesh(nodes=_GOOD_NODES, elements=_GOOD_ELEMS)
    with pytest.raises(ValueError, match="out of range"):
        m.fix_nodes_func([-1], lambda x, y: (0.0, 0.0))


def test_fix_nodes_func_valid_path_unaffected():
    m = Mesh(nodes=_GOOD_NODES, elements=_GOOD_ELEMS)
    m.fix_nodes_func([0, 1], lambda x, y: (x, y))
    assert 0 in m.fixed_dofs and 1 in m.fixed_dofs
    assert m.prescribed_vals.get(0) == 0.0
    assert m.prescribed_vals.get(2) == 1.0
