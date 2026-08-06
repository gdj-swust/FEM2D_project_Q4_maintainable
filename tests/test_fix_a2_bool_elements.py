"""A2 (P3, R-α): 布尔单元索引数组必须拒绝 — 禁止静默构造退化单元.

审查现象 (2026-08-05): elements=[[True,True,False]] 静默转 [[1,1,0]]
(重复节点退化单元), 构造不报错 — 与项目既有 bool 拒绝策略
(mesh._validate_node_id / require_dof_index_array) 相悖。

修复: 浮点分支前显式拒绝 elems_raw.dtype.kind == "b" (构造 +
replace_elements 两条入口)。

判别性: 回滚本提交 → 布尔构造静默通过回归。
"""
import numpy as np
import pytest

from fem2d.mesh import Mesh

NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])


def test_constructor_bool_elements_rejected():
    # 回滚 → 静默接受并转成 [[1,1,0]] 重复节点退化单元
    with pytest.raises(ValueError, match="boolean"):
        Mesh(NODES, [[True, True, False]])


def test_constructor_bool_2d_array_rejected():
    with pytest.raises(ValueError, match="boolean"):
        Mesh(NODES, np.ones((1, 3), dtype=bool))


def test_replace_elements_bool_rejected():
    m = Mesh(NODES, ELEMS)
    with pytest.raises(ValueError, match="布尔数组"):
        m.replace_elements(np.array([[True, False, True]]))


def test_constructor_int_elements_unchanged():
    # 绿侧: 整数/浮点索引路径行为不变
    m = Mesh(NODES, ELEMS)
    assert m.elements.dtype == np.int64
    m2 = Mesh(NODES, [[0.0, 1.0, 3.0]])  # 恰为整数的浮点仍接受
    assert m2.elements.tolist() == [[0, 1, 3]]
