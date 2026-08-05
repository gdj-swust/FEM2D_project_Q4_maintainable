"""A7 (P3, R-α): fix_node 惰性 set 延迟落盘 — 行为逐位不变锁定.

审查现象 (2026-08-05): fix_node 每次全量重建 fixed_dofs (set + sorted),
bc_apply 对整边逐节点调用 → 10 万节点整边固支 ≈ 10¹⁰ 次操作 (O(n² log n)
结构性超线性)。

修复 (mesh.py 领地): fixed_dofs 改 property, 内部维护 set — fix_node
只更新 set (O(1)/次), 数组在首次读取时一次性重建 (O(n log n))。
内容恒为排序去重 int64, 与历史 np.unique 语义一致 → 求解结果逐位
不变 (本文件金标准为修复前基线录制)。

判别性: PROMPT 规定 A7 判别性 = 行为不变 (位移/反力逐位) + 性能由
代码路径证明 (无新测试, 审查者看 diff)。本文件为行为锁定 (两侧均绿)
+ 大调用量冒烟 (回滚 O(n²) 会拖到分钟级)。
"""
import warnings

import numpy as np
import pytest

from fem2d.mesh import Mesh
from fem2d.solver import solve

NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])

# 修复前 (基线录制) 的逐位金标准 — fixed_dofs + prescribed + u + reactions
GOLD_FIXED = [0, 1, 2, 4, 5]
GOLD_PRESCRIBED = [(0, 0.0), (1, 0.0), (2, 0.001), (4, 0.0), (5, 0.0)]
GOLD_U = [0.0, 0.0, 0.001, 0.0006500000000000003, 0.0, 0.0,
          9.074074074074067e-05, 0.0002592592592592596]
GOLD_REACTIONS = [-534.900284900285, 34.90028490028488, 534.900284900285,
                  -0.9999999999999005, -36.90028490028497]


def _golden_mesh():
    m = Mesh(NODES, ELEMS, E=1e6, nu=0.3, thickness=1.0)
    for n in (0, 2):
        m.fix_node(n, "both", 0.0)
    m.fix_node(1, "x", 1e-3)
    m.fix_node(1, "x", 1e-3)   # 完全相同重复: 无警告
    m.add_force(2, 1.0, 2.0)
    return m


def test_fix_node_state_bit_exact_gold():
    m = _golden_mesh()
    assert m.fixed_dofs.tolist() == GOLD_FIXED
    assert sorted(m.prescribed_vals.items()) == GOLD_PRESCRIBED


def test_fix_node_solve_bit_exact_gold():
    r = solve(_golden_mesh(), verbose=False)
    assert r["u"].tolist() == GOLD_U
    assert r["reactions"].tolist() == GOLD_REACTIONS


def test_fixed_dofs_read_only_after_fix_node():
    m = _golden_mesh()
    assert not m.fixed_dofs.flags.writeable
    with pytest.raises(ValueError):
        m.fixed_dofs[0] = 99


def test_lazy_set_matches_sorted_unique_contract():
    # 多次 fix_node 后: 数组 = 排序去重; 相同内容重复读取返回同一对象
    m = _golden_mesh()
    arr1 = m.fixed_dofs
    arr2 = m.fixed_dofs
    assert arr1 is arr2
    assert arr1.dtype == np.int64
    assert np.all(np.diff(arr1) > 0)   # 严格递增 = 排序去重


def test_fix_node_overwrite_warns_on_different_value():
    m = Mesh(NODES, ELEMS)
    m.fix_node(1, "x", 1e-3)
    with pytest.warns(UserWarning, match="overwriting"):
        m.fix_node(1, "x", 2e-3)
    assert m.prescribed_vals[2] == 2e-3


def test_fix_node_identical_repeat_no_warning():
    m = Mesh(NODES, ELEMS)
    m.fix_node(1, "x", 1e-3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m.fix_node(1, "x", 1e-3)
    assert len(caught) == 0


def test_fixed_dofs_external_reassign_still_validated():
    # property setter 后 validate_state 仍校验直接赋值 (既有契约)
    m = Mesh(NODES, ELEMS)
    m.fix_node(1, "x", 0.0)
    m.fixed_dofs = np.array([0, 0], dtype=int)
    with pytest.raises(ValueError, match="重复"):
        m.validate_state()


def test_many_fix_node_calls_content_correct():
    # 3000 节点逐节点固支 — 回滚 O(n² log n) 版本此测试耗时分钟级
    n = 3000
    pts = np.column_stack([
        np.linspace(0.0, 1.0, n), np.zeros(n)])
    tri = np.array([[i, i + 1, i + 2] for i in range(n - 2)])
    m = Mesh(pts, tri)
    for i in range(n):
        m.fix_node(i, "both", 0.0)
    assert m.fixed_dofs.tolist() == list(range(2 * n))
    assert len(m.prescribed_vals) == 2 * n
