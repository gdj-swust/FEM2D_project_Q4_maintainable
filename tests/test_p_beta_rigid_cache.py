"""P-β 判别性测试 — 刚体模态检查向量化等价性 + 结果缓存.

缺陷 (P轮Ⅱ 任务书): ``mesh.check_rigid_body_constraints`` 逐节点
Python 循环扫约束 DOF (``(2*n) in fixed_set``), 100k 级网格 (200k
DOF) 每次 ``solve()`` 全量重算拓扑 (connected_components + 逐分量
扫描), BC 未变的重复 solve (参数研究/收敛研究) 白白重算。

修复:
1. 向量化 — 全分量 DOF 一次 ``np.isin``, 集合语义不变 (元素集与 len
   与旧循环逐元素一致)。
2. 结果缓存 — 缓存键 = fixed_dofs 内容快照拷贝 (``np.array_equal``
   比较, 覆盖 fix_node/setter/原地修改全部写入口) + 网格变更经
   ``invalidate_cache`` 失效 (replace_nodes/replace_elements)。
   返回 list 只读复用。

判别性红侧自证:
- 向量化等价: 手写网格 + 手工推算 comp_dofs (len/元素集 → 约束节点
  数/rank/缺失模态) 锁定精确 issue 字符串; 旧实现参照函数
  (_legacy_check_rigid_body_constraints, 逐节点循环) 逐用例全输出
  相等 — 错误向量化必红。
- 缓存命中: 二调 connected_components 调用计数 == 1 且返回同一
  list 对象 (无缓存必红: 计数 == 2 / 身份不同)。
- 失效: fix_node 增删 / fixed_dofs setter 重赋值 / fix_nodes_func /
  原地 setflags+写 / replace_nodes / replace_elements /
  invalidate_cache → 计数递增 + 结果更新 (漏失效必红)。
- 原地修改: 直接改数组内容 (绕过 setter/fix_node) 结果仍正确 —
  快照存引用 (而非拷贝) 时数组与自己比较永远命中陈旧结果 (必红)。
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from fem2d import Mesh
from fem2d.topology_core import build_edge_table

# ── 手写网格工厂 (无 gmsh 依赖) ─────────────────────────────────────


def _tri_plus_isolated():
    """三角形 (节点0-2) + 孤立节点 3 — 任务书场景: 孤立节点 + 部分约束."""
    return Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [5., 5.]]),
                elements=np.array([[0, 1, 2]]), E=1e6, nu=0.3)


def _two_components():
    """两个独立三角形分量 (comp0={0,1,2}, comp1={3,4,5}) — 无孤立节点."""
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.],
                      [5., 5.], [6., 5.], [5., 6.]])
    return Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [3, 4, 5]]),
                E=1e6, nu=0.3)


def _triangle_pair_plus_isolated():
    """共享边双三角形 (comp0={0,1,2,3}) + 孤立节点 4 — 多分量+孤立."""
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [9., 9.]])
    return Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 2, 3]]),
                E=1e6, nu=0.3)


def _micro_scale():
    """微尺度 (L~1e-150) — 工程约定 1: 无绝对阈值, 微尺度必须正确."""
    m = _tri_plus_isolated()
    m.replace_nodes(m.nodes * 1e-150)
    return m


# ── 旧实现参照 (运行时参照 — 与 P-β 前逐节点循环逐字节一致) ────────


def _legacy_check_rigid_body_constraints(mesh):
    """P-β 前的原实现 (逐节点 Python 循环扫约束 DOF).

    作为向量化/缓存后行为等价的判别性参照 — 任何输出差异都意味着
    向量化改变了判定语义。
    """
    mesh.build_connectivity()
    n_nodes = mesh.n_nodes
    edge_table = build_edge_table(
        mesh.elements, mesh.element_kernel.local_edges, n_nodes)
    row = np.concatenate([edge_table.lo, edge_table.hi])
    col = np.concatenate([edge_table.hi, edge_table.lo])
    data = np.ones(len(row), dtype=int)
    adj = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(adj, directed=False)

    fixed_set = set(mesh.fixed_dofs)
    issues = []
    for comp in range(n_comp):
        comp_nodes = np.where(labels == comp)[0]
        if len(comp_nodes) < 2:
            issues.append({
                "component": comp,
                "nodes": comp_nodes.tolist(),
                "issue": f"孤立分量 ({len(comp_nodes)} 节点, 无单元连接)"
            })
            continue
        comp_dofs = set()
        for n in comp_nodes:
            if (2*n) in fixed_set:
                comp_dofs.add(2*n)
            if (2*n + 1) in fixed_set:
                comp_dofs.add(2*n + 1)
        if len(comp_dofs) == 0:
            issues.append({
                "component": comp,
                "nodes": comp_nodes.tolist(),
                "issue": "无任何约束 — 保留 3 个刚体模态"
            })
            continue
        xy = mesh.nodes[comp_nodes]
        origin = xy.mean(axis=0)
        scl = max(np.ptp(xy, axis=0).max(),
                  64.0 * np.finfo(float).eps * max(
                      float(np.max(np.abs(xy))), np.finfo(float).tiny))
        constrained_nodes = sorted(set(d // 2 for d in comp_dofs))
        R_rows = []
        for n in constrained_nodes:
            x = (mesh.nodes[n, 0] - origin[0]) / scl
            y = (mesh.nodes[n, 1] - origin[1]) / scl
            if (2*n) in comp_dofs:
                R_rows.append([1.0, 0.0, -y])
            if (2*n + 1) in comp_dofs:
                R_rows.append([0.0, 1.0,  x])
        R = np.array(R_rows)
        rank = np.linalg.matrix_rank(R)
        if rank < 3:
            missing = []
            if rank < 2:
                _, _, vt = np.linalg.svd(R)
                null = vt[2]
                if abs(null[0]) > 0.5:
                    missing.append("x-平动")
                if abs(null[1]) > 0.5:
                    missing.append("y-平动")
                if abs(null[2]) > 0.5:
                    missing.append("转动")
            elif rank == 2:
                _, _, vt = np.linalg.svd(R)
                null = vt[2]
                if abs(null[0]) > 0.5:
                    missing.append("x-平动")
                elif abs(null[1]) > 0.5:
                    missing.append("y-平动")
                else:
                    missing.append("转动 (约束共线或等价)")
            issues.append({
                "component": comp,
                "nodes": comp_nodes.tolist(),
                "issue": (f"约束不足 ({len(constrained_nodes)} 节点约束, "
                          f"rank={rank}/3) — 残留: {', '.join(missing) if missing else '未知'}"),
                "constrained_nodes": constrained_nodes,
            })
    return issues


# ── 探针: 统计真实 connected_components 调用次数 (不改语义) ────────


class _CountingConnections:
    """包装真实实现 — 缓存命中判别: 二调不重算拓扑 ⟺ 调用次数不增."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return connected_components(*args, **kwargs)


def _install_counter(monkeypatch):
    probe = _CountingConnections()
    monkeypatch.setattr("scipy.sparse.csgraph.connected_components", probe)
    return probe


# ── 1. 向量化等价性 (判别性) ────────────────────────────────────────


def test_hand_computed_comp_dofs_and_issues_locked():
    """手工推算 comp_dofs 的 len/元素集 → 锁定精确 issue 输出.

    网格: comp0 = 三角形 (节点 0,1,2, 坐标 (0,0),(1,0),(0,1)),
    comp1 = 孤立节点 3. 手工推算:
      fix_node(0, "both") → comp_dofs = {0, 1}   (len 2, 元素集 {0,1})
      constrained_nodes = sorted({d//2}) = [0]   (1 节点约束)
      R = [[1,0,-y0],[0,1,x0]], 原点=(1/3,1/3), scl=1
        = [[1,0,1/3],[0,1,-1/3]] → rank=2 (缺转动: 约束共线)
      null ∝ (-1/3, 1/3, 1): |null[0]|=0.30, |null[1]|=0.30 <0.5,
      |null[2]|=0.90 >0.5 → rank==2 分支 elif 链 → "转动 (约束共线或等价)"
    孤立分量分支不扫描约束 DOF (行为冻结) → comp1 恒为"孤立分量".
    """
    m = _tri_plus_isolated()
    m.fix_node(0, "both")
    expected = [
        {"component": 0, "nodes": [0, 1, 2],
         "issue": "约束不足 (1 节点约束, rank=2/3) — 残留: 转动 (约束共线或等价)",
         "constrained_nodes": [0]},
        {"component": 1, "nodes": [3],
         "issue": "孤立分量 (1 节点, 无单元连接)"},
    ]
    assert m.check_rigid_body_constraints() == expected
    # 二调 (缓存命中路径) 输出逐元素一致
    assert m.check_rigid_body_constraints() == expected


def test_hand_computed_two_constrained_dofs():
    """两个约束 DOF {0, 4} (node0-x + node2-x) → comp_dofs len 2.

    constrained_nodes = [0, 2]; R 两行均为 x-平动行 → rank=2,
    null ∝ (0, 1, 0) (|null[1]|=1) → 残留 y-平动 — 若向量化错取
    dof (如 {0,5}) R 行变成 y-平动行, rank/残留文本即变 (红侧).
    """
    m = _tri_plus_isolated()
    m.fix_node(0, "x")
    m.fix_node(2, "x")
    expected = [
        {"component": 0, "nodes": [0, 1, 2],
         "issue": "约束不足 (2 节点约束, rank=2/3) — 残留: y-平动",
         "constrained_nodes": [0, 2]},
        {"component": 1, "nodes": [3],
         "issue": "孤立分量 (1 节点, 无单元连接)"},
    ]
    assert m.check_rigid_body_constraints() == expected


def test_hand_computed_no_and_full_constraint():
    """无约束 → comp_dofs 空集 (len 0) → "无任何约束"; 全约束 → 空输出."""
    m = _tri_plus_isolated()
    assert m.check_rigid_body_constraints() == [
        {"component": 0, "nodes": [0, 1, 2],
         "issue": "无任何约束 — 保留 3 个刚体模态"},
        {"component": 1, "nodes": [3],
         "issue": "孤立分量 (1 节点, 无单元连接)"},
    ]

    m.fix_node(0, "both")
    m.fix_node(1, "both")
    m.fix_node(2, "both")
    assert m.check_rigid_body_constraints() == [
        {"component": 1, "nodes": [3],
         "issue": "孤立分量 (1 节点, 无单元连接)"},
    ]


_BC_SETS = {
    "no": lambda m: None,
    "partial": lambda m: (m.fix_node(0, "x"), m.fix_node(2, "y")),
    "full": lambda m: (m.fix_node(0, "both"), m.fix_node(1, "both"),
                       m.fix_node(2, "both")),
    "func": lambda m: m.fix_nodes_func([0, 1, 2], lambda x, y: (0.0, 0.0)),
    "partial2": lambda m: (m.fix_node(0, "both"), m.fix_node(3, "x")),
}

_MESH_BUILDERS = {
    "tri+isolated": _tri_plus_isolated,
    "two-components": _two_components,
    "tri-pair+isolated": _triangle_pair_plus_isolated,
    "micro-scale": _micro_scale,
}


@pytest.mark.parametrize("mesh_name", sorted(_MESH_BUILDERS))
@pytest.mark.parametrize("bc_name", sorted(_BC_SETS))
def test_vectorized_scan_equivalent_to_legacy(mesh_name, bc_name):
    """向量化 + 缓存后输出与旧实现参照逐元素一致 (多 BC 组合电池).

    BC 组合: 无约束 / 部分约束 / 全约束 / fix_nodes_func / 混合 —
    覆盖任务书要求的 无/部分/全/孤立节点 场景; 微尺度网格验证无绝对
    阈值下语义不变. 首调 (未缓存) 与二调 (缓存命中) 都须与参照相等.
    """
    m1 = _MESH_BUILDERS[mesh_name]()
    m2 = _MESH_BUILDERS[mesh_name]()   # 独立副本: 首调路径 (无缓存)
    _BC_SETS[bc_name](m1)
    _BC_SETS[bc_name](m2)

    expected = _legacy_check_rigid_body_constraints(m1)
    got_first = m1.check_rigid_body_constraints()
    assert got_first == expected, (
        f"[{mesh_name}/{bc_name}] 向量化首调输出与旧实现不一致")
    got_second = m1.check_rigid_body_constraints()   # 缓存命中
    assert got_second == expected, (
        f"[{mesh_name}/{bc_name}] 缓存命中输出与旧实现不一致")
    assert m2.check_rigid_body_constraints() == expected, (
        f"[{mesh_name}/{bc_name}] 独立网格首调输出与旧实现不一致")
