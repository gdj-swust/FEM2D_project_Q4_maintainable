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


# ── 2. 缓存命中 (判别性红侧) ────────────────────────────────────────


def test_second_call_hits_cache_single_topology_pass(monkeypatch):
    """同一网格两次检查 → 拓扑只算一次 + 返回同一 list 对象 (只读复用).

    无缓存时: connected_components 调用 2 次, 且二调返回新建 list
    (身份断言必红).
    """
    m = _triangle_pair_plus_isolated()
    m.fix_node(0, "both")
    m.fix_node(1, "both")
    m.fix_node(2, "both")
    probe = _install_counter(monkeypatch)

    first = m.check_rigid_body_constraints()
    assert probe.calls == 1
    second = m.check_rigid_body_constraints()
    assert probe.calls == 1, (
        f"二调重算了拓扑 (connected_components 调用 {probe.calls} 次, "
        "期望 1 — 缓存未命中)")
    assert second is first, "二调未复用缓存结果 (返回了新建 list)"


def test_cache_hit_path_skips_connectivity_and_components(monkeypatch):
    """缓存命中路径零预处理 — 不调 build_connectivity 也不扫图.

    首调建缓存后, 二调 (BC/网格未变) 必须直接返回, 连接重建
    (spy) 与 connected_components (探针) 都不触发.
    """
    m = _tri_plus_isolated()
    m.fix_node(0, "both")
    m.check_rigid_body_constraints()          # 首调: 建连接 + 建缓存
    build_calls = [0]
    original_build = m.build_connectivity

    def spy_build():
        build_calls[0] += 1
        return original_build()

    m.build_connectivity = spy_build          # 实例级 spy (仅本测试)
    probe = _install_counter(monkeypatch)
    m.check_rigid_body_constraints()
    assert probe.calls == 0, (
        f"缓存命中路径重算连通分量: 调用 {probe.calls} 次, 期望 0")
    assert build_calls[0] == 0, "缓存命中路径重调 build_connectivity"


# ── 3. 失效路径 (判别性红侧) ────────────────────────────────────────


def test_fix_node_invalidate_and_update(monkeypatch):
    """fix_node 增删约束 → 缓存失效 (计数+1) 且结果更新."""
    m = _tri_plus_isolated()
    probe = _install_counter(monkeypatch)

    issues1 = m.check_rigid_body_constraints()
    assert probe.calls == 1
    assert issues1[0]["issue"] == "无任何约束 — 保留 3 个刚体模态"

    m.fix_node(0, "both")
    issues2 = m.check_rigid_body_constraints()
    assert probe.calls == 2, "fix_node 后缓存未失效 (计数未递增)"
    assert issues2[0]["issue"] != issues1[0]["issue"], "fix_node 后结果未更新"

    m.fix_node(1, "both")
    m.fix_node(2, "both")
    issues3 = m.check_rigid_body_constraints()
    assert probe.calls == 3
    assert issues3 == [
        {"component": 1, "nodes": [3],
         "issue": "孤立分量 (1 节点, 无单元连接)"},
    ], "约束增足后结果未更新"


def test_fixed_dofs_setter_invalidate(monkeypatch):
    """fixed_dofs setter 重赋值 → 缓存显式失效 (白盒) + 结果更新."""
    m = _tri_plus_isolated()
    m.fix_node(0, "both")
    m.check_rigid_body_constraints()
    assert m._rigid_cache is not None

    m.fixed_dofs = np.array([0, 1, 2, 3, 4, 5], dtype=int)
    assert m._rigid_cache is None, "setter 未清刚体缓存 (白盒失效要求)"
    issues = m.check_rigid_body_constraints()
    assert issues == [
        {"component": 1, "nodes": [3],
         "issue": "孤立分量 (1 节点, 无单元连接)"},
    ]

    # 内容比较兜底: 即使 setter 失效被移除, 新内容 ≠ 旧快照 → 仍重算
    probe = _install_counter(monkeypatch)
    m.check_rigid_body_constraints()          # BC 未变 → 缓存命中 (0 次)
    m.fixed_dofs = np.array([], dtype=int)    # BC 变更 → 失效
    m.check_rigid_body_constraints()          # 重算 (1 次)
    assert probe.calls == 1


def test_fix_nodes_func_invalidate(monkeypatch):
    """fix_nodes_func (批量函数约束) → 缓存失效 (计数+1) 且结果更新."""
    m = _tri_plus_isolated()
    probe = _install_counter(monkeypatch)
    m.check_rigid_body_constraints()
    assert probe.calls == 1

    m.fix_nodes_func([0, 1, 2], 0.0)
    issues = m.check_rigid_body_constraints()
    assert probe.calls == 2, "fix_nodes_func 后缓存未失效"
    assert issues == [
        {"component": 1, "nodes": [3],
         "issue": "孤立分量 (1 节点, 无单元连接)"},
    ]


def test_inplace_fixed_dofs_mutation_invalidate(monkeypatch):
    """原地修改 fixed_dofs 数组内容 (绕过 setter/fix_node) → 结果正确.

    数组只读 by design — 测试显式 setflags(write=True) 模拟绕过 API
    的原地写。若缓存存数组引用而非快照拷贝, 数组与自己比较永远
    命中 → 返回陈旧结果 (此测试必红). 内容往返两次, 每次结果都须
    反映新 BC.
    """
    m = _triangle_pair_plus_isolated()
    for n in range(4):
        m.fix_node(n, "both")          # comp0 全约束 (dofs 0-7)
    probe = _install_counter(monkeypatch)

    issues1 = m.check_rigid_body_constraints()
    assert probe.calls == 1
    assert issues1 == [
        {"component": 1, "nodes": [4],
         "issue": "孤立分量 (1 节点, 无单元连接)"},
    ]

    # 原地改写同一数组: 清空 comp0 约束 (dofs 0-7 → 8,9,8,9,...;
    # dofs 8,9 = 孤立节点 4, comp0 一个都占不到 → 无任何约束)
    arr = m.fixed_dofs
    assert not arr.flags.writeable
    arr.setflags(write=True)
    arr[:] = np.array([8, 9] * 4, dtype=int)
    issues2 = m.check_rigid_body_constraints()
    assert probe.calls == 2, "原地修改后缓存未失效 (快照存引用必红)"
    assert issues2[0]["issue"] == "无任何约束 — 保留 3 个刚体模态", (
        "原地修改后结果未反映新 BC (陈旧缓存)")

    # 恢复原约束 → 结果回到充分约束状态
    arr[:] = np.arange(8, dtype=int)
    issues3 = m.check_rigid_body_constraints()
    assert probe.calls == 3
    assert issues3 == issues1


def test_mesh_replacement_invalidate(monkeypatch):
    """replace_nodes/replace_elements/invalidate_cache → 缓存失效.

    结果与旧实现参照一致 (坐标/拓扑变更后 R 矩阵与分量分解都变).
    """
    m = _triangle_pair_plus_isolated()
    m.fix_node(0, "both")
    m.fix_node(3, "both")
    probe = _install_counter(monkeypatch)
    m.check_rigid_body_constraints()
    assert probe.calls == 1

    # 坐标平移 (不动拓扑) — R 矩阵变, 结果必须重算
    m.replace_nodes(m.nodes + np.array([10.0, 0.0]))
    m.check_rigid_body_constraints()
    assert probe.calls == 2, "replace_nodes 后缓存未失效"
    assert m.check_rigid_body_constraints() == \
        _legacy_check_rigid_body_constraints(m)

    # 单元重连: 孤立节点从 4 换到 0 (拓扑变, 分量分解变)
    m.replace_elements(np.array([[1, 2, 3], [2, 3, 4]]))
    m.check_rigid_body_constraints()
    assert probe.calls == 3, "replace_elements 后缓存未失效"
    assert m.check_rigid_body_constraints() == \
        _legacy_check_rigid_body_constraints(m)

    # 显式 invalidate_cache → 下一次必重算
    m.invalidate_cache()
    m.check_rigid_body_constraints()
    assert probe.calls == 4, "invalidate_cache 后缓存未失效"


# ── 4. 性能判别场景 (100k 级网格, 正确性 + 二调命中) ────────────────


def _grid_mesh(n):
    """n×n 三角网格 — n² 节点, 2n(n-1) 单元 (向量化构造, 无 gmsh)."""
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    nodes = np.column_stack([xs.ravel(), ys.ravel()])
    quads = np.arange(n * n).reshape(n, n)
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = quads[i, j], quads[i, j + 1], quads[i + 1, j + 1], quads[i + 1, j]
            tris.append((a, b, d))
            tris.append((b, c, d))
    return Mesh(nodes=nodes, elements=np.array(tris), E=1e6, nu=0.3)


def test_large_grid_correct_and_second_call_cached(monkeypatch):
    """100k 级网格 (90k 节点 ≈ 180k DOF): 结果正确 + 二调拓扑零重算.

    只断言计数与正确性 (墙钟时间断言在 CI 机器上会抖动误报 —
    首调/二调耗时对比为汇报项, 见性能对比表).
    """
    m = _grid_mesh(300)                # 90k 节点 / 179,400 单元
    m.fix_node(0, "both")
    m.fix_node(0 + 299, "both")        # 左下 + 右下角
    m.fix_node(299 * 300, "both")      # 左上角 — 非共线 → rank 3
    probe = _install_counter(monkeypatch)

    assert m.check_rigid_body_constraints() == []   # 单连通分量, 约束充分
    assert probe.calls == 1
    assert m.check_rigid_body_constraints() == []
    assert probe.calls == 1, (
        f"100k 级网格二调重算拓扑: 调用 {probe.calls} 次, 期望 1")
