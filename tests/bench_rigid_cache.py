"""P-β 性能对比基准 — 刚体模态检查向量化 + 结果缓存 (首调/二调).

用法: python tests/bench_rigid_cache.py
无 gmsh 依赖 (手写网格数组); 不做墙钟断言 — CI 机器上时间抖动大,
判别性由 tests/test_p_beta_rigid_cache.py 的 connected_components
调用计数测试承担 (二调计数必须 == 1)。本脚本输出对比表:

    legacy (旧实现: 逐节点循环 + 每次全量拓扑)   <A> ms
    新实现首调 (向量化 + 冷缓存)                 <B> ms
    新实现二调 (缓存命中)                        <C> ms  ← 应接近 0
    connected_components 单独                     <D> ms  ← 首调主导项

100k 级网格: 300×300 = 90k 节点 / 179,400 单元 / 180k DOF。
"""
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# 直接 `python tests/bench_rigid_cache.py` 时 sys.path[0]=tests/ —
# 与 scripts/ 下脚本同款引导 (editable install 指向其他 worktree 时
# 会静默测到旧实现, 数据失真)。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fem2d import Mesh
from fem2d.topology_core import build_edge_table


def grid_mesh(n):
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


def apply_bcs(mesh, n):
    """三处非共线角点全约束 (左下/右下/左上) — 约束充分, 返回空 issues."""
    mesh.fix_node(0, "both")
    mesh.fix_node(n - 1, "both")
    mesh.fix_node(n * (n - 1), "both")


def legacy_scan(mesh):
    """P-β 前实现 (逐节点 Python 循环扫约束 DOF) — 计时参照.

    与旧 check_rigid_body_constraints 相同: build_connectivity
    (含边表, 首次全量预处理) + csr + connected_components +
    逐节点 set 查询循环。
    """
    mesh.build_connectivity()
    fixed = mesh.fixed_dofs
    fixed_set = set(fixed.tolist())
    n_nodes = mesh.n_nodes
    edge_table = build_edge_table(
        mesh.elements, mesh.element_kernel.local_edges, n_nodes)
    row = np.concatenate([edge_table.lo, edge_table.hi])
    col = np.concatenate([edge_table.hi, edge_table.lo])
    data = np.ones(len(row), dtype=int)
    adj = csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    n_comp, labels = connected_components(adj, directed=False)
    total = 0
    for comp in range(n_comp):
        comp_nodes = np.where(labels == comp)[0]
        for node in comp_nodes:
            if (2 * node) in fixed_set:
                total += 1
            if (2 * node + 1) in fixed_set:
                total += 1
    return total


def ms(t_sec):
    return f"{t_sec * 1e3:8.2f}"


def main():
    n = 300
    # 旧实现: 每次调用全量拓扑 + 循环
    m_legacy = grid_mesh(n)
    apply_bcs(m_legacy, n)
    t0 = time.perf_counter()
    legacy_res = legacy_scan(m_legacy)
    t_legacy = time.perf_counter() - t0

    # 新实现首调: 向量化 + 冷缓存 (含 build_connectivity + 连通分解)
    m_new = grid_mesh(n)
    apply_bcs(m_new, n)
    t0 = time.perf_counter()
    issues_first = m_new.check_rigid_body_constraints()
    t_first = time.perf_counter() - t0

    # 新实现二调: 缓存命中 (BC/网格未变)
    t0 = time.perf_counter()
    issues_second = m_new.check_rigid_body_constraints()
    t_second = time.perf_counter() - t0

    # connected_components 单独 — 首调的主导项
    m_cc = grid_mesh(n)
    m_cc.build_connectivity()
    edge_table = build_edge_table(
        m_cc.elements, m_cc.element_kernel.local_edges, m_cc.n_nodes)
    row = np.concatenate([edge_table.lo, edge_table.hi])
    col = np.concatenate([edge_table.hi, edge_table.lo])
    data = np.ones(len(row), dtype=int)
    adj = csr_matrix((data, (row, col)), shape=(m_cc.n_nodes, m_cc.n_nodes))
    t0 = time.perf_counter()
    connected_components(adj, directed=False)
    t_cc = time.perf_counter() - t0

    assert legacy_res == 6 and not issues_first and not issues_second
    print(f"网格: {n * n} 节点 / {len(m_new.elements)} 单元 / "
          f"{2 * n * n} DOF")
    print(f"legacy 旧实现 (逐节点循环, 全量拓扑) : {ms(t_legacy)} ms"
          f"  (constraint dofs found={legacy_res})")
    print(f"新实现首调 (向量化 + 冷缓存)        : {ms(t_first)} ms"
          f"  (issues={len(issues_first)})")
    print(f"新实现二调 (缓存命中)               : {ms(t_second)} ms"
          f"  (issues={len(issues_second)}, 应接近 0)")
    print(f"connected_components 单独            : {ms(t_cc)} ms"
          f"  (参考: 首调中 build_connectivity 全量预处理才为主导)")
    speedup = t_legacy / t_second if t_second > 0 else float("inf")
    print(f"二调 vs legacy: {speedup:.0f}x; "
          f"首调 vs legacy: {t_legacy / t_first:.2f}x")


if __name__ == "__main__":
    main()
