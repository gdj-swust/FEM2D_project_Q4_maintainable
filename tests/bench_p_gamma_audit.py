"""P-gamma query-path audit -- EdgeTable.index_of dynamic measurement.

任务书任务 2: 查询路径审计 (index_of / EdgeIncidence). 本脚本用
monkeypatch 计数 EdgeTable.index_of 的调用次数与耗时占比, 覆盖
端到端典型流程 (Mesh + BC + solve + error estimate + refinement
indicator), 并按网格规模验证调用次数 scaling.

结论 (可复现): index_of 是 O(log E) 二分查找, 调用点只在
fem2d/error_est.py 的残差型路径 (element_refinement_indicator
内部 _collect_loaded_edges/_neumann_edge_residuals) 与 solve 的
面力组装 — 每次调用成本 ~4 us, 调用次数 = O(边界边数), 与单元
总数无关 (50x25->200, 100x50->400, 200x100->800 次). 低频
(每个 estimate 调用一次), 非内循环 — 保持不动, 不批量化:
批量化收益上限 ~2 ms/次调用, 且调用点在禁碰模块 error_est.py,
其逐边 fail-掩码语义与批量路径冲突.

用法: python tests/bench_p_gamma_audit.py
无 gmsh 依赖 (手写网格数组); 不做墙钟断言 (判别性由等价测试承担).
"""
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fem2d import Mesh
from fem2d import solver, error_est
from fem2d.topology_core import EdgeTable


def _quad_mesh(nx, ny):
    """nx x ny CPS4 网格 (0..2 x 0..1)."""
    xs = np.linspace(0.0, 2.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def idx(i, j):
        return i * (ny + 1) + j

    quads = np.array(
        [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
         for i in range(nx) for j in range(ny)], dtype=np.int64)
    return nodes, quads


class _Counter:
    """monkeypatch EdgeTable.index_of, 统计调用次数与查找耗时."""

    def __init__(self):
        self.n = 0
        self.t = 0.0
        self._orig = EdgeTable.index_of

    def __enter__(self):
        def counting(table, a, b):
            self.n += 1
            t0 = time.perf_counter()
            r = self._orig(table, a, b)
            self.t += time.perf_counter() - t0
            return r

        EdgeTable.index_of = counting
        return self

    def __exit__(self, *exc):
        EdgeTable.index_of = self._orig


def _model(nx, ny):
    """典型教学模型: 底边固定 + 右边界压力."""
    nodes, elements = _quad_mesh(nx, ny)
    ny_ = ny

    def idx(i, j):
        return i * (ny_ + 1) + j

    mesh = Mesh(nodes=nodes, elements=elements, E=210e9, nu=0.3,
                elem_type="CPS4")
    for i in range(nx + 1):
        mesh.fix_node(idx(i, 0), "both")
    for j in range(ny_):
        mesh.add_pressure(idx(nx, j), idx(nx, j + 1), 1e6)
    mesh.build_connectivity()
    return mesh


def main():
    print("audit: EdgeTable.index_of per pipeline stage "
          "(100x50 mesh, 5000 elems, bottom fixed + right pressure)")
    mesh = _model(100, 50)
    result = solver.solve(mesh, verbose=False)

    stages = [
        ("build_connectivity", lambda: mesh.build_connectivity()),
        ("solve (incl. BC assembly)", lambda: solver.solve(mesh, verbose=False)),
        ("estimate SPR", lambda: error_est.estimate(mesh, result, method="SPR", verbose=False)),
        ("estimate L2", lambda: error_est.estimate(mesh, result, method="L2", verbose=False)),
        ("estimate weighted", lambda: error_est.estimate(mesh, result, method="weighted", verbose=False)),
        ("refinement indicator (residual)", lambda: error_est.element_refinement_indicator(mesh, result)),
    ]
    for name, fn in stages:
        with _Counter() as c:
            t0 = time.perf_counter()
            fn()
            dt = time.perf_counter() - t0
        pct = c.t / dt * 100
        print(f"  {name:36s} {dt * 1000:9.2f} ms   index_of x{c.n:6d}"
              f"   lookup {c.t * 1000:8.3f} ms ({pct:5.2f}%)")

    print("\naudit: call-count scaling with mesh size "
          "(refinement indicator, bottom fixed + right pressure)")
    for nx, ny in ((50, 25), (100, 50), (200, 100)):
        mesh = _model(nx, ny)
        result = solver.solve(mesh, verbose=False)
        with _Counter() as c:
            error_est.element_refinement_indicator(mesh, result)
        print(f"  {nx:4d}x{ny:3d}  elems={nx * ny:6d}"
              f"  boundary_edges~{2 * (nx + ny):5d}  index_of x{c.n}")

    print("\nconclusion: O(log E) binary search; calls = O(boundary edges),"
          "\nnot O(elements); low frequency (once per estimate call); keep.")


if __name__ == "__main__":
    main()
