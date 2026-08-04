"""终轮回归 — 数值漂移对照脚本 (早期 baseline vs 当前 main 共用).

用法: python scripts/regression_compare.py <model.msh> <E> <nu> <t> <fx> <fy> <method>
  - 左边 (x=min) 全部节点固定, 右边 (x=max) 的边界边施加均布面力 (fx, fy)
  - 输出: max|u| 应变能 eta 应力峰值 max|σx|/max|σy|/max|τxy| (tab 分隔)
不依赖边界几何分类 — 只用坐标 + 纯拓扑 (add_traction 的边界边检查),
保证两个版本施加相同的 BC。
"""
import os
import sys

import numpy as np

# 脚本位于 scripts/ 下 — 审计必须针对本项目代码。editable install 指向
# 其他 worktree 时 sys.path 无 cwd, `python scripts/xxx.py` 会 import 到
# 外部 fem2d 副本 (曾静默测到旧实现, 数据失真)。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fem2d import Mesh, estimate_error, solve
from fem2d.gmsh_adapter import import_msh


def main():
    msh_path, E, nu, t, fx, fy, method = sys.argv[1:8]
    g = import_msh(msh_path, plane_type="stress")
    mesh = Mesh(
        g.nodes, g.elements, E=float(E), nu=float(nu), thickness=float(t),
        plane_type="stress", elem_type=g.elem_type)
    mesh.build_connectivity()

    left = set(mesh.nodes_on_edge("x", "min").tolist())
    right = set(mesh.nodes_on_edge("x", "max").tolist())
    for n in left:
        mesh.fix_node(int(n), "both", 0.0)
    n_edges = 0
    for a, b in mesh.boundary_edges:
        if a in right and b in right:
            mesh.add_traction(int(a), int(b), float(fx), float(fy))
            n_edges += 1
    assert n_edges > 0, f"右边界未找到面力边: {msh_path}"

    res = solve(mesh, method=method, verbose=False)
    u = res["u"]
    from fem2d.assembly import assemble_sparse
    K = assemble_sparse(mesh)
    strain_energy = 0.5 * float(u @ K.dot(u))
    z2 = estimate_error(mesh, res, verbose=False)
    s = res["stress"]
    print(f"max|u|={np.max(np.abs(u)):.10e} "
          f"energy={strain_energy:.10e} "
          f"eta={z2['eta']:.10e} "
          f"sxmax={np.max(np.abs(s[:, 0])):.10e} "
          f"symax={np.max(np.abs(s[:, 1])):.10e} "
          f"txymax={np.max(np.abs(s[:, 2])):.10e}")


if __name__ == "__main__":
    main()
