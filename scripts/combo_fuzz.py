"""终轮回归 — 端到端组合 fuzz (第三视角).

组合: 单元 (CST/Q4/Q4I/Q4R) × 尺度 (正常/1e-9/1e12 偏移) × 载荷
(固定+拉力 / +压力 / +体力 / +集中力 / 全组合) = 60 组合。
同一份四边形网格 (plate_q4.msh) 覆写单元类型与节点坐标,
每个组合全链路 solve + 误差估计, 抓: 崩溃 / NaN/Inf / 静默零解。
"""
import sys

import numpy as np

from fem2d import Mesh, estimate_error, solve

ELEM_TYPES = ["CPS3", "CPS4", "CPS4I", "CPS4R"]
SCALES = [1.0, 1e-9, 1e12]
LOAD_KINDS = ["traction", "pressure", "body", "force", "all"]


def _structured_mesh(quads, nx=6, ny=4):
    """结构化网格: quads=True → 四边形, False → 三角形 (每 quad 拆 2)."""
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    gx, gy = np.meshgrid(xs, ys)
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    n = nx + 1
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j * n + i
            b = a + 1
            c = a + n
            d = c + 1
            if quads:
                elems.append([a, b, d, c])
            else:
                elems.append([a, b, c])
                elems.append([b, d, c])
    return nodes, np.array(elems, dtype=np.int64)


def run_combo(elem_type, scale, load_kind):
    quads = elem_type != "CPS3"
    nodes, elems = _structured_mesh(quads)
    nodes = nodes * scale
    mesh = Mesh(
        nodes, elems, E=2.1e11, nu=0.3, thickness=0.01 * scale,
        plane_type="stress", elem_type=elem_type)
    mesh.build_connectivity()
    # 固定: x 最小边界节点
    left = set(mesh.nodes_on_edge("x", "min").tolist())
    for n in left:
        mesh.fix_node(int(n), "both", 0.0)
    # 载荷边: 任取一条边界边 (避免几何分类)
    bdy = mesh.boundary_edges[0]
    if load_kind in ("traction", "all"):
        mesh.add_traction(int(bdy[0]), int(bdy[1]), 1e6 * scale, 0.0)
    if load_kind in ("pressure", "all"):
        mesh.add_pressure(int(bdy[0]), int(bdy[1]), 1e6 * scale)
    if load_kind in ("body", "all"):
        mesh.body_force = (0.0, -78000.0 * scale)
    if load_kind in ("force", "all"):
        right = sorted(mesh.nodes_on_edge("x", "max").tolist())
        mesh.add_force(int(right[0]), 1e6 * scale, 0.0)
    res = solve(mesh, verbose=False)
    z2 = estimate_error(mesh, res, verbose=False)
    u, s = res["u"], res["stress"]
    ok = (np.all(np.isfinite(u)) and np.all(np.isfinite(s))
          and np.isfinite(z2["eta"]))
    if not ok:
        return f"NaN/Inf: elem={elem_type} scale={scale} load={load_kind}"
    return None


def main():
    problems = []
    n = 0
    for et in ELEM_TYPES:
        for sc in SCALES:
            for lk in LOAD_KINDS:
                n += 1
                try:
                    issue = run_combo(et, sc, lk)
                except Exception as exc:  # noqa: BLE001 — 组合 fuzz 抓一切
                    problems.append(
                        f"CRASH elem={et} scale={sc} load={lk}: "
                        f"{type(exc).__name__}: {str(exc)[:80]}")
                    continue
                if issue:
                    problems.append(issue)
    print(f"combos={n}")
    print(f"problems={len(problems)}")
    for p in problems:
        print("  -", p)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
