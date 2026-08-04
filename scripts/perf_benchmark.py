"""FEM2D 性能基准 — 各规模组装/求解/后处理时间表 (JSON 输出).

用法::

    python scripts/perf_benchmark.py                 # 默认 1k/10k/100k 单元
    python scripts/perf_benchmark.py --scale 300000  # 指定目标单元规模
    python scripts/perf_benchmark.py --all           # 1k/10k/100k/300k 全档
    python scripts/perf_benchmark.py --out bench.json

每档 (Q4 + CST 规则网格) 分阶段计时: 连接/几何构建、K 组装、求解
(solver 按规模 auto 选择)、应力、L2 恢复、Z2 误差估计; 输出 JSON 供
未来改动对比 (组装/求解时间不退化)。另跑 1e-150 微尺度冒烟确认
性能路径无绝对阈值 (有限结果即可, 不计时)。

JSON 键 (可对比项):
    machine / platform / numpy / scipy — 环境指纹
    scales: [{scale, elem_type, n_elem, n_nodes, n_dof, nnz, solver,
              stages: {connect_ms, assemble_ms, solve_ms, stress_ms,
                       l2_ms, estimate_ms}}]
"""
import argparse
import json
import os
import platform
import sys
import time
import tracemalloc

import numpy as np

# 脚本位于 scripts/ 下 — 基准必须测量本项目代码。editable install 指向
# 其他 worktree 时 sys.path 无 cwd, `python scripts/perf_benchmark.py`
# 会 import 到外部 fem2d 副本 (曾静默测到旧实现, 数据失真)。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fem2d import Mesh, solve
from fem2d.assembly import assemble_sparse_vectorized
from fem2d.error_est import estimate
from fem2d.stress import nodal_L2_projection

# 目标单元规模档位 (实际网格为规则方形, 单元数 ≥ 目标)
DEFAULT_SCALES = (1000, 10_000, 100_000)
ALL_SCALES = (1000, 10_000, 100_000, 300_000)


def _grid(nx, ny, code):
    """规则网格: Q4 为 nx×ny 四边形, CST 为 2×nx×ny 三角形."""
    gx = np.linspace(0.0, 1.0, nx + 1)
    gy = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(gx, gy)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    if code == "CPS3":
        quads = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                quads.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
        tri = []
        for n0, n1, n2, n3 in quads:
            tri.append([n0, n1, n2])
            tri.append([n0, n2, n3])
        return nodes, np.array(tri, dtype=np.int64)
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elems.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    return nodes, np.array(elems, dtype=np.int64)


def _corner_dofs(nx, ny):
    """4 角节点 DOF (刚体模态约束充分)."""
    ncol = nx + 1
    corners = [0, nx, ny * ncol, ny * ncol + nx]
    dofs = []
    for n in corners:
        dofs.extend([2 * n, 2 * n + 1])
    return np.array(dofs, dtype=np.int64)


def _timeit(fn, track_mem):
    if track_mem:
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    out = fn()
    dt = (time.perf_counter() - t0) * 1000.0
    if track_mem:
        peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
        return out, dt, peak_mb
    return out, dt, None


def run_scale(scale, code, track_mem=False):
    """跑一个规模档位, 返回 stage 时间 (可选内存峰值) dict."""
    nx = int(np.ceil(np.sqrt(scale)))
    if code == "CPS3":
        nx = int(np.ceil(np.sqrt(scale / 2.0)))
    nodes, elems = _grid(nx, nx, code)
    n_dof = 2 * len(nodes)
    mesh = Mesh(nodes, elems, E=2.1e11, nu=0.3, plane_type="stress",
                fixed_dofs=_corner_dofs(nx, nx), elem_type=code)
    F = np.zeros(n_dof)
    F[2 * nx] = 1e6  # 右下角 x 向集中力

    if track_mem:
        tracemalloc.start()
    _, t_connect, m_connect = _timeit(
        lambda: mesh.build_connectivity(), track_mem)
    K, t_assemble, m_assemble = _timeit(
        lambda: assemble_sparse_vectorized(mesh), track_mem)
    nnz = int(K.nnz)
    res, t_solve, m_solve = _timeit(
        lambda: solve(mesh, verbose=False), track_mem)  # auto: ≥10 万 DOF 用 CG
    _, t_stress, m_stress = _timeit(
        lambda: mesh.element_kernel.compute_response(
            mesh, res["u"][mesh.element_dofs]), track_mem)
    _, t_l2, m_l2 = _timeit(
        lambda: nodal_L2_projection(mesh, res["stress"]), track_mem)
    _, t_est, m_est = _timeit(
        lambda: estimate(mesh, res, verbose=False), track_mem)
    if track_mem:
        tracemalloc.stop()
    stages = {
        "connect_ms": round(t_connect, 3),
        "assemble_ms": round(t_assemble, 3),
        "solve_ms": round(t_solve, 3),
        "stress_ms": round(t_stress, 3),
        "l2_ms": round(t_l2, 3),
        "estimate_ms": round(t_est, 3),
    }
    if track_mem:
        stages.update({
            "connect_peak_mb": round(m_connect, 1),
            "assemble_peak_mb": round(m_assemble, 1),
            "solve_peak_mb": round(m_solve, 1),
            "stress_peak_mb": round(m_stress, 1),
            "l2_peak_mb": round(m_l2, 1),
            "estimate_peak_mb": round(m_est, 1),
        })
    return {
        "scale": scale, "elem_type": code,
        "n_elem": int(mesh.n_elements), "n_nodes": int(mesh.n_nodes),
        "n_dof": n_dof, "nnz": nnz,
        "solver": res.get("linear_solver", "unknown")["name"],
        "stages": stages,
    }


def micro_scale_smoke():
    """1e-150 几何冒烟: 性能路径必须无绝对阈值 (全有限, 不崩溃)."""
    nx = 20
    nodes, elems = _grid(nx, nx, "CPS4")
    nodes = nodes * 1e-150
    mesh = Mesh(nodes, elems, E=2.1e11, nu=0.3, plane_type="stress",
                fixed_dofs=_corner_dofs(nx, nx), elem_type="CPS4")
    mesh.build_connectivity()
    assemble_sparse_vectorized(mesh)
    F = np.zeros(mesh.n_dof)
    F[2 * nx] = 1e-150
    res = solve(mesh, verbose=False)
    s = mesh.element_kernel.compute_response(
        mesh, res["u"][mesh.element_dofs])
    nodal_L2_projection(mesh, res["stress"])
    eta = estimate(mesh, res, verbose=False)["eta"]
    if not (np.all(np.isfinite(res["u"])) and np.all(np.isfinite(s[0]))
            and np.isfinite(eta)):
        raise RuntimeError("微尺度冒烟产生 NaN/Inf — 性能路径存在绝对阈值")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", type=int, default=None,
                    help="单档单元规模 (如 300000)")
    ap.add_argument("--all", action="store_true", help="1k/10k/100k/300k 全档")
    ap.add_argument("--mem", action="store_true",
                    help="同时记录每阶段内存峰值 (tracemalloc, 时间略偏大)")
    ap.add_argument("--out", default="perf_results.json",
                    help="JSON 输出路径 (默认 perf_results.json)")
    args = ap.parse_args()
    if args.scale:
        scales = (args.scale,)
    elif args.all:
        scales = ALL_SCALES
    else:
        scales = DEFAULT_SCALES

    import scipy

    out = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scales": [],
    }
    for scale in scales:
        for code in ("CPS4", "CPS3"):
            print(f"[bench] {code} ~{scale} 单元 ...", flush=True)
            out["scales"].append(run_scale(scale, code, track_mem=args.mem))

    print("[bench] 微尺度冒烟 (1e-150 几何) ...", flush=True)
    micro_scale_smoke()
    out["micro_scale_smoke_1e-150"] = "ok"

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[bench] 结果写入 {args.out}")


if __name__ == "__main__":
    sys.exit(main())
