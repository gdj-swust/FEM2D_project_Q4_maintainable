"""终轮回归 — 端到端组合 fuzz (第三视角).

组合: 单元 (CST/Q4/Q4I/Q4R) × 尺度 (正常/1e-9/1e12 偏移) × 载荷
(固定+拉力 / +压力 / +体力 / +集中力 / 全组合) = 60 组合。
同一份结构化网格覆写单元类型与节点坐标,
每个组合全链路 solve + 误差估计, 抓: 崩溃 / NaN/Inf / 静默零解 /
平衡残差。载荷边恒取右边自由边 (曾取 boundary_edges[0], 与固定边
选取无关但可能任意; 固定边加载会静默改变工况)。
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

from fem2d import (
    Mesh,
    assemble_loads,
    assemble_sparse,
    estimate_error,
    solve,
)

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


def _equilibrium_scale(mesh, u):
    """相对尺度判定: 特征位移 + 全局平衡 (禁止绝对阈值).

    特征位移 u_char = max|F| / median|K_diag| — 与 u 同量纲, 微尺度/
    大坐标模型自动跟随输入尺度 (1e-150 模型 u 极小但 u_char 同步缩小,
    正常解恒在 u_char 的 O(1) 邻域)。
    全局平衡: 外载荷 + 支反力自平衡 ⟺ Σ(K·u) ≈ 0; 载荷若静默施加到
    固定 DOF 或丢失, 平衡即破缺。
    """
    F_vec = assemble_loads(mesh, mesh.n_dof)
    K = assemble_sparse(mesh)
    k_scale = float(np.median(np.abs(K.diagonal())))
    f_max = float(np.max(np.abs(F_vec)))
    u_char = f_max / max(k_scale, np.finfo(float).tiny)
    load_norm = float(np.sum(np.abs(F_vec)))
    rel_net = float(np.abs(np.sum(K @ u))) / max(
        load_norm, np.finfo(float).tiny)
    return u_char, float(np.max(np.abs(u))), rel_net


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
    # 载荷边: 恒取右边自由边 — boundary_edges[0] 是任一边 (曾可能是
    # 被固定边, 载荷施加到固定 DOF 被消去, 工况静默改变).
    bdy = max(
        mesh.boundary_edges,
        key=lambda e: (float(mesh.nodes[int(e[0])][0])
                       + float(mesh.nodes[int(e[1])][0])))
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
    u_char, u_max, rel_net = _equilibrium_scale(mesh, u)
    if u_char > 0 and u_max <= u_char * 1e-10:
        # 静默零解: 载荷被丢弃时 isfinite 全过, 只有相对尺度能抓
        return (f"静默零解: elem={elem_type} scale={scale} "
                f"load={load_kind} (max|u|={u_max:.3e}, "
                f"u_char={u_char:.3e})")
    if rel_net > 1e-6:
        # 平衡破缺: 载荷施加方向/位置错误时解仍有限且非零
        return (f"平衡残差: elem={elem_type} scale={scale} "
                f"load={load_kind} (Σ|Ku|/Σ|F|={rel_net:.3e})")
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
