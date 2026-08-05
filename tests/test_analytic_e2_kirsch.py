"""E2 解析解验证 — Kirsch 圆孔应力集中 (无限板远场拉应力 σ∞).

解析解 (Kirsch 1898, 标准弹性力学):
    σθθ = σ∞/2·(1 + a²/r² − (1 + 3a⁴/r⁴)·cos2θ)
    σθθ,max = 3σ∞ (孔边 θ=π/2),  应力集中系数 Kt = σθθ,max/σ∞ = 3

有限板大域近似 (任务书方案): 板宽 2w = 20a ≥ 10a, 1/4 板 [0,w]².
w/a=10 时有限宽效应把 Kt 放大到 ~3.13 (+4.4%, Howland 型有限宽修正
方向), 仍满足 |Kt−3|/3 < 10% 门槛 — 依据见 docs/analytic_verification.md。

模型: 孔心在原点, 径向网格: 射线从孔边 (r=a) 打到方形外边
(r_max(θ) = w/max(|cosθ|,|sinθ|))。对称 BC (y=0 轴: uy=0, x=0 轴:
ux=0), 右缘 x=w 逐边施加均匀 σ∞, 上缘自由。平面应力, E=210 GPa,
ν=0.3, t=1, σ∞=1 MPa。

FEM Kt: 孔边一圈单元 (任一节点在 r=a) 的质心 σθθ 最大值 / σ∞。
σθθ 由应力张量旋转换算: σθθ = σx·sin²θ + σy·cos²θ − 2τxy·sinθ·cosθ。

判别性 (回滚必红):
  - 收敛: Kt 随加密稳定 (相邻层差递减), 最细层 |Kt−3|/3 < 10%
  - 方向性: 角度网格相位平移半格, Kt 变化 < 10% (网格方向不敏感)
"""
import numpy as np

from fem2d import Mesh, solve

A, W = 1.0, 10.0
SIGMA_INF = 1e6
LEVELS = ((32, 16), (64, 32), (128, 64))  # (n_ang, n_rad)
KT_TOL = 0.10  # |Kt−3|/3 相对门槛


def _kirsch_mesh(n_ang, n_rad, phase=0.0):
    """1/4 板径向网格 → (Mesh, nodes).

    phase: 角度网格相位平移 (方向性实验用, 平移半个扇区).
    """
    nodes = []
    for i in range(n_ang + 1):
        th = 0.5 * np.pi * (i + phase) / n_ang
        c, s = np.cos(th), np.sin(th)
        rmax = W / max(abs(c), abs(s))
        for j in range(n_rad + 1):
            r = A + (rmax - A) * j / n_rad
            nodes.append([r * c, r * s])
    nodes = np.array(nodes)
    elems = []
    for i in range(n_ang):
        for j in range(n_rad):
            n0 = i * (n_rad + 1) + j
            n1 = n0 + 1
            n2 = (i + 1) * (n_rad + 1) + j + 1
            n3 = n2 - 1
            elems.append([n0, n1, n2])
            elems.append([n0, n2, n3])
    elems = np.array(elems, dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=1.0,
             plane_type="stress", elem_type="CST")
    for j in range(n_rad + 1):
        m.fix_node(j, "y")                        # θ=0 轴 (y=0): uy=0
        m.fix_node(n_ang * (n_rad + 1) + j, "x")  # θ=π/2 轴 (x=0): ux=0
    return m, nodes


def _solve_kirsch(n_ang, n_rad, phase=0.0):
    m, nodes = _kirsch_mesh(n_ang, n_rad, phase)
    # 右缘 x=w: 逐边施加均匀远场拉力 σ∞
    right = sorted(np.flatnonzero(np.abs(nodes[:, 0] - W) < 1e-9),
                   key=lambda n: nodes[n, 1])
    for a_, b_ in zip(right, right[1:]):
        m.add_traction(int(a_), int(b_), SIGMA_INF, 0.0)
    result = solve(m, method="elimination", verbose=False)
    # 孔边一圈单元 (任一节点在 r=a) 的质心 σθθ 最大值
    kt = 0.0
    for eid, conn in enumerate(m.elements):
        if not np.any(np.abs(np.linalg.norm(nodes[conn], axis=1) - A) < 1e-9):
            continue
        sx, sy, txy = result["stress"][eid]
        xc, yc = m.centroids[eid]
        thc = np.arctan2(yc, xc)
        s_theta = (sx * np.sin(thc)**2 + sy * np.cos(thc)**2
                   - 2 * txy * np.sin(thc) * np.cos(thc))
        kt = max(kt, s_theta)
    return kt / SIGMA_INF


def test_kirsch_kt_converges_toward_3():
    """Kt 随加密收敛 (相邻层差递减), 最细层 |Kt−3|/3 < 10% (判别)."""
    kt = [_solve_kirsch(na, nr) for na, nr in LEVELS]
    diffs = [abs(b - a) for a, b in zip(kt, kt[1:])]
    assert all(b < a for a, b in zip(diffs, diffs[1:])), \
        f"Kt 序列未收敛稳定: {kt}, 层间差 {diffs}"
    rel = abs(kt[-1] - 3.0) / 3.0
    assert rel < KT_TOL, \
        f"最细层 Kt={kt[-1]:.4f}, |Kt−3|/3={rel:.1%} ≥ 10%"


def test_kirsch_kt_mesh_direction_insensitive():
    """角度网格相位平移半格后 Kt 变化 < 10% (网格方向不敏感, 判别)."""
    kt0 = _solve_kirsch(64, 32, phase=0.0)
    kt1 = _solve_kirsch(64, 32, phase=0.5)
    for name, val in (("phase=0", kt0), ("phase=0.5", kt1)):
        rel = abs(val - 3.0) / 3.0
        assert rel < KT_TOL, \
            f"{name} Kt={val:.4f}, |Kt−3|/3={rel:.1%} ≥ 10%"
    spread = abs(kt0 - kt1) / 3.0
    assert spread < KT_TOL, \
        f"网格方向敏感: Kt={kt0:.4f} vs {kt1:.4f}, 差 {spread:.1%} ≥ 10%"
