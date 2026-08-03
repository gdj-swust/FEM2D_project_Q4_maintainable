"""平面应力/应变验证 — 单向拉伸 + 厚壁圆筒内压

从 run.py 导入: from fem2d.verification import run_plane_verification
(2026-08-02 自 tests/verify_plane.py 迁入: wheel 打包不含 tests,
--self-test 不能依赖 tests 包; tests/verify_plane.py 保留为薄壳。)
"""
import sys

import numpy as np

from fem2d import Mesh, solve
from fem2d.stress import principal_stresses


def run_plane_verification():
    """返回 (pass_count, fail_count)

    Test 1: 单向拉伸 — plane stress vs plane strain
            纯直角坐标，验证 D 矩阵和 von Mises。
    Test 2: 厚壁圆筒内压 — Lame 解析解 (plane strain)
            应力张量旋转: σ_rr = σ_xx·cos²θ + σ_yy·sin²θ + 2τ_xy·sinθcosθ
    """
    PASS, FAIL = 0, 0

    def check(name, computed, theory, tol=0.05):
        nonlocal PASS, FAIL
        # 分母地板用 tiny 而非固定 1e-30 — 理论值恰为零时 1e-30 会使
        # 微小误差被误判为 0 通过 (微尺度约定: 禁止绝对阈值)。
        rel = abs(computed - theory) / (abs(theory) + np.finfo(float).tiny)
        status = "PASS" if rel < tol else "FAIL"
        print(f"  {status}  {name}: err={rel*100:.1f}%  ({computed:.4e} vs {theory:.4e})")
        if status == "PASS":
            PASS += 1
        else:
            FAIL += 1

    # ═══════════════════════════════════════════════════════
    # Test 1: 单向拉伸 (纯直角坐标)
    # ═══════════════════════════════════════════════════════
    print("\n  Test 1: 单向拉伸 — plane stress vs plane strain")
    L, H, t = 1.0, 0.5, 0.01
    E, nu, sigma = 2.1e11, 0.3, 1e6
    nx, ny = 4, 2

    nodes = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes.append([L * i / nx, H * j / ny])
    nodes = np.array(nodes)
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elems.append([n0, n0 + 1, n0 + nx + 2])
            elems.append([n0, n0 + nx + 2, n0 + nx + 1])
    elems = np.array(elems, dtype=int)

    for pt in ("stress", "strain"):
        m = Mesh(nodes=nodes, elements=elems, E=E, nu=nu, thickness=t,
                 plane_type=pt, elem_type="CST")
        # 最小约束: 左端仅 ux=0 + 单点 uy=0 防刚体。禁止对左端全固定 —
        # 那会阻止泊松收缩, 在固定端引入圣维南弯曲边界层, 使"单向拉伸"
        # 不再有解析解 (旧版测得 0.8% 假误差即源于此)。
        for n in range(len(nodes)):
            if abs(nodes[n, 0]) < 1e-10:
                m.fix_node(n, "x")
            if abs(nodes[n, 0]) < 1e-10 and abs(nodes[n, 1]) < 1e-10:
                m.fix_node(n, "y")
        n_right = sorted([n for n in range(len(nodes))
                           if abs(nodes[n, 0] - L) < 1e-10],
                         key=lambda n: nodes[n, 1])
        for a, b in zip(n_right, n_right[1:]):
            m.add_traction(a, b, sigma, 0)

        r = solve(m, verbose=False)
        u_fe = np.max(np.abs(r["u"]))
        vm_fe = r["vm_stress"].max() * 1e-6

        if pt == "stress":
            u_th, vm_th = sigma * L / E, sigma * 1e-6
        else:
            u_th = sigma * L / E * (1 - nu**2)
            vm_th = np.sqrt(0.5 * (1 + nu**2 + (1 - nu)**2))

        check(f"plane_{pt} u_max", u_fe, u_th)
        check(f"plane_{pt} σ_vM ", vm_fe, vm_th)

    # ═══════════════════════════════════════════════════════
    # Test 2: 厚壁圆筒内压 — Lame (plane strain) + 张量旋转
    # ═══════════════════════════════════════════════════════
    print("\n  Test 2: 厚壁圆筒内压 — Lame 解 + 应力旋转 (plane strain)")
    a, b_out, p = 1.0, 2.0, 1e6
    nr, nth = 16, 72

    nodes = []
    for i in range(nth):
        ang = 2 * np.pi * i / nth
        ca, sa = np.cos(ang), np.sin(ang)
        for j in range(nr + 1):
            r = a + (b_out - a) * j / nr
            nodes.append([r * ca, r * sa])
    nodes = np.array(nodes)

    elems = []
    for i in range(nth):
        i_next = (i + 1) % nth
        for j in range(nr):
            n0 = i * (nr + 1) + j
            n2 = i_next * (nr + 1) + j + 1
            n3 = i_next * (nr + 1) + j
            elems.append([n0, n0 + 1, n2])
            elems.append([n0, n2, n3])
    elems = np.array(elems, dtype=int)

    m = Mesh(nodes=nodes, elements=elems, E=E, nu=nu, thickness=1.0,
             plane_type="strain", elem_type="CST")
    # 最小约束 — 全部沿切向, 不约束任何节点的径向位移:
    # 内压自由膨胀下 Lame 解 u_r(a) ≠ 0, 旧版把内边界节点 0 两方向
    # 都固定 → 该点径向位移被强制为零, 与解析解冲突, 污染内环检查区
    # (σ_θ/σ_r 误差虚高)。切向三元组: (a,0) 与 (b,0) 的 uy (θ=0 处
    # 切向 = +y) 杀 Ty+Rz (a≠b 两式联立), (0,a) 的 ux (θ=π/2 处
    # 切向 = -x) 杀 Tx — 刚体模态全消, 径向解完整保留。
    m.fix_node(0, "y")                       # (a, 0): 切向
    m.fix_node(nr, "y")                      # (b, 0): 切向, 联立上式杀 Ty/Rz
    m.fix_node((nth // 4) * (nr + 1), "x")   # (0, a): θ=π/2 切向, 杀 Tx

    m.build_connectivity()
    for ea, eb in m.boundary_edges:
        ra, rb = np.linalg.norm(nodes[ea]), np.linalg.norm(nodes[eb])
        if abs(ra - a) < 0.06 and abs(rb - a) < 0.06:
            m.add_pressure(int(ea), int(eb), p)

    r = solve(m, verbose=False)

    # Lame: σ_θ(r)=p·a²/(b²-a²)·(1+b²/r²), σ_r(r)=p·a²/(b²-a²)·(1-b²/r²)
    factor = p * a**2 / (b_out**2 - a**2)
    rc = np.linalg.norm(m.centroids, axis=1)
    inner = rc < a + 0.06

    # 应力张量旋转: [σ_rr, σ_θθ, τ_rθ]^T = T(θ) · [σ_xx, σ_yy, τ_xy]^T
    # σ_rr = σ_xx·cos²θ + σ_yy·sin²θ + 2τ_xy·cosθ·sinθ
    err_t, err_r = [], []
    for eid in np.flatnonzero(inner):
        s1, _, _, _ = principal_stresses(r["stress"][eid:eid+1])
        sx, sy, txy = r["stress"][eid]
        xc, yc = m.centroids[eid]
        r_c = np.linalg.norm([xc, yc])
        ang = np.arctan2(yc, xc)
        c, s = np.cos(ang), np.sin(ang)

        theta_fe = s1[0]                                     # σ_θθ ≈ σ_1 (主应力)
        theta_th = factor * (1 + b_out**2 / r_c**2)
        err_t.append(abs(theta_fe - theta_th) / theta_th)

        r_fe = sx*c*c + sy*s*s + 2*txy*c*s                   # σ_rr = 张量旋转
        r_th = factor * (1 - b_out**2 / r_c**2)
        err_r.append(abs(r_fe - r_th)
                     / (abs(r_th) + np.finfo(float).tiny))

    err_t = np.mean(err_t) if err_t else 1.0
    err_r = np.mean(err_r) if err_r else 1.0
    print(f"  {'PASS' if err_t<0.10 else 'FAIL'}  σ_θ (主应力): err={err_t*100:.1f}%")
    print(f"  {'PASS' if err_r<0.10 else 'FAIL'}  σ_r (张量旋转): err={err_r*100:.1f}%")
    if err_t < 0.10: PASS += 1
    else: FAIL += 1
    if err_r < 0.10: PASS += 1
    else: FAIL += 1

    return PASS, FAIL


if __name__ == "__main__":
    p, f = run_plane_verification()
    print("\n" + "=" * 55)
    print(f"  {p} PASS, {f} FAIL")
    print("=" * 55)
    sys.exit(0 if f == 0 else 1)
