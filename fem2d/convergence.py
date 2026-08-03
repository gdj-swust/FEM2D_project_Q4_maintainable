"""CST/Q4/Q4R convergence study — Bathe §4.3.5 / §5.3.3

Verifies displacement-element convergence using the Timoshenko-Goodier
parabolic shear cantilever (smooth solution, no point-load singularity):

  u ~ O(h^2)   displacement
  sigma ~ O(h) stress
  eta ~ O(h)   Z2 energy error estimate

Reference: Richardson-extrapolated FE limit (not beam theory).

Usage: python -m fem2d.convergence
"""
import numpy as np

from .element import get_element_kernel
from .error_est import estimate
from .mesh import Mesh
from .solver import solve
from .stress import point_in_element


def _gen_cantilever_mesh(L, H, nx, ny, elem_type="CPS3"):
    """Generate a structured CST, Q4 or Q4R cantilever mesh."""
    xs = np.linspace(0, L, nx + 1)
    ys = np.linspace(-H / 2, H / 2, ny + 1)

    nodes = []
    for y in ys:
        for x in xs:
            nodes.append([x, y])
    nodes = np.array(nodes)

    npx = nx + 1
    is_quad = get_element_kernel(elem_type).nodes_per_element == 4
    elements = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * npx + i
            n10 = j * npx + (i + 1)
            n01 = (j + 1) * npx + i
            n11 = (j + 1) * npx + (i + 1)
            if is_quad:
                elements.append([n00, n10, n11, n01])
            else:
                elements.append([n00, n10, n11])
                elements.append([n00, n11, n01])
    elements = np.array(elements, dtype=int)
    return nodes, elements


def _timoshenko_tip_deflection(L, H, t, P, E, nu):
    """Timoshenko-Goodier tip deflection for parabolic shear cantilever.

    u_y(L, 0) = PL^3/(3EI) * [1 + 3(1+nu)/(2*(L/H)^2)]

    The second term is the shear correction; vanishes for slender beams.
    P > 0 = downward load => u_y < 0 (downward deflection).
    (公式曾缺负号, 返回正挠度 — 与 docstring/约定矛盾, 方向错误被
    调用方 abs() 掩盖; 2026-08 与 _parabolic_shear_traction 统一
    P>0 = 向下, u_y < 0。)
    """
    I = t * H**3 / 12.0
    eb = P * L**3 / (3.0 * E * I)  # Euler-Bernoulli (P>0 → 下弯)
    shear_correction = 3.0 * (1.0 + nu) / (2.0 * (L / H)**2)
    return -eb * (1.0 + shear_correction)


def _parabolic_shear_traction(y, H, t, P):
    """Parabolic shear stress: tau_xy(y) = -P/(2I)*(H^2/4 - y^2).

    This is the exact traction distribution on the right end face for the
    Timoshenko-Goodier cantilever.  P > 0 = downward total shear
    (约定与 _timoshenko_tip_deflection 统一; 曾注释声称 P<0 为向下,
    实际 FE 载荷被施加成向上 +10000 N, 与理论方向相反)。
    """
    I = t * H**3 / 12.0
    return -P / (2.0 * I) * (H**2 / 4.0 - y**2)


def _find_tip_node(mesh, tol=1e-6):
    """Find the node at the tip (x=max, y~0) of the cantilever."""
    tip_nodes = mesh.nodes_on_edge("x", "max", tol)
    best = None
    best_dy = float('inf')
    for n in tip_nodes:
        nid = int(n)
        y = mesh.nodes[nid, 1]
        if abs(y) < best_dy:
            best_dy = abs(y)
            best = nid
    return best


def run_cantilever_convergence(
        refinements=5, verbose=True, elem_type="CPS3"):
    """Timoshenko parabolic shear cantilever convergence study.

    Structured CST or Q4 mesh, each level refines 2x in each direction.
    Left end: fixed (u=v=0).  Right end: exact parabolic shear traction.

    Reference value: Richardson-extrapolated FE limit from the two finest
    meshes.  The Timoshenko-Goodier closed form is printed for comparison
    but NOT used for rate fitting (avoids the systematic error of comparing
    a 2D FE solution against a beam theory).
    """
    L, H, t = 5.0, 1.0, 0.1
    E, nu = 210e9, 0.3
    P_mag = 10000.0  # total shear magnitude [N]
    I = t * H**3 / 12.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {elem_type} Convergence - Timoshenko Parabolic Shear Cantilever")
        print(f"  L={L}m  H={H}m  t={t}m  E={E:.1e}Pa  nu={nu}  P={P_mag}N")
        print("  Bathe sec 4.3.5: expected O(h) energy/stress, O(h^2) displ")
        print("  Reference: Richardson-extrapolated FE limit")
        print(f"{'='*60}\n")

    # Timoshenko-Goodier closed form (comparison only, NOT the rate reference)
    # P>0 = 向下 (与 _parabolic_shear_traction 同一约定, 2026-08 统一)
    uy_tip_tg = abs(_timoshenko_tip_deflection(L, H, t, P_mag, E, nu))

    results = {"h": [], "uy_tip": [], "sigma_sample": [], "n_dof": [],
               "eta": [], "total_error": []}

    for level in range(refinements):
        nx = 4 * (2 ** level)   # start at nx=4, not 2
        ny = 2 * (2 ** level)

        nodes, elements = _gen_cantilever_mesh(
            L, H, nx, ny, elem_type=elem_type)
        mesh = Mesh(nodes=nodes, elements=elements, E=E, nu=nu,
                    thickness=t, plane_type="stress",
                    elem_type=elem_type)

        # BC: left end fixed
        left_nodes = mesh.nodes_on_edge("x", "min", tol=1e-6)
        for n in left_nodes:
            mesh.fix_node(int(n), "both", 0.0)

        # Right end: parabolic shear traction (downward, P > 0)
        right_nodes = mesh.nodes_on_edge("x", "max", tol=1e-6)
        right_sorted = sorted(right_nodes, key=lambda n: mesh.nodes[int(n), 1])

        def exact_shear_traction(x, y):
            del x
            return _parabolic_shear_traction(y, H, t, P_mag)

        for k in range(len(right_sorted) - 1):
            a, b = int(right_sorted[k]), int(right_sorted[k + 1])
            # loads_core 在每条边的 3 点 Gauss 位置调用该函数，因此
            # 二次抛物线面力被精确积分，不会把载荷离散误差混入单元收敛率。
            mesh.add_traction(a, b, 0.0, exact_shear_traction)

        result = solve(mesh, method="elimination", verbose=verbose)

        u2 = result["u"].reshape(-1, 2)
        tip_nid = _find_tip_node(mesh)
        uy_tip = abs(u2[tip_nid, 1])

        # 弯曲应力: 不取 max|σ_xx|（会被固定端角点奇异性污染），
        # 改为在 x=L/2, y=H/2 处采样（远离边界条件突变点，光滑解区域）。
        # 该点 TG 理论值: σ_xx = -P(L-L/2)(H/2)/I = PLH/(4I)
        mid_eid = point_in_element(mesh, L/2, H/2)
        if mid_eid >= 0:
            s_sample_fem = abs(result["stress"][mid_eid, 0])
        else:
            s_sample_fem = np.max(np.abs(result["stress"][:, 0]))
        s_sample_tg = P_mag * L * H / (4.0 * I)  # |σ_xx| at (L/2, H/2): M*y/I = P(L/2)(H/2)/I

        z2 = estimate(mesh, result, method="SPR", verbose=False)

        h_char = L / nx

        results["h"].append(h_char)
        results["uy_tip"].append(uy_tip)
        results["sigma_sample"].append(s_sample_fem)
        results["n_dof"].append(mesh.n_dof)
        results["eta"].append(z2["eta"])
        results["total_error"].append(z2["total_error"])

        if verbose:
            uy_err_vs_tg = abs(uy_tip - uy_tip_tg) / uy_tip_tg * 100
            s_err_vs_tg = abs(s_sample_fem - s_sample_tg) / s_sample_tg * 100
            print(f"  h={h_char:.4f}  nDOF={mesh.n_dof:5d}  "
                  f"uy_tip={uy_tip:.6e} (vs TG: {uy_err_vs_tg:.1f}%)  "
                  f"s_xx@L/2,H/2={s_sample_fem:.3e} (vs TG: {s_err_vs_tg:.1f}%)  "
                  f"eta={z2['eta']:.2f}%")

    h = np.array(results["h"])
    uy_tip = np.array(results["uy_tip"])
    s_max = np.array(results["sigma_sample"])
    eta_vals = np.array(results["eta"])

    # Richardson extrapolation
    if len(h) >= 2:
        r = h[-2] / h[-1]
        uy_richardson = uy_tip[-1] + (uy_tip[-1] - uy_tip[-2]) / (r**2 - 1)
    else:
        uy_richardson = uy_tip[-1]

    # 使用 Richardson 外推值作参考 (不是最细网格自参考 — 那会让 e_N≡0)。
    # 分母 1e-30 绝对地板曾使微尺度误差序列失真 (与 error_est 同族,
    # 参考值恒非零 (Richardson), tiny 仅防除零。
    uy_err = np.abs(uy_tip - uy_richardson) / (
        np.abs(uy_richardson) + np.finfo(float).tiny)
    s_ref_actual = s_max[-1] + (s_max[-1] - s_max[-2]) / (r**1 - 1) if len(h) >= 2 else s_max[-1]
    s_err = np.abs(s_max - s_ref_actual) / (
        np.abs(s_ref_actual) + np.finfo(float).tiny)

    # Per-level local convergence rates。
    # 曾 1e-15 地板 + 1e-14 门槛: 精细层误差 (<1e-14) 的速率被截成 0,
    # 收敛序列 [1e-3,1e-6,1e-10,1e-15,1e-16] 的最细层报 k=0.00 。仅当下一层误差恰为 0 (机器精度收敛) 才跳过。
    per_level = []
    for i in range(len(h) - 1):
        r_local = np.log(h[i] / h[i+1])
        if uy_err[i + 1] > 0.0:
            ku = np.log(np.maximum(uy_err[i], np.finfo(float).tiny)
                        / uy_err[i + 1]) / r_local
        else:
            ku = 0.0
        if s_err[i + 1] > 0.0:
            ks = np.log(np.maximum(s_err[i], np.finfo(float).tiny)
                        / s_err[i + 1]) / r_local
        else:
            ks = 0.0
        if eta_vals[i + 1] > 0.0:
            ke = np.log(np.maximum(eta_vals[i], np.finfo(float).tiny)
                        / eta_vals[i + 1]) / r_local
        else:
            ke = 0.0
        per_level.append((ku, ks, ke))

    # Global fit: only fit asymptotic region (finest 3-4 levels, skip coarsest)
    uy_rate = s_rate = e_rate = 0.0
    h_fit = np.array([], dtype=float)
    if len(h) >= 3:
        n_skip = max(1, len(h) - 4)  # skip coarsest 1-2 levels (not asymptotic)
        h_fit = h[n_skip:]
        uy_err_fit = uy_err[n_skip:]
        s_err_fit = s_err[n_skip:]
        eta_fit = eta_vals[n_skip:]
        log_h = np.log(h_fit)
        # 地板只防 log(0) — 曾 1e-15 截平真实速率
        uy_rate = np.polyfit(log_h, np.log(np.maximum(
            uy_err_fit, np.finfo(float).tiny)), 1)[0]
        s_rate = np.polyfit(log_h, np.log(np.maximum(
            s_err_fit, np.finfo(float).tiny)), 1)[0]
        e_rate = np.polyfit(log_h, np.log(np.maximum(
            eta_fit, np.finfo(float).tiny)), 1)[0]

    if verbose:
        print("\n  Timoshenko-Goodier closed form (comparison only):")
        print(f"    uy_tip = {uy_tip_tg:.6e} m")
        print(f"    sigma_xx@L/2,H/2 = {s_sample_tg:.3e} Pa")
        print("\n  NOTE: stress sampled at (L/2, H/2) — a point far from the clamped")
        print("  corner singularity (BC type jump at x=0, y=+-H/2). This avoids the")
        print("  singularity-driven divergence that would contaminate max|sigma_xx|.")
        print("  uy_tip and eta are unaffected — both are global quantities.")
        print("\n  Richardson-extrapolated FE reference:")
        print(f"    uy_tip ~ {uy_richardson:.6e} m")
        # 曾标 "self-referenced vs finest" — 实际误差基于 Richardson 参考
        # (2026-08-03 去除自参考偏差后)
        print("\n  Per-level convergence rates (Richardson ref):")
        for i in range(len(per_level)):
            ku, ks, ke = per_level[i]
            print(f"    h={h[i]:.4f} -> {h[i+1]:.4f}:  "
                  f"k_u={ku:+.2f}  k_s={ks:+.2f}  k_e={ke:+.2f}")
        if len(h_fit):
            print(
                f"\n  Asymptotic convergence rates "
                f"(Richardson ref, finest {len(h_fit)} levels):")
            print(
                f"    Tip displacement:     k = {uy_rate:.2f}  "
                f"(expected: ~2.0)")
            print(
                f"    sigma_xx @ L/2,H/2:  k = {s_rate:.2f}  "
                f"(expected: ~1.0)")
            print(
                f"    Z2 energy (eta):      k = {e_rate:.2f}  "
                f"(expected: ~1.0)")

            if (1.5 < uy_rate < 2.5 and 0.5 < s_rate < 1.5
                    and 0.5 < e_rate < 1.5):
                print(
                    f"\n  [PASS] Convergence rates consistent with "
                    f"{elem_type} expectations")
                print("         u~O(h^2), sigma~O(h), eta~O(h)")
            else:
                print(
                    f"\n  [CHECK] Fitted rates: u={uy_rate:.2f} "
                    f"s={s_rate:.2f} e={e_rate:.2f}")
                print("          Expected: u~2.0  s~1.0  e~1.0")
        else:
            print(
                "\n  Asymptotic rates require at least 3 refinement levels.")

    results["uy_rate"] = uy_rate
    results["s_rate"] = s_rate
    results["e_rate"] = e_rate
    results["uy_tip_theory"] = uy_tip_tg
    results["sigma_sample_tg"] = s_sample_tg
    results["uy_richardson"] = uy_richardson

    return results


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("  CST/Q4/Q4R Convergence Study - Bathe sec 4.3.5 / sec 5.3.3")
    print("  Timoshenko Parabolic Shear Cantilever")
    print(f"{'='*60}")
    for _elem_type in ("CPS3", "CPS4", "CPS4R"):
        run_cantilever_convergence(
            refinements=5, verbose=True, elem_type=_elem_type)
