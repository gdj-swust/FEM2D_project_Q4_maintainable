"""Z² 误差估计对材料与几何尺度的完全不变性 (审查 P2-6).

error_est 曾只归一化应力 — 柔度 D_inv 与积分面积 dA 仍是绝对尺度。
同一无量纲应力场 (正常尺度解出的应力数组) 在极端但可表示的材料/
几何尺度下, t·dA·D_inv·σ² 中间量溢出/下溢:
  - 几何 1e-150 (t 不缩放): eta 静默跳到 100% (分子分母同时撞 tiny 地板)
  - 几何 1e-150 (t 同步缩放): eta 下溢到 0%
  - 组合极端: NaN
修复后 eta / worst_elem / elem_contrib 在正常/1e-150 几何/1e150 材料
下完全一致 (相对差 < 1e-6)。

注: E=1e-150 材料 + 几何 1e150 的组合在 solver.py 内部能量计算处
(超出本文件边界) 报 NaN, 端到端用例仅覆盖求解器可表示的组合。
"""
import numpy as np

from fem2d import Mesh, solve
from fem2d.error_est import estimate


def _cantilever_mesh():
    """悬臂梁 CST 网格 (与 20260803 b/c 系列同款)."""
    L, H, nx, ny = 5.0, 1.0, 8, 4
    ncol = nx + 1
    nodes = [[i * L / nx, j * H / ny] for j in range(ny + 1)
             for i in range(nx + 1)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (j * ncol + i, j * ncol + i + 1,
                          (j + 1) * ncol + i, (j + 1) * ncol + i + 1)
            elems.append([a, b, d]); elems.append([a, d, c])
    return np.array(nodes, dtype=float), np.array(elems), nx, ny


def _build_mesh(E, geo_scale, thickness, tau=1e5):
    nodes, elems, nx, ny = _cantilever_mesh()
    ncol = nx + 1
    m = Mesh(nodes=nodes * geo_scale, elements=elems, E=E, nu=0.3,
             thickness=thickness, plane_type="stress")
    for i in range(ny + 1):
        m.fix_node(i * ncol, "both", 0.0)
    right = [i * ncol + ncol - 1 for i in range(ny + 1)]
    for k in range(ny):
        m.add_traction(right[k], right[k + 1], 0.0, tau)
    return m


def _reference():
    """正常尺度求解一次 — 应力场作为同一无量纲应力场喂给各尺度网格."""
    m = _build_mesh(210e9, 1.0, 0.1)
    result = solve(m, method="elimination", verbose=False)
    return m, result


# 验收矩阵: 正常尺度 / 1e-150 几何 / 1e150 材料 + 反向极端与组合
_SCALE_CASES = [
    ("正常尺度", 210e9, 1.0, 0.1),
    ("1e150 材料", 1e150, 1.0, 0.1),
    ("1e-150 材料", 1e-150, 1.0, 0.1),
    ("1e-150 几何", 210e9, 1e-150, 0.1 * 1e-150),
    ("1e150 几何", 210e9, 1e150, 0.1 * 1e150),
    ("1e150 材料 + 1e-150 几何", 1e150, 1e-150, 0.1 * 1e-150),
    ("1e-150 材料 + 1e150 几何", 1e-150, 1e150, 0.1 * 1e150),
]


def test_eta_worst_contrib_invariant_to_material_geometry_scale():
    """同一无量纲应力场: eta / worst_elem / elem_contrib 与尺度无关 —
    曾 1e-150 几何 eta 跳 100%, t 同步缩放跳 0%, 组合溢出 NaN."""
    ref_mesh, result = _reference()
    ref = estimate(ref_mesh, result, method="SPR", verbose=False)
    for label, E, gs, th in _SCALE_CASES:
        z = estimate(_build_mesh(E, gs, th), result,
                     method="SPR", verbose=False)
        rel = abs(z["eta"] - ref["eta"]) / max(abs(ref["eta"]), 1e-300)
        assert rel < 1e-6, (f"{label}: eta {z['eta']:.6f}% vs 基准 "
                            f"{ref['eta']:.6f}% (rel {rel:.2e})")
        assert z["worst_elem"] == ref["worst_elem"], label
        np.testing.assert_allclose(z["elem_contrib"], ref["elem_contrib"],
                                   rtol=1e-6, atol=1e-9, err_msg=label)
        assert abs(z["elem_contrib"].sum() - 100.0) < 0.5, label
        assert np.all(np.isfinite(z["elem_contrib"])), label


def test_scale_invariance_all_recovery_methods():
    """L2 / weighted 走同一归一化路径, 极端尺度下同样不变."""
    ref_mesh, result = _reference()
    for method in ("L2", "weighted"):
        ref = estimate(ref_mesh, result, method=method, verbose=False)
        for E, gs, th in ((1e150, 1.0, 0.1),
                          (210e9, 1e-150, 0.1 * 1e-150),
                          (1e-150, 1e150, 0.1 * 1e150)):
            z = estimate(_build_mesh(E, gs, th), result,
                         method=method, verbose=False)
            rel = abs(z["eta"] - ref["eta"]) / max(abs(ref["eta"]), 1e-300)
            assert rel < 1e-6, (f"{method} E={E:.0e} gs={gs:.0e}: "
                                f"eta {z['eta']:.6f}% vs {ref['eta']:.6f}%")
            assert z["worst_elem"] == ref["worst_elem"], method
            np.testing.assert_allclose(
                z["elem_contrib"], ref["elem_contrib"],
                rtol=1e-6, atol=1e-9, err_msg=f"{method} {E:.0e}/{gs:.0e}")


def test_end_to_end_eta_invariant_extreme_scale():
    """完整 solve→estimate 链路在极端尺度同样不变 (可表示的组合)."""
    ref_mesh, result = _reference()
    ref = estimate(ref_mesh, result, method="SPR", verbose=False)["eta"]
    for E, gs, th in ((1e150, 1.0, 0.1),
                      (1e-150, 1.0, 0.1),
                      (210e9, 1e-150, 0.1 * 1e-150),
                      (1e150, 1e-150, 0.1 * 1e-150)):
        m = _build_mesh(E, gs, th)
        r = solve(m, method="elimination", verbose=False)
        eta = estimate(m, r, method="SPR", verbose=False)["eta"]
        assert abs(eta - ref) / ref < 1e-6, \
            f"E={E:.0e} gs={gs:.0e}: eta {eta:.6f}% vs 基准 {ref:.6f}%"


def test_residual_indicator_finite_and_ordered_at_extreme_geometry():
    """element_refinement_indicator 在极端几何下必须有限且排序不变 —
    曾 h_e²·jump² 中间量溢出为 inf, 全部单元并列 argmax=0 (排序静默
    失效); 修复为对数空间累加 (logsumexp)."""
    from fem2d.error_est import element_refinement_indicator
    ref_mesh, result = _reference()
    ind0 = element_refinement_indicator(ref_mesh, result)
    assert np.all(np.isfinite(ind0))
    for E, gs, th in ((210e9, 1e-150, 0.1 * 1e-150),
                      (210e9, 1e150, 0.1 * 1e150),
                      (1e150, 1e150, 0.1 * 1e150)):
        ind = element_refinement_indicator(_build_mesh(E, gs, th), result)
        assert np.all(np.isfinite(ind)), f"E={E:.0e} gs={gs:.0e} 溢出"
        assert int(np.argmax(ind)) == int(np.argmax(ind0)), \
            f"E={E:.0e} gs={gs:.0e} 排序漂移"
        assert np.max(ind) > 0.0


def test_total_error_scales_linearly_with_load_at_extreme_geometry():
    """绝对报告量乘回正确: 1e-150 几何下 total_error 仍随载荷线性缩放
    (曾中间量下溢为 0, 载荷比例关系丢失)."""
    big = estimate(_build_mesh(210e9, 1e-150, 0.1 * 1e-150, tau=1e5),
                   solve(_build_mesh(210e9, 1e-150, 0.1 * 1e-150, tau=1e5),
                         method="elimination", verbose=False),
                   verbose=False)["total_error"]
    assert big > 0.0, "1e-150 几何 total_error 下溢为 0"
    small = estimate(
        _build_mesh(210e9, 1e-150, 0.1 * 1e-150, tau=1e-10),
        solve(_build_mesh(210e9, 1e-150, 0.1 * 1e-150, tau=1e-10),
              method="elimination", verbose=False),
        verbose=False)["total_error"]
    expected = big * (1e-10 / 1e5)  # total_error ∝ 载荷
    assert small > 0.0, "1e-10 载荷 total_error 下溢为 0"
    assert abs(small / expected - 1.0) < 0.05, \
        f"total_error 尺度失真: {small:.3e}, 应 ~{expected:.3e}"
