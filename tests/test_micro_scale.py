"""尺度不变判别性测试 — closure_tol 1.0 floor 移除 + 曲率阈值相对化。

两类修复各带判别性断言 (revert 修复后本文件必须失败):

1. ``fit_closed_ellipse`` 的 ``closure_tol`` 曾含 ``max(..., 1.0)``
   物理尺度下限: 微尺度模型 (跨度 ≲2e-13) 整环首末顶点间距恒
   ≤ eps×1.0×32 ≈ 7.1e-15, 被误判为"重复闭合点" → 静默 ``coords[:-1]``
   截掉末顶点, 被截顶点的拟合残差不再被验证 (primitive_samples 63/64)。
2. ``_classify_general_curve`` 的绝对曲率阈值 ``1e-8``/``1e-14``
   (量纲 1/长度): 大坐标 (1e12 级) 平滑曲线 κ < 1e-8 被降级成
   "通用曲线", 拐点被漏计; 相对化 (按 characteristic span 归一) 后
   四档尺度 (1e-150/1e-13/1e0/1e12) 分类行为必须一致。

``_classify_open_arc`` 的 ``radius < max(scale, fit_span)*1e6`` 门
与 ``topology._closed_conic_segment`` 的 scale 传参 (第二处 1.0 floor)
在微尺度下的正确性由环状网格端到端测试覆盖。
"""
import numpy as np
import pytest

from fem2d import Mesh, detect_boundaries, solve
from fem2d.boundary.geometry import classify, fit_closed_ellipse


# ── 形状构造 ──────────────────────────────────────────────────────

def _circle(radius, n=64):
    """端点不重复的闭合圆环采样 — 首末间距 = 闭合边长度 (非 0)."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
    ])


def _s_curve(span, n=128):
    """S 形曲线: 一次变号 (1 个拐点), |κ| 变化系数 ~0.465 ∈ [0.15,0.5)."""
    x = np.linspace(0.0, span, n, endpoint=False)
    return np.column_stack([
        x,
        1e-4 * span * np.sin(2.0 * np.pi * x / span),
    ])


def _parabola(half_span, n=96):
    """浅抛物线: 无拐点, |κ| 变化系数 ~0.294 ∈ [0.15,0.5),
    命中 "曲线 R=[...]" 分支 (mean κ 高于相对化后的阈值)."""
    x = np.linspace(-half_span, half_span, n)
    return np.column_stack([x, 0.5 * x * x / half_span])


def _annulus_mesh(r_outer, r_inner, n=64, E=210e9, nu=0.3, thickness=1.0):
    """微尺度圆环网格 (外环 CCW, 内环 CW 的三角形环带)."""
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    outer = np.column_stack([r_outer * np.cos(angles),
                             r_outer * np.sin(angles)])
    inner = np.column_stack([r_inner * np.cos(angles),
                             r_inner * np.sin(angles)])
    nodes = np.vstack([outer, inner])
    elements = []
    for i in range(n):
        j = (i + 1) % n
        elements.append([i, j, n + j])
        elements.append([i, n + j, n + i])
    return Mesh(
        nodes=nodes, elements=np.array(elements),
        E=E, nu=nu, thickness=thickness, plane_type="stress",
    )


# 微尺度判别: 跨度 ≤ ~2e-13 时旧 closure_tol 的 1.0 floor 必截末顶点
_MICRO_RADII = (1e-14, 1e-150)


# ── 1. closure_tol 不再含 1.0 物理尺度下限 ─────────────────────────

def test_fit_closed_ellipse_keeps_last_vertex_micro_scale():
    """微尺度闭合椭圆: 末顶点不能被当成"重复闭合点"截掉 —
    曾 63/64 (静默丢失, 被截顶点残差不再被验证)."""
    for radius in _MICRO_RADII:
        coords = _circle(radius)
        ellipse, info = fit_closed_ellipse(coords)
        assert ellipse is not None, f"r={radius:g}: 微尺度拟合失败"
        assert info["primitive_samples"] == 64, \
            f"r={radius:g}: 末顶点被截 (primitive_samples {info['primitive_samples']})"
        # 拟合结果仍须准确 (截断修复不引入拟合回归)
        assert np.isclose(ellipse[2], radius, rtol=1e-2), \
            f"r={radius:g}: 半长轴 {ellipse[2]:.3g} ≠ {radius:g}"


def test_classify_micro_circle_validates_full_loop():
    """topology 生产路径 (classify closed=True): 整环 64 顶点全部
    参与 conic 验证 — 曾末顶点被截后 primitive_samples = 63."""
    for radius in _MICRO_RADII:
        coords = _circle(radius)
        segment_type, label, info = classify(
            coords, scale=2.0 * radius, is_outer=True, closed=True)
        assert segment_type == "arc", \
            f"r={radius:g}: 微尺度整圆被降级为 {segment_type} {label}"
        assert info["primitive_samples"] == 64, \
            f"r={radius:g}: 末顶点被截 ({info['primitive_samples']})"
        assert np.isclose(info["radius"], radius, rtol=1e-2)


def test_fit_closed_ellipse_degenerate_loop_no_crash():
    """全同坐标 (ptp=0, characteristic 落到 tiny 兜底): 必须不崩,
    干净返回 None — tiny 兜底路径真实执行."""
    coords = np.tile(np.array([1.0, 2.0]), (12, 1))
    ellipse, info = fit_closed_ellipse(coords)
    assert ellipse is None
    assert info == {}


# ── 2. 曲率阈值相对化: 大坐标不降级、拐点不漏计 ─────────────────────

def test_large_coordinate_s_curve_not_degraded_no_missed_inflection():
    """1e12 级 S 曲线: κ̄ ≈ 2.5e-15 < 绝对 1e-8 (旧) → 曾降级成
    "通用曲线" 且拐点漏计 (inflections=0); 相对化后保持
    "曲线 R=[...]" 且拐点如实计 1 个."""
    coords = _s_curve(1e12)
    segment_type, label, info = classify(
        coords, scale=1e12, is_outer=True)
    assert segment_type == "curve"
    assert "通用曲线" not in label, f"大坐标平滑曲线被降级: {label}"
    assert info["inflection_count"] == 1, \
        f"拐点漏计: {info['inflection_count']} (应 1)"
    # R 范围与模型跨度成比例 (尺度不变)
    assert np.isclose(info["R_min"] / 1e12, 253.4, rtol=0.05), \
        f"R_min 尺度失真: {info['R_min']:.4g}"
    assert np.isclose(info["R_max"] / 1e12, 5163.0, rtol=0.05), \
        f"R_max 尺度失真: {info['R_max']:.4g}"


def test_large_coordinate_parabola_keeps_radius_range():
    """1e12 级浅抛物线: κ̄ ≈ 7.1e-13 < 绝对 1e-8 (旧) → 曾降级成
    "通用曲线"; 相对化后保持 "曲线 R=[...]" 半径区间标签."""
    coords = _parabola(1e12)
    segment_type, label, info = classify(
        coords, scale=2e12, is_outer=True)
    assert segment_type == "curve"
    assert "R=[" in label, f"大坐标抛物线被降级: {label}"
    assert np.isclose(info["R_min"] / 1e12, 1.0003, rtol=0.05), \
        f"R_min 尺度失真: {info['R_min']:.4g}"
    assert np.isclose(info["R_max"] / 1e12, 2.7405, rtol=0.05), \
        f"R_max 尺度失真: {info['R_max']:.4g}"


# ── 3. 四档尺度一致 (1e-150 / 1e-13 / 1e0 / 1e12) ─────────────────

_SCALE_TIERS = (1e-150, 1e-13, 1.0, 1e12)


def test_s_curve_classification_invariant_across_scale_tiers():
    """同一无量纲 S 曲线: 四档尺度下分类不退化、拐点数恒 1、
    κ̄×span 与 R×span 比例恒定 (κ 量纲 1/长度, 归一后必须无尺度依赖)."""
    for span in _SCALE_TIERS:
        coords = _s_curve(span)
        segment_type, label, info = classify(
            coords, scale=span, is_outer=True)
        assert segment_type == "curve", f"span={span:g}: {label}"
        assert info["inflection_count"] == 1, \
            f"span={span:g}: 拐点 {info['inflection_count']} (应 1)"
        assert np.isclose(
            info["curvature_mean"] * span, 2.51e-3, rtol=0.05), \
            f"span={span:g}: κ̄×span 漂移 {info['curvature_mean'] * span:.4g}"
        assert np.isclose(info["R_min"] / span, 253.4, rtol=0.05), \
            f"span={span:g}: R_min 比例漂移"
        assert np.isclose(info["R_max"] / span, 5163.0, rtol=0.05), \
            f"span={span:g}: R_max 比例漂移"


def test_parabola_classification_invariant_across_scale_tiers():
    """同一无量纲浅抛物线: 四档尺度下 "曲线 R=[...]" 分支不退化,
    半径区间与跨度严格成比例."""
    for half_span in _SCALE_TIERS:
        coords = _parabola(half_span)
        segment_type, label, info = classify(
            coords, scale=2.0 * half_span, is_outer=True)
        assert segment_type == "curve", f"span={half_span:g}: {label}"
        assert "R=[" in label, \
            f"span={half_span:g}: 曲线被降级: {label}"
        assert np.isclose(info["R_min"] / half_span, 1.0003, rtol=0.05), \
            f"span={half_span:g}: R_min 比例漂移"
        assert np.isclose(info["R_max"] / half_span, 2.7405, rtol=0.05), \
            f"span={half_span:g}: R_max 比例漂移"


# ── 4. 端到端: 微尺度圆环 detect + solve ───────────────────────────

def test_micro_annulus_boundary_detection_full_loop_validation():
    """1e-13 级圆环 (跨度 1.2e-13 ≤ 2e-13): 内外环均须整圆分类且
    primitive_samples = 64 (末顶点参与验证, 曾 63)."""
    mesh = _annulus_mesh(6e-14, 3e-14)
    segments = detect_boundaries(mesh)
    assert len(segments) == 2, f"圆环应得 2 段, 实际 {len(segments)}"
    by_role = {bool(s["info"].get("is_outer")): s for s in segments}
    assert set(by_role) == {True, False}, "应同时检出外环与内孔"
    for role, radius in ((True, 6e-14), (False, 3e-14)):
        seg = by_role[role]
        assert seg["type"] == "arc", f"{'外环' if role else '内孔'}降级: {seg['type']}"
        assert seg["info"]["primitive_samples"] == 64, \
            f"{'外环' if role else '内孔'}末顶点被截 " \
            f"({seg['info']['primitive_samples']})"
        assert np.isclose(seg["info"]["radius"], radius, rtol=1e-2), \
            f"{'外环' if role else '内孔'}半径 {seg['info']['radius']:.3g}"


def test_micro_annulus_full_solve_succeeds():
    """微尺度 (1e-13) 圆环全流程求解: 位移/应力有限, 残差闭合.
    位移量级须与载荷/材料/几何量纲一致 (~1e-24)."""
    mesh = _annulus_mesh(6e-14, 3e-14)
    mesh.fix_node(0, "both", 0.0)
    mesh.fix_node(1, "both", 0.0)
    mesh.add_traction(40, 41, 1.0, 0.0)
    result = solve(mesh, method="elimination", verbose=False)
    assert np.all(np.isfinite(result["u"])), "位移含 NaN/inf"
    assert np.all(np.isfinite(result["stress"])), "应力含 NaN/inf"
    assert result["residual"] < 1e-6, f"残差未闭合: {result['residual']}"
    assert 1e-26 < np.max(np.abs(result["u"])) < 1e-22, \
        f"位移量级失真: {np.max(np.abs(result['u'])):.3e}"


@pytest.mark.parametrize("scale", [1e-13, 1e-150])
def test_micro_solve_square_plate_scale_invariant(scale):
    """复用求解器既有微尺度网格结构 (test_error_est_scale_invariance
    同款条带): 求解本身不受边界容差修改影响, 两档微尺度均成功."""
    ncol, nrow = 8, 4
    xs = np.linspace(0.0, scale, ncol)
    ys = np.linspace(0.0, 0.5 * scale, nrow + 1)
    nodes, elements = [], []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            nodes.append((x, y))
            if i < ncol - 1 and j < nrow:
                a, b, c = j * ncol + i, j * ncol + i + 1, (j + 1) * ncol + i
                d = (j + 1) * ncol + i + 1
                elements.append([a, b, d])
                elements.append([a, d, c])
    mesh = Mesh(
        nodes=np.array(nodes), elements=np.array(elements),
        E=210e9, nu=0.3, thickness=1.0, plane_type="stress")
    for j in range(nrow + 1):
        mesh.fix_node(j * ncol, "both", 0.0)
    for j in range(nrow):
        mesh.add_traction(
            j * ncol + ncol - 1, (j + 1) * ncol + ncol - 1, 0.0, 1.0)
    result = solve(mesh, method="elimination", verbose=False)
    assert np.all(np.isfinite(result["u"])), f"scale={scale:g}: 位移非有限"
    assert result["residual"] < 1e-6
