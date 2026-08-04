"""2026-08-03 二轮审计回归测试 — 绝对阈值/静默错误家族 (12 项修复).

覆盖: fixed_dofs 布尔掩码 / 微尺度集中力 / CST 纳米面积 / preprocess
NaN/连接宽度/蝴蝶形/空单元 / 刚体 rank==2 诊断 / error_est 幅值不变性
/ band 整除校验 / .spec 未知键警告 / gouraud 微尺度色标。
"""
import contextlib
import io
import os

import numpy as np
import pytest


def _square_two_cst(scale=1.0):
    from fem2d import Mesh
    s = scale
    return Mesh(nodes=np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=float),
                elements=np.array([[0, 1, 2], [0, 2, 3]]), E=210e9, nu=0.3,
                thickness=1.0, plane_type="stress", elem_type="CPS3")


def _triangle_mesh():
    from fem2d import Mesh
    return Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3,
                thickness=1.0)


# ═══════════════════════════════════════════════════════════════
# P1: fixed_dofs 布尔掩码曾静默折叠成 {0,1}, 约束错 DOF
# ═══════════════════════════════════════════════════════════════

def test_fixed_dofs_boolean_mask_rejected():
    """布尔掩码必须拒绝 — 曾 asarray(float) 变成 [0,0,1,1] unique 折叠
    成 {0,1}, 用户想约束的 DOF 被静默换成节点 0."""
    from fem2d import Mesh
    mask = np.array([False, False, True, True,
                     False, False, False, False])
    with pytest.raises(TypeError, match="boolean mask"):
        Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
             elements=np.array([[0, 1, 2], [1, 3, 2]]), E=210e9, nu=0.3,
             thickness=1.0, fixed_dofs=mask)


def test_fixed_dofs_boolean_mask_rejected_on_reassign():
    """构造后重绑布尔掩码同样拒绝 (validate_state 路径)."""
    from fem2d import Mesh, solve
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
             elements=np.array([[0, 1, 2], [1, 3, 2]]), E=210e9, nu=0.3,
             thickness=1.0)
    m.fix_node(0, "both", 0.0)
    m.fixed_dofs = np.array([True, True, False, False,
                             False, False, False, False])
    with pytest.raises(TypeError, match="boolean mask"):
        solve(m, method="elimination", verbose=False)


# ═══════════════════════════════════════════════════════════════
# 微尺度集中力不得静默丢弃 (abs>1e-30 过滤器)
# ═══════════════════════════════════════════════════════════════

def test_tiny_concentrated_force_not_dropped():
    """1e-31 N 微尺度集中力不得静默丢弃 (曾 abs>1e-30 过滤, 求解"成功"
    但位移全零 — 静默错误结果, 审计 2026-08-03)."""
    from fem2d import solve
    m = _triangle_mesh()
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.add_force(2, 1e-31, 0.0)
    assert len(m.concentrated_forces) == 1, "1e-31 集中力被静默丢弃"
    r = solve(m, method="elimination", verbose=False)
    assert r["u"].reshape(-1, 2)[2, 0] > 0, "节点 2 应受 +x 载荷"


def test_zero_force_still_not_recorded():
    """零力调用 (默认参数) 不得产生记录 (与旧行为一致)."""
    m = _triangle_mesh()
    m.add_force(2)
    assert len(m.concentrated_forces) == 0


# ═══════════════════════════════════════════════════════════════
# CST 标量刚度路径的纳米面积阈值 (曾绝对 1e-30)
# ═══════════════════════════════════════════════════════════════

def test_cst_scalar_stiffness_nano_element():
    """边长 1e-15 的合法纳米 CST 标量刚度不得拒绝 — 曾绝对 1e-30 面积
    阈值 (面积 5e-31) 误判退化, 批量路径却正常."""
    from fem2d import Mesh
    s = 1e-15
    m = Mesh(nodes=np.array([[0, 0], [s, 0], [0, s]], dtype=float),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3,
             thickness=1e-9, plane_type="stress", elem_type="CPS3")
    Ke = m.element_kernel.stiffness(m, 0)
    assert Ke.shape == (6, 6) and np.all(np.isfinite(Ke))


def test_cst_scalar_stiffness_truly_degenerate_still_rejected():
    """真正零面积 (共线) 单元仍须拒绝."""
    from fem2d import Mesh
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [2., 0.]]),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3,
             thickness=1.0, elem_type="CPS3")
    with pytest.raises(ValueError, match="degenerate"):
        m.element_kernel.stiffness(m, 0)


def test_cst_scalar_stiffness_far_from_origin():
    """远离原点的合法单元不得误判退化 — 面积判据曾用坐标绝对值作尺度,
    1e7 偏移的单位三角形 (面积 0.5 < 判据 1.42) 被拒; 批量路径却正常
   ."""
    from fem2d import Mesh
    off = 1e7
    m = Mesh(nodes=np.array([[off, off], [off + 1, off], [off, off + 1]]),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3,
             thickness=1.0, elem_type="CPS3")
    Ke = m.element_kernel.stiffness(m, 0)
    assert np.all(np.isfinite(Ke)) and Ke.shape == (6, 6)


# ═══════════════════════════════════════════════════════════════
# preprocess 校验缺口 (裸异常 / 漏检 / 误导性 ok)
# ═══════════════════════════════════════════════════════════════

def test_validate_mesh_nan_coords_clean_error():
    """NaN/Inf 坐标必须报 MeshValidationError, 不得透传 cKDTree 裸异常."""
    from fem2d.preprocess import validate_mesh, MeshValidationError
    nodes = np.array([[0, 0], [1, 0], [0, 1], [np.nan, np.nan]])
    with pytest.raises(MeshValidationError, match="NaN"):
        validate_mesh(nodes, np.array([[0, 1, 2]]))


def test_validate_mesh_connectivity_width_rejected():
    """5/6 列连接曾抛裸 KeyError — 必须报 MeshValidationError."""
    from fem2d.preprocess import validate_mesh, MeshValidationError
    with pytest.raises(MeshValidationError, match="宽度"):
        validate_mesh(np.zeros((6, 2)), np.array([[0, 1, 2, 3, 4, 5]]))


def test_validate_mesh_empty_elements_rejected():
    """空单元集曾 ok=True 误导下游 — 必须致命."""
    from fem2d.preprocess import validate_mesh, MeshValidationError
    with pytest.raises(MeshValidationError, match="不包含任何单元"):
        validate_mesh(np.zeros((3, 2)), np.zeros((0, 3), dtype=int))


def test_validate_mesh_bowtie_q4_detected():
    """蝴蝶形 (自交) Q4 净面积为正曾漏检, 求解时才被拒 — 导入必须诊断."""
    from fem2d.preprocess import validate_mesh
    nodes = np.array([[0, 0], [2, 0], [0, 1], [1, -1]], dtype=float)
    rep = validate_mesh(nodes, np.array([[0, 1, 2, 3]]))
    assert not rep["ok"], "蝴蝶形 Q4 未被判退化"
    assert any("退化单元" in e for e in rep["errors"])


# ═══════════════════════════════════════════════════════════════
# 刚体约束 rank==2 诊断 (曾一律误报"转动")
# ═══════════════════════════════════════════════════════════════

def test_rigid_body_rank2_diagnosis_names_missing_mode():
    """三个仅 y 约束的节点 rank=2, 缺失的是 x 平动 — 不得误报转动."""
    from fem2d import Mesh
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3,
             thickness=1.0)
    m.fix_node(0, "y", 0.0)
    m.fix_node(1, "y", 0.0)
    m.fix_node(2, "y", 0.0)
    issues = m.check_rigid_body_constraints()
    assert issues, "应报告约束不足"
    assert "x-平动" in issues[0]["issue"], \
        f"应提示 x-平动, 得到: {issues[0]['issue']}"


# ═══════════════════════════════════════════════════════════════
# error_est: 相对指标必须与应力幅值无关 (曾 1e-30 地板静默低估)
# ═══════════════════════════════════════════════════════════════

def _cantilever_eta_jump(tau):
    from fem2d import Mesh, solve
    from fem2d.error_est import estimate, compute_traction_jumps
    L, H, t, nx, ny = 5.0, 1.0, 0.1, 8, 4
    ncol = nx + 1
    nodes = [[i * L / nx, j * H / ny] for j in range(ny + 1)
             for i in range(nx + 1)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (j * ncol + i, j * ncol + i + 1,
                          (j + 1) * ncol + i, (j + 1) * ncol + i + 1)
            elems.append([a, b, d])
            elems.append([a, d, c])
    m = Mesh(nodes=np.array(nodes), elements=np.array(elems),
             E=210e9, nu=0.3, thickness=t, plane_type="stress")
    for i in range(ny + 1):
        m.fix_node(i * ncol, "both", 0.0)
    right = [i * ncol + ncol - 1 for i in range(ny + 1)]
    for k in range(ny):
        m.add_traction(right[k], right[k + 1], 0.0, tau)
    r = solve(m, method="elimination", verbose=False)
    eta = estimate(m, r, method="SPR", verbose=False)["eta"]
    jr = max(j["jump_rel"] for j in compute_traction_jumps(m, r["stress"]))
    return eta, jr


def test_error_est_amplitude_invariance():
    """Z2 eta 与牵引跳跃必须与应力幅值无关 — 曾 1e-30 地板使 eta 从
    65.7% 静默跌到 1.0%, jump_rel 从 1.61 跌到 0.15."""
    eta_big, jr_big = _cantilever_eta_jump(1e5)
    eta_tiny, jr_tiny = _cantilever_eta_jump(1e-32)
    assert abs(eta_big - eta_tiny) < 1e-6, \
        f"eta 幅值依赖: {eta_big:.4f}% vs {eta_tiny:.4f}%"
    assert abs(jr_big - jr_tiny) < 1e-6, \
        f"jump_rel 幅值依赖: {jr_big:.4f} vs {jr_tiny:.4f}"


def test_traction_jump_sigma_ref_nonpositive_rejected():
    """显式 sigma_ref 非正必须拒绝 (曾 max(...,1e-30) 静默覆盖)."""
    from fem2d.error_est import compute_traction_jumps
    m = _square_two_cst()
    m.build_connectivity()
    stress = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="must be > 0"):
        compute_traction_jumps(m, stress, sigma_ref=0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        compute_traction_jumps(m, stress, sigma_ref=-1.0)


# ═══════════════════════════════════════════════════════════════
# band 区间整除校验 (曾求解白跑, 绘图阶段才崩)
# ═══════════════════════════════════════════════════════════════

def test_band_range_divisible_required():
    """(max-min)/step 非整数必须配置期拒绝 — 曾求解成功后在绘图阶段
    抛 ValueError."""
    from fem2d.config import AnalysisConfig
    with pytest.raises(ValueError, match="非整数"):
        AnalysisConfig(band_min=0, band_max=10, band_step=4)
    AnalysisConfig(band_min=0, band_max=8, band_step=4)       # 整除: 合法
    AnalysisConfig(band_min=0, band_max=1, band_step=1e-4)    # 10000 层仍合法


# ═══════════════════════════════════════════════════════════════
# .spec 未知键必须警告 (曾静默忽略, 拼错键名载荷不生效)
# ═══════════════════════════════════════════════════════════════

def test_spec_unknown_key_warns(tmp_path):
    """拼错的键名 (tracion) 必须 WARN — 曾静默忽略导致载荷不生效."""
    from fem2d.config import AnalysisConfig
    from fem2d.input_source import resolve_spec_overrides
    geo = tmp_path / "_unkkey.geo"
    geo.write_text("Point(1) = {0, 0, 0, 0.5};\n", encoding="utf-8")
    spec = tmp_path / "_unkkey.spec"
    spec.write_text("mesh = _unkkey.geo\ntracion = right:1e6,0\nE = 210e9\n",
                    encoding="utf-8")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        resolve_spec_overrides(str(spec), AnalysisConfig())
    out = buf.getvalue()
    assert "tracion" in out and "WARN" in out, f"未警告: {out!r}"


# ═══════════════════════════════════════════════════════════════
# gouraud 微尺度近常场色标 (曾 vmax=vmin+1.0 塌缩为单色)
# ═══════════════════════════════════════════════════════════════

def test_spr_representative_path_linear_exact_on_distorted_quads():
    """SPR (ne,ncomp) 路径在扭曲四边形上必须线性精确 — 曾把代表应力
    挂在多边形质心 (与自然中心差 ~3.7% 边长), 线性场恢复误差 5e-3;
    节点均值位置恢复误差 ~2e-16."""
    from fem2d import Mesh
    from fem2d.spr import spr_recovery
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.1, 0.9], [2.0, 1.0],
        [0.0, 2.0], [1.0, 2.0], [2.0, 2.0]], dtype=float)
    elems = np.array([[0, 1, 4, 3], [1, 2, 5, 4],
                      [3, 4, 7, 6], [4, 5, 8, 7]])
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=1.0,
             plane_type="stress", elem_type="CPS4R")
    m.build_connectivity()
    a, b = 1.0e6, -0.5e6
    centers = np.mean(nodes[elems], axis=1)
    s_rep = np.column_stack([a * centers[:, 0] + b * centers[:, 1],
                             np.zeros(m.n_elements),
                             np.zeros(m.n_elements)])
    rec = spr_recovery(m, s_rep)
    for i, nd in enumerate(nodes):
        exact = a * nd[0] + b * nd[1]
        assert abs(rec[i, 0] - exact) < 1e-9 * max(abs(exact), 1.0), \
            f"节点 {i}: SPR 恢复 {rec[i, 0]:.6e} vs 精确 {exact:.6e}"


def test_nodal_l2_projection_orphan_node_clean_error():
    """孤立节点 → L2 投影必须报网格诊断, 不得透传 splu
    'Factor is exactly singular' 裸异常."""
    from fem2d import Mesh
    from fem2d.stress import nodal_L2_projection
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [5, 5]], dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2, 3]]), E=210e9,
             nu=0.3, thickness=1.0, elem_type="CPS4")
    m.build_connectivity()
    with pytest.raises(ValueError, match="孤立节点"):
        nodal_L2_projection(m, np.ones((1, 3)))


def test_boundary_self_intersection_scale_invariant():
    """微尺度边界不得误报自交 — 曾 1.0 下限容差 + 接缝微小闭合边
    使每条边界都被判自交拒绝."""
    from fem2d.boundary.topology import has_boundary_self_intersection
    t = np.linspace(0, 2 * np.pi, 129)
    for s in (1.0, 1e-6, 1e-16):
        coords = np.column_stack([s * 5.0 * np.cos(t), s * 2.5 * np.sin(t)])
        assert has_boundary_self_intersection(coords.tolist()) is False, \
            f"scale={s} 合法边界被误报自交"
    bowtie = [[0, 0], [1, 1], [0, 1], [1, 0]]
    assert has_boundary_self_intersection(bowtie) is True, "真实自交漏检"
    dup = [[0, 0], [1, 0], [0.5, 0.5], [1, 0], [0, 1]]
    assert has_boundary_self_intersection(dup) is True, "重复节点漏检"


def test_curvature_closing_seam_no_spurious_breakpoint():
    """闭合链接缝不得产生伪曲率断点 — 曾闭合重复点/微小闭合边使
    segment_by_curvature 在接缝误报."""
    from fem2d.boundary.geometry import curvature, segment_by_curvature
    t = np.linspace(0, 2 * np.pi, 41)
    coords = np.column_stack([np.cos(t), np.sin(t)])
    kappa = curvature(coords)
    assert segment_by_curvature(kappa, scale=10) == [], \
        "闭合圆被误报曲率断点"
    assert abs(kappa[0] - kappa[-1]) < 1e-12, "接缝 κ 不一致"


def test_curvature_micro_scale_not_all_zero():
    """微尺度模型曲率不得全零 — 曾绝对 1e-15 零长判据使曲率分段
    静默失效."""
    from fem2d.boundary.geometry import curvature
    s = 1e-16
    t = np.linspace(0, 2 * np.pi, 41)
    coords = np.column_stack([s * np.cos(t), s * np.sin(t)])
    kappa = curvature(coords)
    assert kappa[10] > 0, f"微尺度曲率为零: {kappa[10]}"


def test_gouraud_micro_field_range_not_collapsed():
    """微尺度近常场 (span 1e-16 绝对单位) 的 gouraud 色标必须保持场
    尺度 — 曾绝对 1e-15 阈值 + 1.0 pad, 色标变 [1e-12, 1.0] 单色塌缩
   ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fem2d.visualize import _plot_gouraud_contour, _display_triangulation
    m = _square_two_cst(scale=1e-9)
    m.build_connectivity()
    vals = 1e-12 + np.linspace(0.0, 1e-16, m.n_nodes)
    fig, ax = plt.subplots()
    try:
        tri, _ = _display_triangulation(m.nodes, m.elements)
        _, vmin, vmax = _plot_gouraud_contour(ax, tri, vals, m, 12, "SPR")
    finally:
        plt.close(fig)
    assert vmax - vmin <= 1e-11, \
        f"色标被抬到 [{vmin:.3e}, {vmax:.3e}], 微尺度云图塌缩"
    assert abs(vmin - 1e-12) < 1e-15, f"vmin 偏离场尺度: {vmin:.3e}"
