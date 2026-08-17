"""2026-08-03 第四轮外部审查第二批回归测试.

覆盖: Z2 三处归一化细节 (total_error 下溢 / s_scale 未含 stress_qp /
次正规直接除法) / BC 公共系统校验 (K NaN / F NaN / penalty 强度门槛) /
微尺度标签 .6g。
"""
import numpy as np
import pytest

from scipy.sparse import csr_matrix


def _cst_rect_mesh(nx=2, ny=1):
    """CST 矩形网格 (nx×ny 四边形各切 2 三角形)."""
    from fem2d import Mesh
    ncol = nx + 1
    nodes = [[i, j] for j in range(ny + 1) for i in range(ncol)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (j * ncol + i, j * ncol + i + 1,
                          (j + 1) * ncol + i, (j + 1) * ncol + i + 1)
            elems.append([a, b, d]); elems.append([a, d, c])
    m = Mesh(nodes=np.array(nodes, dtype=float),
             elements=np.array(elems), E=210e9, nu=0.3,
             thickness=0.1, plane_type="stress")
    m.build_connectivity()
    return m, np.array(elems)


# ═══════════════════════════════════════════════════════════════
# Z2: 三处归一化细节 (第四轮外部审查复现)
# ═══════════════════════════════════════════════════════════════

def test_z2_total_error_keeps_scale():
    """total_error 必须随载荷线性缩放 — 曾对乘回尺度后的 elem_err
    再平方求和, 载荷 1e-200 时 elem_err² ~1e-400 下溢为 0 (审查复现:
    total_error 1.012 → 0). 归一化空间求和再乘回后 1e-200 载荷
    total_error ≈ 1.012e-205."""
    from fem2d import Mesh, solve
    from fem2d.error_est import estimate
    L, H, t, nx, ny = 5.0, 1.0, 0.1, 8, 4
    ncol = nx + 1
    nodes = [[i * L / nx, j * H / ny] for j in range(ny + 1)
             for i in range(nx + 1)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (j * ncol + i, j * ncol + i + 1,
                          (j + 1) * ncol + i, (j + 1) * ncol + i + 1)
            elems.append([a, b, d]); elems.append([a, d, c])
    def run(tau):
        m = Mesh(nodes=np.array(nodes), elements=np.array(elems),
                 E=210e9, nu=0.3, thickness=t, plane_type="stress")
        for i in range(ny + 1):
            m.fix_node(i * ncol, "both", 0.0)
        right = [i * ncol + ncol - 1 for i in range(ny + 1)]
        for k in range(ny):
            m.add_traction(right[k], right[k + 1], 0.0, tau)
        r = solve(m, method="elimination", verbose=False)
        return estimate(m, r, method="SPR", verbose=False)["total_error"]
    base = run(1e5)
    # SPR-BC-2026-001 重基线: 边界节点改由内部 patch 恢复后 base
    # 1.012 → 0.722 (估计更准, 载荷线性缩放比仍 1.000000); 地板只防
    # total_error 塌缩/下溢回归, 0.5 以下仍判异常。
    assert base > 0.5, f"基准 total_error {base} 异常"
    small = run(1e-200)
    expected = base * 1e-205
    assert small > 0.0, "1e-200 载荷 total_error 下溢为 0"
    assert abs(small / expected - 1.0) < 0.05, \
        f"total_error 尺度失真: 1e-200 得 {small:.3e}, 应 ~{expected:.3e}"


def test_z2_s_scale_includes_quadrature_stress():
    """s_scale 必须覆盖积分点应力 — 曾只取单元代表应力: 构造积分点
    应力正负抵消、单元平均为零的弯曲型场, s_scale≈0 使归一化爆炸
    (审查复现: eta=2.12e153%). 现在取 max(|stress|,|stress_qp|,|s_star|),
    eta 必须有限且合理."""
    from fem2d.error_est import estimate
    m, elems = _cst_rect_mesh()
    n_elem = len(elems)
    amp = 1e-200
    # 单元代表应力 = 0 (上下排积分点应力正负抵消的平均); 积分点应力
    # 非零: 上排 +amp, 下排 −amp (σx 分量)
    stress = np.zeros((n_elem, 3))
    stress_qp = np.zeros((n_elem, 1, 3))
    for eid in range(n_elem):
        ymid = float(m.centroids[eid, 1])
        stress_qp[eid, 0, 0] = amp if ymid >= 0.5 else -amp
    result = {"stress": stress, "stress_qp": stress_qp,
              "nodal_displacements": np.zeros((m.n_nodes, 2)),
              "displacement": np.zeros((m.n_dof,))}
    z = estimate(m, result, method="SPR", verbose=False)
    assert np.isfinite(z["eta"]) and 0.0 < z["eta"] < 100.0, \
        f"eta={z['eta']:.3e} 非有限或越界 (s_scale 未覆盖 stress_qp)"
    assert np.isfinite(z["total_error"]) and z["total_error"] > 0.0


# ═══════════════════════════════════════════════════════════════
# BC: 公共系统校验 + penalty 强度 (第四轮外部审查复现)
# ═══════════════════════════════════════════════════════════════

def test_bc_rejects_nan_stiffness():
    """K 含 NaN/Inf 必须显式拒绝 — 曾: 全约束分支返回 NaN 反力;
    有自由 DOF 时透传到 SuperLU 报误导性 'Factor is exactly
    singular'."""
    from fem2d.bc import apply_elimination, apply_penalty
    K_nan = csr_matrix(np.array([[np.nan, 0.0], [0.0, 1.0]]))
    F = np.zeros(2)
    with pytest.raises(ValueError, match="NaN/Inf"):
        apply_elimination(K_nan, F, [], [0, 1], [0.0, 0.0])
    with pytest.raises(ValueError, match="NaN/Inf"):
        apply_elimination(K_nan, F, [0], [1], [0.0])
    with pytest.raises(ValueError, match="NaN/Inf"):
        apply_penalty(K_nan, F, [0])


def test_solve_all_fixed_nan_stiffness_rejected():
    """全约束 + K 含 NaN 必须拒绝 — 曾 dirichlet_only 分支直接
    K·u−F 算反力, K 含 NaN 时静默返回 NaN 反力."""
    from fem2d.solver import _solve_linear_system
    K = csr_matrix(np.eye(8))
    K[0, 0] = np.nan
    fixed = list(range(8))
    with pytest.raises(ValueError, match="NaN/Inf"):
        _solve_linear_system(
            K, np.zeros(8), [], fixed, np.zeros(8),
            method="elimination", linear_solver="auto", n_dof=8, log=print)


def test_apply_penalty_rejects_nan_force():
    """apply_penalty 必须拒绝含 NaN 的 F — 曾接受并返回 NaN 载荷
   ."""
    from fem2d.bc import apply_penalty
    K = csr_matrix(np.eye(6) * 1e6)
    with pytest.raises(ValueError, match="NaN/Inf"):
        apply_penalty(K, np.array([0., 1, np.nan, 0, 0, 0]), [0])


def test_penalty_strength_minimum_1e4():
    """显式 penalty 必须 ≥ 1e4·max|K_ii| — 曾只要求 ≥ max|K_ii|:
    实测 K=1e12, penalty=1e12 时 u=0.5 (约束误差 50%) 被接受
   . 1e4 倍时约束误差 < 0.01%."""
    from fem2d.bc import apply_penalty
    from scipy.sparse.linalg import spsolve
    K = csr_matrix(np.eye(6) * 1e12)
    F = np.zeros(6)
    with pytest.raises(ValueError, match="1e4"):
        apply_penalty(K, F, [0], [1.0], penalty=1e12)      # = max|K_ii|
    # 1e4 倍门槛: 规定 u=1, 约束残差 = K/(K+p) = 1e-4 < 0.1%
    K_mod, F_mod, _ = apply_penalty(K, F, [0], [1.0], penalty=1e16)
    u = spsolve(K_mod, F_mod)
    assert 1.0 - u[0] < 1e-3, f"1e4 倍罚因子约束残差过大: {1-u[0]:.3e}"


# ═══════════════════════════════════════════════════════════════
# 微尺度标签 .6g (第四轮外部审查低优先级项)
# ═══════════════════════════════════════════════════════════════

def test_micro_scale_arc_label_scientific():
    """微尺度开放圆弧标签必须科学计数 — 曾 .3f 显示 R=0.000
    (第四轮外部审查: geometry/bc_apply/naming 的 .3f/.4f).
    轮 2 插件 3 起开放圆弧标签格式为 ρ=.. (spec: "圆弧 ρ=..,
    圆心(..,..)") — 断言 token 同步更新, 意图 (科学计数 + 无 0.000)
    不变."""
    from fem2d.boundary.geometry import classify
    R = 1e-16
    theta = np.linspace(0, np.pi, 17)
    coords = np.column_stack([R * np.cos(theta), R * np.sin(theta)])
    seg_type, label, _ = classify(coords, R, True)
    assert seg_type == "arc", f"半圆弧误判为 {seg_type}: {label}"
    assert "ρ=1e-16" in label, f"标签未用科学计数: {label}"
    assert "0.000" not in label, f"标签精度塌缩: {label}"
