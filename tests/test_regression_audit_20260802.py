"""2026-08-02 外部审计回归测试 — 10 项修复.

覆盖: 网格重绑缓存失效 / 求解前状态校验 / 点定位容差量纲 / 退化容差
逐单元缩放 / 边界残差三点 Gauss / 平衡诊断尺度 / Q4R 闭合边长宽比 /
Timoshenko 载荷方向 / 只读数组 / 直接赋值路由。
"""
import warnings

import numpy as np
import pytest


def _square_mesh(elem_type="CPS3", scale=1.0):
    """[-0.5, 0.5]² 方形, 2 个单元 (共享对角线)."""
    s = scale
    nodes = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5],
                      [-0.5, 0.5]], dtype=float) * s
    if elem_type in ("CPS3",):
        elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    else:
        elems = np.array([[0, 1, 2, 3]], dtype=int)
    from fem2d import Mesh
    return Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
                thickness=0.01, plane_type="stress", elem_type=elem_type)


# ═══════════════════════════════════════════════════════════════
# P1: 网格数组重绑定必须路由到 replace_nodes (缓存失效)
# ═══════════════════════════════════════════════════════════════

def test_direct_nodes_assignment_routes_to_replace_nodes():
    """先构建缓存再重绑 mesh.nodes — 结果必须与 replace_nodes 一致.

    旧行为: 直接赋值绕过缓存失效, 几何缓存过期 → ~55% 静默误差.
    """
    from fem2d import Mesh, solve

    def make():
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
                 thickness=1.0, plane_type="stress", elem_type="CPS3")
        for n in (0, 3):
            m.fix_node(n, "both", 0.0)
        m.add_force(1, 1e6, 0.0)
        return m

    scaled = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)

    ma = make()
    ma.build_connectivity()
    ma.replace_nodes(scaled)
    ua = solve(ma, method="elimination", verbose=False)["u"].reshape(-1, 2)[1, 0]

    mb = make()
    mb.build_connectivity()
    mb.nodes = scaled  # 直接赋值 — setter 路由
    ub = solve(mb, method="elimination", verbose=False)["u"].reshape(-1, 2)[1, 0]

    assert abs(ua - ub) / abs(ua) < 1e-12


def test_direct_nodes_assignment_validates_shape():
    """重绑错误形状的 nodes 必须报错 (经 setter 校验)."""
    m = _square_mesh()
    with pytest.raises(ValueError, match="形状"):
        m.nodes = np.zeros((3, 2))


def test_inplace_array_write_blocked():
    """nodes/elements 数组必须只读 — 原地写入被 numpy 拒绝."""
    m = _square_mesh()
    with pytest.raises(ValueError):
        m.nodes[0, 0] = 5.0
    with pytest.raises(ValueError):
        m.elements[0, 0] = 7


def test_replace_elements_invalidates_cache():
    """replace_elements 后求解必须使用新拓扑 (缓存失效)."""
    from fem2d import Mesh, solve
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 2, 3]]),
             E=210e9, nu=0.3, thickness=1.0, plane_type="stress")
    for n in (0, 3):
        m.fix_node(n, "both", 0.0)
    m.add_force(1, 1e6, 0.0)
    m.build_connectivity()
    # 换成另一对角线划分 (拓扑等价, 但连接数组变化)
    m.elements = np.array([[0, 1, 3], [1, 2, 3]], dtype=int)
    r = solve(m, method="elimination", verbose=False)
    assert np.all(np.isfinite(r["u"]))


# ═══════════════════════════════════════════════════════════════
# P2: 求解前状态校验 (构造后可写入非法参数)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("attr,value", [
    ("thickness", -1.0), ("thickness", 0.0),
    ("E", -1e9), ("E", 0.0),
    ("nu", 0.6), ("nu", -2.0),
    ("plane_type", "stress2"),
])
def test_invalid_state_rejected_at_solve(attr, value):
    """构造后写入非法物理参数 → solve 入口必须拒绝 (快速失败)."""
    from fem2d import solve
    m = _square_mesh()
    for n in (0, 1, 2, 3):
        m.fix_node(n, "both", 0.0)
    setattr(m, attr, value)
    with pytest.raises(ValueError):
        solve(m, method="elimination", verbose=False)


# ═══════════════════════════════════════════════════════════════
# P3: 点定位容差量纲混用 (大尺度坐标)
# ═══════════════════════════════════════════════════════════════

def test_point_location_large_scale_rejects_outside_point():
    """1e12 尺度网格: 域外点不得被误判在单元内.

    旧行为: 长度容差 1e-12×span=1 传入无量纲重心坐标判断,
    重心坐标 [-0.5, 0.75, 0.75] 的域外点被接受.
    """
    from fem2d.stress import point_in_element
    s = 1e12
    m = _square_mesh(scale=s)   # 单元0: (0,0),(s,0),(s,s); 单元1: (0,0),(s,s),(0,s)
    # (0.5s, 1.5s): y 超出两个三角形的 y 上限 s — 重心坐标对单元0
    # 为 [0.5, -1.0, 1.5], 旧容差 (tol=1) 下 -1.0 ≥ -1 被误判在单元内
    assert point_in_element(m, 0.5 * s, 1.5 * s) == -1
    # 域内点仍能定位
    assert point_in_element(m, 0.25 * s, 0.25 * s) >= 0


def test_point_location_normal_scale_still_works():
    """正常尺度下点定位行为不变."""
    from fem2d.stress import point_in_element
    m = _square_mesh()
    assert point_in_element(m, 0.0, 0.0) >= 0
    assert point_in_element(m, 0.4, 0.4) >= 0   # 对角线上 — 任一单元合法
    assert point_in_element(m, 10.0, 10.0) == -1


# ═══════════════════════════════════════════════════════════════
# M1: 退化容差逐单元缩放 (多尺度网格)
# ═══════════════════════════════════════════════════════════════

def test_jacobian_report_slender_cst_detected():
    """极瘦 CST 单元必须判退化 (复测 2026-08-02 反例).

    detJ=1e-14 (最小角 ~5.7e-13°) 的单元: 逐单元 detJ 缩放是恒真测试
    (|detJ| ≤ 1e-15·|detJ| 对非零 detJ 永假), 旧实现全部放行;
    degeneracy_measure = 2A/h_max² = 1e-14 < 1e-8 必须抓住.
    """
    from fem2d import Mesh
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1e-14]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
             thickness=1.0, plane_type="stress", elem_type="CPS3")
    m.build_connectivity()
    report = m.element_kernel.jacobian_report(m)
    assert not report.ok, "极瘦 CST 单元未被判退化"
    assert report.degenerate == 1


def test_jacobian_report_slender_q4_detected():
    """细长 Q4 (长宽比 1e12) 必须判退化 — detJ 缩放同样漏判."""
    from fem2d import Mesh
    nodes = np.array([[0, 0], [1e12, 0], [1e12, 1], [0, 1]], dtype=float)
    elems = np.array([[0, 1, 2, 3]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
             thickness=1.0, plane_type="stress", elem_type="CPS4")
    m.build_connectivity()
    report = m.element_kernel.jacobian_report(m)
    assert not report.ok
    assert report.degenerate == 1


def test_rewritten_bc_state_rejected_at_solve():
    """构造后重写 BC/载荷为非法值 → solve 入口拒绝 (复测建议)."""
    from fem2d import solve
    m = _square_mesh()
    for n in range(4):
        m.fix_node(n, "both", 0.0)

    m.prescribed_vals = {}
    m.fixed_dofs = np.array([999], dtype=int)          # 越界 DOF
    with pytest.raises(ValueError, match="out of range"):
        solve(m, method="elimination", verbose=False)
    m.fixed_dofs = np.array([0, 1], dtype=int)

    # API 自带参数校验 — 直接操纵列表绕过 API 才能复现"构造后重写"
    m.concentrated_forces.append({"node": 7, "force": (1e6, 0.0)})  # 节点越界
    with pytest.raises(ValueError, match="out of range"):
        solve(m, method="elimination", verbose=False)
    m.concentrated_forces = []

    m.surface_tractions.append({"nodes": (0, 9), "traction": (0.0, 1e6)})
    with pytest.raises(ValueError, match="out of range"):
        solve(m, method="elimination", verbose=False)


def test_jacobian_report_multiscale_no_false_degenerate():
    """大单元 1e9 + 小单元 1 共存: 小单元不得被全局 detJ 容差误判退化."""
    from fem2d import Mesh
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],          # 小三角形
        [1e9, 1e9], [2e9, 1e9], [1e9, 2e9],          # 大三角形
    ], dtype=float)
    elems = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
             thickness=1.0, plane_type="stress", elem_type="CPS3")
    m.build_connectivity()   # jacobian_determinants 需要几何缓存
    report = m.element_kernel.jacobian_report(m)
    assert report.ok, f"小单元被误判: bad={report.bad}, tolerance={report.tolerance:.2e}"


# ═══════════════════════════════════════════════════════════════
# M2: 边界牵引残差三点 Gauss (线性面力)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("s", [0.5, 1.0, 2.0])
def test_boundary_residual_linear_traction_scaling(s):
    """线性面力残差的边长缩放必须精确 (复测 2026-08-02).

    ty = y (y∈[-s/2,s/2]) 的线性面力: 边中点值为 0, 但真实残差
    η = √(h_e·∫y²dy) = √(s⁴/12) ≠ 0。旧中点采样给出 0; 第一版 Gauss
    实现多乘一个边长, 非单位边长下偏大 √s 倍 (s=2 偏大 √2)。

    构造: 纯 Dirichlet (u=0, σ=0), 左竖边施加线性面力 → 残差完全
    来自 t̄, 理论值可精确比较。
    """
    from fem2d import Mesh, solve
    from fem2d.error_est import element_refinement_indicator
    nodes = np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2],
                      [-s/2, s/2]], dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 2, 3]]),
             E=210e9, nu=0.3, thickness=0.01, plane_type="stress")
    for n in range(4):
        m.fix_node(n, "both", 0.0)
    m.add_traction(3, 0, lambda x, y: y, 0.0)   # 竖边 x=-s/2, ty=y 线性
    result = solve(m, method="elimination", verbose=False)
    ind = element_refinement_indicator(m, result)
    theory = np.sqrt(s**4 / 12.0)
    assert np.isclose(ind.sum(), theory, rtol=1e-10), \
        f"s={s}: 指标 {ind.sum():.6f} ≠ 理论 √(s⁴/12)={theory:.6f}"


# ═══════════════════════════════════════════════════════════════
# M4: Timoshenko 载荷方向 (P>0 = 向下)
# ═══════════════════════════════════════════════════════════════

def test_timoshenko_shear_traction_sign():
    """右端面合剪力必须向下 (负) — 与闭式解方向一致.

    旧行为: FE 载荷被施加成向上 +10000 N, 与注释声称的向下相反,
    方向错误被 abs() 掩盖.
    """
    from fem2d.convergence import _parabolic_shear_traction, _timoshenko_tip_deflection
    L, H, t, P_mag, E, nu = 5.0, 1.0, 0.1, 10000.0, 210e9, 0.3
    yy = np.linspace(-H / 2, H / 2, 501)
    total = t * np.trapezoid(
        _parabolic_shear_traction(yy, H, t, P_mag), yy)
    assert total < 0.0, f"合剪力应向下, 实际 {total:+.0f} N"
    uy = _timoshenko_tip_deflection(L, H, t, P_mag, E, nu)
    assert uy < 0.0, "P>0 向下载荷的闭式挠度应为负"


# ═══════════════════════════════════════════════════════════════
# M5: 平衡诊断尺度 (微牛模型不被 1N 下限污染)
# ═══════════════════════════════════════════════════════════════

def test_multiscale_mesh_not_rejected():
    """多尺度网格 (1 m + 1 nm 单元) 不得被全局容差误判退化 (复测反例).

    旧行为: span×1e-8 全局容差 (≈1e-8) 大于纳米单元边长 (1e-9),
    正常边被误报为退化边.
    """
    from fem2d.preprocess import validate_mesh
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],          # 1 m 三角形
        [0.0, 2.0], [1e-9, 2.0], [0.0, 2.0 + 1e-9],  # 1 nm 三角形
    ], dtype=float)
    elems = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    report = validate_mesh(nodes, elems, "CPS3")
    assert report["ok"], f"多尺度网格被误拒: {report['errors']}"
    assert not any("退化边" in e for e in report["errors"])


def test_nano_q4_point_location():
    """纳米尺度 Q4 点定位: 域外点不得被 1.0 尺度下限误判在单元内."""
    from fem2d import Mesh
    from fem2d.stress import point_in_element
    s = 1e-12
    nodes = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2, 3]]),
             E=210e9, nu=0.3, thickness=1.0, plane_type="stress",
             elem_type="CPS4")
    # 距单元 100 倍宽度的点 (旧容差 1e-10×1 远超单元 → 误判在单元内)
    assert point_in_element(m, 100.0 * s, 0.5 * s) == -1
    assert point_in_element(m, 0.5 * s, 0.5 * s) >= 0


def test_degenerate_q4i_reports_element():
    """退化 Q4I (中心 Jacobian 奇异) 必须给出带单元编号的诊断,
    而非裸 LinAlgError."""
    from fem2d import Mesh
    nodes = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)  # 全共线
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2, 3]]),
             E=210e9, nu=0.3, thickness=1.0, plane_type="stress",
             elem_type="CPS4I")
    with pytest.raises(ValueError, match="degenerate"):
        m.build_connectivity()


def test_kernel_mutual_recursion_detected():
    """第三方内核两个默认方法都未覆盖 → NotImplementedError 而非递归."""
    from fem2d.element import ElementKernel

    class _BothDefault(ElementKernel):
        name = "TESTBOTH"
        aliases = ()
        nodes_per_element = 3
        local_edges = ((0, 1), (1, 2), (2, 0))

        def build_geometry(self, nodes, elements):
            return {"areas": np.ones(len(elements)),
                    "centroids": np.zeros((len(elements), 2))}

        def stiffness_batch(self, mesh, element_slice=None):
            return np.zeros((mesh.n_elements, 6, 6))

        def jacobian_determinants(self, mesh):
            return np.ones((mesh.n_elements, 1))

        def body_force_vector(self, mesh, eid, body_force):
            return np.zeros(6)

        def shape_values_at(self, coords, x, y, tol=1e-12):
            return None

        def verify_mesh(self, mesh, verbose=True):
            return True

    from fem2d import Mesh
    m = Mesh(nodes=np.zeros((3, 2)), elements=np.array([[0, 1, 2]]),
             E=210e9, nu=0.3, thickness=1.0, elem_type="CPS3")
    m.element_kernel = _BothDefault()   # 替换为两个默认方法都未覆盖的内核
    with pytest.raises(NotImplementedError, match="response_at_quadrature"):
        m.element_kernel.compute_response(m, np.zeros(6))


@pytest.mark.parametrize("s", [1e-12, 1e-16])
def test_nano_shared_edge_sides_stress(s):
    """纳米网格共享边的双侧应力必须返回两侧 (复测反例).

    旧行为: 全局 span×1e-8 容差 > 纳米边长, 所有边被判退化跳过,
    sides 静默退化为单侧, average 不平均.
    s=1e-16 是判别用例: coord_ulp 曾用 max(..., 1.0) 抬到 1.4e-14,
    1e-16 边长全部被判退化 — s=1e-12 单独通过旧实现, 不能覆盖缺陷.
    """
    from fem2d import Mesh
    from fem2d.stress import stress_at_point
    nodes = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 3], [1, 2, 3]]),
             E=210e9, nu=0.3, thickness=1.0, plane_type="stress",
             elem_type="CPS3")
    m.build_connectivity()
    result = {"stress": np.array([[1., 2., 3.], [10., 12., 14.]])}
    sides = stress_at_point(m, result, 0.5 * s, 0.5 * s, mode="sides")
    assert isinstance(sides, tuple) and len(sides) == 2, \
        f"sides 应返回两侧, 得到 {sides!r}"
    avg = stress_at_point(m, result, 0.5 * s, 0.5 * s, mode="average")
    assert np.allclose(avg, 0.5 * (sides[0] + sides[1])), \
        f"average 应等于两侧均值, 得到 {avg}"


def test_body_programmatic_validation():
    """程序化 body 非法输入必须被 config 拒绝 (复测反例)."""
    from fem2d.config import AnalysisConfig
    with pytest.raises(ValueError, match="二元组"):
        AnalysisConfig(body=(1.0,))               # 1 分量
    with pytest.raises(ValueError, match="二元组"):
        AnalysisConfig(body=(1.0, 2.0, 3.0))      # 3 分量
    with pytest.raises(ValueError, match="有限数值"):
        AnalysisConfig(body=(float("nan"), 0.0))  # NaN
    with pytest.raises(ValueError, match="有限数值"):
        AnalysisConfig(body=(True, 0.0))          # 布尔
    # 合法形式
    AnalysisConfig(body=(0.0, -78000.0))
    AnalysisConfig(body=(lambda x, y: x, 0.0))


def test_multiscale_no_duplicate_node_warning():
    """多尺度网格 (1 m + 1 nm) 不得报告重复节点 (复测反例)."""
    from fem2d.preprocess import validate_mesh
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.0, 2.0], [1e-9, 2.0], [0.0, 2.0 + 1e-9],
    ], dtype=float)
    elems = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    report = validate_mesh(nodes, elems, "CPS3")
    assert not any("重复节点" in w for w in report["warnings"]), \
        f"纳米正常节点被误报重复: {report['warnings']}"


def test_von_mises_extreme_stress_no_nan():
    """极端应力 (1e308) 下 von Mises 不得为 NaN (归一化消除 inf−inf).

    vm 真实值可能超过 float64 上限 (1e308 分量 → vm≈2e308) — 此时
    inf 是数学正确结果, NaN 才是错误; 结果由 solver 的 isfinite
    检查拒绝.
    """
    from fem2d.material import von_mises
    stress = np.array([[1e308, 1e308, 1e308],
                       [1e-300, 1e-300, 1e-300],
                       [0.0, 0.0, 0.0]])
    vm = von_mises(stress, "stress")
    assert not np.any(np.isnan(vm)), f"vm 含 NaN: {vm}"
    vm_strain = von_mises(stress, "strain", nu=0.3)
    assert not np.any(np.isnan(vm_strain))
    # 常规值不回归 (含跨尺度混合: 大值+小值混合曾产生 NaN)
    assert np.isclose(von_mises(np.array([[100., 0., 0.]]), "stress")[0],
                      100.0)
    mixed = np.array([[1e308, 0.0, 0.0]])
    assert not np.isnan(von_mises(mixed, "stress")[0])


def test_degenerate_q4i_gauss_point_reports_quadrature():
    """中心 Jacobian 正常但某 Gauss 点奇异 → 报 quadrature id (复测反例).

    四边形 [0,0],[1,0],[1,1],[-0.5,b]: b 经数值求根使 detJ 在 Gauss 点
    (0.577,0.577) 为零; 中心 detJ=0.137 非零 (中心检查不拦), 循环内
    逐 Gauss 点检查必须拦截.
    """
    from fem2d import Mesh
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [-0.5, -0.401923788646684]],
        dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2, 3]]),
             E=210e9, nu=0.3, thickness=1.0, plane_type="stress",
             elem_type="CPS4I")
    with pytest.raises(ValueError, match="quadrature point 0"):
        m.build_connectivity()


def test_body_callable_supported():
    """整体 callable body 必须可用: body(x,y) → (bx,by) (复测反例)."""
    from fem2d.config import AnalysisConfig
    config = AnalysisConfig(body=lambda x, y: (0.0, -1.0))
    assert callable(config.body)
    # 契约校验: 返回值必须为 2 元组 (由 bc_apply 装配时校验,
    # 实际装配路径见 test_body_callable_apply_bcs_returns_tuple)


def test_body_callable_apply_bcs_returns_tuple():
    """整体 callable 体力: apply_bcs 返回契约必须是二元组 (复测反例).

    曾返回函数本身 — runner 固定 ``bfx, bfy = apply_bcs(...)`` 解包
    函数对象直接 TypeError; 新增测试只覆盖了配置/导入, 未穿过
    装配层, 契约问题漏网.
    """
    from fem2d.bc_apply import _apply_body_force
    from fem2d.config import AnalysisConfig
    from fem2d import Mesh
    import numpy as np
    m = Mesh(nodes=np.array([[1.0, 2.0], [2.0, 2.0], [1.5, 3.0]]),
             elements=np.array([[0, 1, 2]]),
             E=210e9, nu=0.3, thickness=1.0, plane_type="stress")
    config = AnalysisConfig(body=lambda x, y: (x, -y))  # 域相关: 形心 (1.5, 2.33)
    # 解包本身即契约验证 — 返回非二元组会在此 TypeError
    bfx, bfy = _apply_body_force(config, m, batch_mode=True)
    assert callable(bfx) and callable(bfy)
    # 分量独立求值: bfx(x,y) = 第一分量, bfy(x,y) = 第二分量
    assert bfx(1.5, 2.0) == 1.5 and bfy(1.5, 2.0) == -2.0
    assert m.body_force is config.body, "mesh.body_force 应透传整体函数"


def test_body_callable_hole_domain_no_probe():
    """带孔/凹域: 节点形心在孔洞内 — 不得在任意位置预调用 body (复测反例).

    方形环网格的节点形心 = (0,0) = 孔洞中心; 体力在 r=0 奇异但所有
    材料积分点合法。旧实现预调用形心 → ZeroDivisionError 误拒.
    """
    from fem2d.bc_apply import _apply_body_force
    from fem2d.config import AnalysisConfig
    from fem2d import Mesh
    import numpy as np
    nodes = np.array([
        [-2, -2], [2, -2], [2, 2], [-2, 2],      # 外圈
        [-1, -1], [1, -1], [1, 1], [-1, 1],      # 内圈 (孔洞)
    ], dtype=float)
    elems = np.array([[0, 1, 5, 4], [1, 2, 6, 5],
                      [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
             thickness=1.0, plane_type="stress", elem_type="CPS4")

    def radial(x, y):
        r2 = x * x + y * y
        if r2 == 0:
            raise ZeroDivisionError("r=0 奇点")
        return x / r2, y / r2

    config = AnalysisConfig(body=radial)
    bfx, bfy = _apply_body_force(config, m, batch_mode=True)
    assert callable(bfx) and callable(bfy)
    assert m.body_force is radial


def test_body_callable_bad_return_reported_at_gauss():
    """非法返回值不得在装配阶段裸崩 — 由 evaluate_vector_field 在
    Gauss 点给出带坐标的 ValueError."""
    from fem2d import Mesh
    from fem2d.bc_apply import _apply_body_force
    from fem2d.config import AnalysisConfig
    from fem2d.loads import assemble as assemble_loads
    import numpy as np
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3,
             thickness=1.0, plane_type="stress")
    config = AnalysisConfig(body=lambda x, y: (1.0,))  # 1 元组, 非法
    _apply_body_force(config, m, batch_mode=True)     # 设置阶段不报
    with pytest.raises(ValueError, match="二元组"):
        assemble_loads(m, m.n_dof)
    # 0-D ndarray 同样报 ValueError 而非泄漏 TypeError (复测 2026-08-02)
    m2 = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
              elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3,
              thickness=1.0, plane_type="stress")
    config2 = AnalysisConfig(body=lambda x, y: np.array(1.0))  # 0-D, 非法
    _apply_body_force(config2, m2, batch_mode=True)
    with pytest.raises(ValueError, match="二元组"):
        assemble_loads(m2, m2.n_dof)


def test_body_numpy_scalar_components():
    """np.float32/np.int64 体力分量必须被接受 (numbers.Real)."""
    from fem2d.config import AnalysisConfig
    import numpy as np
    AnalysisConfig(body=(np.float32(1.0), np.int64(2)))
    # 仍拒绝布尔
    with pytest.raises(ValueError, match="有限数值"):
        AnalysisConfig(body=(np.int64(1), True))


def test_symmetric_pressure_ring_balance_passes():
    """对称网格 + 自平衡内压: 力矩判据不得误杀 (高强度审计发现).

    完美对称网格下 Σ|单项力矩| 互相抵消到浮点噪声级, 相对判据分母
    退化 → rel≈0.26 误判失衡 (ΣM 绝对值仅 ~1e-12 N·m, 物理完美平衡).
    """
    from fem2d import Mesh, solve
    E, nu, t = 210e9, 0.3, 0.01
    ang = np.linspace(0, 2 * np.pi, 9)[:-1]
    nodes = np.vstack([np.column_stack([np.cos(ang), np.sin(ang)]), [0, 0]])
    elems = np.array([[i, (i + 1) % 8, 8] for i in range(8)], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=E, nu=nu, thickness=t,
             plane_type="stress")
    m.fix_node(8, "both", 0.0)
    m.fix_node(0, "y", 0.0)
    for i in range(8):
        m.add_pressure(i, (i + 1) % 8, 1e6)
    r = solve(m, method="elimination", verbose=False)
    assert np.all(np.isfinite(r["u"]))


def test_plane_verification_runs():
    """fem2d.verification 的平面应力/应变验证可被 pytest 直接调用."""
    from fem2d.verification import run_plane_verification
    p, f = run_plane_verification()
    assert f == 0, f"平面验证 {p} PASS, {f} FAIL"


def test_cylinder_constraint_leaves_inner_radial_free():
    """厚壁圆筒验证的最小约束必须不钉死内边界节点的径向位移 (包 6).

    判别性: 旧版 fix_node(0, "both") 把内边界 θ=0 节点 ux=uy=0, 该点
    径向位移被强制为零, 与 Lame 自由膨胀 u_r(a)≈9.08e-6 m 冲突 (相对
    误差 100% 必失败); 修复后三个约束全部沿切向, 节点 0 可自由径向
    运动, FE u_x 应接近 Lame 解。
    """
    import numpy as np
    from fem2d import Mesh, solve

    a, b_out, p, E, nu = 1.0, 2.0, 1e6, 2.1e11, 0.3
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
    # 与 verification.py 同步的切向最小约束 (θ=0 处 uy 切向、θ=π/2 处 ux 切向)
    m.fix_node(0, "y")
    m.fix_node(nr, "y")
    m.fix_node((nth // 4) * (nr + 1), "x")
    m.build_connectivity()
    for ea, eb in m.boundary_edges:
        ra = np.linalg.norm(nodes[ea])
        rb = np.linalg.norm(nodes[eb])
        if abs(ra - a) < 0.06 and abs(rb - a) < 0.06:
            m.add_pressure(int(ea), int(eb), p)
    r = solve(m, verbose=False)

    # Lame (plane strain): u_r(a) = (1+ν)a²p/(E(b²−a²))·[(1−2ν)a + b²/a]
    u_r_lame = ((1 + nu) * a**2 * p / (E * (b_out**2 - a**2))
                * ((1 - 2 * nu) * a + b_out**2 / a))
    assert abs(r["u"][0]) > 0.5 * u_r_lame, \
        "内边界节点 0 径向位移被约束钉死 (旧约束回归)"
    rel = abs(r["u"][0] - u_r_lame) / u_r_lame
    assert rel < 0.10, f"u_x(node0) 偏离 Lame {rel*100:.1f}%"


def test_balance_failure_message_renders():
    """失衡报错消息必须能渲染 — 判据与消息共享同一分母.

    回归: 判据重构为双判据后消息仍引用已删除的 f_scale 变量,
    失衡时抛 NameError 而非平衡诊断 (ruff F821 发现)。正常求解路径
    平衡恒成立, 该消息路径只能单元测试覆盖。
    """
    import numpy as np

    from fem2d.solver import _balance_failure_message
    msg = _balance_failure_message(
        np.array([1.0, -2.0]), 1e-8, 1e-13, 3.0, 1e-6)
    assert "Global equilibrium NOT satisfied" in msg
    assert "ΣF rel=" in msg and "ΣM rel=" in msg

def test_micro_newton_model_balance_passes():
    """微牛量级载荷模型: 平衡检查必须通过 (1N 固定下限移除后不误杀)."""
    from fem2d import solve
    m = _square_mesh()
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.add_force(2, 1e-9, 0.0)
    m.add_force(3, -1e-9, 0.0)   # 自平衡
    r = solve(m, method="elimination", verbose=False)
    # 曾 OR 恒真: 微牛位移 ~1e-21, abs(u).max()<1e-9 永远成立, balance_ok
    # 无论真假都通过. 自平衡载荷必须显式通过平衡检查.
    assert r["balance_ok"] is True


# ═══════════════════════════════════════════════════════════════
# M6: Q4R 长宽比含闭合边 3→0
# ═══════════════════════════════════════════════════════════════

def test_q4r_aspect_ratio_includes_closing_edge():
    """闭合边 (3→0) 是唯一短边时, 长宽比告警必须触发.

    单元 [0,0],[100,0],[95,100],[0,5]: 前 3 条边 ~100, 闭合边 = 5.
    旧实现只数 3 条边 → AR≈1 不告警; 修复后 AR≈20 → RuntimeWarning.
    """
    from fem2d import Mesh, solve
    nodes = np.array([[0, 0], [100, 0], [95, 100], [0, 5]], dtype=float)
    elems = np.array([[0, 1, 2, 3]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
             thickness=1.0, plane_type="stress", elem_type="CPS4R")
    for n in (0, 1, 2):
        m.fix_node(n, "both", 0.0)
    m.add_force(3, 1e6, 0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve(m, method="elimination", verbose=False)
    assert any(issubclass(w.category, RuntimeWarning)
               and "长宽比" in str(w.message) for w in caught)


def test_normalize_element_orientation_cw_to_ccw():
    """CW 单元导入后必须归一化为 CCW (高强度审计: 随包 demo 是 CW 几何)."""
    from fem2d.gmsh_adapter import normalize_element_orientation
    import numpy as np
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
    cw = np.array([[0, 2, 1]])        # 顺时针 (负面积)
    ccw = normalize_element_orientation(nodes, cw)
    s = 0.5 * ((nodes[ccw[0, 1], 0] - nodes[ccw[0, 0], 0])
               * (nodes[ccw[0, 2], 1] - nodes[ccw[0, 0], 1])
               - (nodes[ccw[0, 2], 0] - nodes[ccw[0, 0], 0])
               * (nodes[ccw[0, 1], 1] - nodes[ccw[0, 0], 1]))
    assert s > 0, "CW 单元未翻转为 CCW"
    # 已 CCW 的不动
    ok = np.array([[0, 1, 2]])
    assert np.array_equal(normalize_element_orientation(nodes, ok), ok)
    # 输入不被原地修改
    assert np.array_equal(cw, np.array([[0, 2, 1]]))
    # 空输入
    assert normalize_element_orientation(nodes, np.empty((0, 3))).shape == (0, 3)


def test_spec_no_plot_truthy_values():
    """.spec 的 no_plot 必须接受 1/0/true/false/yes/no (曾只认 "true")."""
    import os
    from fem2d.config import AnalysisConfig
    from fem2d.input_source import resolve_spec_overrides
    with open("models/_nptest.geo", "w") as f:
        f.write("Point(1) = {0, 0, 0, 0.5};\n")
    try:
        for val, expect in (("1", True), ("0", False), ("true", True),
                            ("false", False), ("yes", True), ("no", False)):
            p = f"models/_nptest_{val}.spec"
            with open(p, "w") as f:
                f.write(f"mesh = _nptest.geo\nno_plot = {val}\n")
            cfg = AnalysisConfig()
            resolve_spec_overrides(p, cfg)
            assert cfg.no_plot is expect, f"no_plot={val} → {cfg.no_plot}"
            os.unlink(p)
    finally:
        os.unlink("models/_nptest.geo")


def test_weighted_error_consistent_with_spr_on_linear_field():
    """weighted 误差估计不得混用均值恢复与积分点原始应力 (高强度审计).

    精确线性应力场: SPR/L2 报 0, weighted 修复前报 ~14.5% 虚假误差.
    """
    from fem2d import Mesh, solve
    from fem2d.error_est import estimate
    E, nu, t = 210e9, 0.3, 0.01
    m = Mesh(nodes=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
             elements=np.array([[0, 1, 2, 3]], dtype=int), E=E, nu=nu,
             thickness=t, plane_type="stress", elem_type="CPS4")
    m.fix_node(0, "both", 0.0)
    m.fix_node(3, "y", 0.0)
    m.fix_node(1, "x", 1e-4)
    m.fix_node(2, "x", 1e-4)
    r = solve(m, method="elimination", verbose=False)
    eta_w = estimate(m, r, method="weighted", verbose=False)["eta"]
    assert eta_w < 1e-6, f"weighted 线性场虚假误差: {eta_w:.4f}%"


def test_to_node_simple_is_arithmetic_mean():
    """_to_node 的 simple 方法必须是算术平均, 不得被 dA 加权污染."""
    from fem2d import Mesh
    from fem2d.visualize import _to_node
    # 歪扭 Q4: 各 Gauss 点 dA 不同 → simple(算术) ≠ weighted(dA 加权)
    nodes = np.array([[0, 0], [2, 0], [1.5, 2], [0, 1.5]], dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2, 3]]), E=210e9,
             nu=0.3, thickness=1.0, plane_type="stress", elem_type="CPS4")
    m.build_connectivity()
    data = np.ones((1, 4, 3)) * np.array([[[1., 2., 3.],
                                           [2., 4., 6.],
                                           [3., 6., 9.],
                                           [4., 8., 12.]]])
    simple = _to_node(m, None, integration_values=data, method="simple")
    weighted = _to_node(m, None, integration_values=data, method="weighted")
    # simple: 每个节点 = 4 个 Gauss 点算术平均 → 第 1 分量 2.5
    assert abs(simple[0, 0] - 2.5) < 1e-12, f"simple 非算术平均: {simple}"
    # 歪扭网格上两者必须不同 (曾相同)
    assert not np.allclose(simple, weighted), "simple 被 dA 加权污染"


def test_unwrap_angle_range_major_arc():
    """跨 ±π 切线的优弧角度必须正确 (高强度审计: 200° 弧曾报 340°)."""
    from fem2d.boundary.geometry import _unwrap_angle_range, classify
    import numpy as np

    def to_atan2(deg_list):
        return np.mod(np.deg2rad(deg_list) + np.pi, 2 * np.pi) - np.pi

    cases = [
        ([160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 0], 200.0),
        ([170, 190, 210, 230, 250, 270, 290, 310, 330, 350, 0], 190.0),
        ([10, 20, 30, 40, 50, 60, 70, 80, 90], 80.0),
    ]
    for deg_list, expect in cases:
        got = np.degrees(_unwrap_angle_range(to_atan2(deg_list)))
        assert abs(got - expect) < 1e-9, f"弧 {deg_list[0]}°→{deg_list[-1]}°: {got} ≠ {expect}"
    # 真实 classify 路径: 270° 弧 (45°→315° 长路)
    t = np.deg2rad(np.linspace(45, 315, 28))
    coords = np.array([[2 * np.cos(a), 2 * np.sin(a)] for a in t])
    _, _, info = classify(coords, 4.0, True, closed=False)
    assert abs(np.degrees(info["angle"]) - 270.0) < 1e-6


def test_orient2d_exact_sign_near_collinear():
    """orient2d 对 1-3 ulp 近共线输入必须精确 (高强度审计).

    对照口径用 float.hex() 精确值 — str() 是最短往返表示非精确十进制,
    曾因此误报 ~11% 符号错误。Dekker expansion 在 IEEE 754 double 下
    分裂只有 ε 级精度, 现用 as_integer_ratio 整数精确判定 (平台无关).
    """
    from decimal import Decimal, getcontext
    getcontext().prec = 100
    from fem2d.boundary.predicates import orient2d
    import numpy as np

    def hex_dec(x):
        h = float(x).hex()
        sign = -1 if h.startswith("-") else 1
        h = h.lstrip("+-")
        mant, _, exp = h.partition("p")
        ip, _, frac = mant[2:].partition(".")
        val = Decimal(int(ip + frac, 16))
        val *= Decimal(2) ** (int(exp) - 4 * len(frac))
        return sign * val

    def exact_sign(ax, ay, bx, by, cx, cy):
        # oracle 必须从原始坐标直接算 (曾先做 float(ax-cx) 浮点差再
        # 精确化 — 减法舍入使符号可翻转, 复测 2026-08-02 反例)
        d = ((hex_dec(ax) - hex_dec(cx)) * (hex_dec(by) - hex_dec(cy))
             - (hex_dec(ay) - hex_dec(cy)) * (hex_dec(bx) - hex_dec(cx)))
        return 1 if d > 0 else (-1 if d < 0 else 0)

    # 复测者反例: 普通量级坐标, 浮点差路径曾返回错误符号
    assert orient2d(
        0.0, 0.0,
        0.9998957067728043, 1.0001200437545776,
        0.1428422438246863, 0.14287429196493964) > 0.0

    rng = np.random.default_rng(2026)
    for _ in range(5000):
        ax, ay = float(rng.uniform(-1e6, 1e6)), float(rng.uniform(-1e6, 1e6))
        bx, by = float(rng.uniform(-1e6, 1e6)), float(rng.uniform(-1e6, 1e6))
        t = float(rng.uniform(0, 1))
        cx, cy = ax + t * (bx - ax), ay + t * (by - ay)
        cx += np.spacing(cx) * int(rng.integers(0, 4))   # 1-3 ulp 微扰
        expect = exact_sign(ax, ay, bx, by, cx, cy)
        got = orient2d(ax, ay, bx, by, cx, cy)
        got_sign = 1 if got > 0 else (-1 if got < 0 else 0)
        assert expect == got_sign, (
            f"近共线符号错误: A=({ax},{ay}) B=({bx},{by}) C=({cx},{cy}) "
            f"expect={expect} got={got_sign}")
    # 精确共线 → 0 (整数坐标)
    assert orient2d(1.0, 2.0, 3.0, 4.0, 5.0, 6.0) == 0.0
    assert orient2d(0.0, 0.0, 1.0, 0.0, 0.5, 1.0) > 0.0   # C 在 AB 左侧
    assert orient2d(0.0, 0.0, 1.0, 0.0, 0.5, -1.0) < 0.0  # 右侧


def test_gmsh_geometry_saveall_always_enforced():
    """所有 .geo 路径必须最终生效 Mesh.SaveAll=1 — 多重赋值 (1→0)
    也不能绕过 (复测 2026-08-02)."""
    import os
    import tempfile
    from scripts.gmsh_runner import _geometry_without_explicit_save
    cases = [
        ("Point(1) = {0, 0, 0, 0.5};\n", "clean"),
        ("Mesh.SaveAll = 1;\nPoint(1) = {0, 0, 0, 0.5};\n", "saveall1"),
        ("Mesh.SaveAll = 1;\nMesh.SaveAll = 0;\n"
         "Point(1) = {0, 0, 0, 0.5};\n", "1-then-0"),
        ("Mesh.SaveAll = 0;\nPoint(1) = {0, 0, 0, 0.5};\n", "saveall0"),
    ]
    for text, label in cases:
        with tempfile.NamedTemporaryFile("w", suffix=".geo",
                                         delete=False) as f:
            f.write(text)
            p = f.name
        try:
            path, tmp = _geometry_without_explicit_save(p)
            # 必须生成临时副本 (零复制已取消)
            assert tmp is not None and os.path.isfile(tmp), \
                f"{label}: 未生成副本"
            content = open(path, encoding="utf-8").read()
            # 末尾追加的 SaveAll=1 必须最后出现 (覆盖前面任何赋值)
            assert content.rstrip().endswith("Mesh.SaveAll = 1;"), \
                f"{label}: SaveAll=1 未最终生效"
        finally:
            if tmp and os.path.isfile(tmp):
                os.unlink(tmp)
            os.unlink(p)


def test_node_id_must_be_integer():
    """fix_node(1.5) 曾静默约束错误 DOF — 节点编号必须整数."""
    from fem2d import Mesh
    import numpy as np
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3, thickness=1.0)
    for bad in (1.5, True):
        with pytest.raises(TypeError):
            m.fix_node(bad, "both")
        with pytest.raises(TypeError):
            m.add_force(bad, 1e6, 0.0)
        with pytest.raises(TypeError):
            m.add_traction(bad, 0, 1e6, 0.0)
    m.fix_node(1.0, "both", 0.0)   # 整数值浮点兼容
    assert m.fixed_dofs.tolist() == [2, 3]


def test_parse_vec2_rejects_nonfinite():
    """CLI NaN/Inf 体力/面力曾静默不施加 — 现在必须报错."""
    from fem2d.loads import parse_vec2
    for expr in ("nan,0", "inf,0", "0,1e999"):
        with pytest.raises(ValueError, match="有限数值"):
            parse_vec2(expr)
    parse_vec2("1e6,0")   # 合法不受影响


def test_parse_vec2_fullwidth_comma_accepted():
    """全角逗号必须正常解析 — 曾报"需要两个分量"误导用户."""
    from fem2d.loads import parse_vec2
    bx, by = parse_vec2("1e6，0")
    assert bx == 1e6 and by == 0.0
    bx, by = parse_vec2("0，-78000")
    assert bx == 0.0 and by == -78000.0


def test_parse_vec2_expr_syntax_error_reports_expression():
    """表达式语法错误必须 ValueError 且带原表达式 — 曾裸 SyntaxError 无
    载荷上下文."""
    from fem2d.loads import parse_vec2
    with pytest.raises(ValueError, match="sin\\("):
        parse_vec2("0,sin(x")
    with pytest.raises(ValueError, match="__import__"):
        parse_vec2("0,__import__('os')")   # AST 白名单拒绝任意调用


def test_isoband_band_count_capped():
    """isoband 层数无上限曾耗尽内存/OverflowError — 现限制 10000 (审计)."""
    from fem2d.config import AnalysisConfig
    with pytest.raises(ValueError, match="10000"):
        AnalysisConfig(band_min=0, band_max=1, band_step=1e-12)
    AnalysisConfig(band_min=0, band_max=1, band_step=1e-4)   # 10000 层合法


def test_q4_large_origin_geometry_stable():
    """大坐标原点 + 小局部尺寸: 面积/形心不得消差.

    修复前 1e6 原点 + 0.01 边长 → 面积误差 22%、形心偏差 6 万;
    0.001 边长合法单元被判退化。
    """
    from fem2d.element.q4 import _polygon_geometry
    import numpy as np
    for side in (0.1, 0.01, 0.001):
        ox, oy = 1e6, -1e6
        coords = np.array([[[ox, oy], [ox + side, oy],
                            [ox + side, oy + side], [ox, oy + side]]])
        area, cent = _polygon_geometry(coords)
        # 误差 ≤ 输入坐标固有 ulp 级 (1e6+side 无法精确表示, 面积偏差
        # 来源是坐标舍入而非算法 — 修复前 0.01 边长误差 22%)
        assert abs(area[0] - side * side) / (side * side) < 1e-6, \
            f"边长 {side}: 面积 {area[0]} vs {side*side}"
        assert abs(cent[0, 0] - (ox + side / 2)) < 1e-6, "形心 x 偏差"
        assert abs(cent[0, 1] - (oy + side / 2)) < 1e-6, "形心 y 偏差"


def test_rigid_body_scale_invariance():
    """刚体模态检查的尺度不变性: 1e-16 尺寸不得误报转动模态."""
    from fem2d import Mesh, solve
    import numpy as np
    s = 1e-16
    m = Mesh(nodes=np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=float),
             elements=np.array([[0, 1, 2, 3]], dtype=int), E=210e9, nu=0.3,
             thickness=s, plane_type="stress", elem_type="CPS4")
    m.fix_node(0, "both", 0.0)
    m.fix_node(3, "x", 0.0)
    m.add_force(1, 1e-30, 0.0)
    r = solve(m, method="elimination", verbose=False)
    assert np.all(np.isfinite(r["u"]))


def test_apply_penalty_argument_validation():
    """apply_penalty 公开 API: 长度不等/负 DOF 必须拒绝."""
    from fem2d.bc import apply_penalty
    import numpy as np
    K = np.diag(np.ones(4))
    F = np.zeros(4)
    with pytest.raises(ValueError, match="长度必须相等"):
        apply_penalty(K, F, [0, 1], [0.1])       # 长度不等
    with pytest.raises(ValueError, match="out of range"):
        apply_penalty(K, F, [-1], [0.1])         # 负 DOF
    apply_penalty(K, F, [0, 1], [0.1, 0.2])     # 合法


def test_near_integer_connectivity_rejected():
    """1.000001 的连接关系必须拒绝 (np.allclose 曾静默四舍五入)."""
    from fem2d.preprocess import validate_mesh, MeshValidationError
    import numpy as np
    nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    with pytest.raises(MeshValidationError, match="non-integer"):
        validate_mesh(nodes, np.array([[0, 1.000001, 2]]))
    rep = validate_mesh(nodes, np.array([[0.0, 1.0, 2.0]]))
    assert rep["ok"]


def test_band_step_extreme_friendly_error():
    """band_step=1e-320 必须友好报错 (曾 int(inf) 抛 OverflowError)."""
    from fem2d.config import AnalysisConfig
    with pytest.raises(ValueError, match="10000"):
        AnalysisConfig(band_min=0, band_max=1, band_step=1e-320)


def test_material_rejects_nan():
    """D_matrix(nan, ...) 必须拒绝 (曾返回全 nan 矩阵)."""
    from fem2d.material import D_matrix
    with pytest.raises(ValueError):
        D_matrix(float("nan"), 0.3)
    with pytest.raises(ValueError):
        D_matrix(210e9, float("inf"))


def test_large_id_near_integer_rejected():
    """大编号 100001.25 不得被 np.allclose 静默取整."""
    from fem2d import Mesh
    import numpy as np
    nodes = np.zeros((100002, 2))
    with pytest.raises(ValueError, match="integers"):
        Mesh(nodes=nodes, elements=np.array([[100001.25, 0, 1]]),
             E=210e9, nu=0.3, thickness=1.0)


@pytest.mark.parametrize("s", [1e-6, 1e-16])
def test_micro_scale_edge_load_not_dropped(s):
    """微尺度网格边载荷不得被绝对 1e-15 下限静默丢弃.

    s=1e-16 是判别用例: 旧判据 64·eps·max(全局坐标, 1.0) ≈ 1.4e-14,
    边长 1e-16 会被误判退化 — 测试放回旧实现即失败; s=1e-6 单独
    通过旧实现, 不能覆盖缺陷.
    """
    from fem2d import Mesh, solve
    import numpy as np
    nodes = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 2, 3]]),
             E=210e9, nu=0.3, thickness=1e-6, plane_type="stress")
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.fix_node(3, "both", 0.0)
    m.add_traction(2, 3, 0.0, 1.0)   # 顶边 ty=1 Pa
    r = solve(m, method="elimination", verbose=False)
    Ry = r["reactions"].reshape(-1, 2)[:, 1].sum()
    theory = 1.0 * s * 1e-6          # p·L·t
    assert abs(Ry + theory) < 1e-18, f"边载荷被丢弃: ΣRy={Ry} vs 理论 {theory}"


def test_micro_scale_pressure_normal_no_absolute_threshold():
    """微尺度 (s=1e-16) 法向压力: boundary_outward_normal 不得用绝对
    1e-15 阈值判退化 — 曾与边载荷同病."""
    from fem2d import Mesh, solve
    import numpy as np
    s = 1e-16
    nodes = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 2, 3]]),
             E=210e9, nu=0.3, thickness=1e-6, plane_type="stress")
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.fix_node(3, "both", 0.0)
    m.add_pressure(2, 3, 1.0)        # 顶边 p=1 Pa (外法向 +y)
    r = solve(m, method="elimination", verbose=False)
    Ry = r["reactions"].reshape(-1, 2)[:, 1].sum()
    theory = 1.0 * s * 1e-6          # p·L·t
    assert abs(Ry - theory) < 1e-18, f"压力载荷错误: ΣRy={Ry} vs 理论 {theory}"


def test_micro_scale_mesh_import_no_false_degenerate():
    """微尺度网格导入不得因 1.0 下限 ULP 误报重复节点/退化边.

    preprocess._coordinate_ulp 曾用 max(..., 1.0), 1e-16 网格的容差被
    抬到 ~1.4e-14, 全部节点判重复、全部边判退化.
    """
    from fem2d.preprocess import validate_mesh
    import numpy as np
    s = 1e-16
    nodes = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=float)
    rep = validate_mesh(nodes, np.array([[0, 1, 2], [0, 2, 3]]))
    assert not rep.get("errors"), f"微尺度网格被误报错误: {rep.get('errors')}"
    assert not any("重复节点" in w for w in rep.get("warnings", [])), \
        f"微尺度节点被误报重复: {rep.get('warnings')}"


def test_construct_direct_load_float_integer_normalized():
    """构造函数直传整数值浮点节点号须规范化写回 (曾组装时 IndexError).

    2.0 通过验证后仍以 float 留在记录里, 组装时 2*nid 触发 IndexError
    — 必须真正保存 int.
    """
    from fem2d import Mesh, solve
    import numpy as np
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3, thickness=1.0)
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.concentrated_forces = [{"node": 2.0, "force": (1e6, 0.0)}]
    r = solve(m, method="elimination", verbose=False)
    nid = m.concentrated_forces[0]["node"]
    assert nid == 2 and type(nid) is int, f"节点号未写回 int: {nid!r}"
    assert r["u"].reshape(-1, 2)[2, 0] > 0, "节点 2 应受 +x 载荷"


def test_construct_direct_surface_traction_float_nodes_normalized():
    """构造函数直传浮点节点对面力须规范化写回 (曾组装时 IndexError)."""
    from fem2d import Mesh, solve
    import numpy as np
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3, thickness=1.0)
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.surface_tractions = [{"nodes": (0.0, 1.0), "traction": (1e6, 0.0)}]
    solve(m, method="elimination", verbose=False)
    nodes = m.surface_tractions[0]["nodes"]
    assert nodes == (0, 1) and all(type(n) is int for n in nodes), \
        f"面力节点未写回 int: {nodes!r}"


def test_construct_direct_load_non_integer_rejected():
    """构造函数直传非整数节点号必须拒绝 (1.5 曾在组装时 IndexError)."""
    from fem2d import Mesh, solve
    import numpy as np
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3, thickness=1.0)
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.concentrated_forces = [{"node": 1.5, "force": (1e6, 0.0)}]
    with pytest.raises(TypeError, match="integer"):
        solve(m, method="elimination", verbose=False)


def test_construct_interior_edge_traction_rejected_at_solve():
    """构造函数直传内部边面力必须在 solve 时即拒绝.

    曾先成功求解、误差估计阶段才崩溃 — 同一模型"先成功再失败"的行为
    不一致; validate_state 须覆盖所有原始面力记录.
    """
    from fem2d import Mesh, solve
    import numpy as np
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 2, 3]]),
             E=210e9, nu=0.3, thickness=1.0)
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.surface_tractions = [{"nodes": (0, 2), "traction": (1.0, 0.0)}]
    with pytest.raises(ValueError, match="interior edge"):
        solve(m, method="elimination", verbose=False)


def test_spec_no_plot_whitelist():
    """.spec no_plot 非法字符串必须报错 (曾静默当 False)."""
    import os
    from fem2d.config import AnalysisConfig
    from fem2d.input_source import resolve_spec_overrides
    with open("models/_nptest2.geo", "w") as f:
        f.write("Point(1) = {0, 0, 0, 0.5};\n")
    try:
        with open("models/_npbad.spec", "w") as f:
            f.write("mesh = _nptest2.geo\nno_plot = banana\n")
        with pytest.raises(ValueError, match="no_plot"):
            resolve_spec_overrides("models/_npbad.spec", AnalysisConfig())
    finally:
        os.unlink("models/_nptest2.geo")
        if os.path.isfile("models/_npbad.spec"):
            os.unlink("models/_npbad.spec")
