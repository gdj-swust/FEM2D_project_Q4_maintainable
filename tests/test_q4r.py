"""Q4R 沙漏系数入口校验 + 独立物理锚点 (R-ε 包 E1).

判别性: 旧实现 validate_hourglass_coefficient 对字符串/None 冒裸
TypeError ("ufunc 'isfinite' not supported"), 对数组冒 ambiguous-truth
ValueError, 0/负值/NaN 虽拒绝但消息无参数名上下文. 以下断言锁定
checks.require_finite_positive 的统一消息格式 (<name>=<value> — 原因).

锚点测试 (文件底部): 审查发现唯一公式测试
test_hourglass_coefficient_valid_sanity 是"同一公式两入口互证" —
沙漏稳定项符号翻转/系数缩放/投影矩阵错 → 两侧同错照绿. 补两个
独立推导的闭式锚点: 常应力 patch test (本构矩阵测试内手写) +
checkerboard 模态反力闭式值 (2×2 高斯对二次被积函数精确).
"""
import numpy as np
import pytest

from fem2d import Mesh, solve
from fem2d.assembly import assemble_sparse
from fem2d.element.q4r import (
    HOURGLASS_COEFFICIENT,
    element_stiffness,
    validate_hourglass_coefficient,
)
from fem2d.patch_test import _gen_irregular_q4_patch

COORDS = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _two_quad_q4r_mesh():
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
    ])
    elements = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])
    return Mesh(
        nodes, elements, E=210e9, nu=0.3, thickness=0.1,
        plane_type="stress", elem_type="CPS4R",
    )


@pytest.mark.parametrize("bad", ["0.5", "abc", None, [0.1], np.array([0.1, 0.2])])
def test_hourglass_coefficient_non_numeric_rejected(bad):
    """字符串/None/容器 → TypeError 带参数名 (旧实现: 裸 numpy TypeError)."""
    with pytest.raises(TypeError, match="hourglass_coefficient="):
        validate_hourglass_coefficient(bad)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_hourglass_coefficient_nonfinite_rejected(bad):
    """NaN/Inf → ValueError "must be finite" (旧实现: 无参数名上下文)."""
    with pytest.raises(ValueError, match="must be finite"):
        validate_hourglass_coefficient(bad)


@pytest.mark.parametrize("bad", [0.0, -1e-3, -1.0])
def test_hourglass_coefficient_nonpositive_rejected(bad):
    """0/负值 → ValueError "must be > 0" (旧实现: 无参数名上下文)."""
    with pytest.raises(ValueError, match="must be > 0"):
        validate_hourglass_coefficient(bad)


def test_hourglass_coefficient_valid_scalars_accepted():
    """合法标量 (int/float/numpy scalar) → 通过, 返回 float 规范化值."""
    for good in (HOURGLASS_COEFFICIENT, 1, np.float64(0.05)):
        assert validate_hourglass_coefficient(good) == float(good)


def test_element_stiffness_rejects_bad_hourglass():
    """标量入口 element_stiffness: 非法系数在公式前拒绝."""
    with pytest.raises(TypeError, match="hourglass_coefficient="):
        element_stiffness(COORDS, 210e9, 0.3, 0.1, "stress",
                          hourglass_coefficient="0.5")
    with pytest.raises(ValueError, match="must be > 0"):
        element_stiffness(COORDS, 210e9, 0.3, 0.1, "stress",
                          hourglass_coefficient=0.0)


def test_stiffness_batch_rejects_bad_hourglass():
    """批量入口 stiffness_batch (生产路径): 非法系数在装配前拒绝."""
    mesh = _two_quad_q4r_mesh()
    mesh.build_connectivity()
    mesh.element_kernel.hourglass_coefficient = "0.5"
    with pytest.raises(TypeError, match="hourglass_coefficient="):
        mesh.element_kernel.stiffness_batch(mesh)
    mesh.element_kernel.hourglass_coefficient = 0.0
    with pytest.raises(ValueError, match="must be > 0"):
        mesh.element_kernel.stiffness_batch(mesh)


def test_hourglass_coefficient_valid_sanity():
    """合法自定义系数两条路径均正常 — 改动校验不得破坏公式路径."""
    mesh = _two_quad_q4r_mesh()
    mesh.build_connectivity()
    mesh.element_kernel.hourglass_coefficient = 0.5
    K_batch = mesh.element_kernel.stiffness_batch(mesh)
    K_scalar = element_stiffness(
        mesh.nodes[mesh.elements[0]], mesh.E, mesh.nu, mesh.thickness,
        mesh.plane_type, hourglass_coefficient=0.5)
    assert np.all(np.isfinite(K_batch))
    assert np.all(np.isfinite(K_scalar))
    assert np.allclose(K_batch[0], K_scalar, rtol=1e-12)


# ── R-ε E1: 独立物理锚点 (审查 2026-08-05 建议"若只修一条, 优先此条") ──

def test_q4r_patch_test_constant_stress_exact():
    """独立锚点 (1/2): 常应力 patch test — 本构矩阵在测试内手写.

    4 单元不规则 patch (内部节点 4 留自由), 边界钉到仿射位移场;
    三个独立常应力态 × 平面应力/应变. 位移场由柔度 C = D⁻¹ 导出,
    D 用平面态闭式直接构造 — 与生产 D_matrix/单元公式无共享代码.
    Q4R 单点积分对常应变精确; 沙漏项在仿射场恒为零不参与.
    """
    E, nu = 2.1e11, 0.3
    D_by_plane = {
        "stress": E / (1 - nu ** 2) * np.array([
            [1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu) / 2.0]]),
        "strain": E / ((1 + nu) * (1 - 2 * nu)) * np.array([
            [1 - nu, nu, 0.0], [nu, 1 - nu, 0.0],
            [0.0, 0.0, (1 - 2 * nu) / 2.0]]),
    }
    nodes, elements = _gen_irregular_q4_patch()
    boundary = [0, 1, 2, 3, 5, 6, 7, 8]  # 内部节点 4 自由 (真实 patch test)
    interior = 4
    for plane, D in D_by_plane.items():
        C = np.linalg.inv(D)
        for sx, sy, txy in (
                (1.0e6, 0.0, 0.0), (0.0, 1.0e6, 0.0), (0.0, 0.0, 1.0e6)):
            eps = C @ np.array([sx, sy, txy])
            mesh = Mesh(nodes=nodes, elements=elements, E=E, nu=nu,
                        thickness=1.0, plane_type=plane, elem_type="CPS4R")
            for n in boundary:
                x, y = nodes[n]
                mesh.fix_node(n, "x", eps[0] * x + 0.5 * eps[2] * y)
                mesh.fix_node(n, "y", 0.5 * eps[2] * x + eps[1] * y)
            result = solve(mesh, method="elimination", verbose=False)
            # Q4R 只报质心单点值 (response_at_quadrature 约定) — (ne, 1, 3).
            # 相对误差以应力尺度为基准 — 零应力分量含 ~1e-11 数值噪声,
            # 逐分量 rtol 会误伤.
            stress_err = np.max(np.abs(
                result["stress_qp"][:, 0] - np.array([sx, sy, txy])))
            assert stress_err <= 1e-10 * max(
                abs(sx), abs(sy), abs(txy))
            x, y = nodes[interior]
            u_ana = np.array([eps[0] * x + 0.5 * eps[2] * y,
                              0.5 * eps[2] * x + eps[1] * y])
            u_num = result["u"].reshape(-1, 2)[interior]
            assert np.allclose(u_num, u_ana, rtol=1e-10, atol=0.0)


def test_q4r_hourglass_stabilization_closed_form():
    """独立锚点 (2/2): 沙漏稳定项闭式值 — checkerboard 模态反力.

    单位方单元素钉死到 x 方向 checkerboard 位移 (1,-1,1,-1): 单点
    积分在中心处该模态应变为零 → 反力完全来自沙漏项. 该模态应变
    (2η, 0, 2ξ) 配 D_shear = diag(2μ,2μ,μ) 是 ξ/η 的二次式, 2×2
    高斯积分精确 → hᵀK_shear h = 4μt (det=1/4 权重在内), 反力
    f = K·u = αμt·(1,-1,1,-1), 应变能 ½uᵀf = 2αμt.

    旧 sanity 测试两侧共享实现 — 稳定项符号翻转/系数缩放/投影矩阵
    错 → 同错同绿; 此闭式值独立于生产公式, 任何稳定项错误均偏离.
    """
    for E, nu, t in ((2.1e11, 0.3, 1.0), (1000.0, 0.0, 1.0),
                     (2.1e11, 0.3, 0.1)):
        mesh = Mesh(nodes=COORDS, elements=np.array([[0, 1, 2, 3]]),
                    E=E, nu=nu, thickness=t, plane_type="stress",
                    elem_type="CPS4R")
        K = assemble_sparse(mesh).toarray()
        u = np.zeros(8)
        u[0::2] = (1.0, -1.0, 1.0, -1.0)
        f = K @ u
        mu = E / (2.0 * (1.0 + nu))
        expected = HOURGLASS_COEFFICIENT * mu * t
        assert np.allclose(f[0::2],
                           expected * np.array([1.0, -1.0, 1.0, -1.0]),
                           rtol=1e-12, atol=0.0)
        assert np.max(np.abs(f[1::2])) <= 1e-9 * expected
        assert np.isclose(0.5 * u @ f,
                          2.0 * HOURGLASS_COEFFICIENT * mu * t, rtol=1e-12)
