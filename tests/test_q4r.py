"""Q4R 沙漏系数入口校验 — 组 C (q4r 沙漏 + sigma_ref).

判别性: 旧实现 validate_hourglass_coefficient 对字符串/None 冒裸
TypeError ("ufunc 'isfinite' not supported"), 对数组冒 ambiguous-truth
ValueError, 0/负值/NaN 虽拒绝但消息无参数名上下文. 以下断言锁定
checks.require_finite_positive 的统一消息格式 (<name>=<value> — 原因).
"""
import numpy as np
import pytest

from fem2d import Mesh
from fem2d.element.q4r import (
    HOURGLASS_COEFFICIENT,
    element_stiffness,
    validate_hourglass_coefficient,
)

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
