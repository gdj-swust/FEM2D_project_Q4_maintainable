"""位移模长/模型跨度 hypot 溢出守卫测试 (审查修复包第 1 项).

旧实现 np.sqrt(u²+v²) 在 |u| ~ 1e308 时先平方溢出成 inf — 有限模长被
算成 inf, 变形放大系数退 0、摘要打印 inf。判别性: 放回旧实现 (平方和
再开方) 以下测试必须失败。
"""
import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from fem2d import Mesh
from fem2d.reporting import displacement_scale, print_result_summary
from fem2d.solver import _small_deformation_check


def _plate():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        E=2.1e11, nu=0.3, thickness=0.01, plane_type="stress",
        elem_type="CPS4")


def test_displacement_scale_finite_for_huge_ux():
    """ux=1e308 有限 → 模长有限, 放大系数 > 0 (旧实现退 0)."""
    mesh = _plate()
    u = np.array([1e308, 0.0, 1e308, 0.0, 1e308, 0.0, 1e308, 0.0])
    scale = displacement_scale(mesh, u)
    assert np.isfinite(scale) and scale > 0.0
    # span=1.0, mag.max()=1e308 → scale = 1e-309 (量级可锁, 子正常数舍入放宽)
    assert np.isclose(scale, 1e-309, rtol=1e-3)


def test_print_result_summary_huge_ux_not_inf():
    """摘要"最大总位移"在 ux=1e308 时打印有限值 (旧实现打印 inf)."""
    mesh = _plate()
    mesh.validate_state()  # 构建 centroids (solve 前的正常状态)
    result = {
        "u": np.array([1e308, 0.0, 1e308, 0.0, 1e308, 0.0, 1e308, 0.0]),
        "vm_stress": np.array([1.0, 1.0]),
        "residual": 1e-17,
        "condition_info": None,
    }
    z2 = {
        "total_error": 1e-3, "energy_norm": 1e-2, "eta": 1.0,
        "worst_elem": 0, "elem_contrib": [0.5, 0.5],
        "stress_jumps": {"avg_jump": 1e-4, "max_jump": 1e-3},
    }
    q = {
        "grade": "A", "area_min": 1e-4, "area_max": 1e-4, "area_mean": 1e-4,
        "area_cv": 0.0, "ratio_max": 1.0, "ratio_ok": 1, "ratio_warn": 0,
        "ratio_bad": 0, "angle_min": 90.0, "angle_max": 90.0, "angle_ok": 1,
        "angle_warn": 0, "angle_bad": 0, "jacobian_neg": 0,
    }

    class _Cfg:
        plane = "stress"
        nu = 0.3
        E = 2.1e11

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_result_summary(_Cfg(), mesh, result, z2, q, 1e-309,
                             False, 2.0, 2.0)
    out = buf.getvalue()
    assert "1.000000e+308" in out  # 模长有限且值正确
    assert "inf" not in out        # 旧实现平方溢出 → 打印 inf


def test_small_deformation_span_hypot_no_overflow():
    """模型跨度 ~1e308 时 model_span 必须有限 — 位移比 >0.1 仍能告警.

    旧实现 (ptp_x²+ptp_y²)^0.5 溢出成 inf → 位移比恒 0 → 大变形静默放行.
    """
    mesh = Mesh(
        nodes=np.array(
            [[0., 0.], [1e308, 0.], [1e308, 1e308], [0., 1e308]]),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        E=2.1e11, nu=0.3, thickness=0.01, plane_type="stress",
        elem_type="CPS4")
    u = np.zeros(8)
    u[1] = 2e307  # 位移变化 = 0.141 × model_span (1.414e308) — 必须告警
    with pytest.warns(RuntimeWarning, match="Small-deformation"):
        _small_deformation_check(mesh, u)
