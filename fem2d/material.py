"""Topology-independent linear-elastic material operations."""
import sys

import numpy as np

from .checks import require_finite_positive, require_nu_valid


def D_matrix(E, nu, plane_type="stress"):
    """Return the isotropic 2-D constitutive matrix."""
    require_finite_positive(E, "E")
    require_nu_valid(nu, "nu")

    D = np.zeros((3, 3))
    if plane_type == "stress":
        factor = E / (1.0 - nu * nu)
        D[0, 0] = D[1, 1] = 1.0
        D[0, 1] = D[1, 0] = nu
        D[2, 2] = (1.0 - nu) / 2.0
        D *= factor
    elif plane_type == "strain":
        if nu > 0.45:
            print(
                f"  [WARN] ν = {nu} > 0.45 in plane strain → "
                "near-incompressible material.",
                file=sys.stderr,
            )
            print(
                "         Standard displacement formulations may exhibit "
                "volumetric locking; verify mesh convergence and use a "
                "mixed/selective formulation when needed as ν → 0.5.",
                file=sys.stderr,
            )
        factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D[0, 0] = D[1, 1] = 1.0 - nu
        D[0, 1] = D[1, 0] = nu
        D[2, 2] = (1.0 - 2.0 * nu) / 2.0
        D *= factor
    else:
        raise ValueError(
            f"plane_type='{plane_type}' — must be 'stress' or 'strain'. "
            "Bathe Table 4.3 supports these two 2-D idealizations.")
    return D


def von_mises(stress, plane_type="stress", nu=0.3):
    """Return von Mises stress for one or many in-plane stress vectors.

    归一化计算防溢出: 极端但有限的应力 (如 1e308) 下分量平方 → inf,
    inf−inf → NaN。先按最大|分量|缩放为无量纲量, 再乘回尺度。
    """
    stress = np.asarray(stress)
    if stress.ndim < 2 or stress.shape[-1] != 3:
        # 标量/1-D/末维≠3 曾冒裸 IndexError (fuzz 2026-08-03)
        raise ValueError(
            f"von_mises: stress 必须为 (..., 3) 数组 [σx, σy, τxy], "
            f"got {stress.shape}")
    if not np.all(np.isfinite(stress)):
        raise ValueError(
            "von_mises: stress contains NaN/Inf — 对非法输入静默返回 "
            "NaN 曾掩盖上游错误")
    scale = np.maximum(
        np.max(np.abs(stress), axis=-1), np.finfo(float).tiny)
    sx = stress[..., 0] / scale
    sy = stress[..., 1] / scale
    txy = stress[..., 2] / scale
    if plane_type == "strain":
        sz = nu * (sx + sy)
        vm_hat = np.sqrt(
            0.5 * (
                (sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2
            ) + 3.0 * txy**2
        )
    elif plane_type == "stress":
        vm_hat = np.sqrt(
            sx**2 + sy**2 - sx * sy + 3.0 * txy**2)
    else:
        raise ValueError(
            f"plane_type='{plane_type}' — must be 'stress' or 'strain'.")
    # 乘回可能真溢出 (vm 超过 float64 上限) — 数学正确的结果, 静默返回
    # inf; NaN 已由归一化消除 (inf−inf 不再发生)
    with np.errstate(over="ignore"):
        return scale * vm_hat
