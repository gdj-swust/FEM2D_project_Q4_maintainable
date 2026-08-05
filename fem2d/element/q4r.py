"""Stabilized one-point quadrilateral (Q4R).

The constitutive part is integrated at the natural-coordinate centre.  The
two non-affine displacement modes left by one-point integration are controlled
with a geometry-aware, positive-semidefinite projector stabilization:

``K = K_one_point + alpha * H (H.T K_shear H) H.T``.

Here the columns of ``H`` span the orthogonal complement of all affine nodal
displacement fields.  Consequently the stabilization is exactly zero for
translations, rotation and every constant-strain field.  ``K_shear`` is a
2x2-Gauss reference stiffness formed with shear modulus only, which gives the
stabilization the correct physical units without adding bulk locking.

This is a compact affine-projector Q4R formulation; it is not intended to
reproduce a vendor-specific hourglass algorithm coefficient-for-coefficient.

⚠️ 措辞澄清: 以下限制是**本实现所用 compact
hourglass 稳定公式的理论/数值特性**, 不是编码错误, 也不能推广为
"所有 Q4R 单元都必然如此"。商业软件采用更复杂的沙漏控制 (增强
假设应变/自适应缩放) 后适用范围可能更宽。Q4 (全积分) 与 Q4I
(非协调) 完全不受影响。

Known limitation (measured 2026-08): the stabilization scale
``H.T K_shear H`` (full 2x2-Gauss shear stiffness) does not decay with
element aspect ratio, while the one-point bending stiffness decays as
``(h/L)^3``. On thin plates / few-row meshes the balance drifts:

* aspect ratio >= 50  or single-row meshes: stabilization dominates ->
  solution strongly over-stiff (e.g. 2% of analytic cantilever tip);
* medium ratios (L/h ~ 10, few rows): hourglass modes dominate ->
  over-soft (e.g. ~5x analytic tip) while hourglass energy reports
  >90% in both cases, so the ratio is NOT a reliability indicator.

Conventional multi-row meshes with element aspect ratio < 10 verify
correctly (patch test, cantilever ~1.0, SCF, convergence rates). For
thin/slender models prefer Q4I (CPS4I), which has no stabilization term.
"""
import warnings

import numpy as np

from .. import material
from ..checks import require_finite_positive
from .base import register_element
from .q4 import (
    GAUSS_POINTS,
    Q4Element,
    _batch_kinematics,
)

HOURGLASS_COEFFICIENT = 0.10


def _affine_complement(coords):
    """Return an orthonormal basis for non-affine nodal displacements.

    ``coords`` is shaped ``(ne, 4, 2)`` and the result ``(ne, 8, 2)``.
    Centring/scaling changes conditioning only; it does not change the affine
    subspace.
    """
    ne = coords.shape[0]
    shifted = coords - coords.mean(axis=1, keepdims=True)
    scale = np.max(np.linalg.norm(shifted, axis=2), axis=1)
    scale = np.maximum(scale, np.finfo(float).tiny)
    xy = shifted / scale[:, None, None]

    def triangle_twice_area(a, b, c):
        ab = b - a
        ac = c - a
        return ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]

    # Cofactors of [1, x_i, y_i]^T give its one-dimensional nullspace.
    # The two displacement hourglass modes are that scalar vector in the
    # x- and y-DOFs respectively. This is exactly the same affine complement
    # as a complete 8x6 QR, without one LAPACK factorization per element.
    h = np.column_stack([
        triangle_twice_area(xy[:, 1], xy[:, 2], xy[:, 3]),
        -triangle_twice_area(xy[:, 0], xy[:, 2], xy[:, 3]),
        triangle_twice_area(xy[:, 0], xy[:, 1], xy[:, 3]),
        -triangle_twice_area(xy[:, 0], xy[:, 1], xy[:, 2]),
    ])
    norm = np.linalg.norm(h, axis=1)
    if np.any(norm <= 100.0 * np.finfo(float).eps):
        raise ValueError(
            "Q4R affine projector is singular; check element geometry")
    h /= norm[:, None]
    bases = np.zeros((ne, 8, 2), dtype=float)
    bases[:, 0::2, 0] = h
    bases[:, 1::2, 1] = h
    return bases


def validate_hourglass_coefficient(value):
    """Q4R 沙漏系数必须为有限正数 — 负系数会生成负刚度/负稳定能.

    统一标量校验 (checks.require_finite_positive): 字符串/容器/None
    带参数名 TypeError, NaN/Inf/0/负值 ValueError; 标量与批量入口
    (element_stiffness / stiffness_batch) 共用此校验.
    """
    return require_finite_positive(value, "hourglass_coefficient")


def element_stiffness(coords, E, nu, t=1.0, plane="stress",
                      hourglass_coefficient=HOURGLASS_COEFFICIENT):
    """Return one stabilized one-point Q4 stiffness matrix."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (4, 2):
        raise ValueError("Q4R coordinates must have shape (4, 2)")
    validate_hourglass_coefficient(hourglass_coefficient)
    _, B0_all, det0_all = _batch_kinematics(
        coords[None, :, :], np.array([[0.0, 0.0]]))
    _, B_gp_all, det_gp_all = _batch_kinematics(
        coords[None, :, :], GAUSS_POINTS)
    det0 = float(det0_all[0, 0])
    det_gp = det_gp_all[0]
    if det0 <= 0.0 or np.any(det_gp <= 0.0):
        raise ValueError("Q4R has a non-positive Jacobian determinant")

    B0 = B0_all[0, 0]
    D = material.D_matrix(E, nu, plane)
    w0 = 2.0 * np.sqrt(t) * np.sqrt(det0)
    B0w = w0 * B0
    K_reduced = B0w.T @ D @ B0w

    mu = E / (2.0 * (1.0 + nu))
    D_shear = np.diag([2.0 * mu, 2.0 * mu, mu])
    K_shear = np.zeros((8, 8))
    for q in range(4):
        B = B_gp_all[0, q]
        w = np.sqrt(t) * np.sqrt(det_gp[q])
        Bw = w * B
        K_shear += Bw.T @ D_shear @ Bw

    H = _affine_complement(coords[None, :, :])[0]
    reduced_hg = H.T @ K_shear @ H
    K_hourglass = hourglass_coefficient * H @ reduced_hg @ H.T
    K = K_reduced + K_hourglass
    return 0.5 * (K + K.T)


class Q4RElement(Q4Element):
    """One-point Q4 with affine-projector hourglass stabilization."""

    name = "Q4R"
    aliases = ("CPS4R", "CPE4R")
    recovery_family = "q4"
    hourglass_coefficient = HOURGLASS_COEFFICIENT

    def response_at_quadrature(self, mesh, u_e):
        # 本构仅在中心一点积分 — 2×2 高斯点 (q4_B_gauss) 只用于构造沙漏
        # 稳定刚度, 那里的应变包含被稳定项"罚"住但不进入应变能的沙漏
        # 模式。若沿用 Q4 的 2×2 点输出, 应力会混入 O(30%) 量级的虚假
        # 单元内振荡, 污染 SPR/Z2 恢复与云图。Abaqus CPS4R 同样只报
        # 质心单点值。
        D = material.D_matrix(mesh.E, mesh.nu, mesh.plane_type)
        strain = np.einsum("eij,ej->ei", mesh.q4r_B_center, u_e)
        stress = np.einsum("ab,eb->ea", D, strain)
        # dA 含单点高斯权重 w = 2×2 = 4: A_e = ∫∫detJ dξdη ≈ 4·detJ₀。
        # 只返回 detJ₀ 会让 ΣdA 只有单元面积的 1/4, 破坏
        # dA_qp.sum() == areas 契约并压低 Z2 的能量绝对值。
        return (
            stress[:, None, :],
            strain[:, None, :],
            4.0 * mesh.q4r_detJ_center[:, None],
        )

    def recovery_shape_matrix(self, mesh):
        # 中心自然坐标 (0,0) 的 Q4 形函数: 四值均为 0.25
        del mesh  # 协议签名, 常量形函数不依赖几何
        return np.full((1, 4), 0.25)

    def recovery_weights(self, mesh):
        # 单点高斯权重 4 (见 response_at_quadrature)
        return 4.0 * mesh.q4r_detJ_center[:, None]

    def build_geometry(self, nodes, elements):
        geometry = super().build_geometry(nodes, elements)
        coords = nodes[elements]
        _, B0, det0 = _batch_kinematics(
            coords, np.array([[0.0, 0.0]]))
        geometry.update({
            "q4r_B_center": B0[:, 0],
            "q4r_detJ_center": det0[:, 0],
            "q4r_hourglass_basis": _affine_complement(coords),
            "q4r_reduced_hourglass": None,
        })
        return geometry

    def stiffness_batch(self, mesh, element_slice=None):
        validate_hourglass_coefficient(self.hourglass_coefficient)
        selector = slice(None) if element_slice is None else element_slice
        D = material.D_matrix(mesh.E, mesh.nu, mesh.plane_type)
        B0 = mesh.q4r_B_center[selector]
        det0 = mesh.q4r_detJ_center[selector]
        w0 = (2.0 * np.sqrt(mesh.thickness) * np.sqrt(det0))[:, None, None]
        B0w = B0 * w0
        DB0 = np.einsum("ab,ebj->eaj", D, B0w)
        K_reduced = np.einsum("eai,eaj->eij", B0w, DB0)

        mu = mesh.E / (2.0 * (1.0 + mesh.nu))
        D_shear = np.diag([2.0 * mu, 2.0 * mu, mu])
        K_shear = np.zeros_like(K_reduced)
        B_gauss = mesh.q4_B_gauss[selector]
        det_gauss = mesh.q4_detJ_gauss[selector]
        for q in range(4):
            B = B_gauss[:, q]
            w = (np.sqrt(mesh.thickness) * np.sqrt(det_gauss[:, q])
                 )[:, None, None]
            Bw = B * w
            DB = np.einsum("ab,ebj->eaj", D_shear, Bw)
            K_shear += np.einsum("eai,eaj->eij", Bw, DB)

        H = mesh.q4r_hourglass_basis[selector]
        reduced_hg = np.einsum(
            "eia,eij,ejb->eab", H, K_shear, H)
        reduced_hg = 0.5 * (
            reduced_hg + np.transpose(reduced_hg, (0, 2, 1)))
        if mesh.q4r_reduced_hourglass is None:
            mesh.q4r_reduced_hourglass = np.full(
                (mesh.n_elements, 2, 2), np.nan, dtype=float)
        mesh.q4r_reduced_hourglass[selector] = reduced_hg
        # reduced_hg 依赖 D(E,ν,plane) — 记录材料指纹, hourglass_energy
        # 据此检测装配后改材料/平面态导致的静默过期 (几何缓存不随材料
        # 失效)。plane_type 缺失会导致 stress→strain 切换后沙漏能沿用
        # 旧 D 矩阵 (积分点应力相对差 ~9%)。
        mesh._q4r_hourglass_material = (
            mesh.E, mesh.nu, mesh.thickness, mesh.plane_type)
        K_hourglass = self.hourglass_coefficient * np.einsum(
            "eia,eab,ejb->eij", H, reduced_hg, H)
        K_hourglass = 0.5 * (
            K_hourglass + np.transpose(K_hourglass, (0, 2, 1)))
        K = K_reduced + K_hourglass
        return 0.5 * (K + np.transpose(K, (0, 2, 1)))

    def stiffness(self, mesh, eid):
        return element_stiffness(
            mesh.nodes[mesh.elements[eid]], mesh.E, mesh.nu,
            mesh.thickness, mesh.plane_type,
            self.hourglass_coefficient)

    def hourglass_energy(self, mesh, u_e):
        """Return non-negative stabilization energy for each element."""
        if (
                mesh.q4r_reduced_hourglass is None
                or not np.all(np.isfinite(
                    mesh.q4r_reduced_hourglass))
                or getattr(mesh, "_q4r_hourglass_material", None)
                != (mesh.E, mesh.nu, mesh.thickness, mesh.plane_type)):
            # 材料指纹不匹配: 装配后改过 E/ν/t — 旧缓存静默过期,
            # 必须按当前材料重算沙漏稳定项。
            self.stiffness_batch(mesh)
        generalized = np.einsum(
            "eia,ei->ea", mesh.q4r_hourglass_basis, u_e)
        energy = 0.5 * np.einsum(
            "ea,eab,eb->e",
            generalized,
            self.hourglass_coefficient * mesh.q4r_reduced_hourglass,
            generalized,
        )
        # K_hourglass is positive semidefinite by construction.  Remove only
        # signed round-off at its exact affine nullspace (e.g. -1e-29 J).
        # 显著负能量 = 负系数或实现错误 — 无条件截零会掩盖问题。
        neg_mask = energy < -1e-12 * max(
            float(np.max(np.abs(energy))), 1.0)
        if np.any(neg_mask):
            warnings.warn(
                f"Q4R hourglass energy 显著为负 "
                f"(min={float(np.min(energy)):.3e} J) — 沙漏系数必须为"
                "正或存在实现错误", RuntimeWarning)
        return np.maximum(energy, 0.0)

    def verify_mesh(self, mesh, verbose=True):
        log = print if verbose else (lambda *args, **kwargs: None)
        report = self.jacobian_report(mesh)
        if not report.ok:
            log(f"  [FAIL] Q4R Jacobian check: {len(report.bad)} bad elements")
            return False

        Ke = self.stiffness_batch(mesh)
        eig = np.linalg.eigvalsh(Ke)
        scale = np.maximum(
            np.max(np.abs(eig), axis=1), np.finfo(float).tiny)
        n_zero = np.sum(np.abs(eig) <= 1e-9 * scale[:, None], axis=1)
        negative = np.any(eig < -1e-10 * scale[:, None], axis=1)

        H = mesh.q4r_hourglass_basis
        max_affine_leak = 0.0
        coords_all = mesh.nodes[mesh.elements]
        for eid, coords in enumerate(coords_all):
            x, y = coords[:, 0], coords[:, 1]
            affine_modes = (
                np.tile([1.0, 0.0], 4),
                np.tile([0.0, 1.0], 4),
                np.column_stack([-y, x]).ravel(),
                np.column_stack([x, np.zeros(4)]).ravel(),
                np.column_stack([np.zeros(4), y]).ravel(),
                np.column_stack([0.5 * y, 0.5 * x]).ravel(),
            )
            Kh = self.hourglass_coefficient * (
                H[eid] @ mesh.q4r_reduced_hourglass[eid] @ H[eid].T)
            kh_scale = max(
                float(np.linalg.norm(Kh)), np.finfo(float).tiny)
            for mode in affine_modes:
                # 与 Q4/Q4I 同约定: 连模态范数一起除 (||Kh||·||mode||),
                # 彻底无量纲 — 旋转/剪切模态幅值 ~ 坐标量级, 只除 ||Kh||
                # 会让泄漏量带长度量纲 (1e6 单元尺寸时误报 FAIL)
                mode_norm = float(np.linalg.norm(mode))
                denom = kh_scale * mode_norm
                if denom > np.finfo(float).tiny:
                    max_affine_leak = max(
                        max_affine_leak,
                        float(np.linalg.norm(Kh @ mode) / denom))

        ok = (np.all(n_zero == 3) and not np.any(negative)
              and max_affine_leak < 1e-10)
        if ok:
            log(f"  [OK] All {mesh.n_elements} Q4R elements have exactly "
                "3 rigid-body zero modes; affine hourglass leakage "
                f"{max_affine_leak:.2e}")
        else:
            bad_rank = np.flatnonzero(n_zero != 3)
            log(f"  [FAIL] Q4R verification: bad-rank elements="
                f"{bad_rank.tolist()}, negative={bool(np.any(negative))}, "
                f"affine leakage={max_affine_leak:.3e}")
        return ok


Q4R = register_element(Q4RElement())
