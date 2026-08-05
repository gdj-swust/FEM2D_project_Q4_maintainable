r"""Four-node quadrilateral with incompatible modes (QM6).

The bilinear Q4 element is over-stiff in bending because its assumed strain
field cannot represent the linear bending strain of a distorted element; the
error shows up as parasitic shear ("shear locking") and makes bending-dominated
models converge slowly.  Wilson's incompatible-mode quadrilateral adds two
internal displacement bubbles

.. math::

    N_5 = 1 - \\xi^2, \\qquad N_6 = 1 - \\eta^2

whose degrees of freedom are eliminated by static condensation, so the global
system keeps exactly the same 8 nodal degrees of freedom, node ordering and
edge topology as ``Q4``.

The plain Wilson element (Q6) fails the constant-stress patch test on distorted
meshes.  Two corrections are applied here, together known as QM6
(Taylor-Beresford-Wilson 1976) and equivalent to the Simo-Rifai enhanced
assumed strain condition:

1. the incompatible-mode gradients use the Jacobian at the element centre
   :math:`J_0` rather than at the integration point;
2. the resulting operator is scaled by :math:`\det J_0 / \det J(\xi,\eta)`.

The scaling is what makes the enhanced operator orthogonal to constant stress,

.. math::

    \int_\Omega B_a \, \mathrm{d}\Omega
    = \det J_0 \int_{-1}^{1}\!\!\int_{-1}^{1} M(\xi,\eta)
      \,\mathrm{d}\xi\,\mathrm{d}\eta = 0 ,

because the enhanced natural modes :math:`M` are odd in :math:`\xi,\eta`.
Without it the bubble amplitudes do not vanish for a linear displacement field
and the element fails the patch test on any non-parallelogram shape.

Because the element is geometrically identical to Q4 (same nodes, same bilinear
map), no change is needed in the boundary subsystem, traction integration,
visualization or point location.  Aliases follow Abaqus ``CPS4I``/``CPE4I``.
"""
import numpy as np

from .. import material
from .base import register_element
from .q4 import (
    GAUSS_POINTS,
    Q4Element,
    _batch_jacobian_determinants,
    _polygon_geometry,
    shape_derivatives,
    shape_values,
)


def incompatible_derivatives(xi, eta):
    """Natural derivatives of the two bubble modes, shaped ``(2, 2)``.

    Rows are ``d/dxi`` and ``d/deta``; columns are the modes
    :math:`1-\\xi^2` and :math:`1-\\eta^2`.
    """
    return np.array([
        [-2.0 * xi, 0.0],
        [0.0, -2.0 * eta],
    ])


def _strain_operator(gradients, n_field):
    """Assemble a plane strain-displacement operator from nodal gradients."""
    B = np.zeros((3, 2 * n_field))
    B[0, 0::2] = gradients[0]
    B[1, 1::2] = gradients[1]
    B[2, 0::2] = gradients[1]
    B[2, 1::2] = gradients[0]
    return B


def element_operators(coords):
    """Return ``(B_gauss, Bi_gauss, detJ_gauss)`` for one element.

    ``B_gauss[q]`` is the compatible 3x8 operator, ``Bi_gauss[q]`` the
    incompatible 3x4 operator built from the centre Jacobian.
    """
    # 以第一个节点为原点居中 — 与批量路径一致 (大坐标原点下绝对坐标
    # 乘加引入 ~ulp(原点) 误差)
    coords = np.asarray(coords, dtype=float)
    coords = coords - coords[:1]
    jacobian_centre = shape_derivatives(0.0, 0.0) @ coords
    if abs(np.linalg.det(jacobian_centre)) <= np.finfo(float).tiny:
        raise ValueError("Q4I centre Jacobian is singular")

    det_centre = float(np.linalg.det(jacobian_centre))
    n_gauss = len(GAUSS_POINTS)
    B_gauss = np.empty((n_gauss, 3, 8))
    Bi_gauss = np.empty((n_gauss, 3, 4))
    det_gauss = np.empty(n_gauss)
    for q, (xi, eta) in enumerate(GAUSS_POINTS):
        dN_nat = shape_derivatives(xi, eta)
        jacobian = dN_nat @ coords
        det = float(np.linalg.det(jacobian))
        if abs(det) <= np.finfo(float).tiny:
            raise ValueError("Q4I Jacobian is singular")
        det_gauss[q] = det
        B_gauss[q] = _strain_operator(
            np.linalg.solve(jacobian, dN_nat), 4)
        Bi_gauss[q] = (det_centre / det) * _strain_operator(
            np.linalg.solve(
                jacobian_centre, incompatible_derivatives(xi, eta)), 2)
    return B_gauss, Bi_gauss, det_gauss


def _batch_operators(coords):
    """Vectorized ``element_operators`` over all elements."""
    ne = coords.shape[0]
    n_gauss = len(GAUSS_POINTS)
    local = coords - coords[:, :1, :]
    dN_centre = shape_derivatives(0.0, 0.0)
    jac_centre = np.einsum("ai,eib->eab", dN_centre, local)
    det_centre = (jac_centre[:, 0, 0] * jac_centre[:, 1, 1]
                  - jac_centre[:, 0, 1] * jac_centre[:, 1, 0])

    # 退化检测: 中心 Jacobian 奇异时缩聚无解, 下方 np.linalg.solve 会
    # 裸抛 LinAlgError 且无单元编号 — 先定位并给出网格诊断
    h_elem2 = np.max(np.ptp(coords, axis=1), axis=1) ** 2
    bad_mask = np.abs(det_centre) <= 1e-12 * np.maximum(
        h_elem2, np.finfo(float).tiny)
    if np.any(bad_mask):
        first = int(np.flatnonzero(bad_mask)[0])
        raise ValueError(
            f"Q4I element {first} is degenerate (center Jacobian "
            f"det = {det_centre[first]:.3e}, 单元特征尺寸² = "
            f"{h_elem2[first]:.3e}) — 修复网格后重新求解")

    B_all = np.zeros((ne, n_gauss, 3, 8))
    Bi_all = np.zeros((ne, n_gauss, 3, 4))
    det_all = np.empty((ne, n_gauss))
    for q, (xi, eta) in enumerate(GAUSS_POINTS):
        dN_nat = shape_derivatives(xi, eta)
        jac = np.einsum("ai,eib->eab", dN_nat, local)
        det_all[:, q] = (jac[:, 0, 0] * jac[:, 1, 1]
                         - jac[:, 0, 1] * jac[:, 1, 0])
        # 逐 Gauss 点退化检查: 折叠单元可能中心 Jacobian 正常而某个
        # Gauss 点奇异 — 不检查会裸抛 LinAlgError (中心检查只覆盖 det_centre)
        bad_q = np.abs(det_all[:, q]) <= 1e-12 * np.maximum(
            h_elem2, np.finfo(float).tiny)
        if np.any(bad_q):
            first = int(np.flatnonzero(bad_q)[0])
            raise ValueError(
                f"Q4I element {first} quadrature point {q} is degenerate "
                f"(detJ = {det_all[first, q]:.3e}) — 修复网格后重新求解")
        gradients = np.linalg.solve(
            jac, np.broadcast_to(dN_nat, (ne, 2, 4)))
        B_all[:, q, 0, 0::2] = gradients[:, 0]
        B_all[:, q, 1, 1::2] = gradients[:, 1]
        B_all[:, q, 2, 0::2] = gradients[:, 1]
        B_all[:, q, 2, 1::2] = gradients[:, 0]

        dNi_nat = incompatible_derivatives(xi, eta)
        gradients_i = np.linalg.solve(
            jac_centre, np.broadcast_to(dNi_nat, (ne, 2, 2)))
        # detJ0/detJ scaling enforces the enhanced-strain orthogonality
        # condition that the patch test relies on.
        gradients_i = gradients_i * (det_centre / det_all[:, q])[:, None, None]
        Bi_all[:, q, 0, 0::2] = gradients_i[:, 0]
        Bi_all[:, q, 1, 1::2] = gradients_i[:, 1]
        Bi_all[:, q, 2, 0::2] = gradients_i[:, 1]
        Bi_all[:, q, 2, 1::2] = gradients_i[:, 0]
    return B_all, Bi_all, det_all


def _condensation_blocks(mesh, selector):
    """Return ``(K_uu, K_ua, K_aa)`` for the selected elements."""
    B = mesh.q4i_B_gauss[selector]
    Bi = mesh.q4i_Bi_gauss[selector]
    det = mesh.q4i_detJ_gauss[selector]
    D = material.D_matrix(mesh.E, mesh.nu, mesh.plane_type)

    n = len(B)
    K_uu = np.zeros((n, 8, 8))
    K_ua = np.zeros((n, 8, 4))
    K_aa = np.zeros((n, 4, 4))
    for q in range(B.shape[1]):
        Bq, Biq = B[:, q], Bi[:, q]
        # 先加权再二次型 (与 q4 同策略): B̃=√(t·detJ)·B — 微尺度几何下
        # BᵀDB ~ E/L² 中间量会溢出 Inf
        if np.any(det[:, q] <= 0.0):
            # sqrt(max(...,0)) 会把负 Jacobian 静默夹成零 (审查发现)
            raise ValueError(
                "Q4I non-positive Jacobian det(J) at Gauss point "
                f"q={q} (min {float(det[:, q].min()):.3e})")
        w = np.sqrt(mesh.thickness) * np.sqrt(det[:, q])
        Bq = Bq * w[:, None, None]
        Biq = Biq * w[:, None, None]
        DB = np.einsum("ab,ebj->eaj", D, Bq)
        DBi = np.einsum("ab,ebj->eaj", D, Biq)
        K_uu += np.einsum("eai,eaj->eij", Bq, DB)
        K_ua += np.einsum("eai,eaj->eij", Bq, DBi)
        K_aa += np.einsum("eai,eaj->eij", Biq, DBi)
    return K_uu, K_ua, K_aa


def element_stiffness(coords, E, nu, t=1.0, plane="stress"):
    """Condensed QM6 stiffness matrix for one element."""
    D = material.D_matrix(E, nu, plane)
    B_gauss, Bi_gauss, det_gauss = element_operators(coords)
    K_uu = np.zeros((8, 8))
    K_ua = np.zeros((8, 4))
    K_aa = np.zeros((4, 4))
    for q, det in enumerate(det_gauss):
        if det <= 0.0:
            raise ValueError(
                f"Q4I has non-positive Jacobian det(J)={det:.3e}")
        B, Bi = B_gauss[q], Bi_gauss[q]
        w = np.sqrt(t) * np.sqrt(det)
        Bw, Biw = w * B, w * Bi
        K_uu += Bw.T @ D @ Bw
        K_ua += Bw.T @ D @ Biw
        K_aa += Biw.T @ D @ Biw
    Ke = K_uu - K_ua @ np.linalg.solve(K_aa, K_ua.T)
    return 0.5 * (Ke + Ke.T)


class Q4IElement(Q4Element):
    """Incompatible-mode (QM6) bilinear quadrilateral kernel.

    Geometry, loads, point location and recovery are inherited from
    :class:`Q4Element` via ``cache_prefix``; only the stiffness
    condensation and enhanced-strain response differ.
    """

    name = "Q4I"
    aliases = ("CPS4I", "CPE4I")
    cache_prefix = "q4i"

    def build_geometry(self, nodes, elements):
        coords = nodes[elements]
        signed_areas, centroids = _polygon_geometry(coords)
        B_gauss, Bi_gauss, det_gauss = _batch_operators(coords)
        N_gauss = np.array(
            [shape_values(xi, eta) for xi, eta in GAUSS_POINTS])
        extra_check_points = np.array([
            [-1.0, -1.0], [1.0, -1.0],
            [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
        ])
        det_check = np.column_stack([
            det_gauss,
            _batch_jacobian_determinants(coords, extra_check_points),
        ])
        return {
            "signed_areas": signed_areas,
            "areas": np.abs(signed_areas),
            "centroids": centroids,
            "q4i_N_gauss": N_gauss,
            "q4i_B_gauss": B_gauss,
            "q4i_Bi_gauss": Bi_gauss,
            "q4i_detJ_gauss": det_gauss,
            "q4i_detJ_check": det_check,
        }

    def stiffness_batch(self, mesh, element_slice=None):
        selector = slice(None) if element_slice is None else element_slice
        K_uu, K_ua, K_aa = _condensation_blocks(mesh, selector)
        Ke = K_uu - np.einsum(
            "eij,ejk->eik", K_ua,
            np.linalg.solve(K_aa, np.transpose(K_ua, (0, 2, 1))))
        return 0.5 * (Ke + np.transpose(Ke, (0, 2, 1)))

    def stiffness(self, mesh, eid):
        return element_stiffness(
            mesh.nodes[mesh.elements[eid]], mesh.E, mesh.nu,
            mesh.thickness, mesh.plane_type)

    def _cached_enhancement(self, mesh):
        """缓存缩聚增强块 (K_ua, K_aa) — 材料指纹失效, 同 Q4R 沙漏模式.

        response_at_quadrature 每次都调 enhanced_amplitudes; 无缓存时
        每次应力恢复都把全部单元的 4 点缩聚重做一遍。
        """
        cached = getattr(mesh, "_q4i_enhancement", None)
        material = getattr(mesh, "_q4i_enhancement_material", None)
        # 指纹含 plane_type: D(E,ν,plane) 随平面态变化, 缺失会导致
        # stress→strain 切换后缩聚块沿用旧 D (积分点应力相对差 ~3-10%)。
        if cached is None or material != (
                mesh.E, mesh.nu, mesh.thickness, mesh.plane_type):
            _, K_ua, K_aa = _condensation_blocks(mesh, slice(None))
            mesh._q4i_enhancement = (K_ua, K_aa)
            mesh._q4i_enhancement_material = (
                mesh.E, mesh.nu, mesh.thickness, mesh.plane_type)
        return mesh._q4i_enhancement

    def enhanced_amplitudes(self, mesh, u_e):
        """Recover the condensed bubble amplitudes ``alpha`` per element."""
        K_ua, K_aa = self._cached_enhancement(mesh)
        rhs = -np.einsum("eij,ei->ej", K_ua, np.asarray(u_e, dtype=float))
        return np.linalg.solve(K_aa, rhs[:, :, None])[:, :, 0]

    def response_at_quadrature(self, mesh, u_e):
        u_e = np.asarray(u_e, dtype=float)
        D = material.D_matrix(mesh.E, mesh.nu, mesh.plane_type)
        alpha = self.enhanced_amplitudes(mesh, u_e)
        strain = (np.einsum("eqij,ej->eqi", mesh.q4i_B_gauss, u_e)
                  + np.einsum("eqij,ej->eqi", mesh.q4i_Bi_gauss, alpha))
        stress = np.einsum("ab,eqb->eqa", D, strain)
        return stress, strain, mesh.q4i_detJ_gauss.copy()

    def verify_mesh(self, mesh, verbose=True):
        log = print if verbose else (lambda *args, **kwargs: None)
        report = self.jacobian_report(mesh)
        if not report.ok:
            log(f"  [FAIL] Q4I Jacobian check: {len(report.bad)} bad elements")
            return False

        max_partition = float(np.max(np.abs(
            np.sum(mesh.q4i_N_gauss, axis=1) - 1.0)))

        # Rigid-body test on the condensed element: a rigid nodal field must
        # produce zero enhanced amplitude and zero strain.
        coords = mesh.nodes[mesh.elements]
        centre = coords.mean(axis=1)
        modes = np.zeros((mesh.n_elements, 3, 8))
        modes[:, 0, 0::2] = 1.0
        modes[:, 1, 1::2] = 1.0
        modes[:, 2, 0::2] = -(coords[:, :, 1] - centre[:, 1, None])
        modes[:, 2, 1::2] = coords[:, :, 0] - centre[:, 0, None]

        max_rb = 0.0
        # 与 Q4 同约定: 应变算子范数 (B ~ 1/h) × 模态范数 (旋转 ~ h),
        # 彻底无量纲, 与单元尺寸/坐标偏移均无关
        b_scale = max(
            float(np.max(np.abs(mesh.q4i_B_gauss))),
            np.finfo(float).tiny)
        for mode in range(3):
            u_e = modes[:, mode, :]
            _, strain, _ = self.compute_response(mesh, u_e)
            mode_norm = max(
                float(np.max(np.abs(u_e))), np.finfo(float).tiny)
            max_rb = max(
                max_rb,
                float(np.max(np.abs(strain))) / (b_scale * mode_norm))

        ok = max_partition < 1e-12 and max_rb < 1e-10
        if ok:
            log(f"  [OK] All {mesh.n_elements} Q4I elements pass "
                "partition-of-unity + rigid-body checks")
        else:
            log(f"  [FAIL] Q4I verification: partition={max_partition:.3e}, "
                f"rigid-body={max_rb:.3e}")
        return ok


Q4I = register_element(Q4IElement())
