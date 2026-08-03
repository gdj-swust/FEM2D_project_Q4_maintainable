"""Four-node bilinear isoparametric quadrilateral (Q4).

The node order follows Abaqus CPS4/CPE4 convention around the boundary:
``1 -> 2 -> 3 -> 4`` counter-clockwise. Stiffness, body force and raw stress
use full 2x2 Gauss integration.
"""
import numpy as np

from .. import material
from .base import (
    ElementKernel,
    evaluate_vector_field,
    register_element,
)

_G = 1.0 / np.sqrt(3.0)
GAUSS_POINTS = np.array([
    [-_G, -_G],
    [ _G, -_G],
    [ _G,  _G],
    [-_G,  _G],
])


def shape_values(xi, eta):
    """Bilinear shape values ``[N1,N2,N3,N4]``."""
    return 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])


def shape_derivatives(xi, eta):
    """Natural derivatives shaped ``(2,4)``: rows d/dxi and d/deta."""
    return 0.25 * np.array([
        [-(1.0 - eta),  (1.0 - eta),
          (1.0 + eta), -(1.0 + eta)],
        [-(1.0 - xi),  -(1.0 + xi),
          (1.0 + xi),   (1.0 - xi)],
    ])


def B_matrix(coords, xi, eta):
    """Return ``(B, detJ)`` for one Q4 element and natural point."""
    coords = np.asarray(coords, dtype=float)
    dN_nat = shape_derivatives(xi, eta)
    jacobian = dN_nat @ coords
    det_j = float(np.linalg.det(jacobian))
    if abs(det_j) <= np.finfo(float).tiny:
        raise ValueError("Q4 Jacobian is singular")
    gradients = np.linalg.solve(jacobian, dN_nat)
    B = np.zeros((3, 8))
    B[0, 0::2] = gradients[0]
    B[1, 1::2] = gradients[1]
    B[2, 0::2] = gradients[1]
    B[2, 1::2] = gradients[0]
    return B, det_j


def element_stiffness(coords, E, nu, t=1.0, plane="stress"):
    """Full-integration Q4 stiffness matrix."""
    # 以第一个节点为原点居中 — 与 _batch_kinematics 一致 (绝对坐标的
    # 浮点乘加在大坐标原点下引入 ~ulp(原点) 误差, 1e12 偏移实测刚度
    # 偏差 ~1e-4, 审计 2026-08-03)
    coords = np.asarray(coords, dtype=float)
    coords = coords - coords[:1]
    D = material.D_matrix(E, nu, plane)
    Ke = np.zeros((8, 8))
    for xi, eta in GAUSS_POINTS:
        B, det_j = B_matrix(coords, xi, eta)
        if det_j <= 0.0:
            raise ValueError(
                f"Q4 has non-positive Jacobian det(J)={det_j:.3e}")
        w = np.sqrt(t) * np.sqrt(det_j)
        Bw = w * B
        Ke += Bw.T @ D @ Bw
    return 0.5 * (Ke + Ke.T)


def _polygon_geometry(coords):
    """Vectorized signed area and polygon centroid.

    基于局部坐标 (每单元以其第一个节点为原点): 绝对坐标下鞋带公式
    cross = x·y_next − x_next·y 对大坐标原点 + 小局部尺寸灾难性消差
    (审计 2026-08: 1e6 原点 + 0.01 边长 → 面积误差 22%、形心偏差 6 万、
    0.001 边长合法单元被判退化)。
    """
    origin = coords[:, :1, :]           # (ne, 1, 2) — 每单元第一个节点
    local = coords - origin
    x, y = local[:, :, 0], local[:, :, 1]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * np.sum(cross, axis=1)
    centroid = origin[:, 0, :] + local.mean(axis=1)
    valid = np.abs(signed_area) > np.finfo(float).tiny
    if np.any(valid):
        denom = 6.0 * signed_area[valid]
        centroid[valid, 0] = origin[valid, 0, 0] + np.sum(
            (x[valid] + x_next[valid]) * cross[valid], axis=1) / denom
        centroid[valid, 1] = origin[valid, 0, 1] + np.sum(
            (y[valid] + y_next[valid]) * cross[valid], axis=1) / denom
    return signed_area, centroid


def _batch_kinematics(coords, sample_points):
    """Return batched shape values, B matrices and Jacobian determinants.

    Jacobian 基于局部坐标 (每单元以其第一个节点为原点) — 形函数导数
    行和为零, 平移数学上不变, 但绝对坐标的浮点乘加会引入 ~ulp(原点)
    误差, 大坐标原点下刚度相对变化实测 ~9e-5 (审计 2026-08)。
    """
    ne = coords.shape[0]
    nq = len(sample_points)
    N_all = np.zeros((nq, 4))
    B_all = np.zeros((ne, nq, 3, 8))
    det_all = np.zeros((ne, nq))
    local = coords - coords[:, :1, :]
    for q, (xi, eta) in enumerate(sample_points):
        N_all[q] = shape_values(xi, eta)
        dN_nat = shape_derivatives(xi, eta)
        jac = np.einsum("ai,eib->eab", dN_nat, local)
        det = (
            jac[:, 0, 0] * jac[:, 1, 1]
            - jac[:, 0, 1] * jac[:, 1, 0])
        det_all[:, q] = det
        nonsingular = np.abs(det) > np.finfo(float).tiny
        if not np.any(nonsingular):
            continue
        gradients = np.zeros((ne, 2, 4))
        inv_det = 1.0 / det[nonsingular]
        jac_valid = jac[nonsingular]
        gradients[nonsingular, 0] = (
            jac_valid[:, 1, 1, None] * dN_nat[0]
            - jac_valid[:, 0, 1, None] * dN_nat[1]
        ) * inv_det[:, None]
        gradients[nonsingular, 1] = (
            -jac_valid[:, 1, 0, None] * dN_nat[0]
            + jac_valid[:, 0, 0, None] * dN_nat[1]
        ) * inv_det[:, None]
        B_all[:, q, 0, 0::2] = gradients[:, 0]
        B_all[:, q, 1, 1::2] = gradients[:, 1]
        B_all[:, q, 2, 0::2] = gradients[:, 1]
        B_all[:, q, 2, 1::2] = gradients[:, 0]
    return N_all, B_all, det_all


def _batch_jacobian_determinants(coords, sample_points):
    """Return only Jacobian determinants without allocating B matrices."""
    det_all = np.empty((coords.shape[0], len(sample_points)), dtype=float)
    local = coords - coords[:, :1, :]
    for q, (xi, eta) in enumerate(sample_points):
        dN_nat = shape_derivatives(xi, eta)
        jac = np.einsum("ai,eib->eab", dN_nat, local)
        det_all[:, q] = (
            jac[:, 0, 0] * jac[:, 1, 1]
            - jac[:, 0, 1] * jac[:, 1, 0])
    return det_all


class Q4Element(ElementKernel):
    """Full-integration bilinear quadrilateral kernel."""

    name = "Q4"
    aliases = ("CPS4", "CPE4")
    nodes_per_element = 4
    local_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    recovery_family = "q4"
    # 几何缓存前缀: Q4I 继承本类但生成 q4i_* 缓存, 通过该前缀统一访问
    cache_prefix = "q4"

    def _cache(self, mesh, suffix):
        return getattr(mesh, f"{self.cache_prefix}_{suffix}")

    def build_geometry(self, nodes, elements):
        coords = nodes[elements]
        signed_areas, centroids = _polygon_geometry(coords)
        N_gp, B_gp, det_gp = _batch_kinematics(coords, GAUSS_POINTS)
        extra_check_points = np.array([
            [-1.0, -1.0], [1.0, -1.0],
            [1.0, 1.0], [-1.0, 1.0], [0.0, 0.0],
        ])
        det_check = np.column_stack([
            det_gp,
            _batch_jacobian_determinants(coords, extra_check_points),
        ])
        return {
            "signed_areas": signed_areas,
            "areas": np.abs(signed_areas),
            "centroids": centroids,
            "q4_N_gauss": N_gp,
            "q4_B_gauss": B_gp,
            "q4_detJ_gauss": det_gp,
            "q4_detJ_check": det_check,
        }

    def stiffness_batch(self, mesh, element_slice=None):
        selector = slice(None) if element_slice is None else element_slice
        B_gauss = mesh.q4_B_gauss[selector]
        det_gauss = mesh.q4_detJ_gauss[selector]
        D = material.D_matrix(mesh.E, mesh.nu, mesh.plane_type)
        Ke = np.zeros((len(B_gauss), 8, 8))
        for q in range(4):
            B = B_gauss[:, q]
            # 先加权再二次型: B̃ = √(t·detJ)·B, Ke = B̃ᵀDB̃ —
            # 曾先算 BᵀDB ~ E/L² 中间量, 微尺度几何 (L~1e-150) 下溢出
            # Inf 后再乘 detJ 已来不及 (外部审查, 2026-08-03)。数学等价,
            # 中间量尺度 O(√t·D) 不随 L 发散。
            if np.any(det_gauss[:, q] <= 0.0):
                # 曾 sqrt(max(...,0)) 把负 Jacobian 静默夹成零 (第三轮
                # 外部审查) — 显式拒绝, 与标量路径一致
                raise ValueError(
                    "Q4 non-positive Jacobian det(J) at Gauss point "
                    f"q={q} (min {float(det_gauss[:, q].min()):.3e})")
            w = (np.sqrt(mesh.thickness) * np.sqrt(det_gauss[:, q])
                 )[:, None, None]
            Bw = B * w
            # 注意 einsum 下标: "eai,eaj" 共享应变分量 a —
            # "eai,ebj" 会让 a/b 独立求和, 数值全错 (2026-08-03 修复时踩坑)
            Ke += np.einsum("eai,eaj->eij", Bw, np.einsum(
                "ab,ebj->eaj", D, Bw))
        return 0.5 * (Ke + np.transpose(Ke, (0, 2, 1)))

    def stiffness(self, mesh, eid):
        return element_stiffness(
            mesh.nodes[mesh.elements[eid]], mesh.E, mesh.nu,
            mesh.thickness, mesh.plane_type)

    def response_at_quadrature(self, mesh, u_e):
        D = material.D_matrix(mesh.E, mesh.nu, mesh.plane_type)
        strain = np.einsum("eqij,ej->eqi", mesh.q4_B_gauss, u_e)
        stress = np.einsum("ab,eqb->eqa", D, strain)
        return stress, strain, mesh.q4_detJ_gauss.copy()

    def jacobian_determinants(self, mesh):
        return self._cache(mesh, "detJ_check")

    def degeneracy_measure(self, mesh):
        """Q4 形状退化指标: 面积/最长边² — 无量纲.

        矩形退化为长宽比倒数; 扭歪再乘 sinθ; 极细长/扭歪 → 0。
        """
        coords = mesh.nodes[mesh.elements]
        edges = np.roll(coords, -1, axis=1) - coords
        h_max2 = np.max(np.sum(edges * edges, axis=2), axis=1)
        return np.abs(mesh.areas) / np.maximum(
            h_max2, np.finfo(float).tiny)

    def body_force_vector(self, mesh, eid, body_force):
        coords = mesh.nodes[mesh.elements[eid]]
        fe = np.zeros(8)
        for q, N in enumerate(self._cache(mesh, "N_gauss")):
            xg, yg = N @ coords
            bx, by = evaluate_vector_field(body_force, xg, yg)
            if not (np.isfinite(bx) and np.isfinite(by)):
                raise ValueError(
                    "Body force callable returned NaN/Inf at "
                    f"{self.name} Gauss point ({xg:.4g},{yg:.4g}) in element {eid}")
            factor = mesh.thickness * self._cache(mesh, "detJ_gauss")[eid, q]
            fe[0::2] += factor * N * bx
            fe[1::2] += factor * N * by
        return fe

    def body_force_batch(self, mesh, body_force, element_slice=None):
        constant = self._constant_body_force(body_force)
        if constant is None:
            return None
        bx, by = constant
        selector = slice(None) if element_slice is None else element_slice
        det_gauss = self._cache(mesh, "detJ_gauss")[selector]
        nodal_weight = mesh.thickness * np.einsum(
            "qn,eq->en", self._cache(mesh, "N_gauss"), det_gauss)
        fe = np.empty((len(nodal_weight), 8), dtype=float)
        fe[:, 0::2] = nodal_weight * bx
        fe[:, 1::2] = nodal_weight * by
        return fe

    def shape_values_at(self, coords, x, y, tol=1e-12):
        target = np.array([x, y], dtype=float)
        natural = np.zeros(2)
        # 局部单元尺度, 下限取坐标 ULP — 固定 1.0 下限在纳米尺度
        # (边长 1e-12) 下容差 1e-10×1 远超单元本身, 域外点被误判在单元内
        coord_ulp = 64.0 * np.finfo(float).eps * float(
            np.max(np.abs(coords)))
        scale = max(float(np.ptp(coords[:, 0])),
                    float(np.ptp(coords[:, 1])), coord_ulp)
        for _ in range(20):
            xi, eta = natural
            N = shape_values(xi, eta)
            residual = N @ coords - target
            if np.linalg.norm(residual, ord=np.inf) <= tol * scale:
                break
            jac = shape_derivatives(xi, eta) @ coords
            if abs(np.linalg.det(jac)) <= np.finfo(float).tiny:
                return None
            natural -= np.linalg.solve(jac.T, residual)
        xi, eta = natural
        if xi < -1.0 - tol or xi > 1.0 + tol:
            return None
        if eta < -1.0 - tol or eta > 1.0 + tol:
            return None
        N = shape_values(xi, eta)
        if np.linalg.norm(N @ coords - target, ord=np.inf) > 10.0 * tol * scale:
            return None
        return N

    def recovery_quadrature(self, mesh, eid):
        return self._cache(mesh, "N_gauss"), self._cache(
            mesh, "detJ_gauss")[eid].copy()

    def recovery_shape_matrix(self, mesh):
        return self._cache(mesh, "N_gauss")

    def recovery_weights(self, mesh):
        return self._cache(mesh, "detJ_gauss")

    def verify_mesh(self, mesh, verbose=True):
        log = print if verbose else (lambda *args, **kwargs: None)
        report = self.jacobian_report(mesh)
        if not report.ok:
            log(f"  [FAIL] Q4 Jacobian check: {len(report.bad)} bad elements")
            return False

        max_partition = 0.0
        max_rb = 0.0
        for eid, conn in enumerate(mesh.elements):
            coords = mesh.nodes[conn]
            for q, N in enumerate(mesh.q4_N_gauss):
                max_partition = max(max_partition, abs(np.sum(N) - 1.0))
                B = mesh.q4_B_gauss[eid, q]
                center = coords.mean(axis=0)
                modes = (
                    np.tile([1.0, 0.0], 4),
                    np.tile([0.0, 1.0], 4),
                    np.column_stack([
                        -(coords[:, 1] - center[1]),
                        coords[:, 0] - center[0],
                    ]).ravel(),
                )
                for mode in modes:
                    rb = float(np.linalg.norm(B @ mode))
                    # 相对判据: B ~ 1/h 且旋转模态幅值 ~ h, 绝对比较会被
                    # 坐标偏移 (ε·|x0|/h) 放大; 连模态范数一起除 (||B||·||mode||)
                    # 后彻底无量纲, 与单元尺寸/坐标偏移均无关 (与 Q4I 同约定)
                    b_norm = float(np.linalg.norm(B))
                    mode_norm = float(np.linalg.norm(mode))
                    denom = b_norm * mode_norm
                    if denom > np.finfo(float).tiny:
                        max_rb = max(max_rb, rb / denom)
        ok = max_partition < 1e-12 and max_rb < 1e-10
        if ok:
            log(f"  [OK] All {mesh.n_elements} Q4 elements pass "
                "partition-of-unity + rigid-body checks")
        else:
            log(f"  [FAIL] Q4 verification: partition={max_partition:.3e}, "
                f"rigid-body={max_rb:.3e}")
        return ok


Q4 = register_element(Q4Element())
