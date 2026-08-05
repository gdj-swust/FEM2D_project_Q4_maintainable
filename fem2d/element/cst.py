"""CST constant-strain triangle — self-contained kernel (Bathe §5.3.2).

All CST-specific formulas (shape functions, B-matrix, stiffness, verification)
live here. The CSTElement kernel class wraps them into the ElementKernel protocol.
"""

import numpy as np

from ..material import D_matrix, von_mises
from .base import (
    ElementKernel,
    evaluate_vector_field,
    register_element,
)

# ═══════════════════════════════════════════════════════════════
# 1. 形函数 — Bathe §5.3.2 Eq (5.43)
# ═══════════════════════════════════════════════════════════════

def _shape_coeffs(xi, yi, xj, yj, xk, yk):
    """CST 形函数系数 (a, b, c) 和单元面积.

    Bathe §5.3.2 Eq (5.43):
      a_1 = x₂y₃ - x₃y₂,  b_1 = y₂ - y₃,  c_1 = x₃ - x₂  (循环置换)
    """
    a = np.array([xj * yk - xk * yj,
                   xk * yi - xi * yk,
                   xi * yj - xj * yi])
    b = np.array([yj - yk, yk - yi, yi - yj])
    c = np.array([xk - xj, xi - xk, xj - xi])
    area = 0.5 * ((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
    return a, b, c, area


def _shape_values(a, b, c, area, x, y):
    """CST 单元内点 (x,y) 处的三个形函数值.

    Bathe §5.3.2 Eq (5.44): N_i(x,y) = (a_i + b_i*x + c_i*y) / (2A)
    """
    return (a + b * x + c * y) / (2.0 * area)


# ═══════════════════════════════════════════════════════════════
# 2. 应变-位移矩阵 B — Bathe §5.3.2 Example 5.18
# ═══════════════════════════════════════════════════════════════

def _B_matrix(b, c, area):
    """CST 单元的应变-位移矩阵 B (3×6).

    B = (1/2A) · [b₁ 0 b₂ 0 b₃ 0; 0 c₁ 0 c₂ 0 c₃; c₁ b₁ c₂ b₂ c₃ b₃]
    """
    B = np.zeros((3, 6))
    f = 1.0 / (2.0 * area)
    for i in range(3):
        col = 2 * i
        B[0, col] = b[i] * f
        B[1, col + 1] = c[i] * f
        B[2, col] = c[i] * f
        B[2, col + 1] = b[i] * f
    return B


def _batch_B_matrix(b_coeffs, c_coeffs, signed_areas):
    """批量构造所有 CST 单元的 B 矩阵 (Bathe §5.3.2 Eq 5.43).

    Returns (n_elem, 3, 6) ndarray.
    """
    ne = len(signed_areas)
    inv_2A = 0.5 / signed_areas
    B = np.zeros((ne, 3, 6))
    B[:, 0, 0] = b_coeffs[:, 0] * inv_2A
    B[:, 1, 1] = c_coeffs[:, 0] * inv_2A
    B[:, 2, 0] = c_coeffs[:, 0] * inv_2A
    B[:, 2, 1] = b_coeffs[:, 0] * inv_2A
    B[:, 0, 2] = b_coeffs[:, 1] * inv_2A
    B[:, 1, 3] = c_coeffs[:, 1] * inv_2A
    B[:, 2, 2] = c_coeffs[:, 1] * inv_2A
    B[:, 2, 3] = b_coeffs[:, 1] * inv_2A
    B[:, 0, 4] = b_coeffs[:, 2] * inv_2A
    B[:, 1, 5] = c_coeffs[:, 2] * inv_2A
    B[:, 2, 4] = c_coeffs[:, 2] * inv_2A
    B[:, 2, 5] = b_coeffs[:, 2] * inv_2A
    return B


# ═══════════════════════════════════════════════════════════════
# 3. 单元刚度矩阵 — Bathe §5.3.1 Eq (5.27)
# ═══════════════════════════════════════════════════════════════

def _element_stiffness(xi, yi, xj, yj, xk, yk, E, nu, t=1.0, plane="stress"):
    """CST 三角形单元刚度矩阵 K_e (6×6).

    Bathe Eq (5.27): K_e = t·A·Bᵀ D B  (1 点积分精确).
    """
    _, b, c, area = _shape_coeffs(xi, yi, xj, yj, xk, yk)
    # 面积判据基于单元自身尺寸 (最大边长), 不是坐标绝对值 —
    # 绝对 1e-30 会拒纳米单元; 改用 max|x_i,y_i| 又误拒远离原点的
    # 合法单元 (1e7 偏移的单位三角形, 面积 0.5 < 1.42 判据) — 批量路径
    # 用坐标差构造, 天然与原点无关
    scl = max(np.hypot(xj - xi, yj - yi), np.hypot(xk - xj, yk - yj),
              np.hypot(xi - xk, yi - yk), np.finfo(float).tiny)
    if abs(area) <= 64.0 * np.finfo(float).eps * scl * scl:
        raise ValueError(f"Element area {area:.3e} ≈ 0 — degenerate triangle")
    B = _B_matrix(b, c, area)
    D = D_matrix(E, nu, plane)
    w = np.sqrt(t) * np.sqrt(abs(area))
    Bw = w * B
    K_e = Bw.T @ D @ Bw
    # 对称性检查 + 浮点舍入修正
    skew = K_e - K_e.T
    scale = max(np.linalg.norm(K_e, ord=np.inf), 1.0)
    if np.linalg.norm(skew, ord=np.inf) / scale > 1e-12:
        raise RuntimeError("CST element stiffness matrix is asymmetric")
    return 0.5 * (K_e + K_e.T)


# ═══════════════════════════════════════════════════════════════
# 4. 单元验证 — Bathe §5.3.3 完备性条件 + 刚体模态
# ═══════════════════════════════════════════════════════════════

def _check_completeness(xi, yi, xj, yj, xk, yk, tol=1e-12):
    """验证 CST 单元的完备性条件: ΣN_i(x,y) = 1."""
    x0, y0 = xi, yi
    a, b, c, area = _shape_coeffs(0.0, 0.0, xj - x0, yj - y0, xk - x0, yk - y0)
    test_points = [
        ((xj - x0 + xk - x0) / 3.0, (yj - y0 + yk - y0) / 3.0),
        (0.0, 0.0),
        (xj - x0, yj - y0),
        (xk - x0, yk - y0),
    ]
    max_err = 0.0
    for lx, ly in test_points:
        N = _shape_values(a, b, c, area, lx, ly)
        max_err = max(max_err, abs(np.sum(N) - 1.0))
    return max_err < tol, max_err


def _check_rigid_body(B, xi, yi, xj, yj, xk, yk, tol=1e-12):
    """验证 B 矩阵的刚体模态: B·u_rb = 0."""
    xc = (xi + xj + xk) / 3.0
    yc = (yi + yj + yk) / 3.0
    rb_modes = [
        np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        np.array([-(yi - yc), xi - xc, -(yj - yc), xj - xc, -(yk - yc), xk - xc]),
    ]
    max_strain = 0.0
    for u_rb in rb_modes:
        strain_norm = np.linalg.norm(B @ u_rb)
        max_strain = max(max_strain, strain_norm)
    return max_strain < tol, max_strain


def _verify_element(xi, yi, xj, yj, xk, yk, tol=1e-12):
    """完整 CST 单元验证: 完备性 + 刚体模态."""
    _, b, c, area = _shape_coeffs(xi, yi, xj, yj, xk, yk)
    if abs(area) <= np.finfo(float).tiny:
        # 退化 (共线/零面积) 单元 — _B_matrix 的 1/(2A) 抛裸
        # ZeroDivisionError, 验证路径崩溃而非报告失败
        return {
            "completeness": False, "rigid_body": False,
            "completeness_err": float("inf"),
            "rigid_body_err": float("inf"),
            "area": float(area), "all_ok": False,
        }
    B = _B_matrix(b, c, area)
    h_char = max(np.sqrt(2.0 * abs(area)), np.finfo(float).tiny)
    adaptive_tol = max(tol, np.finfo(float).eps * 10.0 / h_char)
    comp_ok, comp_err = _check_completeness(xi, yi, xj, yj, xk, yk, tol)
    rb_ok, rb_err = _check_rigid_body(B, xi, yi, xj, yj, xk, yk, adaptive_tol)
    return {
        "completeness": comp_ok, "rigid_body": rb_ok,
        "completeness_err": comp_err, "rigid_body_err": rb_err,
        "area": area, "all_ok": comp_ok and rb_ok,
    }


def _verify_all_elements(mesh, verbose=True):
    """验证网格中所有 CST 单元的完备性和刚体模态."""
    log = print if verbose else (lambda *a, **k: None)
    log(f"\n[Verify] Bathe §5.3.3 CST 单元完备性检验 ({mesh.n_elements} elements) ...")
    all_ok = True
    for eid, (i, j, k) in enumerate(mesh.elements):
        result = _verify_element(
            float(mesh.nodes[i, 0]), float(mesh.nodes[i, 1]),
            float(mesh.nodes[j, 0]), float(mesh.nodes[j, 1]),
            float(mesh.nodes[k, 0]), float(mesh.nodes[k, 1]))
        if not result["all_ok"]:
            all_ok = False
            log(f"  [FAIL] Element {eid}: completeness={result['completeness']} "
                f"(err={result['completeness_err']:.2e}), "
                f"rigid_body={result['rigid_body']} "
                f"(err={result['rigid_body_err']:.2e})")
    if all_ok:
        log(f"  [OK] All {mesh.n_elements} CST elements pass "
            "completeness + rigid body check")
    return all_ok


# ═══════════════════════════════════════════════════════════════
# 5. CST Kernel — ElementKernel 协议实现
# ═══════════════════════════════════════════════════════════════

class CSTElement(ElementKernel):
    """Vectorized constant-strain triangle kernel."""

    name = "CST"
    aliases = ("CPS3", "CPE3", "C2D3")
    nodes_per_element = 3
    local_edges = ((0, 1), (1, 2), (2, 0))
    recovery_family = "cst"

    # ── geometry ──

    def build_geometry(self, nodes, elements):
        p0 = nodes[elements[:, 0]]
        p1 = nodes[elements[:, 1]]
        p2 = nodes[elements[:, 2]]
        v1 = p1 - p0
        v2 = p2 - p0
        signed_areas = 0.5 * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        return {
            "signed_areas": signed_areas,
            "areas": np.abs(signed_areas),
            "b_coeffs": np.column_stack([
                p1[:, 1] - p2[:, 1],
                p2[:, 1] - p0[:, 1],
                p0[:, 1] - p1[:, 1],
            ]),
            "c_coeffs": np.column_stack([
                p2[:, 0] - p1[:, 0],
                p0[:, 0] - p2[:, 0],
                p1[:, 0] - p0[:, 0],
            ]),
            "centroids": (p0 + p1 + p2) / 3.0,
        }

    # ── stiffness ──

    def stiffness_batch(self, mesh, element_slice=None):
        selector = slice(None) if element_slice is None else element_slice
        D = D_matrix(mesh.E, mesh.nu, mesh.plane_type)
        B = _batch_B_matrix(
            mesh.b_coeffs[selector],
            mesh.c_coeffs[selector],
            mesh.signed_areas[selector],
        )
        w = (np.sqrt(mesh.thickness) * np.sqrt(mesh.areas[selector])
             )[:, None, None]
        Bw = B * w
        Ke = np.transpose(Bw, (0, 2, 1)) @ (D @ Bw)
        return 0.5 * (Ke + np.transpose(Ke, (0, 2, 1)))

    def stiffness(self, mesh, eid):
        coords = mesh.nodes[mesh.elements[eid]].ravel()
        return _element_stiffness(
            *coords, mesh.E, mesh.nu, mesh.thickness, mesh.plane_type)

    # ── response ──

    def compute_response(self, mesh, u_e):
        D = D_matrix(mesh.E, mesh.nu, mesh.plane_type)
        B = _batch_B_matrix(mesh.b_coeffs, mesh.c_coeffs, mesh.signed_areas)
        strain = np.sum(B * u_e[:, None, :], axis=2)
        stress = strain @ D.T
        vm = von_mises(stress, mesh.plane_type, mesh.nu)
        return stress, strain, vm

    def response_at_quadrature(self, mesh, u_e):
        # CST 的超收敛点只有形心一个: 常应力场复制到 3 个 Hammer 点会让
        # SPR 法方程带上同量级的 Δ=Σδδᵀ≻0 (等效岭正则), 恢复的应力梯度
        # 被系统性压低, 线性精确性退化为 O(h)。单点采样 (与 Q4R 同模式)
        # 使 SPR 对线性场精确复现 (Barlow 点语义)。
        stress, strain, _ = self.compute_response(mesh, u_e)
        return (
            stress[:, None, :],
            strain[:, None, :],
            np.asarray(mesh.areas, dtype=float)[:, None],
        )

    # ── Jacobian ──

    def jacobian_determinants(self, mesh):
        return (2.0 * mesh.signed_areas)[:, None]

    def degeneracy_measure(self, mesh):
        """CST 形状退化指标: 2A/h_max² = sin(最小夹角量级) — 无量纲.

        等边三角形 = √3/2 ≈ 0.866; 面积塌缩 (极瘦/共线) → 0。
        """
        p0 = mesh.nodes[mesh.elements[:, 0]]
        p1 = mesh.nodes[mesh.elements[:, 1]]
        p2 = mesh.nodes[mesh.elements[:, 2]]
        l01 = np.linalg.norm(p1 - p0, axis=1)
        l12 = np.linalg.norm(p2 - p1, axis=1)
        l20 = np.linalg.norm(p0 - p2, axis=1)
        h_max2 = np.maximum(np.maximum(l01, l12), l20) ** 2
        return (2.0 * np.abs(mesh.signed_areas)
                / np.maximum(h_max2, np.finfo(float).tiny))

    # ── loads ──

    def body_force_vector(self, mesh, eid, body_force):
        coords = mesh.nodes[mesh.elements[eid]]
        area = mesh.areas[eid]
        gauss = (
            (1.0 / 3.0, (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0)),
            (1.0 / 3.0, (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)),
            (1.0 / 3.0, (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)),
        )
        fe = np.zeros(self.dofs_per_element)
        for weight, shape in gauss:
            N = np.asarray(shape)
            xg, yg = N @ coords
            bx, by = evaluate_vector_field(body_force, xg, yg)
            if not (np.isfinite(bx) and np.isfinite(by)):
                raise ValueError(
                    "Body force callable returned NaN/Inf at "
                    f"Hammer point ({xg:.4g},{yg:.4g}) in element {eid}")
            fe[0::2] += mesh.thickness * weight * area * N * bx
            fe[1::2] += mesh.thickness * weight * area * N * by
        return fe

    def body_force_batch(self, mesh, body_force, element_slice=None):
        constant = self._constant_body_force(body_force)
        if constant is None:
            return None
        bx, by = constant
        selector = slice(None) if element_slice is None else element_slice
        nodal_weight = (
            mesh.thickness * mesh.areas[selector] / 3.0)
        fe = np.empty((len(nodal_weight), 6), dtype=float)
        fe[:, 0::2] = nodal_weight[:, None] * bx
        fe[:, 1::2] = nodal_weight[:, None] * by
        return fe

    # ── point location ──

    def shape_values_at(self, coords, x, y, tol=1e-12):
        p0, p1, p2 = coords
        det_j = ((p1[0] - p0[0]) * (p2[1] - p0[1])
                 - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        if det_j <= 0.0:
            return None
        point = np.array([x, y])
        n1 = ((p1[0] - point[0]) * (p2[1] - point[1])
              - (p2[0] - point[0]) * (p1[1] - point[1])) / det_j
        n2 = ((p2[0] - point[0]) * (p0[1] - point[1])
              - (p0[0] - point[0]) * (p2[1] - point[1])) / det_j
        N = np.array([n1, n2, 1.0 - n1 - n2])
        if np.all(N >= -tol) and np.all(N <= 1.0 + tol):
            return N
        return None

    # ── recovery ──

    def recovery_quadrature(self, mesh, eid):
        # 3 点 Hammer 规则 — 与单点 SPR 采样解耦 (Q4R 同模式):
        # nodal_L2_projection 的一致性质量阵需要满秩积分规则,
        # 单点 N^T N 只有秩 1 会奇异。
        N = np.array([
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ])
        return N, np.full(3, mesh.areas[eid] / 3.0)

    def recovery_shape_matrix(self, mesh):
        # SPR 采样点 = 形心 (CST 唯一的超收敛点, 见 response_at_quadrature)
        del mesh
        return np.full((1, 3), 1.0 / 3.0)

    def recovery_weights(self, mesh):
        return np.asarray(mesh.areas, dtype=float)[:, None]

    # ── verification ──

    def verify_mesh(self, mesh, verbose=True):
        return _verify_all_elements(mesh, verbose=verbose)


# ── 注册 ──

CST = register_element(CSTElement())

