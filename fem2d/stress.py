"""Element response and stress recovery.

Raw element response is delegated to the active element kernel. Recovery
utilities use generic connectivity and kernel-provided integration rules.
"""
import numpy as np
from scipy.sparse import coo_matrix

from .spr import spr_recovery


def compute_stresses(mesh, u):
    """Return representative element stress, strain and von Mises arrays."""
    mesh.build_connectivity()
    u = np.asarray(u, dtype=float)
    if u.shape != (mesh.n_dof,):
        raise ValueError(f"u must have shape ({mesh.n_dof},), got {u.shape}")
    if not np.all(np.isfinite(u)):
        raise ValueError(
            "compute_stresses: u contains NaN/Inf — 位移向量非法")
    return mesh.element_kernel.compute_response(
        mesh, u[mesh.element_dofs])


def principal_stresses(stress):
    """Compute in-plane principal stresses and maximum in-plane shear."""
    stress = np.asarray(stress)
    if stress.ndim != 2 or stress.shape[1] != 3:
        # (n,2)/(n,)/(标量) 曾冒裸 IndexError — 形状契约前置校验
        raise ValueError(
            "principal_stresses: stress 必须为 (n, 3) 数组 "
            f"[σx, σy, τxy], got {stress.shape}")
    if not np.all(np.isfinite(stress)):
        raise ValueError(
            "principal_stresses: stress contains NaN/Inf — "
            "主应力对非法输入静默返回 NaN 曾掩盖上游错误")
    sx, sy, txy = stress[:, 0], stress[:, 1], stress[:, 2]
    # 防溢出: (0.5(sx-sy))² 在 |s|~1e308 时平方 inf; 0.5*(sx+sy) 在
    # sx=sy=1e308 时先加后乘也 inf — 先除以 2 再相加
    average = 0.5 * sx + 0.5 * sy
    half_diff = 0.5 * sx - 0.5 * sy
    radius = np.hypot(half_diff, txy)
    theta = 0.5 * np.arctan2(txy, half_diff)   # = atan2(2txy, sx−sy)
    return average + radius, average - radius, radius, theta


def nodal_average(mesh, elem_stress, weights=None):
    """Recover nodal stress by weighted averaging of element responses.

    ``weights`` 可为:
      * ``None`` — 算术平均 (每节点取所属单元响应均值)
      * ``"area"`` — 按单元面积加权
      * ``(n_elem,)`` ndarray — 任意单元权重

    统一向量化实现 (bincount), 是 nodal_simple/nodal_weighted 的共同内核。
    """
    mesh.build_connectivity()
    elem_stress = np.asarray(elem_stress)
    if elem_stress.ndim != 2 or elem_stress.shape[0] != mesh.n_elements:
        raise ValueError(
            "elem_stress must have shape (n_elem, n_comp), got "
            f"{elem_stress.shape}")
    if not np.all(np.isfinite(elem_stress)):
        raise ValueError(
            "nodal_average: elem_stress contains NaN/Inf — 恢复输入非法")
    n_comp = elem_stress.shape[1]
    if weights is None:
        w = np.ones(mesh.n_elements, dtype=float)
    elif isinstance(weights, str) and weights == "area":
        w = np.asarray(mesh.areas, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != (mesh.n_elements,):
            raise ValueError(
                f"weights must have shape ({mesh.n_elements},), got {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError("nodal_average: weights contain NaN/Inf")

    flat_nodes = mesh.elements.ravel()
    repeated = np.repeat(w, mesh.elements.shape[1])
    denom = np.bincount(flat_nodes, weights=repeated, minlength=mesh.n_nodes)
    orphan = np.flatnonzero(denom == 0.0)
    if len(orphan):
        # 曾 denom==0 → 1.0 静默填 0 应力, 与 L2 路径抛错行为分叉 —
        # 孤立节点在三条恢复路径统一显式报错
        raise ValueError(
            f"节点平均恢复失败: {len(orphan)} 个孤立节点未被任何单元引用, "
            f"加权计数为零 (节点: {orphan[:10].tolist()}) — "
            "请先移除孤立节点")
    recovered = np.empty((mesh.n_nodes, n_comp))
    for component in range(n_comp):
        num = np.bincount(
            flat_nodes,
            weights=repeated * np.repeat(
                elem_stress[:, component], mesh.elements.shape[1]),
            minlength=mesh.n_nodes,
        )
        recovered[:, component] = num / denom
    return recovered


def nodal_simple(mesh, elem_stress):
    """Recover nodal stress by arithmetic averaging.

    等价于 ``nodal_average(mesh, elem_stress)`` — 兼容别名.
    """
    return nodal_average(mesh, elem_stress)


def nodal_weighted(mesh, elem_stress):
    """Recover nodal stress by element-area weighting.

    等价于 ``nodal_average(mesh, elem_stress, weights="area")`` — 兼容别名.
    """
    return nodal_average(mesh, elem_stress, weights="area")


def nodal_L2_projection(mesh, elem_stress):
    """Consistent-mass L2 projection using the active kernel quadrature.

    ``elem_stress`` may be representative values ``(ne,ncomp)`` or
    integration-point values ``(ne,nqp,ncomp)``.

    内核提供批量恢复规则 (recovery_shape_matrix/recovery_weights) 时走
    向量化 einsum + COO scatter (大网格 ~2 个数量级提速); 否则退回
    逐单元 recovery_quadrature 协议 (兼容第三方内核)。
    """
    mesh.build_connectivity()
    elem_stress = np.asarray(elem_stress, dtype=float)
    if elem_stress.ndim not in (2, 3):
        raise ValueError(
            "elem_stress must have shape (ne,ncomp) or (ne,nqp,ncomp)")
    if elem_stress.shape[0] != mesh.n_elements:
        raise ValueError(
            f"elem_stress first dimension must be {mesh.n_elements}, "
            f"got {elem_stress.shape[0]}")
    if not np.all(np.isfinite(elem_stress)):
        raise ValueError(
            "nodal_L2_projection: elem_stress contains NaN/Inf — "
            "恢复输入非法")
    n_nodes = mesh.n_nodes
    n_comp = elem_stress.shape[-1]
    kernel = mesh.element_kernel
    conn = mesh.elements
    ne = conn.shape[0]
    npe = conn.shape[1]

    shape = kernel.recovery_shape_matrix(mesh)
    weights = kernel.recovery_weights(mesh)
    if shape is not None and weights is not None:
        N = np.asarray(shape, dtype=float)           # (nqp, npe)
        dA = np.asarray(weights, dtype=float)        # (ne, nqp)
        nqp = N.shape[0]
        # 恢复采样规则 ≠ 质量阵积分规则的内核 (Q4R/CST: 单点 SPR vs
        # 多点 L2) 不能复用采样规则做质量阵 — 单点 NᵀN 只有秩 1,
        # 一致质量阵会奇异。退回逐单元 recovery_quadrature 协议。
        _, dA_ref = kernel.recovery_quadrature(mesh, 0)
        if len(dA_ref) != nqp:
            shape = weights = None
    if shape is not None and weights is not None:
        N = np.asarray(shape, dtype=float)           # (nqp, npe)
        dA = np.asarray(weights, dtype=float)        # (ne, nqp)
        nqp = N.shape[0]
        if N.shape[1] != npe:
            raise ValueError(
                f"{kernel.name} recovery shape matrix has shape {N.shape}; "
                f"expected (n_sample, {npe}).")
        if dA.shape != (mesh.n_elements, nqp):
            raise ValueError(
                f"{kernel.name} recovery weights have shape {dA.shape}; "
                f"expected ({mesh.n_elements}, {nqp}).")

        # 一致质量阵: local_mass_e = Nᵀ diag(dA_e) N — 一次性 einsum
        local_mass = np.einsum("qi,qj,eq->eij", N, N, dA)
        if elem_stress.ndim == 2:
            stress_qp = np.broadcast_to(
                elem_stress[:, None, :], (mesh.n_elements, nqp, n_comp))
        else:
            stress_qp = elem_stress
            if stress_qp.shape[1] == 1 and nqp != 1:
                # 单点采样内核 (如 Q4R/CST): 应力只在质心采样, 但质量阵
                # 必须用内核的积分规则 (单点 N^T N 只有秩 1, 一致质量阵
                # 会奇异)。代表应力沿积分点广播即可。
                stress_qp = np.broadcast_to(
                    stress_qp[:, 0, :][:, None, :],
                    (mesh.n_elements, nqp, n_comp))
            elif stress_qp.shape[1] != nqp:
                raise ValueError(
                    f"{stress_qp.shape[1]} stress samples, but "
                    f"{nqp} quadrature samples are required")
        local_rhs = np.einsum("qi,eqc,eq->eic", N, stress_qp, dA)

        rows = np.broadcast_to(conn[:, None, :], (ne, npe, npe)).ravel()
        cols = np.broadcast_to(conn[:, :, None], (ne, npe, npe)).ravel()
        mass = coo_matrix(
            (local_mass.ravel(), (rows, cols)),
            shape=(n_nodes, n_nodes)).tocsr()
        flat = conn.ravel()
        rhs = np.zeros((n_nodes, n_comp))
        for component in range(n_comp):
            rhs[:, component] = np.bincount(
                flat, weights=local_rhs[:, :, component].ravel(),
                minlength=n_nodes)
    else:
        # 逐单元兼容路径 (外部内核无批量恢复规则)
        rows, cols, values = [], [], []
        rhs = np.zeros((n_nodes, n_comp))
        for eid in range(mesh.n_elements):
            conn_e = conn[eid]
            N, dA_e = kernel.recovery_quadrature(mesh, eid)
            local_mass = np.einsum("qi,qj,q->ij", N, N, dA_e)
            if elem_stress.ndim == 2:
                stress_qp = np.broadcast_to(
                    elem_stress[eid], (len(dA_e), n_comp))
            else:
                stress_qp = elem_stress[eid]
                if stress_qp.shape[0] == 1 and len(dA_e) != 1:
                    stress_qp = np.broadcast_to(
                        stress_qp[0], (len(dA_e), n_comp))
                elif stress_qp.shape[0] != len(dA_e):
                    raise ValueError(
                        f"Element {eid}: {stress_qp.shape[0]} stress "
                        f"samples, but {len(dA_e)} quadrature samples "
                        "are required")
            local_rhs = np.einsum("qi,qc,q->ic", N, stress_qp, dA_e)
            for p, ni in enumerate(conn_e):
                for q, nj in enumerate(conn_e):
                    rows.append(int(ni))
                    cols.append(int(nj))
                    values.append(local_mass[p, q])
                rhs[ni] += local_rhs[p]
        mass = coo_matrix(
            (values, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    from scipy.sparse.linalg import splu
    orphan = np.flatnonzero(np.asarray(mass.diagonal()) == 0.0)
    if len(orphan):
        # 孤立节点行全零 → splu 抛 "Factor is exactly singular" 裸异常,
        # 与网格无关的底层报错
        raise ValueError(
            f"L2 投影失败: {len(orphan)} 个孤立节点未被任何单元引用, "
            f"一致质量矩阵奇异 (节点: {orphan[:10].tolist()}) — "
            "请先移除孤立节点")
    factor = splu(mass.tocsc())
    recovered = np.zeros((n_nodes, n_comp))
    for component in range(n_comp):
        recovered[:, component] = factor.solve(rhs[:, component])
    return recovered


def point_in_element(mesh, x, y):
    """Return the id of the element containing ``(x,y)``, or ``-1``."""
    mesh.build_connectivity()
    return mesh.element_kernel.find_containing_element(mesh, x, y)


def stress_at_point(mesh, result, x, y, mode="element"):
    """Query representative, two-sided, averaged or recovered point stress.

    模式语义:
      'element'  — 查询点所在单元的**代表应力** (单元均值, 与点位置无关)
      'sides'    — 共享边上两侧单元的**代表应力** (单元均值, 非点插值)
      'average'  — 两侧单元代表应力的**算术平均** (非点插值)
      'recovered'— SPR/L2 恢复场的**逐点插值** (唯一含点位置信息的模式)
    需要查询点处应力时用 'recovered'; 其余模式是单元级语义。
    共享节点 (>2 单元邻接) 时 sides/average 取查询边命中的第一对
    相邻单元 — 行为确定但语义任意, 请改用 'recovered' 获得点值。
    """
    valid_modes = {"element", "sides", "average", "recovered"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown stress query mode '{mode}'. "
            f"Expected one of {sorted(valid_modes)}.")
    if not isinstance(result, dict) or "stress" not in result:
        # 非 solve() 输出曾冒裸 KeyError — 契约前置校验
        raise ValueError(
            "stress_at_point: result 必须是 solve() 的返回 dict 且含 "
            f"'stress' 键, got {type(result).__name__}")

    eid = point_in_element(mesh, x, y)
    if eid < 0:
        raise ValueError(f"({x:.4f},{y:.4f}) not in mesh")
    if mode == "element":
        return result["stress"][eid]

    conn = mesh.elements[eid]
    if mode == "recovered":
        if "_spr_cache" not in result:
            result["_spr_cache"] = spr_recovery(
                mesh, result.get("stress_qp", result["stress"]))
        shape = mesh.element_kernel.shape_values_at(
            mesh.nodes[conn], x, y)
        if shape is None:
            return result["stress"][eid]
        return shape @ result["_spr_cache"][conn]

    # 点-边定位容差必须用局部边长尺度 + 坐标 ULP — 全局 span×1e-8
    # 在纳米网格 (边长 1e-12) 下大于所有真实边长, 全部边被判退化,
    # 双侧应力静默退化为单侧 (复测 2026-08-02)
    coord_ulp = 64.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(mesh.nodes))), np.finfo(float).tiny)
    neighbor = -1
    point = np.array([x, y])
    for ia, ib in mesh.element_kernel.local_edges:
        a, b = int(conn[ia]), int(conn[ib])
        pa, pb = mesh.nodes[a], mesh.nodes[b]
        edge = pb - pa
        length = float(np.linalg.norm(edge))
        if length <= coord_ulp:
            continue  # 浮点重合节点 (真退化)
        param = float((point - pa) @ edge) / (length * length)
        projection = pa + param * edge
        # 距离容差: 边自身长度的相对量 (1e-8) + ULP 兜底; 参数容差无量纲
        if (np.linalg.norm(point - projection)
                < max(1e-8 * length, coord_ulp)
                and -1e-8 <= param <= 1.0 + 1e-8):
            for other in mesh.elem_neighbors[eid]:
                other_conn = mesh.elements[other]
                if a in other_conn and b in other_conn:
                    neighbor = other
                    break
            break

    if neighbor < 0:
        return result["stress"][eid]
    first, second = result["stress"][eid], result["stress"][neighbor]
    if mode == "sides":
        return first, second
    return 0.5 * (first + second)
