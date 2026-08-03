"""SPR 应力恢复 — Zienkiewicz-Zhu Patch Recovery.

Bathe §4.3.6: 对每个节点的周围单元片做线性最小二乘拟合 σ*(x,y) = a + bx + cy,
从单元响应采样值恢复节点处的改进应力。

CST 使用三点代表值；Q4 / Q4E 可传入 2×2 Gauss (Barlow) 点响应。

实现分两条路径:

* 批量路径 —— 以 CSR 形式一次性铺开全部 (节点, 单元, 积分点) 采样，
  用 ``np.add.reduceat`` 段归约直接装配 3×3 法方程，再批量求解。
* 精确路径 —— 仅对批量路径判定为秩不足或病态的节点启用，
  保留原有的三环 patch 扩张 + ``lstsq`` + 条件数守卫逐点逻辑。

两条路径对良态节点给出相同的拟合平面（同一组采样、同一仿射重参数化），
病态节点则完全走原算法，因此恢复结果与逐点实现一致。
"""
import numpy as np

# 批量路径的良态判据: cond(A) <= 1e6 (法方程特征值比 1e-12)。
# 比原逐点实现的 1e8 更严格 —— 落在两者之间的节点被推给精确路径，
# 因此判据只会更保守, 不会改变结果。
_WELL_CONDITIONED = 1.0e-12

# 单批采样数上限, 控制大网格上的临时内存峰值。
_SAMPLE_BLOCK = 4_000_000


def recovery_sample_positions(mesh, n_sample):
    """返回全部单元的恢复采样点物理坐标 ``(ne, n_sample, 2)``。

    内核若提供与单元无关的恢复形函数矩阵, 则一次 ``einsum`` 即可；
    否则退回逐单元 ``recovery_quadrature`` 协议 (兼容第三方内核)。
    """
    kernel = mesh.element_kernel
    shape = kernel.recovery_shape_matrix(mesh)
    if shape is not None:
        shape = np.asarray(shape, dtype=float)
        if shape.shape[0] != n_sample:
            raise ValueError(
                f"{kernel.name} recovery rule provides {shape.shape[0]} "
                f"sample points, but {n_sample} stress samples were given")
        return np.einsum("qn,end->eqd", shape, mesh.nodes[mesh.elements])

    positions = np.empty((mesh.n_elements, n_sample, 2), dtype=float)
    for eid, conn in enumerate(mesh.elements):
        shape, _ = kernel.recovery_quadrature(mesh, eid)
        if shape.shape[0] != n_sample:
            raise ValueError(
                f"Element {eid}: {n_sample} stress samples, "
                f"but kernel returned {shape.shape[0]} recovery points")
        positions[eid] = shape @ mesh.nodes[conn]
    return positions


def _prepare_samples(mesh, elem_stress):
    """校验输入并返回 ``(sample_xy, sample_values)``。"""
    elem_stress = np.asarray(elem_stress, dtype=float)
    if elem_stress.ndim == 1:
        elem_stress = elem_stress.reshape(-1, 1)
    if elem_stress.ndim not in (2, 3):
        raise ValueError(
            "elem_stress must have shape (ne,ncomp) or (ne,nqp,ncomp)")
    mesh.build_connectivity()
    if elem_stress.shape[0] != mesh.n_elements:
        raise ValueError(
            f"elem_stress first dimension must be {mesh.n_elements}, "
            f"got {elem_stress.shape[0]}")

    if elem_stress.ndim == 2:
        # (ne,ncomp): 代表应力定义在单元自然中心 (CST 形心 / Q4R 中心
        # 单点 / Q4·Q4I 对称高斯均值 — 物理位置均为节点坐标均值), 不是
        # 多边形质心; 扭曲四边形上两者差 ~3.7% 边长, SPR 线性精确性从
        # 2e-16 退化到 5e-3 (审计 2026-08-03)
        positions = np.mean(mesh.nodes[mesh.elements], axis=1)
        return positions[:, None, :], elem_stress[:, None, :]
    positions = recovery_sample_positions(mesh, elem_stress.shape[1])
    return positions, elem_stress


def _node_patch_csr(mesh):
    """返回 ``node_to_elems`` 的 CSR 指针/索引数组。"""
    table = mesh.node_to_elems
    ptr = getattr(table, "ptr", None)
    if ptr is not None:
        return (np.asarray(ptr, dtype=np.int64),
                np.asarray(table.flat, dtype=np.int64))
    # 兼容手工构造的 list[list[int]]
    lengths = np.fromiter(
        (len(row) for row in table), dtype=np.int64, count=len(table))
    ptr = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=ptr[1:])
    flat = np.fromiter(
        (eid for row in table for eid in row),
        dtype=np.int64, count=int(ptr[-1]))
    return ptr, flat


def _normal_matrix(dx, dy, starts, counts):
    """返回段归约得到的 ``A^T A``, 形状 ``(n_group, 3, 3)``。"""
    normal = np.empty((counts.size, 3, 3), dtype=float)
    normal[:, 0, 0] = counts
    s_x = np.add.reduceat(dx, starts)
    s_y = np.add.reduceat(dy, starts)
    normal[:, 0, 1] = normal[:, 1, 0] = s_x
    normal[:, 0, 2] = normal[:, 2, 0] = s_y
    normal[:, 1, 1] = np.add.reduceat(dx * dx, starts)
    normal[:, 1, 2] = normal[:, 2, 1] = np.add.reduceat(dx * dy, starts)
    normal[:, 2, 2] = np.add.reduceat(dy * dy, starts)
    return normal


def _fit_node_block(mesh, sample_xy, sample_values, ptr, flat,
                    node_lo, node_hi, recovered):
    """对 ``[node_lo, node_hi)`` 的节点批量拟合, 返回需精确处理的节点。"""
    n_qp = sample_xy.shape[1]
    counts = np.diff(ptr)[node_lo:node_hi]
    node_ids = np.arange(node_lo, node_hi, dtype=np.int64)

    active = counts > 0
    empty_nodes = node_ids[~active]
    node_ids = node_ids[active]
    counts = counts[active]
    if node_ids.size == 0:
        return empty_nodes

    sample_counts = counts * n_qp
    starts = np.zeros(node_ids.size, dtype=np.int64)
    np.cumsum(sample_counts[:-1], out=starts[1:])

    # 每个节点的单元在 flat 中连续 → 直接取切片再展开积分点。
    elem_slice = flat[ptr[node_lo]:ptr[node_hi]] if np.all(active) else (
        np.concatenate([flat[ptr[nid]:ptr[nid + 1]] for nid in node_ids]))
    elem_of = np.repeat(elem_slice, n_qp)
    qp_of = np.tile(np.arange(n_qp, dtype=np.int64), elem_slice.size)

    sx = sample_xy[elem_of, qp_of, 0]
    sy = sample_xy[elem_of, qp_of, 1]
    values = sample_values[elem_of, qp_of, :]
    n_comp = values.shape[1]

    per_sample_node = np.repeat(np.arange(node_ids.size), sample_counts)
    node_x = mesh.nodes[node_ids, 0]
    node_y = mesh.nodes[node_ids, 1]

    # ── 1. 复现原实现的 ring-0 接受判据 (以节点为中心的归一化设计矩阵) ──
    dx0 = sx - node_x[per_sample_node]
    dy0 = sy - node_y[per_sample_node]
    h0 = np.maximum.reduceat(np.hypot(dx0, dy0), starts)
    h0 = np.where(h0 > 0.0, h0, 1.0)
    normal0 = _normal_matrix(
        dx0 / h0[per_sample_node], dy0 / h0[per_sample_node],
        starts, sample_counts)
    eig0 = np.linalg.eigvalsh(normal0)
    ok = (sample_counts >= 3) & (
        eig0[:, 0] >= _WELL_CONDITIONED * np.maximum(eig0[:, 2], 0.0))

    unresolved = np.concatenate([empty_nodes, node_ids[~ok]])
    if not np.any(ok):
        return unresolved

    # ── 2. 良态节点: 以 patch 均值为中心、按 patch 尺度归一化后拟合 ──
    keep = ok[per_sample_node]
    counts_ok = sample_counts[ok]
    starts_ok = np.zeros(counts_ok.size, dtype=np.int64)
    np.cumsum(counts_ok[:-1], out=starts_ok[1:])
    sx_ok, sy_ok = sx[keep], sy[keep]
    values_ok = values[keep]
    per_sample_ok = np.repeat(np.arange(counts_ok.size), counts_ok)

    inv_count = 1.0 / counts_ok
    mean_x = np.add.reduceat(sx_ok, starts_ok) * inv_count
    mean_y = np.add.reduceat(sy_ok, starts_ok) * inv_count
    dx = sx_ok - mean_x[per_sample_ok]
    dy = sy_ok - mean_y[per_sample_ok]
    h = np.maximum.reduceat(np.hypot(dx, dy), starts_ok)
    h = np.where(h > 0.0, h, 1.0)
    dx = dx / h[per_sample_ok]
    dy = dy / h[per_sample_ok]

    normal = _normal_matrix(dx, dy, starts_ok, counts_ok)
    rhs = np.empty((counts_ok.size, 3, n_comp), dtype=float)
    for comp in range(n_comp):
        column = values_ok[:, comp]
        rhs[:, 0, comp] = np.add.reduceat(column, starts_ok)
        rhs[:, 1, comp] = np.add.reduceat(dx * column, starts_ok)
        rhs[:, 2, comp] = np.add.reduceat(dy * column, starts_ok)

    target = node_ids[ok]
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return np.concatenate([unresolved, target])

    xn = (mesh.nodes[target, 0] - mean_x) / h
    yn = (mesh.nodes[target, 1] - mean_y) / h
    fitted = (coefficients[:, 0, :]
              + coefficients[:, 1, :] * xn[:, None]
              + coefficients[:, 2, :] * yn[:, None])
    bad = ~np.all(np.isfinite(fitted), axis=1)
    if np.any(bad):
        unresolved = np.concatenate([unresolved, target[bad]])
        fitted[bad] = 0.0
    recovered[target] = fitted
    return unresolved


def _fit_nodes_exact(mesh, sample_xy, sample_values, nodes, recovered):
    """原逐点实现: 三环 patch 扩张 + lstsq + 条件数守卫。"""
    n_comp = sample_values.shape[-1]
    node_elems = mesh.node_to_elems
    for raw_nid in nodes:
        nid = int(raw_nid)
        patch_set = set(node_elems[nid])
        frontier = set(patch_set)
        ring = 0
        while ring < 3:
            ids = sorted(patch_set)
            if ids:
                xy_abs = sample_xy[ids].reshape(-1, 2)
                if len(xy_abs) >= 3:
                    xy = xy_abs - mesh.nodes[nid]
                    h_scale = max(np.linalg.norm(xy, axis=1).max(), np.finfo(float).tiny)
                    a_test = np.ones((len(xy), 3))
                    a_test[:, 1:] = xy / h_scale
                    if np.linalg.matrix_rank(a_test) == 3:
                        if np.linalg.cond(a_test) <= 1e8:
                            break
            new_frontier = set()
            for eid in frontier:
                for neighbor in mesh.elem_neighbors[eid]:
                    if neighbor not in patch_set:
                        new_frontier.add(neighbor)
            if not new_frontier:
                break
            patch_set.update(new_frontier)
            frontier = new_frontier
            ring += 1

        patch = sorted(patch_set)
        if not patch:
            recovered[nid] = 0.0
            continue
        xy_patch = sample_xy[patch].reshape(-1, 2)
        value_patch = sample_values[patch].reshape(-1, n_comp)
        count = len(xy_patch)

        if count < 3:
            recovered[nid] = (
                value_patch.mean(axis=0) if count > 0 else 0.0)
            continue

        design = np.ones((count, 3))
        design[:, 1:] = xy_patch
        x_mean, y_mean = design[:, 1].mean(), design[:, 2].mean()
        design[:, 1] -= x_mean
        design[:, 2] -= y_mean
        h_patch = max(np.linalg.norm(design[:, 1:], axis=1).max(), np.finfo(float).tiny)
        design[:, 1:] /= h_patch
        xn = (mesh.nodes[nid, 0] - x_mean) / h_patch
        yn = (mesh.nodes[nid, 1] - y_mean) / h_patch

        for comp in range(n_comp):
            values = value_patch[:, comp]
            try:
                coef, _, rank, sv = np.linalg.lstsq(
                    design, values, rcond=None)
                accept = rank == 3
                if accept and len(sv) >= 3:
                    accept = sv[0] / (sv[2] + np.finfo(float).tiny) <= 1e8
                if accept:
                    recovered[nid, comp] = (
                        coef[0] + coef[1] * xn + coef[2] * yn)
                else:
                    recovered[nid, comp] = np.average(values)
            except np.linalg.LinAlgError:
                recovered[nid, comp] = np.mean(values)


def spr_recovery(mesh, elem_stress):
    """Superconvergent Patch Recovery — 逐节点最小二乘拟合

    对每个节点:
      1. 收集周围单元的代表值，或积分点应力
      2. 拟合 p(x,y) = a + bx + cy
      3. 在节点位置求值 → 恢复后的节点应力

    ``elem_stress`` 可为 ``(ne,ncomp)``，也可为 ``(ne,nqp,ncomp)``。
    后者的采样位置由当前 element kernel 的恢复积分规则给出，因此
    Q4/Q4E 会使用 2×2 Gauss/Barlow 点。

    支持单分量 (如 vm, s1) 或多分量 (如 [σ_x, σ_y, τ_xy])。
    """
    sample_xy, sample_values = _prepare_samples(mesh, elem_stress)
    n_comp = sample_values.shape[-1]
    recovered = np.zeros((mesh.n_nodes, n_comp))

    ptr, flat = _node_patch_csr(mesh)
    per_node = np.diff(ptr) * sample_xy.shape[1]
    n_nodes = mesh.n_nodes
    unresolved = []

    node_lo = 0
    while node_lo < n_nodes:
        node_hi = node_lo + 1
        budget = int(per_node[node_lo])
        while (node_hi < n_nodes
               and budget + per_node[node_hi] <= _SAMPLE_BLOCK):
            budget += int(per_node[node_hi])
            node_hi += 1
        unresolved.append(_fit_node_block(
            mesh, sample_xy, sample_values, ptr, flat,
            node_lo, node_hi, recovered))
        node_lo = node_hi

    pending = (np.unique(np.concatenate(unresolved))
               if unresolved else np.empty(0, dtype=np.int64))
    if pending.size:
        _fit_nodes_exact(mesh, sample_xy, sample_values, pending, recovered)
    return recovered

