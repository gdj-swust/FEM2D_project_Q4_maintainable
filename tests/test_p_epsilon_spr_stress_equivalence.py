"""P-ε 判别性红侧自证: spr/stress 残余 Python 循环消除的逐位等价门禁.

重构纪律: spr.py / stress.py 的一切数值改动必须是"等价计算路径",
输出逐位一致 (np.array_equal, 不允许 1 ULP 差异)。本文件内嵌
``_ref_*`` 参考实现 —— P-ε 重构前 (main=1902d6a, v9.28.0) 的
spr_recovery / nodal_L2_projection 数值核心原样快照 (语义冻结,
禁止修改)。重构前新测试对旧实现全绿 → 重构后仍全绿, 即等价性证明。

SPR-BC-2026-001 (v9.29.0) 重基线说明: 生产 spr_recovery 新增边界
节点 patch 替换 (ZZ92 §2.3) — 参考管线同步加入 ``_ref_boundary_patch_table``
(朴素逐节点重实现, 与生产 CSRLists 增强表逐行逐元素一致)。内部节点
语义仍与 P-ε 冻结快照逐位一致; 拟合核心 ``_ref_fit_node_block`` /
``_ref_fit_nodes_exact`` 的数值段未动。

覆盖:
* CST / Q4 / Q4R / Q4I 各 4 类手写网格 (规则 / 扭曲 / 微尺度 / 带孔)
* SPR: (ne,ncomp) 代表值 + (ne,nqp,ncomp) 积分点采样
* L2: 批量路径 (Q4/Q4I) + else 分支均匀堆叠路径 (CST/Q4R)
* 第三方内核 (无批量恢复规则) 的回退路径
* 100k 网格冒烟 (防 O(n²) 回归, 无 gmsh)
"""
import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from fem2d import Mesh
from fem2d.spr import spr_recovery
from fem2d.stress import nodal_L2_projection

# ────────────────────────────────────────────────────────────────────────
# 旧实现参考 (P-ε 重构前快照, 数值语义冻结 — 禁止修改)
# ────────────────────────────────────────────────────────────────────────

_REF_WELL_CONDITIONED = 1.0e-12
_REF_SAMPLE_BLOCK = 4_000_000


def _ref_normal_matrix(dx, dy, starts, counts):
    """段归约 A^T A, 形状 (n_group, 3, 3) — 旧实现原样."""
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


def _ref_boundary_patch_table(mesh):
    """SPR-BC-2026-001 边界增强表参考 — 朴素逐节点重实现.

    边界节点 b 的行替换为最近内部候选节点 (邻接单元顶点 − 边界集,
    为空扩 ring-1; 候选须 patch 非空; 并列取最小节点号) 的 patch 行;
    无候选保留自身行。内部节点行原样。返回 list[list[int]] —
    与生产 _boundary_patch_table 的 CSRLists 逐行逐元素一致。
    """
    mesh.build_connectivity()
    table = mesh.node_to_elems
    rows = [list(table.ids(n)) for n in range(mesh.n_nodes)]
    if not mesh.boundary_edges:
        return rows
    bset = set()
    for lo, hi in mesh.boundary_edges:
        bset.add(int(lo))
        bset.add(int(hi))
    for b in sorted(bset):
        cands = set()
        for eid in rows[b]:
            cands.update(int(v) for v in mesh.elements[int(eid)])
        cands -= bset
        if not cands:
            ring1 = set(rows[b])
            for eid in list(ring1):
                ring1.update(int(nb) for nb in mesh.elem_neighbors[int(eid)])
            for eid in ring1:
                cands.update(int(v) for v in mesh.elements[int(eid)])
            cands -= bset
        cands = {n for n in cands if rows[n]}
        if not cands:
            continue
        xb, yb = mesh.nodes[b]
        i = min(cands, key=lambda n: (
            float(np.hypot(mesh.nodes[n, 0] - xb, mesh.nodes[n, 1] - yb)),
            n))
        rows[b] = list(rows[i])
    return rows


def _ref_node_patch_csr(mesh, table=None):
    """node_to_elems 的 CSR 指针/索引数组 — 旧实现原样."""
    if table is None:
        table = mesh.node_to_elems
    ptr = getattr(table, "ptr", None)
    if ptr is not None:
        return (np.asarray(ptr, dtype=np.int64),
                np.asarray(table.flat, dtype=np.int64))
    lengths = np.fromiter(
        (len(row) for row in table), dtype=np.int64, count=len(table))
    ptr = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=ptr[1:])
    flat = np.fromiter(
        (eid for row in table for eid in row),
        dtype=np.int64, count=int(ptr[-1]))
    return ptr, flat


def _ref_fit_node_block(mesh, sample_xy, sample_values, ptr, flat,
                        node_lo, node_hi, recovered):
    """批量拟合 [node_lo, node_hi) — 旧实现原样 (含逐节点切片拼接 +
    逐分量 rhs 组装)。"""
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

    dx0 = sx - node_x[per_sample_node]
    dy0 = sy - node_y[per_sample_node]
    h0 = np.maximum.reduceat(np.hypot(dx0, dy0), starts)
    h0 = np.where(h0 > 0.0, h0, 1.0)
    normal0 = _ref_normal_matrix(
        dx0 / h0[per_sample_node], dy0 / h0[per_sample_node],
        starts, sample_counts)
    eig0 = np.linalg.eigvalsh(normal0)
    ok = (sample_counts >= 3) & (
        eig0[:, 0] >= _REF_WELL_CONDITIONED * np.maximum(eig0[:, 2], 0.0))

    unresolved = np.concatenate([empty_nodes, node_ids[~ok]])
    if not np.any(ok):
        return unresolved

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

    normal = _ref_normal_matrix(dx, dy, starts_ok, counts_ok)
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


def _ref_fit_nodes_exact(mesh, sample_xy, sample_values, nodes, recovered,
                         table=None):
    """旧逐点实现: 三环 patch 扩张 + 逐分量 lstsq + 条件数守卫 — 原样."""
    n_comp = sample_values.shape[-1]
    node_elems = mesh.node_to_elems if table is None else table
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
                    h_scale = max(np.linalg.norm(xy, axis=1).max(),
                                  np.finfo(float).tiny)
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
        h_patch = max(np.linalg.norm(design[:, 1:], axis=1).max(),
                      np.finfo(float).tiny)
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


def _ref_recovery_sample_positions(mesh, n_sample):
    """旧回退实现: 逐单元 recovery_quadrature + 逐单元 matmul — 原样."""
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


def _ref_prepare_samples(mesh, elem_stress):
    """旧采样准备: 2D → 质心单点; 3D → 恢复规则采样 — 原样."""
    raw = np.asarray(elem_stress)
    if np.iscomplexobj(raw):
        raise ValueError(
            "spr_recovery: elem_stress 必须为实数 — complex 虚部会被丢弃")
    if raw.dtype.kind not in ("i", "u", "f", "b"):
        raise TypeError(
            f"spr_recovery: elem_stress 必须为数值数组, got dtype {raw.dtype}")
    elem_stress = np.asarray(raw, dtype=float)
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
    if not np.all(np.isfinite(elem_stress)):
        raise ValueError(
            "spr_recovery: elem_stress contains NaN/Inf — 恢复输入非法")

    if elem_stress.ndim == 2:
        positions = np.mean(mesh.nodes[mesh.elements], axis=1)
        return positions[:, None, :], elem_stress[:, None, :]
    positions = _ref_recovery_sample_positions(mesh, elem_stress.shape[1])
    return positions, elem_stress


def _ref_spr_recovery(mesh, elem_stress):
    """旧 SPR 管线: 批处理块循环 + 精确路径 — 原样 (边界增强表为
    SPR-BC-2026-001 重基线新增)."""
    sample_xy, sample_values = _ref_prepare_samples(mesh, elem_stress)
    n_comp = sample_values.shape[-1]
    recovered = np.zeros((mesh.n_nodes, n_comp))

    table = _ref_boundary_patch_table(mesh)
    ptr, flat = _ref_node_patch_csr(mesh, table)
    per_node = np.diff(ptr) * sample_xy.shape[1]
    n_nodes = mesh.n_nodes
    unresolved = []

    node_lo = 0
    while node_lo < n_nodes:
        node_hi = node_lo + 1
        budget = int(per_node[node_lo])
        while (node_hi < n_nodes
               and budget + per_node[node_hi] <= _REF_SAMPLE_BLOCK):
            budget += int(per_node[node_hi])
            node_hi += 1
        unresolved.append(_ref_fit_node_block(
            mesh, sample_xy, sample_values, ptr, flat,
            node_lo, node_hi, recovered))
        node_lo = node_hi

    pending = (np.unique(np.concatenate(unresolved))
               if unresolved else np.empty(0, dtype=np.int64))
    if pending.size:
        _ref_fit_nodes_exact(mesh, sample_xy, sample_values, pending,
                             recovered, table)
    return recovered


def _ref_l2_batch_assembly(mesh, N_all, dA_all, stress_qp):
    """旧批量 L2 组装 (含逐分量 bincount rhs) — 原样."""
    conn = mesh.elements
    ne, npe = conn.shape
    n_nodes = mesh.n_nodes
    n_comp = stress_qp.shape[-1]
    local_mass = np.einsum("eqi,eqj,eq->eij", N_all, N_all, dA_all)
    local_rhs = np.einsum("eqi,eqc,eq->eic", N_all, stress_qp, dA_all)
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
    return mass, rhs


def _ref_l2_stress_qp(elem_stress, nqp, n_comp, n_elem):
    """积分点应力准备 — 旧实现原样."""
    if elem_stress.ndim == 2:
        return np.broadcast_to(
            elem_stress[:, None, :], (n_elem, nqp, n_comp))
    stress_qp = elem_stress
    if stress_qp.shape[1] == 1 and nqp != 1:
        return np.broadcast_to(
            stress_qp[:, 0, :][:, None, :], (n_elem, nqp, n_comp))
    if stress_qp.shape[1] != nqp:
        raise ValueError(
            f"{stress_qp.shape[1]} stress samples, but "
            f"{nqp} quadrature samples are required")
    return stress_qp


def _ref_nodal_L2_projection(mesh, elem_stress):
    """旧 L2 投影管线: 批量/均匀堆叠/逐单元 + 逐分量 splu — 原样."""
    mesh.build_connectivity()
    raw_stress = np.asarray(elem_stress)
    if np.iscomplexobj(raw_stress):
        raise ValueError(
            "nodal_L2_projection: elem_stress 必须为实数 — "
            "complex 虚部会被静默丢弃")
    elem_stress = np.asarray(raw_stress, dtype=float)
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
    n_comp = elem_stress.shape[-1]
    n_nodes = mesh.n_nodes
    kernel = mesh.element_kernel
    conn = mesh.elements
    ne = conn.shape[0]
    npe = conn.shape[1]

    shape = kernel.recovery_shape_matrix(mesh)
    weights = kernel.recovery_weights(mesh)
    if shape is not None and weights is not None:
        N = np.asarray(shape, dtype=float)
        dA = np.asarray(weights, dtype=float)
        nqp = N.shape[0]
        _, dA_ref = kernel.recovery_quadrature(mesh, 0)
        if len(dA_ref) != nqp:
            shape = weights = None
    if shape is not None and weights is not None:
        N = np.asarray(shape, dtype=float)
        dA = np.asarray(weights, dtype=float)
        nqp = N.shape[0]
        if N.shape[1] != npe:
            raise ValueError(
                f"{kernel.name} recovery shape matrix has shape {N.shape}; "
                f"expected (n_sample, {npe}).")
        if dA.shape != (mesh.n_elements, nqp):
            raise ValueError(
                f"{kernel.name} recovery weights have shape {dA.shape}; "
                f"expected ({mesh.n_elements}, {nqp}).")

        stress_qp = _ref_l2_stress_qp(elem_stress, nqp, n_comp, ne)
        mass, rhs = _ref_l2_batch_assembly(
            mesh, np.broadcast_to(N, (ne, nqp, npe)), dA, stress_qp)
    else:
        N_list, dA_list = [], []
        nqp, n_shape, uniform = None, None, True
        for eid in range(mesh.n_elements):
            N, dA_e = kernel.recovery_quadrature(mesh, eid)
            if N is None or dA_e is None:
                uniform = False
            elif nqp is None:
                nqp, n_shape = len(dA_e), N.shape
            elif len(dA_e) != nqp or N.shape != n_shape:
                uniform = False
            N_list.append(N)
            dA_list.append(dA_e)
        if uniform and nqp is not None:
            N_all = np.stack(N_list)
            dA_all = np.stack(dA_list)
            stress_qp = _ref_l2_stress_qp(elem_stress, nqp, n_comp, ne)
            mass, rhs = _ref_l2_batch_assembly(mesh, N_all, dA_all,
                                               stress_qp)
        else:
            rows, cols, values = [], [], []
            rhs = np.zeros((n_nodes, n_comp))
            for eid in range(mesh.n_elements):
                conn_e = conn[eid]
                N, dA_e = N_list[eid], dA_list[eid]
                local_mass = np.einsum("qi,qj,q->ij", N, N, dA_e)
                stress_qp = _ref_l2_stress_qp(
                    elem_stress[eid:eid + 1], len(dA_e), n_comp, 1)[0]
                local_rhs = np.einsum("qi,qc,q->ic", N, stress_qp, dA_e)
                for p, ni in enumerate(conn_e):
                    for q, nj in enumerate(conn_e):
                        rows.append(int(ni))
                        cols.append(int(nj))
                        values.append(local_mass[p, q])
                    rhs[ni] += local_rhs[p]
            mass = coo_matrix(
                (values, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    orphan = np.flatnonzero(np.asarray(mass.diagonal()) == 0.0)
    if len(orphan):
        raise ValueError(
            f"L2 投影失败: {len(orphan)} 个孤立节点未被任何单元引用, "
            f"一致质量矩阵奇异 (节点: {orphan[:10].tolist()}) — "
            "请先移除孤立节点")
    factor = splu(mass.tocsc())
    recovered = np.zeros((n_nodes, n_comp))
    for component in range(n_comp):
        recovered[:, component] = factor.solve(rhs[:, component])
    return recovered


# ────────────────────────────────────────────────────────────────────────
# 手写网格 (无 gmsh) — 规则 / 扭曲 / 微尺度 / 带孔
# ────────────────────────────────────────────────────────────────────────

def _grid_mesh(elem_type, nx=8, ny=8, distort=0.0, scale=1.0, seed=3):
    """矩形网格; distort 抖动内部节点, scale 整体缩放 (微尺度)."""
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    gx, gy = np.meshgrid(xs, ys)
    nodes = np.column_stack([gx.ravel(), gy.ravel()]) * scale
    if distort:
        rng = np.random.default_rng(seed)
        d = rng.normal(0.0, distort, size=nodes.shape)
        span = max(nodes[:, 0].max() - nodes[:, 0].min(), nodes[:, 0].max())
        tol = 1e-9 * span
        xmin, xmax = nodes[:, 0].min(), nodes[:, 0].max()
        ymin, ymax = nodes[:, 1].min(), nodes[:, 1].max()
        interior = ((nodes[:, 0] > xmin + tol) & (nodes[:, 0] < xmax - tol)
                    & (nodes[:, 1] > ymin + tol) & (nodes[:, 1] < ymax - tol))
        nodes[interior] += d[interior]
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            elems.append([a, a + 1, a + nx + 2, a + nx + 1])
    if elem_type == "CPS3":
        tris = []
        for (a, b, c, d) in elems:
            tris.append([a, b, c])
            tris.append([b, d, c])
        elems = tris
    return Mesh(nodes=nodes, elements=np.array(elems, dtype=int),
                elem_type=elem_type)


def _hole_mesh(elem_type, nx=12, ny=12):
    """中心挖圆孔网格: 删除孔内单元 + 孤立节点重编号."""
    base = _grid_mesh("CPS4", nx, ny)
    base.build_connectivity()
    c = base.centroids
    r2 = (c[:, 0] - 0.5) ** 2 + (c[:, 1] - 0.5) ** 2
    keep_e = r2 >= 0.22 ** 2
    quads = base.elements[keep_e]
    used = np.unique(quads)
    node_map = -np.ones(base.n_nodes, dtype=int)
    node_map[used] = np.arange(used.size)
    nodes = base.nodes[used]
    if elem_type == "CPS3":
        tris = []
        for (a, b, c, d) in node_map[quads]:
            tris.append([a, b, c])
            tris.append([b, d, c])
        return Mesh(nodes=nodes, elements=np.array(tris, dtype=int),
                    elem_type=elem_type)
    return Mesh(nodes=nodes, elements=node_map[quads], elem_type=elem_type)


_MESH_CASES = {
    "regular": lambda et: _grid_mesh(et, 8, 8),
    "distorted": lambda et: _grid_mesh(et, 8, 8, distort=0.02),
    "micro": lambda et: _grid_mesh(et, 8, 8, scale=1e-6),
    "hole": _hole_mesh,
}


def _stress_input(mesh, nqp, seed=11):
    """确定性应力输入: (ne, ncomp) 代表值 + (ne, nqp, ncomp) 采样."""
    rng = np.random.default_rng(seed)
    if nqp is None:
        return rng.normal(size=(mesh.n_elements, 3))
    return rng.normal(size=(mesh.n_elements, nqp, 3))


# ────────────────────────────────────────────────────────────────────────
# 逐位等价测试
# ────────────────────────────────────────────────────────────────────────

SPR_CASES = [
    # (elem_type, nqp 或 None, 网格用例)
    ("CPS3", None), ("CPS4", None), ("CPS4", 4),
    ("CPS4R", None), ("CPS4R", 1), ("CPS4I", None), ("CPS4I", 4),
]


@pytest.mark.parametrize("elem_type,nqp", SPR_CASES)
@pytest.mark.parametrize("case", sorted(_MESH_CASES))
def test_spr_equivalence(elem_type, nqp, case):
    """spr_recovery 新旧逐位等价 (CST/Q4/Q4R/Q4I × 规则/扭曲/微尺度/带孔)."""
    mesh = _MESH_CASES[case](elem_type)
    stress = _stress_input(mesh, nqp)
    assert np.array_equal(
        spr_recovery(mesh, stress), _ref_spr_recovery(mesh, stress)), (
        f"spr_recovery 逐位不等: {elem_type} × {case} × nqp={nqp}")


L2_CASES = [
    ("CPS3", None), ("CPS3", 3),
    ("CPS4", None), ("CPS4", 4),
    ("CPS4R", None), ("CPS4R", 1), ("CPS4I", None), ("CPS4I", 4),
]


@pytest.mark.parametrize("elem_type,nqp", L2_CASES)
@pytest.mark.parametrize("case", sorted(_MESH_CASES))
def test_l2_equivalence(elem_type, nqp, case):
    """nodal_L2_projection 新旧逐位等价 (批量路径 + else 均匀堆叠路径)."""
    mesh = _MESH_CASES[case](elem_type)
    stress = _stress_input(mesh, nqp)
    assert np.array_equal(
        nodal_L2_projection(mesh, stress),
        _ref_nodal_L2_projection(mesh, stress)), (
        f"nodal_L2_projection 逐位不等: {elem_type} × {case} × nqp={nqp}")


# ────────────────────────────────────────────────────────────────────────
# 第三方内核回退路径 (无 recovery_shape_matrix / recovery_weights)
# ────────────────────────────────────────────────────────────────────────

class _ThirdPartyKernel:
    """外部内核: 无批量恢复规则, 逐单元 recovery_quadrature (均匀 3 点)."""

    name = "third_party_uniform"
    local_edges = ((0, 1), (1, 2), (2, 0))

    @staticmethod
    def recovery_shape_matrix(mesh):
        return None

    @staticmethod
    def recovery_weights(mesh):
        return None

    @staticmethod
    def recovery_quadrature(mesh, eid):
        N = np.array([
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ])
        return N, np.full(3, mesh.areas[eid] / 3.0)


def _mesh_with_third_party_kernel(case):
    mesh = _MESH_CASES[case]("CPS3")
    mesh.build_connectivity()
    mesh.element_kernel = _ThirdPartyKernel()
    return mesh


@pytest.mark.parametrize("case", sorted(_MESH_CASES))
def test_spr_third_party_fallback_equivalence(case):
    """回退路径 (逐单元 recovery_quadrature) SPR 逐位等价."""
    mesh = _mesh_with_third_party_kernel(case)
    stress = _stress_input(mesh, 3)
    assert np.array_equal(
        spr_recovery(mesh, stress), _ref_spr_recovery(mesh, stress)), (
        f"SPR 第三方内核回退路径逐位不等: {case}")


@pytest.mark.parametrize("case", sorted(_MESH_CASES))
def test_l2_third_party_uniform_equivalence(case):
    """第三方内核均匀规则 → else 分支均匀堆叠路径 L2 逐位等价."""
    mesh = _mesh_with_third_party_kernel(case)
    stress = _stress_input(mesh, 3)
    assert np.array_equal(
        nodal_L2_projection(mesh, stress),
        _ref_nodal_L2_projection(mesh, stress)), (
        f"L2 第三方内核均匀堆叠路径逐位不等: {case}")


class _ThirdPartyNonUniformKernel(_ThirdPartyKernel):
    """非均匀恢复规则: 偶单元 3 点 Hammer, 奇单元 2 点子集
    (nqp/shape 逐单元不同) → L2 逐单元契约路径 (行为冻结)."""

    _N3 = np.array([
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ])

    @classmethod
    def recovery_quadrature(cls, mesh, eid):
        if eid % 2 == 0:
            return cls._N3, np.full(3, mesh.areas[eid] / 3.0)
        return cls._N3[:2], np.full(2, mesh.areas[eid] / 2.0)


@pytest.mark.parametrize("case", sorted(_MESH_CASES))
def test_l2_third_party_non_uniform_equivalence(case):
    """第三方内核非均匀规则 → 逐单元 else 分支 (行为冻结) L2 逐位等价."""
    mesh = _MESH_CASES[case]("CPS3")
    mesh.build_connectivity()
    mesh.element_kernel = _ThirdPartyNonUniformKernel()
    # 单点采样输入 → 逐单元 nqp 广播路径 (见 _l2_stress_qp 契约)
    stress = _stress_input(mesh, 1)
    assert np.array_equal(
        nodal_L2_projection(mesh, stress),
        _ref_nodal_L2_projection(mesh, stress)), (
        f"L2 第三方内核逐单元路径逐位不等: {case}")


# ────────────────────────────────────────────────────────────────────────
# 100k 冒烟 (防 O(n²) 回归; 无时间断言 — 绝对阈值禁用)
# ────────────────────────────────────────────────────────────────────────

def test_100k_smoke():
    """100k 单元 Q4: SPR + L2 全路径可跑通, 输出形状/有限性正确."""
    mesh = _grid_mesh("CPS4", 317, 316)
    stress_2d = _stress_input(mesh, None, seed=5)
    stress_qp = _stress_input(mesh, 4, seed=6)
    spr_2d = spr_recovery(mesh, stress_2d)
    spr_qp = spr_recovery(mesh, stress_qp)
    l2_2d = nodal_L2_projection(mesh, stress_2d)
    l2_qp = nodal_L2_projection(mesh, stress_qp)
    for name, out in (("SPR(2D)", spr_2d), ("SPR(3D)", spr_qp),
                      ("L2(2D)", l2_2d), ("L2(3D)", l2_qp)):
        assert out.shape == (mesh.n_nodes, 3), name
        assert np.all(np.isfinite(out)), name
