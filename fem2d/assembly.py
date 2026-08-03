"""Global stiffness assembly for registered 2-D element kernels.

Element kernels compute local matrices.  This module only maps local degrees
of freedom into the global system, so its code is independent of the number of
nodes in an element.
"""
import inspect

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix

ASSEMBLY_BATCH_ELEMENTS = 50000


def _check_local_symmetry(Ke, kernel_name):
    """scatter 前批量检查单元刚度对称性 — 全局 _check_symmetry 只抽样
    256 行, 未抽中位置的非对称项会漏检 (第五轮外部审查: 构造 1000×1000
    仅未抽中位置非对称的矩阵被放行)。Ke 是小型密集矩阵, 全量检查成本
    可忽略; 第三方非对称内核在这里被拒绝, 全局抽样检查保留兜底。
    """
    skew = np.abs(Ke - np.transpose(Ke, (0, 2, 1)))
    scale = max(float(np.abs(Ke).max()), 1.0)
    if skew.size and float(skew.max()) > 1e-10 * scale:
        bad = int(np.count_nonzero(skew > 1e-10 * scale))
        raise RuntimeError(
            f"{kernel_name} kernel returned asymmetric local stiffness "
            f"(relative skew {float(skew.max())/scale:.3e} > 1e-10, "
            f"{bad} entries) — element stiffness bug or transposed "
            "DOF mapping")


def _check_symmetry(K):
    """Verify stiffness symmetry and return K.

    抽样核对若干行 — 组装错位 (索引 bug) 破坏所有行, 抽样足以捕获;
    全量 K−K.T 转置+相减的峰值内存翻倍在 10 万 DOF 级不可接受,
    0.5*(K+K.T) 对称化同样复制整个矩阵。Ke 级对称性由
    _check_local_symmetry 在 scatter 前全量承担 (第五轮外部审查:
    抽样会漏掉未抽中位置的非对称项), 此处保留为全局兜底。
    """
    n = K.shape[0]
    if K.nnz and not np.all(np.isfinite(K.data)):
        # 装配溢出 (微尺度几何 BᵀDB ~ E/L² 中间量 Inf) 曾透传到 splu,
        # 误报 "Factor is exactly singular"; NaN > tolerance 恒 False,
        # 对称检查也放行 — 显式拒绝 (外部审查, 2026-08-03)
        n_bad = int(np.count_nonzero(~np.isfinite(K.data)))
        raise ValueError(
            f"Assembled stiffness contains {n_bad} NaN/Inf entries — "
            "element stiffness overflow (check geometry scale / E / t)")
    n_samples = min(n, 256)
    sample = np.unique(np.linspace(0, n - 1, n_samples).astype(np.intp))
    sub = K[sample]                      # 行采样 (S, n)
    skew = sub - K.T[sample]             # K[j, sample[i]] − K[sample[i], j]
    sk_norm = np.linalg.norm(skew.data) if skew.nnz else 0.0
    kd_norm = np.linalg.norm(sub.data) if sub.nnz else 0.0
    rel = sk_norm / kd_norm if kd_norm > 0 else 0.0
    if rel > 1e-10:
        raise RuntimeError(
            "Assembled stiffness matrix is asymmetric "
            f"(relative skew = {rel:.3e} > 1e-10). "
            "This indicates a bug in element stiffness or assembly logic.")
    return K


def assemble_sparse(mesh):
    """Assemble the global matrix in COO form and return CSR."""
    mesh.build_connectivity()
    return assemble_sparse_vectorized(mesh)


def assemble_sparse_vectorized(
        mesh, batch_elements=ASSEMBLY_BATCH_ELEMENTS):
    """Vectorized COO scatter with bounded temporary memory.

    Local stiffness matrices are computed in batches of ``batch_elements``,
    then written into preallocated (rows, cols, data) triplets and converted
    once to CSR — O(nnz) instead of O(batches × nnz) accumulation, with
    peak memory ≈ final matrix + one batch (preallocation replaces the
    concat copy of all batches held simultaneously).
    """
    mesh.build_connectivity()
    ne = mesh.n_elements
    nd = mesh.n_dof
    dofs = mesh.element_dofs
    nldof = dofs.shape[1]

    if batch_elements is None or batch_elements <= 0:
        batch_elements = ne
    batch_elements = max(1, int(batch_elements))
    stiffness_batch = mesh.element_kernel.stiffness_batch
    try:
        accepts_slice = len(inspect.signature(stiffness_batch).parameters) >= 2
    except (TypeError, ValueError):
        accepts_slice = True
    full_Ke = None

    total = ne * nldof * nldof
    # 行/列索引由 element_dofs 广播而来 — 一次性生成 (保持 int32 省一半内存)。
    # 标准 COO 语义: data[i,j,k] = K_e[j,k] → K[rows, cols] = K[dofs[i,j], dofs[i,k]],
    # 即 rows 对应 j 维、cols 对应 k 维 (曾写反 — 内置刚度对称故数值不受影响,
    # 但第三方非对称内核会得到转置)。
    rows = np.broadcast_to(
        dofs[:, :, None], (ne, nldof, nldof)).reshape(-1)
    cols = np.broadcast_to(
        dofs[:, None, :], (ne, nldof, nldof)).reshape(-1)
    data = np.empty(total, dtype=float)
    off = 0
    for start in range(0, ne, batch_elements):
        stop = min(start + batch_elements, ne)
        element_slice = slice(start, stop)
        count = stop - start
        if accepts_slice:
            Ke = np.asarray(
                stiffness_batch(mesh, element_slice),
                dtype=float,
            )
        else:
            # Backward compatibility for third-party kernels implementing the
            # original ``stiffness_batch(mesh)`` protocol.
            if full_Ke is None:
                full_Ke = np.asarray(stiffness_batch(mesh), dtype=float)
            Ke = full_Ke[element_slice]
        expected = (count, nldof, nldof)
        if Ke.shape != expected:
            raise RuntimeError(
                f"{mesh.element_kernel.name} kernel returned stiffness shape "
                f"{Ke.shape}; expected {expected}.")
        _check_local_symmetry(Ke, getattr(mesh.element_kernel, 'name', 'unknown'))
        block = count * nldof * nldof
        data[off:off + block] = Ke.ravel()
        off += block
    if total:
        K = coo_matrix((data, (rows, cols)), shape=(nd, nd)).tocsr()
    else:
        K = csr_matrix((nd, nd), dtype=float)
    K.sum_duplicates()
    K.eliminate_zeros()
    return _check_symmetry(K)


def assemble_lil_reference(mesh):
    """Teaching/reference assembly using scalar LIL accumulation.

    ⚠️ 参考实现 (验证性冗余, 外部审查 2026-08-03): LIL 逐单元组装,
    O(n_elem × nldof²) 时间、逐单元 Python 循环 — 仅用于与稀疏/向量化
    路径交叉验证, 生产用 assemble_sparse / assemble_sparse_vectorized。
    保留在 fem2d 内供 tests/benchmark 直接 import (公共导出契约)。
    """
    mesh.build_connectivity()
    K_lil = lil_matrix((mesh.n_dof, mesh.n_dof))
    for eid in range(mesh.n_elements):
        Ke = np.asarray(mesh.element_kernel.stiffness(mesh, eid))
        dofs = mesh.element_dofs[eid]
        expected = (len(dofs), len(dofs))
        if Ke.shape != expected:
            raise RuntimeError(
                f"{mesh.element_kernel.name} kernel returned local stiffness "
                f"shape {Ke.shape}; expected {expected}.")
        _check_local_symmetry(Ke[None], getattr(mesh.element_kernel, 'name', 'unknown'))
        for p in range(len(dofs)):
            for q in range(len(dofs)):
                K_lil[dofs[p], dofs[q]] += Ke[p, q]
    return _check_symmetry(K_lil.tocsr())


def assemble_expand(mesh):
    """Teaching/reference full-matrix expansion ``K = sum(L.T Ke L)``.

    ⚠️ Reference only — O(n_dof²) 内存 (稠密矩阵), 生产路径请用
    assemble_sparse / assemble_sparse_vectorized.
    """
    mesh.build_connectivity()
    K = np.zeros((mesh.n_dof, mesh.n_dof))
    for eid in range(mesh.n_elements):
        Ke = np.asarray(mesh.element_kernel.stiffness(mesh, eid))
        dofs = mesh.element_dofs[eid]
        expected = (len(dofs), len(dofs))
        if Ke.shape != expected:
            raise RuntimeError(
                f"{mesh.element_kernel.name} kernel returned local stiffness "
                f"shape {Ke.shape}; expected {expected}.")
        _check_local_symmetry(Ke[None], getattr(mesh.element_kernel, 'name', 'unknown'))
        K[np.ix_(dofs, dofs)] += Ke
    return K


