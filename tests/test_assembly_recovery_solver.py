"""Assembly-pattern caching, recovery vectorization and solver options.

Every fast path added here has a slow reference path in the package; the tests
compare the two rather than trusting recorded numbers.  ``test_repeated_solve``
covers a defect that the cached CSR pattern introduced during development: the
returned matrix shared its index arrays with the cache, so an in-place
``eliminate_zeros`` on the result silently corrupted every later assembly.
"""
import numpy as np
import pytest

from fem2d import Mesh
from fem2d.assembly import (
    assemble_expand,
    assemble_lil_reference,
    assemble_sparse,
    assemble_sparse_vectorized,
)
from fem2d.error_est import estimate
from fem2d.solver import solve
from fem2d.spr import spr_recovery
from fem2d.stress import nodal_L2_projection, nodal_simple, nodal_weighted


def _distorted_mesh(nx=6, ny=4, elem_type="CPS4", seed=7):
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 2.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    interior = ((nodes[:, 0] > 1e-9) & (nodes[:, 0] < 2.0 - 1e-9)
                & (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1.0 - 1e-9))
    nodes[interior] += rng.uniform(
        -0.04, 0.04, size=(int(interior.sum()), 2))

    def idx(i, j):
        return i * (ny + 1) + j

    quads = np.array(
        [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
         for i in range(nx) for j in range(ny)], dtype=np.int64)
    elements = (np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
                if elem_type == "CPS3" else quads)
    mesh = Mesh(nodes=nodes, elements=elements, E=2.1e11, nu=0.3,
                thickness=0.01, plane_type="stress", elem_type=elem_type)
    for node in mesh.nodes_on_edge("x", "min"):
        mesh.fix_node(int(node), "both")
    right = mesh.nodes_on_edge("x", "max")
    for node in right:
        mesh.add_force(int(node), 0.0, -1.0e3 / len(right))
    mesh.body_force = (0.0, -7850.0 * 9.81)
    return mesh


def test_reference_assemblies_not_reexported_at_top_level():
    """验证性参考装配只从 fem2d.assembly 导入 — 顶层导出已收敛.

    生产路径 (assemble_sparse/assemble_sparse_vectorized) 仍保留顶层;
    lil_reference/expand 为验证冗余, 实现保留在 assembly 模块.
    """
    import fem2d
    assert not hasattr(fem2d, "assemble_lil_reference")
    assert not hasattr(fem2d, "assemble_expand")
    assert hasattr(fem2d, "assemble_sparse")
    assert hasattr(fem2d, "assemble_sparse_vectorized")
    from fem2d.assembly import assemble_expand, assemble_lil_reference
    assert callable(assemble_expand) and callable(assemble_lil_reference)


def test_cached_pattern_matches_reference_assemblies():
    for elem_type in ("CPS3", "CPS4", "CPS4I", "CPS4R"):
        mesh = _distorted_mesh(elem_type=elem_type)
        fast = assemble_sparse(mesh).toarray()
        dense = assemble_expand(mesh)
        lil = assemble_lil_reference(mesh).toarray()
        scale = np.abs(dense).max()
        assert np.allclose(fast, dense, rtol=1e-10, atol=1e-9 * scale)
        assert np.allclose(fast, lil, rtol=1e-10, atol=1e-9 * scale)
        # Symmetry must survive the scatter.
        assert np.allclose(fast, fast.T, atol=1e-9 * scale)


def test_batched_and_pattern_assembly_agree():
    mesh = _distorted_mesh(nx=8, ny=5)
    pattern_based = assemble_sparse_vectorized(mesh).toarray()
    batched = assemble_sparse_vectorized(mesh, batch_elements=3).toarray()
    scale = np.abs(pattern_based).max()
    assert np.allclose(pattern_based, batched, rtol=1e-10, atol=1e-9 * scale)


def test_repeated_assembly_and_solve_are_stable():
    """Regression: the cached pattern must not be mutated by its consumers."""
    mesh = _distorted_mesh(nx=8, ny=5)
    first = assemble_sparse(mesh)
    for _ in range(3):
        again = assemble_sparse(mesh)
        assert again.shape == first.shape
        assert again.nnz == first.nnz
        assert np.allclose(again.toarray(), first.toarray())

    reference = solve(mesh, verbose=False)
    for _ in range(3):
        repeat = solve(mesh, verbose=False)
        assert np.allclose(repeat["u"], reference["u"], rtol=1e-12, atol=0.0)
        assert repeat["balance_ok"]


def test_pattern_is_rebuilt_after_mesh_change():
    mesh = _distorted_mesh(nx=4, ny=3)
    small = assemble_sparse(mesh)
    nodes = np.vstack([mesh.nodes, [[3.0, 0.0]]])
    bigger = Mesh(nodes=nodes, elements=mesh.elements, E=mesh.E, nu=mesh.nu,
                  thickness=mesh.thickness, elem_type=mesh.elem_type)
    grown = assemble_sparse(bigger)
    assert grown.shape[0] == small.shape[0] + 2


def _reference_boundary_rows(mesh):
    """SPR-BC-2026-001: 边界节点行替换为最近内部候选的 patch (naive).

    与生产 _boundary_patch_table 同规则: 候选 = 邻接单元顶点 − 边界集,
    为空扩 ring-1; 候选须 patch 非空; 并列取最小节点号; 无候选保留
    自身行。内部节点行原样。返回 list[list[int]]。
    """
    rows = [list(mesh.node_to_elems.ids(n)) for n in range(mesh.n_nodes)]
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


def _reference_spr(mesh, sample_xy, sample_values):
    """Deliberately naive per-node least-squares recovery.

    Mirrors the production exact path (``spr._fit_nodes_exact``) including
    its ring expansion: accept at ring 0 when the patch design matrix is
    full-rank and well-conditioned, otherwise expand the patch up to ring 3.
    Ring-0 patches with fewer than 3 samples (e.g. CST single-centroid
    sampling on corner/boundary nodes) therefore need the expansion to
    match the batched path's fallback.  Boundary nodes start from the
    enhanced patch rows (SPR-BC-2026-001, ZZ92 §2.3) — same table the
    production exact path receives.
    """
    n_comp = sample_values.shape[-1]
    recovered = np.zeros((mesh.n_nodes, n_comp))
    rows = _reference_boundary_rows(mesh)
    for nid in range(mesh.n_nodes):
        patch = set(rows[nid])
        frontier = set(patch)
        ring = 0
        while ring < 3:
            ids = sorted(patch)
            if ids:
                xy = sample_xy[ids].reshape(-1, 2)
                if len(xy) >= 3:
                    design = np.ones((len(xy), 3))
                    design[:, 1] = xy[:, 0] - mesh.nodes[nid, 0]
                    design[:, 2] = xy[:, 1] - mesh.nodes[nid, 1]
                    scale = max(
                        np.linalg.norm(design[:, 1:], axis=1).max(), np.finfo(float).tiny)
                    design[:, 1:] /= scale
                    if (np.linalg.matrix_rank(design) == 3
                            and np.linalg.cond(design) <= 1e8):
                        break
            new_frontier = set()
            for eid in frontier:
                for neighbor in mesh.elem_neighbors[eid]:
                    if neighbor not in patch:
                        new_frontier.add(neighbor)
            if not new_frontier:
                break
            patch.update(new_frontier)
            frontier = new_frontier
            ring += 1

        patch = sorted(patch)
        xy = sample_xy[patch].reshape(-1, 2)
        values = sample_values[patch].reshape(-1, n_comp)
        design = np.ones((len(xy), 3))
        design[:, 1] = xy[:, 0] - mesh.nodes[nid, 0]
        design[:, 2] = xy[:, 1] - mesh.nodes[nid, 1]
        scale = max(np.linalg.norm(design[:, 1:], axis=1).max(), np.finfo(float).tiny)
        design[:, 1:] /= scale
        if len(xy) < 3 or np.linalg.matrix_rank(design) < 3:
            recovered[nid] = values.mean(axis=0)
            continue
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        recovered[nid] = coefficients[0]
    return recovered


def test_spr_matches_naive_least_squares():
    from fem2d.spr import recovery_sample_positions

    for elem_type in ("CPS3", "CPS4", "CPS4I"):
        mesh = _distorted_mesh(elem_type=elem_type)
        result = solve(mesh, verbose=False)
        stress_qp = result["stress_qp"]
        positions = recovery_sample_positions(mesh, stress_qp.shape[1])
        fast = spr_recovery(mesh, stress_qp)
        slow = _reference_spr(mesh, positions, stress_qp)
        scale = np.abs(slow).max()
        assert np.allclose(fast, slow, rtol=1e-8, atol=1e-8 * scale), elem_type


def test_recovery_accepts_element_and_quadrature_input():
    mesh = _distorted_mesh()
    result = solve(mesh, verbose=False)
    for field in (result["stress"], result["stress_qp"]):
        recovered = spr_recovery(mesh, field)
        assert recovered.shape == (mesh.n_nodes, 3)
        assert np.all(np.isfinite(recovered))
    single = spr_recovery(mesh, result["vm_stress"])
    assert single.shape == (mesh.n_nodes, 1)


def test_l2_projection_matches_scalar_quadrature_path():
    """The batched projection must agree with the per-element protocol."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import splu

    mesh = _distorted_mesh()
    result = solve(mesh, verbose=False)
    stress_qp = result["stress_qp"]
    fast = nodal_L2_projection(mesh, stress_qp)

    n_nodes, n_comp = mesh.n_nodes, stress_qp.shape[-1]
    rows, cols, values = [], [], []
    rhs = np.zeros((n_nodes, n_comp))
    for eid, conn in enumerate(mesh.elements):
        N, dA = mesh.element_kernel.recovery_quadrature(mesh, eid)
        local = np.einsum("qi,qj,q->ij", N, N, dA)
        local_rhs = np.einsum("qi,qc,q->ic", N, stress_qp[eid], dA)
        for p, ni in enumerate(conn):
            for q, nj in enumerate(conn):
                rows.append(int(ni))
                cols.append(int(nj))
                values.append(local[p, q])
            rhs[ni] += local_rhs[p]
    mass = coo_matrix((values, (rows, cols)),
                      shape=(n_nodes, n_nodes)).tocsr()
    factor = splu(mass.tocsc())
    slow = np.column_stack(
        [factor.solve(rhs[:, c]) for c in range(n_comp)])
    assert np.allclose(fast, slow, rtol=1e-9, atol=1e-9 * np.abs(slow).max())


def test_nodal_averages_match_interpreted_reference():
    mesh = _distorted_mesh()
    result = solve(mesh, verbose=False)
    stress = result["stress"]

    reference = np.zeros((mesh.n_nodes, stress.shape[1]))
    counts = np.zeros(mesh.n_nodes)
    for eid, conn in enumerate(mesh.elements):
        for nid in conn:
            reference[nid] += stress[eid]
            counts[nid] += 1
    counts[counts == 0] = 1.0
    reference /= counts[:, None]
    assert np.allclose(nodal_simple(mesh, stress), reference)
    # Area weighting must stay distinct from arithmetic averaging.
    assert not np.allclose(nodal_weighted(mesh, stress), reference)


def test_error_estimate_is_kernel_driven_not_family_driven():
    """Q4I is a new kernel; the estimator must handle it without edits."""
    for elem_type in ("CPS3", "CPS4", "CPS4I"):
        mesh = _distorted_mesh(elem_type=elem_type)
        result = solve(mesh, verbose=False)
        report = estimate(mesh, result, verbose=False)
        assert np.isfinite(report["eta"]) and report["eta"] > 0.0
        assert report["elem_error"].shape == (mesh.n_elements,)
        assert report["elem_contrib"].sum() == pytest.approx(100.0, rel=1e-9)


def test_linear_solver_options_agree():
    mesh = _distorted_mesh(nx=12, ny=8)
    reference = solve(mesh, verbose=False, linear_solver="direct")
    for name in ("cg", "cg-block", "ilu"):
        result = solve(mesh, verbose=False, linear_solver=name)
        scale = np.abs(reference["u"]).max()
        assert np.allclose(result["u"], reference["u"],
                           rtol=0.0, atol=1e-8 * scale), name
        assert result["linear_solver"]["name"] == "cg"
    assert reference["linear_solver"]["name"] == "direct"


def test_unknown_linear_solver_is_rejected():
    mesh = _distorted_mesh(nx=3, ny=2)
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        solve(mesh, verbose=False, linear_solver="magic")
