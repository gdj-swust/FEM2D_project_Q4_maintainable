"""Q4I (QM6 incompatible-mode quadrilateral) verification tests.

The element is only worth shipping if it keeps every property Q4 already has
(patch test, rigid-body modes, exactly three global zero energy modes) while
removing shear locking.  Each of those claims is asserted here.
"""
import numpy as np
import pytest

from fem2d import Mesh
from fem2d.assembly import assemble_expand, assemble_sparse
from fem2d.element import get_element_kernel, registered_element_types
from fem2d.element.q4i import element_stiffness
from fem2d.patch_test import run_patch_test
from fem2d.solver import solve


def _grid(nx, ny, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    xs = np.linspace(x0, x1, nx + 1)
    ys = np.linspace(y0, y1, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def idx(i, j):
        return i * (ny + 1) + j

    quads = np.array(
        [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
         for i in range(nx) for j in range(ny)], dtype=np.int64)
    return nodes, quads


def _cook_mesh(n, elem_type="CPS4I"):
    """Cook's skew membrane: (0,0),(48,44),(48,60),(0,44)."""
    corners = np.array([[0.0, 0.0], [48.0, 44.0], [48.0, 60.0], [0.0, 44.0]])
    u = np.linspace(0.0, 1.0, n + 1)
    U, V = np.meshgrid(u, u, indexing="ij")
    shape = np.stack(
        [(1 - U) * (1 - V), U * (1 - V), U * V, (1 - U) * V], axis=-1)
    nodes = (shape @ corners).reshape(-1, 2)
    _, quads = _grid(n, n)
    mesh = Mesh(nodes=nodes, elements=quads, E=1.0, nu=1.0 / 3.0,
                thickness=1.0, plane_type="stress", elem_type=elem_type)
    for node in np.flatnonzero(np.abs(nodes[:, 0]) < 1e-9):
        mesh.fix_node(int(node), "both")
    right = np.flatnonzero(np.abs(nodes[:, 0] - 48.0) < 1e-9)
    order = right[np.argsort(nodes[right, 1])]
    for a, b in zip(order[:-1], order[1:]):
        mesh.add_traction(int(a), int(b), 0.0, 1.0 / 16.0)
    return mesh, nodes


def test_q4i_is_registered_with_abaqus_aliases():
    registered = set(registered_element_types())
    assert {"Q4I", "CPS4I", "CPE4I"} <= registered
    kernel = get_element_kernel("CPS4I")
    assert kernel is get_element_kernel("Q4I")
    assert kernel.nodes_per_element == 4
    assert kernel.dofs_per_element == 8
    # Same edge topology as Q4 -> boundary/traction code needs no changes.
    assert kernel.local_edges == get_element_kernel("CPS4").local_edges


def test_q4i_codes_resolve_the_correct_plane():
    """Abaqus 单元码 → 平面态判型 (CPS4I=stress, CPE4I=strain).

    原 .inp 导入路径 (read_inp) 已移除 — 单元码的注册由
    test_q4i_is_registered_with_abaqus_aliases 覆盖, 这里只验证
    cli 的平面态解析逻辑 (生产路径仍由它决定 .geo/.msh 算例的平面态).
    """
    from fem2d.cli import _resolve_plane_type

    for element_type, expected_plane in (
            ("CPS4I", "stress"), ("CPE4I", "strain")):
        assert _resolve_plane_type(element_type) == expected_plane

    with pytest.raises(ValueError, match="plane-strain"):
        _resolve_plane_type("CPE4I", "stress")
    with pytest.raises(ValueError, match="plane-stress"):
        _resolve_plane_type("CPS4I", "strain")


def test_q4i_passes_constant_stress_patch_test():
    """The detJ0/detJ orthogonality scaling is what makes this pass."""
    for plane in ("stress", "strain"):
        report = run_patch_test(verbose=False, plane=plane,
                                elem_type="CPS4I")
        assert report["all_passed"], (
            f"Q4I patch test failed for plane {plane}")


def test_q4i_reproduces_pure_bending_exactly():
    """Q4 locks at 8/9 of the exact tip deflection; QM6 must be exact."""
    length, height, E = 10.0, 2.0, 1000.0
    moment = 1.0
    inertia = height ** 3 / 12.0
    exact = moment * length ** 2 / (2.0 * E * inertia)

    results = {}
    for elem_type in ("CPS4", "CPS4I"):
        nodes, quads = _grid(10, 1, 0.0, length, -height / 2, height / 2)
        mesh = Mesh(nodes=nodes, elements=quads, E=E, nu=0.0, thickness=1.0,
                    plane_type="stress", elem_type=elem_type)
        left = np.flatnonzero(np.abs(nodes[:, 0]) < 1e-9)
        for node in left:
            mesh.fix_node(int(node), "x")
        mesh.fix_node(int(left[np.argmin(np.abs(nodes[left, 1]))]), "y")
        right = np.flatnonzero(np.abs(nodes[:, 0] - length) < 1e-9)
        order = right[np.argsort(nodes[right, 1])]
        for a, b in zip(order[:-1], order[1:]):
            mesh.add_traction(int(a), int(b),
                              lambda x, y: moment * y / inertia, 0.0)
        result = solve(mesh, verbose=False)
        tip = order[np.argmin(np.abs(nodes[order, 1]))]
        results[elem_type] = abs(result["u"][2 * tip + 1])

    assert results["CPS4I"] == pytest.approx(exact, rel=1e-10)
    # Q4 shear locking with one element through the thickness.
    assert 0.85 < results["CPS4"] / exact < 0.90


def test_q4i_converges_faster_than_q4_on_cooks_membrane():
    reference = 23.9642
    errors = {}
    for elem_type in ("CPS4", "CPS4I"):
        mesh, nodes = _cook_mesh(4, elem_type)
        result = solve(mesh, verbose=False)
        tip = int(np.argmin(np.abs(nodes[:, 0] - 48.0)
                            + np.abs(nodes[:, 1] - 52.0)))
        value = result["u"][2 * tip + 1]
        errors[elem_type] = abs(value - reference) / reference

    assert errors["CPS4I"] < 0.05, errors
    # Roughly a five-fold error reduction on the same 32-element mesh.
    assert errors["CPS4I"] < errors["CPS4"] / 4.0, errors


def test_q4i_has_exactly_three_zero_energy_modes_on_distorted_mesh():
    rng = np.random.default_rng(3)
    nodes, quads = _grid(4, 3)
    interior = ((nodes[:, 0] > 1e-9) & (nodes[:, 0] < 1.0 - 1e-9)
                & (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1.0 - 1e-9))
    perturbed = nodes.copy()
    perturbed[interior] += rng.uniform(-0.05, 0.05,
                                       size=(int(interior.sum()), 2))
    mesh = Mesh(nodes=perturbed, elements=quads, E=1000.0, nu=0.3,
                thickness=1.0, elem_type="CPS4I")
    K = assemble_sparse(mesh).toarray()
    eigenvalues = np.linalg.eigvalsh(K)
    zeros = int(np.sum(eigenvalues < 1e-8 * eigenvalues.max()))
    assert zeros == 3, f"expected 3 rigid-body modes, found {zeros}"


def test_q4i_scalar_and_batched_stiffness_agree():
    rng = np.random.default_rng(11)
    nodes, quads = _grid(3, 3)
    perturbed = nodes.copy()
    interior = ((nodes[:, 0] > 1e-9) & (nodes[:, 0] < 1.0 - 1e-9)
                & (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1.0 - 1e-9))
    perturbed[interior] += rng.uniform(-0.06, 0.06,
                                       size=(int(interior.sum()), 2))
    mesh = Mesh(nodes=perturbed, elements=quads, E=2.1e11, nu=0.3,
                thickness=0.02, elem_type="CPS4I")
    mesh.build_connectivity()
    batched = mesh.element_kernel.stiffness_batch(mesh)
    for eid, conn in enumerate(mesh.elements):
        scalar = element_stiffness(
            mesh.nodes[conn], mesh.E, mesh.nu, mesh.thickness,
            mesh.plane_type)
        assert np.allclose(scalar, batched[eid], rtol=1e-11, atol=0.0)
        assert np.allclose(scalar, scalar.T, atol=1e-9 * np.abs(scalar).max())

    # The condensed element must still assemble identically through every
    # assembly backend.
    dense = assemble_expand(mesh)
    sparse = assemble_sparse(mesh).toarray()
    assert np.allclose(dense, sparse, rtol=1e-10,
                       atol=1e-8 * np.abs(dense).max())


def test_q4i_enhanced_amplitudes_vanish_for_linear_field():
    """A linear displacement field must not excite the bubble modes."""
    rng = np.random.default_rng(5)
    nodes, quads = _grid(3, 2)
    perturbed = nodes.copy()
    interior = ((nodes[:, 0] > 1e-9) & (nodes[:, 0] < 1.0 - 1e-9)
                & (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1.0 - 1e-9))
    perturbed[interior] += rng.uniform(-0.07, 0.07,
                                       size=(int(interior.sum()), 2))
    mesh = Mesh(nodes=perturbed, elements=quads, E=1.0, nu=0.25,
                thickness=1.0, elem_type="CPS4I")
    mesh.build_connectivity()

    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    u = np.empty(mesh.n_dof)
    u[0::2] = 0.3 + 0.11 * x - 0.04 * y
    u[1::2] = -0.2 + 0.07 * x + 0.09 * y
    u_e = u[mesh.element_dofs]

    alpha = mesh.element_kernel.enhanced_amplitudes(mesh, u_e)
    assert np.max(np.abs(alpha)) < 1e-12

    stress, strain, _ = mesh.element_kernel.compute_response(mesh, u_e)
    assert np.allclose(strain[:, 0], 0.11, atol=1e-12)
    assert np.allclose(strain[:, 1], 0.09, atol=1e-12)
    assert np.allclose(strain[:, 2], 0.03, atol=1e-12)
