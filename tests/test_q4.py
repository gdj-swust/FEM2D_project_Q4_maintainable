"""End-to-end tests for the full-integration Q4 element."""
import numpy as np

from fem2d import Mesh, solve


def _grid(nx, ny, x0, x1, y0, y1):
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


def _two_quad_mesh():
    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
    ])
    elements = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
    ])
    return Mesh(
        nodes, elements, E=210e9, nu=0.3, thickness=0.1,
        plane_type="stress", elem_type="CPS4",
    )


def _loaded_two_quad_mesh():
    mesh = _two_quad_mesh()
    for nid in (0, 3):
        mesh.fix_node(nid, "both", 0.0)
    mesh.add_traction(2, 5, 1.0e6, 0.0)
    return mesh


def test_q4_patch_tests_plane_stress_and_strain():
    from fem2d.patch_test import run_patch_test

    for elem_type, plane in (("CPS4", "stress"), ("CPE4", "strain")):
        report = run_patch_test(
            elem_type=elem_type, plane=plane, verbose=False)
        assert report["all_passed"]
        assert all(
            case["stress_qp_error"] < report["tol"]
            for case in report["tests"])


def test_q4_pure_bending_closed_form_tip_deflection():
    """积分点位置锁定 (R-ε E2): 单行 Q4 纯弯闭式解 (ν=0).

    v_tip/v_exact = 2n²/(2n²+(L/h)²); n=10, L/h=5 → 恰 8/9 (经典
    Q4 弯曲比值). 闭式对 n∈{1..30}, L/h∈{4,5,10} 在 1e-13 内成立;
    ν≠0 时共轭曲率改变比值 → 只锁 ν=0. 常数/仿射应变场在任何采样
    位置应变相同, 旧测试对积分点位置盲视; 纯弯是非仿射场 — 2×2
    积分点错位 (如 ±0.9) 破坏对二次被积函数的精确积分 → 弯曲刚度
    错误 → 比值偏离闭式 (实测 0.366 vs 0.889).
    """
    length, height, E, moment, nx = 10.0, 2.0, 1000.0, 1.0, 10
    inertia = height ** 3 / 12.0
    v_exact = moment * length ** 2 / (2.0 * E * inertia)

    nodes, quads = _grid(nx, 1, 0.0, length, -height / 2, height / 2)
    mesh = Mesh(nodes=nodes, elements=quads, E=E, nu=0.0, thickness=1.0,
                plane_type="stress", elem_type="CPS4")
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
    ratio = abs(result["u"][2 * tip + 1]) / v_exact

    expected = 2 * nx ** 2 / (2 * nx ** 2 + (length / height) ** 2)
    assert np.isclose(ratio, expected, rtol=1e-9)


def test_q4_affine_field_is_exact_at_all_gauss_points():
    from fem2d.material import D_matrix

    nodes = np.array([
        [0.0, 0.0], [1.2, -0.1], [1.1, 1.0], [-0.1, 0.9],
    ])
    mesh = Mesh(
        nodes, np.array([[0, 1, 2, 3]]),
        E=70e9, nu=0.25, thickness=0.2, elem_type="CPS4",
    )
    strain = np.array([1.0e-3, -2.0e-3, 0.5e-3])
    for nid, (x, y) in enumerate(nodes):
        mesh.fix_node(nid, "x", strain[0] * x + 0.5 * strain[2] * y)
        mesh.fix_node(nid, "y", 0.5 * strain[2] * x + strain[1] * y)

    result = solve(mesh, verbose=False)
    expected_stress = D_matrix(mesh.E, mesh.nu, "stress") @ strain
    assert np.allclose(result["strain_qp"][0], strain, rtol=1e-11, atol=1e-13)
    assert np.allclose(
        result["stress_qp"][0], expected_stress, rtol=1e-11, atol=1e-4)
    assert np.isclose(result["dA_qp"].sum(), mesh.areas[0])


def test_q4_end_to_end_equilibrium_and_response_shapes():
    mesh = _loaded_two_quad_mesh()
    result = solve(mesh, verbose=False)

    assert result["u"].shape == (mesh.n_dof,)
    assert result["stress"].shape == (mesh.n_elements, 3)
    assert result["stress_qp"].shape == (mesh.n_elements, 4, 3)
    assert result["strain_qp"].shape == (mesh.n_elements, 4, 3)
    assert result["dA_qp"].shape == (mesh.n_elements, 4)
    assert np.all(np.isfinite(result["u"]))
    assert np.linalg.norm(result["force_balance"]) < 1e-8
    assert abs(result["moment_balance"]) < 1e-8


def test_q4_body_force_and_sparse_assembly_are_consistent():
    from fem2d.assembly import assemble_lil_reference, assemble_sparse
    from fem2d.loads import assemble as assemble_loads

    mesh = _two_quad_mesh()
    mesh.body_force = (2.0, -3.0)
    force = assemble_loads(mesh, mesh.n_dof)
    volume = mesh.thickness * np.sum(mesh.areas)
    assert np.isclose(force[0::2].sum(), 2.0 * volume)
    assert np.isclose(force[1::2].sum(), -3.0 * volume)

    fast = assemble_sparse(mesh).toarray()
    reference = assemble_lil_reference(mesh).toarray()
    relative_difference = (
        np.linalg.norm(fast - reference) / np.linalg.norm(reference))
    assert relative_difference < 1e-14


def test_q4_recovery_z2_and_residual_indicator():
    from fem2d.error_est import element_refinement_indicator, estimate
    from fem2d.stress import nodal_L2_projection

    mesh = _loaded_two_quad_mesh()
    result = solve(mesh, verbose=False)
    nodal = nodal_L2_projection(mesh, result["stress_qp"])
    assert nodal.shape == (mesh.n_nodes, 3)
    assert np.all(np.isfinite(nodal))

    for method in ("L2", "SPR", "weighted"):
        z2 = estimate(mesh, result, method=method, verbose=False)
        assert np.isfinite(z2["eta"])
        assert np.all(np.isfinite(z2["elem_error"]))

    indicator = element_refinement_indicator(mesh, result)
    assert indicator.shape == (mesh.n_elements,)
    assert np.all(np.isfinite(indicator))


def test_q4_point_location_and_shape_interpolation():
    from fem2d.stress import point_in_element

    mesh = _two_quad_mesh()
    assert point_in_element(mesh, 0.25, 0.5) == 0
    assert point_in_element(mesh, 1.75, 0.5) == 1
    assert point_in_element(mesh, 3.0, 0.5) == -1


def test_q4_quality_uses_jacobian_samples():
    from fem2d.quality import evaluate

    good = _two_quad_mesh()
    quality = evaluate(good)
    assert quality["grade"] == "A"
    assert quality["jacobian_neg"] == 0
    assert quality["angle_min"] == 90.0

    nodes = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ])
    folded = Mesh(
        nodes, np.array([[0, 1, 3, 2]]), elem_type="CPS4")
    folded_quality = evaluate(folded)
    assert folded_quality["grade"] == "F"
    assert folded_quality["jacobian_neg"] > 0


def test_q4_visualization_paths():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from fem2d.visualize import plot_contour, plot_mesh, plot_three

    mesh = _loaded_two_quad_mesh()
    result = solve(mesh, verbose=False)

    fig, ax = plt.subplots()
    try:
        plot_mesh(mesh, ax=ax)
        assert len(ax.collections) + len(ax.lines) > 0, \
            "plot_mesh 未画出任何内容 (曾零断言)"
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    try:
        plot_contour(
            mesh, result["vm_stress"], ax=ax,
            shading="flat", location="element")
        assert len(ax.collections) > 0, \
            "plot_contour 未画出任何 collection (曾零断言)"
    finally:
        plt.close(fig)

    fig, ax = plt.subplots()
    plot_contour(
        mesh, result["u"][0::2], ax=ax,
        shading="gouraud", location="node")
    plt.close(fig)

    plot_three(mesh, result, tag="ux", scale=1.0)
    plt.close("all")


def test_q4_convergence_mesh_generator_returns_quads():
    from fem2d.convergence import _gen_cantilever_mesh, run_cantilever_convergence

    nodes, elements = _gen_cantilever_mesh(
        5.0, 1.0, 4, 2, elem_type="CPS4")
    assert nodes.shape == (15, 2)
    assert elements.shape == (8, 4)

    study = run_cantilever_convergence(
        refinements=3, verbose=False, elem_type="CPS4")
    assert study["eta"][-1] < study["eta"][0]
    assert 1.5 < study["uy_rate"] < 2.5
