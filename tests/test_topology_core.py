"""Tests for the vectorized topology kernel and the element locator.

These structures replaced interpreted per-element loops, so every test here
compares against a deliberately naive reference implementation rather than
against stored numbers.
"""
import numpy as np
import pytest

from fem2d import Mesh
from fem2d.topology_core import (
    CSRLists,
    EdgeIncidence,
    ElementLocator,
    build_edge_table,
    element_neighbor_table,
    node_element_table,
)


def _grid(nx, ny, quad=True, lx=2.0, ly=1.0):
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def idx(i, j):
        return i * (ny + 1) + j

    quads = np.array(
        [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
         for i in range(nx) for j in range(ny)], dtype=np.int64)
    if quad:
        return nodes, quads
    return nodes, np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def _reference_topology(elements, local_edges, n_nodes):
    """Straightforward interpreted reference for the vectorized tables."""
    node_to_elems = [[] for _ in range(n_nodes)]
    for eid, conn in enumerate(elements):
        for nid in conn:
            node_to_elems[int(nid)].append(eid)
    edge_to_elems = {}
    for eid, conn in enumerate(elements):
        for ia, ib in local_edges:
            a, b = int(conn[ia]), int(conn[ib])
            edge_to_elems.setdefault((min(a, b), max(a, b)), []).append(eid)
    boundary = sorted(k for k, v in edge_to_elems.items() if len(v) == 1)
    neighbors = [[] for _ in range(len(elements))]
    for (a, b), eids in edge_to_elems.items():
        if len(eids) == 2:
            neighbors[eids[0]].append(eids[1])
            neighbors[eids[1]].append(eids[0])
    return node_to_elems, edge_to_elems, boundary, neighbors


def test_topology_tables_match_interpreted_reference():
    for quad in (True, False):
        nodes, elements = _grid(5, 4, quad=quad)
        local = (((0, 1), (1, 2), (2, 3), (3, 0)) if quad
                 else ((0, 1), (1, 2), (2, 0)))
        n_nodes = nodes.shape[0]
        ref_n2e, ref_edges, ref_boundary, ref_neigh = _reference_topology(
            elements, local, n_nodes)

        n2e = node_element_table(elements, n_nodes)
        assert [sorted(row) for row in n2e] == [
            sorted(row) for row in ref_n2e]

        table = build_edge_table(elements, local, n_nodes)
        mapping = table.as_mapping()
        assert isinstance(mapping, EdgeIncidence)
        assert len(mapping) == len(ref_edges)
        for key, eids in ref_edges.items():
            assert key in mapping
            assert sorted(mapping[key]) == sorted(eids)
        boundary = sorted(
            map(tuple,
                np.column_stack([table.lo[table.boundary_mask()],
                                 table.hi[table.boundary_mask()]]).tolist()))
        assert boundary == ref_boundary

        neigh = element_neighbor_table(table, len(elements))
        assert [sorted(row) for row in neigh] == [
            sorted(row) for row in ref_neigh]


def test_edge_incidence_rejects_unknown_keys():
    nodes, elements = _grid(2, 2)
    table = build_edge_table(elements, ((0, 1), (1, 2), (2, 3), (3, 0)),
                             nodes.shape[0])
    mapping = table.as_mapping()
    assert (10 ** 6, 10 ** 6 + 1) not in mapping
    assert "not-an-edge" not in mapping
    with pytest.raises(KeyError):
        mapping[(10 ** 6, 10 ** 6 + 1)]
    # Reversed keys resolve to the same canonical edge.
    a, b = int(table.lo[0]), int(table.hi[0])
    assert mapping[(a, b)] == mapping[(b, a)]
    assert dict(mapping.items()) == table.as_dict()


def test_csr_lists_behaves_like_list_of_lists():
    lists = CSRLists(np.array([0, 2, 2, 5]), np.array([7, 8, 1, 2, 3]))
    assert len(lists) == 3
    assert lists[0] == [7, 8]
    assert lists[1] == []
    assert lists[2] == [1, 2, 3]
    assert lists[-1] == [1, 2, 3]
    assert list(lists) == [[7, 8], [], [1, 2, 3]]
    assert lists == [[7, 8], [], [1, 2, 3]]
    assert lists != [[7, 8], [], [1, 2]]
    assert np.array_equal(lists.ids(2), np.array([1, 2, 3]))
    with pytest.raises(IndexError):
        lists.ids(5)


def test_non_manifold_edge_is_reported():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, -1.0],
                      [1.5, 0.5]])
    elements = np.array([[0, 1, 2], [0, 3, 1], [1, 4, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="Non-manifold"):
        build_edge_table(elements, ((0, 1), (1, 2), (2, 0)), nodes.shape[0])


def test_locator_keeps_two_by_two_bucket_invariant():
    """Every element box must touch at most 2x2 cells, or queries can miss."""
    rng = np.random.default_rng(17)
    for quad in (True, False):
        for shape in ((1, 1), (7, 3), (24, 12), (40, 17)):
            nodes, elements = _grid(*shape, quad=quad)
            locator = ElementLocator(nodes, elements)
            assert locator.max_bucket_span() <= 4

        nodes, elements = _grid(8, 6, quad=quad)
        interior = ((nodes[:, 0] > 1e-9) & (nodes[:, 0] < 2.0 - 1e-9)
                    & (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1.0 - 1e-9))
        jittered = nodes.copy()
        jittered[interior] += rng.uniform(
            -0.05, 0.05, size=(int(interior.sum()), 2))
        assert ElementLocator(jittered, elements).max_bucket_span() <= 4


def test_point_location_matches_exhaustive_scan():
    rng = np.random.default_rng(23)
    probes = np.vstack([
        rng.uniform([0.0, 0.0], [2.0, 1.0], size=(300, 2)),
        np.array([[1.3, 0.27], [0.0, 0.0], [2.0, 1.0], [1.0, 0.5],
                  [2.5, 0.5], [-0.1, 0.2]]),
    ])
    for quad in (True, False):
        nodes, elements = _grid(12, 7, quad=quad)
        mesh = Mesh(nodes=nodes, elements=elements, E=1.0, nu=0.3,
                    elem_type="CPS4" if quad else "CPS3")
        mesh.build_connectivity()
        kernel = mesh.element_kernel
        tol = 1e-12 * 2.0
        for x, y in probes:
            fast = kernel.find_containing_element(mesh, x, y)
            slow = -1
            for eid, conn in enumerate(mesh.elements):
                if kernel.shape_values_at(
                        mesh.nodes[conn], x, y, tol) is not None:
                    slow = eid
                    break
            assert fast == slow, (x, y, fast, slow)


def test_locator_cache_is_invalidated_with_topology():
    nodes, elements = _grid(3, 3)
    mesh = Mesh(nodes=nodes, elements=elements, E=1.0, nu=0.3,
                elem_type="CPS4")
    mesh.build_connectivity()
    first = mesh.locator
    assert mesh.locator is first
    mesh.invalidate_cache()
    mesh.build_connectivity()
    assert mesh.locator is not first


def test_locator_zero_span_axis_micro_scale():
    """零跨度轴 (共线退化网格) 的定位器不得崩溃 — extent 曾回落绝对
    1.0, 微尺度网格分桶随 1.0 发散 (审计 2026-08-03)."""
    from fem2d.topology_core import ElementLocator
    s = 1e-16
    nodes = np.array([[0.0, 0.0], [s, 0.0], [2 * s, 0.0]], dtype=float)
    elems = np.array([[0, 1, 2]])   # 共线退化 (y 跨度 = 0)
    loc = ElementLocator(nodes, elems)
    # 定位不崩溃且返回合法候选 (退化网格候选集不保证命中, 只验证无异常)
    loc.candidates(s, 0.0)
    loc.candidates(0.0, 0.0)
    # 非退化微尺度网格必须精确定位
    nodes2 = np.array([[0.0, 0.0], [s, 0.0], [0.0, s]], dtype=float)
    loc2 = ElementLocator(nodes2, np.array([[0, 1, 2]]))
    assert 0 in loc2.candidates(s * 0.5, s * 0.5)
