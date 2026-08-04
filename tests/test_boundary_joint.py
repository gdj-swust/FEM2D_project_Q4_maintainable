"""Joint Gmsh-semantic and mesh-geometric boundary segmentation tests."""

import numpy as np
import pytest

from fem2d import Mesh
from fem2d.boundary import (
    BoundaryDiagnostics,
    _resolve_edge_indices,
    build_boundary_segments,
    validate_boundary_segments,
)
from fem2d.boundary.geometry import classify
from fem2d.boundary.topology import (
    _loop_probe_point,
    _point_in_loop,
    has_boundary_self_intersection,
)
from fem2d.regions import (
    CadCurveRegion,
    CurveRegion,
    RegionRegistry,
    canonical_edge,
    ordered_edge_chains,
)


def _strip_mesh():
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])
    elements = np.array([
        [0, 1, 4],
        [0, 4, 5],
        [1, 2, 3],
        [1, 3, 4],
    ])
    return Mesh(
        nodes=nodes,
        elements=elements,
        E=1.0,
        nu=0.25,
        thickness=1.0,
        elem_type="CPS3",
    )


def _curve(
        name, physical_tag, records, entity_types=None):
    """Build one CurveRegion from (a, b, entity_tag, entity_type) records."""
    records = tuple(
        (int(a), int(b), int(tag), str(kind))
        for a, b, tag, kind in records
    )
    edges = tuple(sorted({
        canonical_edge(a, b) for a, b, _, _ in records
    }))
    tags = tuple(sorted({tag for _, _, tag, _ in records}))
    kinds = (
        tuple(entity_types)
        if entity_types is not None
        else tuple(sorted({kind for _, _, _, kind in records}))
    )
    return CurveRegion(
        name=name,
        physical_tag=physical_tag,
        entity_tags=tags,
        entity_types=kinds,
        node_ids=tuple(sorted({
            node for edge in edges for node in edge
        })),
        edge_pairs=edges,
        edge_entities=records,
    )


def _selected_edges(segments, name):
    result = set()
    for index in _resolve_edge_indices(name, segments):
        nodes = segments[index]["nodes"]
        result.update(
            canonical_edge(a, b)
            for a, b in zip(nodes, nodes[1:])
        )
    return result


def test_membership_changes_split_one_geometric_line_exactly():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("load", 101, [(0, 1, 11, "Line")]),
        _curve("load_aux", 102, [(1, 2, 12, "Line")]),
        _curve("bottom_all", 103, [
            (0, 1, 11, "Line"),
            (1, 2, 12, "Line"),
        ]),
    ])

    segments = build_boundary_segments(mesh, registry=registry)

    assert validate_boundary_segments(mesh, segments)
    assert _selected_edges(segments, "load") == {(0, 1)}
    assert _selected_edges(segments, "load_aux") == {(1, 2)}
    assert _selected_edges(segments, "bottom_all") == {(0, 1), (1, 2)}


def test_cad_entity_transition_is_a_hard_split():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("bottom_all", 103, [
            (0, 1, 11, "Line"),
            (1, 2, 12, "Line"),
        ]),
    ])

    segments = build_boundary_segments(mesh, registry=registry)
    selected = [
        segments[index]
        for index in _resolve_edge_indices("bottom_all", segments)
    ]

    assert len(selected) == 2
    assert {
        segment["info"]["entity_tags"] for segment in selected
    } == {(11,), (12,)}


def test_short_name_never_expands_multiple_physical_groups():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("load_a", 101, [(0, 1, 11, "Line")]),
        _curve("load_b", 102, [(1, 2, 12, "Line")]),
    ])

    segments = build_boundary_segments(mesh, registry=registry)

    assert _resolve_edge_indices("load", segments) == []
    assert _resolve_edge_indices("load_a", segments)
    assert _resolve_edge_indices("load_b", segments)


def test_explicit_fuzzy_name_reports_ambiguity():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("load_a", 101, [(0, 1, 11, "Line")]),
        _curve("load_b", 102, [(1, 2, 12, "Line")]),
    ])
    segments = build_boundary_segments(mesh, registry=registry)

    try:
        _resolve_edge_indices("~load", segments)
    except ValueError as error:
        assert "多个候选" in str(error)
        assert "load_a" in str(error)
        assert "load_b" in str(error)
    else:
        raise AssertionError("Expected an ambiguous fuzzy selector to fail")


def test_unlabelled_remainder_is_supplied_without_expanding_named_region():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("fixed_half", 101, [(0, 1, 11, "Line")]),
    ])

    segments = build_boundary_segments(mesh, registry=registry)

    assert validate_boundary_segments(mesh, segments)
    assert _selected_edges(segments, "fixed_half") == {(0, 1)}
    assert any(
        canonical_edge(a, b) == (1, 2)
        for segment in segments
        if "fixed_half" not in segment["info"].get("physical_names", ())
        for a, b in zip(segment["nodes"], segment["nodes"][1:])
    )


def test_disconnected_components_with_one_name_remain_separate():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("supports", 101, [
            (0, 5, 14, "Line"),
            (2, 3, 12, "Line"),
        ]),
    ])

    segments = build_boundary_segments(mesh, registry=registry)
    selected = _resolve_edge_indices("supports", segments)

    assert validate_boundary_segments(mesh, segments)
    assert len(selected) == 2
    assert _selected_edges(segments, "supports") == {(0, 5), (2, 3)}


def test_per_edge_cad_metadata_does_not_override_geometry():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("mixed", 101, [
            (0, 1, 11, "Line"),
            (1, 2, 12, "Line"),
            (2, 3, 13, "Circle"),
        ], entity_types=("Line", "Circle")),
    ])

    segments = build_boundary_segments(mesh, registry=registry)
    selected = [
        segments[index] for index in _resolve_edge_indices("mixed", segments)
    ]

    bottom = next(
        segment for segment in selected
        if canonical_edge(
            segment["nodes"][0], segment["nodes"][1]) in {(0, 1), (1, 2)}
    )
    right = next(
        segment for segment in selected
        if {
            canonical_edge(a, b)
            for a, b in zip(segment["nodes"], segment["nodes"][1:])
        } == {(2, 3)}
    )
    assert bottom["type"] == "line"
    assert bottom["info"]["cad_entity_types"] == ("Line",)
    assert right["type"] == "line"
    assert right["info"]["cad_entity_types"] == ("Circle",)


def test_unlabelled_cad_entities_still_control_segmentation():
    mesh = _strip_mesh()
    registry = RegionRegistry(cad_curves=[
        CadCurveRegion(
            entity_tag=11,
            entity_type="Line",
            node_ids=(0, 1),
            edge_pairs=((0, 1),),
            surface_occurrences=((1, 11),),
        ),
        CadCurveRegion(
            entity_tag=12,
            entity_type="Line",
            node_ids=(1, 2),
            edge_pairs=((1, 2),),
            surface_occurrences=((1, 12),),
        ),
    ])

    segments = build_boundary_segments(mesh, registry=registry)
    cad_segments = [
        segment for segment in segments
        if segment["info"].get("entity_tags")
    ]

    assert validate_boundary_segments(mesh, segments)
    assert {
        segment["info"]["entity_tags"] for segment in cad_segments
    } == {(11,), (12,)}


def test_multi_surface_internal_cad_interface_is_not_an_external_boundary():
    mesh = _strip_mesh()
    registry = RegionRegistry(cad_curves=[
        CadCurveRegion(
            entity_tag=11, entity_type="Line",
            node_ids=(0, 1, 2),
            edge_pairs=((0, 1), (1, 2)),
            surface_occurrences=((1, 11),),
        ),
        CadCurveRegion(
            entity_tag=12, entity_type="Line",
            node_ids=(2, 3), edge_pairs=((2, 3),),
            surface_occurrences=((2, 12),),
        ),
        CadCurveRegion(
            entity_tag=90, entity_type="Line",
            node_ids=(1, 4), edge_pairs=((1, 4),),
            surface_occurrences=((1, 90), (2, -90)),
        ),
    ])
    diagnostics = BoundaryDiagnostics()

    segments = build_boundary_segments(
        mesh, registry=registry,
        diagnostics=diagnostics, strict=True)

    assert validate_boundary_segments(mesh, segments)
    assert not diagnostics.errors
    assert all(
        90 not in segment["info"].get("entity_tags", ())
        for segment in segments)


def test_internal_physical_curve_is_reported_and_strict_mode_rejects_it():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("internal_load", 111, [(1, 4, 91, "Line")]),
    ])
    diagnostics = BoundaryDiagnostics()

    segments = build_boundary_segments(
        mesh, registry=registry, diagnostics=diagnostics)

    assert validate_boundary_segments(mesh, segments)
    assert diagnostics.dropped_physical_names == ("internal_load",)
    assert diagnostics.errors[0].code == "physical_curve_internal"
    try:
        build_boundary_segments(
            mesh, registry=registry,
            diagnostics=BoundaryDiagnostics(), strict=True)
    except ValueError as error:
        assert "internal_load" in str(error)
    else:
        raise AssertionError("Strict boundary mode accepted an internal group")


def test_edge_labels_fallback_rejects_internal_and_missing_physical_edges():
    mesh = _strip_mesh()
    diagnostics = BoundaryDiagnostics()
    labels = {
        (0, 1): frozenset({"valid", "partly_internal"}),
        (1, 4): frozenset({"internal", "partly_internal"}),
        (0, 99): frozenset({"missing"}),
    }

    try:
        build_boundary_segments(
            mesh, edge_labels=labels,
            diagnostics=diagnostics, strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "edge-labels fallback strict mode accepted invalid Physical Curves")

    codes = {issue.code for issue in diagnostics.errors}
    assert "physical_curve_internal" in codes
    assert "physical_curve_partly_internal" in codes
    assert "physical_curve_missing_edges" in codes
    assert set(diagnostics.dropped_physical_names) == {
        "internal", "missing", "partly_internal",
    }


def test_orphan_physical_curve_survives_import_as_diagnostic():
    """孤儿 Physical Curve 记录必须作为诊断报告.

    孤儿记录 (边的节点不属于位移网格) 经 unmapped_records 传入
    (原 EdgeLabelMap 语义, 已随 read_inp 移除 — 此处用带同名属性的
    轻量 dict 等价构造, 验证边界构建不静默丢弃它们).
    """
    class _LabelMap(dict):
        def __init__(self, *args, unmapped_records=(), **kwargs):
            super().__init__(*args, **kwargs)
            self.unmapped_records = tuple(unmapped_records)

    mesh = _strip_mesh()
    labels = _LabelMap(unmapped_records=(
        {"original_edge": (5, 6), "names": ("orphan_load",),
         "reason": "undefined_node"},
    ))
    diagnostics = BoundaryDiagnostics()
    try:
        build_boundary_segments(
            mesh, edge_labels=labels,
            diagnostics=diagnostics, strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Orphan Physical Curve disappeared without strict failure")
    assert diagnostics.dropped_physical_names == ("orphan_load",)
    assert {
        issue.code for issue in diagnostics.errors
    } == {"physical_curve_unmapped_nodes"}


def test_case_colliding_and_cli_unsafe_physical_names_are_rejected():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("Load", 101, [(0, 1, 11, "Line")]),
        _curve("load", 102, [(1, 2, 12, "Line")]),
        _curve("bad:name", 103, [(2, 3, 13, "Line")]),
    ])
    diagnostics = BoundaryDiagnostics()

    try:
        build_boundary_segments(
            mesh, registry=registry,
            diagnostics=diagnostics, strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous Physical Curve names were accepted")

    assert {
        issue.code for issue in diagnostics.errors
    } >= {
        "physical_name_case_collision",
        "physical_name_cli_delimiter",
    }


def test_numeric_physical_name_is_rejected_before_index_selection():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("1", 101, [(0, 1, 11, "Line")]),
    ])
    diagnostics = BoundaryDiagnostics()

    try:
        build_boundary_segments(
            mesh, registry=registry,
            diagnostics=diagnostics, strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Numeric Physical Curve name was accepted")

    assert {
        issue.code for issue in diagnostics.errors
    } == {"physical_name_numeric"}


def test_complete_cad_registry_requires_every_external_mesh_edge():
    mesh = _strip_mesh()
    registry = RegionRegistry(
        cad_curves=[
            CadCurveRegion(
                entity_tag=11, entity_type="Line",
                node_ids=(0, 1, 2),
                edge_pairs=((0, 1), (1, 2)),
                surface_occurrences=((1, 11),),
            ),
        ],
        cad_boundary_complete=True,
    )
    diagnostics = BoundaryDiagnostics()

    try:
        build_boundary_segments(
            mesh, registry=registry,
            diagnostics=diagnostics)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Incomplete complete-CAD coverage was accepted without strict mode")

    gap = next(
        issue for issue in diagnostics.errors
        if issue.code == "cad_boundary_coverage_gap")
    assert gap.edge_count == 4


def test_complete_cad_registry_detects_external_internal_mismatch():
    mesh = _strip_mesh()
    registry = RegionRegistry(
        cad_curves=[
            CadCurveRegion(
                entity_tag=90, entity_type="Line",
                node_ids=(1, 4), edge_pairs=((1, 4),),
                surface_occurrences=((1, 90),),
            ),
        ],
        cad_boundary_complete=True,
    )
    diagnostics = BoundaryDiagnostics()

    try:
        build_boundary_segments(
            mesh, registry=registry,
            diagnostics=diagnostics, strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("CAD/mesh boundary mismatch was accepted")

    assert {
        issue.code for issue in diagnostics.errors
    } >= {
        "cad_boundary_internal_mismatch",
        "cad_boundary_coverage_gap",
    }


def test_physical_group_reports_partially_unmeshed_cad_entities():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        CurveRegion(
            name="load",
            physical_tag=101,
            entity_tags=(11, 12),
            entity_types=("Line", "Spline"),
            node_ids=(0, 1),
            edge_pairs=((0, 1),),
            edge_entities=((0, 1, 11, "Line"),),
        ),
    ])
    diagnostics = BoundaryDiagnostics()

    try:
        build_boundary_segments(
            mesh, registry=registry,
            diagnostics=diagnostics, strict=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Partially unmeshed Physical Curve was accepted")

    assert diagnostics.dropped_physical_names == ("load",)
    assert {
        issue.code for issue in diagnostics.errors
    } == {"physical_curve_unmeshed_entity"}


def test_ordered_edge_chains_rejects_zero_length_topological_edge():
    try:
        ordered_edge_chains([(0, 1), (1, 1)])
    except ValueError as error:
        assert "zero-length topological edges" in str(error)
    else:
        raise AssertionError("Self-loop boundary edge was silently dropped")


def test_physical_name_text_cannot_change_topological_role():
    mesh = _strip_mesh()
    registry = RegionRegistry(curves=[
        _curve("内孔_load", 111, [(0, 1, 11, "Line")]),
    ])

    segments = build_boundary_segments(mesh, registry=registry)
    selected = segments[_resolve_edge_indices("内孔_load", segments)[0]]

    assert selected["info"]["is_outer"] is True
    assert _resolve_edge_indices("hole", segments) == []


def test_self_intersection_accepts_unique_vertices_and_finds_overlap():
    bow_tie = [(0.0, 0.0), (2.0, 2.0),
               (0.0, 2.0), (2.0, 0.0)]
    overlap = [(0.0, 0.0), (3.0, 0.0), (1.0, 0.0),
               (1.0, 2.0), (0.0, 2.0)]
    rectangle = [(0.0, 0.0), (2.0, 0.0),
                 (2.0, 1.0), (0.0, 1.0)]

    assert has_boundary_self_intersection(bow_tie)
    assert has_boundary_self_intersection(overlap)
    assert not has_boundary_self_intersection(rectangle)


def test_zero_area_boundary_loop_is_rejected_before_classification():
    from fem2d.boundary.topology import detect

    class _CollapsedMesh:
        nodes = np.array([
            [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        ])
        boundary_edges = [(0, 1), (1, 2), (2, 0)]

        def build_connectivity(self):
            return None

    try:
        detect(_CollapsedMesh())
    except ValueError as error:
        assert "degenerate enclosed area" in str(error)
    else:
        raise AssertionError("Zero-area boundary loop was accepted")


def test_nearly_closed_open_arc_is_not_promoted_to_full_circle():
    angles = np.deg2rad(np.linspace(10.0, 350.0, 80))
    coords = np.column_stack([np.cos(angles), np.sin(angles)])

    segment_type, label, info = classify(
        coords, scale=2.0, is_outer=True)

    assert segment_type == "arc"
    assert "整圆" not in label
    assert info["angle"] < 2.0 * np.pi


def test_large_unrepeated_loop_uses_explicit_topological_closure():
    center = np.array([1.0e9, -2.0e9])
    radius = 2.5e6
    angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    coords = center + radius * np.column_stack([
        np.cos(angles), np.sin(angles),
    ])

    segment_type, label, info = classify(
        coords, scale=2.0 * radius, is_outer=True, closed=True)

    assert segment_type == "arc"
    assert "整圆" in label
    assert np.isclose(info["angle"], 2.0 * np.pi)


def test_large_coordinate_mesh_loop_remains_closed():
    center = np.array([1.0e9, -2.0e9])
    radius = 2.5e6
    angles = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    perimeter = center + radius * np.column_stack([
        np.cos(angles), np.sin(angles),
    ])
    nodes = np.vstack([center, perimeter])
    elements = np.array([
        [0, index + 1, ((index + 1) % len(perimeter)) + 1]
        for index in range(len(perimeter))
    ])
    mesh = Mesh(
        nodes=nodes,
        elements=elements,
        E=1.0,
        nu=0.25,
        thickness=1.0,
        elem_type="CPS3",
    )

    segments = build_boundary_segments(mesh)

    assert len(segments) == 1
    assert segments[0]["closed"]
    assert segments[0]["type"] == "arc"
    assert segments[0]["info"]["is_outer"] is True


def test_concave_c_loop_probe_is_strictly_inside():
    coords = np.array([
        [2.0, 2.0],
        [9.0, 2.0],
        [9.0, 2.5],
        [2.5, 2.5],
        [2.5, 7.5],
        [9.0, 7.5],
        [9.0, 8.0],
        [2.0, 8.0],
    ])
    mean = coords.mean(axis=0)
    probe = _loop_probe_point(coords[:, 0], coords[:, 1])

    assert not _point_in_loop(
        mean[0], mean[1], coords[:, 0], coords[:, 1])
    assert _point_in_loop(
        probe[0], probe[1], coords[:, 0], coords[:, 1])


def test_validate_boundary_segments_canonicalizes_edge_keys():
    """边界边键规范化契约 (pkg11 A4).

    判别性: 分段节点可任意换向 (反向边必须与正向边视为同一条),
    边界边为 numpy int64 也必须可哈希判重 — 曾内联 _canon 与
    canonical_edge 双实现, 本测试锁死唯一实现的行为。
    """
    mesh = _strip_mesh()
    mesh.build_connectivity()
    # 覆盖全部 6 条边界边; 段 2 反向遍历 {2,3,4}→[4,3,2] 验证换向
    segments = [
        {"nodes": [0, 1, 2]},
        {"nodes": [4, 3, 2]},
        {"nodes": [4, 5, 0]},
    ]
    assert validate_boundary_segments(mesh, segments)
    # 重复一条反向边 → duplicated 必须被检出
    segments.append({"nodes": [0, 5]})
    with pytest.raises(ValueError, match="duplicated"):
        validate_boundary_segments(mesh, segments)
