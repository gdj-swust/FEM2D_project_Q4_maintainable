"""Regression tests for arcs, splines, and generic smooth boundaries."""
import numpy as np

from fem2d import Mesh, detect_boundaries
from fem2d.boundary import (
    segments_from_physical_curves,
    validate_boundary_segments,
)
from fem2d.boundary.geometry import classify


def _fan_mesh(points):
    points = np.asarray(points, dtype=float)
    nodes = np.vstack([points.mean(axis=0), points])
    n = len(points)
    elements = np.array([
        [0, index + 1, ((index + 1) % n) + 1]
        for index in range(n)
    ])
    return Mesh(
        nodes=nodes, elements=elements,
        E=1.0, nu=0.3, thickness=1.0,
    )


def test_exact_open_circular_arc_is_recognized():
    angles = np.linspace(-0.4, 1.2, 41)
    center = np.array([3.5, -1.25])
    radius = 2.75
    coords = center + radius * np.column_stack([
        np.cos(angles), np.sin(angles),
    ])

    segment_type, _, info = classify(
        coords, scale=10.0, is_outer=True)

    assert segment_type == "arc"
    assert np.allclose(info["center"], center, rtol=1e-11, atol=1e-11)
    assert np.isclose(info["radius"], radius, rtol=1e-11)


def test_parabola_is_kept_as_general_curve():
    x = np.linspace(-2.0, 2.0, 61)
    coords = np.column_stack([x, 0.35 * x * x + 0.1 * x])

    segment_type, label, info = classify(
        coords, scale=5.0, is_outer=True)

    assert segment_type == "curve"
    assert "曲线" in label
    assert info["geometry"] == "general"
    assert not info["closed"]


def test_smooth_nonconic_loop_is_not_split_at_curvature_extrema():
    angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    radius = 1.0 + 0.18 * np.cos(3.0 * angles)
    points = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
    ])

    segments = detect_boundaries(_fan_mesh(points))

    assert len(segments) == 1
    assert segments[0]["type"] == "curve"
    assert segments[0]["closed"]
    assert segments[0]["info"]["geometry"] == "general"


def test_physical_spline_is_rebuilt_as_one_complete_chain():
    bottom = np.column_stack([
        np.linspace(-2.0, 2.0, 9, endpoint=False),
        np.zeros(9),
    ])
    right = np.column_stack([
        np.full(5, 2.0),
        np.linspace(0.0, 1.0, 5, endpoint=False),
    ])
    top_x = np.linspace(2.0, -2.0, 41, endpoint=False)
    top = np.column_stack([
        top_x,
        1.0 + 0.18 * np.sin(np.pi * (top_x - 2.0) / -4.0),
    ])
    left = np.column_stack([
        np.full(5, -2.0),
        np.linspace(1.0, 0.0, 5, endpoint=False),
    ])
    points = np.vstack([bottom, right, top, left])
    mesh = _fan_mesh(points)

    first_top = 1 + len(bottom) + len(right)
    first_left = first_top + len(top)
    top_nodes = list(range(first_top, first_left)) + [first_left]
    edge_labels = {
        (a, b): "free_spline"
        for a, b in zip(top_nodes, top_nodes[1:])
    }

    segments = segments_from_physical_curves(
        mesh, edge_labels, geo_path=None)
    validate_boundary_segments(mesh, segments)
    physical = [
        segment for segment in segments
        if segment["info"].get("physical_name") == "free_spline"
    ]

    assert len(physical) == 1
    assert physical[0]["type"] == "curve"
    assert len(physical[0]["nodes"]) == len(top_nodes)
    assert set(physical[0]["nodes"]) == set(top_nodes)


def test_one_physical_name_can_cover_disconnected_outer_and_hole_loops():
    n = 24
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    outer = 2.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    inner = np.column_stack([np.cos(angles), np.sin(angles)])
    nodes = np.vstack([outer, inner])
    elements = np.array([
        [i, (i + 1) % n, n + (i + 1) % n, n + i]
        for i in range(n)
    ])
    mesh = Mesh(
        nodes=nodes, elements=elements,
        E=1.0, nu=0.3, thickness=1.0, elem_type="CPS4",
    )
    edge_labels = {}
    for offset in (0, n):
        for index in range(n):
            edge_labels[
                (offset + index, offset + (index + 1) % n)
            ] = "walls"

    segments = segments_from_physical_curves(
        mesh, edge_labels, geo_path=None)
    validate_boundary_segments(mesh, segments)

    assert len(segments) == 2
    assert all(
        segment["info"].get("physical_name") == "walls"
        for segment in segments)
    assert sum("外边" in segment["label"] for segment in segments) == 1
    assert sum("内孔" in segment["label"] for segment in segments) == 1
