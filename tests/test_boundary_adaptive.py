"""Regression tests for adaptive boundary primitive detection."""
import numpy as np

from fem2d import Mesh, detect_boundaries
from fem2d.boundary.geometry import classify
from fem2d.boundary.topology import _point_in_loop


def _fan_mesh(points):
    points = np.asarray(points, dtype=float)
    nodes = np.vstack([points.mean(axis=0), points])
    n = len(points)
    elements = np.array([
        [0, i + 1, ((i + 1) % n) + 1] for i in range(n)
    ])
    return Mesh(
        nodes=nodes, elements=elements,
        E=1.0, nu=0.3, thickness=1.0,
    )


def _regular_loop(n, radius=1.0):
    return np.array([
        [
            radius * np.cos(2.0 * np.pi * i / n),
            radius * np.sin(2.0 * np.pi * i / n),
        ]
        for i in range(n)
    ])


def _rounded_rectangle():
    radius = 0.5
    points = []

    def line(start, end, count):
        start = np.asarray(start)
        end = np.asarray(end)
        return [
            start + (end - start) * value
            for value in np.linspace(0.0, 1.0, count, endpoint=False)
        ]

    def arc(center, angle_start, angle_end, count):
        center = np.asarray(center)
        return [
            center + radius * np.array([np.cos(value), np.sin(value)])
            for value in np.linspace(
                angle_start, angle_end, count, endpoint=False)
        ]

    points += line((-1.5, -1.0), (1.5, -1.0), 12)
    points += arc((1.5, -0.5), -np.pi / 2.0, 0.0, 8)
    points += line((2.0, -0.5), (2.0, 0.5), 6)
    points += arc((1.5, 0.5), 0.0, np.pi / 2.0, 8)
    points += line((1.5, 1.0), (-1.5, 1.0), 12)
    points += arc((-1.5, 0.5), np.pi / 2.0, np.pi, 8)
    points += line((-2.0, 0.5), (-2.0, -0.5), 6)
    points += arc((-1.5, -0.5), np.pi, 3.0 * np.pi / 2.0, 8)
    return np.asarray(points)


def _subdivide_chords(points, subdivisions):
    result = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        result.extend(
            start + (end - start) * value
            for value in np.linspace(
                0.0, 1.0, subdivisions, endpoint=False))
    return np.asarray(result)


def test_coarse_regular_polygons_remain_lines_without_cad_semantics():
    for sample_count in (8, 10, 12, 16):
        segments = detect_boundaries(
            _fan_mesh(_regular_loop(sample_count)))
        assert len(segments) == sample_count
        assert {segment["type"] for segment in segments} == {"line"}
        assert all(
            len(segment["nodes"]) == 2
            and not segment["closed"]
            for segment in segments)


def test_subdivided_regular_polygons_preserve_line_primitives():
    for primitive_count in (8, 10, 12, 16):
        polygonized = _subdivide_chords(
            _regular_loop(primitive_count), subdivisions=4)
        segments = detect_boundaries(_fan_mesh(polygonized))

        assert len(segments) == primitive_count
        assert {segment["type"] for segment in segments} == {"line"}
        assert all(
            len(segment["nodes"]) == 5
            and not segment["closed"]
            for segment in segments)


def test_closed_physical_curve_classification_refits_chord_vertices():
    polygonized = _subdivide_chords(_regular_loop(24, radius=8.0), 9)
    closed = np.vstack([polygonized, polygonized[0]])

    segment_type, _, info = classify(closed, scale=135.0, is_outer=False)

    assert segment_type == "arc"
    assert info["primitive_samples"] == 24
    assert np.isclose(info["radius"], 8.0, rtol=1e-12)


def test_true_hexagon_remains_six_lines():
    segments = detect_boundaries(_fan_mesh(_regular_loop(6)))
    assert len(segments) == 6
    assert {segment["type"] for segment in segments} == {"line"}


def test_g1_rounded_rectangle_preserves_eight_primitives():
    segments = detect_boundaries(_fan_mesh(_rounded_rectangle()))
    lines = [segment for segment in segments if segment["type"] == "line"]
    arcs = [segment for segment in segments if segment["type"] == "arc"]

    assert len(segments) == 8
    assert len(lines) == 4
    assert len(arcs) == 4
    assert all(
        np.isclose(segment["info"]["radius"], 0.5, rtol=1e-10)
        for segment in arcs)


def test_point_in_loop_is_scale_aware_and_deduplicates_vertices():
    for origin, span in (
            (0.0, 1e-20),
            (1e12, 100.0),
            (-1e12, 1e-3)):
        xs = np.array([
            origin, origin + span, origin + span,
            origin + span, origin, origin,
        ])
        ys = np.array([
            origin, origin, origin,
            origin + span, origin + span, origin,
        ])
        assert _point_in_loop(
            origin + 0.5 * span, origin + 0.5 * span, xs, ys)
        assert not _point_in_loop(
            origin + 2.0 * span, origin + 0.5 * span, xs, ys)
        assert _point_in_loop(origin, origin, xs, ys)


def test_detected_ring_has_gmsh_outer_and_hole_orientation():
    n = 12
    outer = _regular_loop(n, radius=2.0)
    inner = _regular_loop(n, radius=1.0)
    nodes = np.vstack([outer, inner])
    elements = np.array([
        [i, (i + 1) % n, n + (i + 1) % n, n + i]
        for i in range(n)
    ])
    mesh = Mesh(
        nodes=nodes, elements=elements,
        E=1.0, nu=0.3, thickness=1.0, elem_type="CPS4",
    )

    segments = detect_boundaries(mesh)
    oriented_area = {}
    for segment in segments:
        loop_id = segment["info"]["loop_id"]
        coords = segment["coords"]
        oriented_area[loop_id] = oriented_area.get(loop_id, 0.0) + (
            0.5 * np.sum(
                coords[:-1, 0] * coords[1:, 1]
                - coords[1:, 0] * coords[:-1, 1])
        )

    outer_loop = next(
        segment["info"]["loop_id"]
        for segment in segments
        if segment["info"]["is_outer"])
    hole_loop = next(
        segment["info"]["loop_id"]
        for segment in segments
        if not segment["info"]["is_outer"])
    assert oriented_area[outer_loop] > 0.0
    assert oriented_area[hole_loop] < 0.0
