"""输入防护测试: 续行/3D坐标/重合节点/ELSET注释/混合单元/重复曲线."""
import os
import tempfile
from unittest.mock import patch

import numpy as np


class _AliasMesh:
    def __init__(self):
        self.generate_calls = 0

    def generate(self, dimension):
        self.generate_calls += 1
        raise AssertionError(f"mesh.generate({dimension}) must not be called")


class _AliasModel:
    def __init__(self, groups):
        self.mesh = _AliasMesh()
        self.geo = type(
            "_AliasGeo", (), {"synchronize": lambda self: None})()
        self.groups = tuple(groups)

    def getPhysicalGroups(self):
        return [(1, tag) for tag, _, _ in self.groups]

    def getPhysicalName(self, dimension, physical_tag):
        assert int(dimension) == 1
        return next(
            name for tag, name, _ in self.groups
            if tag == int(physical_tag))

    def getEntitiesForPhysicalGroup(self, dimension, physical_tag):
        assert int(dimension) == 1
        return next(
            entities for tag, _, entities in self.groups
            if tag == int(physical_tag))


class _AliasGmsh:
    def __init__(self, groups):
        self.model = _AliasModel(groups)
        self.initialized = False
        self.finalized = False
        self.opened_source = ""

    def isInitialized(self):
        return self.initialized and not self.finalized

    def initialize(self):
        self.initialized = True

    def finalize(self):
        self.finalized = True

    def open(self, path):
        with open(path, "r", encoding="utf-8") as stream:
            self.opened_source = stream.read()


def _temporary_geo(source):
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".geo", delete=False, encoding="utf-8")
    handle.write(source)
    handle.close()
    return handle.name


def test_overlapping_physical_curves_are_accepted():
    """同一几何曲线可以合法属于多个 Physical Curve."""
    from fem2d.preprocess import read_geo_groups
    t = tempfile.NamedTemporaryFile(mode='w', suffix='.geo', delete=False, encoding='utf-8')
    t.write('Physical Curve("a",1)={1,2};\nPhysical Curve("b",2)={2,3};\n')
    t.close()
    try:
        groups = read_geo_groups(t.name)
        assert groups == {"a": [1, 2], "b": [2, 3]}
    finally:
        os.unlink(t.name)


def test_geo_group_api_uses_final_entity_tags_without_meshing():
    from fem2d.preprocess import read_geo_groups

    path = _temporary_geo(
        'Physical Curve("left", 101) = {4};\nMesh 2;\n')
    fake = _AliasGmsh([(101, "left", (37,))])
    try:
        assert read_geo_groups(path, gmsh_module=fake) == {
            "left": [37]}
        assert fake.model.mesh.generate_calls == 0
        # 消毒逻辑统一到 scripts.gmsh_runner.sanitize_geo_source (曾双实现
        # 分叉, 审计 2026-08-03) — 断言剥离生效而非特定文案
        assert "removed" in fake.opened_source
        assert "Mesh 2;" not in fake.opened_source
    finally:
        os.unlink(path)


def test_geo_group_api_handles_multiline_definitions_and_comments():
    from fem2d.preprocess import read_geo_groups

    path = _temporary_geo(
        'Physical Curve(\\n'
        '  "curved", 102 // parser comment\\n'
        ') = {\\n'
        '  4, // split across lines\\n'
        '  9\\n'
        '};\\n')
    fake = _AliasGmsh([(102, "curved", (21, 22))])
    try:
        assert read_geo_groups(path, gmsh_module=fake) == {
            "curved": [21, 22]}
    finally:
        os.unlink(path)


def test_geo_group_api_preserves_overlapping_physical_curves():
    from fem2d.preprocess import read_geo_groups

    path = _temporary_geo("// groups supplied by parsed CAD model\n")
    fake = _AliasGmsh([
        (101, "left", (37,)),
        (102, "all_boundary", (37, 38)),
    ])
    try:
        assert read_geo_groups(path, gmsh_module=fake) == {
            "left": [37],
            "all_boundary": [37, 38],
        }
    finally:
        os.unlink(path)


def test_geo_group_api_unavailable_uses_regex_fallback():
    from fem2d.gmsh_adapter import GmshUnavailableError
    from fem2d.preprocess import read_geo_groups

    path = _temporary_geo(
        'Physical Curve("fallback", 101) = {4, 9};\n')
    try:
        with patch(
                "fem2d.gmsh_adapter._load_gmsh_module",
                side_effect=GmshUnavailableError("not installed")):
            assert read_geo_groups(path) == {"fallback": [4, 9]}
    finally:
        os.unlink(path)


def test_geo_group_alias_ignores_entity_absent_from_inp():
    from fem2d import Mesh
    from fem2d.boundary import _resolve_edge_indices, build_boundary_segments
    from fem2d.regions import canonical_edge

    mesh = Mesh(
        nodes=np.array([
            [0.0, 0.0], [1.0, 0.0],
            [1.0, 1.0], [0.0, 1.0],
        ]),
        elements=np.array([[0, 1, 2], [0, 2, 3]]),
        E=1.0, nu=0.25, elem_type="CPS3",
    )
    labels = {
        (0, 1): frozenset({"Line38"}),
        (1, 2): frozenset({"Line37"}),
    }
    with patch(
            "fem2d.preprocess.read_geo_groups",
            return_value={"left": [37]}):
        segments = build_boundary_segments(
            mesh, edge_labels=labels, geo_path="ignored.geo")

    selected = _resolve_edge_indices("left", segments)
    assert len(selected) == 1
    segment = segments[selected[0]]
    selected_edges = {
        canonical_edge(a, b)
        for a, b in zip(segment["nodes"], segment["nodes"][1:])
    }
    assert selected_edges == {(1, 2)}
    assert all(
        "Line38" not in segment.get("info", {}).get(
            "physical_names", ())
        for segment in segments)

def test_boundary_labels_from_mesh_edges():
    """边界分段与语义覆盖 — 重叠标签由边标签映射提供 (原 ELSET 路径已移除)."""
    from fem2d import Mesh
    from fem2d.boundary import (
        _resolve_edge_indices,
        build_boundary_segments,
        semantic_coverage,
    )
    mesh = Mesh(
        nodes=np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        elements=np.array([[0, 1, 2], [0, 2, 3]]),
        E=1.0, nu=0.25, thickness=1.0,
        elem_type="CPS3")
    labels = {
        (0, 1): frozenset({'Line1', 'left', 'all_boundary'}),
    }
    segments = build_boundary_segments(mesh, edge_labels=labels)
    assert len(_resolve_edge_indices('left', segments)) == 1
    assert len(_resolve_edge_indices('all_boundary', segments)) == 1
    assert _resolve_edge_indices('Line1', segments) == []
    report = semantic_coverage(mesh, segments)
    assert report["physical_names"] == ("all_boundary", "left")
    assert report["covered_edges"] == 1
    assert report["total_boundary_edges"] == 4


def test_body_force_space_after_comma():
    """体力 '0, -78000' 和 '0,-78000' 应等价."""
    from scripts.geo_spec import parse_spec
    t = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
    t.write('类型 矩形板\n宽 3.0\n高 2.0\n网格 0.1\n体力 0, -78000\n')
    t.close()
    try:
        spec = parse_spec(t.name)
        assert spec['body_force'] == [0.0, -78000.0], f"Got {spec['body_force']}"
    finally:
        os.unlink(t.name)

def test_convergence_rate_on_synthetic_data():
    """合成 O(h²) 数据应回归出 ~2.0."""
    from scripts.convergence_study import convergence_rate
    h = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    exact = 1.0
    values = exact + 0.1 * h**2  # pure O(h²) with known exact
    k = convergence_rate(h, values, finest_value=exact)
    assert 1.5 < k < 2.5, f"Expected k~2.0 for O(h^2) data, got {k:.2f}"

def test_convergence_rate_eta_no_exclude():
    """eta 模式(finest_value=None)不应排除最细网格."""
    from scripts.convergence_study import convergence_rate
    h = np.array([1.0, 0.5, 0.25])
    values = 100 * h  # pure O(h)
    k = convergence_rate(h, values, finest_value=None)
    assert 0.7 < k < 1.3, f"Expected k~1.0, got {k:.2f}"
