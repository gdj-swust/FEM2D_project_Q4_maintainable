"""Deterministic tests for the Gmsh topology adapter.

These tests use a small fake of the documented Gmsh model/mesh API. They
exercise FEM2D's mapping logic without requiring a Gmsh installation in CI.
The real executable integration tests remain in ``test_boundary_gmsh.py``.
"""
import tempfile
from pathlib import Path

import numpy as np

from fem2d.boundary import (
    _resolve_edge_indices,
    segments_from_region_registry,
    validate_boundary_segments,
)
from fem2d.gmsh_adapter import generate_from_geo
from fem2d.mesh import Mesh


class _FakeOption:
    def __init__(self):
        self.values = {}

    def setNumber(self, name, value):
        self.values[str(name)] = float(value)


class _FakeMeshAPI:
    _curve_nodes = {
        1: (10, 20),
        2: (20, 30),
        3: (30, 40),
        4: (40, 10),
    }

    def __init__(self):
        self.generated_dimension = None
        self.clear_called = False

    def generate(self, dimension):
        self.generated_dimension = int(dimension)

    def clear(self):
        self.clear_called = True

    def getNodes(self):
        tags = np.array([10, 20, 30, 40], dtype=np.int64)
        coords = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
        ])
        return tags, coords, np.empty(0)

    def getNodesForPhysicalGroup(self, dimension, physical_tag):
        groups = {
            (0, 301): [30],
            (1, 101): [40, 10],
            (1, 102): [20, 30],
            (1, 103): [10, 20, 30, 40],
            (2, 201): [10, 20, 30, 40],
        }
        return np.array(
            groups[(int(dimension), int(physical_tag))], dtype=np.int64
        ), np.empty(0)

    def getElements(self, dimension, entity_tag=-1):
        dimension = int(dimension)
        entity_tag = int(entity_tag)
        if dimension == 1:
            a, b = self._curve_nodes[entity_tag]
            return (
                [1],
                [np.array([100 + entity_tag], dtype=np.int64)],
                [np.array([a, b], dtype=np.int64)],
            )
        if dimension == 2 and entity_tag in (-1, 1):
            return (
                [2],
                [np.array([1001, 1002], dtype=np.int64)],
                [np.array(
                    [10, 20, 30, 10, 30, 40], dtype=np.int64)],
            )
        return [], [], []

    def getElementProperties(self, element_type):
        if int(element_type) == 1:
            return "Line 2", 1, 1, 2, np.empty(0), 2
        if int(element_type) == 2:
            return "Triangle 3", 2, 1, 3, np.empty(0), 3
        raise KeyError(element_type)


class _FakeQ4MeshAPI(_FakeMeshAPI):
    def getElements(self, dimension, entity_tag=-1):
        dimension = int(dimension)
        entity_tag = int(entity_tag)
        if dimension == 2 and entity_tag in (-1, 1):
            return (
                [3],
                [np.array([2001], dtype=np.int64)],
                [np.array([10, 20, 30, 40], dtype=np.int64)],
            )
        return super().getElements(dimension, entity_tag)

    def getElementProperties(self, element_type):
        if int(element_type) == 3:
            return "Quadrangle 4", 2, 1, 4, np.empty(0), 4
        return super().getElementProperties(element_type)


class _FakeModel:
    def __init__(self, mesh_api=None):
        self.mesh = mesh_api or _FakeMeshAPI()
        self.geo = type(
            "_FakeGeo", (), {"synchronize": lambda self: None})()

    def getPhysicalGroups(self):
        return [(0, 301), (1, 101), (1, 102), (1, 103), (2, 201)]

    def getPhysicalName(self, dimension, tag):
        return {
            (0, 301): "load_point",
            (1, 101): "left",
            (1, 102): "right",
            (1, 103): "all_boundary",
            (2, 201): "domain",
        }[(int(dimension), int(tag))]

    def getEntitiesForPhysicalGroup(self, dimension, tag):
        return {
            (0, 301): [3],
            (1, 101): [4],
            (1, 102): [2],
            (1, 103): [1, 2, 3, 4],
            (2, 201): [1],
        }[(int(dimension), int(tag))]

    def getEntityType(self, dimension, tag):
        return {0: "Point", 1: "Line", 2: "Plane Surface"}[int(dimension)]

    def getBoundary(
            self, entities, combined=False, oriented=True, recursive=False):
        assert entities == [(2, 1)]
        assert combined is False and oriented is True and recursive is False
        return [(1, 1), (1, 2), (1, 3), (1, -4)]


class _FakeGmsh:
    def __init__(self, model=None):
        self.model = model or _FakeModel()
        self.option = _FakeOption()
        self.opened = None
        self.opened_source = None
        self.initialized = False
        self.finalized = False

    def isInitialized(self):
        return self.initialized and not self.finalized

    def initialize(self):
        self.initialized = True

    def finalize(self):
        self.finalized = True

    def open(self, path):
        self.opened = str(path)
        self.opened_source = Path(path).read_text(encoding="utf-8")


def _generate(plane_type="stress"):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "square.geo"
    path.write_text(
        'SetFactory("OpenCASCADE");\n'
        'Mesh.Format = 39; Save "script-owned.inp";\n',
        encoding="utf-8",
    )
    fake = _FakeGmsh()
    result = generate_from_geo(
        path, plane_type=plane_type, gmsh_module=fake)
    return temporary, fake, result


def test_gmsh_api_maps_physical_groups_to_mesh_regions():
    temporary, fake, result = _generate()
    try:
        assert result.elem_type == "CPS3"
        assert result.nodes.shape == (4, 2)
        assert result.elements.tolist() == [[0, 1, 2], [0, 2, 3]]
        assert result.node_tag_to_index == {10: 0, 20: 1, 30: 2, 40: 3}
        assert result.element_tag_to_index == {1001: 0, 1002: 1}
        assert fake.model.mesh.generated_dimension == 2
        assert fake.model.mesh.clear_called
        assert "script-owned.inp" not in fake.opened_source
        assert fake.finalized

        registry = result.regions
        assert registry.cad_boundary_complete is True
        assert len(registry.cad_curves) == 4
        assert {
            curve.entity_tag for curve in registry.cad_curves
        } == {1, 2, 3, 4}
        assert all(
            len(curve.edge_pairs) == 1
            for curve in registry.cad_curves)
        assert registry.by_name("load_point", 0)[0].node_ids == (2,)
        all_boundary = registry.by_name("all_boundary", 1)[0]
        assert len(all_boundary.edge_pairs) == 4
        assert {
            (a, b, entity_type)
            for a, b, _, entity_type in all_boundary.edge_entities
        } == {
            (0, 1, "Line"),
            (1, 2, "Line"),
            (2, 3, "Line"),
            (0, 3, "Line"),
        }
        surface = registry.by_name("domain", 2)[0]
        assert surface.element_ids == (0, 1)
        assert surface.oriented_boundary_entities == (1, 2, 3, -4)
        assert {
            curve.name
            for curve in registry.surface_boundary_curves("domain")
        } == {"left", "right", "all_boundary"}
    finally:
        temporary.cleanup()


def test_region_segments_preserve_overlapping_physical_curves():
    temporary, _, result = _generate()
    try:
        mesh = Mesh(
            nodes=result.nodes,
            elements=result.elements,
            E=1.0,
            nu=0.25,
            thickness=1.0,
            elem_type=result.elem_type,
        )
        result.regions.validate_against_mesh(mesh)
        segments = segments_from_region_registry(mesh, result.regions)
        assert len(segments) == 4
        assert validate_boundary_segments(mesh, segments)
        assert len(_resolve_edge_indices("left", segments)) == 1
        assert len(_resolve_edge_indices("right", segments)) == 1
        assert len(_resolve_edge_indices("all_boundary", segments)) == 4
        assert np.isclose(
            result.regions.curve_length("all_boundary", mesh.nodes), 4.0)
        assert np.isclose(
            result.regions.surface_area("domain", mesh), 1.0)
    finally:
        temporary.cleanup()


def test_plane_strain_selects_cpe_element_code():
    temporary, _, result = _generate(plane_type="strain")
    try:
        assert result.elem_type == "CPE3"
    finally:
        temporary.cleanup()


def test_quad_mode_extracts_q4_and_sets_recombination_options():
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "square_q4.geo"
    path.write_text("// fake Q4 model\n", encoding="utf-8")
    fake = _FakeGmsh(_FakeModel(_FakeQ4MeshAPI()))
    try:
        result = generate_from_geo(
            path, quad=True, gmsh_module=fake)
        assert result.elem_type == "CPS4"
        assert result.elements.tolist() == [[0, 1, 2, 3]]
        assert result.element_tag_to_index == {2001: 0}
        assert fake.option.values["Mesh.RecombineAll"] == 1.0
        assert fake.option.values["Mesh.Algorithm"] == 8.0
    finally:
        temporary.cleanup()


def test_unmeshed_cad_surface_does_not_create_phantom_boundaries():
    class _ModelWithConstructionSurface(_FakeModel):
        def getEntities(self, dimension):
            assert int(dimension) == 2
            return [(2, 1), (2, 99)]

    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "construction.geo"
    path.write_text("// fake construction surface\n", encoding="utf-8")
    fake = _FakeGmsh(_ModelWithConstructionSurface())
    try:
        result = generate_from_geo(path, gmsh_module=fake)
        assert result.regions.cad_boundary_complete is True
        assert {
            curve.entity_tag for curve in result.regions.cad_curves
        } == {1, 2, 3, 4}
    finally:
        temporary.cleanup()
