"""Semantic mesh regions imported from a preprocessor.

The finite-element mesh answers *how to integrate*.  Regions answer *where a
user intent applies*: Physical Points map to nodes, Physical Curves map to
mesh edges, and Physical Surfaces map to displacement elements.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


def canonical_edge(a, b):
    """Return one hashable orientation-independent edge key."""
    a = int(a)
    b = int(b)
    return (min(a, b), max(a, b))


def ordered_edge_chains(edges: Iterable[tuple[int, int]]):
    """Order disconnected non-branching edge components into node chains.

    A physical group may legitimately contain several disconnected curves.
    Branching, however, is ambiguous for line integration and is rejected.
    Closed components repeat their first node at the end.

    The adjacency walk is linear; deterministic node/candidate ordering makes
    the full routine O(E log E).  Earlier versions repeatedly scanned all
    remaining edges while walking each node, which became quadratic on very
    dense CAD boundaries.
    """
    raw_edges = [(int(a), int(b)) for a, b in edges]
    self_loops = sorted({
        int(a) for a, b in raw_edges if int(a) == int(b)
    })
    if self_loops:
        raise ValueError(
            "A curve region contains zero-length topological edges at mesh "
            f"nodes {self_loops[:5]}.")

    all_edges = {canonical_edge(a, b) for a, b in raw_edges}
    adjacency = defaultdict(set)
    for a, b in all_edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    branches = sorted(
        node for node, neighbors in adjacency.items()
        if len(neighbors) > 2
    )
    if branches:
        raise ValueError(
            "A curve region branches at mesh nodes "
            f"{branches[:5]}; define non-branching Physical Curves.")

    chains = []
    visited_nodes = set()
    for seed_node in sorted(adjacency):
        if seed_node in visited_nodes:
            continue
        component_nodes = set()
        component_edges = set()
        frontier = [seed_node]
        while frontier:
            node = frontier.pop()
            if node in component_nodes:
                continue
            component_nodes.add(node)
            for neighbor in adjacency[node]:
                component_edges.add(canonical_edge(node, neighbor))
                if neighbor not in component_nodes:
                    frontier.append(neighbor)
        visited_nodes.update(component_nodes)

        endpoints = sorted(
            node for node in component_nodes
            if len(adjacency[node]) == 1)
        if len(endpoints) not in (0, 2):
            raise ValueError(
                "A curve region has invalid open-chain topology: expected "
                f"0 or 2 endpoints, found {len(endpoints)}.")
        start = endpoints[0] if endpoints else min(component_nodes)
        chain = [start]
        used = set()
        previous = None
        current = start
        while True:
            candidates = sorted(
                neighbor for neighbor in adjacency[current]
                if canonical_edge(current, neighbor) not in used)
            if previous in candidates and len(candidates) > 1:
                candidates.remove(previous)
            if not candidates:
                break
            nxt = candidates[0]
            used.add(canonical_edge(current, nxt))
            chain.append(nxt)
            previous, current = current, nxt
            if current == start:
                break

        if used != component_edges:
            missing = sorted(component_edges - used)
            raise ValueError(
                "Curve-chain reconstruction did not consume every component "
                f"edge; unconsumed edges: {missing[:5]}.")
        if len(chain) >= 2:
            chains.append(tuple(chain))
    return tuple(chains)


@dataclass(frozen=True)
class PointRegion:
    name: str
    physical_tag: int
    entity_tags: tuple[int, ...]
    node_ids: tuple[int, ...]
    source: str = "gmsh_api"


@dataclass(frozen=True)
class CurveRegion:
    name: str
    physical_tag: int
    entity_tags: tuple[int, ...]
    entity_types: tuple[str, ...]
    node_ids: tuple[int, ...]
    edge_pairs: tuple[tuple[int, int], ...]
    source: str = "gmsh_api"
    # Exact per-edge CAD provenance:
    # (node_a, node_b, gmsh_entity_tag, gmsh_entity_type).
    #
    # ``entity_types`` above remains the group-level summary for backwards
    # compatibility.  ``edge_entities`` lets boundary segmentation attach CAD
    # metadata without assuming that every entity in a Physical Curve has the
    # same geometry type.
    edge_entities: tuple[tuple[int, int, int, str], ...] = ()

    @property
    def components(self):
        return ordered_edge_chains(self.edge_pairs)


@dataclass(frozen=True)
class CadCurveRegion:
    """One Gmsh CAD curve on the boundary of an active 2-D surface.

    Unlike :class:`CurveRegion`, this record exists even when the curve has no
    Physical Group.  It is the hard segmentation boundary that prevents two
    distinct CAD entities from being glued together and re-guessed from mesh
    curvature.
    """

    entity_tag: int
    entity_type: str
    node_ids: tuple[int, ...]
    edge_pairs: tuple[tuple[int, int], ...]
    # (surface entity tag, signed/oriented curve entity tag)
    surface_occurrences: tuple[tuple[int, int], ...] = ()
    source: str = "gmsh_api"

    @property
    def components(self):
        return ordered_edge_chains(self.edge_pairs)


@dataclass(frozen=True)
class SurfaceRegion:
    name: str
    physical_tag: int
    entity_tags: tuple[int, ...]
    entity_types: tuple[str, ...]
    element_ids: tuple[int, ...]
    oriented_boundary_entities: tuple[int, ...] = ()
    source: str = "gmsh_api"


@dataclass
class RegionRegistry:
    """Dimension-aware semantic regions attached to one imported mesh."""

    points: list[PointRegion] = field(default_factory=list)
    curves: list[CurveRegion] = field(default_factory=list)
    surfaces: list[SurfaceRegion] = field(default_factory=list)
    cad_curves: list[CadCurveRegion] = field(default_factory=list)
    source: str = "gmsh_api"
    # True only when the registry was extracted from the complete boundary of
    # every active Gmsh 2-D entity.  Hand-built/legacy registries leave this
    # false so partial CAD metadata is treated as advisory.
    cad_boundary_complete: bool = False

    def by_name(self, name, dimension=None):
        """Return all regions with an exact case-insensitive name.

        dimension 必须为 None 或 0/1/2 — 非法维度曾裸 KeyError
        (collections[int(dimension)] 越界), 契约: 带参数名 ValueError。
        """
        needle = str(name).casefold()
        collections = {
            0: self.points,
            1: self.curves,
            2: self.surfaces,
        }
        if dimension is not None and int(dimension) not in collections:
            raise ValueError(
                f"by_name: dimension={dimension!r} — 仅支持 0/1/2 "
                "(0=点, 1=曲线, 2=曲面) 或 None (全部)")
        selected = (
            collections.values()
            if dimension is None else [collections[int(dimension)]])
        return [
            region
            for collection in selected
            for region in collection
            if region.name.casefold() == needle
        ]

    def validate_indices(self, node_count, element_count):
        """Validate imported indices before they reach load or BC assembly."""
        errors = []
        for region in self.points:
            invalid = [
                node for node in region.node_ids
                if not 0 <= node < node_count
            ]
            if invalid:
                errors.append(
                    f"Point region {region.name!r} has invalid nodes "
                    f"{invalid[:5]}.")
        for region in self.curves:
            invalid_nodes = [
                node for node in region.node_ids
                if not 0 <= node < node_count
            ]
            invalid_edges = [
                edge for edge in region.edge_pairs
                if any(not 0 <= node < node_count for node in edge)
            ]
            if invalid_nodes or invalid_edges:
                errors.append(
                    f"Curve region {region.name!r} has invalid node/edge "
                    "indices.")
        for region in self.cad_curves:
            invalid_edges = [
                edge for edge in region.edge_pairs
                if any(not 0 <= node < node_count for node in edge)
            ]
            if invalid_edges:
                errors.append(
                    f"CAD curve entity {region.entity_tag} has invalid "
                    "node/edge indices.")
        for region in self.surfaces:
            invalid = [
                element for element in region.element_ids
                if not 0 <= element < element_count
            ]
            if invalid:
                errors.append(
                    f"Surface region {region.name!r} has invalid elements "
                    f"{invalid[:5]}.")
        if errors:
            raise ValueError("Region index validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors))
        return True

    def validate_against_mesh(self, mesh):
        """Check that curve edges exist and surface elements are in range."""
        self.validate_indices(len(mesh.nodes), len(mesh.elements))
        mesh.build_connectivity()
        mesh_edges = {
            canonical_edge(a, b) for a, b in mesh.edge_to_elems
        }
        errors = []
        for region in self.curves:
            missing = sorted({
                canonical_edge(*edge) for edge in region.edge_pairs
            } - mesh_edges)
            if missing:
                errors.append(
                    f"Curve region {region.name!r} contains {len(missing)} "
                    f"edges absent from the 2-D mesh: {missing[:5]}.")
        for region in self.cad_curves:
            missing = sorted({
                canonical_edge(*edge) for edge in region.edge_pairs
            } - mesh_edges)
            if missing:
                errors.append(
                    f"CAD curve entity {region.entity_tag} contains "
                    f"{len(missing)} edges absent from the 2-D mesh: "
                    f"{missing[:5]}.")
        if errors:
            raise ValueError("Region/mesh validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors))
        return True

    def curve_length(self, name, nodes):
        """Return the total discrete length of all matching curve regions."""
        nodes = np.asarray(nodes, dtype=float)
        edges = {
            canonical_edge(a, b)
            for region in self.by_name(name, dimension=1)
            for a, b in region.edge_pairs
        }
        return float(sum(
            np.linalg.norm(nodes[b] - nodes[a]) for a, b in edges))

    def surface_area(self, name, mesh):
        """Return FEM integration area of all matching surface regions."""
        element_ids = {
            element
            for region in self.by_name(name, dimension=2)
            for element in region.element_ids
        }
        if not element_ids:
            return 0.0
        return float(np.sum(mesh.areas[sorted(element_ids)]))

    def surface_boundary_curves(self, name):
        """Return Physical Curves attached to matching Physical Surfaces."""
        boundary_entities = {
            abs(int(entity))
            for surface in self.by_name(name, dimension=2)
            for entity in surface.oriented_boundary_entities
        }
        return [
            curve for curve in self.curves
            if boundary_entities.intersection(curve.entity_tags)
        ]

    def summary(self):
        """返回区域统计 (公开 API, 供集成方汇总报告)."""
        return {
            "point_regions": len(self.points),
            "curve_regions": len(self.curves),
            "cad_curve_entities": len(self.cad_curves),
            "surface_regions": len(self.surfaces),
            "point_nodes": sum(len(r.node_ids) for r in self.points),
            "curve_edges": sum(len(r.edge_pairs) for r in self.curves),
            "cad_curve_edges": sum(len(r.edge_pairs) for r in self.cad_curves),
            "surface_elements": sum(len(r.element_ids) for r in self.surfaces),
            "cad_boundary_complete": bool(self.cad_boundary_complete),
        }