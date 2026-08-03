"""Build boundary segment dictionaries from ordered mesh-edge chains.

The public boundary model intentionally stays dictionary-based for backward
compatibility.  This class owns the validation, orientation and geometry
classification needed whenever Gmsh metadata is mapped onto those chains.
"""
from __future__ import annotations

import numpy as np

from ..regions import canonical_edge
from .geometry import (
    classify,
    piecewise_smooth_breakpoints,
    sharp_corner_indices,
)
from .segment_utils import (
    LoopContext,
    deduplicate_consecutive_nodes,
)
from .topology import _signed_loop_area


class BoundarySegmentBuilder:
    """Convert ordered node chains into validated boundary segments."""

    def __init__(
            self, mesh, edge_context, edge_metadata, scale, segments=None):
        self.mesh = mesh
        self.edge_context = edge_context
        self.edge_metadata = edge_metadata
        self.scale = float(scale)
        self.segments = segments if segments is not None else []

    def append_chain(
            self, nodes, name="", source_info=None,
            allow_geometric_split=True):
        """Validate, orient, classify and append one ordered boundary chain."""
        nodes = deduplicate_consecutive_nodes(nodes)
        self._validate_nodes(nodes, name)
        edges = [
            canonical_edge(a, b) for a, b in zip(nodes, nodes[1:])
        ]
        context = LoopContext.from_edges(
            name, edges, self.edge_context)
        closed = len(nodes) >= 4 and nodes[0] == nodes[-1]

        if name and allow_geometric_split:
            preview_type, _, _ = classify(
                self.mesh.nodes[nodes],
                self.scale,
                context.is_outer,
                closed=closed,
            )
            # A conic already passed strict all-point residual checks. Keep it
            # intact even if Gmsh represented it with several CAD entities.
            if preview_type not in {"arc", "ellipse"}:
                subchains = self.split_at_structural_breaks(nodes)
                if len(subchains) > 1:
                    for subchain in subchains:
                        self.append_chain(
                            subchain,
                            name,
                            source_info,
                            allow_geometric_split=False,
                        )
                    return

        self._orient(nodes, edges, context, closed)
        coords = self.mesh.nodes[nodes]
        segment_type, label, info = classify(
            coords,
            self.scale,
            context.is_outer,
            closed=closed,
        )
        info = dict(info)
        info.update({
            "is_outer": context.is_outer,
            "loop_depth": context.loop_depth,
            "loop_id": context.loop_id,
        })
        if source_info:
            info.update(source_info)
        if name:
            info.update({
                "source": "physical_curve",
                "physical_name": name,
            })
            label = f"{name} | {label}"
        self.segments.append({
            "type": segment_type,
            "nodes": nodes,
            "coords": coords,
            "label": label,
            "info": info,
            "closed": closed,
        })

    def split_at_structural_breaks(self, nodes):
        """Split at actual corners/straight-run transitions, never by noise."""
        nodes = deduplicate_consecutive_nodes(nodes)
        if len(nodes) < 2:
            raise ValueError(
                "Physical Curve produced a boundary chain with fewer than "
                "two distinct consecutive nodes.")

        closed = len(nodes) >= 4 and nodes[0] == nodes[-1]
        if closed:
            return self._split_closed_chain(nodes)
        return self._split_open_chain(nodes)

    def metadata_for_edges(self, edges):
        """Collect deterministic CAD provenance for a group of mesh edges."""
        entity_tags = set()
        entity_types = set()
        surface_entities = set()
        for edge in edges:
            metadata = self.edge_metadata.get(edge, {})
            entity_tags.update(metadata.get("entity_tags", ()))
            entity_types.update(metadata.get("cad_entity_types", ()))
            surface_entities.update(
                metadata.get("surface_entity_tags", ()))
        return {
            "entity_tags": tuple(sorted(map(int, entity_tags))),
            "cad_entity_types": tuple(sorted(map(str, entity_types))),
            "surface_entity_tags": tuple(
                sorted(map(int, surface_entities))),
        }

    def _split_closed_chain(self, nodes):
        loop = nodes[:-1]
        coords = self.mesh.nodes[loop]
        positions = sorted(set(
            sharp_corner_indices(coords)
            + piecewise_smooth_breakpoints(coords)
        ))
        if len(positions) < 2:
            return [nodes]

        result = []
        for index, start in enumerate(positions):
            end = positions[(index + 1) % len(positions)]
            subchain = [loop[start]]
            current = start
            while current != end:
                current = (current + 1) % len(loop)
                subchain.append(loop[current])
            if len(subchain) >= 2 and subchain[0] != subchain[-1]:
                result.append(subchain)
        return result or [nodes]

    def _split_open_chain(self, nodes):
        coords = self.mesh.nodes[nodes]
        positions = [0] + [
            index for index in sharp_corner_indices(coords)
            if 0 < index < len(nodes) - 1
        ] + [len(nodes) - 1]
        positions = sorted(set(positions))
        return [
            nodes[start:end + 1]
            for start, end in zip(positions, positions[1:])
            if end > start
        ]

    def _validate_nodes(self, nodes, name):
        if len(nodes) < 2:
            raise ValueError(
                f"Physical Curve {name!r} produced a single-point segment.")
        edge_lengths = np.linalg.norm(
            np.diff(self.mesh.nodes[nodes], axis=0), axis=1)
        coordinate_scale = max(
            float(np.ptp(self.mesh.nodes[:, 0])),
            float(np.ptp(self.mesh.nodes[:, 1])),
            np.finfo(float).tiny,
        )
        tolerance = (
            np.finfo(float).eps * coordinate_scale * 32.0)
        if np.any(edge_lengths <= tolerance):
            raise ValueError(
                f"Physical Curve {name!r} contains a zero-length boundary "
                "edge; repair coincident mesh nodes before applying loads.")

    def _orient(self, nodes, edges, context, closed):
        if closed:
            area = _signed_loop_area(self.mesh.nodes[nodes])
            if (
                    context.is_outer and area < 0.0
                    or not context.is_outer and area > 0.0):
                nodes.reverse()
            return

        expected = self.edge_context[edges[0]].get("direction")
        if expected and (nodes[0], nodes[1]) != expected:
            nodes.reverse()
