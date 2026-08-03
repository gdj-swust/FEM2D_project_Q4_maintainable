"""Safe user-facing merge of adjacent Gmsh conic curve entities."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..regions import canonical_edge, ordered_edge_chains
from .geometry import classify
from .segment_utils import (
    mesh_scale,
    segment_edges,
    segment_info,
    segment_physical_names,
    segment_sort_key,
    tuple_value,
)
from .topology import _signed_loop_area

MERGEABLE_CAD_CONICS = frozenset({"circle", "ellipse"})
TESSELLATED_CONIC_CAD_KINDS = frozenset({"line"})
MIN_TESSELLATED_CONIC_SEGMENTS = 16

# A 20-piece 2:1 ellipse turns by 35.15 degrees at its high-curvature ends,
# while a regular octagon turns by 45 degrees.  Combined with the strict
# whole-chain conic fit and the 16-fragment minimum for Line tessellations,
# this gate accepts the former without hiding coarse polygon corners.
MAX_JOIN_TURN_RADIANS = np.deg2rad(40.0)


class ConicSegmentMerger:
    """Merge only conic segments that share one exact semantic contract."""

    def __init__(self, mesh, segments):
        self.mesh = mesh
        self.segments = segments

    def merge(self):
        """Return segments with compatible conic components coalesced."""
        if len(self.segments) < 2:
            return self.segments

        groups = defaultdict(list)
        for index, segment in enumerate(self.segments):
            key = self._merge_key(segment)
            if key is not None:
                groups[key].append(index)

        consumed = set()
        merged_segments = []
        for key in sorted(groups, key=str):
            indices = groups[key]
            if len(indices) < 2:
                continue
            edge_owner = self._unique_edge_owners(indices)
            if edge_owner is None:
                continue
            for chain in ordered_edge_chains(edge_owner):
                component_edges = {
                    canonical_edge(a, b)
                    for a, b in zip(chain, chain[1:])
                }
                source_indices = sorted({
                    edge_owner[edge] for edge in component_edges
                })
                if len(source_indices) < 2:
                    continue
                if not self._tangent_continuous(
                        component_edges, edge_owner):
                    continue
                merged = self._merge_component(
                    chain, source_indices)
                if merged is None:
                    continue
                consumed.update(source_indices)
                merged_segments.append(merged)

        if not merged_segments:
            return self.segments
        result = [
            segment
            for index, segment in enumerate(self.segments)
            if index not in consumed
        ]
        result.extend(merged_segments)
        result.sort(key=segment_sort_key)
        return result

    @staticmethod
    def _merge_key(segment):
        """Return the immutable semantic identity required for merging."""
        info = segment_info(segment)
        physical_names = segment_physical_names(segment)
        if not physical_names:
            return None

        normalized_types = {
            str(entity_type).strip().casefold()
            for entity_type in tuple_value(
                info.get("cad_entity_types", ()))
        }
        if (
                len(normalized_types) != 1
                or not normalized_types.issubset(
                    MERGEABLE_CAD_CONICS
                    | TESSELLATED_CONIC_CAD_KINDS)):
            return None

        loop_id = int(info.get("loop_id", -1))
        if loop_id < 0:
            return None
        return (
            physical_names,
            tuple(sorted(map(
                int, tuple_value(info.get("physical_tags", ()))))),
            loop_id,
            int(info.get("loop_depth", 0)),
            bool(info.get("is_outer", False)),
            tuple(sorted(map(
                int,
                tuple_value(info.get("surface_entity_tags", ())),
            ))),
            tuple(sorted(normalized_types)),
        )

    def _unique_edge_owners(self, indices):
        """Map every edge to one source segment, or reject ambiguity."""
        edge_owner = {}
        for index in indices:
            for edge in segment_edges(self.segments[index]):
                if edge in edge_owner:
                    return None
                edge_owner[edge] = index
        return edge_owner

    def _tangent_continuous(self, component_edges, edge_owner):
        """Check the discrete tangent only where CAD entities meet."""
        incidence = defaultdict(list)
        for a, b in component_edges:
            owner = edge_owner[(a, b)]
            incidence[a].append((b, owner))
            incidence[b].append((a, owner))

        for node, incident in incidence.items():
            if len(incident) != 2:
                continue
            (first, first_owner), (second, second_owner) = incident
            if first_owner == second_owner:
                continue
            first_vector = self.mesh.nodes[first] - self.mesh.nodes[node]
            second_vector = self.mesh.nodes[second] - self.mesh.nodes[node]
            first_length = float(np.linalg.norm(first_vector))
            second_length = float(np.linalg.norm(second_vector))
            if (
                    first_length <= np.finfo(float).tiny
                    or second_length <= np.finfo(float).tiny):
                return False
            cosine = np.dot(first_vector, second_vector) / (
                first_length * second_length)
            interior_angle = float(np.arccos(np.clip(
                cosine, -1.0, 1.0)))
            if abs(np.pi - interior_angle) > MAX_JOIN_TURN_RADIANS:
                return False
        return True

    def _merge_component(self, chain, source_indices):
        """Build one merged segment after all semantic/geometric gates."""
        source_segments = [
            self.segments[index] for index in source_indices
        ]
        nodes = list(map(int, chain))
        closed = len(nodes) >= 4 and nodes[0] == nodes[-1]
        self._restore_source_direction(nodes, source_segments, closed)

        first_info = segment_info(source_segments[0])
        is_outer = bool(first_info.get("is_outer", False))
        coords = self.mesh.nodes[nodes]
        segment_type, geometric_label, geometric_info = classify(
            coords,
            mesh_scale(self.mesh.nodes),
            is_outer,
            closed=closed,
        )
        cad_kind = self._single_cad_kind(source_segments)
        if (
                cad_kind in TESSELLATED_CONIC_CAD_KINDS
                and len(source_segments)
                < MIN_TESSELLATED_CONIC_SEGMENTS):
            return None
        if not self._classification_matches(
                cad_kind, segment_type):
            return None

        physical_names = tuple(sorted({
            name
            for segment in source_segments
            for name in segment_physical_names(segment)
        }, key=str.casefold))
        display_name = " + ".join(physical_names)
        entity_tags = self._collect_ints(
            source_segments, "entity_tags")

        info = dict(geometric_info)
        info.update({
            "is_outer": is_outer,
            "loop_depth": int(first_info.get("loop_depth", 0)),
            "loop_id": int(first_info.get("loop_id", -1)),
            "source": "gmsh_api+cad+geometry",
            "physical_name": display_name,
            "physical_names": physical_names,
            "physical_tags": self._collect_ints(
                source_segments, "physical_tags"),
            "entity_tags": entity_tags,
            "cad_entity_types": self._collect_strings(
                source_segments, "cad_entity_types"),
            "surface_entity_tags": self._collect_ints(
                source_segments, "surface_entity_tags"),
            "merged_cad_entities": True,
            "merged_entity_count": len(entity_tags),
            "merged_segment_count": len(source_segments),
            "join_turn_limit_degrees": float(np.degrees(
                MAX_JOIN_TURN_RADIANS)),
        })
        return {
            "type": segment_type,
            "nodes": nodes,
            "coords": coords,
            "label": f"{display_name} | {geometric_label}",
            "info": info,
            "closed": closed,
        }

    def _restore_source_direction(
            self, nodes, source_segments, closed):
        """Orient an ordered component like its source topology."""
        directed_edges = {
            (int(a), int(b))
            for segment in source_segments
            for a, b in zip(
                segment["nodes"], segment["nodes"][1:])
        }
        if (
                len(nodes) >= 2
                and (nodes[0], nodes[1]) not in directed_edges
                and (nodes[1], nodes[0]) in directed_edges):
            nodes.reverse()

        if not closed:
            return
        is_outer = bool(
            segment_info(source_segments[0]).get(
                "is_outer", False))
        area = _signed_loop_area(self.mesh.nodes[nodes])
        if (is_outer and area < 0.0) or (
                not is_outer and area > 0.0):
            nodes.reverse()

    @staticmethod
    def _single_cad_kind(source_segments):
        kinds = {
            str(kind).strip().casefold()
            for segment in source_segments
            for kind in tuple_value(
                segment_info(segment).get(
                    "cad_entity_types", ()))
        }
        return next(iter(kinds))

    @staticmethod
    def _classification_matches(cad_kind, segment_type):
        if cad_kind == "circle":
            # Several short Circle entities can be a CAD approximation of an
            # ellipse.  The complete chain still has to pass the strict
            # ellipse fit and the join-turn gate before reaching this point.
            return segment_type in {"arc", "ellipse"}
        if cad_kind == "ellipse":
            return segment_type in {"arc", "ellipse"}
        if cad_kind in TESSELLATED_CONIC_CAD_KINDS:
            return segment_type in {"arc", "ellipse"}
        return False

    @staticmethod
    def _collect_ints(segments, key):
        return tuple(sorted({
            int(value)
            for segment in segments
            for value in tuple_value(
                segment_info(segment).get(key, ()))
        }))

    @staticmethod
    def _collect_strings(segments, key):
        return tuple(sorted({
            str(value)
            for segment in segments
            for value in tuple_value(
                segment_info(segment).get(key, ()))
        }, key=str.casefold))


def merge_compatible_cad_conics(mesh, segments):
    """Functional façade for the focused merger class."""
    return ConicSegmentMerger(mesh, segments).merge()
