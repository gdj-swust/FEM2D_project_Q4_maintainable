"""Normalize Gmsh Physical Group labels into semantic boundary groups."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from ..regions import canonical_edge
from .validation import validate_physical_curve_names

AUTO_ENTITY_NAME = re.compile(
    r"^(Line|Circle|Ellipse|Spline|BSpline)\d+$")
CAD_CURVE_KINDS = (
    "Line",
    "Circle",
    "Ellipse",
    "Spline",
    "BSpline",
)


@dataclass(frozen=True)
class PhysicalEdgeMap:
    """Validated semantic memberships and their CAD hard partitions."""

    boundary_edges: set
    named_edges: dict
    combined_names: dict


class PhysicalEdgeMapper:
    """Map T3D2/ELSET records and optional ``.geo`` aliases to mesh edges.

    ⚠️ 遗留路径 (2026-08-03 标记): 生产调用链 runner._import_mesh 恒传
    edge_labels=None, 本类只在 edge_labels 非空时激活 — 实际不可达。
    保留供 segments_from_physical_curves 公共 API 契约; 若未来复活,
    _geo_curve_aliases 的 5 种 kind 全猜 (Line3/Circle3 同 id 张冠李戴)
    需改为从 .geo 解析真实实体类型。
    """

    def __init__(
            self, mesh, edge_labels, edge_partitions,
            geo_path, diagnostics):
        self.mesh = mesh
        self.edge_labels = edge_labels
        self.edge_partitions = edge_partitions
        self.diagnostics = diagnostics
        self.curve_to_names = self._geo_curve_aliases(geo_path)

        mesh.build_connectivity()
        self.boundary_edges = {
            canonical_edge(a, b) for a, b in mesh.boundary_edges
        }
        self.mesh_edges = {
            canonical_edge(a, b) for a, b in mesh.edge_to_elems
        }
        self.memberships = defaultdict(set)
        self.boundary_by_name = defaultdict(set)
        self.internal_by_name = defaultdict(set)
        self.missing_by_name = defaultdict(set)
        self.unmapped_by_name = defaultdict(list)

    def build(self):
        for (a, b), raw_value in self.edge_labels.items():
            self._record_edge(
                canonical_edge(a, b),
                self._semantic_names_for_value(raw_value),
            )
        for record in getattr(
                self.edge_labels, "unmapped_records", ()):
            names = self._semantic_names_for_value(
                record.get("names", ()))
            for name in names:
                self.unmapped_by_name[name].append(record)

        validate_physical_curve_names(
            self.diagnostics.declared_physical_names,
            self.diagnostics,
        )
        self._report_invalid_memberships()
        named_edges, combined_names = self._combined_groups()
        return PhysicalEdgeMap(
            boundary_edges=self.boundary_edges,
            named_edges=named_edges,
            combined_names=combined_names,
        )

    @staticmethod
    def _geo_curve_aliases(geo_path):
        aliases = defaultdict(set)
        if not geo_path:
            return aliases

        # Local import avoids coupling preprocess module initialization to the
        # boundary package.
        from ..preprocess import read_geo_groups

        for name, entity_ids in (
                read_geo_groups(geo_path) or {}).items():
            for entity_id in entity_ids:
                for kind in CAD_CURVE_KINDS:
                    aliases[f"{kind}{entity_id}"].add(name)
        return aliases

    def _semantic_names_for_value(self, raw_value):
        raw_names = (
            (raw_value,)
            if isinstance(raw_value, str)
            else tuple(raw_value)
        )
        names = set()
        for raw_name in map(str, raw_names):
            if raw_name in self.curve_to_names:
                names.update(self.curve_to_names[raw_name])
            elif not AUTO_ENTITY_NAME.fullmatch(raw_name):
                names.add(raw_name)
        self.diagnostics.register_declared(names)
        return names

    def _record_edge(self, edge, names):
        if edge in self.boundary_edges:
            self.memberships[edge].update(names)
            for name in names:
                self.boundary_by_name[name].add(edge)
        elif edge in self.mesh_edges:
            for name in names:
                self.internal_by_name[name].add(edge)
        else:
            for name in names:
                self.missing_by_name[name].add(edge)

    def _report_invalid_memberships(self):
        for name in self.diagnostics.declared_physical_names:
            internal = self.internal_by_name[name]
            boundary_part = self.boundary_by_name[name]
            if internal:
                code = (
                    "physical_curve_partly_internal"
                    if boundary_part else "physical_curve_internal")
                self.diagnostics.add(
                    code,
                    "error",
                    f"Physical Curve {name!r} contains "
                    f"{len(internal)} internal 2-D mesh edges; only exterior "
                    "domain edges can receive boundary traction/pressure.",
                    physical_name=name,
                    edge_count=len(internal),
                )
            missing = self.missing_by_name[name]
            if missing:
                self.diagnostics.add(
                    "physical_curve_missing_edges",
                    "error",
                    f"Physical Curve {name!r} references "
                    f"{len(missing)} edges absent from the displacement mesh.",
                    physical_name=name,
                    edge_count=len(missing),
                )
            unmapped = self.unmapped_by_name[name]
            if unmapped:
                self.diagnostics.add(
                    "physical_curve_unmapped_nodes",
                    "error",
                    f"Physical Curve {name!r} has {len(unmapped)} edge "
                    "records whose nodes were removed or never belonged to "
                    "the 2-D displacement mesh.",
                    physical_name=name,
                    edge_count=len(unmapped),
                )

    def _combined_groups(self):
        named_edges = defaultdict(set)
        combined_names = {}
        for edge, names in self.memberships.items():
            if not names:
                continue
            ordered_names = tuple(sorted(
                names, key=str.casefold))
            display_name = " + ".join(ordered_names)
            combined_names[display_name] = ordered_names
            partition = tuple(sorted(
                int(tag)
                for tag in self.edge_partitions.get(edge, ())))
            named_edges[(display_name, partition)].add(edge)
        return named_edges, combined_names


def map_physical_edges(
        mesh, edge_labels, edge_partitions, geo_path, diagnostics):
    """Functional façade for legacy edge-label mapping."""
    return PhysicalEdgeMapper(
        mesh,
        edge_labels,
        edge_partitions,
        geo_path,
        diagnostics,
    ).build()
