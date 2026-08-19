"""Normalize Gmsh Physical Group labels into semantic boundary groups.

生产可达 (2026-08-19 实证修订): 带 Physical Curve 的 gmsh 模型必经
runner.build_boundary_segments → naming 模块 registry 分支 →
segments_from_region_registry → segments_from_physical_curves →
map_physical_edges 工厂实例化 PhysicalEdgeMapper; 另有
tests/test_clean_round.py 直接导入与 docs/api_contract.md 锁定的
内部 import 测试注入点 — 本模块不可删。
"""
from __future__ import annotations

import re
import warnings
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

    生产 registry 路径可达: 带 Physical Curve 的 gmsh 模型经
    naming 模块 registry 分支 → segments_from_physical_curves →
    map_physical_edges 工厂实例化本类; 测试与公共契约锁定, 不可删。
    遗留注意: _geo_curve_aliases 的 5 种 kind 全猜 (Line3/Circle3
    同 id 张冠李戴), 如需精确实体类型应从 .geo 解析。
    """

    def __init__(
            self, mesh, edge_labels, edge_partitions,
            geo_path, diagnostics):
        warnings.warn(
            "PhysicalEdgeMapper 已弃用: 改用 map_physical_edges() 工厂; "
            "类计划在 v10 删除.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        # 保持局部 import: build_boundary_segments 的 geo 别名解析是测试
        # patch 点 (patch("fem2d.preprocess.read_geo_groups") 注入) —
        # 模块级直接名绑定会复制旧函数引用, 使 patch 失效
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mapper = PhysicalEdgeMapper(
            mesh,
            edge_labels,
            edge_partitions,
            geo_path,
            diagnostics,
        )
    return mapper.build()
