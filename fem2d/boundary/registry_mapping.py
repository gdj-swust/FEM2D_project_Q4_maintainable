"""Map an in-memory Gmsh region registry onto displacement-mesh edges."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..regions import canonical_edge
from .validation import validate_physical_curve_names


@dataclass(frozen=True)
class RegionBoundaryMap:
    """Normalized inputs consumed by the generic chain reconstruction path."""

    edge_labels: dict
    edge_partitions: dict
    edge_metadata: dict
    physical_metadata: dict
    edge_entity_tags: dict
    edge_entity_types: dict


class RegionBoundaryMapper:
    """Validate CAD ownership and preserve exact Physical Curve overlap."""

    def __init__(self, mesh, registry, diagnostics):
        self.mesh = mesh
        self.registry = registry
        self.diagnostics = diagnostics

        mesh.build_connectivity()
        self.boundary_edges = {
            canonical_edge(a, b) for a, b in mesh.boundary_edges
        }
        self.mesh_edges = {
            canonical_edge(a, b) for a, b in mesh.edge_to_elems
        }
        self.memberships = defaultdict(set)
        self.physical_metadata = defaultdict(lambda: {
            "physical_tags": set(),
            "entity_tags": set(),
            "entity_types": set(),
        })
        self.edge_entity_tags = defaultdict(set)
        self.edge_entity_types = defaultdict(set)
        self.edge_surface_tags = defaultdict(set)

    def build(self):
        """Return one deterministic boundary map for downstream segmentation."""
        self.diagnostics.register_declared(
            region.name for region in self.registry.curves)
        validate_physical_curve_names(
            self.diagnostics.declared_physical_names,
            self.diagnostics,
        )
        for cad_curve in getattr(
                self.registry, "cad_curves", ()):
            self._record_cad_curve(cad_curve)
        self._validate_cad_partition()
        for region in self.registry.curves:
            self._record_physical_region(region)
        return self._export()

    def _record_cad_curve(self, cad_curve):
        """Validate and index one authoritative Gmsh CAD curve entity."""
        cad_edges = {
            canonical_edge(*edge)
            for edge in cad_curve.edge_pairs
        }
        occurrences = tuple(cad_curve.surface_occurrences)
        surface_tags = {
            int(surface_tag)
            for surface_tag, _ in occurrences
        }
        missing = cad_edges - self.mesh_edges
        boundary_part = cad_edges & self.boundary_edges
        internal_part = (
            cad_edges & self.mesh_edges
        ) - self.boundary_edges

        if getattr(
                self.registry, "cad_boundary_complete", False):
            self._validate_complete_cad_curve(
                cad_curve,
                cad_edges,
                occurrences,
                surface_tags,
                missing,
                boundary_part,
                internal_part,
            )

        for edge in cad_edges & self.boundary_edges:
            self.edge_entity_tags[edge].add(
                int(cad_curve.entity_tag))
            self.edge_entity_types[edge].add(
                str(cad_curve.entity_type))
            self.edge_surface_tags[edge].update(
                int(surface_tag)
                for surface_tag, _ in occurrences)

    def _validate_complete_cad_curve(
            self, curve, edges, occurrences, surface_tags,
            missing, boundary_part, internal_part):
        """Check one curve under a complete Gmsh CAD-boundary contract."""
        if not edges:
            self.diagnostics.add(
                "cad_curve_empty",
                "error",
                f"Active CAD boundary curve entity "
                f"{curve.entity_tag} ({curve.entity_type}) "
                "contains no usable first-order mesh edges.",
                entity_tag=curve.entity_tag,
            )
        if not occurrences:
            self.diagnostics.add(
                "cad_curve_unowned",
                "error",
                f"CAD curve entity {curve.entity_tag} has no active "
                "surface occurrence in a complete Gmsh registry.",
                entity_tag=curve.entity_tag,
                edge_count=len(edges),
            )
        if len(surface_tags) > 2:
            self.diagnostics.add(
                "cad_curve_nonmanifold",
                "error",
                f"CAD curve entity {curve.entity_tag} bounds "
                f"{len(surface_tags)} active surfaces; a 2-D manifold "
                "curve may bound at most two.",
                entity_tag=curve.entity_tag,
                edge_count=len(edges),
            )
        if missing:
            self.diagnostics.add(
                "cad_curve_missing_edges",
                "error",
                f"CAD curve entity {curve.entity_tag} contains "
                f"{len(missing)} line-mesh edges absent from the "
                "displacement mesh.",
                entity_tag=curve.entity_tag,
                edge_count=len(missing),
            )
        if len(surface_tags) == 1 and internal_part:
            self.diagnostics.add(
                "cad_boundary_internal_mismatch",
                "error",
                f"CAD curve entity {curve.entity_tag} bounds one "
                f"active surface but {len(internal_part)} of its edges "
                "are internal in the displacement mesh.",
                entity_tag=curve.entity_tag,
                edge_count=len(internal_part),
            )
        elif len(surface_tags) >= 2 and boundary_part:
            self.diagnostics.add(
                "cad_interface_exposed",
                "error",
                f"CAD curve entity {curve.entity_tag} is shared by "
                f"{len(surface_tags)} active surfaces but "
                f"{len(boundary_part)} of its edges appear on the "
                "external mesh boundary.",
                entity_tag=curve.entity_tag,
                edge_count=len(boundary_part),
            )
        self._validate_interface_orientation(
            curve, occurrences, surface_tags, edges)

    def _validate_interface_orientation(
            self, curve, occurrences, surface_tags, edges):
        if len(surface_tags) != 2:
            return
        signs = {
            1 if int(oriented_tag) > 0 else -1
            for _, oriented_tag in occurrences
        }
        if len(signs) == 1:
            self.diagnostics.add(
                "cad_interface_orientation",
                "warning",
                f"Shared CAD curve entity {curve.entity_tag} has "
                "the same orientation in both adjacent surfaces; "
                "verify surface orientation in Gmsh.",
                entity_tag=curve.entity_tag,
                edge_count=len(edges),
            )

    def _validate_cad_partition(self):
        for edge, entity_tags in self.edge_entity_tags.items():
            if len(entity_tags) > 1:
                self.diagnostics.add(
                    "cad_entity_overlap",
                    "error",
                    f"Boundary edge {edge} belongs to multiple CAD curve "
                    f"entities {sorted(entity_tags)}.",
                    edge_count=1,
                )

        if not getattr(
                self.registry, "cad_boundary_complete", False):
            return
        uncovered = (
            self.boundary_edges - set(self.edge_entity_tags))
        if uncovered:
            self.diagnostics.add(
                "cad_boundary_coverage_gap",
                "error",
                f"Complete Gmsh CAD metadata covers "
                f"{len(self.boundary_edges) - len(uncovered)}/"
                f"{len(self.boundary_edges)} external mesh edges; "
                f"{len(uncovered)} edges have no owning CAD curve entity.",
                edge_count=len(uncovered),
            )

    def _record_physical_region(self, region):
        """Validate one Physical Curve and attach all exact memberships."""
        metadata = self.physical_metadata[region.name]
        metadata["physical_tags"].add(int(region.physical_tag))
        metadata["entity_tags"].update(region.entity_tags)
        metadata["entity_types"].update(region.entity_types)

        exact_tags, exact_types = self._exact_edge_entities(region)
        region_edges = {
            canonical_edge(*edge) for edge in region.edge_pairs
        }
        boundary_part = region_edges & self.boundary_edges
        internal = (
            region_edges & self.mesh_edges
        ) - self.boundary_edges
        self._validate_physical_region(
            region,
            region_edges,
            boundary_part,
            internal,
            exact_tags,
        )
        for edge in boundary_part:
            self.memberships[edge].add(region.name)
            if edge in exact_tags:
                self.edge_entity_tags[edge].update(
                    exact_tags[edge])
                self.edge_entity_types[edge].update(
                    exact_types[edge])
            else:
                # Compatibility path for registries without exact per-edge
                # provenance. Group-level data stays advisory.
                self.edge_entity_tags[edge].update(
                    region.entity_tags)
                self.edge_entity_types[edge].update(
                    region.entity_types)

    @staticmethod
    def _exact_edge_entities(region):
        exact_tags = defaultdict(set)
        exact_types = defaultdict(set)
        for a, b, entity_tag, entity_type in tuple(
                getattr(region, "edge_entities", ())):
            edge = canonical_edge(a, b)
            exact_tags[edge].add(int(entity_tag))
            exact_types[edge].add(str(entity_type))
        return exact_tags, exact_types

    def _validate_physical_region(
            self, region, region_edges, boundary_part,
            internal, exact_tags):
        missing = region_edges - self.mesh_edges
        if exact_tags:
            meshed_tags = {
                tag for tags in exact_tags.values() for tag in tags
            }
            unmeshed_tags = (
                set(map(int, region.entity_tags)) - meshed_tags)
            if unmeshed_tags:
                self.diagnostics.add(
                    "physical_curve_unmeshed_entity",
                    "error",
                    f"Physical Curve {region.name!r} contains CAD entities "
                    f"{sorted(unmeshed_tags)} with no usable line mesh in "
                    "the displacement domain.",
                    physical_name=region.name,
                )
        if not region_edges:
            self.diagnostics.add(
                "physical_curve_empty",
                "error",
                f"Physical Curve {region.name!r} contains no active mesh "
                "edges.",
                physical_name=region.name,
            )
        if missing:
            self.diagnostics.add(
                "physical_curve_missing_edges",
                "error",
                f"Physical Curve {region.name!r} references "
                f"{len(missing)} edges absent from the displacement mesh.",
                physical_name=region.name,
                edge_count=len(missing),
            )
        if internal:
            code = (
                "physical_curve_partly_internal"
                if boundary_part else "physical_curve_internal")
            self.diagnostics.add(
                code,
                "error",
                f"Physical Curve {region.name!r} contains "
                f"{len(internal)} internal 2-D mesh edges; only exterior "
                "domain edges can receive boundary traction/pressure.",
                physical_name=region.name,
                edge_count=len(internal),
            )

    def _export(self):
        edge_labels = {
            edge: tuple(sorted(names, key=str.casefold))
            for edge, names in self.memberships.items()
        }
        edge_partitions = {
            edge: tuple(sorted(tags))
            for edge, tags in self.edge_entity_tags.items()
            if edge in self.boundary_edges and tags
        }
        edge_metadata = {
            edge: {
                "entity_tags": tuple(sorted(
                    self.edge_entity_tags[edge])),
                "cad_entity_types": tuple(sorted(
                    self.edge_entity_types[edge])),
                "surface_entity_tags": tuple(sorted(
                    self.edge_surface_tags[edge])),
            }
            for edge in self.boundary_edges
            if self.edge_entity_tags[edge]
        }
        return RegionBoundaryMap(
            edge_labels=edge_labels,
            edge_partitions=edge_partitions,
            edge_metadata=edge_metadata,
            physical_metadata=self.physical_metadata,
            edge_entity_tags=self.edge_entity_tags,
            edge_entity_types=self.edge_entity_types,
        )


def map_region_registry(mesh, registry, diagnostics):
    """Functional façade for exact Gmsh registry mapping."""
    return RegionBoundaryMapper(
        mesh, registry, diagnostics).build()
