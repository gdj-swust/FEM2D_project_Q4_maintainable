"""Boundary topology, geometry, Gmsh semantics and user selection.

子包结构:
  topology         — mesh edges → validated/nested/oriented loops
  geometry         — curvature and primitive classifiers
  physical_mapping — Gmsh Physical Group semantic mapping
  registry_mapping — exact Gmsh CAD/Physical registry mapping
  segment_builder  — ordered chains → public segment dictionaries
  conic_merge      — conservative CAD conic presentation merge
  selectors        — exact CLI boundary-name resolution
  naming           — public orchestration and reporting façade
"""

from .model import BoundaryDiagnostics, BoundaryIssue
from .naming import (
    _resolve_edge_indices,
    build_boundary_segments,
    describe_geometry,
    parse_edge_name,
    print_segments,
    segments_from_physical_curves,
    segments_from_region_registry,
    semantic_coverage,
    validate_boundary_segments,
)
from .topology import detect

__all__ = [
    "BoundaryDiagnostics",
    "BoundaryIssue",
    "_resolve_edge_indices",
    "build_boundary_segments",
    "describe_geometry",
    "detect",
    "parse_edge_name",
    "print_segments",
    "segments_from_physical_curves",
    "segments_from_region_registry",
    "semantic_coverage",
    "validate_boundary_segments",
]
