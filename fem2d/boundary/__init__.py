"""Boundary topology, geometry, Gmsh semantics and user selection.

显式管线 (阶段 2 插件化重构后):
  topology         — mesh edges → validated/nested/oriented loops
  detectors        — 识别器注册表 (几何判定层: line → circle → ellipse
                     → general; 插件经 register_detector 接入, 见
                     docs/boundary_plugins.md)
  geometry         — 曲率/拟合原语 + classify facade (实现 = 注册表)
  physical_mapping — Gmsh Physical Group semantic mapping
  registry_mapping — exact Gmsh CAD/Physical registry mapping
  segment_builder  — ordered chains → public segment dictionaries
  conic_merge      — conservative CAD conic presentation merge
  selectors        — exact CLI boundary-name resolution
  naming           — public orchestration and reporting façade

原生实体信息 (Gmsh line/circle/ellipse/bspline) 沿管线经
``info["cad_entity_types"]`` 与 classify 的 ``native_entities`` 参数
传递不丢失 — 内置探测器不消费, 供插件识别器参考.
"""
from .detectors import (
    Detection,
    Detector,
    DetectorRegistry,
    default_registry,
    register_detector,
)
from . import plugins  # noqa: F401  — 正式插件默认注册 (轮 2, 注册即生效)
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
    "Detection",
    "Detector",
    "DetectorRegistry",
    "_resolve_edge_indices",
    "build_boundary_segments",
    "default_registry",
    "describe_geometry",
    "detect",
    "parse_edge_name",
    "print_segments",
    "register_detector",
    "segments_from_physical_curves",
    "segments_from_region_registry",
    "semantic_coverage",
    "validate_boundary_segments",
]
