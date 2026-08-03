"""FEM2D — extensible 2-D displacement finite-element solver."""

from .assembly import (
    assemble_expand,
    assemble_lil_reference,
    assemble_sparse,
    assemble_sparse_vectorized,
)
from .bc import apply_elimination, apply_penalty
from .boundary import (
    BoundaryDiagnostics,
    BoundaryIssue,
    build_boundary_segments,
    describe_geometry,
    parse_edge_name,
    print_segments,
    segments_from_physical_curves,
    segments_from_region_registry,
    semantic_coverage,
    validate_boundary_segments,
)
from .boundary import (
    detect as detect_boundaries,
)
from .element import (
    # CST
    CST,
    # Q4 / Q4R
    Q4,
    Q4I,
    Q4R,
    CSTElement,
    # Kernel protocol & registry
    ElementKernel,
    Q4Element,
    Q4IElement,
    Q4RElement,
    get_element_kernel,
    register_element,
    registered_element_types,
    verify_all_elements,
)
from .error_est import estimate as estimate_error
from .gmsh_adapter import (
    GmshImportResult,
    GmshTopologyError,
    GmshUnavailableError,
    generate_from_geo,
)
from .loads import (
    assemble as assemble_loads,
)
from .loads import (
    make_edge_profile_func,
    parse_traction,
    parse_vec2,
)
from .material import D_matrix, von_mises
from .mesh import Mesh
from .patch_test import run_patch_test
from .preprocess import (
    parse_geo_fem_config,
    parse_spec_config,
    read_geo_groups,
)

# run_cantilever_convergence 经 __getattr__ 惰性导出 (见文末):
# 避免 __init__ 急切导入 convergence 模块 (python -m fem2d.convergence
# 会触发 runpy 警告), 同时保持顶层 API 兼容。
from .quality import (
    evaluate as evaluate_mesh_quality,
)
from .quality import (
    report as report_mesh_quality,
)
from .regions import (
    CadCurveRegion,
    CurveRegion,
    PointRegion,
    RegionRegistry,
    SurfaceRegion,
)
from .solver import estimate_condition, solve
from .spr import spr_recovery
from .stress import (
    compute_stresses,
    nodal_L2_projection,
    nodal_simple,
    nodal_weighted,
    point_in_element,
    principal_stresses,
    stress_at_point,
)

__all__ = [
    "CST",
    "Q4",
    "Q4I",
    "Q4R",
    "BoundaryDiagnostics",
    "BoundaryIssue",
    "CSTElement",
    "CadCurveRegion",
    "CurveRegion",
    "D_matrix",
    "ElementKernel",
    "GmshImportResult",
    "GmshTopologyError",
    "GmshUnavailableError",
    "Mesh",
    "PointRegion",
    "Q4Element",
    "Q4IElement",
    "Q4RElement",
    "RegionRegistry",
    "SurfaceRegion",
    "apply_elimination",
    "apply_penalty",
    "assemble_expand",
    "assemble_lil_reference",
    "assemble_loads",
    "assemble_sparse",
    "assemble_sparse_vectorized",
    "build_boundary_segments",
    "compute_stresses",
    "describe_geometry",
    "detect_boundaries",
    "estimate_condition",
    "estimate_error",
    "evaluate_mesh_quality",
    "generate_from_geo",
    "get_element_kernel",
    "make_edge_profile_func",
    "nodal_L2_projection",
    "nodal_simple",
    "nodal_weighted",
    "parse_edge_name",
    "parse_geo_fem_config",
    "parse_spec_config",
    "parse_traction",
    "parse_vec2",
    "point_in_element",
    "principal_stresses",
    "print_segments",
    "read_geo_groups",
    "register_element",
    "registered_element_types",
    "report_mesh_quality",
    "run_cantilever_convergence",
    "run_patch_test",
    "segments_from_physical_curves",
    "segments_from_region_registry",
    "semantic_coverage",
    "solve",
    "spr_recovery",
    "stress_at_point",
    "validate_boundary_segments",
    "verify_all_elements",
    "von_mises",
]


def __getattr__(name):
    """PEP 562 惰性导出 — 顶层 API 兼容且不急切导入子模块.

    ``from fem2d import run_cantilever_convergence`` 仍可用, 但 convergence
    模块只在真正访问该名字时才导入 (避免 ``python -m fem2d.convergence``
    的 runpy 提前导入警告)。
    """
    if name == "run_cantilever_convergence":
        from .convergence import run_cantilever_convergence
        return run_cantilever_convergence
    raise AttributeError(name)
