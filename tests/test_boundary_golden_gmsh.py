"""边界金标准快照 — Gmsh 真实网格层 (skipif 无 gmsh).

覆盖 models/ 全部带边界曲线的模型, 双路径:
  automatic — detect (纯拓扑/几何)
  registry  — build_boundary_segments(registry=r.regions) (物理曲线 + CAD)

gmsh 网格节点编号随版本可能漂移, 故本层快照不含节点 ID 数组
(include_nodes=False) 与逐边法向明细; 锁定的是版本无关语义:
段集合/标签/info/边界边计数/法向一致性/打印输出. 逐位对比仍针对
入库金标准文本 (判别性).

生成器: FEM2D_UPDATE_GOLDEN=1 (仅本地, CI 不设置).
"""
import pytest

from tests.conftest import GMSH_AVAILABLE, GMSH_UNAVAILABLE_REASON
from tests.boundary_snapshot import (
    build_snapshot,
    compare_golden,
    edge_coverage_check,
)

from fem2d import Mesh, build_boundary_segments, detect_boundaries
from fem2d.gmsh_adapter import generate_from_geo
from fem2d.regions import canonical_edge

pytestmark = pytest.mark.skipif(
    not GMSH_AVAILABLE, reason=GMSH_UNAVAILABLE_REASON)

MODELS = [
    "demo_complex.geo",      # CPS4 Q4 网格 + 圆角矩形 + 半圆凹口 + 椭圆孔(20段Line近似)
    "multi_hole_plate.geo",  # CPS3, 3 组圆孔环 (24+24+25 段 Circle)
    "curved_beam.geo",       # CPS3, 曲梁: 圆弧内/外边界 + 直边端部
    "l_bracket.geo",         # CPS3, L 形支架 6 直边
]


def _load_model(name):
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "models" / name
    result = generate_from_geo(str(path))
    return Mesh(nodes=result.nodes, elements=result.elements,
                E=210e9, nu=0.3, thickness=0.01,
                elem_type=result.elem_type), result.regions


def _golden_stem(name):
    """金标准文件主干 — 去掉 .geo 扩展名 (文件名无点)."""
    return name.split(".")[0]


def _canonical_edges(segments):
    edges = set()
    for seg in segments:
        for a, b in zip(seg["nodes"], seg["nodes"][1:]):
            edges.add(canonical_edge(a, b))
    return edges


def _assert_mesh_coverage(mesh, segments):
    """版本无关覆盖校验: 边界边集合与网格一致 (节点 ID 逐位断言换为
    计数 + 完整性, 节点编号本身跨 gmsh 版本可漂移)."""
    mesh.build_connectivity()
    expected = {canonical_edge(a, b) for a, b in mesh.boundary_edges}
    found = _canonical_edges(segments)
    assert found == expected, (
        f"边界边覆盖漂移: 多 {len(found - expected)} 缺 {len(expected - found)}")


@pytest.mark.parametrize("name", MODELS)
def test_automatic_path_golden(name):
    mesh, _ = _load_model(name)
    segs = detect_boundaries(mesh)
    _assert_mesh_coverage(mesh, segs)
    compare_golden(
        f"gmsh_{_golden_stem(name)}_automatic.json",
        build_snapshot(mesh, segs, include_nodes=False))


@pytest.mark.parametrize("name", MODELS)
def test_registry_path_golden(name):
    mesh, registry = _load_model(name)
    segs = build_boundary_segments(mesh, registry=registry)
    _assert_mesh_coverage(mesh, segs)
    compare_golden(
        f"gmsh_{_golden_stem(name)}_registry.json",
        build_snapshot(mesh, segs, include_nodes=False))
