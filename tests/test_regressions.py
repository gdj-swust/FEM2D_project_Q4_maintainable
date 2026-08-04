"""回归测试: 覆盖本轮所有关键修复, 防止未来重构静默破坏."""
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# ═══════════════════════════════════════════════════════════════
# 1. .geo 原文件保护
# ═══════════════════════════════════════════════════════════════

def test_geo_original_not_deleted():
    """不改 lc 时原始 .geo 文件不被修改或删除."""
    import hashlib
    geo = str(MODELS_DIR / 'test_spec.geo')
    assert os.path.isfile(geo)
    with open(geo, 'rb') as f:
        h1 = hashlib.sha256(f.read()).hexdigest()
    # 模拟 run.py .geo 处理路径 (不调用 Gmsh)
    from fem2d.preprocess import parse_geo_fem_config
    parse_geo_fem_config(geo)
    with open(geo, 'rb') as f:
        h2 = hashlib.sha256(f.read()).hexdigest()
    assert h1 == h2, f'.geo file was modified! {h1} != {h2}'


def test_physical_curve_square_stays_lines():
    """整个方形一个 Physical Curve → 仍得到 4 条直边，非圆弧."""
    from fem2d import Mesh
    from fem2d.boundary.naming import segments_from_physical_curves
    # 构造: 方形四边在同一个 Physical Curve "square"
    # (原 .inp T3D2/ELSET 语义已随 read_inp 移除 — 边标签映射直接构造)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
             thickness=0.01, elem_type='CPS3')
    labels = {
        (0, 1): 'square', (1, 2): 'square',
        (2, 3): 'square', (3, 0): 'square',
    }
    segs = segments_from_physical_curves(m, labels)
    assert segs is not None, 'Physical Curve should return segments'
    assert len(segs) == 4, f'Expected 4 edges, got {len(segs)}'
    for s in segs:
        assert s['type'] == 'line', 'Square edge should be line, got ' + s['type']


def test_physical_curves_none_labels_returns_none():
    """生产守卫: edge_labels 为 None (生产链路恒如此) 必须返回 None —
    曾无此测试, 未来重构可能让 legacy 映射在生产路径复活."""
    from fem2d import Mesh
    from fem2d.boundary.naming import segments_from_physical_curves
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
             thickness=0.01, elem_type='CPS3')
    assert segments_from_physical_curves(m, None) is None
    assert segments_from_physical_curves(m, {}) is None


def test_small_deformation_asserts_false():
    """大载荷 → small_deformation_ok 必须为 False + RuntimeWarning."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    mesh.fix_node(0, 'both', 0.0); mesh.fix_node(2, 'both', 0.0)
    mesh.add_force(3, 1e15, 0.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = solve(mesh, verbose=False)
        assert result['small_deformation_ok'] is False, 'Must detect large deformation'
        assert len([x for x in w if issubclass(x.category, RuntimeWarning)]) > 0


# ═══════════════════════════════════════════════════════════════
# 2. 嵌套边界: 外域→孔→材料岛
# ═══════════════════════════════════════════════════════════════

def test_nested_boundary_island_in_hole():
    """孔中材料岛应被识别为外边 (depth=2, even)."""
    from fem2d import Mesh, detect_boundaries
    # 外边 6x6, 孔 2x2@(2,2), 岛 0.5x0.5@(2.75,2.75)
    nodes = np.array([
        [0,0],[6,0],[6,6],[0,6],           # 外边 0-3
        [2,2],[4,2],[4,4],[2,4],           # 孔 4-7
        [2.75,2.75],[3.25,2.75],[3.25,3.25],[2.75,3.25],  # 岛 8-11
    ], dtype=float)
    elems = np.array([
        [0,1,2],[0,2,3],
        [4,5,6],[4,6,7],
        [8,9,10],[8,10,11],
    ], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    segs = detect_boundaries(mesh)
    outer = [s for s in segs if '外边' in s.get('label', '')]
    inner = [s for s in segs if '内孔' in s.get('label', '')]
    # 外边4段 + 岛4段 = 8 outer, 孔4段 = 4 inner
    assert len(outer) == 8, f"Expected 8 outer (4 border + 4 island), got {len(outer)}"
    assert len(inner) == 4, f"Expected 4 inner (hole), got {len(inner)}"


def test_two_disconnected_squares_both_outer():
    """两个断开方板应该都是外边."""
    from fem2d import Mesh, detect_boundaries
    nodes = np.array([
        [0,0],[1,0],[1,1],[0,1],
        [3,0],[4,0],[4,1],[3,1],
    ], dtype=float)
    elems = np.array([
        [0,1,2],[0,2,3],
        [4,5,6],[4,6,7],
    ], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    segs = detect_boundaries(mesh)
    outer = [s for s in segs if '外边' in s.get('label', '')]
    inner = [s for s in segs if '内孔' in s.get('label', '')]
    assert len(outer) == 8, f"Expected 8 outer, got {len(outer)}"
    assert len(inner) == 0, f"Expected 0 inner, got {len(inner)}"


# ═══════════════════════════════════════════════════════════════
# 3. 容差校验
# ═══════════════════════════════════════════════════════════════

def test_nodes_on_edge_tol_validation():
    """tol=-1, NaN, Inf 必须报错."""
    from fem2d import Mesh
    nodes = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    elems = np.array([[0,1,2],[1,3,2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for bad_tol in [-1, -0.1, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            mesh.nodes_on_edge('x', 'min', tol=bad_tol)

def test_nodes_on_edge_tol_zero():
    """tol=0 只选严格相等的节点."""
    from fem2d import Mesh
    nodes = np.array([[0,0],[1e-9,0],[1,0]], dtype=float)
    elems = np.array([[0,1,2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    result = mesh.nodes_on_edge('x', 'min', tol=0)
    assert len(result) == 1, f"tol=0 should select exactly 1 node, got {len(result)}"

def test_nodes_on_edge_micro_scale():
    """1μm 模型: 距边界5nm的节点不应被选为边界节点."""
    from fem2d import Mesh
    # 1μm × 1μm 模型
    nodes = np.array([[0,0],[5e-9,5e-9],[1e-6,1e-6]], dtype=float)
    elems = np.array([[0,1,2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=1e-6)
    result = mesh.nodes_on_edge('x', 'min')
    assert len(result) == 1, f"Micro-scale: expected 1 node on edge, got {len(result)}"


# ═══════════════════════════════════════════════════════════════
# 4. 求解器
# ═══════════════════════════════════════════════════════════════

def test_uniform_translation_zero_stress():
    """均匀平移应产生零应力."""
    from fem2d import Mesh, solve
    nodes = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    elems = np.array([[0,1,2],[1,3,2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for i in range(4):
        mesh.fix_node(i, 'both', 1.0)
    result = solve(mesh, verbose=False)
    assert np.max(np.abs(result['stress'])) < 1e-10

def test_penalty_has_reactions():
    """penalty 法应返回反力."""
    from fem2d import Mesh, solve
    nodes = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    elems = np.array([[0,1,2],[1,3,2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    mesh.fix_node(0, 'both', 0.0); mesh.fix_node(2, 'both', 0.0)
    mesh.add_force(3, 1000.0, 0.0)
    result = solve(mesh, method='penalty', verbose=False)
    assert result['reactions'] is not None, "penalty must return reactions"
    assert len(result['reactions']) > 0


# ═══════════════════════════════════════════════════════════════
# 5. gmsh 物理组路径: test_spec 孔整圆回归
# ═══════════════════════════════════════════════════════════════

def test_test_spec_hole_is_circle():
    """test_spec 模型的孔必须为闭合整圆 (gmsh 物理组路径).

    原实现从 test_spec.inp 的 T3D2/ELSET 重建边界 (read_inp 已移除);
    现在走生产路径: .geo → generate_from_geo (Gmsh API) → registry.
    """
    from collections import Counter

    from fem2d import Mesh
    from fem2d.boundary.naming import build_boundary_segments
    from fem2d.gmsh_adapter import GmshUnavailableError, generate_from_geo
    geo = str(MODELS_DIR / 'test_spec.geo')
    if not os.path.isfile(geo):
        pytest.skip('models/test_spec.geo is unavailable')
    try:
        g = generate_from_geo(geo)
    except GmshUnavailableError:
        pytest.skip(
            'gmsh Python API unavailable — skipping gmsh-dependent test')
    m = Mesh(nodes=g.nodes, elements=g.elements, E=210e9, nu=0.3,
             thickness=0.01, elem_type=g.elem_type)
    segs = build_boundary_segments(m, registry=g.regions)
    assert segs is not None, 'Physical Curve should return segments'
    holes = [s for s in segs if '内孔' in s.get('label', '')]
    assert len(holes) > 0, 'Expected hole segments, got 0'
    for h in holes:
        assert h['type'] == 'arc', 'Hole edge should be arc, got ' + h['type']
    # 孔由若干段圆弧组成 (每个 Circle CAD 实体一段), 必须同圆心同半径
    centers = {tuple(np.round(h['info']['center'], 6)) for h in holes}
    radii = {round(h['info']['radius'], 6) for h in holes}
    assert len(centers) == 1, f'Hole arcs must share one center, got {centers}'
    assert len(radii) == 1, f'Hole arcs must share one radius, got {radii}'
    # 各段角度之和必须等于整圆 2π (孔不能是半个圆或缺角)
    total_angle = sum(h['info']['angle'] for h in holes)
    assert abs(total_angle - 2 * np.pi) < 1e-6, (
        f'Hole arcs must tile a full circle, total angle {total_angle}')
    # 闭环: 每个端点恰好被两个弧段共享 (闭合整圆的拓扑属性)
    endpoints = Counter()
    for h in holes:
        endpoints[tuple(h['coords'][0])] += 1
        endpoints[tuple(h['coords'][-1])] += 1
    assert all(v == 2 for v in endpoints.values()), (
        'Hole arcs must form a closed loop')
