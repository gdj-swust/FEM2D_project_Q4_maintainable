"""极端拓扑边界识别判别性测试 (2026-08-04, P1-8).

自交环 / 嵌套孔两种极端网格的边界识别行为锁定 — 只锁当前行为,
禁止改动 fem2d/boundary/ 识别逻辑。网格纯手建 (无 gmsh 依赖,
参照 test_boundary_extreme 现有写法): 自交环必须响亮报错而非
静默错边; 嵌套孔边界边集合必须与显式钉死的期望集合完全一致。

判别性: 期望集合显式写死 (非取自网格) — 故意弄乱一条边界边
(如把孔边改共享) → 网格真实边界变化 → 集合断言必红。
"""
import numpy as np
import pytest

from fem2d import Mesh, build_boundary_segments, detect_boundaries


def _segment_edges(segments):
    """段节点对集合 (排序后) — 边界边集合的唯一规范形式."""
    edges = set()
    for seg in segments:
        for a, b in zip(seg["nodes"], seg["nodes"][1:]):
            edges.add(tuple(sorted((int(a), int(b)))))
    return edges


def _bowtie_mesh():
    """蝴蝶形边界: 两三角形共享左边缘 (0-2), 边界 0-1-2-3-0 自交于
    (0.5, 0.5) — 单条自交闭环."""
    nodes = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
    elems = np.array([[0, 1, 2], [0, 3, 2]])
    return Mesh(nodes=nodes, elements=elems, E=1e6, nu=0.3, thickness=1.0)


def _holed_plate_mesh():
    """8×8 板带 4×4 孔 (二级嵌套): 外环 0-3 + 内环 4-7, 8 三角形连接."""
    nodes = np.array([[0, 0], [8, 0], [8, 8], [0, 8],
                      [2, 2], [6, 2], [6, 6], [2, 6]], dtype=float)
    elems = np.array([[0, 1, 4], [0, 4, 7], [1, 2, 5], [1, 5, 4],
                      [2, 3, 6], [2, 6, 5], [3, 0, 7], [3, 7, 6]], dtype=int)
    return Mesh(nodes=nodes, elements=elems, E=1e6, nu=0.3, thickness=1.0)


# 嵌套孔期望边界边 (端点排序后) — 显式钉死, 不取自网格
# (判别性: 弄乱网格/识别器漏边 → 红)
EXPECTED_HOLE_EDGES = {(0, 1), (1, 2), (2, 3), (0, 3),
                       (4, 5), (5, 6), (6, 7), (4, 7)}


def test_self_intersecting_loop_rejected_loudly():
    """自交环必须响亮报错 — 静默返回错边段集合 (或自交识别丢失
    → 零面积报错) 都会使本测试变红."""
    m = _bowtie_mesh()
    with pytest.raises(ValueError, match="Self-intersecting"):
        detect_boundaries(m)
    with pytest.raises(ValueError, match="Self-intersecting"):
        build_boundary_segments(m)


def test_nested_hole_boundary_edge_set_exact():
    """嵌套孔: 识别出的边界边集合必须与显式期望完全一致 (4 外边 +
    4 孔边), 无静默错边 (内部边不得混入、孔边不得漏)."""
    m = _holed_plate_mesh()
    segments = build_boundary_segments(m)
    got = _segment_edges(segments)
    assert got == EXPECTED_HOLE_EDGES, (
        f"边界边集合漂移: 多 {sorted(got - EXPECTED_HOLE_EDGES)} "
        f"缺 {sorted(EXPECTED_HOLE_EDGES - got)}")
    outer = [s for s in segments if '外边' in s.get('label', '')]
    inner = [s for s in segments if '内孔' in s.get('label', '')]
    assert len(outer) == 4 and len(inner) == 4, \
        f"两层嵌套: {len(outer)}外边 {len(inner)}内孔 (期望 4+4)"
