"""边界检测极端测试 — 退化/多孔/嵌套/近切.

曾为模块级执行脚本 (无 def test_*), pytest 无法逐项统计 —
改为标准测试函数 (第四轮外部审查)。
"""
import numpy as np

from fem2d import Mesh, detect_boundaries
from fem2d.boundary.topology import _point_in_loop


def test_tiny_polygon_point_in_loop():
    """边长 ~0.0001 的小正方形."""
    tiny_x = np.array([0, 0.0001, 0.0001, 0])
    tiny_y = np.array([0, 0, 0.0001, 0.0001])
    assert _point_in_loop(5e-5, 5e-5, tiny_x, tiny_y), "极小正方形内"
    assert not _point_in_loop(1, 1, tiny_x, tiny_y), "极小正方形外"


def test_near_collinear_orient2d():
    """三点近共线 orient2d 符号必须正确."""
    from fem2d.boundary.predicates import orient2d
    for offset in (1e-12, 1e-14, 1e-15, 1e-16, 0):
        o = orient2d(0, 0, 1, 0, 0.5, offset)
        if offset > 0:
            assert o > 0, f"offset={offset}: o={o}"
        elif offset < 0:
            assert o < 0, f"offset={offset}: o={o}"
        else:
            assert o == 0, f"共线: o={o}"


def test_point_on_edge_and_vertex():
    """测点恰在边上/顶点 — 文档语义: 判为内部 (曾字面 True 占位)."""
    square_x = np.array([0, 10, 10, 0])
    square_y = np.array([0, 0, 10, 10])
    assert _point_in_loop(5, 5, square_x, square_y), "(5,5) 内部"
    assert not _point_in_loop(15, 5, square_x, square_y), "(15,5) 外部"
    assert _point_in_loop(5, 0, square_x, square_y) is True, "(5,0) 底边上"
    assert _point_in_loop(0, 0, square_x, square_y) is True, "(0,0) 顶点"


def test_u_shape_non_convex():
    """U 形 (非凸): 主体/凹槽/底部/右柱."""
    ux = np.array([0, 10, 10, 8, 8, 2, 2, 0])
    uy = np.array([0, 0, 10, 10, 2, 2, 10, 10])
    assert _point_in_loop(1, 5, ux, uy), "U形左柱 (1,5)"
    assert not _point_in_loop(5, 5, ux, uy), "U形凹槽 (5,5) 外部"
    assert _point_in_loop(5, 1, ux, uy), "U形底部 (5,1)"
    assert _point_in_loop(9, 5, ux, uy), "U形右柱 (9,5)"


def test_extreme_aspect_ratio():
    """极细长矩形 (高宽比 1000:1)."""
    thin_x = np.array([0, 1000, 1000, 0])
    thin_y = np.array([0, 0, 1, 1])
    assert _point_in_loop(500, 0.5, thin_x, thin_y), "(500,0.5) 在内"
    assert not _point_in_loop(-1, 0.5, thin_x, thin_y), "(-1,0.5) 在外"


def test_island_in_hole_three_levels():
    """岛中孔 (嵌套三层): 外边 10×10, 孔 6×6, 岛 2×2 —
    精确锁定 4 外边段 + 4 内孔段 (曾只断言 inner>0)."""
    nodes = np.array([
        [0, 0], [10, 0], [10, 10], [0, 10],     # 外边 0-3
        [2, 2], [8, 2], [8, 8], [2, 8],         # 孔 4-7
        [4, 4], [6, 4], [6, 6], [4, 6],         # 岛 8-11
    ], dtype=float)
    elems = np.array([
        [0, 1, 4], [0, 4, 7], [1, 2, 5], [1, 5, 4],
        [2, 3, 6], [2, 6, 5], [3, 0, 7], [3, 7, 6],
        [4, 5, 8], [4, 8, 11], [5, 6, 9], [5, 9, 8],
        [6, 7, 10], [6, 10, 9], [7, 4, 11], [7, 11, 10],
    ], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
                thickness=0.01)
    segs = detect_boundaries(mesh)
    outer = [s for s in segs if '外边' in s.get('label', '')]
    inner = [s for s in segs if '内孔' in s.get('label', '')]
    assert len(outer) == 4 and len(inner) == 4, \
        f"三层嵌套: {len(outer)}外边 {len(inner)}内孔 (期望 4+4)"
