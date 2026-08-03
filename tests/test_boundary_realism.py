"""边界检测精度验证 — 闭环 / 多连通 / 定向 / 自交.

曾为模块级执行脚本 (无 def test_*), pytest 无法逐项统计 —
改为标准测试函数。
"""
import numpy as np

from fem2d import Mesh, detect_boundaries
from fem2d.boundary.predicates import orient2d
from fem2d.boundary.topology import has_boundary_self_intersection


def test_self_intersection_detection():
    """Z 形锯齿环自交; 矩形/三角形无自交."""
    zigzag = np.array([[0, 0], [5, 5], [0, 3], [5, 2], [0, 0]])
    assert has_boundary_self_intersection(zigzag), "Z 形自交环"
    rect = np.array([[0, 0], [10, 0], [10, 5], [0, 5], [0, 0]])
    assert not has_boundary_self_intersection(rect), "矩形无自交"
    tri = np.array([[0, 0], [5, 10], [10, 0], [0, 0]])
    assert not has_boundary_self_intersection(tri), "三角形无自交"


def test_ellipse_detect():
    """48 边形椭圆 (a=2, b=1) 必须检测为 ellipse/arc."""
    n = 48
    center = np.array([[0.0, 0.0]])
    pts = np.array(
        [[2*np.cos(2*np.pi*i/n), 1*np.sin(2*np.pi*i/n)]
         for i in range(n)])
    nodes = np.vstack([center, pts])
    elems = np.array(
        [[0, i + 1, i + 2 if i + 2 <= n else 1] for i in range(n)],
        dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
                thickness=0.01)
    segs = detect_boundaries(mesh)
    types = segs[0]['type'] if segs else 'none'
    assert types in ('ellipse', 'arc'), f"椭圆检测为 {types}"


def test_orient2d_precision():
    """三点近共线符号正确."""
    o = orient2d(0, 0, 1, 0, 0.5, 1e-14)
    assert o > 0, f"近共线正向: {o}"
    o2 = orient2d(0, 0, 1, 0, 0.5, -1e-14)
    assert o2 < 0, f"反向符号: {o2}"


def test_rectangle_four_lines():
    """规则矩形网格 (4×2 四边形) → 4 条直线边."""
    nx, ny = 4, 2
    nodes = np.array(
        [[i, j] for j in range(ny + 1) for i in range(nx + 1)],
        dtype=float)
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elems.extend(
                [[n0, n0+1, n0+nx+2], [n0, n0+nx+2, n0+nx+1]])
    mesh = Mesh(nodes=nodes, elements=np.array(elems, dtype=int),
                E=210e9, nu=0.3, thickness=0.01)
    segs = detect_boundaries(mesh)
    lines = [s for s in segs if s['type'] == 'line']
    assert len(lines) == len(segs) == 4, \
        f"{len(lines)}/{len(segs)} 段为直线 (期望 4/4)"
