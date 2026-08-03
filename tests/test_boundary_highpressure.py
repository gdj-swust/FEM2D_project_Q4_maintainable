"""高压综合测试: 多孔/异形孔/非90°边.

曾为模块级执行脚本 (无 def test_*), pytest 无法逐项统计,
缺 gmsh 包时模块级 raise 还是 collection error — 改为标准测试函数
(第四轮外部审查)。
"""
import math
import os
import tempfile

import numpy as np
import pytest

from fem2d import Mesh, solve
from fem2d.boundary import validate_boundary_segments
from fem2d.boundary.naming import build_boundary_segments
from fem2d.gmsh_adapter import generate_from_geo

try:
    import gmsh  # noqa: F401
except (ImportError, OSError):
    # importorskip 只捕 ImportError; gmsh 加载动态库缺 libGLU 时抛
    # OSError, 必须同样转 skip (否则 CI 收集阶段直接中断)
    pytest.skip(
        "Gmsh Python API unavailable or native dependency missing",
        allow_module_level=True)


class G:  # Gmsh script builder
    def __init__(self, lc=0.15):
        self.s = ["lc = " + str(lc) + ";"]
        self._pt = 0; self._ln = 0; self._cl = 0
        self._loop_curves = {}
    def pt(self, x, y, lc=None):
        self._pt += 1; l = ("lc*0.5" if lc is None else str(lc))
        self.s.append("Point(%d)={%.6f,%.6f,0,%s};" % (self._pt, x, y, l))
        return self._pt
    def line(self, a, b):
        self._ln += 1; self.s.append("Line(%d)={%d,%d};" % (self._ln, a, b))
        return self._ln
    def circle(self, start, center, end):
        self._ln += 1; self.s.append(
            "Circle(%d)={%d,%d,%d};" % (self._ln, start, center, end))
        return self._ln
    def ellipse(self, start, center, major, end):
        self._ln += 1
        self.s.append(
            "Ellipse(%d)={%d,%d,%d,%d};"
            % (self._ln, start, center, major, end))
        return self._ln
    def curve_loop(self, ids):
        self._cl += 1; rng = ",".join(str(i) for i in ids)
        self.s.append("Curve Loop(%d)={%s};" % (self._cl, rng))
        self._loop_curves[self._cl] = tuple(ids)
        return self._cl
    def plane_surface(self, outer_cl, hole_cls):
        ids = ",".join(str(c) for c in [outer_cl] + list(hole_cls))
        self.s.append("Plane Surface(1)={%s};" % ids)
    def finish(self):
        self.s.append("Mesh.Format=39; Mesh 2;")
        return "\n".join(self.s)
    def physical_curve(self, name, loop_id):
        ids = self._loop_curves[loop_id]
        self.s.append(
            'Physical Curve("%s")={%s};'
            % (name, ",".join(str(i) for i in ids)))
    def polygon(self, pts, lc=None):
        n = len(pts); first = None
        for x, y in pts:
            pid = self.pt(x, y, lc)
            if first is None: first = pid
        ids = list(range(first, first + n))
        for i in range(n):
            self.line(ids[i], ids[(i+1) % n])
        return self.curve_loop(list(range(self._ln - n + 1, self._ln + 1)))
    def circle_hole(self, cx, cy, R, n_seg=20, lc=None):
        ctr = self.pt(cx, cy, lc)
        pts = []
        for i in range(n_seg):
            a = 2*math.pi*i/n_seg
            pts.append(self.pt(cx+R*math.cos(a), cy+R*math.sin(a), lc))
        for i in range(n_seg):
            self.circle(pts[i], ctr, pts[(i+1) % n_seg])
        return self.curve_loop(
            list(range(self._ln - n_seg + 1, self._ln + 1)))
    def ellipse_hole(self, cx, cy, a, b, n_seg=28, lc=None):
        ctr = self.pt(cx, cy, lc)
        major = self.pt(cx + a, cy, lc)
        points = [
            major,
            self.pt(cx, cy + b, lc),
            self.pt(cx - a, cy, lc),
            self.pt(cx, cy - b, lc),
        ]
        curves = [
            self.ellipse(
                points[i], ctr, major, points[(i + 1) % 4])
            for i in range(4)
        ]
        return self.curve_loop(curves)
    def square_hole(self, cx, cy, s, lc=None):
        h = s/2
        return self.polygon(
            [(cx-h, cy-h), (cx+h, cy-h), (cx+h, cy+h), (cx-h, cy+h)], lc)
    def triangle_hole(self, cx, cy, s, lc=None):
        h = s*math.sqrt(3)/2
        return self.polygon(
            [(cx, cy+h*2/3), (cx-s/2, cy-h/3), (cx+s/2, cy-h/3)], lc)


def gmsh_api(g):
    geo_str = g.finish()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geo',
                                     delete=False) as f:
        f.write(geo_str)
        geo = f.name
    try:
        r = generate_from_geo(geo)
        return r.nodes, r.elements, r.elem_type, r.regions
    finally:
        if os.path.exists(geo):
            try:
                os.unlink(geo)
            except OSError:
                pass


def check(nodes, elems, etype, registry, want_holes, name):
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
             thickness=0.01, elem_type=etype)
    segs = build_boundary_segments(m, registry=registry)
    outer = [s for s in segs if '外边' in s.get('label', '')]
    inner = [s for s in segs if '内孔' in s.get('label', '')]
    assert len(inner) >= want_holes, \
        f"{name}: 内孔段 {len(inner)} < 期望 {want_holes} (外边 {len(outer)})"
    validate_boundary_segments(m, segs)   # ValueError → 测试失败
    return segs, m


def test_parallelogram_3_holes():
    """平行四边形(非90°) + 圆/三角/方3孔 — 内孔段 1+3+4 >= 6."""
    g = G(0.12)
    outer = g.polygon([(0, 0), (8, 0), (10, 5), (2, 5)])
    h1 = g.circle_hole(3, 2.5, 0.6, 20)
    h2 = g.triangle_hole(7, 1.5, 1.0)
    h3 = g.square_hole(5, 3.5, 0.7)
    g.plane_surface(outer, [h1, h2, h3])
    for label, loop in (
            ("outer", outer), ("round_hole", h1),
            ("triangle_hole", h2), ("square_hole", h3)):
        g.physical_curve(label, loop)
    n, e, et, reg = gmsh_api(g)
    segs, _ = check(n, e, et, reg, 3, "Parallelogram")
    inner_count = len([s for s in segs if '内孔' in s.get('label', '')])
    assert inner_count >= 6, f"内孔段 {inner_count} (应 8=1圆+3三角+4方)"


def test_star_5_round_holes():
    """五角星外边 + 5 圆孔."""
    g = G(0.08)
    Ro, Ri, n_star = 3.0, 1.2, 5
    pts = []
    for i in range(2 * n_star):
        a = 2*math.pi*i/(2 * n_star) - math.pi/2
        r = Ro if i % 2 == 0 else Ri
        pts.append((r*math.cos(a), r*math.sin(a)))
    outer = g.polygon(pts)
    holes = []
    for k in range(5):
        a = 2*math.pi*k/5 - math.pi/2
        holes.append(g.circle_hole(1.9*math.cos(a), 1.9*math.sin(a), 0.3, 16))
    g.plane_surface(outer, holes)
    g.physical_curve("outer", outer)
    for index, loop in enumerate(holes, 1):
        g.physical_curve("hole_%d" % index, loop)
    n, e, et, reg = gmsh_api(g)
    segs, _ = check(n, e, et, reg, 5, "Star")
    outers = [s for s in segs if '外边' in s.get('label', '')]
    assert len(outers) >= 8, f"星形外边段 {len(outers)} < 8"


def test_ellipse_10_random_holes():
    """椭圆外边 + 10 随机孔(圆/方/三角)."""
    g = G(0.15)
    outer_cl = g.ellipse_hole(0, 0, 6, 4, lc=0.075)
    rng = np.random.default_rng(42)
    holes = []
    for k in range(10):
        cx, cy = rng.uniform(-4, 4), rng.uniform(-2.5, 2.5)
        R, s = rng.uniform(0.2, 0.6), rng.choice(['c', 's', 't'])
        if s == 'c':
            holes.append(g.circle_hole(cx, cy, R, 16))
        elif s == 's':
            holes.append(g.square_hole(cx, cy, R*0.7))
        else:
            holes.append(g.triangle_hole(cx, cy, R))
    g.plane_surface(outer_cl, holes)
    g.physical_curve("outer_ellipse", outer_cl)
    for index, loop in enumerate(holes, 1):
        g.physical_curve("hole_%d" % index, loop)
    n, e, et, reg = gmsh_api(g)
    segs, _ = check(n, e, et, reg, 10, "Ellipse10Holes")
    outer_types = set(s['type'] for s in segs if '外边' in s.get('label', ''))
    assert bool(outer_types & {'arc', 'ellipse'}), \
        f"椭圆外边类型 {outer_types}"


def test_irregular_hexagon_3_mixed_holes():
    """不规则六边形 + 椭圆/三角/圆 3孔."""
    g = G(0.12)
    outer = g.polygon([(0, 0), (8, 0), (10, 3), (7, 7), (2, 8), (-1, 4)])
    h1 = g.ellipse_hole(3, 3, 0.8, 0.4, 24)
    h2 = g.triangle_hole(6, 2, 0.7)
    h3 = g.circle_hole(5, 5.5, 0.5, 20)
    g.plane_surface(outer, [h1, h2, h3])
    for label, loop in (
            ("outer", outer), ("ellipse_hole", h1),
            ("triangle_hole", h2), ("round_hole", h3)):
        g.physical_curve(label, loop)
    n, e, et, reg = gmsh_api(g)
    check(n, e, et, reg, 3, "Hexagon3Holes")


def test_solve_parallelogram_with_bc():
    """平行四边形模型施加 BC 并求解 — 解必须有限."""
    g = G(0.15)
    outer = g.polygon([(0, 0), (8, 0), (10, 5), (2, 5)])
    hole = g.circle_hole(4, 2.5, 0.5, 20)
    g.plane_surface(outer, [hole])
    g.physical_curve("outer", outer)
    g.physical_curve("hole", hole)
    n, e, et, reg = gmsh_api(g)
    m = Mesh(nodes=n, elements=e, E=2.1e11, nu=0.3,
             thickness=0.01, elem_type=et)
    segs = build_boundary_segments(m, registry=reg)
    for s in segs:
        if s['type'] != 'line':
            continue
        xs = s['coords'][:, 0]
        if xs.mean() < 2:                      # 左边固定
            for nid in s['nodes']:
                m.fix_node(int(nid), 'both', 0)
        elif xs.mean() > 8:                    # 右边拉力
            ns = s['nodes']
            for a, b in zip(ns, ns[1:]):
                m.add_traction(int(a), int(b), 1e6, 0)
    r = solve(m, verbose=False)
    assert np.all(np.isfinite(r['u'])), "解含 NaN/Inf"
    assert np.all(np.isfinite(r['vm_stress'])), "应力含 NaN/Inf"
