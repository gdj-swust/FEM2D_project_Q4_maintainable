"""复杂几何边界检测 — Gmsh 生成 + 多密度对比.

曾为模块级执行脚本 (无 def test_*), pytest 无法逐项统计,
缺 gmsh 包时模块级 raise 还是 collection error — 改为标准测试函数
。
"""
from pathlib import Path

import pytest

from tests.conftest import GMSH_AVAILABLE, mesh_from_geo

from fem2d import Mesh, detect_boundaries
from fem2d.gmsh_adapter import generate_from_geo

pytestmark = pytest.mark.skipif(
    not GMSH_AVAILABLE, reason="Gmsh Python API unavailable or native dependency missing")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _detect(geo_str):
    """生成 + 检测, 返回 (segs, outer, inner, types)."""
    nodes, elems, etype = mesh_from_geo(geo_str)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
                thickness=0.01, elem_type=etype)
    segs = detect_boundaries(mesh)
    outer = [s for s in segs if '外边' in s.get('label', '')]
    inner = [s for s in segs if '内孔' in s.get('label', '')]
    types = set(s['type'] for s in segs)
    return segs, outer, inner, types


HOLE_PLATE = """
lc = {lc};
Point(1)={{-1,-1,0,lc}}; Point(2)={{1,-1,0,lc}};
Point(3)={{1,1,0,lc}}; Point(4)={{-1,1,0,lc}};
Line(1)={{1,2}}; Line(2)={{2,3}}; Line(3)={{3,4}}; Line(4)={{4,1}};
Point(5)={{0,0,0,lc*0.5}};
n=24;
For i In {{0:n-1}}
  ang=2*Pi*i/n;
  Point(100+i)={{0.3*Cos(ang), 0.3*Sin(ang), 0, lc*0.5}};
EndFor
For i In {{0:n-1}}
  Circle(200+i)={{100+i, 5, 100+((i+1)%n)}};
EndFor
Curve Loop(1)={{1,2,3,4}}; Curve Loop(2)={{200:200+n-1}};
Plane Surface(1)={{1,2}};
Mesh.Format=39; Mesh 2;
"""


def test_plate_hole_three_densities():
    """带圆孔方板, 粗/细网格对比 (lc=0.3/0.1/0.03)."""
    for lc in (0.3, 0.1, 0.03):
        _, outer, inner, types = _detect(HOLE_PLATE.format(lc=lc))
        assert len(inner) >= 1, f"lc={lc}: 无内孔"
        assert len(outer) >= 1, f"lc={lc}: 无外边"
        assert 'arc' in types or 'ellipse' in types, \
            f"lc={lc}: 无圆弧段 {types}"


MULTI_HOLE = """
lc = {lc};
Point(1)={{0,0,0,lc}}; Point(2)={{6,0,0,lc}};
Point(3)={{6,4,0,lc}}; Point(4)={{0,4,0,lc}};
Line(1)={{1,2}}; Line(2)={{2,3}}; Line(3)={{3,4}}; Line(4)={{4,1}};
n=20;
// 大孔 R=0.8 @ (1.5, 2), 中心=Point(10)
Point(10)={{1.5,2,0,lc*0.5}};
For i In {{0:n-1}}
  ang=2*Pi*i/n;
  Point(100+i)={{1.5+0.8*Cos(ang), 2+0.8*Sin(ang), 0, lc*0.6}};
EndFor
For i In {{0:n-1}}
  Circle(200+i)={{100+i, 10, 100+((i+1)%n)}};
EndFor
// 中孔 R=0.5 @ (4, 2), 中心=Point(20)
Point(20)={{4,2,0,lc*0.5}};
For i In {{0:n-1}}
  ang=2*Pi*i/n;
  Point(300+i)={{4+0.5*Cos(ang), 2+0.5*Sin(ang), 0, lc*0.6}};
EndFor
For i In {{0:n-1}}
  Circle(400+i)={{300+i, 20, 300+((i+1)%n)}};
EndFor
// 小孔 R=0.25 @ (2.5, 1), 中心=Point(30)
Point(30)={{2.5,1,0,lc*0.5}};
For i In {{0:n-1}}
  ang=2*Pi*i/n;
  Point(500+i)={{2.5+0.25*Cos(ang), 1+0.25*Sin(ang), 0, lc*0.5}};
EndFor
For i In {{0:n-1}}
  Circle(600+i)={{500+i, 30, 500+((i+1)%n)}};
EndFor
Curve Loop(1)={{1,2,3,4}};
Curve Loop(2)={{200:219}}; Curve Loop(3)={{400:419}}; Curve Loop(4)={{600:619}};
Plane Surface(1)={{1,2,3,4}};
Mesh.Format=39; Mesh 2;
"""


def test_three_holes_different_sizes():
    """三孔板 (不同大小) → 3 内孔."""
    _, outer, inner, _ = _detect(MULTI_HOLE.format(lc=0.15))
    assert len(inner) == 3, f"内孔 {len(inner)} ≠ 3"
    assert len(outer) >= 1


L_SHAPE = """
lc = {lc};
Point(1)={{0,0,0,lc}}; Point(2)={{4,0,0,lc}};
Point(3)={{4,2,0,lc}}; Point(4)={{2,2,0,lc}};
Point(5)={{2,6,0,lc}}; Point(6)={{0,6,0,lc}};
Line(1)={{1,2}}; Line(2)={{2,3}}; Line(3)={{3,4}};
Line(4)={{4,5}}; Line(5)={{5,6}}; Line(6)={{6,1}};
Curve Loop(1)={{1,2,3,4,5,6}};
Plane Surface(1)={{1}};
Mesh.Format=39; Mesh 2;
"""


def test_l_shape():
    """L 形板: 无内孔, 外边 ≥4 段, 全直线."""
    _, outer, inner, types = _detect(L_SHAPE.format(lc=0.1))
    assert len(inner) == 0, f"L形: 内孔 {len(inner)}"
    assert len(outer) >= 4, f"L形: 外边 {len(outer)} 段"
    assert types == {'line'}, f"L形: {types}"


RING = """
lc = {lc};
Point(1)={{0,0,0,lc*0.3}};
n=32;
For i In {{0:n-1}}
  ang=2*Pi*i/n;
  Point(100+i)={{2*Cos(ang), 2*Sin(ang), 0, lc}};
  Point(200+i)={{1*Cos(ang), 1*Sin(ang), 0, lc*0.7}};
EndFor
For i In {{0:n-1}}
  Circle(300+i)={{100+i, 1, 100+((i+1)%n)}};
  Circle(400+i)={{200+i, 1, 200+((i+1)%n)}};
EndFor
Curve Loop(1)={{300:331}}; Curve Loop(2)={{400:431}};
Plane Surface(1)={{1,2}};
Mesh.Format=39; Mesh 2;
"""


def test_concentric_ring():
    """同心圆环: 1 外边 + 1 内孔."""
    _, outer, inner, _ = _detect(RING.format(lc=0.15))
    assert len(outer) >= 1 and len(inner) >= 1, \
        f"圆环: {len(outer)}外边 {len(inner)}内孔"


def test_demo_complex_geo_file():
    """demo_complex 直接从 .geo 生成."""
    r = generate_from_geo(str(MODELS_DIR / "demo_complex.geo"))
    mesh = Mesh(nodes=r.nodes, elements=r.elements, E=210e9, nu=0.3,
                thickness=0.01, elem_type=r.elem_type)
    segs = detect_boundaries(mesh)
    outer = [s for s in segs if '外边' in s.get('label', '')]
    inner = [s for s in segs if '内孔' in s.get('label', '')]
    types = set(s['type'] for s in segs)
    assert len(inner) >= 1, "demo_complex 含内孔(椭圆)"
    assert 'arc' in types or 'ellipse' in types, f"有圆弧段 {types}"
    assert len(outer) >= 4, f"外边 {len(outer)} < 4"
