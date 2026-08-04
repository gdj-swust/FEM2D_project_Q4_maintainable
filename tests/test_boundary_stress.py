"""高压测试: 多孔 + 近切 + 高密度.

曾为模块级执行脚本 (无 def test_*), pytest 无法逐项统计,
缺 gmsh 包时模块级 raise 还是 collection error — 改为标准测试函数
。
"""
import pytest

from tests.conftest import GMSH_AVAILABLE, GMSH_UNAVAILABLE_REASON, mesh_from_geo

from fem2d import Mesh, detect_boundaries

pytestmark = pytest.mark.skipif(
    not GMSH_AVAILABLE, reason=GMSH_UNAVAILABLE_REASON)


def inner_closed_components(segments):
    """Select hole loops from topology metadata, independent of label text."""
    return [
        segment for segment in segments
        if segment.get("closed", False)
        and segment.get("info", {}).get("is_outer") is False
    ]


def _detect(geo_str):
    n, e, et = mesh_from_geo(geo_str)
    m = Mesh(nodes=n, elements=e, E=210e9, nu=0.3, thickness=0.01,
             elem_type=et)
    return detect_boundaries(m), len(n)


def test_near_tangent_holes():
    """两孔近切 (间隙 0.001) → 2 个闭合 arc 内孔."""
    geo = """
lc=0.08; n=24;
Point(1)={0,0,0,lc}; Point(2)={6,0,0,lc};
Point(3)={6,4,0,lc}; Point(4)={0,4,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
// 左孔 R=0.6@(1.5,2)
Point(10)={1.5,2,0,lc*0.3};
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(100+i)={1.5+0.6*Cos(ang), 2+0.6*Sin(ang), 0, lc*0.4};
EndFor
For i In {0:n-1}
  Circle(300+i)={100+i, 10, 100+((i+1)%n)};
EndFor
// 右孔 R=0.6@(2.701,2) 间隙=2.701-0.6-(1.5+0.6)=0.001
Point(20)={2.701,2,0,lc*0.3};
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(200+i)={2.701+0.6*Cos(ang), 2+0.6*Sin(ang), 0, lc*0.4};
EndFor
For i In {0:n-1}
  Circle(400+i)={200+i, 20, 200+((i+1)%n)};
EndFor
Curve Loop(1)={1,2,3,4};
Curve Loop(2)={300:323}; Curve Loop(3)={400:423};
Plane Surface(1)={1,2,3};
Mesh.Format=39; Mesh 2;
"""
    s, n_nodes = _detect(geo)
    inn = inner_closed_components(s)
    assert len(inn) == 2 and all(x["type"] == "arc" for x in inn), \
        f"近切双孔: {len(inn)}内孔 (期望2) {n_nodes}节点"


def test_hole_near_boundary():
    """孔贴外边界 (gap=0.005) → 1 个闭合 arc 内孔."""
    geo = """
lc=0.05; n=24;
Point(1)={0,0,0,lc}; Point(2)={4,0,0,lc};
Point(3)={4,3,0,lc}; Point(4)={0,3,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Point(10)={0.355,1.5,0,lc*0.3};
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(100+i)={0.355+0.35*Cos(ang), 1.5+0.35*Sin(ang), 0, lc*0.4};
EndFor
For i In {0:n-1}
  Circle(300+i)={100+i, 10, 100+((i+1)%n)};
EndFor
Curve Loop(1)={1,2,3,4}; Curve Loop(2)={300:323};
Plane Surface(1)={1,2};
Mesh.Format=39; Mesh 2;
"""
    s, n_nodes = _detect(geo)
    inn = inner_closed_components(s)
    assert len(inn) == 1 and inn[0]["type"] == "arc", \
        f"贴边孔: {len(inn)}内孔 (期望1) {n_nodes}节点"


def test_dense_plate_hole():
    """高密度 plate_hole (lc=0.02) → 1 个闭合 arc 内孔."""
    geo = """
lc=0.02; n=32;
Point(1)={-1,-1,0,lc}; Point(2)={1,-1,0,lc};
Point(3)={1,1,0,lc}; Point(4)={-1,1,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Point(10)={0,0,0,lc*0.5};
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(100+i)={0.3*Cos(ang), 0.3*Sin(ang), 0, lc*0.5};
EndFor
For i In {0:n-1}
  Circle(300+i)={100+i, 10, 100+((i+1)%n)};
EndFor
Curve Loop(1)={1,2,3,4}; Curve Loop(2)={300:331};
Plane Surface(1)={1,2};
Mesh.Format=39; Mesh 2;
"""
    s, n_nodes = _detect(geo)
    inn = inner_closed_components(s)
    arc = [x for x in s if x['type'] == 'arc']
    assert len(inn) == 1 and inn[0]["type"] == "arc" and len(arc) >= 1, \
        f"密plate_hole: {len(inn)}内孔 {len(arc)}arc段 {n_nodes}节点"
