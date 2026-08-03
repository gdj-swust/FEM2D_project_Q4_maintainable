"""边界检测系统测试 — 全部由 Gmsh 生成真实网格.

曾为模块级执行脚本 (无 def test_*), pytest 无法逐项统计,
缺 gmsh 包时模块级 raise 还是 collection error — 改为标准测试函数
。
"""
import os
import tempfile

import pytest

from fem2d import Mesh, detect_boundaries
from fem2d.gmsh_adapter import generate_from_geo

try:
    import gmsh  # noqa: F401
except (ImportError, OSError):
    # importorskip 只捕 ImportError; gmsh 加载动态库缺 libGLU 时抛
    # OSError, 必须同样转 skip (否则 CI 收集阶段直接中断)
    pytest.skip(
        "Gmsh Python API unavailable or native dependency missing",
        allow_module_level=True)


def _mesh_from_geo(geo_str):
    """Gmsh API 生成网格, 返回 (nodes, elements, elem_type)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geo',
                                     delete=False) as fh:
        fh.write(geo_str)
        geo = fh.name
    try:
        r = generate_from_geo(geo)
        return r.nodes, r.elements, r.elem_type
    finally:
        if os.path.exists(geo):
            try:
                os.unlink(geo)
            except OSError:
                pass


def _detect(nodes, elems, etype):
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3,
             thickness=0.01, elem_type=etype)
    return detect_boundaries(m)


# ── 1. 三角形 ──
def test_triangle_mesh_all_lines():
    geo = """lc=0.15;
Point(1)={0,0,0,lc}; Point(2)={6,0,0,lc}; Point(3)={3,5.2,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,1};
Curve Loop(1)={1,2,3}; Plane Surface(1)={1};
Mesh.Format=39; Mesh 2;"""
    n, e, et = _mesh_from_geo(geo)
    s = _detect(n, e, et)
    assert set(x['type'] for x in s) == {'line'}, \
        f"三角形: {len(s)}段 {set(x['type'] for x in s)}"


# ── 2. 星形 (尖角) ──
def test_star_mesh_all_lines():
    geo = """lc=0.06; n=5; R=2; r=0.8;
For i In {0:n-1}
  ang=2*Pi*i/n-Pi/2;
  Point(100+i*2)={R*Cos(ang),R*Sin(ang),0,lc};
  Point(100+i*2+1)={r*Cos(ang+Pi/n),r*Sin(ang+Pi/n),0,lc};
EndFor
For i In {0:2*n-1}
  Line(200+i)={100+i,100+((i+1)%(2*n))};
EndFor
Curve Loop(1)={200:209}; Plane Surface(1)={1};
Mesh.Format=39; Mesh 2;"""
    n, e, et = _mesh_from_geo(geo)
    s = _detect(n, e, et)
    assert set(x['type'] for x in s) == {'line'}, \
        f"星形: {len(n)}节点 {len(s)}段 {set(x['type'] for x in s)}"


# ── 3. 正六边形 ──
def test_hexagon_mesh_all_lines():
    geo = """lc=0.1; n=6; R=2;
For i In {0:n-1}
  ang=2*Pi*i/n; Point(1+i)={R*Cos(ang),R*Sin(ang),0,lc};
EndFor
For i In {0:n-1}
  Line(10+i)={1+i,1+((i+1)%n)};
EndFor
Curve Loop(1)={10:15}; Plane Surface(1)={1};
Mesh.Format=39; Mesh 2;"""
    n, e, et = _mesh_from_geo(geo)
    s = _detect(n, e, et)
    assert set(x['type'] for x in s) == {'line'}, \
        f"六边形: {len(s)}段 {set(x['type'] for x in s)}"


# ── 4. 正圆 (32点, 应检测为 arc) ──
def test_circle_mesh_detected_as_arc():
    geo = """lc=0.08; n=32; R=1.5;
Point(1)={0,0,0,lc*0.5};
For i In {0:n-1}
  ang=2*Pi*i/n; Point(100+i)={R*Cos(ang),R*Sin(ang),0,lc*0.5};
EndFor
For i In {0:n-1}
  Circle(200+i)={100+i,1,100+((i+1)%n)};
EndFor
Curve Loop(1)={200:231}; Plane Surface(1)={1};
Mesh.Format=39; Mesh 2;"""
    n, e, et = _mesh_from_geo(geo)
    s = _detect(n, e, et)
    assert len(s) == 1 and s[0]['type'] == 'arc' and \
        s[0].get('closed', False), f"正圆: {s[0]['type'] if s else '?'}"


# ── 5. 椭圆 (a=3, b=1.5) ──
def test_ellipse_mesh_detected():
    geo = """lc=0.1; n=36; a=3; b=1.5;
Point(1)={0,0,0,lc*0.5};
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(100+i)={a*Cos(ang),b*Sin(ang),0,lc*0.5};
EndFor
For i In {0:n-1}
  Circle(200+i)={100+i,1,100+((i+1)%n)};
EndFor
Curve Loop(1)={200:235}; Plane Surface(1)={1};
Mesh.Format=39; Mesh 2;"""
    n, e, et = _mesh_from_geo(geo)
    s = _detect(n, e, et)
    assert s and s[0]['type'] in ('arc', 'ellipse'), \
        f"椭圆: type={s[0]['type'] if s else '?'}"


# ── 6. 圆角矩形 (4直线+4圆角) ──
def test_rounded_rect_4_lines_4_arcs():
    geo = """lc=0.06; W=4; H=2; R=0.5;
// 底边: 左到右
Point(1)={-W/2+R,-H/2,0,lc}; Point(2)={W/2-R,-H/2,0,lc};
// 右下圆角 center=(W/2-R, -H/2+R)
Point(10)={W/2-R, -H/2+R, 0, lc*0.3};
Point(3)={W/2, -H/2+R, 0, lc};
// 右边
Point(4)={W/2, H/2-R, 0, lc};
// 右上圆角
Point(20)={W/2-R, H/2-R, 0, lc*0.3};
Point(5)={W/2-R, H/2, 0, lc};
// 顶边
Point(6)={-W/2+R, H/2, 0, lc};
// 左上圆角
Point(30)={-W/2+R, H/2-R, 0, lc*0.3};
Point(7)={-W/2, H/2-R, 0, lc};
// 左边
Point(8)={-W/2, -H/2+R, 0, lc};
// 左下圆角
Point(40)={-W/2+R, -H/2+R, 0, lc*0.3};
// 线段
Line(1)={1,2};
Circle(2)={2,10,3}; Line(3)={3,4};
Circle(4)={4,20,5}; Line(5)={5,6};
Circle(6)={6,30,7}; Line(7)={7,8};
Circle(8)={8,40,1};
Curve Loop(1)={1,2,3,4,5,6,7,8};
Plane Surface(1)={1};
Mesh.Format=39; Mesh 2;"""
    n, e, et = _mesh_from_geo(geo)
    s = _detect(n, e, et)
    n_lines = sum(1 for x in s if x['type'] == 'line')
    n_arcs = sum(1 for x in s if x['type'] == 'arc')
    assert len(s) == 8 and n_lines == 4 and n_arcs == 4, \
        f"圆角矩形: {len(s)}段 ({n_lines}直线+{n_arcs}圆弧)"


# ── 7. 三孔板 (不同大小) ──
def test_three_holes_detected():
    geo = """lc=0.12; n=24;
Point(1)={0,0,0,lc}; Point(2)={8,0,0,lc};
Point(3)={8,5,0,lc}; Point(4)={0,5,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
// 大孔 R=0.9@(2,2.5)
Point(10)={2,2.5,0,lc*0.5};
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(100+i)={2+0.9*Cos(ang),2.5+0.9*Sin(ang),0,lc*0.5};
EndFor
For i In {0:n-1}
  Circle(300+i)={100+i,10,100+((i+1)%n)};
EndFor
// 中孔 R=0.5@(5,2.5)
Point(20)={5,2.5,0,lc*0.4};
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(200+i)={5+0.5*Cos(ang),2.5+0.5*Sin(ang),0,lc*0.4};
EndFor
For i In {0:n-1}
  Circle(400+i)={200+i,20,200+((i+1)%n)};
EndFor
// 小孔 R=0.3@(3.5,1)
Point(30)={3.5,1,0,lc*0.3};
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(500+i)={3.5+0.3*Cos(ang),1+0.3*Sin(ang),0,lc*0.3};
EndFor
For i In {0:n-1}
  Circle(600+i)={500+i,30,500+((i+1)%n)};
EndFor
Curve Loop(1)={1,2,3,4};
Curve Loop(2)={300:323}; Curve Loop(3)={400:423}; Curve Loop(4)={600:623};
Plane Surface(1)={1,2,3,4};
Mesh.Format=39; Mesh 2;"""
    n, e, et = _mesh_from_geo(geo)
    s = _detect(n, e, et)
    inners = [x for x in s if '内孔' in x.get('label', '')]
    arcs = [x for x in s if x['type'] == 'arc']
    assert len(inners) == 3 and len(arcs) >= 3, \
        f"三孔板: {len(inners)}内孔 {len(arcs)}arc ({len(n)}节点)"


# ── 8. 嵌套方孔 (3层) ──
def test_nested_holes_detected():
    geo = """lc=0.1;
Point(1)={0,0,0,lc}; Point(2)={10,0,0,lc};
Point(3)={10,10,0,lc}; Point(4)={0,10,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Point(5)={2,2,0,lc*0.6}; Point(6)={8,2,0,lc*0.6};
Point(7)={8,8,0,lc*0.6}; Point(8)={2,8,0,lc*0.6};
Line(5)={5,6}; Line(6)={6,7}; Line(7)={7,8}; Line(8)={8,5};
Point(9)={4,4,0,lc*0.5}; Point(10)={6,4,0,lc*0.5};
Point(11)={6,6,0,lc*0.5}; Point(12)={4,6,0,lc*0.5};
Line(9)={9,10}; Line(10)={10,11}; Line(11)={11,12}; Line(12)={12,9};
Curve Loop(1)={1,2,3,4};
Curve Loop(2)={5,6,7,8}; Curve Loop(3)={9,10,11,12};
Plane Surface(1)={1,2,3};
Mesh.Format=39; Mesh 2;"""
    n, e, et = _mesh_from_geo(geo)
    s = _detect(n, e, et)
    inners = [x for x in s if '内孔' in x.get('label', '')]
    outers = [x for x in s if '外边' in x.get('label', '')]
    assert len(inners) >= 2, f"三层嵌套: {len(inners)}内孔 {len(outers)}外边"
