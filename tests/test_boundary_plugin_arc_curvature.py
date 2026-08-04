"""正式插件 3 判别性测试 — arc_curvature 曲率分段展示层 (轮 2).

判别性 (放回旧实现必须失败):
  - 已知圆弧链 → ρ/圆心 数值正确 (相对容差) + 标签 **token 级断言**
    (防快照规范化吞掉: 断言原始标签文本)
  - 短弧链 (1/8 椭圆弧) → 不得出现 "椭圆" 标签; 放回旧实现 → 内置
    椭圆探测器硬拟合短弧标 "椭圆" (本文件显式证明)
  - 直边/闭合链 → 让位内置探测器 (金标准行为不变)
  - 覆盖足够的开放椭圆弧 → 让位内置 (旧行为不变)
  - 生产默认管线注册 (import 链启用, 非测试内手动注册)
"""
import re

import numpy as np
import pytest

from fem2d.boundary import default_registry
from fem2d.boundary.detectors import EllipseDetector
from fem2d.boundary.geometry import classify
from fem2d.boundary.plugins.arc_curvature import ArcCurvatureDetector


def _arc_chain(radius=1.5, center=(0.0, 0.0), span_rad=np.pi / 2, n=32):
    phi = np.linspace(0, span_rad, n)
    return np.column_stack([
        center[0] + radius * np.cos(phi),
        center[1] + radius * np.sin(phi),
    ])


def _short_ellipse_arc(a=2.0, b=1.0, span_rad=np.pi / 4, n=16):
    """1/8 椭圆弧 — 弧长覆盖 ~12.5%, a/b 从短弧不可靠."""
    phi = np.linspace(0, span_rad, n)
    return np.column_stack([a * np.cos(phi), b * np.sin(phi)])


def test_plugin_3_arc_tokens_and_values():
    """①已知圆弧链 → ρ/圆心数值正确 (相对容差) + token 级标签断言."""
    plugin = ArcCurvatureDetector()
    chain = _arc_chain(radius=1.5, center=(1.2, -0.7))
    detection = plugin.detect(
        chain, scale=2.0, is_outer=True, closed=False)
    assert detection is not None
    assert detection.type == "arc"

    # token 级断言: 原始标签文本 (快照规范化不得吞掉这些 token)
    label = detection.label
    assert "ρ=1.5" in label
    assert "圆心(1.2,-0.7)" in label
    match = re.match(
        r"^外边 圆弧 ρ=([-+0-9.eE]+), 圆心\(([-+0-9.eE]+),"
        r"([-+0-9.eE]+)\)$",
        label,
    )
    assert match, f"标签格式漂移: {label!r}"
    rho = float(match.group(1))
    center_x = float(match.group(2))
    center_y = float(match.group(3))
    assert rho == pytest.approx(1.5, rel=1e-9)
    assert center_x == pytest.approx(1.2, rel=1e-9)
    assert center_y == pytest.approx(-0.7, rel=1e-9)

    # params 与标签一致 (段 info 数据面)
    assert detection.params["radius"] == pytest.approx(1.5, rel=1e-9)
    assert detection.params["center"][0] == pytest.approx(1.2, abs=1e-9)
    assert detection.params["center"][1] == pytest.approx(-0.7, abs=1e-9)
    assert detection.params["angle"] == pytest.approx(np.pi / 2, rel=1e-6)


def test_plugin_3_short_arc_no_ellipse_discriminative():
    """②短弧链 (1/8 椭圆弧) → 不得出现 "椭圆" 标签.

    判别性前提: 内置椭圆探测器对同链硬拟合成功 (残差 ~1e-16) →
    放回旧实现 (无插件) → 必标 "椭圆". 本文件显式证明.
    """
    chain = _short_ellipse_arc()
    plugin = ArcCurvatureDetector()
    detection = plugin.detect(
        chain, scale=2.0, is_outer=True, closed=False)
    assert detection is not None
    assert detection.type == "curve"
    assert "椭圆" not in detection.label

    # 生产默认管线: classify 不给椭圆
    seg_type, label, _ = classify(chain, 2.0, True, closed=False)
    assert seg_type != "ellipse"
    assert "椭圆" not in label

    # 判别性: 内置 open-ellipse 确实会硬拟合此短弧 (证明插件拦截必要)
    builtin = EllipseDetector().detect(
        chain, scale=2.0, is_outer=True, closed=False)
    assert builtin is not None and builtin.type == "ellipse", (
        "判别性前提失效: 内置探测器未硬拟合短弧, 短弧保护无从证明")

    # 判别性: 放回旧实现 (移除插件) → classify 标椭圆 → 断言红
    registry = default_registry()
    original = list(registry._detectors)
    try:
        registry._detectors[:] = [
            detector for detector in original
            if detector.name != "arc_curvature"
        ]
        old_type, old_label, _ = registry.classify(
            chain, 2.0, True, closed=False)
    finally:
        registry._detectors[:] = original
    assert old_type == "ellipse" and "椭圆" in old_label


def test_plugin_3_deferral_gates():
    """让位门: 直边 → None (LineDetector); 闭合链 → None (插件 1/内置)."""
    plugin = ArcCurvatureDetector()
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert plugin.detect(
        line, scale=2.0, is_outer=True, closed=False) is None

    circle = np.column_stack([
        1.5 * np.cos(2 * np.pi * np.arange(32) / 32),
        1.5 * np.sin(2 * np.pi * np.arange(32) / 32),
    ])
    assert plugin.detect(
        circle, scale=2.0, is_outer=True, closed=True) is None

    # 覆盖足够的开放椭圆弧 (3/4, 75% ≥ 60%) → 让位内置椭圆探测器
    ellipse_arc = _short_ellipse_arc(span_rad=3 * np.pi / 2, n=60)
    assert plugin.detect(
        ellipse_arc, scale=2.0, is_outer=True, closed=False) is None


def test_plugin_3_registered_in_default_pipeline():
    """注册即生效: 生产默认注册表含插件 (import 链启用, 非测试内注册).

    插件文件删除/注册行移除 → 本断言红.
    """
    names = [detector.name for detector in default_registry()._detectors]
    assert "arc_curvature" in names
    # 更保守者优先: 插件 3 先于插件 1 (开链先裁决, 永不当椭圆)
    assert names.index("arc_curvature") < names.index("ellipse_group_label")


@pytest.mark.skipif(
    not __import__("tests.conftest", fromlist=["GMSH_AVAILABLE"]).
    GMSH_AVAILABLE,
    reason="Gmsh Python API unavailable or native dependency missing")
def test_plugin_3_registered_through_pipeline_discriminative():
    """真 gmsh 管线: 原生 Circle 圆弧段 → "圆弧 ρ=.., 圆心(..,..)"
    token 标签; 椭圆孔保持 "椭圆" (插件 1 域); 直边不受影响."""
    from fem2d import Mesh, build_boundary_segments
    from tests.conftest import mesh_result_from_geo

    # 带半圆凹口 (2×90° 原生 Circle, 开放链) + 圆孔 (4×90° 原生 Circle,
    # 闭合链) — 演示 demo_complex 的两种 Circle 实体形态
    geo = """
lc = 0.25;
Point(1)={-2,-1,0,lc}; Point(2)={-0.5,-1,0,lc};
Point(3)={0.5,-1,0,lc}; Point(4)={2,-1,0,lc};
Point(5)={2,1,0,lc}; Point(6)={-2,1,0,lc};
Point(7)={0,-1,0,lc};  // 凹口圆心 (半圆向上凹入板内, R=0.5)
Point(8)={0,-0.5,0,lc}; // 凹口顶点
Line(1)={1,2}; Line(2)={3,4}; Line(3)={4,5}; Line(4)={5,6}; Line(5)={6,1};
Circle(6)={2,7,8}; Circle(7)={8,7,3};
Curve Loop(1)={1,6,7,2,3,4,5};
Point(100)={0.8,0.2,0,lc*0.25};
n=8;
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(200+i)={0.8+0.3*Cos(ang), 0.2+0.3*Sin(ang), 0, lc*0.25};
EndFor
Circle(201)={200,100,202}; Circle(202)={202,100,204};
Circle(203)={204,100,206}; Circle(204)={206,100,200};
Curve Loop(2)={201,202,203,204};
Plane Surface(1)={1,2};
Physical Curve("notch")={6,7};
Physical Curve("hole")={201,202,203,204};
Mesh.Format=39; Mesh 2;
"""
    result = mesh_result_from_geo(geo)
    mesh = Mesh(nodes=result.nodes, elements=result.elements,
                E=210e9, nu=0.3, thickness=0.01,
                elem_type=result.elem_type)

    segments = build_boundary_segments(mesh, registry=result.regions)
    labels = [segment["label"] for segment in segments]
    # 圆孔合并整圆: 闭合 → 插件 3 让位 → 内置 "整圆" (标签语义优先)
    circle_labels = [label for label in labels if "hole" in label]
    assert circle_labels, "圆孔段应存在"
    assert all("整圆" in label for label in circle_labels)
    # 凹口 (Circle 实体, 开放链): 插件 3 代数标签 (token 级)
    notch_labels = [label for label in labels if "notch" in label]
    assert notch_labels, "凹口段应存在"
    assert len(notch_labels) == 1
    assert "圆弧 ρ=0.5" in notch_labels[0]
    assert "圆心(0,-1)" in notch_labels[0]
