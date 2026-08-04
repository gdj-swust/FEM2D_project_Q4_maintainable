"""示例插件判别性测试 — circle 标签探测器真插真测 (阶段 3).

判别性 (放回旧实现必须失败):
  - 本文件 import 插件类 → 插件文件删除/改名 → ImportError 红
  - 注册后段标签含 " [Gmsh 原生圆]" + info["native_circle"] — 插件
    未接入 (无注册行/注册失败) → 断言红
  - 未注册时标签无插件标记 (默认管线与金标准逐位一致)
"""
import numpy as np
import pytest

from fem2d.boundary import default_registry, register_detector
from fem2d.boundary.plugins.circle_label import (
    NativeCircleLabelDetector,
)


def _circle_chain(n=32, radius=1.5):
    angle = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([
        radius * np.cos(angle), radius * np.sin(angle)])


def test_plugin_unit_contract_discriminative():
    """插件单元契约: 原生 Circle + 圆/圆弧链 → 标签增强 Detection;
    无原生实体/非 Circle 实体/非圆链 → None (让位上游). 期望值硬编码 —
    插件退化 (丢委托/丢标记) 必红."""
    plugin = NativeCircleLabelDetector()

    detection = plugin.detect(
        _circle_chain(),
        scale=2.0,
        is_outer=True,
        closed=True,
        native_entities=("Circle",),
    )
    assert detection is not None
    assert detection.type == "arc"
    assert detection.label == "外边 整圆 R=1.5 [Gmsh 原生圆]"
    assert detection.params["native_circle"] is True
    assert 0.0 <= detection.confidence <= 1.0

    # 开放圆弧 + 原生 Circle → 同样标记 (段类型 arc)
    arc = _circle_chain()[0:9]
    open_detection = plugin.detect(
        arc, scale=2.0, is_outer=True, closed=False,
        native_entities=("Circle",))
    assert open_detection is not None
    assert open_detection.type == "arc"
    assert "[Gmsh 原生圆]" in open_detection.label

    # 无原生实体信息 → None (纯几何链不触发插件)
    assert plugin.detect(
        _circle_chain(), scale=2.0, is_outer=True, closed=True) is None
    # 非 Circle 原生实体 → None
    assert plugin.detect(
        _circle_chain(), scale=2.0, is_outer=True, closed=True,
        native_entities=("Line",)) is None
    # 椭圆链 → 委托判定失败 → None
    ellipse = np.column_stack([
        2.0 * np.cos(2 * np.pi * np.arange(40) / 40),
        1.0 * np.sin(2 * np.pi * np.arange(40) / 40),
    ])
    assert plugin.detect(
        ellipse, scale=2.0, is_outer=True, closed=True,
        native_entities=("Circle",)) is None


def test_plugin_not_registered_pipeline_unchanged():
    """未注册插件时管线输出无插件标记 — 金标准锁定的默认行为."""
    from fem2d import detect_boundaries
    from tests.test_boundary_golden_deterministic import _circle_fan

    names = [d.name for d in default_registry()._detectors]
    assert "native_circle_label" not in names
    segs = detect_boundaries(_circle_fan())
    assert all(
        "[Gmsh 原生圆]" not in s["label"] for s in segs)


@pytest.mark.skipif(
    not __import__("tests.conftest", fromlist=["GMSH_AVAILABLE"]).
    GMSH_AVAILABLE,
    reason="Gmsh Python API unavailable or native dependency missing")
def test_plugin_registered_through_pipeline_discriminative():
    """插件一行注册 → 真实管线生效: Gmsh Circle 实体圆孔段标签增强,
    Line 实体段不受影响. 判别性: 去掉注册行 → 断言必红."""
    from tests.conftest import mesh_result_from_geo
    from fem2d import Mesh, build_boundary_segments

    geo = """
lc = 0.25;
Point(1)={-1,-1,0,lc}; Point(2)={1,-1,0,lc};
Point(3)={1,1,0,lc}; Point(4)={-1,1,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Point(5)={0,0,0,lc*0.5};
n=8;
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(100+i)={0.3*Cos(ang), 0.3*Sin(ang), 0, lc*0.25};
EndFor
// 4×90° 原生 Circle 弧 (Gmsh 要求弧 <180°) — 环网格 ~6-7 边/弧
Circle(200)={100,5,102}; Circle(201)={102,5,104};
Circle(202)={104,5,106}; Circle(203)={106,5,100};
Curve Loop(1)={1,2,3,4}; Curve Loop(2)={200,201,202,203};
Plane Surface(1)={1,2};
Physical Curve("hole")={200,201,202,203};
Mesh.Format=39; Mesh 2;
"""
    result = mesh_result_from_geo(geo)
    mesh = Mesh(nodes=result.nodes, elements=result.elements,
                E=210e9, nu=0.3, thickness=0.01,
                elem_type=result.elem_type)

    registry = default_registry()
    original = list(registry._detectors)
    try:
        # 注册 1 行 — 插件接入点
        register_detector(NativeCircleLabelDetector())
        # 插件插入注册表前端 (优先级契约): 未命中必须让位内置探测器
        assert registry._detectors[0].name == "native_circle_label"
        segments = build_boundary_segments(mesh, registry=result.regions)
    finally:
        registry._detectors[:] = original

    circle_segs = [
        s for s in segments
        if s["info"].get("native_circle") is True
    ]
    # 4 条原生 Circle 弧经 conic_merge 合并为 1 个整圆段, 插件标记在
    # 合并 classify (native_entities=("Circle",)) 上生效
    # (判别性: 插件未注册 → 0 段).
    assert len(circle_segs) == 1, (
        f"插件未生效: 圆孔合并段应含 native_circle 标记, "
        f"实际 {len(circle_segs)} 段")
    assert "[Gmsh 原生圆]" in circle_segs[0]["label"]
    assert circle_segs[0]["type"] == "arc"
    assert "整圆" in circle_segs[0]["label"]

    # Line 实体段不受插件影响
    line_segs = [
        s for s in segments
        if "Line" in s["info"].get("cad_entity_types", ())
    ]
    assert line_segs, "网格应含 Line 实体段"
    assert all(
        "[Gmsh 原生圆]" not in s["label"] for s in line_segs)
