"""正式插件 1 判别性测试 — ellipse_group_label 组级椭圆标签 (轮 2).

判别性 (放回旧实现必须失败):
  - 真 Ellipse 原生链 (粗采样, 内置角点门会拒) → 原生直读零门标椭圆;
    无插件时内置探测器拒 → 断言红
  - 残差超严格门 (2%) 但内置门 (3%) 内的闭合链 → 标签必须不出现;
    无插件时内置椭圆探测器会标 "椭圆" → 断言红 (本文件显式证明)
  - 闭合整圆链 (含原生 Circle) → 让位内置圆探测器 ("整圆"标签优先)
  - 生产默认管线注册 (import fem2d.boundary 即生效, 非测试内手动注册)
  - 求解逐位不变 (回归对照)
"""
import numpy as np
import pytest

from fem2d.boundary import default_registry
from fem2d.boundary.geometry import classify
from fem2d.boundary.plugins.ellipse_group_label import (
    EllipseGroupLabelDetector,
    RELATIVE_DEVIATION_LIMIT,
)


def _ellipse_chain(n=40, a=2.0, b=1.0):
    """闭合整环椭圆链 (a:b = 2:1, 与内置 20 段演示链同族)."""
    phi = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([a * np.cos(phi), b * np.sin(phi)])


def _smooth_noisy_ellipse(amp=0.04, n=40, k=3):
    """平滑周期径向扰动椭圆链 — 拟合残差落在 (2%, 3%) 内:
    超插件严格门, 仍在内置门内 (判别性窗口)."""
    phi = 2.0 * np.pi * np.arange(n) / n
    base = _ellipse_chain(n)
    scale = 1.0 + amp * np.sin(k * phi + 0.7)
    return base * scale[:, None]


def _circle_chain(n=32, radius=1.5):
    phi = 2.0 * np.pi * np.arange(n) / n
    return np.column_stack([
        radius * np.cos(phi), radius * np.sin(phi)])


def test_plugin_1_native_direct_discriminative():
    """①真 Ellipse 原生命令链 → 标签必须出现 (原生直读, 零拟合门).

    粗采样 12 点链: 转角 30° > 内置角点门 (12<16 原语 → 20°) → 内置
    椭圆探测器必拒. 原生实体 = CAD 真值 → 插件直接读参标椭圆.
    放回旧实现 (无插件) → 该链落入 通用曲线 → 断言红.
    """
    plugin = EllipseGroupLabelDetector()
    chain = _ellipse_chain(n=12)
    detection = plugin.detect(
        chain, scale=2.0, is_outer=True, closed=True,
        native_entities=("Ellipse",))
    assert detection is not None
    assert detection.type == "ellipse"
    assert detection.label == "外边 椭圆 a=2 b=1"
    assert detection.params["semi_major"] == pytest.approx(2.0, rel=1e-9)
    assert detection.params["semi_minor"] == pytest.approx(1.0, rel=1e-9)

    # 同链无原生实体 → 点云兜底: 内置角点门拒 → None (宁缺毋滥)
    assert plugin.detect(
        chain, scale=2.0, is_outer=True, closed=True) is None

    # 原生 Circle 实体 + 闭合整圆链 → 让位内置圆探测器
    assert plugin.detect(
        _circle_chain(), scale=2.0, is_outer=True, closed=True,
        native_entities=("Circle",)) is None


def test_plugin_1_strict_gate_discriminative():
    """②残差超严格门 (2%) → 标签必须不出现.

    判别性窗口: 残差 2.5% (超插件门) 且 4.2% (内置 max 门内) — 无插件
    时内置椭圆探测器必标 "椭圆". 本文件显式证明: 移除插件 → 椭圆出现.
    """
    chain = _smooth_noisy_ellipse()
    plugin = EllipseGroupLabelDetector()
    detection = plugin.detect(
        chain, scale=2.0, is_outer=True, closed=True)
    assert detection is not None           # 保守兜底仍在
    assert detection.type == "curve"
    assert "椭圆" not in detection.label

    # 生产默认管线 (插件已注册): classify 同样不给椭圆
    seg_type, label, _ = classify(chain, 2.0, True, closed=True)
    assert seg_type != "ellipse"
    assert "椭圆" not in label

    # 判别性: 放回旧实现 (移除插件) → 内置宽松门 (3%) 标椭圆
    registry = default_registry()
    original = list(registry._detectors)
    try:
        registry._detectors[:] = [
            detector for detector in original
            if detector.name != "ellipse_group_label"
        ]
        old_type, old_label, _ = registry.classify(
            chain, 2.0, True, closed=True)
    finally:
        registry._detectors[:] = original
    assert old_type == "ellipse", (
        "判别性窗口失效: 无插件时内置探测器未标椭圆, "
        "插件严格门无从证明")
    assert "椭圆" in old_label

    # 门限常量可回归: 2% 是相对偏差门 (与椭圆拟合残差同量纲)
    assert RELATIVE_DEVIATION_LIMIT == 0.02


def test_plugin_1_clean_ellipse_passthrough():
    """干净整环椭圆 (残差 ~0) → 标签与内置逐位一致 (金标准零变化)."""
    plugin = EllipseGroupLabelDetector()
    chain = _ellipse_chain(n=40)
    detection = plugin.detect(
        chain, scale=2.0, is_outer=True, closed=True)
    assert detection is not None
    assert detection.label == "外边 椭圆 a=2 b=1"
    assert detection.type == "ellipse"

    # 让位: 开链不属于插件 1 (插件 3 裁决)
    assert plugin.detect(
        chain[:30], scale=2.0, is_outer=True, closed=False) is None


def test_plugin_1_circle_deferral_pipeline():
    """闭合整圆链 → 默认管线维持 "整圆" 标签 (插件不劫持圆)."""
    chain = _circle_chain()
    seg_type, label, _ = classify(chain, 2.0, True, closed=True)
    assert seg_type == "arc"
    assert "整圆" in label and "椭圆" not in label


def test_plugin_1_registered_in_default_pipeline():
    """注册即生效: 生产默认注册表含插件 (import 链启用, 非测试内注册).

    插件文件删除/注册行移除 → 本断言红.
    """
    names = [detector.name for detector in default_registry()._detectors]
    assert "ellipse_group_label" in names
    # 前端优先: 插件须在 line 之前 (闭合链先经插件判定)
    assert names.index("ellipse_group_label") < names.index("line")


@pytest.mark.skipif(
    not __import__("tests.conftest", fromlist=["GMSH_AVAILABLE"]).
    GMSH_AVAILABLE,
    reason="Gmsh Python API unavailable or native dependency missing")
def test_plugin_1_registered_through_pipeline_discriminative():
    """真 gmsh 管线: 演示椭圆孔 (20 段 Line 近似, 残差 0.78% < 2%)
    在**生产默认管线**输出组级椭圆标签; 边界边/节点逐位不变."""
    from fem2d import Mesh, build_boundary_segments
    from tests.conftest import mesh_result_from_geo

    # 物理组名用 ASCII: 临时 .geo 以平台默认编码写盘, 中文名会乱码
    # (真实模型文件 demo_complex.geo 为 UTF-8, 中文名路径由金标准覆盖)
    geo = """
lc = 0.06;
Point(1)={-2,-1.5,0,lc}; Point(2)={2,-1.5,0,lc};
Point(3)={2,1.5,0,lc}; Point(4)={-2,1.5,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
cx=0.3; cy=0.15; a1=0.4; b1=0.2; n=20;
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(100+i)={cx+a1*Cos(ang), cy+b1*Sin(ang), 0, lc};
EndFor
For i In {0:n-1}
  Line(200+i)={100+i, 100+((i+1)%n)};
EndFor
Curve Loop(1)={1,2,3,4}; Curve Loop(2)={200:200+n-1};
Plane Surface(1)={1,2};
Physical Curve("ellipse_hole")={200:200+n-1};
Mesh.Format=39; Mesh 2;
"""
    result = mesh_result_from_geo(geo)
    mesh = Mesh(nodes=result.nodes, elements=result.elements,
                E=210e9, nu=0.3, thickness=0.01,
                elem_type=result.elem_type)

    # 生产默认管线 (插件经 import 已注册, 不手动注册)
    segments = build_boundary_segments(mesh, registry=result.regions)
    hole_segs = [
        segment for segment in segments
        if "ellipse_hole" in segment["label"]
    ]
    assert len(hole_segs) == 1, "ellipse_hole 组应合并为 1 个整环段"
    label = hole_segs[0]["label"]
    # 组级椭圆标签 (a/b 值随网格采样, 只做 token 断言)
    assert "椭圆 a=" in label and " b=" in label, f"插件未生效: {label}"
    assert hole_segs[0]["type"] == "ellipse"
    # 求解数据面 (nodes/边界边) 不受插件影响: 段节点为整环闭合链
    nodes = hole_segs[0]["nodes"]
    assert nodes[0] == nodes[-1]

    # 无插件管线 (判别性对照): 同一网格段集合的节点序列逐位一致
    registry = default_registry()
    original = list(registry._detectors)
    try:
        registry._detectors[:] = [
            detector for detector in original
            if detector.name != "ellipse_group_label"
        ]
        stripped = build_boundary_segments(mesh, registry=result.regions)
    finally:
        registry._detectors[:] = original
    assert len(stripped) == len(segments)
    for with_plugin, without_plugin in zip(segments, stripped):
        assert with_plugin["nodes"] == without_plugin["nodes"]
        assert np.array_equal(
            with_plugin["coords"], without_plugin["coords"])


@pytest.mark.skipif(
    not __import__("tests.conftest", fromlist=["GMSH_AVAILABLE"]).
    GMSH_AVAILABLE,
    reason="Gmsh API unavailable; solve regression uses gmsh mesh")
def test_plugin_1_solve_bitwise_unchanged():
    """③求解逐位不变: 插件注册与否 → 位移数组逐位一致 (回归对照)."""
    from fem2d import Mesh, build_boundary_segments, solve
    from tests.conftest import mesh_result_from_geo

    geo = """
lc = 0.08;
Point(1)={0,0,0,lc}; Point(2)={1,0,0,lc};
Point(3)={1,1,0,lc}; Point(4)={0,1,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
n=16;
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(100+i)={0.5+0.3*Cos(ang), 0.5+0.15*Sin(ang), 0, lc};
EndFor
For i In {0:n-1}
  Line(200+i)={100+i, 100+((i+1)%n)};
EndFor
Curve Loop(1)={1,2,3,4}; Curve Loop(2)={200:200+n-1};
Plane Surface(1)={1,2};
Physical Curve("hole")={200:200+n-1};
Mesh.Format=39; Mesh 2;
"""
    result = mesh_result_from_geo(geo)

    def _solve_with(registry):
        mesh = Mesh(nodes=result.nodes, elements=result.elements,
                    E=210e9, nu=0.3, thickness=0.01,
                    elem_type=result.elem_type)
        segments = build_boundary_segments(mesh, registry=registry)
        for index, segment in enumerate(segments):
            if "hole" in segment["label"]:
                mesh.fix_node(int(segment["nodes"][0]), "both", 0.0)
            elif segment["info"].get("is_outer"):
                for a, b in zip(segment["nodes"], segment["nodes"][1:]):
                    mesh.add_traction(int(a), int(b), 0.0, -1e6)
        return solve(mesh)["u"]

    with_plugin = _solve_with(result.regions)
    registry = default_registry()
    original = list(registry._detectors)
    try:
        registry._detectors[:] = [
            detector for detector in original
            if detector.name != "ellipse_group_label"
        ]
        without_plugin = _solve_with(result.regions)
    finally:
        registry._detectors[:] = original
    assert np.array_equal(with_plugin, without_plugin), (
        "插件注册改变了求解结果 — 标签层插件不得触碰数值链")
