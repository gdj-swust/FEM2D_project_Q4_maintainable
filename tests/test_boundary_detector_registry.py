"""识别器注册表契约测试 (阶段 2 插件化重构).

锁定: 注册表顺序/接口签名/Detection 输出面/原生实体信息管线传递/
段 schema 稳定化. 行为逐位不变由金标准快照 (test_boundary_golden_*)
承担; 本文件锁接口结构 (判别性: 接口退化 → 本文件必红).
"""
import numpy as np
import pytest

from fem2d.boundary.detectors import (
    CircleDetector,
    Detection,
    Detector,
    DetectorRegistry,
    EllipseDetector,
    GeneralCurveDetector,
    LineDetector,
    default_registry,
)
from fem2d.boundary.geometry import classify
from fem2d.boundary.validation import validate_segment_schema

LINE_CHAIN = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
CIRCLE_CHAIN = np.column_stack([
    1.5 * np.cos(2 * np.pi * np.arange(32) / 32),
    1.5 * np.sin(2 * np.pi * np.arange(32) / 32),
])
ELLIPSE_CHAIN = np.column_stack([
    2.0 * np.cos(2 * np.pi * np.arange(40) / 40),
    1.0 * np.sin(2 * np.pi * np.arange(40) / 40),
])
ARC_CHAIN = np.column_stack([
    1.5 * np.cos(np.linspace(0, np.pi / 2, 20)),
    1.5 * np.sin(np.linspace(0, np.pi / 2, 20)),
])


def _wavy_chain():
    x = np.linspace(0, 6, 60)
    return np.column_stack([x, 0.4 * np.sin(x)])


class _SpyDetector(Detector):
    """记录 native_entities 接收情况的探测识别器 (恒返回 None, 不干扰)."""

    name = "spy"
    seen = []

    def detect(
            self, points, *, scale, is_outer, closed, native_entities=()):
        _SpyDetector.seen.append(tuple(native_entities))
        return None


def test_default_registry_order():
    """注册表顺序 = 正式插件优先 + 旧 classify 探测顺序.

    轮 2 起默认注册: ellipse_group_label (插件 1, 闭合整椭圆) →
    line → circle → ellipse → general (插件 3 arc_curvature 随
    轮 2 第 3 插件提交注册, 顺序见 docs/boundary_plugins.md).
    插件判定优先, 未命中让位内置."""
    names = [detector.name for detector in default_registry()._detectors]
    assert names == [
        "ellipse_group_label",
        "line", "circle", "ellipse", "general",
    ]


def test_registry_classify_equals_geometry_facade():
    """注册表 classify 与 geometry.classify facade 输出逐位一致."""
    registry = default_registry()
    for coords in (LINE_CHAIN, CIRCLE_CHAIN, ELLIPSE_CHAIN,
                   ARC_CHAIN, _wavy_chain()):
        assert registry.classify(
            coords, 2.0, True, closed=None) == classify(
                coords, 2.0, True, closed=None)


def test_detector_interface_contract():
    """单个探测器按接口签名输入输出; 输入面与任务书一致:
    点链 + 可选原生实体信息 + 尺度 → 类型/参数/标签/置信度/残差."""
    line = LineDetector()
    detection = line.detect(
        LINE_CHAIN, scale=2.0, is_outer=True, closed=False)
    assert isinstance(detection, Detection)
    assert detection.type == "line"
    assert detection.label == "外边 直边 (0,0)→(2,0)"
    assert detection.params["axis"] == "y"
    assert 0.0 <= detection.confidence <= 1.0
    assert detection.residual >= 0.0

    circle = CircleDetector()
    closed_circle = circle.detect(
        CIRCLE_CHAIN, scale=2.0, is_outer=True, closed=True)
    assert closed_circle is not None
    assert closed_circle.type == "arc"
    assert closed_circle.label == "外边 整圆 R=1.5"
    assert abs(closed_circle.params["radius"] - 1.5) < 1e-6

    open_arc = circle.detect(
        ARC_CHAIN, scale=2.0, is_outer=True, closed=False)
    assert open_arc is not None
    assert open_arc.type == "arc"
    assert "圆弧" in open_arc.label

    ellipse = EllipseDetector()
    closed_ellipse = ellipse.detect(
        ELLIPSE_CHAIN, scale=2.0, is_outer=True, closed=True)
    assert closed_ellipse is not None
    assert closed_ellipse.type == "ellipse"
    assert closed_ellipse.label == "外边 椭圆 a=2 b=1"

    # 圆链对 ellipse 探测器 → None (轴比互斥), 椭圆链对 circle → None
    assert ellipse.detect(
        CIRCLE_CHAIN, scale=2.0, is_outer=True, closed=True) is None
    assert circle.detect(
        ELLIPSE_CHAIN, scale=2.0, is_outer=True, closed=True) is None

    # 非圆/非椭圆/非直线的点链 → 全部拒绝, 落到 general 兜底
    general = GeneralCurveDetector()
    wavy = general.detect(
        _wavy_chain(), scale=2.0, is_outer=True, closed=False)
    assert wavy is not None and wavy.type == "curve"


def test_general_detector_is_fallback():
    """general 恒返回 → classify 永不落空; 空注册表响亮报错."""
    registry = DetectorRegistry([GeneralCurveDetector()])
    seg_type, _, _ = registry.classify(LINE_CHAIN, 2.0, True)
    assert seg_type == "curve"

    empty = DetectorRegistry([])
    with pytest.raises(AssertionError, match="兜底"):
        empty.classify(LINE_CHAIN, 2.0, True)


def test_registry_order_is_priority():
    """登记顺序 = 优先级: general 提前 → 所有链都被判 curve (判别性)."""
    registry = DetectorRegistry([
        GeneralCurveDetector(), LineDetector(), CircleDetector()])
    seg_type, _, _ = registry.classify(LINE_CHAIN, 2.0, True)
    assert seg_type == "curve"


def test_register_duplicate_and_type_errors():
    registry = DetectorRegistry()
    registry.add(LineDetector())
    with pytest.raises(ValueError, match="已注册"):
        registry.add(LineDetector())
    with pytest.raises(TypeError, match="Detector"):
        registry.add("not-a-detector")  # type: ignore[arg-type]


def test_remove_and_detectors_lifecycle():
    """注册表生命周期: remove 幂等, detectors 只读视图, 基类未实现即报错."""
    registry = DetectorRegistry([LineDetector()])
    assert [d.name for d in registry.detectors()] == ["line"]
    assert registry.remove("line") is True
    assert registry.detectors() == ()
    assert registry.remove("line") is False  # 幂等: 不存在返回 False

    bare = Detector()
    with pytest.raises(NotImplementedError, match="detect 未实现"):
        bare.detect(LINE_CHAIN, scale=1.0, is_outer=True, closed=False)


def test_register_detector_api():
    """公开注册入口 (fem2d.boundary.register_detector) — 插件接入点."""
    from fem2d.boundary import register_detector
    registry = default_registry()
    original = list(registry._detectors)
    try:
        register_detector(_SpyDetector())
        names = [d.name for d in registry._detectors]
        assert "spy" in names
    finally:
        registry._detectors[:] = original


def test_native_entities_threaded_through_registry():
    """原生实体信息沿 classify 传入探测器不丢失 (一等公民输入面)."""
    _SpyDetector.seen = []
    spy = _SpyDetector()
    registry = DetectorRegistry(
        [spy, LineDetector(), CircleDetector()])
    registry.classify(
        CIRCLE_CHAIN, 2.0, True, closed=True,
        native_entities=("Circle",))
    assert ("Circle",) in spy.seen


@pytest.mark.skipif(
    not __import__("tests.conftest", fromlist=["GMSH_AVAILABLE"]).
    GMSH_AVAILABLE,
    reason="Gmsh Python API unavailable or native dependency missing")
def test_native_entities_flow_through_build_pipeline():
    """注册表路径 build_boundary_segments: CAD 实体类型 (Circle/Line)
    沿 segment_builder → classify 管线传入探测器 (真实 gmsh 网格)."""
    from tests.conftest import mesh_result_from_geo
    from fem2d import build_boundary_segments

    geo = """
lc = 0.25;
Point(1)={-1,-1,0,lc}; Point(2)={1,-1,0,lc};
Point(3)={1,1,0,lc}; Point(4)={-1,1,0,lc};
Line(1)={1,2}; Line(2)={2,3}; Line(3)={3,4}; Line(4)={4,1};
Point(5)={0,0,0,lc*0.5};
n=16;
For i In {0:n-1}
  ang=2*Pi*i/n;
  Point(100+i)={0.3*Cos(ang), 0.3*Sin(ang), 0, lc*0.5};
EndFor
For i In {0:n-1}
  Circle(200+i)={100+i, 5, 100+((i+1)%n)};
EndFor
Curve Loop(1)={1,2,3,4}; Curve Loop(2)={200:200+n-1};
Plane Surface(1)={1,2};
Physical Curve("hole")={200:200+n-1};
Mesh.Format=39; Mesh 2;
"""
    result = mesh_result_from_geo(geo)
    from fem2d import Mesh
    mesh = Mesh(nodes=result.nodes, elements=result.elements,
                E=210e9, nu=0.3, thickness=0.01,
                elem_type=result.elem_type)

    _SpyDetector.seen = []
    spy = _SpyDetector()
    registry = default_registry()
    original = list(registry._detectors)
    try:
        # 插入首位: 其余探测器会先命中, spy 须在判定前观察每次 classify
        registry._detectors.insert(0, spy)
        build_boundary_segments(mesh, registry=result.regions)
    finally:
        registry._detectors[:] = original

    seen_types = {kind for entry in spy.seen for kind in entry}
    assert "Circle" in seen_types, f"Circle 实体类型未传入探测器: {spy.seen}"
    assert "Line" in seen_types, f"Line 实体类型未传入探测器: {spy.seen}"


def test_segment_schema_validation():
    """段 schema 稳定化: 类型/节点链/坐标/标签/参数 缺一不可."""
    from fem2d import detect_boundaries
    from tests.test_boundary_golden_deterministic import _q4_square

    segs = detect_boundaries(_q4_square("CPS4"))
    assert validate_segment_schema(segs) is True

    bad = dict(segs[0])
    del bad["info"]
    with pytest.raises(ValueError, match="缺 schema 键"):
        validate_segment_schema([bad])

    bad_type = dict(segs[0])
    bad_type["type"] = "bogus"
    with pytest.raises(ValueError, match="受控枚举"):
        validate_segment_schema([bad_type])

    bad_coords = dict(segs[0])
    bad_coords["coords"] = bad_coords["coords"][:-1]
    with pytest.raises(ValueError, match="coords 行数"):
        validate_segment_schema([bad_coords])

    bad_nodes = dict(segs[0])
    bad_nodes["nodes"] = ["x", "y"]
    with pytest.raises(ValueError, match="整数节点"):
        validate_segment_schema([bad_nodes])

    with pytest.raises(TypeError, match="不是 dict"):
        validate_segment_schema(["not-a-segment"])


def test_schema_gate_location_contract():
    """schema 门禁位置契约 (判别性):
    - validate_boundary_segments 允许只含 nodes 的部分 dict (覆盖检查
      契约, test_boundary_joint 锁定);
    - validate_segment_schema 拒绝残缺段;
    - build_boundary_segments 返回的段全量通过 schema (管线出口)."""
    from fem2d import build_boundary_segments, validate_boundary_segments
    from fem2d.boundary.validation import validate_segment_schema
    from tests.test_boundary_golden_deterministic import _q4_square

    mesh = _q4_square("CPS4")
    partial = [
        {"nodes": [0, 1]},
        {"nodes": [1, 2]},
        {"nodes": [2, 3]},
        {"nodes": [3, 0]},
    ]
    validate_boundary_segments(mesh, partial)

    with pytest.raises(ValueError, match="缺 schema 键"):
        validate_segment_schema(partial)

    segments = build_boundary_segments(mesh)
    assert validate_segment_schema(segments) is True
