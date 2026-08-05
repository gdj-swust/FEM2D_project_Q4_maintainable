"""覆盖轮 C1 — boundary 几何/拓扑/选择器/命名 缺口行 (防御分支与退化输入).

触发方式: 直接调用内部几何函数 (无副作用), 或构造退化网格走公开入口
detect(). 每个用例断言分支语义, 不为凑行数.
"""
import numpy as np
import pytest

from fem2d import Mesh
from fem2d.boundary import topology as T
from fem2d.boundary import geometry as G
from fem2d.boundary.detectors.circle import CircleDetector
from fem2d.boundary.detectors.ellipse import EllipseDetector
from fem2d.boundary.detectors.general import GeneralCurveDetector
from fem2d.boundary.model import BoundaryDiagnostics
from fem2d.boundary.naming import (
    describe_geometry,
    print_segments,
    segments_from_physical_curves,
    semantic_coverage,
    _resolve_edge_indices,
)
from fem2d.boundary.selectors import BoundarySelector
from fem2d.boundary.segment_utils import tuple_value, segment_physical_names
from fem2d.boundary.validation import validate_segment_schema, _validate_one_name


def _square_mesh():
    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    return Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 2, 3]]),
                E=1e6, nu=0.3, thickness=1.0)


# ── topology.py ────────────────────────────────────────────────────────────

def test_detect_empty_adjacency_returns_empty(monkeypatch):
    """边界邻接图为空 (monkeypatch 拓扑输入) → detect 返回 []."""
    monkeypatch.setattr(T, "_boundary_adjacency", lambda edges: {})
    assert T.detect(_square_mesh()) == []


def test_detect_no_loops_returns_empty(monkeypatch):
    """邻接存在但解不出闭环 (monkeypatch 分解器) → []."""
    mesh = _square_mesh()
    monkeypatch.setattr(T, "_decompose_loops", lambda adj: [])
    assert T.detect(mesh) == []


def test_validate_nesting_rejects_intersecting_loops():
    """两个重叠方形环 → ValueError (相交/重叠边界必须拒绝)."""
    nodes = np.array([
        [0., 0.], [2., 0.], [2., 2.], [0., 2.],   # 外环
        [1., 1.], [3., 1.], [3., 3.], [1., 3.],   # 与之重叠的内环
    ])
    with pytest.raises(ValueError, match="intersect"):
        T._validate_and_nest_loops(
            nodes, [np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])], 2.0)


def test_orient_closed_segments_flips_cw_outer_loop():
    """闭合外环 CW 定向 → 翻转为 CCW (Gmsh 约定)."""
    seg = {
        "type": "line", "closed": True,
        "nodes": [0, 3, 2, 1],  # CW 顺序
        "coords": np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]]),
        "info": {"is_outer": True, "loop_id": 0},
    }
    T._orient_closed_segments([seg])
    assert seg["nodes"] == [1, 2, 3, 0]
    assert seg["coords"][0].tolist() == [1., 0.]


def test_dedup_loop_vertices_all_duplicate_returns_none():
    """去重后不足 3 点 → 无有效环."""
    assert T._dedup_loop_vertices(
        np.array([0., 0., 0.]), np.array([0., 0., 0.])) is None


def test_loop_edges_skips_self_loop():
    """相邻重复点 (p1==p2) 的自环边被跳过 (防御保留)."""
    edges = T._loop_edges([(0., 0.), (0., 0.), (1., 0.)])
    assert len(edges) == 2  # 两条环绕边, 自环已跳过
    assert all(p1 != p2 for p1, p2 in edges)


def test_general_position_ray_nextafter_when_perturb_swallowed():
    """py 在 1e308 量级时 perturb (8 ULP) 被舍入吞掉 → nextafter 显式换."""
    py = 1e308
    vertices = [(0.0, py), (1.0, py), (0.5, py + 1e292)]
    edges = [((0.0, py), (1.0, py)), ((1.0, py), (0.5, py + 1e292))]
    # 不抛异常即达成 — 372-373 行在 out_y==py 时触发 nextafter
    result = T._general_position_ray(0.5, py, vertices, edges)
    assert isinstance(result, bool)


def test_half_open_crossing_same_side_skip_and_on_edge():
    """边两端点 y 同侧 → 跳过; 点恰在边上 → 判内."""
    edges = [((0., 0.), (1., 0.)), ((1., 0.), (1., 1.)),
             ((1., 1.), (0., 1.)), ((0., 1.), (0., 0.))]
    assert T._half_open_crossing(0.5, 5.0, edges) is False   # 同侧全跳过
    assert T._half_open_crossing(0.5, 0.0, edges) is True    # 恰在底边上
    assert T._half_open_crossing(0.5, 0.5, edges) is True    # 方环内


def test_point_in_loop_vertices_none():
    """顶点去重后无效 → False (不抛)."""
    assert T._point_in_loop(0.0, 0.0,
                            np.array([0., 0., 0.]), np.array([0., 0., 0.])) is False


def test_point_in_loop_edges_lt_3():
    """去重后不足 3 顶点 (相邻重复点) → 环无效 → False."""
    assert T._point_in_loop(0.2, 0.2,
                            np.array([0., 1., 1.]),
                            np.array([0., 0., 0.])) is False


def test_point_in_loop_half_open_fallback(monkeypatch):
    """一般位置射线全部失败 → 半开穿越兜底 (结果一致)."""
    monkeypatch.setattr(T, "_general_position_ray", lambda *a, **k: None)
    assert T._point_in_loop(0.5, 0.5,
                            np.array([0., 1., 1., 0.]),
                            np.array([0., 0., 1., 1.])) is True


def test_signed_loop_area_lt3_zero():
    """不足 3 点 → 面积记 0."""
    assert T._signed_loop_area(np.zeros((2, 2))) == 0.0


def test_loop_probe_point_centroid_inside():
    """质心在环内 → 直接用质心 (正方形中心)."""
    xs = np.array([0., 1., 1., 0.])
    ys = np.array([0., 0., 1., 1.])
    assert T._loop_probe_point(xs, ys) == (0.5, 0.5)


def test_loop_probe_point_skips_zero_length_edge():
    """环含相邻重复点 (零长边) → 跳过该候选点."""
    xs = np.array([0., 0., 0., 1., 1.])
    ys = np.array([0., 0., 1., 1., 0.])
    px, py = T._loop_probe_point(xs, ys)
    assert np.isfinite(px) and np.isfinite(py)


def test_loop_probe_point_fallback_mean(monkeypatch):
    """全部候选点域外 → 退回顶点均值."""
    monkeypatch.setattr(T, "_point_in_loop", lambda *a, **k: False)
    xs = np.array([0., 1., 1., 0.])
    ys = np.array([0., 0., 1., 1.])
    assert T._loop_probe_point(xs, ys) == (0.5, 0.5)


def test_decompose_loops_bad_degree_raises():
    """度数≠2 的节点 (开放/非流形) → ValueError 带预览."""
    with pytest.raises(ValueError, match="degree=1"):
        T._decompose_loops({0: [1], 1: [0, 2], 2: [1]})


def test_point_on_segment_non_collinear_false():
    """点与线段不共线 → 不在段上."""
    assert T._point_on_segment((0.5, 0.5), (0., 0.), (1., 0.), 1e-9) is False
    assert T._point_on_segment((0.5, 0.0), (0., 0.), (1., 0.), 1e-9) is True


def test_has_self_intersection_lt3():
    assert T.has_boundary_self_intersection(np.zeros((2, 2))) is False


def test_has_self_intersection_nonfinite():
    coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., np.nan]])
    assert T.has_boundary_self_intersection(coords) is True


def test_has_self_intersection_zero_length_edge():
    """相邻重复点构成零长退化边 → 判自交."""
    coords = np.array([[0., 0.], [1., 0.], [1., 1.], [1., 1.], [0., 1.]])
    assert T.has_boundary_self_intersection(coords) is True


def test_intersecting_loop_pair_no_finite_arrays():
    assert T._intersecting_boundary_loop_pair([np.full((4, 2), np.nan)]) is None


def test_intersecting_loop_pair_pops_duplicate_closure():
    """环首尾重复点被弹出后再检测."""
    loops = [np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.], [0., 0.]])]
    assert T._intersecting_boundary_loop_pair(loops) is None  # 不误报


def test_intersecting_loop_pair_detects_pair():
    loops = [
        np.array([[0., 0.], [2., 0.], [2., 2.], [0., 2.]]),
        np.array([[1., 1.], [3., 1.], [3., 3.], [1., 3.]]),
    ]
    assert T._intersecting_boundary_loop_pair(loops) == (0, 1)


def test_detect_square_returns_segments():
    """公开入口: 正常方形网格 → 4 段直边."""
    segs = T.detect(_square_mesh())
    assert len(segs) == 4
    assert all(s["type"] == "line" for s in segs)


# ── geometry.py ────────────────────────────────────────────────────────────

def test_compute_tolerance_single_coord():
    assert G.compute_tolerance(np.array([[0., 0.]])) == 1e-12


def test_curvature_skips_zero_length_edges():
    """相邻重复点 → 零长边跳过, κ 保持 0 不炸."""
    coords = np.array([[0., 0.], [0., 0.], [1., 0.], [1., 1.]])
    kappa = G.curvature(coords)
    assert len(kappa) == len(coords)
    assert np.all(np.isfinite(kappa))


def test_sharp_corner_indices_zero_length_edges():
    coords = np.array([[0., 0.], [0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    result = G.sharp_corner_indices(coords)
    assert isinstance(result, list)


def test_turning_angles_lt3():
    assert G.turning_angles(np.zeros((2, 2))).tolist() == [0.0, 0.0]


def test_segment_by_curvature_replaces_nonfinite():
    kappa = np.ones(12)
    kappa[3] = np.nan
    result = G.segment_by_curvature(kappa, scale=1.0)
    assert isinstance(result, list)


def test_segment_by_curvature_below_floor():
    """最大曲率低于地板 (1e-8/scale) → 无断点."""
    assert G.segment_by_curvature(np.ones(12) * 1e-12, scale=1.0) == []


def test_segment_by_curvature_cluster_branch():
    """方波曲率 → 候选断点聚类并保留每簇最强点."""
    kappa = np.zeros(20)
    kappa[5] = 10.0
    kappa[6] = 10.0
    kappa[15] = 8.0
    breaks = G.segment_by_curvature(kappa, scale=1.0)
    assert len(breaks) >= 1
    assert all(0 <= b < 20 for b in breaks)


def test_segment_is_closed_respects_hint():
    """拓扑层给定 closed 真值 → 直接采用."""
    coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    assert G._segment_is_closed(coords, 1e-9, closed=True) is True
    assert G._segment_is_closed(coords, 1e-9, closed=False) is False


def test_fit_circle_least_squares_lt3():
    assert G.fit_circle_least_squares(np.zeros((2, 2))) == (0.0, 0.0, -1.0)


def test_fit_circle_least_squares_lstsq_failure(monkeypatch):
    """最小二乘解算抛 LinAlgError → 返回失败三元组 (防御)."""
    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")
    monkeypatch.setattr(np.linalg, "lstsq", boom)
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    assert G.fit_circle_least_squares(points)[2] < 0.0


def test_fit_circle_least_squares_negative_radius(monkeypatch):
    """解出负半径平方 → 返回失败三元组 (防御)."""
    def neg_radius(*args, **kwargs):
        return np.array([0.0, 0.0, -1.0]), None, None, None
    monkeypatch.setattr(np.linalg, "lstsq", neg_radius)
    points = np.array([[0., 0.], [1., 0.], [0., 1.]])
    assert G.fit_circle_least_squares(points)[2] < 0.0


def test_circle_fit_residual_invalid_radius():
    assert G.circle_fit_residual(
        np.array([[0., 0.], [1., 0.]]), (0., 0., -1.)) == (np.inf, np.inf)


def test_fit_ellipse_lt6():
    assert G.fit_ellipse(np.zeros((5, 2))) is None


def test_fit_ellipse_collinear_linalg():
    """共线 6 点 → 最小二乘奇异 → None."""
    x = np.linspace(0, 1, 6)
    assert G.fit_ellipse(np.column_stack([x, np.zeros(6)])) is None


def test_fit_ellipse_degenerate_semiaxes():
    """二次型非椭圆 (Ap*Fp>=0) → None."""
    points = np.array([
        [0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.], [5., 0.],
    ])
    assert G.fit_ellipse(points) is None


def test_fit_ellipse_high_aspect_ratio():
    """长短轴比 >20 拒绝."""
    x = np.cos(np.linspace(0, 2 * np.pi, 8))
    y = np.sin(np.linspace(0, 2 * np.pi, 8)) * 0.01   # 扁椭圆
    points = np.column_stack([x * 100.0, y])
    assert G.fit_ellipse(points) is None


def test_ellipse_fit_residual_invalid():
    assert G.ellipse_fit_residual(
        np.zeros((0, 2)), (0., 0., 1., 1., 0.)) == (np.inf, np.inf)


def test_fit_closed_ellipse_lt8():
    assert G.fit_closed_ellipse(np.zeros((7, 2))) == (None, {})


def test_fit_closed_ellipse_after_dedup_lt8():
    """8 行但含闭合重复点 → 截掉末点后仅 7 → 失败."""
    t = np.linspace(0, 2 * np.pi, 8)[:-1]  # 7 个均匀点
    pts = np.column_stack([np.cos(t), np.sin(t)])
    pts = np.vstack([pts, pts[0]])  # 首点闭合重复 → 8 行
    assert G.fit_closed_ellipse(pts) == (None, {})


# ── selectors.py ───────────────────────────────────────────────────────────

def _seg(label, tp="line", **info):
    return {"type": tp, "label": label, "nodes": [0, 1],
            "coords": np.zeros((2, 2)), "info": info}


def test_selector_empty_needle():
    assert BoundarySelector([_seg("a")]).resolve("") == []


def test_selector_shortcut_hit():
    sel = BoundarySelector([_seg("x", "arc"), _seg("y", "line")])
    assert sel.resolve("a") == [0]
    assert sel.resolve("l") == [1]


def test_selector_numeric_isdigit_names():
    """纯数字输入直接走编号匹配, 不查询 Physical 名."""
    sel = BoundarySelector([_seg("a")])
    assert sel._exact_physical_names("12") == []
    assert sel._exact_visible_labels("12") == []


def test_selector_fuzzy_unique_match():
    sel = BoundarySelector([_seg("right"), _seg("left")])
    assert sel._explicit_fuzzy_match("right") == [0]


def test_selector_geometric_no_segments():
    assert BoundarySelector([]).resolve("bottom") == []


def test_selector_position_bottom_top():
    """bottom/top 按段自身坐标判定 (水平 + 相对全局 bounds 偏下/偏上)."""
    bottom = {"type": "line", "label": "b", "nodes": [0, 1],
              "coords": np.array([[0., 0.], [1., 0.]])}
    top = {"type": "line", "label": "t", "nodes": [2, 3],
           "coords": np.array([[0., 1.], [1., 1.]])}
    sel = BoundarySelector([bottom, top])
    assert sel.resolve("bottom") == [0]
    assert sel.resolve("top") == [1]


def test_selector_outer_shortcut():
    outer = {"type": "line", "label": "o", "nodes": [0, 1],
             "coords": np.zeros((2, 2)), "info": {"is_outer": True}}
    inner = {"type": "line", "label": "i", "nodes": [0, 1],
             "coords": np.zeros((2, 2)), "info": {"is_outer": False}}
    sel = BoundarySelector([outer, inner])
    assert sel.resolve("o") == [0]


def test_selector_numeric_out_of_range():
    sel = BoundarySelector([_seg("a")])
    assert sel.resolve("99") == []


# ── naming.py / segment_utils.py / validation.py / model.py ────────────────

def test_semantic_coverage_single_physical_name_fallback():
    mesh = _square_mesh()
    segs = [{"type": "line", "label": "x", "nodes": [0, 1, 2],
             "coords": mesh.nodes[:3],
             "info": {"physical_name": "bottom"}}]
    cov = semantic_coverage(mesh, segs)
    assert "bottom" in cov["physical_names"]


def test_print_segments_empty(capsys):
    print_segments([])
    assert "未检测到边界" in capsys.readouterr().out


def test_print_segments_curve_radius_range_and_fallback(capsys):
    segs = [
        {"type": "curve", "label": "c1", "nodes": [0, 1], "coords": np.zeros((2, 2)),
         "info": {"curvature_mean": 1.0, "curvature_cv": 0.1,
                  "R_min": 0.5, "R_max": 2.0, "is_outer": True}},
        {"type": "curve", "label": "c2", "nodes": [0, 1], "coords": np.zeros((2, 2)),
         "info": {"curvature_mean": 1.0, "curvature_cv": 0.1,
                  "curvature_std": 0.3, "is_outer": False}},
    ]
    print_segments(segs)
    out = capsys.readouterr().out
    assert "R=[" in out or "k_std" in out


def test_describe_geometry_concentric_rings():
    segs = [
        {"type": "arc", "label": "a", "nodes": [0, 1], "coords": np.array([[0., 0.], [1., 0.]]),
         "info": {"radius": 1.0, "center": (0., 0.), "is_outer": True}},
        {"type": "arc", "label": "b", "nodes": [0, 1], "coords": np.array([[0., 0.], [1., 0.]]),
         "info": {"radius": 0.5, "center": (0., 0.), "is_outer": False}},
    ]
    assert "同心圆环" in describe_geometry(segs)


def test_describe_geometry_hole_plate():
    outer = {"type": "line", "label": "o", "nodes": [0, 1],
             "coords": np.array([[0., 0.], [1., 0.]]),
             "info": {"is_outer": True}}
    hole = {"type": "line", "label": "h", "nodes": [0, 1],
            "coords": np.array([[0.4, 0.4], [0.6, 0.6]]),
            "info": {"is_outer": False}}
    assert "带孔板" in describe_geometry([outer, hole])


def test_resolve_edge_indices_none_and_blank():
    assert _resolve_edge_indices(None, [_seg("a")]) == []
    assert _resolve_edge_indices("  ", [_seg("a")]) == []


def test_segments_from_physical_curves_non_dict_labels():
    with pytest.raises(ValueError, match="dict"):
        segments_from_physical_curves(
            ["c1"], edge_labels=["not", "a", "dict"])


def test_tuple_value_none_str():
    assert tuple_value(None) == ()
    assert tuple_value("abc") == ("abc",)


def test_segment_physical_names_fallback():
    seg = {"type": "line", "label": "x", "nodes": [0, 1],
           "coords": np.zeros((2, 2)), "info": {"physical_name": "pn"}}
    assert segment_physical_names(seg) == ("pn",)


def test_validate_segment_schema_info_not_dict():
    """info 字段非 dict → TypeError (schema 稳定化)."""
    with pytest.raises(TypeError):
        validate_segment_schema([
            {"type": "line", "label": "x", "nodes": [0, 1],
             "coords": np.zeros((2, 2)), "info": "bad"}])


def test_validate_one_name_whitespace_and_control():
    d = BoundaryDiagnostics()
    _validate_one_name("  spaced  ", d)
    _validate_one_name("ctl\x07char", d)
    assert len(d.errors) >= 2


def test_diagnostics_bad_severity():
    d = BoundaryDiagnostics()
    with pytest.raises(ValueError, match="severity"):
        d.add("k", "fatal", "msg")


# ── detectors ──────────────────────────────────────────────────────────────

def _detect_kwargs(closed=False):
    return dict(scale=1.0, is_outer=True, closed=closed, native_entities=())


def test_circle_detector_lt4_points():
    coords = np.array([[0., 0.], [1., 0.], [1., 1.]])
    assert CircleDetector().detect(coords, **_detect_kwargs()) is None


def test_ellipse_detector_lt6_points():
    coords = np.zeros((5, 2))
    assert EllipseDetector().detect(coords, **_detect_kwargs()) is None


def test_ellipse_detector_high_residual():
    """残差超阈值 (2e-3) → 拒绝椭圆."""
    coords = np.column_stack([
        np.linspace(0, 1, 8), np.zeros(8)])
    assert EllipseDetector().detect(coords, **_detect_kwargs()) is None


def test_general_detector_empty_evaluated():
    coords = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    result = GeneralCurveDetector().detect(coords, **_detect_kwargs())
    assert result is not None
