"""覆盖轮 C1 — boundary 模块第二轮缺口 (质心分支/退化拟合/上下文校验)."""
import numpy as np
import pytest

from fem2d.boundary import topology as T
from fem2d.boundary import geometry as G
from fem2d.boundary.detectors.ellipse import EllipseDetector
from fem2d.boundary.detectors.general import GeneralCurveDetector
from fem2d.boundary.selectors import BoundarySelector
from fem2d.boundary.segment_utils import LoopContext


# ── topology.py 第二轮 ─────────────────────────────────────────────────────

def test_loop_probe_centroid_outside_mean_inside(monkeypatch):
    """质心域外 (凹环) 但顶点均值在环内 → 均值分支."""
    def fake_in_loop(px, py, xs, ys):
        return abs(px - 1.0) < 1e-9 and abs(py - 1.0) < 1e-9
    monkeypatch.setattr(T, "_point_in_loop", fake_in_loop)
    xs = np.array([0., 2., 2., 1., 1., 0.])
    ys = np.array([0., 0., 1., 1., 2., 2.])
    # mean=(1,1) 命中; 质心 (≈0.78,0.78) 不命中 → 走均值分支
    assert T._loop_probe_point(xs, ys) == (1.0, 1.0)


def test_loop_probe_zero_length_edge_skipped(monkeypatch):
    """全部域内检查失败 + 含零长边 → 边循环跳过零长边后兜底均值."""
    monkeypatch.setattr(T, "_point_in_loop", lambda *a, **k: False)
    xs = np.array([0., 0., 0., 2., 2., 0.])
    ys = np.array([0., 0., 2., 2., 0., 0.])
    px, py = T._loop_probe_point(xs, ys)
    assert (px, py) == (float(np.mean(xs)), float(np.mean(ys)))


def test_intersecting_pair_all_empty_arrays():
    """所有环坐标数组为空 → 无有效数组 → None."""
    assert T._intersecting_boundary_loop_pair(
        [np.zeros((0, 2)), np.zeros((0, 2))]) is None


# ── geometry.py 第二轮 ─────────────────────────────────────────────────────

def test_fit_circle_scale_degenerate():
    """全部同点 → 归一化尺度为 0 → 失败."""
    points = np.array([[0., 0.], [0., 0.], [0., 0.]])
    assert G.fit_circle_least_squares(points)[2] < 0.0


def test_fit_ellipse_lstsq_failure(monkeypatch):
    """SVD/lstsq 解算异常 → None (防御)."""
    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular")
    monkeypatch.setattr(np.linalg, "lstsq", boom)
    pts = np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 8)),
                           np.sin(np.linspace(0, 2 * np.pi, 8))])
    assert G.fit_ellipse(pts) is None


def test_fit_ellipse_degenerate_discriminant():
    """共线点 → 二次型判别式 ≤0 → 非椭圆 → None."""
    pts = np.column_stack([np.linspace(0, 1, 6), np.zeros(6)])
    assert G.fit_ellipse(pts) is None


def test_fit_ellipse_high_aspect_ratio_rejected():
    """真实椭圆但长短轴比 >20 → 拒绝."""
    t = np.linspace(0, 2 * np.pi, 10)
    pts = np.column_stack([10.0 * np.cos(t), 0.05 * np.sin(t)])
    assert G.fit_ellipse(pts) is None


# ── selectors.py 第二轮 ────────────────────────────────────────────────────

def test_selector_outer_shortcut_not_label_collision():
    """段 label 不含 'o' 时 'o' 快捷必须命中 is_outer 段."""
    outer = {"type": "line", "label": "out", "nodes": [0, 1],
             "coords": np.zeros((2, 2)), "info": {"is_outer": True}}
    inner = {"type": "line", "label": "inn", "nodes": [0, 1],
             "coords": np.zeros((2, 2)), "info": {"is_outer": False}}
    sel = BoundarySelector([inner, outer])
    assert sel.resolve("o") == [1]


# ── detectors.py 第二轮 ────────────────────────────────────────────────────

def test_ellipse_detector_residual_gate():
    """椭圆拟合成功但残差超阈 (离群点) → 拒绝椭圆."""
    t = np.linspace(0, 2 * np.pi, 9)
    pts = np.column_stack([2.0 * np.cos(t), np.sin(t)])
    pts[5] = (2.0, 3.0)  # 离群 → max_residual ≥ 2e-3
    result = EllipseDetector().detect(
        pts, scale=1.0, is_outer=True, closed=False, native_entities=())
    assert result is None


def test_general_detector_open_chain_empty_curvature():
    """开放链过短 → 曲率序列为空 → 零向量兜底."""
    coords = np.array([[0., 0.], [1., 0.]])
    result = GeneralCurveDetector().detect(
        coords, scale=1.0, is_outer=True, closed=False, native_entities=())
    assert result is not None


# ── segment_utils.py ───────────────────────────────────────────────────────

def test_loop_context_missing_edge():
    with pytest.raises(ValueError, match="no topology"):
        LoopContext.from_edges(
            "seg", [(0, 1)], {(1, 2): {"loop_id": 0}})


def test_loop_context_inconsistent_topology():
    ctx = {
        (0, 1): {"loop_id": 0, "loop_depth": 0, "is_outer": True},
        (1, 2): {"loop_id": 1, "loop_depth": 0, "is_outer": True},
    }
    with pytest.raises(ValueError, match="inconsistent"):
        LoopContext.from_edges("seg", [(0, 1), (1, 2)], ctx)
