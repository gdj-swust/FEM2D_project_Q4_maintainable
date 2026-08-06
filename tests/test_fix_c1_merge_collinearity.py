"""C1 [P2][R5/F] _merge_adjacent_lines 共线判据修复 — 判别性测试.

审查发现 (review_20260805 R5): 旧合并判据 = axis 三档粗分类相等 +
|pos_i - pos_j| < scale×0.02, 而 tilted 线的 pos 就是起点 x 坐标 —
两条夹角 30° 的非共线斜线只要共享端点且起点 x 落在模型尺度 2% 内,
即被合并成一条 n=3 的弯曲"直边"段 (中点到弦偏离达弦长 13.4%).

修复: 合并判据改为方向向量共线 (归一化方向点积 |cos θ| ≥ 1-1e-6) +
共享端点, 起点坐标仅作挡板 (axis/pos 保留).

判别性 (放回旧判据必红):
- test_30deg_fold_stays_two_segments — 旧判据合并出 nodes=[3,4,5]
  弯曲直边 → 断言 "折线两段独立" 失败
- test_no_bent_line_segments — 旧判据下 gmsh 圆角矩形出 2 段 n=3
  弯曲直边 (maxdev=0.067) → 断言 "line 段无弯曲" 失败
- 合法共线合并 (竖直三节点边) 必须保留 — 防修复过度
"""
import numpy as np
import pytest

from fem2d import Mesh, detect_boundaries
from fem2d.boundary.topology import _merge_adjacent_lines

from tests.conftest import (
    GMSH_AVAILABLE,
    GMSH_UNAVAILABLE_REASON,
    mesh_result_from_geo,
)


def _max_chord_deviation(coords):
    """段内全部点到首尾弦的最大垂直距离 (相对弦长)."""
    chord = coords[-1] - coords[0]
    length = float(np.linalg.norm(chord))
    if length <= np.finfo(float).tiny:
        return 0.0
    deviations = []
    for point in coords[1:-1]:
        d = point - coords[0]
        deviations.append(
            abs(float(np.cross(np.append(chord, 0.0), np.append(d, 0.0))[2]))
            / length)
    return max(deviations, default=0.0) / length


def _fold_heptagon():
    """CCW 七边形: 底边 + 右边 + 顶边 30° 折线 (115°→145°, 各长 0.05).

    折线两段均为 tilted (axis 相等), 起点 x 差 = 0.0211 < scale×0.02
    (scale=1.787) — 旧判据必合并成 nodes=[3,4,5] 弯曲直边 (审查复现
    输出同构). 左侧竖直三节点边 (0,1.55)→(0,0.8)→(0,0) 是合法共线,
    修复后必须仍合并.
    """
    radius = 0.05
    n2 = np.array([2.0, 1.5])
    n3 = n2 + radius * np.array([
        np.cos(np.radians(115.0)), np.sin(np.radians(115.0))])
    n4 = n3 + radius * np.array([
        np.cos(np.radians(145.0)), np.sin(np.radians(145.0))])
    ring = np.array([
        [0.0, 0.0], [2.0, 0.0], n2, n3, n4,
        [0.0, 1.55], [0.0, 0.8],
    ])
    center = ring.mean(axis=0)
    nodes = np.vstack([center, ring])
    elems = np.array([
        [0, 1 + i, 1 + (i + 1) % 7] for i in range(7)
    ], dtype=int)
    return Mesh(nodes=nodes, elements=elems,
                E=1e6, nu=0.3, thickness=1.0, elem_type="CPS3")


def test_30deg_fold_stays_two_segments():
    """30° 折线必须拆成两条独立直边, 不得合并成 n=3 弯曲"直边"."""
    mesh = _fold_heptagon()
    segments = detect_boundaries(mesh)

    # 折线两段 (节点 3→4 与 4→5) 各自独立存在, 均为 n=2 直边 —
    # 折线中点节点 4 只属于这两段
    fold = [
        s for s in segments if 4 in s["nodes"]
    ]
    assert len(fold) == 2, (
        f"30° 折线应拆成 2 段, 实得 {len(fold)}: "
        f"{[s['label'] for s in fold]}")
    assert all(s["type"] == "line" for s in fold)
    assert all(len(s["nodes"]) == 2 for s in fold), (
        f"折线段必须 n=2, 实得 {[len(s['nodes']) for s in fold]}")

    # 旧判据的输出: 一条 n=3 弯曲直边 [3,4,5] — 不得出现
    assert not any(
        set(s["nodes"]) == {3, 4, 5} for s in segments), (
        "30° 折线被合并成弯曲直边段 (共线判据回归)")


def test_no_bent_line_segments():
    """不变量锁: 任何 n≥3 的 line 段, 全部点到弦的偏离 < 1e-9 相对弦长.

    弯曲"直边" (旧判据产物) 偏离 13.4%, 与真直边 (0) 相差 8+ 个量级.
    """
    mesh = _fold_heptagon()
    segments = detect_boundaries(mesh)
    for segment in segments:
        if segment["type"] != "line" or len(segment["nodes"]) < 3:
            continue
        relative = _max_chord_deviation(segment["coords"])
        assert relative < 1e-9, (
            f"line 段 {segment['label']!r} n={len(segment['nodes'])} "
            f"中点到弦偏离 {relative:.3e} — 弯曲段被标成直边")


def test_collinear_merge_preserved():
    """合法共线合并必须保留 (防修复过度): 左侧竖直三节点边仍为 n=3."""
    mesh = _fold_heptagon()
    segments = detect_boundaries(mesh)
    left = [s for s in segments if set(s["nodes"]) == {6, 7, 1}]
    assert len(left) == 1
    assert left[0]["type"] == "line"
    assert len(left[0]["nodes"]) == 3, (
        f"共线竖直边应合并为 n=3, 实得 {left[0]['label']!r}")


# ────────────────────────────────────────────────────────────────
# _merge_adjacent_lines 单元层: 判据语义逐一锁定
# ────────────────────────────────────────────────────────────────

def _seg(nodes, coords, axis, pos):
    return {
        "type": "line",
        "nodes": list(nodes),
        "coords": np.asarray(coords, dtype=float),
        "info": {"axis": axis, "pos": float(pos)},
    }


def test_merge_criterion_non_collinear_rejected():
    """30° 夹角斜线 (共享端点 + pos 在 2% 内) 不得合并."""
    nodes = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 1.4]])
    a = _seg([0, 1], [[0.0, 0.0], [1.0, 1.0]], "tilted", 0.0)
    b = _seg([1, 2], [[1.0, 1.0], [0.2, 1.4]], "tilted", 1.0)
    # 方向 45° vs 75° — 点积 = cos30° ≈ 0.866 < 1-1e-6
    merged = _merge_adjacent_lines([a, b], nodes, scale=10.0)
    assert len(merged) == 2


def test_merge_criterion_collinear_merged():
    """同向共线斜线 (共享端点) 必须合并."""
    nodes = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    a = _seg([0, 1], [[0.0, 0.0], [1.0, 1.0]], "tilted", 0.0)
    b = _seg([1, 2], [[1.0, 1.0], [2.0, 2.0]], "tilted", 1.0)
    # scale=100 → 2% 容差 = 2.0 ≥ pos 差 1.0, pos 挡板放行, 共线判据通过
    merged = _merge_adjacent_lines([a, b], nodes, scale=100.0)
    assert len(merged) == 1
    assert merged[0]["nodes"] == [0, 1, 2]


def test_merge_criterion_antiparallel_blocked_by_pos_guard():
    """反平行共线 (方向点积 = -1, 回头段) — pos 挡板必须阻止合并."""
    nodes = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    a = _seg([0, 1], [[0.0, 0.0], [1.0, 1.0]], "tilted", 0.0)
    b = _seg([1, 2], [[1.0, 1.0], [0.0, 2.0]], "tilted", 1.0)
    # 方向 (1,1)/√2 vs (-1,1)/√2 — |cos θ| = 1, 但 pos 差 = 1 ≥ 2%×scale
    merged = _merge_adjacent_lines([a, b], nodes, scale=10.0)
    assert len(merged) == 2


# ────────────────────────────────────────────────────────────────
# gmsh 真实圆角矩形 (粗网格) — 审查复现场景的判别性层
# ────────────────────────────────────────────────────────────────

def _rounded_rect_geo():
    """圆角矩形 8.0×4.0 圆角 R=0.3, lc=0.8 (审查复现场景的参数化).

    参数经旧判据实测筛选: 该组合下旧判据合并出 2 段 n=3 弯曲直边
    (中点到弦偏离 20.7% 弦长) — 判别性; 圆角弧仅 1-2 条网格边, 同时
    是 C2 描述的"角弧 ≤4 条网格边"悬崖场景. Circle(id) = {start,
    center, end} (Gmsh 语法).
    """
    return """\
lc = 0.8;
W = 4.0; H = 2.0; R = 0.3;
// 直线端点 (圆角弧与直边的切点)
Point(1) = { -W+R, -H, 0, lc};
Point(2) = {  W-R, -H, 0, lc};
Point(3) = {  W,  -H+R, 0, lc};
Point(4) = {  W,   H-R, 0, lc};
Point(5) = {  W-R,  H, 0, lc};
Point(6) = { -W+R,  H, 0, lc};
Point(7) = { -W,   H-R, 0, lc};
Point(8) = { -W,  -H+R, 0, lc};
// 四个圆角弧心
Point(9)  = {  W-R, -H+R, 0, lc};
Point(10) = {  W-R,  H-R, 0, lc};
Point(11) = { -W+R,  H-R, 0, lc};
Point(12) = { -W+R, -H+R, 0, lc};
Line(1) = {1, 2};
Circle(2) = {2, 9, 3};
Line(3) = {3, 4};
Circle(4) = {4, 10, 5};
Line(5) = {5, 6};
Circle(6) = {6, 11, 7};
Line(7) = {7, 8};
Circle(8) = {8, 12, 1};
Curve Loop(100) = {1, 2, 3, 4, 5, 6, 7, 8};
Plane Surface(1) = {100};
Physical Surface("domain", 200) = {1};
Mesh 2;
"""


@pytest.mark.skipif(
    not GMSH_AVAILABLE, reason=GMSH_UNAVAILABLE_REASON)
def test_gmsh_rounded_rect_no_bent_line():
    """粗网格圆角矩形: 不得出现偏离弦的弯曲"直边"段.

    审查复现 (圆角矩形 lc=0.3) 出 2 段 n=3 弯曲直边 maxdev=0.067;
    本测试参数 (8×4, R=0.3, lc=0.8) 经旧判据实测触发同类合并
    (n=3, 偏离 20.7% 弦长) — 修复后 0.
    """
    result = mesh_result_from_geo(_rounded_rect_geo())
    mesh = Mesh(
        nodes=result.nodes, elements=result.elements,
        E=210e9, nu=0.3, thickness=0.01,
        elem_type=result.elem_type)
    segments = detect_boundaries(mesh)
    assert len(segments) >= 4

    bent = [
        (s["label"], _max_chord_deviation(s["coords"]))
        for s in segments
        if s["type"] == "line" and len(s["nodes"]) >= 3
    ]
    for label, relative in bent:
        assert relative < 1e-9, (
            f"gmsh 圆角矩形出弯曲直边: {label!r} 偏离 {relative:.3e}")
