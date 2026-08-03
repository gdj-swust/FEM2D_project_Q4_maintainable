"""_point_in_loop 重构行为锁定测试 (包 5 任务 2).

``_REFERENCE`` 是重构前的完整实现快照 (2026-08-03 基线)。重构后
生产实现必须对同一输入返回完全相同的布尔值 — 任何分支/容差/顺序
变化都会使本文件失败。测试同时覆盖微尺度 / 大坐标 / 边界 / 顶点 /
退化输入, 这是 _point_in_loop 的全部调用上下文 (环嵌套深度判定 +
probe 点选择, 见 topology.py)。
"""
import numpy as np

from fem2d.boundary.topology import _point_in_loop


def _reference(px, py, xs, ys):
    """冻结的旧实现 — 与重构无关, 只做行为对照."""
    from fem2d.boundary.predicates import orient2d

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) != len(ys) or len(xs) < 3:
        return False
    if not (
            np.isfinite(px) and np.isfinite(py)
            and np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return False

    vertices = []
    for x, y in zip(xs, ys):
        point = (float(x), float(y))
        if not vertices or point != vertices[-1]:
            vertices.append(point)
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices.pop()
    if len(vertices) < 3:
        return False

    edges = []
    for i, p1 in enumerate(vertices):
        p2 = vertices[(i + 1) % len(vertices)]
        if p1 == p2:
            continue
        edges.append((p1, p2))
        side = orient2d(
            p1[0], p1[1], p2[0], p2[1], float(px), float(py))
        if side == 0.0 and (
                min(p1[0], p2[0]) <= px <= max(p1[0], p2[0])
                and min(p1[1], p2[1]) <= py <= max(p1[1], p2[1])):
            return True
    if len(edges) < 3:
        return False

    vx = np.array([point[0] for point in vertices])
    vy = np.array([point[1] for point in vertices])
    span = max(float(np.ptp(vx)), float(np.ptp(vy)))
    magnitude = max(
        float(np.max(np.abs(vx))), float(np.max(np.abs(vy))),
        abs(float(px)), abs(float(py)), 1.0)
    margin = max(span * 2.0, np.spacing(magnitude) * 64.0, 1.0)
    out_x = np.nextafter(float(np.max(vx)) + margin, np.inf)
    perturb = max(
        np.spacing(magnitude) * 8.0,
        np.finfo(float).eps * max(span, 1.0) * 8.0)

    factors = (1.0, -1.0, np.sqrt(2.0), -np.sqrt(2.0),
               np.pi, -np.pi, np.e, -np.e)
    for factor in factors:
        out_y = float(py) + perturb * factor
        if out_y == py:
            out_y = np.nextafter(
                float(py), np.inf if factor > 0 else -np.inf)
        if any(
                orient2d(float(px), float(py), out_x, out_y, x, y) == 0.0
                for x, y in vertices):
            continue

        count = 0
        for (x1, y1), (x2, y2) in edges:
            a = orient2d(x1, y1, x2, y2, float(px), float(py))
            b = orient2d(x1, y1, x2, y2, out_x, out_y)
            if a * b >= 0.0:
                continue
            c = orient2d(float(px), float(py), out_x, out_y, x1, y1)
            d = orient2d(float(px), float(py), out_x, out_y, x2, y2)
            if c * d < 0.0:
                count += 1
        return count % 2 == 1

    inside = False
    for (x1, y1), (x2, y2) in edges:
        if (y1 > py) == (y2 > py):
            continue
        side = orient2d(x1, y1, x2, y2, float(px), float(py))
        if side == 0.0:
            return True
        if (side > 0.0) == (y2 > y1):
            inside = not inside
    return inside


# ═══════════════════════════════════════════════════════════════
# 固定输入电池: 覆盖全部调用上下文 + 边界/顶点/退化/微尺度/大坐标
# ═══════════════════════════════════════════════════════════════

def _square(s, origin=(0.0, 0.0)):
    x0, y0 = origin
    return (
        [x0, x0 + s, x0 + s, x0],
        [y0, y0, y0 + s, y0 + s],
    )


def _u_shape():
    """U 形凹多边形: 槽 (5,5) 在外, 两柱在内."""
    xs = [0, 10, 10, 9, 9, 1, 1, 0]
    ys = [0, 0, 10, 10, 9, 9, 10, 10]
    return xs, ys


def _star():
    """五角星 (凹) — 顶点/边界的射线对准敏感形状."""
    angles = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    xs = []
    ys = []
    for i, a in enumerate(angles):
        r = 1.0 if i % 2 == 0 else 0.5
        xs.append(r * np.cos(a))
        ys.append(r * np.sin(a))
    return xs, ys


def _collinear():
    """全部共线 (退化) — 必须返回 False 且不崩溃."""
    return [0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0]


BATTERY = [
    # (标签, xs, ys, 测试点列表)
    ("square", *_square(1.0), [(0.5, 0.5), (2.0, 2.0), (0.0, 0.5),
                                (1.0, 1.0), (0.5, 0.0), (-0.1, 0.5)]),
    ("square_big_coord", *_square(1.0, origin=(1e12, 1e12)),
     [(1e12 + 0.5, 1e12 + 0.5), (1e12 + 5.0, 1e12), (1e12, 1e12),
      (1e12 + 0.0, 1e12 + 0.5)]),
    ("square_micro", *_square(1e-150),
     [(5e-151, 5e-151), (1e-149, 1e-149), (0.0, 5e-151),
      (1e-150, 1e-150), (1e-150, 0.0)]),
    ("u_shape", *_u_shape(), [(1.0, 5.0), (5.0, 5.0), (5.0, 1.0),
                               (9.0, 5.0), (5.0, 10.0), (0.0, 5.0)]),
    ("star", *_star(), [(0.0, 0.0), (0.0, 0.75), (0.0, 1.5),
                         (0.0, 1.0), (0.4, 0.4), (1.0, 0.0)]),
    ("collinear", *_collinear(), [(1.0, 0.0), (1.0, 1.0)]),
    ("two_points", [0.0, 1.0], [0.0, 1.0], [(0.5, 0.5)]),
    ("closed_dup", [0.0, 1.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 1.0, 0.0], [(0.5, 0.5), (2.0, 2.0)]),
]

RANDOM_SEED = 20260803


def _random_battery():
    """固定种子随机多边形 × 规则采样点网格 — 回归面."""
    rng = np.random.default_rng(RANDOM_SEED)
    cases = []
    for _ in range(24):
        n = int(rng.integers(3, 9))
        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
        radius = rng.uniform(0.5, 1.5, n)
        xs = list(radius * np.cos(angles))
        ys = list(radius * np.sin(angles))
        # 含点在环内外的规则网格 (含恰在顶点/边上的概率)
        points = []
        for gx in np.linspace(-2.0, 2.0, 9):
            for gy in np.linspace(-2.0, 2.0, 9):
                points.append((float(gx), float(gy)))
        cases.append((xs, ys, points))
    return cases


def test_reference_matches_production_on_fixed_battery():
    mismatches = []
    for label, xs, ys, points in BATTERY:
        for px, py in points:
            expected = _reference(px, py, xs, ys)
            got = _point_in_loop(px, py, xs, ys)
            if expected != got:
                mismatches.append((label, px, py, expected, got))
    assert not mismatches, f"固定电池分歧: {mismatches[:5]}"


def test_reference_matches_production_on_random_battery():
    mismatches = []
    for xs, ys, points in _random_battery():
        for px, py in points:
            expected = _reference(px, py, xs, ys)
            got = _point_in_loop(px, py, xs, ys)
            if expected != got:
                mismatches.append((px, py, expected, got))
    assert not mismatches, f"随机电池分歧: {mismatches[:5]}"


def test_input_validation_semantics_unchanged():
    """畸形/非有限输入 → False (与旧实现一致)."""
    assert _point_in_loop(0.0, 0.0, [0.0, 1.0], [0.0, 1.0]) is False
    assert _point_in_loop(0.0, 0.0, [], []) is False
    assert _point_in_loop(0.0, 0.0, [0.0, 1.0, 2.0], [0.0, 1.0]) is False
    assert _point_in_loop(np.nan, 0.0, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) is False
    assert _point_in_loop(0.0, np.inf, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]) is False
    assert _point_in_loop(
        0.0, 0.0, [0.0, 1.0, np.nan], [0.0, 0.0, 1.0]) is False


def test_boundary_points_still_inside():
    """边上/顶点上的点必须仍判在内 (环嵌套深度的关键语义)."""
    xs, ys = _square(1.0)
    for px, py in [(0.0, 0.0), (0.5, 0.0), (1.0, 1.0), (0.0, 0.5)]:
        assert _point_in_loop(px, py, xs, ys) is True, (px, py)
