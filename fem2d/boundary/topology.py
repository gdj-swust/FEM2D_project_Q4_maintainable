"""边界拓扑 — 邻接图 → 有序闭环 → 曲率分割 → 分类标注"""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .geometry import (
    bilateral_filter,
    classify,
    curvature,
    fit_closed_ellipse,
    piecewise_smooth_breakpoints,
    segment_by_curvature,
    sharp_corner_indices,
)
from .predicates import orient2d
from .segment_utils import mesh_scale


@dataclass
class BoundaryLoop:
    """Validated boundary-loop geometry plus its nesting role."""

    node_ids: np.ndarray
    area: float
    probe: tuple
    depth: int = 0

    @property
    def is_outer(self):
        return self.depth % 2 == 0


def detect(mesh):
    """Detect validated, oriented geometric segments on every mesh loop.

    Returns ``list[dict]``:
        {"type": "line"|"arc"|"curve",
         "nodes": [node_indices],
         "coords": (n,2) ndarray,
         "label": "外边 直边 x=...",
         "info": {"radius":..., "center":..., ...}}
    """
    mesh.build_connectivity()
    adjacency = _boundary_adjacency(mesh.boundary_edges)
    if not adjacency:
        return []

    node_loops = _decompose_loops(adjacency)
    if not node_loops:
        return []

    scale = mesh_scale(mesh.nodes)
    loops = _validate_and_nest_loops(
        mesh.nodes, node_loops, scale)
    segments = [
        segment
        for loop_id, loop in enumerate(loops)
        for segment in _segment_loop(
            mesh.nodes, loop, loop_id, scale)
    ]
    segments = _merge_adjacent_lines(
        segments, mesh.nodes, scale)
    _orient_closed_segments(segments)
    segments.sort(key=lambda segment: (
        int(segment.get("info", {}).get("loop_depth", 0)),
        int(segment.get("info", {}).get("loop_id", 0)),
        -len(segment["nodes"]),
    ))
    return segments


def _boundary_adjacency(boundary_edges):
    """Build the undirected graph used to recover closed boundary loops."""
    adjacency = defaultdict(set)
    for raw_a, raw_b in boundary_edges:
        a, b = sorted((int(raw_a), int(raw_b)))
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency


def _validate_and_nest_loops(nodes, node_loops, scale):
    """Validate loop geometry and assign even/odd containment depth."""
    loops = []
    area_tolerance = (
        np.finfo(float).eps
        * max(float(scale) ** 2, np.finfo(float).tiny)
        * 128.0)
    for loop_id, node_ids in enumerate(node_loops):
        coords = nodes[node_ids]
        if has_boundary_self_intersection(coords):
            raise ValueError(
                "Self-intersecting, touching or overlapping domain boundary "
                f"detected in loop {loop_id}; repair the Gmsh geometry "
                "before applying boundary conditions or loads.")
        area = abs(_signed_loop_area(coords))
        if not np.isfinite(area) or area <= area_tolerance:
            raise ValueError(
                f"Boundary loop {loop_id} has zero or numerically "
                f"degenerate enclosed area ({area:.6g}); repair collapsed "
                "or duplicate geometry before meshing.")
        probe = _loop_probe_point(coords[:, 0], coords[:, 1])
        loops.append(BoundaryLoop(node_ids, area, probe))

    intersecting_pair = _intersecting_boundary_loop_pair([
        nodes[loop.node_ids] for loop in loops
    ])
    if intersecting_pair is not None:
        first, second = intersecting_pair
        raise ValueError(
            "Distinct domain boundary loops intersect, touch or overlap: "
            f"loops {first} and {second}. Boolean-fragment/repair the Gmsh "
            "surfaces before meshing.")

    for index, loop in enumerate(loops):
        px, py = loop.probe
        loop.depth = sum(
            _point_in_loop(
                px,
                py,
                nodes[container.node_ids, 0],
                nodes[container.node_ids, 1],
            )
            for other_index, container in enumerate(loops)
            if (
                other_index != index
                and loop.area < container.area)
        )
    _orient_loops(nodes, loops)
    return loops


def _orient_loops(nodes, loops):
    """Orient full loops before segmentation: material CCW, holes CW."""
    for loop in loops:
        is_ccw = _signed_loop_area(nodes[loop.node_ids]) > 0.0
        if loop.is_outer != is_ccw:
            loop.node_ids = loop.node_ids[::-1].copy()


def _segment_loop(nodes, loop, loop_id, scale):
    """Classify one validated loop into stable geometric segments."""
    coords = nodes[loop.node_ids]
    filtered_curvature = bilateral_filter(
        curvature(coords), sigma_s=2.0, sigma_r=0.3)
    sharp = sharp_corner_indices(
        coords, angle_threshold_deg=20.0)
    smooth_breakpoints = piecewise_smooth_breakpoints(coords)

    conic = _closed_conic_segment(
        nodes, loop, loop_id)
    if conic is not None:
        return [conic]

    breakpoints = (
        smooth_breakpoints
        if smooth_breakpoints
        else segment_by_curvature(
            filtered_curvature, coords, scale)
    )
    breakpoints = sorted(set(breakpoints) | set(sharp))
    return [
        _classify_loop_chain(
            nodes, chain, loop, loop_id, scale)
        for chain in _split_at_breakpoints(
            loop.node_ids, breakpoints)
        if len(chain) >= 2
    ]


def _closed_conic_segment(nodes, loop, loop_id):
    """Return a whole-loop conic only after strict smoothness and fit gates."""
    if len(loop.node_ids) < 8:
        return None
    ellipse, fit_info = fit_closed_ellipse(
        nodes[loop.node_ids])
    if not ellipse:
        return None
    # ``classify`` owns the sampling-aware corner rule.  Reuse it here so
    # topology detection and Physical Curve reconstruction cannot disagree
    # about a finely tessellated ellipse.
    segment_type, _, _ = classify(
        np.vstack([
            nodes[loop.node_ids],
            nodes[loop.node_ids[0]],
        ]),
        max(float(np.ptp(nodes[:, 0])),
            float(np.ptp(nodes[:, 1])), 1.0),
        loop.is_outer,
        closed=True,
    )
    if segment_type not in {"arc", "ellipse"}:
        return None

    center_x, center_y, semi_major, semi_minor, angle = ellipse
    role_label = "外边" if loop.is_outer else "内孔"
    closed_nodes = (
        list(map(int, loop.node_ids))
        + [int(loop.node_ids[0])]
    )
    common_info = {
        "center": (center_x, center_y),
        "is_outer": loop.is_outer,
        "loop_depth": int(loop.depth),
        "loop_id": int(loop_id),
        **fit_info,
    }
    # 轴比/标签判据统一到 geometry._axis_ratio (曾复制旧 1e-30 分母逻辑,
    # 微尺度椭圆误判整圆)
    from .geometry import _axis_ratio, _semi_axis_label
    _ratio, is_circle = _axis_ratio(semi_major, semi_minor)
    if is_circle:
        radius = 0.5 * (semi_major + semi_minor)
        return {
            "type": "arc",
            "nodes": closed_nodes,
            "coords": nodes[closed_nodes],
            "closed": True,
            "label": f"{role_label} 整圆 R={radius:.6g}",
            "info": {
                **common_info,
                "radius": radius,
                "angle": 2 * np.pi,
            },
        }
    return {
        "type": "ellipse",
        "nodes": closed_nodes,
        "coords": nodes[closed_nodes],
        "closed": True,
        "label": (
            f"{role_label} 椭圆 {_semi_axis_label(semi_major, semi_minor)}"
        ),
        "info": {
            **common_info,
            "semi_major": semi_major,
            "semi_minor": semi_minor,
            "angle": angle,
        },
    }


def _classify_loop_chain(nodes, chain, loop, loop_id, scale):
    """Classify one open or closed chain and attach loop metadata."""
    coords = nodes[chain]
    closed = (
        len(chain) >= 4
        and int(chain[0]) == int(chain[-1])
    )
    segment_type, label, info = classify(
        coords, scale, loop.is_outer, closed=closed)
    info = dict(info)
    info.update({
        "is_outer": loop.is_outer,
        "loop_depth": int(loop.depth),
        "loop_id": int(loop_id),
    })
    return {
        "type": segment_type,
        "nodes": chain,
        "coords": coords,
        "label": label,
        "info": info,
        "closed": closed,
    }


def _orient_closed_segments(segments):
    """Apply the Gmsh convention: material loops CCW, hole loops CW."""
    for segment in segments:
        if (
                not segment.get("closed", False)
                or len(segment["nodes"]) < 4):
            continue
        is_ccw = _signed_loop_area(segment["coords"]) > 0.0
        is_outer = bool(
            segment.get("info", {}).get("is_outer", False))
        if is_outer != is_ccw:
            segment["nodes"] = list(
                reversed(segment["nodes"]))
            segment["coords"] = segment["coords"][::-1]


# ═══════════════════════════════════════════════════════════════
# 内部工具
# ═══════════════════════════════════════════════════════════════

def _dedup_loop_vertices(xs, ys):
    """顶点元组列表: 相邻重复合并 + 闭合首尾重复去掉. 不足 3 → None."""
    vertices = []
    for x, y in zip(xs, ys):
        point = (float(x), float(y))
        if not vertices or point != vertices[-1]:
            vertices.append(point)
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices.pop()
    if len(vertices) < 3:
        return None
    return vertices


def _loop_edges(vertices):
    """环边列表 (相邻顶点对, 首尾环绕). 去重后不可能出现自环, 防御保留."""
    edges = []
    for i, p1 in enumerate(vertices):
        p2 = vertices[(i + 1) % len(vertices)]
        if p1 == p2:
            continue
        edges.append((p1, p2))
    return edges


def _on_loop_boundary(px, py, edges):
    """点是否恰在环边/顶点上 (orient2d==0 且落在包围盒内) — 判在内."""
    for (x1, y1), (x2, y2) in edges:
        side = orient2d(x1, y1, x2, y2, float(px), float(py))
        if side == 0.0 and (
                min(x1, x2) <= px <= max(x1, x2)
                and min(y1, y2) <= py <= max(y1, y2)):
            return True
    return False


def _count_proper_intersections(px, py, out_x, out_y, edges):
    """Gmsh 4×orient2d 规则: 线段与射线 proper 相交计数 (端点不计数)."""
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
    return count


def _general_position_ray(px, py, vertices, edges):
    """尺度感知射线 (显式一般位置): 8 个无理斜率扰动, 奇偶计数.

    固定 ``py + 1e-14`` 扰动对大坐标无效、对微尺度过大; 改用坐标
    ULP 尺度扰动。无理相关因子避免多个顶点与查询点对齐时反复选中
    同一斜率。全部斜率与某顶点共线 → 返回 None (调用方走半开规则).
    """
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
        return _count_proper_intersections(
            px, py, out_x, out_y, edges) % 2 == 1
    return None


def _half_open_crossing(px, py, edges):
    """半开穿越规则兜底 — 极对称/大坐标下一般位置射线可能找不到."""
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


def _point_in_loop(px, py, xs, ys):
    """Robust odd/even point-in-polygon test.

    Keeps Gmsh's 4×``orient2d`` proper-segment-intersection rule, but chooses a
    scale-aware ray in explicit general position. This replaces the fixed
    ``py + 1e-14`` perturbation, which can be either ineffective for large
    coordinates or excessive for microscopic geometry.

    Points exactly on the boundary are considered inside.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) != len(ys) or len(xs) < 3:
        return False
    if not (
            np.isfinite(px) and np.isfinite(py)
            and np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        return False

    vertices = _dedup_loop_vertices(xs, ys)
    if vertices is None:
        return False
    edges = _loop_edges(vertices)
    if _on_loop_boundary(px, py, edges):
        return True
    if len(edges) < 3:
        return False

    result = _general_position_ray(px, py, vertices, edges)
    if result is not None:
        return result
    return _half_open_crossing(px, py, edges)


def _signed_loop_area(coords):
    """Signed shoelace area; positive means counter-clockwise."""
    coords = np.asarray(coords, dtype=float)
    if len(coords) < 3:
        return 0.0
    # Translate near the origin before the shoelace sum.  This preserves the
    # small enclosed area of a model located at very large global coordinates.
    shifted = coords - coords[0]
    x = shifted[:, 0]
    y = shifted[:, 1]
    return 0.5 * float(np.sum(
        x * np.roll(y, -1) - np.roll(x, -1) * y))


def _loop_probe_point(xs, ys):
    """Choose a point strictly inside a possibly concave boundary loop."""
    coords = np.column_stack([xs, ys]).astype(float)
    area = _signed_loop_area(coords)

    # Polygon area centroid is a better first candidate than vertex mean.
    origin = coords[0]
    local = coords - origin
    x = local[:, 0]
    y = local[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    if abs(area) > np.finfo(float).tiny:
        centroid = origin + np.array([
            np.sum((x + np.roll(x, -1)) * cross),
            np.sum((y + np.roll(y, -1)) * cross),
        ]) / (6.0 * area)
        if _point_in_loop(
                centroid[0], centroid[1], coords[:, 0], coords[:, 1]):
            return float(centroid[0]), float(centroid[1])

    mean = coords.mean(axis=0)
    if _point_in_loop(mean[0], mean[1], coords[:, 0], coords[:, 1]):
        return float(mean[0]), float(mean[1])

    span = max(
        float(np.ptp(local[:, 0])),
        float(np.ptp(local[:, 1])),
        np.finfo(float).tiny)
    direction = 1.0 if area >= 0.0 else -1.0
    edge_order = np.argsort(
        -np.linalg.norm(np.roll(coords, -1, axis=0) - coords, axis=1))
    for edge_index in edge_order:
        p1 = coords[edge_index]
        p2 = coords[(edge_index + 1) % len(coords)]
        tangent = p2 - p1
        length = np.linalg.norm(tangent)
        if length <= np.finfo(float).tiny:
            continue
        inward = direction * np.array([-tangent[1], tangent[0]]) / length
        midpoint = 0.5 * (p1 + p2)
        for fraction in (1e-8, 1e-6, 1e-4, 1e-2):
            candidate = midpoint + inward * span * fraction
            if _point_in_loop(
                    candidate[0], candidate[1],
                    coords[:, 0], coords[:, 1]):
                return float(candidate[0]), float(candidate[1])
    return float(mean[0]), float(mean[1])


def _decompose_loops(adj):
    """边界邻接图 → 有序闭环列表.

    Returned loops contain each vertex once; closure is a topological fact
    verified by returning to ``start``, not a duplicated coordinate sample.
    """
    invalid_degrees = {
        int(node): len(neighbors)
        for node, neighbors in adj.items()
        if len(neighbors) != 2
    }
    if invalid_degrees:
        preview = ", ".join(
            f"{node}(degree={degree})"
            for node, degree in sorted(invalid_degrees.items())[:8])
        raise ValueError(
            "Open or non-manifold boundary topology: every node of a 2-D "
            f"domain boundary must have degree 2; found {preview}.")

    visited = set()
    loops = []

    starts_ordered = sorted(adj)

    for start in starts_ordered:
        if start in visited:
            continue
        if len(adj[start]) == 0:
            visited.add(start)
            continue

        loop = [start]
        visited.add(start)
        cur = list(adj[start])[0]
        prev = start

        while cur != start:
            if cur in visited:
                break
            loop.append(cur)
            visited.add(cur)
            nxt = None
            for nb in adj[cur]:
                if nb != prev:
                    nxt = nb
                    break
            if nxt is None:
                break
            prev, cur = cur, nxt

        if cur != start:
            raise ValueError(
                "Open or non-manifold boundary component encountered near "
                f"mesh node {int(cur)}; a 2-D displacement domain must have "
                "closed boundary loops.")
        if len(loop) >= 3:
            loops.append(np.array(loop, dtype=int))

    return loops


def _split_at_breakpoints(loop, breakpoints):
    """在断点处切分闭环 — 每条原始边恰好属于一个分段, 无重复."""
    loop = list(map(int, loop))
    n = len(loop)
    bps = sorted({int(bp) % n for bp in breakpoints})
    if len(bps) < 2:
        return [loop + [loop[0]]]

    segments = []
    for k, start in enumerate(bps):
        end = bps[(k + 1) % len(bps)]
        positions = [start]
        pos = start
        while pos != end:
            pos = (pos + 1) % n
            positions.append(pos)
        seg_nodes = [loop[pos] for pos in positions]
        if len(seg_nodes) >= 2:
            segments.append(seg_nodes)

    return segments if segments else [loop + [loop[0]]]


def _join_chains(a, b):
    """连接两条具有公共端点的有序节点链, 返回合并链或 None."""
    a, b = list(a), list(b)
    if a[-1] == b[0]:   return a + b[1:]
    if a[-1] == b[-1]:  return a + b[-2::-1]
    if a[0] == b[-1]:   return b[:-1] + a
    if a[0] == b[0]:    return b[:0:-1] + a
    return None


def _merge_adjacent_lines(segments, nodes, scale):
    """合并同轴且共享端点的直边."""
    if len(segments) <= 1:
        return segments

    merged = []
    used = set()

    for i, s in enumerate(segments):
        if i in used:
            continue
        if s["type"] != "line":
            merged.append(s)
            continue

        info_i = s["info"]
        for j in range(i + 1, len(segments)):
            if j in used:
                continue
            if segments[j]["type"] != "line":
                continue
            info_j = segments[j]["info"]
            if info_i.get("axis") != info_j.get("axis"):
                continue
            pos_diff = abs(info_i.get("pos", 0.0) - info_j.get("pos", 0.0))
            if pos_diff >= scale * 0.02:
                continue
            joined = _join_chains(s["nodes"], segments[j]["nodes"])
            if joined is not None:
                s["nodes"] = joined
                s["coords"] = nodes[joined]
                used.add(j)

        merged.append(s)

    return merged


# ═══════════════════════════════════════════════════════════════
# 环定向 + 自交检测 — Gmsh findLinks.cpp orientAndSortEdges 移植
# ═══════════════════════════════════════════════════════════════

def _point_on_segment(point, start, end, tolerance):
    if orient2d(
            start[0], start[1], end[0], end[1],
            point[0], point[1]) != 0.0:
        return False
    return (
        min(start[0], end[0]) - tolerance
        <= point[0]
        <= max(start[0], end[0]) + tolerance
        and min(start[1], end[1]) - tolerance
        <= point[1]
        <= max(start[1], end[1]) + tolerance
    )


def _segments_intersect_or_overlap(a, b, c, d, tolerance):
    """Return True for proper crossings, endpoint touches or overlap."""
    ab_c = orient2d(a[0], a[1], b[0], b[1], c[0], c[1])
    ab_d = orient2d(a[0], a[1], b[0], b[1], d[0], d[1])
    cd_a = orient2d(c[0], c[1], d[0], d[1], a[0], a[1])
    cd_b = orient2d(c[0], c[1], d[0], d[1], b[0], b[1])
    if ab_c * ab_d < 0.0 and cd_a * cd_b < 0.0:
        return True
    return (
        (ab_c == 0.0 and _point_on_segment(c, a, b, tolerance))
        or (ab_d == 0.0 and _point_on_segment(d, a, b, tolerance))
        or (cd_a == 0.0 and _point_on_segment(a, c, d, tolerance))
        or (cd_b == 0.0 and _point_on_segment(b, c, d, tolerance))
    )




def _sweep_intersections(records, tolerance, skip_rule, first_only=False):
    """Shared sweep-line: collect intersecting record pairs.

    ``records`` items: (min_x, max_x, min_y, max_y, group, start, end),
    where ``group`` is the owning loop (multi-loop) or the edge index
    (single-loop).  ``skip_rule(a, b)`` excludes candidate pairs (adjacent
    edges in one loop, or same-loop edges).  ``first_only`` keeps the
    original early-exit behaviour of both callers (cost = first hit).
    """
    hits = []
    active = []
    for current in sorted(records, key=lambda item: (item[0], item[2])):
        min_x, _, min_y, max_y, group, start, end = current
        active = [
            previous for previous in active
            if previous[1] + tolerance >= min_x
        ]
        for previous in active:
            _, _, previous_min_y, previous_max_y, other, a, b = previous
            if skip_rule(group, other):
                continue
            if (
                    max_y + tolerance < previous_min_y
                    or previous_max_y + tolerance < min_y):
                continue
            if _segments_intersect_or_overlap(
                    a, b, start, end, tolerance):
                hits.append((other, group))
                if first_only:
                    return hits
        active.append(current)
    return hits


def has_boundary_self_intersection(loop_nodes):
    """Detect crossings, non-adjacent touches and collinear overlaps.

    Input may repeat its first point or list every vertex once.  Sorting edge
    x-bounds keeps dense Gmsh loops near linear in normal cases, while robust
    ``orient2d`` predicates decide each candidate pair.
    """
    vertices = [
        (float(point[0]), float(point[1]))
        for point in loop_nodes
    ]
    if len(vertices) > 1 and vertices[0] == vertices[-1]:
        vertices.pop()
    n = len(vertices)
    if n < 3:
        return False

    coords = np.asarray(vertices, dtype=float)
    if not np.all(np.isfinite(coords)):
        return True
    # 容差基于局部坐标尺度, 不得强制 1.0 下限 — 曾让尺度 ≤1e-14 的合法
    # 模型每条边都被判零长, 误报 "Self-intersecting" 拒绝
    magnitude = max(float(np.max(np.abs(coords))), np.finfo(float).tiny)
    span = max(
        float(np.ptp(coords[:, 0])),
        float(np.ptp(coords[:, 1])),
        np.finfo(float).tiny,
    )
    tolerance = max(
        np.spacing(magnitude) * 32.0,
        np.finfo(float).eps * span * 32.0,
    )

    records = []
    for index in range(n):
        start = vertices[index]
        end = vertices[(index + 1) % n]
        # 闭合链接缝边 (P_{n-1}→P_0) 是环的闭合, 不是零长边特征 — 三角
        # 采样等数值噪声会在此产生 ULP 量级的微小闭合边, 曾使微尺度
        # 模型每条边界都被判自交拒绝
        if index != n - 1 and (
                abs(start[0] - end[0]) <= tolerance
                and abs(start[1] - end[1]) <= tolerance):
            return True
        records.append((
            min(start[0], end[0]), max(start[0], end[0]),
            min(start[1], end[1]), max(start[1], end[1]),
            index, start, end,
        ))

    return bool(_sweep_intersections(
        records, tolerance,
        lambda i, j: abs(i - j) == 1 or {i, j} == {0, n - 1},
        first_only=True))


def _intersecting_boundary_loop_pair(loop_coordinates):
    """Return two loop indices whose edges touch/cross, or ``None``.

    All loops share one sweep so a dense outer boundary with many holes does
    not trigger a separate full Cartesian comparison for every hole.
    """
    arrays = [np.asarray(coords, dtype=float) for coords in loop_coordinates]
    if len(arrays) < 2:
        return None
    finite_arrays = [
        coords for coords in arrays if len(coords)
    ]
    if not finite_arrays:
        return None
    combined = np.vstack(finite_arrays)
    # 同 has_boundary_self_intersection: 1.0 下限破坏微尺度尺度不变性
    #
    magnitude = max(float(np.max(np.abs(combined))), np.finfo(float).tiny)
    span = max(
        float(np.ptp(combined[:, 0])),
        float(np.ptp(combined[:, 1])),
        np.finfo(float).tiny,
    )
    tolerance = max(
        np.spacing(magnitude) * 32.0,
        np.finfo(float).eps * span * 32.0,
    )

    records = []
    for owner, coords in enumerate(arrays):
        vertices = [
            (float(point[0]), float(point[1])) for point in coords
        ]
        if len(vertices) > 1 and vertices[0] == vertices[-1]:
            vertices.pop()
        for index, start in enumerate(vertices):
            end = vertices[(index + 1) % len(vertices)]
            records.append((
                min(start[0], end[0]), max(start[0], end[0]),
                min(start[1], end[1]), max(start[1], end[1]),
                owner, start, end,
            ))

    hits = _sweep_intersections(
        records, tolerance, lambda i, j: i == j, first_only=True)
    if not hits:
        return None
    return tuple(sorted((int(hits[0][0]), int(hits[0][1]))))
