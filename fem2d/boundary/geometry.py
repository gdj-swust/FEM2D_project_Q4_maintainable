"""边界几何分析 — 曲率计算、双边滤波、基元分类、RANSAC/椭圆拟合

参考: Botsch et al., Polygon Mesh Processing, Ch.3-4
      Shewchuk (1997) — orient2d 自适应精度谓词
      Gmsh — geom.tolerance × lc 统一容差 (GEdge.cpp:115, GModel.cpp:439)
"""
import math

import numpy as np

from .predicates import orient2d

_CONIC_MEAN_RESIDUAL_LIMIT = 0.03
_CONIC_MAX_RESIDUAL_LIMIT = 0.06
_REFINED_CONIC_MEAN_LIMIT = 0.05
_REFINED_CONIC_MAX_LIMIT = 0.08
_DEFAULT_CONIC_TURN_LIMIT_DEG = 20.0
_FINE_CONIC_TURN_LIMIT_DEG = 40.0
_FINE_CONIC_MIN_PRIMITIVES = 16


# ═══════════════════════════════════════════════════════════════
# 统一容差 — Gmsh GEdge.cpp:115 tol = geom.tolerance × lc
# ═══════════════════════════════════════════════════════════════

# 坐标尺度 ULP — 唯一实现在 preprocess (与 geometry 各复制一份
# 是冗余, 已合并)。微尺度模型的零长/退化判据: 绝对 1e-15
# 下限会让微尺度模型的每条边都被判零长, 角点全被跳过。
from ..preprocess import _coordinate_ulp as _coords_ulp


def compute_tolerance(coords, geom_tol=1e-6):
    """Gmsh 式统一容差: tol = geom_tolerance × lc.

    lc = 特征长度 = 平均边长 (Gmsh 中为用户指定的网格密度).
    无用户指定时用中值边长估算.
    """
    if len(coords) < 2:
        return 1e-12
    diffs = np.diff(coords, axis=0)
    edge_lens = np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)
    lc = np.median(edge_lens) if len(edge_lens) > 0 else 1.0
    # eps*10 绝对下限会大于整个微尺度模型 (1e-16) — 只防 lc 为 0
    return max(geom_tol * lc, np.finfo(float).tiny)


# ═══════════════════════════════════════════════════════════════
# 离散曲率 (Polygon Mesh Processing §3.2)
# ═══════════════════════════════════════════════════════════════

def curvature(coords, closed=True):
    """离散曲率 κ_i = 转向角θ_i / 平均边长ℓ_i

    闭曲线: 对每个节点i, v₁ = P_{i-1}→P_i, v₂ = P_i→P_{i+1}
    θ_i = atan2(|v₁×v₂|, v₁·v₂)   (有符号转向角)
    ℓ_i = (|v₁| + |v₂|) / 2

    Open chains leave both endpoint curvatures at zero instead of wrapping
    the last tangent back to the first.  This matters for arbitrary Physical
    Curves, where the artificial closing edge can dominate all statistics.
    """
    n = len(coords)
    kappa = np.zeros(n)

    # 零长边判据基于坐标尺度, 不用绝对 1e-15 — 微尺度模型 (边长 1e-16)
    # 否则全部 κ=0, 曲率分段静默失效
    zero_len = min(
        1e-15,
        64.0 * np.finfo(float).eps
        * max(float(np.max(np.abs(coords))), np.finfo(float).tiny),
    )

    # 闭合重复点不是几何特征: 首尾精确重合或闭合边落在浮点噪声内
    # (三角采样圆的 cos(2π)≠1.0, 闭合边 ~1e-16)。重复点处差分退化
    # κ=0, 且 i=0 的 prev 引用到它 — segment_by_curvature 在接缝误报
    # 伪断点
    has_duplicate = bool(
        closed and n > 2
        and (np.array_equal(coords[0], coords[-1])
             or float(np.linalg.norm(coords[-1] - coords[0])) <= zero_len))

    indices = range(n) if closed else range(1, n - 1)
    for i in indices:
        prev = (i - 1) % n
        nxt = (i + 1) % n
        if has_duplicate:
            if i == 0:
                prev = n - 2      # 跳过闭合重复点, 用真实前驱
            if i == n - 1:
                continue          # 重复点与 P_0 同位置, κ 镜像到 kappa[0]
        v1 = coords[i] - coords[prev]
        v2 = coords[nxt] - coords[i]
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 <= zero_len or l2 <= zero_len:
            continue
        # Shewchuk orient2d: P_prev → P_i → P_next 的定向
        # >0 = CCW 左转, <0 = CW 右转, =0 = 共线
        cross_robust = orient2d(
            coords[prev, 0], coords[prev, 1],
            coords[i, 0], coords[i, 1],
            coords[nxt, 0], coords[nxt, 1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        # 用 orient2d 的符号 + arctan2 的量级 (dot 非零时反正切可靠)
        theta = math.atan2(float(cross_robust), dot)
        ell = (l1 + l2) / 2.0
        kappa[i] = theta / ell

    if has_duplicate:
        kappa[n - 1] = kappa[0]   # 重复点与 P_0 同位置, κ 必须一致

    return kappa


# ═══════════════════════════════════════════════════════════════
# 双边滤波 (Polygon Mesh Processing §4.3)
# ═══════════════════════════════════════════════════════════════

def bilateral_filter(signal, sigma_s=2.0, sigma_r=0.3):
    """保边平滑: w_ij = exp(-d_ij²/2σ_s²) · exp(-Δκ²/2σ_r²)

    σ_s: 空间带宽 (控制邻域范围)
    σ_r: 取值范围带宽 (控制多少曲率差异才算"边")
    """
    n = len(signal)
    result = signal.copy()
    half_w = max(1, int(sigma_s * 2))

    for i in range(n):
        total_w = 0.0
        total_v = 0.0
        for di in range(-half_w, half_w + 1):
            j = (i + di) % n
            w_s = np.exp(-di * di / (2.0 * sigma_s * sigma_s))
            diff = signal[i] - signal[j]
            w_r = np.exp(-diff * diff / (2.0 * sigma_r * sigma_r))
            w = w_s * w_r
            total_w += w
            total_v += w * signal[j]
        if total_w > 1e-15:
            result[i] = total_v / total_w

    return result


# ═══════════════════════════════════════════════════════════════
# 曲率驱动的分割 (Polygon Mesh Processing §7.2)
# ═══════════════════════════════════════════════════════════════

def sharp_corner_indices(coords, angle_threshold_deg=35.0):
    """检测明显角点 (相邻边夹角 ≥ angle_threshold_deg).

    用 Shewchuk orient2d 确定转向符号, 避免共线/退化时的浮点误判.
    """
    corners = []
    n = len(coords)
    ulp = _coords_ulp(coords)
    # 闭合重复点 (首尾精确重合或落在 ULP 内): 与 curvature() 同款处理 —
    # 会使闭合链的首个角点 (prev 指向重复点 → 零长边) 被静默跳过
    #
    has_duplicate = bool(
        n > 2
        and (np.array_equal(coords[0], coords[-1])
             or float(np.linalg.norm(coords[-1] - coords[0])) <= ulp))
    for i in range(n):
        if has_duplicate and i == 0:
            prev = n - 2          # 跳过闭合重复点, 用真实前驱
        elif has_duplicate and i == n - 1:
            continue              # 重复点与 P_0 同位置, 角点镜像到 [0]
        else:
            prev = (i - 1) % n
        nxt = (i + 1) % n
        v1 = coords[i] - coords[prev]
        v2 = coords[nxt] - coords[i]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        # 绝对 1e-15 会让微尺度多边形的所有角点被跳过
        if l1 <= ulp or l2 <= ulp:
            continue
        # orient2d: prev → i → nxt 的定向量 (平行四边形的有向面积)
        cross_robust = orient2d(
            coords[prev, 0], coords[prev, 1],
            coords[i, 0], coords[i, 1],
            coords[nxt, 0], coords[nxt, 1])
        dot = np.dot(v1, v2)
        turn = abs(math.degrees(math.atan2(float(cross_robust), dot)))
        if turn >= angle_threshold_deg:
            corners.append(i)
    return corners


def turning_angles(coords):
    """Return signed turning angles for a closed coordinate loop.

    The magnitude is independent of edge length, which makes this signal more
    reliable than curvature when a Gmsh boundary mixes long straight edges
    with finely sampled arcs.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    result = np.zeros(n, dtype=float)
    if n < 3:
        return result

    for i in range(n):
        prev = (i - 1) % n
        nxt = (i + 1) % n
        v1 = coords[i] - coords[prev]
        v2 = coords[nxt] - coords[i]
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 <= np.finfo(float).tiny or l2 <= np.finfo(float).tiny:
            continue
        cross = orient2d(
            coords[prev, 0], coords[prev, 1],
            coords[i, 0], coords[i, 1],
            coords[nxt, 0], coords[nxt, 1])
        result[i] = math.atan2(float(cross), float(np.dot(v1, v2)))
    return result


def piecewise_smooth_breakpoints(coords):
    """Find tangent line/curve transitions on a closed boundary.

    A rounded rectangle has no sharp corner: a whole-loop ellipse fit can look
    numerically acceptable even though long runs of vertices are exactly
    collinear.  Detect those cyclic straight runs from turning angles and put
    breakpoints at their two tangent endpoints.

    Isolated low-curvature samples are deliberately ignored so that a genuine
    ellipse with non-uniform sampling is not split near its low-curvature
    extrema.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n < 6:
        return []

    turns = np.abs(turning_angles(coords))
    significant = turns[turns > 1e-10]
    if len(significant) < 2:
        return []

    reference_turn = float(np.median(significant))
    straight_limit = max(math.radians(0.1), 0.12 * reference_turn)
    straight = turns <= straight_limit
    if not np.any(straight) or np.all(straight):
        return []

    # Three exactly/near-collinear interior samples are enough to establish a
    # real straight run (including a four-way chord subdivision), while two
    # low-curvature samples commonly occur near a smooth spline inflection.
    minimum_run = 3
    starts = [
        i for i in range(n)
        if straight[i] and not straight[(i - 1) % n]
    ]
    breakpoints = set()
    for start in starts:
        end = start
        length = 1
        while length < n and straight[(end + 1) % n]:
            end = (end + 1) % n
            length += 1
        if length < minimum_run:
            continue
        # The tangent endpoints are the curved vertices immediately before
        # and after the collinear interior run.
        breakpoints.add((start - 1) % n)
        breakpoints.add((end + 1) % n)

    return sorted(breakpoints)


def segment_by_curvature(kappa, coords=None, scale=None):
    """Find genuine G1 curvature jumps without splitting smooth splines.

    Curvature extrema and inflection points are valid parts of one arbitrary
    smooth curve, so they are deliberately *not* breakpoints.  A breakpoint
    is emitted only when the median curvature on two arc-length neighborhoods
    changes far more than the variation inside either neighborhood.

    ``coords`` 兼容旧签名 (kappa, coords, scale), 实现不依赖坐标.
    """
    # 同时兼容两参数旧形式 (kappa, scale): 位置调用会把 scale 传进 coords
    if scale is None:
        scale = coords
        coords = None
    del coords  # 兼容参数 — 实现只依赖曲率序列
    n = len(kappa)
    if n < 10:
        return []

    kappa = np.asarray(kappa, dtype=float)
    finite = np.isfinite(kappa)
    if not finite.all():
        kappa = np.where(finite, kappa, 0.0)
    abs_k = np.abs(kappa)
    characteristic = max(float(scale), np.finfo(float).tiny)
    curvature_floor = 1e-8 / characteristic
    if abs_k.max() <= curvature_floor:
        return []

    global_level = max(float(np.median(abs_k)), curvature_floor)
    global_mad = float(np.median(np.abs(abs_k - np.median(abs_k))))
    window = max(2, min(8, n // 16))
    candidates = []
    for i in range(n):
        left = np.array([
            kappa[(i - offset) % n] for offset in range(window - 1, -1, -1)
        ])
        right = np.array([
            kappa[(i + 1 + offset) % n] for offset in range(window)
        ])
        left_level = float(np.median(left))
        right_level = float(np.median(right))
        jump = abs(right_level - left_level)
        left_mad = float(np.median(np.abs(left - left_level)))
        right_mad = float(np.median(np.abs(right - right_level)))
        local_variation = left_mad + right_mad
        relative_jump = jump / max(
            abs(left_level), abs(right_level), global_level)
        separation = jump / max(
            local_variation + 0.1 * global_mad, curvature_floor)
        if relative_jump > 0.55 and separation > 6.0:
            candidates.append((i, separation))

    if not candidates:
        return []

    # Keep only the strongest location in each short cyclic cluster.
    candidates.sort()
    clusters = []
    for index, score in candidates:
        if clusters and index - clusters[-1][-1][0] <= window:
            clusters[-1].append((index, score))
        else:
            clusters.append([(index, score)])
    if (
            len(clusters) > 1
            and clusters[0][0][0] + n - clusters[-1][-1][0] <= window):
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()

    result = []
    for cluster in clusters:
        index, _ = max(cluster, key=lambda item: item[1])
        result.append(index % n)
    return sorted(set(result))


# ═══════════════════════════════════════════════════════════════
# 段分类 — 显式识别器注册表 (detectors.py), 本模块保留 facade
# ═══════════════════════════════════════════════════════════════

def classify(coords, scale, is_outer, closed=None, *, native_entities=()):
    """Classify one boundary chain as line, arc, ellipse or curve.

    ``closed`` is topology truth when supplied.  Standalone callers use a
    tight, scale-aware coordinate comparison so a nearly complete open arc is
    never promoted to a full circle accidentally.

    实现 = detectors.DetectorRegistry 有序判定 (line → circle → ellipse
    → general, 与旧探测顺序逐位一致); 原生实体信息经 native_entities
    传入不丢失 (内置探测器不消费, 插件接口用).
    """
    from .detectors import default_registry  # 局部导入破环: detectors 依赖本模块拟合原语
    return default_registry().classify(
        coords, scale, is_outer, closed=closed,
        native_entities=native_entities)


def _segment_is_closed(coords, tolerance, closed):
    if closed is not None:
        return bool(closed)
    # 1.0 物理尺度下限会把微尺度圆弧 (跨度 1e-16) 的闭合容差抬到
    # ~7e-15 > 整个模型, 开放圆弧被误判为闭合曲线。坐标 ULP 尺度 —
    # 与 _coords_ulp 约定一致。
    coordinate_scale = max(
        float(np.ptp(coords[:, 0])),
        float(np.ptp(coords[:, 1])),
        np.finfo(float).tiny,
    )
    closure_tolerance = max(
        tolerance * 10.0,
        np.finfo(float).eps * coordinate_scale * 32.0,
    )
    return bool(
        np.linalg.norm(coords[0] - coords[-1])
        <= closure_tolerance
    )


def _axis_ratio(semi_major, semi_minor):
    """轴比判据 — 唯一实现 (曾 geometry/topology 各复制一份, 拓扑层
    保留 1e-30 绝对分母使微尺度椭圆误判整圆)。
    纯相对: 分母仅防除零 (微尺度轴仍精确)。返回 (ratio, is_circle)。
    """
    semi_minor = max(float(semi_minor), np.finfo(float).tiny)
    semi_major = max(float(semi_major), np.finfo(float).tiny)
    return semi_major / semi_minor, (
        abs(semi_major - semi_minor) / semi_major < 0.05)


def _semi_axis_label(semi_major, semi_minor):
    """科学计数标签 — 曾 .3f 使微尺度椭圆显示 a=0.000 (第三轮审查)."""
    return f"a={semi_major:.6g} b={semi_minor:.6g}"


def _closed_conic_turn_limit_deg(ellipse, fit_info):
    """Return a conservative, sampling-aware conic smoothness limit.

    A sparse polygon must retain its visible corners.  A finely tessellated,
    low-residual ellipse can legitimately turn by more than 20 degrees near
    the high-curvature ends (the 20-piece 2:1 demo ellipse reaches 35.15
    degrees).  The relaxed limit therefore applies only after at least 16
    fitted primitive samples and a strict whole-loop residual check.
    """
    semi_major = float(ellipse[2])
    semi_minor = float(ellipse[3])
    visibly_non_circular = (
        semi_major / max(semi_minor, np.finfo(float).tiny) > 1.05)
    if (
            visibly_non_circular
            and
            int(fit_info.get("primitive_samples", 0))
            >= _FINE_CONIC_MIN_PRIMITIVES
            and float(fit_info.get("fit_residual", np.inf))
            < _CONIC_MEAN_RESIDUAL_LIMIT
            and float(fit_info.get("fit_residual_max", np.inf))
            < _CONIC_MAX_RESIDUAL_LIMIT):
        return _FINE_CONIC_TURN_LIMIT_DEG
    return _DEFAULT_CONIC_TURN_LIMIT_DEG


def fit_circle_least_squares(points):
    """Stable algebraic circle fit in centered, normalized coordinates."""
    points = np.asarray(points, dtype=float)
    if len(points) < 3 or not np.all(np.isfinite(points)):
        return 0.0, 0.0, -1.0
    origin = np.mean(points, axis=0)
    shifted = points - origin
    coordinate_scale = float(np.sqrt(np.mean(np.sum(shifted ** 2, axis=1))))
    if coordinate_scale <= np.finfo(float).tiny:
        return 0.0, 0.0, -1.0
    normalized = shifted / coordinate_scale
    x = normalized[:, 0]
    y = normalized[:, 1]
    matrix = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(points))])
    rhs = x * x + y * y
    try:
        cx_n, cy_n, constant = np.linalg.lstsq(
            matrix, rhs, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0, 0.0, -1.0
    radius_sq = constant + cx_n * cx_n + cy_n * cy_n
    if radius_sq <= 0.0:
        return 0.0, 0.0, -1.0
    center = origin + coordinate_scale * np.array([cx_n, cy_n])
    radius = coordinate_scale * np.sqrt(radius_sq)
    return float(center[0]), float(center[1]), float(radius)


def circle_fit_residual(points, circle):
    """Return mean/max absolute radial residual for a circle candidate."""
    points = np.asarray(points, dtype=float)
    cx, cy, radius = circle
    if radius <= 0.0 or not np.all(np.isfinite(points)):
        return np.inf, np.inf
    residual = np.abs(
        np.linalg.norm(points - np.array([cx, cy]), axis=1) - radius)
    return float(np.mean(residual)), float(np.max(residual))


def fit_ellipse(points):
    """椭圆拟合: SVD + 椭圆约束强制 → (cx, cy, a, b, theta) 或 None"""
    x, y = points[:, 0], points[:, 1]
    n = len(x)
    if n < 6:
        return None

    xm, ym = x.mean(), y.mean()
    xs, ys = x - xm, y - ym
    scl = np.sqrt(xs.std()**2 + ys.std()**2)
    # 绝对 1e-15 会让微尺度椭圆 (散布 1e-16) 拟合静默返回 None → 整段
    # 落入 curve 分类
    if scl <= _coords_ulp(points):
        return None
    xs, ys = xs / scl, ys / scl

    Dmat = np.column_stack([xs * xs, xs * ys, ys * ys, xs, ys])
    rhs = np.ones(n)
    try:
        coeffs, _, _, _ = np.linalg.lstsq(Dmat, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return None
    A, B, C_, D_, E_ = coeffs
    F = -1.0

    disc = 4 * A * C_ - B * B
    if disc <= 0:
        target = B * B * 1.1 + 1e-6
        factor = target / (4 * A * C_) if A * C_ > 0 else 2.0
        A *= factor
        C_ *= factor

    # Solve the conic center and axes entirely in normalized coordinates.
    # Expanding the coefficients around a large physical origin introduces
    # catastrophic cancellation in F(center), even when the input points are
    # a mathematically exact circle.
    D, E = D_, E_
    disc = 4 * A * C_ - B * B
    if disc <= 1e-15:
        return None

    cx_n = (B * E - 2 * C_ * D) / disc
    cy_n = (B * D - 2 * A * E) / disc
    theta = 0.5 * np.arctan2(B, A - C_) if abs(A - C_) > 1e-15 else np.pi / 4

    ct, st = np.cos(theta), np.sin(theta)
    Ap = A * ct * ct + B * ct * st + C_ * st * st
    Cp = A * st * st - B * ct * st + C_ * ct * ct
    Fp = (
        A * cx_n * cx_n + B * cx_n * cy_n + C_ * cy_n * cy_n
        + D * cx_n + E * cy_n + F
    )

    if Ap * Fp >= 0 or Cp * Fp >= 0:
        return None
    a_e = scl * np.sqrt(-Fp / Ap)
    b_e = scl * np.sqrt(-Fp / Cp)
    cx = xm + scl * cx_n
    cy = ym + scl * cy_n

    if a_e < b_e:
        a_e, b_e = b_e, a_e
        theta += np.pi / 2

    if a_e > scl * 10 or b_e < scl * 0.001:
        return None
    if max(a_e, b_e) / (min(a_e, b_e) + np.finfo(float).tiny) > 20:
        return None

    residual, _ = ellipse_fit_residual(
        points, (cx, cy, a_e, b_e, theta))
    if residual > 0.08:
        return None

    return cx, cy, a_e, b_e, theta


def ellipse_fit_residual(points, ellipse):
    """Return mean/max dimensionless radial residual for an ellipse fit."""
    points = np.asarray(points, dtype=float)
    cx, cy, a_e, b_e, theta = ellipse
    if (
            len(points) == 0 or a_e <= 0.0 or b_e <= 0.0
            or not np.all(np.isfinite(points))):
        return np.inf, np.inf
    ct_r, st_r = np.cos(theta), np.sin(theta)
    x = points[:, 0]
    y = points[:, 1]
    xr = (x - cx) * ct_r + (y - cy) * st_r
    yr = -(x - cx) * st_r + (y - cy) * ct_r
    radial_error = np.abs(
        np.sqrt((xr / a_e) ** 2 + (yr / b_e) ** 2) - 1.0)
    return float(np.mean(radial_error)), float(np.max(radial_error))


def fit_closed_ellipse(points):
    """Fit a closed conic without bias from collinear chord subdivisions.

    Both automatic topology detection and the Physical Curve reconstruction
    path call this function.  Long collinear runs are treated as polygon
    chords: the conic is refitted on their junctions and then validated
    against every boundary node.  If those runs instead belong to a rounded
    rectangle, the global residual gate rejects the whole-loop conic so that
    the caller can preserve its line/arc segmentation.

    Returns ``(ellipse, info)`` where ``ellipse`` is ``None`` when the loop
    should not be represented by one conic.
    """
    coords = np.asarray(points, dtype=float)
    if len(coords) < 8 or not np.all(np.isfinite(coords)):
        return None, {}

    # 1.0 物理尺度下限会让微尺度 (跨度 ≲2e-13) 整环首末顶点间距
    # 恒 ≤ eps×1.0×32 ≈ 7.1e-15, 被误判"重复闭合点"→ 静默截掉末顶点,
    # 被截顶点的拟合残差不再被验证。与 _segment_is_closed 同款 ULP
    # 相对化 — 坐标尺度多大, 容差就多大。
    span = max(float(np.ptp(coords[:, 0])),
               float(np.ptp(coords[:, 1])), np.finfo(float).tiny)
    closure_tol = max(
        compute_tolerance(coords) * 10.0,
        np.finfo(float).eps * span * 32.0)
    if np.linalg.norm(coords[0] - coords[-1]) <= closure_tol:
        coords = coords[:-1]
    if len(coords) < 8:
        return None, {}

    breakpoints = piecewise_smooth_breakpoints(coords)
    ellipse = fit_ellipse(coords)
    if ellipse is None:
        return None, {}
    mean_residual, max_residual = ellipse_fit_residual(coords, ellipse)
    primitive_samples = len(coords)

    if breakpoints:
        high_quality = (
            mean_residual < _CONIC_MEAN_RESIDUAL_LIMIT
            and max_residual < _CONIC_MAX_RESIDUAL_LIMIT)
        if not high_quality:
            return None, {}

        if len(breakpoints) >= 6:
            refined = fit_ellipse(coords[breakpoints])
            if refined is not None:
                refined_mean, refined_max = ellipse_fit_residual(
                    coords, refined)
                if (
                        refined_mean < _REFINED_CONIC_MEAN_LIMIT
                        and refined_max < _REFINED_CONIC_MAX_LIMIT):
                    ellipse = refined
                    mean_residual = refined_mean
                    max_residual = refined_max
                    primitive_samples = len(breakpoints)

    info = {
        "coarse_fit": primitive_samples < 12,
        "primitive_samples": primitive_samples,
        "fit_residual": mean_residual,
        "fit_residual_max": max_residual,
    }
    return ellipse, info


def _unwrap_angle_range(angles):
    """计算角度范围 (处理环绕).

    圆环顺序上最大间隔 = 缺失段, 弧跨度 = 2π − 最大间隔 (含首尾环绕
    间隔)。旧实现只在间隔 > π 时处理环绕 — 跨 ±π 切线的优弧
    (200° 弧 160°→360°) 返回 360°−表示间隙 (=340°) 而非真实跨度
   。
    """
    sorted_angles = np.sort(angles)
    diffs = np.diff(sorted_angles)
    wrap_gap = sorted_angles[0] + 2.0 * np.pi - sorted_angles[-1]
    max_gap = max(float(np.max(diffs)) if diffs.size else 0.0,
                  float(wrap_gap))
    return 2.0 * np.pi - max_gap
