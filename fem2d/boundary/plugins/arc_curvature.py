"""正式插件 3: arc_curvature 曲率分段展示层探测器.

职责 (docs/boundary_plugins.md 优先级规则): **开链/短弧/圆弧/样条**
→ 曲率展示层标签; 闭合整椭圆链属插件 1 (ellipse_group_label),
闭合整圆/直边让位内置探测器.

判定 (禁止拟合 — 代数解):
  圆弧段 (κ 恒定): 曲率 CV < 0.15 → 圆心 = 最远点对 + 弧顶点的
  垂平分线交点 (2×2 线性代数), ρ = 圆心到任一点距离 — 无最小二乘;
  标签 "圆弧 ρ=.., 圆心(..,..)" (token 级可断言).
  开链/样条 (κ 变化): 委托内置 LineDetector (直边让位) 与
  GeneralCurveDetector (保守曲线标签).

短弧保护 (禁止硬拟合): 1/8 椭圆等短弧 (弧长覆盖 <60%) 的椭圆拟合
即使残差完美也不能标 "椭圆" — a/b 从短弧无法可靠确定. 插件在
注册表前端拦截: 椭圆弧长覆盖不足 → 保守曲线标签 (宁缺毋滥).
覆盖足够的开放椭圆弧让位内置 EllipseDetector (维持旧行为).

实现模式 = 委托上游探测器: LineDetector/GeneralCurveDetector 判定
门槛变更时插件自动跟随.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ..detectors import (
    GeneralCurveDetector,
    LineDetector,
    Detection,
    Detector,
)
from ..geometry import (
    _unwrap_angle_range,
    curvature,
    fit_ellipse,
)

# κ 恒定性门 (与 GeneralCurveDetector 类圆分支同阈值 — 插件只把
# "类圆"级恒定链升级为精确代数圆弧, 语义一致)
ARC_CURVATURE_CV_LIMIT = 0.15
# 短弧保护: 椭圆弧长覆盖 <60% → 不得标椭圆 (a/b 不可靠)
MIN_ELLIPSE_ARC_COVERAGE = 0.6
# 最小弧跨度 (与内置开放圆弧 5° 下限一致) — 过短 → 曲线类
MIN_ARC_SPAN_RADIANS = np.deg2rad(5.0)
# 代数残差门: 径向残差 / ρ < 2% (相对尺度, 微尺度模型不受绝对阈值伤)
MAX_RELATIVE_ALGEBRAIC_RESIDUAL = 0.02
# 弓高门: 弧顶距弦 < 弦长×1e-4 → 近平直, 垂平分线交点病态
MIN_SAGITTA_RATIO = 1e-4


def _ellipse_circumference(semi_major: float, semi_minor: float) -> float:
    """Ramanujan 椭圆周长近似 — 覆盖判据只用相对比值."""
    a = float(semi_major)
    b = float(semi_minor)
    return math.pi * (3.0 * (a + b)
                      - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))


class ArcCurvatureDetector(Detector):
    """曲率分段展示层探测器 — 开链/短弧/圆弧/样条 (见模块 docstring)."""

    name = "arc_curvature"

    def __init__(self) -> None:
        self._line = LineDetector()
        self._general = GeneralCurveDetector()

    def detect(
            self,
            points: np.ndarray,
            *,
            scale: float,
            is_outer: bool,
            closed: bool,
            native_entities: Sequence[str] = (),
    ) -> Optional[Detection]:
        if closed:
            # 闭合整椭圆链 → 插件 1; 闭合整圆 → 内置圆探测器
            return None
        coords = np.asarray(points, dtype=float)
        if len(coords) < 4:
            return None
        # 直边让位 LineDetector (标签/参数与其逐位一致)
        if self._line.detect(
                coords, scale=scale, is_outer=is_outer,
                closed=False) is not None:
            return None
        arc = self._arc_algebraic(coords, is_outer)
        if arc is not None:
            return arc
        return self._short_ellipse_guard(coords, scale, is_outer)

    def _arc_algebraic(
            self, coords: np.ndarray,
            is_outer: bool) -> Optional[Detection]:
        """κ 恒定弧的代数解: ρ/圆心 = 三点的垂平分线交点 (禁止拟合)."""
        interior = curvature(coords, closed=False)[1:-1]
        if len(interior) < 2:
            return None
        absolute = np.abs(interior)
        mean_kappa = float(np.mean(absolute))
        coordinate_scale = max(
            float(np.ptp(coords[:, 0])),
            float(np.ptp(coords[:, 1])),
            np.finfo(float).tiny,
        )
        if mean_kappa <= 1e-8 / coordinate_scale:
            return None  # 近平直 (κ 地板与 segment_by_curvature 同款)
        variation = float(
            np.std(absolute)
            / (mean_kappa + np.finfo(float).tiny))
        if variation >= ARC_CURVATURE_CV_LIMIT:
            return None  # κ 不恒定 → 样条/椭圆弧等, 交短弧保护

        center = self._algebraic_center(coords)
        if center is None:
            return None
        radius = float(np.linalg.norm(coords[0] - center))
        if not (np.isfinite(radius) and radius > 0.0):
            return None
        residual = np.abs(
            np.linalg.norm(coords - center, axis=1) - radius)
        mean_residual = float(np.mean(residual))
        max_residual = float(np.max(residual))
        if mean_residual > radius * MAX_RELATIVE_ALGEBRAIC_RESIDUAL:
            return None  # 代数残差超相对门 (2%) → 非精确圆弧

        angles = np.arctan2(
            coords[:, 1] - center[1],
            coords[:, 0] - center[0],
        )
        span = _unwrap_angle_range(angles)
        if span < MIN_ARC_SPAN_RADIANS:
            return None
        # 圆心坐标 ULP 噪声归零 (尺度相对, 与快照噪声地板同族) —
        # 圆心恰在坐标原点时 6g 显示 5e-16 会误导用户
        display_center = np.where(
            np.abs(center) < coordinate_scale * 1e-12,
            0.0,
            center,
        )
        prefix = "外边 " if is_outer else "内孔 "
        return Detection(
            type="arc",
            label=(
                f"{prefix}圆弧 ρ={radius:.6g}, "
                f"圆心({display_center[0]:.6g},{display_center[1]:.6g})"
            ),
            params={
                "radius": radius,
                "center": (float(display_center[0]),
                           float(display_center[1])),
                "angle": span,
                "fit_residual": mean_residual,
                "fit_residual_max": max_residual,
            },
            confidence=1.0 - min(1.0, variation / ARC_CURVATURE_CV_LIMIT),
            residual=variation,
        )

    @staticmethod
    def _algebraic_center(coords: np.ndarray) -> Optional[np.ndarray]:
        """链首最远点 + 弧顶点的垂平分线交点 (O(n), 纯代数, 无最小二乘).

        j = 离链首最远的点 (跨 π 的优弧取直径对, 短弧取另一端), 弧
        顶点 k = 离弦 (i→j) 最远的点 — 三点张成的三角形最稳, 圆心
        得解最精确. 曾 O(n²) 全对最远点: 细网格大弧 (1e5 节点) 变
        成 1e10 次运算 — 分类是逐链调用, 必须线性.
        """
        i = 0
        p_i = coords[i]
        distances = np.linalg.norm(coords - p_i, axis=1)
        j = int(np.argmax(distances))
        chord = float(distances[j])
        if chord <= 0.0:
            return None
        p_j = coords[j]
        # 弧顶点: 离弦 (i→j) 最远的点; 弓高门防近平直 (病态解)
        tangent = p_j - p_i
        relative = coords - p_i
        height = np.abs(
            tangent[0] * relative[:, 1]
            - tangent[1] * relative[:, 0]) / chord
        apex = int(np.argmax(height))
        sagitta = float(height[apex])
        if apex in (i, j) or sagitta <= chord * MIN_SAGITTA_RATIO:
            return None
        p_k = coords[apex]
        u = p_j - p_i
        v = p_k - p_i
        mid_ij = 0.5 * (p_i + p_j)
        mid_ik = 0.5 * (p_i + p_k)
        try:
            center = np.linalg.solve(
                np.array([[u[0], u[1]], [v[0], v[1]]], dtype=float),
                np.array([np.dot(u, mid_ij), np.dot(v, mid_ik)],
                         dtype=float),
            )
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(center)):
            return None
        return center

    def _short_ellipse_guard(
            self, coords: np.ndarray, scale: float,
            is_outer: bool) -> Optional[Detection]:
        """短弧保护: 椭圆弧长覆盖不足 → 保守曲线标签 (禁止硬拟合).

        覆盖足够的开放椭圆弧让位内置 EllipseDetector (旧行为不变);
        圆链 (a≈b) 让位内置圆探测器.
        """
        if len(coords) < 6:
            return None
        ellipse = fit_ellipse(coords)
        if not ellipse:
            return None
        _cx, _cy, semi_major, semi_minor, _theta = ellipse
        if semi_major <= 0.0 or semi_minor <= 0.0:
            return None
        ratio = semi_major / max(semi_minor, np.finfo(float).tiny)
        if ratio < 1.05:
            return None  # 圆弧 → 内置圆探测器
        chain_length = float(np.sum(np.linalg.norm(
            np.diff(coords, axis=0), axis=1)))
        circumference = _ellipse_circumference(
            semi_major, semi_minor)
        if chain_length >= circumference * MIN_ELLIPSE_ARC_COVERAGE:
            return None  # 覆盖足够 → 内置椭圆探测器
        detection = self._general.detect(
            coords, scale=scale, is_outer=is_outer, closed=False)
        assert detection is not None  # general 恒返回
        return detection
