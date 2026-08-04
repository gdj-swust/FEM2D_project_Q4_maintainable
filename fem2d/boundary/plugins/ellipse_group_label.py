"""正式插件 1: 组级椭圆标签探测器 (ellipse_group_label).

职责 (docs/boundary_plugins.md 优先级规则): **闭合整椭圆链** →
"椭圆 a=.., b=.." 组级标签. 开链/短弧/圆弧/样条属插件 3
(arc_curvature) 裁决, 闭合整圆让位内置圆探测器 (整圆标签语义优先).

双路径:
  ① 原生实体直读 — 链的 Gmsh 原生实体含 ellipse/circle → 椭圆身份
     由 CAD 真值保证, 拟合只用于提取参数 (零残差门, 零角点门 —
     粗网格原生椭圆链仍应标椭圆);
  ② 点云拟合兜底 — 闭合组链委托内置 EllipseDetector 判定 (同一
     门槛, 自动跟随), 通过后再过插件**严格门**: 相对偏差 <2%
     (dimensionless 径向残差 mean/max) 且弧长覆盖 ≥90% (链长 /
     拟合椭圆周长, Ramanujan 近似). 通不过 → 保守"通用曲线"标签
     (宁缺毋滥: 阻止内置宽松门把残差超门的闭合链标成椭圆).

实现模式 = 委托上游探测器: 兜底路径不复制判定逻辑, EllipseDetector
判定门槛变更时插件自动跟随; 严格门失败时委托 GeneralCurveDetector
生成保守标签 (与链在无椭圆声明时的兜底输出逐位一致).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from ..detectors import (
    EllipseDetector,
    GeneralCurveDetector,
    Detection,
    Detector,
)
from ..geometry import (
    _axis_ratio,
    _semi_axis_label,
    fit_closed_ellipse,
)

# 严格残差门: 相对偏差 <2% (dimensionless 径向残差, 与 fit 残差同量纲)
RELATIVE_DEVIATION_LIMIT = 0.02
# 弧长覆盖: 闭合链长 ≥ 拟合椭圆周长的 90%
MIN_ARC_COVERAGE = 0.9
# 原生实体直读只认 ellipse/circle 类型 (Line 镶嵌/无实体 → 兜底拟合)
_NATIVE_CONIC_KINDS = frozenset({"ellipse", "circle"})


def _ellipse_circumference(semi_major: float, semi_minor: float) -> float:
    """Ramanujan 椭圆周长近似 — 覆盖判据只用相对比值, 近似误差无害."""
    a = float(semi_major)
    b = float(semi_minor)
    return math.pi * (3.0 * (a + b)
                      - math.sqrt((3.0 * a + b) * (a + 3.0 * b)))


class EllipseGroupLabelDetector(Detector):
    """组级椭圆标签探测器 — 闭合整椭圆链的双路径判定 (见模块 docstring)."""

    name = "ellipse_group_label"

    def __init__(self) -> None:
        self._ellipse = EllipseDetector()
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
        if not closed:
            # 开链/短弧/圆弧/样条 → 插件 3 (arc_curvature) 裁决
            return None
        kinds = {
            str(kind).strip().casefold()
            for kind in native_entities
        }
        if kinds & _NATIVE_CONIC_KINDS:
            return self._native_direct(points, is_outer)
        return self._fit_fallback(points, scale, is_outer)

    def _native_direct(
            self, points: np.ndarray, is_outer: bool) -> Optional[Detection]:
        """① 原生实体直读: ellipse/circle 实体 = CAD 真值, 零门取参."""
        coords = np.asarray(points, dtype=float)
        ellipse, fit_info = fit_closed_ellipse(coords)
        if not ellipse:
            return None
        center_x, center_y, semi_major, semi_minor, angle = ellipse
        _ratio, is_circle = _axis_ratio(semi_major, semi_minor)
        if is_circle:
            return None  # 整圆 → 内置圆探测器 ("整圆 R=.." 标签优先)
        prefix = "外边 " if is_outer else "内孔 "
        mean_residual = float(fit_info.get("fit_residual", np.inf))
        return Detection(
            type="ellipse",
            label=(
                f"{prefix}椭圆 "
                f"{_semi_axis_label(semi_major, semi_minor)}"
            ),
            params={
                "center": (center_x, center_y),
                "semi_major": semi_major,
                "semi_minor": semi_minor,
                "angle": angle,
                **fit_info,
            },
            confidence=1.0 - min(1.0, mean_residual / 0.05),
            residual=mean_residual,
        )

    def _fit_fallback(
            self, points: np.ndarray, scale: float,
            is_outer: bool) -> Optional[Detection]:
        """② 点云拟合兜底: 内置判定 + 严格残差门 + 弧长覆盖门."""
        coords = np.asarray(points, dtype=float)
        base = self._ellipse.detect(
            coords, scale=scale, is_outer=is_outer, closed=True)
        if base is None:
            # 圆链/非椭圆/角点过密 → 让位内置探测器 (保持原状)
            return None
        mean_residual = float(base.residual)
        max_residual = float(
            base.params.get("fit_residual_max", np.inf))
        if (
                mean_residual >= RELATIVE_DEVIATION_LIMIT
                or max_residual >= RELATIVE_DEVIATION_LIMIT):
            return self._conservative(coords, scale, is_outer)
        if not self._coverage_ok(coords, base.params):
            return self._conservative(coords, scale, is_outer)
        return base

    @staticmethod
    def _coverage_ok(coords: np.ndarray, params: dict) -> bool:
        """弧长覆盖: 闭合链长 / 拟合椭圆周长 ≥ 90%.

        闭合链的点即椭圆周长的离散采样, 弦长和略小于周长 —
        覆盖门防的是"闭合但只描了椭圆一小段"的畸形链.
        """
        semi_major = float(params.get("semi_major", 0.0))
        semi_minor = float(params.get("semi_minor", 0.0))
        if semi_major <= 0.0 or semi_minor <= 0.0:
            return False
        chain_length = float(np.sum(np.linalg.norm(
            np.diff(coords, axis=0), axis=1)))
        circumference = _ellipse_circumference(
            semi_major, semi_minor)
        return chain_length >= circumference * MIN_ARC_COVERAGE

    def _conservative(
            self, coords: np.ndarray, scale: float,
            is_outer: bool) -> Detection:
        """严格门失败 → 保守兜底: 通用曲线标签, 阻止内置宽松椭圆门."""
        detection = self._general.detect(
            coords, scale=scale, is_outer=is_outer, closed=True)
        assert detection is not None  # general 恒返回
        return detection
