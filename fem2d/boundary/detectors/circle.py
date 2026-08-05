"""圆原语探测器 — 闭合整圆 (type=arc) + 开放圆弧 (type=arc).

旧 classify 中对应 _classify_closed_conic 的圆分支 +
_classify_open_arc, 探测顺序与门槛逐位一致.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geometry import (
    _axis_ratio,
    _unwrap_angle_range,
    circle_fit_residual,
    compute_tolerance,
    fit_circle_least_squares,
)
from ._shared import _fit_closed_conic, _prefix, _segment_label
from .base import Detection, Detector


class CircleDetector(Detector):
    """圆原语探测 — 闭合整圆 (type=arc) + 开放圆弧 (type=arc).

    旧 classify 中对应 _classify_closed_conic 的圆分支 +
    _classify_open_arc, 探测顺序与门槛逐位一致.
    """

    name = "circle"

    def detect(
            self,
            points: np.ndarray,
            *,
            scale: float,
            is_outer: bool,
            closed: bool,
            native_entities: Sequence[str] = (),
    ) -> Optional[Detection]:
        coords = np.asarray(points, dtype=float)
        prefix = _prefix(is_outer)
        if closed:
            return self._closed_circle(coords, prefix)
        return self._open_arc(coords, scale, prefix)

    def _closed_circle(
            self, coords: np.ndarray, prefix: str) -> Optional[Detection]:
        conic = _fit_closed_conic(coords)
        if conic is None:
            return None
        ellipse, fit_info = conic
        center_x, center_y, semi_major, semi_minor, angle = ellipse
        ratio, is_circle = _axis_ratio(semi_major, semi_minor)
        if not is_circle:
            return None  # 椭圆链 → EllipseDetector
        radius = 0.5 * (semi_major + semi_minor)
        mean_residual = float(fit_info.get("fit_residual", np.inf))
        return Detection(
            type="arc",
            label=f"{prefix}整圆 R={radius:.3g}",
            params={
                "center": (center_x, center_y),
                "radius": radius,
                "angle": 2 * np.pi,
                **fit_info,
            },
            confidence=1.0 - min(1.0, mean_residual / 0.05),
            residual=mean_residual,
        )

    def _open_arc(
            self, coords: np.ndarray, scale: float,
            prefix: str) -> Optional[Detection]:
        if len(coords) < 4:
            return None
        tolerance = compute_tolerance(coords)
        center_x, center_y, radius = fit_circle_least_squares(coords)
        mean_residual, max_residual = circle_fit_residual(
            coords, (center_x, center_y, radius))
        fit_span = max(
            float(np.ptp(coords[:, 0])),
            float(np.ptp(coords[:, 1])),
            tolerance,
        )
        residual_limit = max(
            tolerance * 20.0,
            fit_span * 1e-4,
            np.spacing(max(np.max(np.abs(coords)), np.finfo(float).tiny))
            * 64.0,
        )
        if not (
                radius > 0.0
                and radius < max(scale, fit_span) * 1e6
                and mean_residual < residual_limit * 0.5
                and max_residual < residual_limit):
            return None

        angles = np.arctan2(
            coords[:, 1] - center_y,
            coords[:, 0] - center_x,
        )
        arc_angle = _unwrap_angle_range(angles)
        if arc_angle < np.deg2rad(5.0):
            arc_angle = 0.0
        if arc_angle <= 0.0:
            return None
        kind = "圆角" if arc_angle < np.deg2rad(30.0) else "圆弧"
        return Detection(
            type="arc",
            label=_segment_label(
                coords, prefix, kind, f"R={radius:.6g}"),
            params={
                "radius": radius,
                "center": (center_x, center_y),
                "angle": arc_angle,
                "fit_residual": mean_residual,
                "fit_residual_max": max_residual,
            },
            confidence=1.0 - min(
                1.0, mean_residual / (residual_limit * 0.5)),
            residual=mean_residual,
        )
