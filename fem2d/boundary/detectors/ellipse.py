"""椭圆原语探测器 — 闭合整环椭圆 + 开放椭圆弧 (type=ellipse)."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geometry import (
    _axis_ratio,
    _semi_axis_label,
    ellipse_fit_residual,
    fit_ellipse,
)
from ._shared import _fit_closed_conic, _prefix
from .base import Detection, Detector


class EllipseDetector(Detector):
    """椭圆原语探测 — 闭合整环椭圆 + 开放椭圆弧 (type=ellipse)."""

    name = "ellipse"

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
            return self._closed_ellipse(coords, prefix)
        return self._open_ellipse(coords, prefix)

    def _closed_ellipse(
            self, coords: np.ndarray, prefix: str) -> Optional[Detection]:
        conic = _fit_closed_conic(coords)
        if conic is None:
            return None
        ellipse, fit_info = conic
        center_x, center_y, semi_major, semi_minor, angle = ellipse
        ratio, is_circle = _axis_ratio(semi_major, semi_minor)
        if is_circle:
            return None  # 圆链 → CircleDetector
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

    def _open_ellipse(
            self, coords: np.ndarray, prefix: str) -> Optional[Detection]:
        if len(coords) < 6:
            return None
        ellipse = fit_ellipse(coords)
        if not ellipse:
            return None

        center_x, center_y, semi_major, semi_minor, angle = ellipse
        mean_residual, max_residual = ellipse_fit_residual(
            coords, ellipse)
        if not (
                semi_major > 0.0
                and semi_minor > 0.0
                and max(semi_major, semi_minor)
                / (min(semi_major, semi_minor) + np.finfo(float).tiny) < 20.0
                and mean_residual < 5e-4
                and max_residual < 2e-3):
            return None
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
                "fit_residual": mean_residual,
                "fit_residual_max": max_residual,
            },
            confidence=1.0 - min(1.0, mean_residual / 5e-4),
            residual=mean_residual,
        )
