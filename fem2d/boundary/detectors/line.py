"""直边探测器 — 全部点到端点连线距离 < tolerance*5 即判直边."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geometry import _coords_ulp, compute_tolerance
from ._shared import _prefix, _segment_label
from .base import Detection, Detector


class LineDetector(Detector):
    """直边探测 — 全部点到端点连线距离 < tolerance*5 即判直边."""

    name = "line"

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
        tolerance = compute_tolerance(coords)
        x, y = coords[:, 0], coords[:, 1]
        delta_x = x[-1] - x[0]
        delta_y = y[-1] - y[0]
        length = float(np.hypot(delta_x, delta_y))
        # 曾绝对 1e-15: 微尺度直边全被判零长 → 整环合并成单个 curve 段
        #
        if length <= tolerance or length <= _coords_ulp(coords):
            return None

        distances = [
            abs(
                (x[index] - x[0]) * delta_y
                - (y[index] - y[0]) * delta_x
            ) / length
            for index in range(len(coords))
        ]
        if max(distances) >= tolerance * 5.0:
            return None

        angle = abs(np.degrees(np.arctan2(delta_y, delta_x)))
        if angle < 10.0 or angle > 170.0:
            axis = "y"
        elif 80.0 < angle < 100.0:
            axis = "x"
        else:
            axis = "tilted"
        prefix = _prefix(is_outer)
        residual = min(1.0, float(max(distances)) / (tolerance * 5.0))
        return Detection(
            type="line",
            label=_segment_label(
                coords, prefix, "直边"),
            params={
                "start": (x[0], y[0]),
                "end": (x[-1], y[-1]),
                "pos": y[0] if axis == "y" else x[0],
                "axis": axis,
            },
            confidence=1.0 - residual,
            residual=residual,
        )
