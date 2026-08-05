"""通用曲线兜底探测器 — 恒返回, 保证 classify 必有结果."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..geometry import curvature
from ._shared import _prefix
from .base import Detection, Detector


class GeneralCurveDetector(Detector):
    """通用曲线兜底 — 恒返回, 保证 classify 必有结果."""

    name = "general"

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
        # 曲率量纲 1/长度 — 绝对 1e-8/1e-14 阈值会使大坐标 (1e12 级,
        # R>1e8) 平滑曲线 κ<1e-8 被降级成"通用曲线"、拐点漏计。按
        # characteristic span 归一 (segment_by_curvature 的
        # 1e-8/characteristic 同款模式).
        coordinate_scale = max(
            float(np.ptp(coords[:, 0])),
            float(np.ptp(coords[:, 1])),
            np.finfo(float).tiny,
        )
        curvature_floor = 1e-8 / coordinate_scale
        sign_floor = 1e-14 / coordinate_scale

        values = curvature(coords, closed=closed)
        evaluated = values if closed else values[1:-1]
        if len(evaluated) == 0:
            evaluated = np.zeros(1, dtype=float)
        absolute = np.abs(evaluated)
        mean = float(np.mean(absolute))
        deviation = float(np.std(absolute))
        variation = deviation / (mean + np.finfo(float).tiny)
        length = float(np.sum(np.linalg.norm(
            np.diff(coords, axis=0), axis=1)))
        signs = np.sign(evaluated)
        nonzero_signs = signs[
            np.abs(evaluated) > max(mean * 1e-6, sign_floor)
        ]
        inflections = (
            int(np.sum(nonzero_signs[1:] * nonzero_signs[:-1] < 0))
            if len(nonzero_signs) > 1 else 0
        )
        common_info = {
            "geometry": "general",
            "closed": closed,
            "length": length,
            "curvature_mean": mean,
            "curvature_std": deviation,
            "curvature_cv": variation,
            "inflection_count": inflections,
        }
        if variation < 0.15:
            equivalent_radius = 1.0 / (mean + np.finfo(float).tiny)
            return Detection(
                type="curve",
                label=f"{prefix}类圆 R~{equivalent_radius:.6g}",
                params={
                    **common_info,
                    "equivalent_radius": equivalent_radius,
                },
                confidence=0.5,
                residual=variation,
            )
        if mean > curvature_floor and variation < 0.5:
            minimum_radius = 1.0 / (absolute.max() + np.finfo(float).tiny)
            maximum_radius = (
                1.0 / (
                    absolute[absolute > curvature_floor].min()
                    + np.finfo(float).tiny)
                if (absolute > curvature_floor).any()
                else 1e9 * coordinate_scale
            )
            return Detection(
                type="curve",
                label=(
                    f"{prefix}曲线 R=[{minimum_radius:.6g},"
                    f"{maximum_radius:.6g}]"
                ),
                params={
                    **common_info,
                    "R_min": minimum_radius,
                    "R_max": maximum_radius,
                },
                confidence=0.5,
                residual=variation,
            )
        return Detection(
            type="curve",
            label=f"{prefix}通用曲线 ({len(coords)}点)",
            params=common_info,
            confidence=0.5,
            residual=variation,
        )
