"""边界识别器注册表 — 显式管线中的几何判定层.

管线: 拓扑 (topology.detect / segment_builder 有序链) → 几何 (本模块)
→ 物理组 (physical_mapping / registry_mapping) → 段 (segment_builder)
→ 标签 (naming 打印/描述).

识别器接口 (Detector.detect):
    输入: points 点链 (n,2) + 可选原生实体信息 native_entities + 尺度
    输出: Detection | None — {type, 参数, 标签, 置信度, 残差}

注册表有序, 登记顺序 = 判定优先级 (与旧 geometry.classify 探测顺序
逐位一致, 金标准快照锁定):
  - line:    LineDetector    → 段类型 "line"
  - circle:  CircleDetector  → 段类型 "arc" (闭合整圆 + 开放圆弧 —
             圆原语的两种几何形态, 旧实现同属一个探测分支)
  - ellipse: EllipseDetector → 段类型 "ellipse" (闭合整环椭圆 + 开放椭圆弧)
  - arc:     槽位预留 — 未来独立弧探测器的接入点 (当前开放圆弧由
             circle 判定, 槽位无内置探测器)
  - general: GeneralCurveDetector 兜底 (恒返回)

原生实体信息一等公民: native_entities (Gmsh 实体类型 line/circle/
ellipse/bspline) 沿管线传入 classify 且不丢失; 内置探测器不消费该
信息 (行为冻结), 供插件参考 (如 circle 标签探测器的原生实体提示).

插件接入: register_detector() 将插件插入注册表**前端** (classify
短路于首个非 None 判定 — 追加到末尾的插件会被 general 兜底永远
遮蔽). 插件判定优先, 未命中 (返回 None) 时让位内置探测器.
典型插件模式 = 委托上游探测器改写标签 (见 plugins/circle_label.py).

行为约束: 内置探测器的判定门槛与标签文本与旧 classify 逐位一致 —
任何改动由 tests/boundary_golden/ 金标准快照拦截。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .geometry import (
    _axis_ratio,
    _closed_conic_turn_limit_deg,
    _coords_ulp,
    _segment_is_closed,
    _semi_axis_label,
    circle_fit_residual,
    compute_tolerance,
    curvature,
    ellipse_fit_residual,
    fit_circle_least_squares,
    fit_closed_ellipse,
    fit_ellipse,
    sharp_corner_indices,
    _unwrap_angle_range,
)


@dataclass(frozen=True)
class Detection:
    """识别器输出 — 类型/参数/标签/置信度/残差.

    params 是段 info 的参数部分 (与旧 classify 输出逐位一致);
    confidence/residual 是插件参考元数据, 不写入段 info (段 schema
    由金标准锁定, 不得新增键).
    """

    type: str
    label: str
    params: dict
    confidence: float
    residual: float


class Detector:
    """识别器基类 — 点链 + 可选原生实体信息 + 尺度 → Detection 或 None.

    子类必须实现 detect() 且 name 全局唯一 (注册表按 name 去重).
    """

    name: str = "detector"

    def detect(
            self,
            points: np.ndarray,
            *,
            scale: float,
            is_outer: bool,
            closed: bool,
            native_entities: Sequence[str] = (),
    ) -> Optional[Detection]:
        raise NotImplementedError(
            f"{type(self).__name__}.detect 未实现")


class DetectorRegistry:
    """有序识别器注册表 — 首个返回非 None 的探测器胜出.

    兜底 GeneralCurveDetector 恒返回, 故 classify 永不落空;
    注册表被清空/兜底缺失 → 响亮报错而非静默返回.
    """

    def __init__(self, detectors: Sequence[Detector] = ()):
        self._detectors: list[Detector] = []
        for detector in detectors:
            self.add(detector)

    def add(self, detector: Detector) -> None:
        if not isinstance(detector, Detector):
            raise TypeError(
                f"register_detector: 期望 Detector 实例, 收到 "
                f"{type(detector).__name__}")
        if any(existing.name == detector.name
               for existing in self._detectors):
            raise ValueError(
                f"register_detector: 识别器名 {detector.name!r} 已注册 "
                "— 插件名必须全局唯一")
        self._detectors.append(detector)

    def remove(self, name: str) -> bool:
        """按 name 移除识别器 (测试清理/插件热卸载); 不存在返回 False."""
        for index, detector in enumerate(self._detectors):
            if detector.name == name:
                del self._detectors[index]
                return True
        return False

    def detectors(self) -> Tuple[Detector, ...]:
        return tuple(self._detectors)

    def classify(
            self,
            coords: np.ndarray,
            scale: float,
            is_outer: bool,
            closed: Optional[bool] = None,
            *,
            native_entities: Sequence[str] = (),
    ) -> Tuple[str, str, dict]:
        """(段类型, 标签, info 参数) — 与旧 geometry.classify 同签名同返回.

        closed=None 时按坐标闭合容差自行判定 (旧语义);
        原生实体信息原样传入每个探测器, 不丢失.
        """
        coords = np.asarray(coords, dtype=float)
        tolerance = compute_tolerance(coords)
        is_closed = (
            _segment_is_closed(coords, tolerance, closed)
            if closed is None else bool(closed))
        for detector in self._detectors:
            detection = detector.detect(
                coords,
                scale=scale,
                is_outer=is_outer,
                closed=is_closed,
                native_entities=tuple(str(e) for e in native_entities),
            )
            if detection is not None:
                return detection.type, detection.label, dict(detection.params)
        raise AssertionError(
            "识别器注册表空转: 兜底 GeneralCurveDetector 未注册, "
            "classify 必须恒有结果")


def _prefix(is_outer: bool) -> str:
    return "外边 " if is_outer else "内孔 "


def _segment_label(
        coords: np.ndarray, prefix: str, kind: str, extra: str = "") -> str:
    """构建稳定的用户可见基元标签 (与旧 classify 文本逐位一致)."""
    x0, y0 = map(float, coords[0])
    x1, y1 = map(float, coords[-1])
    base = f"{prefix}{kind}"
    if extra:
        base += f" {extra}"
    return f"{base} ({x0:.6g},{y0:.6g})→({x1:.6g},{y1:.6g})"


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
        conic = _fit_closed_conic(coords, prefix)
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
        conic = _fit_closed_conic(coords, prefix)
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
        # 曲率量纲 1/长度 — 绝对 1e-8/1e-14 阈值曾使大坐标 (1e12 级,
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


# ═══════════════════════════════════════════════════════════════
# 闭合圆锥曲线共用判定 (旧 _classify_closed_conic 的拟合+门槛部分)
# ═══════════════════════════════════════════════════════════════

def _fit_closed_conic(
        coords: np.ndarray, prefix: str):
    """闭合链的整环圆锥拟合 + 平滑门槛 — 通过返回 (ellipse, fit_info),
    Circle/Ellipse 探测器按轴比各自接管. 门槛与旧实现逐位一致."""
    if len(coords) < 8:
        return None
    ellipse, fit_info = fit_closed_ellipse(coords)
    if not ellipse:
        return None
    if sharp_corner_indices(
            coords,
            angle_threshold_deg=_closed_conic_turn_limit_deg(
                ellipse, fit_info)):
        return None
    return ellipse, fit_info


# ═══════════════════════════════════════════════════════════════
# 默认注册表 — 管线 (topology/segment_builder/conic_merge) 共用
# ═══════════════════════════════════════════════════════════════

_DEFAULT_REGISTRY = None


def default_registry() -> DetectorRegistry:
    """惰性单例 — 管线 classify 的统一注册表.

    插件接入: register_detector() 追加到此注册表, 管线本体零改动
    (见 docs/boundary_plugins.md 五步接入).
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = DetectorRegistry([
            LineDetector(),
            CircleDetector(),
            EllipseDetector(),
            GeneralCurveDetector(),
        ])
    return _DEFAULT_REGISTRY


def register_detector(detector: Detector) -> None:
    """向默认注册表登记插件识别器 (name 重复 → ValueError).

    插件插入注册表**前端**: classify 短路于首个非 None 判定, 追加到
    末尾的插件永远不被调用 (general 兜底恒返回). 插件判定优先,
    未命中 (返回 None) 时让位内置探测器 (line→circle→ellipse→general).
    """
    if not isinstance(detector, Detector):
        raise TypeError(
            f"register_detector: 期望 Detector 实例, 收到 "
            f"{type(detector).__name__}")
    registry = default_registry()
    if any(existing.name == detector.name
           for existing in registry._detectors):
        raise ValueError(
            f"register_detector: 识别器名 {detector.name!r} 已注册 "
            "— 插件名必须全局唯一")
    registry._detectors.insert(0, detector)
