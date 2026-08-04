"""识别器基类与有序注册表 — 探测器判定层的公共契约."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from ..geometry import _segment_is_closed, compute_tolerance


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
