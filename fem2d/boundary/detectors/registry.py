"""默认注册表与插件注册入口 — 管线 classify 的统一注册表."""
from __future__ import annotations

from .base import Detector, DetectorRegistry
from .circle import CircleDetector
from .ellipse import EllipseDetector
from .general import GeneralCurveDetector
from .line import LineDetector

_DEFAULT_REGISTRY = None


def default_registry():
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
