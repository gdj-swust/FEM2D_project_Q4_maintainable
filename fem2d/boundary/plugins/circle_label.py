"""示例插件: Gmsh 原生 Circle 实体的圆标签探测器.

接入路径 (docs/boundary_plugins.md 五步): 新增本文件 + 一行注册
``register_detector(NativeCircleLabelDetector())`` — 管线本体零改动.

效果: 链由 Gmsh 原生 Circle 实体构成且被判定为圆 (整圆或圆弧) →
标签追加 " [Gmsh 原生圆]" 标记, info 增加 "native_circle": True —
纯展示层, 不改变边界边集合/载荷路径/压力法向 (金标准快照验证).

实现模式 = 委托上游探测器: 插件不复制判定逻辑, 调用 CircleDetector
得到基础 Detection 后改写标签 — 上游判定门槛变更时插件自动跟随.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..detectors import CircleDetector, Detection, Detector


class NativeCircleLabelDetector(Detector):
    """圆标签探测器 — 对原生 Gmsh Circle 实体构成的圆/圆弧链加展示标签."""

    name = "native_circle_label"

    def __init__(self) -> None:
        self._circle = CircleDetector()

    def detect(
            self,
            points: np.ndarray,
            *,
            scale: float,
            is_outer: bool,
            closed: bool,
            native_entities: Sequence[str] = (),
    ) -> Optional[Detection]:
        kinds = {
            str(kind).strip().casefold()
            for kind in native_entities
        }
        # 仅原生 Circle 实体链生效 — 其他链返回 None 让位内置探测器
        if not kinds or "circle" not in kinds:
            return None

        detection = self._circle.detect(
            points,
            scale=scale,
            is_outer=is_outer,
            closed=closed,
        )
        if detection is None:
            return None

        params = dict(detection.params)
        params["native_circle"] = True
        return Detection(
            type=detection.type,
            label=detection.label + " [Gmsh 原生圆]",
            params=params,
            confidence=detection.confidence,
            residual=detection.residual,
        )
