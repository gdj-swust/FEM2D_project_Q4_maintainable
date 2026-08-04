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

模块结构 (2026-08-05 拆分, 纯搬移 — 金标准快照逐位不变验证):
  base.py      — Detection/Detector/DetectorRegistry (注册表契约)
  line.py      — LineDetector 直边探测器
  circle.py    — CircleDetector 圆/圆弧探测器
  ellipse.py   — EllipseDetector 椭圆探测器
  general.py   — GeneralCurveDetector 兜底探测器
  registry.py  — default_registry 单例 + register_detector 插件入口
  _shared.py   — 标签前缀/段标签/闭合圆锥拟合门槛共用原语
"""
from .base import Detection, Detector, DetectorRegistry
from .circle import CircleDetector
from .ellipse import EllipseDetector
from .general import GeneralCurveDetector
from .line import LineDetector
from .registry import default_registry, register_detector

__all__ = [
    "CircleDetector",
    "Detection",
    "Detector",
    "DetectorRegistry",
    "EllipseDetector",
    "GeneralCurveDetector",
    "LineDetector",
    "default_registry",
    "register_detector",
]
