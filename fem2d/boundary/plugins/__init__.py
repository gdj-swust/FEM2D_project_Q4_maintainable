"""边界识别器正式插件包 (轮 2: 三个正式插件, 默认注册).

每个插件 = 一个 Detector 子类, 经
``fem2d.boundary.register_detector`` 一行注册接入管线
(见 docs/boundary_plugins.md 五步接入手册).

生产默认注册 (注册即生效): import fem2d.boundary 即启用 — 插件
不能只在测试里活. 注册顺序 = 判定优先级 (register_detector 插入
注册表前端), 插件间职责划分与优先级规则见 docs/boundary_plugins.md:
  - 闭合整椭圆链 → ellipse_group_label (插件 1, 严格 2% 门)
  - 开链/短弧/圆弧/样条 → arc_curvature (插件 3, 更保守, 先注册)
  - 闭合整圆 → 让位内置圆探测器 (整圆标签语义优先)

circle_label 是阶段 3 的测试内注册示例, 不在此默认注册 (其判别性
测试断言默认注册表不含该插件).
"""
from ..detectors import register_detector

from .ellipse_group_label import EllipseGroupLabelDetector

register_detector(EllipseGroupLabelDetector())
