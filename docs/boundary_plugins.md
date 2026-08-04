# 边界识别器插件接入手册（五步）

> 版本: 9.23.0 (2026-08-04, 阶段 3 交付物)。目标: **新识别能力 =
> 插一个插件 + 照手册走, 不碰管线本体、不影响载荷链**。
> 示例插件: `fem2d/boundary/plugins/circle_label.py`（Gmsh 原生 Circle
> 圆标签探测器, 判别性测试 `tests/test_boundary_plugin_circle_label.py`）。

## 管线概览（插件插入点）

```
拓扑 (topology.detect / segment_builder 有序链)
  → 几何 (detectors.DetectorRegistry.classify)   ← 插件插在这里
  → 物理组 (physical_mapping / registry_mapping)
  → 段 (segment_builder / conic_merge)
  → 标签 (naming.print_segments / describe_geometry)
```

`classify` 有序调用注册表探测器, **首个返回非 None 的判定胜出**。
内置顺序: `line → circle → ellipse → general`（general 恒返回兜底）。
插件经 `register_detector()` 插入注册表**前端** — 插件判定优先,
未命中（返回 `None`）时自动让位内置探测器。

**行为冻结**: 边界边集合 / 载荷路径 / 压力法向逐位不变 — 由
`tests/boundary_golden/` 金标准快照锁定。插件只影响展示层
（段标签 / info 键）。

## 第一步: 接口签名

识别器必须继承 `fem2d.boundary.detectors.Detector` 并实现:

```python
class MyDetector(Detector):
    name = "my_detector"          # 全局唯一 (注册表按 name 去重)

    def detect(self, points, *, scale, is_outer, closed,
               native_entities=()):
        # points:        (n,2) 点链 (np.ndarray)
        # scale:         模型特征尺度 (几何分类参考)
        # is_outer:      链在外边界 (True) / 内孔 (False)
        # closed:        链是否闭合 (首尾同点)
        # native_entities: Gmsh 原生实体类型元组, 如 ("Circle",) /
        #                  ("Line",) / ("Circle", "Line") — 沿管线
        #                  传递不丢失 (一等公民输入, 内置探测器不消费)
        ...
        return Detection(           # 或 None (不主张此链, 让位后续探测器)
            type="arc",             # 段类型: "line" | "arc" | "ellipse" | "curve"
            label="内孔 整圆 R=0.3",  # 完整用户可见标签 (print_segments 输出)
            params={...},           # 段 info 参数 dict (写入段 info)
            confidence=0.99,        # 0..1 拟合置信度 (插件参考, 不写入段)
            residual=0.001,         # 拟合残差 (插件参考, 不写入段)
        )
```

- `Detection` 的 `params` 会原样写入段 `info` — 新增键会在金标准
  快照中可见, 请明确这是有意输出。
- 常用模式 = **委托上游探测器**: 调用 `CircleDetector().detect(...)`
  得到基础 Detection 后改写标签/参数（见 circle_label.py 示例）—
  不复制判定逻辑, 上游门槛变更时插件自动跟随。

## 第二步: 注册（一行）

```python
from fem2d.boundary import register_detector
from fem2d.boundary.plugins.my_detector import MyDetector

register_detector(MyDetector())
```

- 注册表为进程级单例 (`default_registry()`); 重复 name →
  `ValueError`。
- 插件插入**前端**: 本插件优先判定; 返回 `None` 时让位内置。
- 测试内注册后须恢复: `default_registry()._detectors[:] = 原列表`
  （或 `remove(name)`）, 避免污染后续测试。

## 第三步: 标签输出

插件的 `label` 是段标签的最终文本 — 经 `print_segments` /
`describe_geometry` 直接展示给 CLI 用户; `params` 键写入段 `info`
（`segment_utils.segment_info` 读取）。

生效位置（三处 classify 调用都会咨询注册表）:
1. 自动检测路径 (`topology.detect`) — `native_entities=()` (纯几何);
2. 物理曲线路径 (`segment_builder.append_chain`) — 传边组的
   `cad_entity_types`;
3. CAD 合并路径 (`conic_merge._merge_component`) — 传源段的
   `cad_entity_types` 并集。

**约束**: 插件不得改变段的 `nodes` / `coords` / 边界边集合 — 那是
载荷输入链 (动脉), 改动即破坏金标准快照。

## 第四步: 判别性测试

两个断言缺一不可 (模式见 test_boundary_plugin_circle_label.py):

```python
def test_plugin_unit_contract():
    """插件单元契约: 期望值硬编码 — 插件退化 (丢委托/丢标记) 必红."""
    detection = MyDetector().detect(chain, scale=2.0, is_outer=True,
                                    closed=True, native_entities=("Circle",))
    assert detection is not None
    assert detection.label == "... [期望标签]"   # 硬编码
    # 让位条件: 非目标链 → None

def test_plugin_not_registered_pipeline_unchanged():
    """未注册插件时管线输出无插件效果 (金标准锁定的默认行为)."""

def test_plugin_registered_through_pipeline():   # 需要 gmsh 时 skipif
    """注册 1 行 → 真实管线生效; 判别性: 去掉注册行 → 断言必红."""
    registry = default_registry()
    original = list(registry._detectors)
    try:
        register_detector(MyDetector())
        segments = build_boundary_segments(mesh, registry=reg)
        assert 期望效果 in segments
    finally:
        registry._detectors[:] = original
```

"放回旧实现必须失败": 删掉插件文件 → import 红; 删掉注册行 →
管线测试红; 插件效果断言改为旧标签 → 红。

## 第五步: 快照更新

插件效果进段标签/info 后, 若该输出**成为默认行为**（内置化）, 更新
金标准:

```bash
FEM2D_UPDATE_GOLDEN=1 python -m pytest tests/test_boundary_golden_deterministic.py tests/test_boundary_golden_gmsh.py
python -m pytest -q          # 全量 0 失败
git add tests/boundary_golden/ && git commit -m "金标准快照更新: ..."
```

仅当你确认新输出是**有意行为变更**时才重写金标准 — 任何无声漂移
必须先查明原因。插件效果属于插件测试自身断言, 不写入金标准。

## 已内置探测器的判定门槛（迁移自旧 classify, 行为逐位冻结）

| 探测器 | 段类型 | 判定核心 |
|--------|--------|----------|
| LineDetector | line | 全部点到端点连线距离 < tol×5, 轴 = 倾角分档 |
| CircleDetector | arc | 闭合整圆 (整环椭圆拟合 + 轴比<1.05 + 平滑门) / 开放圆弧 (最小二乘圆拟合 + 残差门) |
| EllipseDetector | ellipse | 闭合整环椭圆 / 开放椭圆弧 (SVD 拟合 + 残差门) |
| GeneralCurveDetector | curve | 恒返回兜底 (曲率统计分档) |

## 正式插件目录 (轮 2, 2026-08-05 — 默认注册, 注册即生效)

三个正式插件经 `fem2d/boundary/plugins/__init__.py` 在 import 链默认
注册 (不是测试内手动注册); `register_detector` 插入注册表**前端**,
故注册顺序 = 判定优先级。**优先级规则 (职责划分, 必须可测)**:

| 优先级 | 插件 | 职责 | 关键门槛 |
|--------|------|------|----------|
| 1 | `arc_curvature` (插件 3) | **开链/短弧/圆弧/样条** — 最保守, 先裁决 | 圆弧: 内部 κ 恒定 (CV<0.15) + 弧跨度 ≥5° + 弓高 ≥1e-4×弦长 + 代数残差 <2%ρ → "圆弧 ρ=.., 圆心(..,..)" (垂平分线交点, **禁止拟合**); 椭圆弧覆盖 <60% → 保守曲线标签 (禁止硬拟合短弧) |
| 2 | `ellipse_group_label` (插件 1) | **闭合整椭圆链** → "椭圆 a=.., b=.." | ①原生 ellipse/circle 实体 → 直接读参 (零门, CAD 真值); ②点云拟合: 委托内置 EllipseDetector + 严格门 (相对偏差 <2% 且弧长覆盖 ≥90%); 失败 → 保守曲线标签 (宁缺毋滥) |
| 3+ | 内置探测器 | 闭合整圆 (→"整圆 R=.."), 直边, 开放椭圆弧 (覆盖足够) | 行为逐位冻结 |

规则要点:
- **闭合整圆让位内置圆探测器** — "整圆 R=.." 标签语义优先于 "椭圆 a=b"
  (multi_hole_plate 的 24 段 Circle 环保持整圆)。
- **开链永不当椭圆** — 插件 3 先裁决: 1/8 椭圆等短弧 (弧长覆盖不足)
  即使椭圆拟合残差完美 (实测 ~1e-16) 也只给曲线类标签。
- **覆盖足够的开放椭圆弧** (≥60%) 让位内置 EllipseDetector (旧行为)。
- 标签冲突以更保守者胜: 插件 3 (开链) 先于插件 1 (闭合); 插件 1 的
  严格门失败 → 保守标签, 阻止内置宽松门 (残差 3%/6%) 标椭圆。

## 插件 2: @组名批量选择 (CLI 输入层, 非 Detector)

`--traction "@椭圆孔:0,-2e6"` (--fix 同理) → 展开为该物理组全部曲线段;
段列表打印处提示 "输入编号 12,13 精细选择；输入 @组名 整组选择"。

- 展开语义: `region_registry.by_name(name, dimension=1)` 返回组内全部
  曲线, 段按 `info["physical_names"]` 精确匹配 (大小写不敏感)。
- 锁定决策 (不可改): 保留编号通道; 组名不存在 → `CliError(exit_code=1)`
  (批处理) / [WARN] (交互); 单个选择串内不支持编号+@混用 (响亮报错);
  不做段编号稳定化 (@组名 = 跨网格编号漂移的不变通道)。
- 判别性: @组名展开结果 == 手输全部编号的施加结果 (逐边 traction 一致)。
