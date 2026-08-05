# CHANGELOG

> 历史修复里程碑汇总 (2026-08-03 起)。源码注释只保留"为什么必须这样做"；
> 修复历史与审计记录迁移至此。更早的历史散见于代码注释与知识库日志。

## --keep-open (2026-08-05) — 批处理命令可选保留交互窗口

- 现象: `python run.py models/demo_complex.geo --fix 2,4 --traction ...`
  CLI 传 BC 即批处理, 计算完直接退出、无窗口 — 脚本调用不卡死是设计
  优点 (保留), 本次加显式选项让批处理命令**可选**进交互。
- `--keep-open`: 批处理命令 (CLI 传 BC) 计算完成后仍进入 interactive_plot
  (窗口停留 + 键盘切换云图, q 退出); 不带该选项 → 现有批处理行为零变化。
- 组合规则: `--keep-open --save x.png` → 先保存再交互 (允许); `--keep-open
  --no-plot` → 互斥, 求解前拦截 CliError exit 1; `--keep-open` + 非 TTY
  stdin → input() EOFError 走现有 INFO 路径退出 0; 无 BC 参数 + 
  `--keep-open` → 幂等。
- 改动面: `fem2d/cli.py` (标志+帮助文本)、`fem2d/config.py` (keep_open
  字段)、`fem2d/runner.py` (_plot 批处理判定一行 + main 互斥校验)。
  `visualize.py`/边界/求解/载荷逻辑零改动。
- 兜底: `_plot` 的批处理判定把 no_plot 设为绝对优先 — CLI 互斥虽在
  main 拦截, API 直调 AnalysisConfig(keep_open+no_plot) 时抑制语义
  也不被推翻。
- 判别性: `tests/test_runner_branches.py` keep-open 节 9 测 — 放回旧
  实现, 5 测失败 (放行交互 / save 组合 / 互斥 exit 1 / 非 TTY 退出 0 /
  no_plot 绝对优先兜底); 批处理不交互行为由 1 测锁死。

## 审查修复包 2026-08-05 — 5 项已复现缺陷 (来自 2026-08-04 外部审查, 未发版)

### 1. 位移模长/模型跨度 hypot 溢出 (P1)

- `np.sqrt(u²+v²)` 在 |u|~1e308 时先平方溢出成 inf — 有限模长算成 inf,
  变形放大系数退 0、摘要打印 inf。改 `np.hypot` (reporting.py ×2,
  solver.py model_span, visualize.py u_mag 同型模式)。
- 判别性: `tests/test_scale_overflow_guard.py` 3 测 — 放回旧实现全失败。

### 2. 奇异守卫吞警告 (P1/P2)

- `_solve_with_singular_guard` 的 `catch_warnings(record=True)` 只把
  MatrixRankWarning 转异常, 其余警告静默丢弃 (UserWarning 探针 0 条
  送达)。非秩警告现于块外原样转发 (块内转发会被记录器重新捕获)。
- 判别性: `tests/test_solver_warning_forward.py` 3 测 — 放回旧守卫,
  警告转发 2 测失败; 秩警告转异常行为零变化。

### 3. CLI 非法输入退出码归类 (P2)

- `--traction right:bad` (parse ValueError 泄漏) 与 `--force 0,nan,0`
  (float("nan") 绕过解析直达 require_finite_scalar 裸 ValueError) 曾归
  内部错误退出码 2 — 现统一转 `CliError(exit_code=1)` (9.22.0 退出码
  矩阵口径: 用户错误 1)。
- 判别性: `test_exit_code_matrix.py` 新增 2 测 — 放回旧实现泄漏 exit 2。

### 4. internal_energy 舍入负值钳 0 (P3)

- 刚体位移 (纯 Dirichlet 均匀位移) 时 uᵀKu 精确为 0, 浮点舍入给
  -4.66e-13。现按问题规模相对判定: |energy| ≤ n·ε·uᵀ|K|u (禁绝对阈值,
  微尺度模型 2.27e-308 正值恒保留) → 钳 0; 明显负值 → RuntimeWarning。
- **golden 变化 (唯一)**: `test_solve_refactor_lock.py` GOLDEN
  dirichlet_only 的 `internal_energy` -4.656612873077393e-13 → 0.0,
  变化原因 = 钳 0 修复本身; 其余键逐位不变。
- 判别性: `tests/test_internal_energy_roundoff_clamp.py` 2 测 — 放回
  旧实现, 刚体位移测失败 (-4.66e-13 != 0.0)。

### 5. .geo 信任边界标注 + SystemCall 黑名单拦截 (P0 条件性)

- Gmsh 脚本支持 `SystemCall` 执行任意系统命令 — 第三方 .geo 即 RCE 面。
  清洗逻辑 (`sanitize_geo_source`, API 与子进程双路径共用) 现黑名单
  拦截 SystemCall 并明确报错 (行号 + 指令原文, `GeoScriptRejected`);
  注释内的 SystemCall 不误拒; `Include` 保留 (项目自带相对 Include
  锚定功能, 有测试锁定), 其文件链由信任边界文档覆盖。
- CLI 路径 (.geo 生成 / Physical Point 回退) 拒绝映射退出码 1。
- README 新增标注: **.geo 是"可信、可执行式输入"** — 只应运行自己编写
  的文件, 不承诺沙箱安全。
- 判别性: `tests/test_geo_script_guard.py` 6 测 — 放回旧清洗逻辑全失败。

## 9.24.0 (2026-08-05) — 边界识别三正式插件 (轮 2, 默认注册)

### 插件 1 — 组级椭圆标签 `ellipse_group_label`

- **双路径**: ①原生实体直读 — 链的 Gmsh 原生实体含 ellipse/circle →
  直接读参标 "椭圆 a=.., b=.." (零拟合门, CAD 真值; 粗采样原生椭圆链
  内置角点门会拒, 插件仍正确标注); ②点云拟合兜底 — 委托内置
  EllipseDetector 判定, 通过后再过插件严格门: 相对偏差 <2% (mean/max
  dimensionless 径向残差) 且弧长覆盖 ≥90% (Ramanujan 周长)。
- **宁缺毋滥**: 严格门失败 → 保守曲线标签 (委托 GeneralCurveDetector),
  阻止内置宽松门 (3%/6%) 把残差超门的闭合链标成椭圆。判别性: 残差
  2.5% 链 — 有插件 → 通用曲线, 无插件 → "椭圆" (本文件显式证明)。
- **闭合整圆让位内置圆探测器** — "整圆 R=.." 标签语义优先
  (multi_hole_plate 24 段 Circle 环保持整圆)。
- 金标准零变化: demo_complex/ellipse_fan 的椭圆链残差 0.78%/0% 均过门,
  插件标签与内置逐位一致。

### 插件 2 — @组名批量选择 (CLI 输入层)

- `--traction "@椭圆孔:0,-2e6"` (--fix 同理) → `region_registry.by_name(
  name, dimension=1)` 展开为该物理组全部曲线段 (段按 physical_names
  精确匹配, 大小写不敏感); 段列表打印处提示 "输入编号 12,13 精细选择；
  输入 @组名 整组选择"。
- 锁定决策: 保留编号通道; 组名不存在 → CliError(exit_code=1); 单个
  选择串内编号+@混用 → 响亮报错; 不做段编号稳定化 (@组名 = 跨网格
  编号漂移的不变通道)。
- 判别性: @组名施加结果 == 手输全部编号施加结果 (逐边 traction 一致);
  旧编号/名称语法零变化 (全量 pytest 锁定); e2e 组缺失退出码 1。

### 插件 3 — 曲率分段展示层 `arc_curvature` (最保守, 先注册)

- **代数圆弧**: 开链内部 κ 恒定 (CV<0.15) + 弧跨度 ≥5° + 弓高 ≥1e-4×弦
  + 代数残差 <2%ρ → "圆弧 ρ=.., 圆心(..,..)" — 圆心 = 最远点对 + 弧
  顶点的垂平分线交点 (2×2 线性代数, **禁止最小二乘拟合**)。
- **短弧保护 (禁止硬拟合)**: 椭圆弧长覆盖 <60% → 保守曲线标签 — 内置
  EllipseDetector 对 1/8 椭圆弧硬拟合残差 ~1e-16 会误标 "椭圆 a=.. b=.."
  (判别性测试显式证明), 插件在注册表前端拦截。
- **覆盖足够的开放椭圆弧** (≥60%) 让位内置 (旧行为不变); 直边/闭合链
  让位内置探测器。
- **优先级规则 (写进手册)**: arc_curvature (开链, 保守) → ellipse_group_
  label (闭合整椭圆) → line → circle → ellipse → general; 标签冲突以
  更保守者胜。

### 工程 (挑剔点执行)

- **detectors.py 拆分为包** (纯搬移提交, 金标准逐位不变验证):
  `detectors/` = base + line + circle + ellipse + general + registry +
  _shared 7 模块; 外部 import 面不变。
- **注册即生效**: 三插件经 `plugins/__init__.py` 在 import 链默认注册
  (非测试内手动注册); 判别性测试全部跑在生产默认管线。
- **金标准快照更新** (5 文件, 变化类型 = 标签层, 逐 diff 审查):
  demo_complex (5 弧标签 → ρ/圆心), curved_beam (16 弧标签 + 3 类圆→
  圆弧精确升级, 跨度恰 ≥5°; 17 短弧 <5° 保持类圆, 与内置下限一致;
  同环段序随标签排序微调 — 编号重排属锁定决策, @组名不变通道),
  multi_hole (72 弧标签); 边界边集合/法向/载荷路径逐位不变 (结构化
  校验: 计数/边集/法向一致, 段节点不变)。
- **契约探针**: G3 组 10 项 (三插件 + @组名) — 探针合计 146, 0 FAIL;
  api_contract.md G2 节新增 6 行契约。
- 全量 pytest 1051 passed 0 failed; 微尺度 (R=1e-16 圆弧标签科学计数)
  回归测试 token 更新 (R= → ρ=, 意图不变)。

## 9.23.0 (2026-08-04) — 边界层插件化重构 (阶段 3: 示例插件 + 接入手册)

### 阶段 3 — 可扩展性验收 (证明"便于更新")

- **示例插件 `fem2d/boundary/plugins/circle_label.py`**: Gmsh 原生
  Circle 实体圆标签探测器 — 委托上游 CircleDetector 判定, 改写标签
  (" [Gmsh 原生圆]") + info 键 (native_circle=True); 纯展示层
- **判别性测试 `tests/test_boundary_plugin_circle_label.py`**: 单元
  契约硬编码期望 (放回旧实现必红) + 未注册时管线不变 + 注册 1 行后
  真实 gmsh 管线生效 (4 条 90° 弧合并整圆段带标记, Line 段不受影响)
- **注册语义修正**: register_detector 插入注册表前端 (classify 短路
  于首个非 None 判定 — 追加到末尾的插件被 general 兜底永久遮蔽);
  类型校验前置 (非 Detector → TypeError)
- **`docs/boundary_plugins.md`**: 五步接入手册 — 接口签名 / 注册 /
  标签输出 / 判别性测试 / 快照更新
- **契约探针**: G2 组 7 项 (注册表顺序/基类接口/classify/插件接口/
  注册生命周期/去重/类型) — 探针合计 129 → 136, 0 FAIL
- 全量 pytest 0 失败; ruff/mypy/vulture/compileall 全过



### 阶段 2 — 显式管线 + 识别器注册表 (行为逐位不变)

- **新增 `fem2d/boundary/detectors.py`**: 识别器注册表 —
  `Detection` (类型/参数/标签/置信度/残差) + `Detector` 接口 (点链 +
  可选原生实体信息 + 尺度 → Detection|None) + `DetectorRegistry`
  (有序, 首个非 None 胜出, 兜底 general 恒返回) + `default_registry`
  单例 + `register_detector` 插件接入点
- **注册表顺序 = 旧 classify 探测顺序**: line → circle (整圆/圆弧,
  段类型 arc) → ellipse (整环/开放, 段类型 ellipse) → general 兜底;
  分类判定门槛与标签文本逐位迁移 — 金标准快照逐位一致验证
- **原生实体信息一等公民**: Gmsh 实体类型 (line/circle/ellipse/
  bspline) 经 segment_builder/conic_merge → classify native_entities
  参数传入探测器不丢失 (内置探测器不消费, 插件接口用)
- **段 schema 稳定化**: `validation.validate_segment_schema`
  (type/nodes/coords/label/info 必要键 + type 受控枚举), 在
  build_boundary_segments 管线出口执行; validate_boundary_segments
  保持部分 dict 覆盖检查契约不变
- **测试**: `tests/test_boundary_detector_registry.py` 12 项 —
  注册表顺序/接口契约/优先级/生命周期/原生实体信息管线传递 (真实
  gmsh 网格 spy 探测器)/schema 门禁位置契约
- 全量 pytest: 1027 passed 0 failed; ruff/mypy/vulture/compileall 全过



### 阶段 1 — 边界输出行为锁定基线

- **新增 `tests/boundary_snapshot.py`**: 边界金标准共享工具 —
  段集合/边界边集合/段定向法向 vs `boundary_outward_normal` 一致性/
  `print_segments` 打印输出四维快照, JSON 逐位对比入库金标准
  (`tests/boundary_golden/`, `FEM2D_UPDATE_GOLDEN=1` 重写)
- **新增 `tests/test_boundary_golden_deterministic.py`**: 无 gmsh 依赖层 —
  CST 三角形/CST 方板/Q4/Q4R/Q4I 方板/嵌套孔板/32 边形整圆/40 边形
  椭圆 8 个模型 × 双入口 (detect == build_boundary_segments 逐位一致)
- **新增 `tests/test_boundary_golden_gmsh.py`**: gmsh 层 (skipif 无 gmsh) —
  demo_complex/multi_hole_plate/curved_beam/l_bracket 四模型 ×
  自动检测/注册表双路径快照 (节点 ID 版本漂移面改为计数+语义锁定)
- **判别性测试**: `test_discriminative_hardcoded_expectations` 显式
  钉死 Q4 方板标签序/整圆闭环节点序/椭圆标签 — 识别器任何重排/
  翻转/改标签必红

 — 并行批次 1+2 (pkg7 文档 + pkg8 工具 + pkg9 尺度 + pkg10 CLI + pkg11 去重 + pkg12 测试基建)

### pkg7 文档数字统刷 + 交接归档

#### 文档数字全面实测刷新 (数字均来自实测输出, 非猜测)
- **PROJECT_SUMMARY / README / docs/ci.md**: 版本 9.17.0→9.21.1、测试数、
  文件数、模型数全部按 2026-08-04 实测刷新:
  - 有 gmsh 本机: **939 collected → 937 passed + 2 skipped, 0 失败**
    (2 skip 恒为 test_geo_models 无捆绑 gmsh.exe 项)
  - 无 gmsh 全新 venv 实测: **918 collected → 898 passed + 20 skipped, 0 失败**
    (numpy 2.5.1 / scipy 1.18.0 / matplotlib 3.11.1 / pytest 9.1.1)
  - tests/ 74 个测试文件 + conftest.py; models/ 21 .geo + 7 .spec + 3 .txt
    (原 5 .txt 为过期值); scripts/ 12 .py + 1 .sh (原 8 个为过期值);
    fem2d/ 30 模块 + boundary/ 13 + element/ 7 = 50 .py (原 26 为过期值)
  - docs/ci.md §3 skip 明细表: 20 项全部实测列明, 新增
    test_output_dir_policy.py:439 (9.21.0 --output-dir 用例) 一项;
    无法复测的数字 (无 gmsh 环境/ numpy 地板组合) 改为
    "以实测为准 + 复现命令" 表述, 不再写死
- **docs/api_contract.md** C 节标题修正: `fem2d/loads/__init__`
  (该包不存在) → `fem2d/loads.py, fem2d/loads_core.py`

#### 交接归档
- `NEXT_SESSION_HANDOFF.md` git mv → `NEXT_SESSION_HANDOFF_ARCHIVED.md`,
  文件头标注 "历史交接（2026-08-03），已被 COMMANDER_HANDOFF_20260804.md 取代",
  并清除其中过期版本/测试数字 (9.17.0 / 504 等)
- `COMMANDER_HANDOFF_20260804.md` (主会话生成, 此前 untracked) git add 入库 —
  仓库根唯一有效交接文档

#### 配置
- .gitignore 补 `.venv*/` (.coverage / wt_* / models/*.msh / __pycache__ 已覆盖)

#### 验收
- 全量 pytest **939 collected → 0 失败** (937 passed + 2 skipped); 纯文档改动,
  无代码/测试文件被触碰; CHANGELOG 历史节零改动

### pkg8 审计工具自证力补强

#### 探针 (scripts/audit_contract_probe.py): 101 → 129 项
- **assemble_loads 期望收紧**: 实测当前行为已抛 ValueError (9.19.0 修过
  n_dof 裸 IndexError), 期望从 (IndexError, ValueError) 收紧为只 ValueError —
  判别性: 注入裸 IndexError 回归 → 探针 FAIL (smoke 锁定)
- **补缺口**: G 组 boundary 6 契约行全覆盖 (detect / build / validate /
  describe+print+parse_edge_name 歧义模糊 / physical_curves 未映射 /
  region_registry / semantic_coverage); E 组 resolve_input_file (.inp/.xyz
  拒)、resolve_spec_overrides、generate_from_geo、parse_spec_config、
  parse_geo_fem_config、read_geo_groups、validate_mesh; I 组 assemble_*;
  H 组 evaluate_mesh_quality / run_patch_test (E/plane/elem_type);
  F 组 register_element; estimate_error 非 dict 结果
- **覆盖声明非纸面**: docstring 附组↔探针数对照表 (合计 129, 可 AST 核对),
  smoke 测试断言声明数与实际 probe() 调用数一致

#### combo_fuzz (scripts/combo_fuzz.py)
- **静默零解判定**: 曾只查 isfinite, 全零解通过 — 新增相对尺度
  u_char = max|F|/median|K_diag| (量纲=长度, 微尺度自动跟随, 无绝对阈值),
  max|u| ≤ u_char×1e-10 → 报"静默零解"。判别性: 注入"solve 载荷装配丢载荷"
  → 60/60 组合全报; 旧版 (isfinite-only) 同场景 0 检出
- **载荷边固定取右边自由边**: 曾取 boundary_edges[0] (任意边, 可能恰是
  固定边 → 工况静默改变)
- **平衡检查**: Σ(K·u) ≈ 0 (外载荷+支反力自平衡), 相对 Σ|F| > 1e-6 → 报
  "平衡残差" (载荷方向/位置错误时 isfinite 与零解检查均通过)

#### fuzz_api (scripts/fuzz_api.py): 值类别非法输入不再整体豁免
- 14 个 silent_ok=True 改为**按生成值过滤**: 仅该值确实合法 (有限实数对
  压力/位移值, 有限正数对 E, ≥1e4 对罚因子, 0..3 对 nid, (2,) 对
  spr_recovery 等) 才允许静默成功; complex/NaN/str/容器等非法类别照常
  断言必须抛异常。判别性: API 静默接受非法输入 → fuzz 必报 (smoke 锁定)
- **收紧暴露真 bug 并修复**: fix_nodes_func('') 空串迭代零次 → 静默 no-op
  (约束消失无提示) → mesh.py 增加 str node_list 守卫 ValueError (最小修复);
  放回旧实现 → fuzz 500 轮报出, exit 1

#### 测试
- 新增 tests/test_tools_smoke.py (11 项): 探针/fuzz_api/combo_fuzz/
  regression_compare/perf_benchmark 各"运行干净 + 已知 FAIL 必被抓出"
  (注入模式参照 test_static_checkers_smoke 断裂 import); 全部无 gmsh 依赖
- 全量 pytest: 939 → **950 collected, 948 passed, 2 skipped (无 gmsh 环境
  skip), 0 失败**; 探针 129 项 0 FAIL; fuzz 500/2000 轮 0 problems;
  combo 60 组合 0 problems; ruff E/F 干净

### pkg10 CLI/交互修正

CLI 是学生第一接触面: 退出码矩阵补全、Q4R 警告覆写分叉、绘图资源泄漏、
Ctrl-C 交互承诺、色条教学标注。

#### 退出码矩阵 (用户错误 1 / 内部错误 2)
- **--plane 与网格单元码冲突 → 1** (曾裸 ValueError 冒泡到顶层 except
  Exception 归为内部错误 2); 与 --elem-type 不兼容同属用户错误, 矩阵测试锁定
- **config 校验失败 (from_args) → 1** (曾 2): 非法参数组合如 --band-min
  缺 --band-step 是用户错误 — 与退出码矩阵一致; test_main_config_value_error
  更新为 1
- **solver 刚体模态拒绝仍为 2** (跨包, 仅报告): 欠约束模型属用户错误,
  建议下一轮归入 1 — solver.py 在冻结区

#### 警告与交互
- **Q4R 专用提示判据改 kernel.name** (曾查 elem_type ∈ CPS4R/CPE4R):
  --elem-type Q4R 覆写后 elem_type=="Q4R" 使警告静默消失; 两条入口
  (spec 自带 CPS4R / CLI 覆写 Q4R) 判别性测试都须出现警告
- **ask 捕 KeyboardInterrupt → CliError(0)**: 向导 banner 承诺"随时
  Ctrl-C 退出"曾只捕 EOFError, Ctrl-C 泄漏整段 traceback 退出码 130;
  与 EOF 分支对称, 主流程捕获后干净退出
- **main 顶层接 KeyboardInterrupt → 130** (审查轮补): 求解/解析中途
  Ctrl-C (非 ask 内) 曾泄漏整段 traceback — 向导承诺覆盖全部阶段
- **--band-min/max/step + --no-plot 打印一行提示** (曾 config 校验后静默
  忽略, 参数看似生效实未生效); 独立自检 (--self-test 无网格) 同样 WARN
  (审查轮补)

#### 绘图资源与标注
- **plot_three 非 save 路径关闭 figure**: Agg 后端 (CI/测试) 曾每次调用
  静默累积一张图; 交互循环按 1-12 切换前先关旧图 — 覆盖全部非 show 分支
- **plot_contour gouraud 显式 location 短路推断**: location='node' +
  n_nodes==n_elements 网格曾把节点标量误判为单元数据抛 "unreliable"
  ValueError (K4 网格判别性测试)
- **统一色条标注数据来源与峰值**: flat → `[element, max=...]`,
  gouraud → `[node, max=...]` (CST 磨平峰值降 ~17% 教训 — 学生一眼看出
  显示的是单元值还是节点值), 与 isoband 的 [bands: ...] 风格对齐
- **_to_node 权重形状不符 WARN 而非静默回退** (静默替换恢复方法曾让学生
  误以为 weighted 生效)

#### 死负载清理
- **_import_mesh 收窄为 5 元组** (曾 7 元组含 edge_labels/sibling_geo 两个
  恒 None 死输出字段); _build_model 解构点同步

#### 测试
+14 判别性测试 (矩阵 plane 冲突 / Q4R 双入口 / Ctrl-C ×4 / Agg fig close /
K4 节点定位 / 色条标注 / WARN 回退 / band 提示 ×2 / _import_mesh 收窄 /
config 退出码更新)。全量 pytest 951 passed 2 skipped 0 失败; 冒烟
(真实 .msh 输入 + --save) 正常。gmsh.exe 未随 worktree 分发, .txt/.geo
子进程链路冒烟受环境限制 (与打包约定一致: gmsh 可执行文件另行分发)。

### pkg9 尺度不变收尾 (pkg9_micro)

#### 修复
- **fit_closed_ellipse closure_tol 移除 1.0 物理尺度下限** (geometry.py):
  微尺度模型 (跨度 ≲2e-13) 整环首末顶点间距恒 ≤ eps×1.0×32 ≈ 7.1e-15,
  被误判"重复闭合点" → 静默 `coords[:-1]` 截掉末顶点, 被截顶点的拟合
  残差不再被验证 (primitive_samples 63/64)。span 改为
  `max(ptp_x, ptp_y, np.finfo(float).tiny)` — 与 `_segment_is_closed`
  同款 ULP 相对化。
- **_classify_general_curve 曲率阈值相对化** (geometry.py): 绝对
  1e-8/1e-14 (量纲 1/长度) 改为 `1e-8/characteristic` 与
  `1e-14/characteristic` (参照 segment_by_curvature 同款模式)。
  大坐标 (1e12 级) 平滑曲线 κ<1e-8 曾降级成"通用曲线"、拐点漏计;
  相对化后分类与拐点计数与模型尺度无关。R_max 兜底 1e9 同步改为
  `1e9×coordinate_scale`。
- **topology.py 两处 1.0 floor**: `_closed_conic_segment` 传 classify
  的模型 scale (190-191) 与 `_general_position_ray` 的 magnitude/margin/
  perturb (356-360) — 全部改为 `np.finfo(float).tiny` 兜底。后者曾使
  微尺度 (跨度 <1) 模型的射线外推量/扰动被抬到 1.0 与 eps×1.0,
  比多边形边长 (1e-15 级) 大 13+ 个量级。

#### 判别性测试 (tests/test_micro_scale.py 新增 11 项)
- 微尺度圆环 (跨度 1.2e-13) 与 1e-150 深档: 闭合检测不误判、
  末顶点不被截 (primitive_samples=64, 旧 63)、残差路径全环验证
- 大坐标 1e12 级 S 曲线/浅抛物线: 不降级成"通用曲线"、拐点不漏计
- 四档尺度 (1e-150/1e-13/1e0/1e12): 分类、拐点数、κ̄×span、
  R_min/max×span 比例恒定
- 微尺度圆环 detect_boundaries + solve 全流程端到端
- 判别性确认: revert 三处修复后本文件 7 failed / 4 passed

#### 验收
- 全量 pytest 948 collected 0 失败; ruff 干净
- regression_compare 10 组合 (5 模型 × 消去/罚函数) 改动前后
  **逐位一致 (相对差 0.0)** — 正常尺度模型行为未受影响

### pkg11 重复实现/契约清理

#### A 组 重复校验/重复实现 (判别性测试全部在场)
- **solver_key 单点校验** (solver._solve_linear_system): 逐字重复两遍的
  白名单校验合并为入口一次, cg-block 别名映射只发生一次, 三个分支
  复用归一键; 非法 solver 名 ("spqr") 在纯 Dirichlet/penalty 分支入口
  同样拒绝 (判别性测试)
- **L2 投影共享内核** (stress.nodal_L2_projection): 提取
  `_l2_batch_assembly`/`_l2_stress_qp`, batch (共享规则广播) 与
  uniform (逐单元堆叠) 两路径只构造参数; 逐单元回退分支复用同一
  广播助手 (消除第三份副本); 两路径与逐单元参考**逐位一致**
  (np.array_equal 判别性测试)
- **canonical_edge 去重** (boundary/naming): 内联 `_canon` 删除,
  直接用已导入实现; numpy int64/反向边判重行为测试锁定
- **_signed_area 复用** (preprocess): 蝴蝶形四边形两片三角分解复用
  同文件 `_signed_area` (曾逐字重复); 正净面积自交四边形判别性测试
- **gmsh 会话统一生命周期** (gmsh_adapter/input_source): 新增
  `_gmsh_session()` 上下文管理器, 4 处 isInitialized→initialize→
  finalize 样板收敛; 外部已初始化会话不代为 finalize (禁止双重
  finalize, 顺序测试在场); 实测 run.py l_bracket.spec 与 Gmsh API
  自管/复用两条路径会话语义正确
- **批量常量引用** (error_est): 硬编码 50000 → assembly 的
  ASSEMBLY_BATCH_ELEMENTS 共享常量
- **收敛速率向量化** (convergence): per-level ku/ks/ke 三块逐字重复
  堆成数组一次向量化 (与独立标量参考逐位一致, 判别性测试在场)
- **无意义 lambda 内联** (error_est): `both_fixed = lambda d1,d2: d1 and d2`
  直接展开为 dof 元组判定

#### B 组 死代码/无效防御
- **patch_test**: `bdy_nodes` 首次计算结果从未使用 → 删除; 模块内
  `sys.path.insert` 死操作 → 删除 (`python -m fem2d.patch_test` 是文档
  运行方式, 根目录已在 path)
- **bc_apply**: `if idx >= len(segs): continue` 死守卫删除 — resolver
  契约保证索引恒在界内, 静默 continue 曾把内部 bug 吞成"面力漏施加"

#### C 组 API 形状/错误机制统一
- **estimate_condition 返回形状统一** (solver): 成功/失败路径键集一致
  (失败补 lambda_min/lambda_max/digits_lost=None, 成功补 error=None);
  reporting 消费方 `.get()` 兜底不变, 行为锁 golden 测试同步
- **regions.by_name 非法维度** `dimension=5` 曾裸 KeyError → 带参数名
  ValueError (0/1/2/None 契约, 判别性测试)
- **material.D_matrix** 平面应变 ν>0.45 告警 print(stderr) → 统一
  warnings.warn(RuntimeWarning, stacklevel=2) (库调用方可过滤/捕获,
  判别性测试)
- **input_source 模块顶 sys.path.insert 移除** → CLI 入口 (runner.main)
  注入项目根 + `_import_scripts()` 惰性注入; 库用户 import 零副作用
  (reload 判别性测试改写为新契约)
- **reconfigure_streams 共享** (errors.py): run_demo.py 与 runner.main
  的 7 行编码安全网收敛为同一函数 (两种入口行为锁定测试)

#### D 组 docstring 与实现对齐 (教学文档)
- error_est.estimate(): 算法描述改为实际积分点公式 (Σ_q w_q·(σ*−σ_h)ᵀ
  D⁻¹(σ*−σ_h), 无 A_e 项, 归一化积分权 + 尺度乘回)
- stress.principal_stresses(): 补第 4 返回 θ = 0.5·atan2(2τxy, σx−σy)
- preprocess.validate_mesh(): tol=None 语义同步为
  max(min(非零单元跨度)×1e-6, 坐标 ULP) (曾声称 min(edge_length)×1e-3)

#### 验收
- 全量 pytest **0 失败** (含 A 组判别性测试); 回归对照
  (elimination/penalty) **相对差 0.0** (字符级一致)
- 冻结区零改动: solver 数值路径 / error_est 公式 / element/ 未触碰

### pkg12 测试基建收敛

#### conftest 公共 fixture (tests/conftest.py)
- **gmsh 可用性守卫**: `GMSH_AVAILABLE` 常量 + `gmsh_available` skipif fixture
  (ImportError + OSError 双捕, 语义与 test_msh_import_audit 原 skipif 块一致);
  统一 6 处重复的 `try: import gmsh` + 模块级 skip 块
- **mesh_from_geo / mesh_result_from_geo**: 临时 .geo → generate_from_geo 的
  4 份同构实现 (test_boundary_complex/gmsh/stress/highpressure) 收敛; 3 元组
  版 + 完整结果版 (含 regions)
- **square_mesh / quad_mesh fixture**: 2×2 单位方板 CPS4 (节点 CCW), 对应
  各测试文件 _square_mesh/_quad helper 语义, 供后续迁移消费

#### 6 文件脚手架迁移 (纯搬运, 断言/期望值零改动)
- test_boundary_complex / test_boundary_gmsh / test_msh_import_audit_20260803 /
  test_physical_point_resolution / test_boundary_highpressure / test_boundary_stress:
  模块级 gmsh skip 统一为 `pytestmark = skipif(not GMSH_AVAILABLE)`; 临时 .geo
  脚手架改调 conftest helper; test_physical_point_resolution 的 _square_mesh
  改 square_mesh fixture
- 语义提示: 4 个文件曾用模块级 `pytest.skip(allow_module_level=True)` (无 gmsh
  环境不收集), 现统一为 skipif (收集但跳过) — 无 gmsh 环境的 collected 数将
  增加 (docs/ci.md 的 no-gmsh 数字需以实测刷新, 属 pkg7 域, 仅报告)

#### 路径卫生 (仓库外任意 cwd 跑 pytest 0 失败)
- test_regression_audit_20260802 (2 处) / test_regression_audit_20260803 /
  test_config / test_boundary_complex demo_complex / test_regressions (2 处):
  `open("models/...")` / 相对路径 → `Path(__file__).resolve().parents[1]` 绝对
  定位 + tmp_path (写入类); test_msh_import_audit `_gmsh_exe` tools/ 相对路径
  绝对化
- 判别性: revert 旧实现 + 从仓库外跑必失败 (test_regressions/test_regression_audit_20260802
  实测 FileNotFoundError)

#### pyproject.toml
- `[tool.pytest.ini_options]`: `--strict-markers` + markers 登记 (parametrize/
  skip/skipif/xfail 显式列出) + `xfail_strict = true` (现有 0 xfail, 无违规)
- `[dev]` 补 `pytest-cov>=4`: docs/coverage.md 的 `--cov=fem2d` 在全新
  `pip install -e .[dev]` 可直接复现 (实测本机 pytest-cov 7.1.0 运行正常)
- filterwarnings 维持现状 (glyph 豁免不扩大)

#### scripts/_test_helpers.py 不建 (按 PROMPT 规则)
- scripts/ 5 份 _mesh/_solved/_ROOT 注入块无测试消费方 (tests 只子进程执行
  脚本模块级代码或导入 _classify 等真实函数, 不导入 helper) — 留待 scripts
  出现测试消费方时再抽

#### 验收
- 每步迁移后全量 pytest **939 collected → 937 passed + 2 skipped, 0 失败**
- 从 `C:\Users\35666` (仓库外) 跑 `python -m pytest <项目路径>`:
  **937 passed, 2 skipped, 0 失败** (判别性: 迁移前该命令在 test_regressions
  /test_regression_audit_20260802 必失败)
- `--strict-markers` 全量通过 (无未登记 marker); xfail_strict 无违规
- 迁移后文件 grep: 无重复 gmsh skip 块、无 `open("models/` 相对路径
- 判别性: revert 旧脚手架 → 全套仍全绿 (证明纯搬运)

## 9.21.1 (2026-08-04) — 审查 9.0 轮收尾

### 发版阻断修复
- **压力 callable 单元素数组校验版本无关化**: float(np.array([x])) 的弃用转换
  随 numpy 版本变化 (2.3.5 仅警告仍转换 → 单元素数组被静默接受; 2.5.1 抛
  TypeError) — loads_schema._load_component_ok 显式按 ndim 处理: 0 维 .item(),
  1 维及以上拒绝, 契约全版本一致

### 审计工具加固
- probe(): expect 为元组时 expect.__name__ 崩溃 (AttributeError) → _expect_label
- fuzz_api: feed 增加"非法输入静默成功"判定 (曾只查异常不查静默接受);
  逐 case 标注合法值 (负压力/合法 nid/域外 -1/契约允许/一维单分量恢复等),
  500+2000 轮 0 problems; apply_penalty case 传参位置修正 (v 曾误传 fixed_dofs)

### 打包卫生
- zip 排除 .git (18MB) 与 .coverage (本机绝对路径) — 审查建议

### 验收
全量 pytest 0 失败; 探针 0; fuzz 500/2000 轮 0; ruff 干净。

## 9.21.0 (2026-08-03) — 第二轮五路并行 + fuzz 收紧闭环

### 五路并行 (审查 8.8 分轮)
- **组 A**: 压力 callable 返回非法类型裸异常 → 统一标量校验 + 边号/Gauss 点上下文; ast.Num 死代码分支删除
- **组 B**: mesh prescribed_vals / nodes_on_edge(tol) / bc apply_penalty(penalty) 绕过统一校验器 → 收敛 require_finite_scalar 等 helper
- **组 C**: q4r 沙漏系数非法值校验; error_est sigma_ref 字符串; compute_traction_jumps 参数校验先于空数据提前返回 (单单元网格静默接受非法参数修复)
- **组 D**: 自检失败污染成功缓存 (patch_checked 先加后验 → 只在 all_passed 后缓存, 失败后同进程重试必须仍失败)
- **组 E**: 审计脚本 sys.path 根目录注入统一 (按文档直接运行); fuzz_api 捕获过宽失真 → 只接受预期异常 + 消息非空; test_output_dir_policy 可移植性 (monkeypatch 代替 chmod / 捕获 (ImportError, OSError))

### fuzz 收紧暴露并闭环
- point_in_element/stress_at_point 坐标 inf/1e308 曾冒裸 OverflowError (kernel int 转换) → 坐标有限性 ValueError 带上下文 + AABB 域外快筛 (有限域外 → -1, 合法路径不变)
- 收紧后 fuzz 500 轮 problems: 0 (原 5 个全部闭环)

### 测试
全量 pytest 0 失败 (含 5 组新增判别性测试); 双冒烟; 探针 FAILS:0; ruff 干净。

## 9.20.0 (2026-08-03) — 四路并行优化

### CI (opt_ci)
- 修 numpy==latest 无效版本 / 矩阵与 pyproject 冲突 / 缺 `pip install -e ".[dev]"`
- 依赖矩阵统一 + 本地 dry-run 验证 + docs/ci.md

### 覆盖率 (opt_coverage)
- 83% → **93%**（+199 判别性测试; runner 100% / input_source 达标; 拒绝恒真断言）

### .geo 链路收尾 (opt_geo_out)
- **--output-dir**: 生成物写入指定目录 (不存在自动创建); 默认行为不变
- 只读目录预检 → 清晰错误 ("请用 --output-dir 指定可写位置"), 发布时刻 OSError 二次兜底
- 同名 .msh 覆盖保护: 本程序生成物覆盖; 来源不明 → WARN + 临时副本 (与手写 .geo 保护同模式)

### 性能 (opt_perf)
- L2 恢复批量堆叠 (CST/Q4R): 300k 单元 12.8s→4.2s (~3×), **逐位一致**
- docs/performance.md 规模-时间表 + scripts/perf_benchmark.py

### 验收
- 全量 pytest **0 失败**; 双冒烟; ruff/死代码干净
- 数值漂移: 优化前后 (4581971 vs 合并后) elimination/penalty 双求解器**逐位一致**
- 契约探针 100 项 0 失败; fuzz 500 轮 0 裸异常

## 9.19.0 (2026-08-03) — API 契约清账 + 复查轮 + 终轮回归对照

### 契约清账 (3 轮)
- **docs/api_contract.md**: 全部公共 API 的契约表 (误用清单 → 应有错误行为 → 状态),
  69+ 项 ✅ 全部带判别性测试锁定; 0.5 验收回填 + M 节复查记录 + M2 测试映射表
- **校验收敛**: 节点索引/DOF/载荷形状/标量有限性收敛为共享 helper
  (fem2d/loads_schema.py 纯搬移, bc_apply 直引), 错误消息统一
  "函数名: 参数名=值 — 原因, 期望"
- **import 去环**: 11 处函数内 import 提升到模块顶 (5 处保留并记录原因)
- **四入口端到端测试**: .txt/.geo/.spec/.msh 各覆盖 (spec 回归教训落地)
- **fuzz**: 单 API 1000 轮 + 组合 60 组, 抓到 estimate_condition 拼错 method
  静默降级 (修: 白名单 ValueError) 等真缺口

### 终轮数值漂移对照
- **10 组合 (5 模型 × 消去/罚函数) 早期 baseline vs main 逐位一致 (相对差 0.0)** —
  全部修复未引入任何数值漂移 (docs/regression_comparison.md)
- 教学用户 8 场景报错质量评分, 修 .txt BC 误写键 (响亮错误 + 边键写法提示)

### 修复汇总 (本版本)
estimate_condition 静默降级/K 非数组 / element_refinement_indicator 缺键 KeyError /
assemble_loads n_dof 裸 IndexError / .txt BC 误写键 / .spec 扩展名重算回归 (9.18.1)

### 测试
569 → **676 passed** (+107 判别性测试, 0 失败)。全量 pytest + 双冒烟 + ruff + 死代码全绿。

## 9.18.0 (2026-08-03) — 第十轮外部审查修复

### P1
- **Physical Point 统一严格判域**: gmsh_adapter 回退路径曾只做 AABB 检查 —
  孔心/凹域缺口点 (AABB 内但不在任何单元内) 回退到最近节点, 集中力静默
  施加到材料域外。现在与 input_source 一致, 回退前用 point_in_element
  判域, 域外拒绝 + 提示改用边界曲线 (孔心判别性测试, occ 内核构造)
- **error_est math.exp OverflowError** (包 3 引入回归): 全有限输入
  (E=1e-150/t=1e150/σ=1e308) 时 log 和 > log(float_max), math.exp 抛异常
  (不受 np.errstate 控制) → 显式 log 阈值分支, 超出双精度诚实返回 0/inf

### P2
- **mesh 构造器裸 IndexError**: nodes/elements 传标量时 0 维数组
  shape[0] 越界 → 先验维度, ValueError 带上下文
- **fix_nodes_func 裸 IndexError**: 越界/负 nid 先索引后校验 →
  范围检查前移 (负 nid 曾静默约束最后一个节点)

### P3
- `.geo/.msh/.spec/.txt` 扩展名判断大小写不敏感 (Windows .MSH 曾被拒;
  注意"最终需 .msh"检查对象是 resolve 后的路径, 非输入扩展名)
- `--self-test --list-boundaries` 文案诚实化 (曾声称"自检不执行"但照常执行)
- `physical_point_from_geo` 大 except 收窄: 内部逻辑错误不再误报
  "Gmsh 不可用" (仅 gmsh 会话失败归因 gmsh_unavailable)
- ruff F811: elem_type dataclass 字段声明与只读 property 同名冲突 → 移除
  字段声明 (构造参数在自定义 __init__)

### 测试
+7 判别性测试 (error_est 溢出 / mesh 标量 ×2 / fix_nodes_func ×3 /
孔心拒绝)。全量 pytest 0 失败。

## 9.17.0 (2026-08-03) — 高强度整修 + 外部审查修复

### 核心数值
- **微尺度绝对阈值家族清除** (~25 处): `max(...,1.0)` 下限、固定 `1e-15/1e-30`
  地板全部改为相对尺度/坐标 ULP/`np.finfo(float).tiny` — 1e-16 级模型
  (边界分段、patch test、应力恢复、载荷、网格质量) 不再被误杀或静默失真
- **单元验证路径修复**: `cst._verify_element` 对退化 (共线) 单元曾抛裸
  ZeroDivisionError → 零面积提前返回失败 (覆盖率分析发现)
- **Q4/Q4I 标量路径大坐标 Jacobian**: 1e12 偏移刚度偏差 ~1e-4 → 首个节点
  居中化 (实测 0 偏差)
- **全装配路径加权一致性** (第四轮外部审查): 加权顺序 (先 B̃=√t·√detJ·B
  再二次型) 曾只覆盖生产批量路径 — 标量 `element_stiffness` 四单元族、
  Q4R 批量路径、参考装配 (lil_reference/expand) 仍旧顺序, L~1e-150 时
  BᵀDB ~ E/L² 溢出; 统一后 1e-150 全路径有限, 正常尺度批量 vs 标量
  一致到 2.3e-16。CST 批量 `√(t·A)` → `√t·√A` 防乘积先下溢
- **isoband 末带静默加宽一倍** (P1): 浮点除法 floor 截断 → round + 尾点归位
- **solver 诊断**: trivial 误标 / 小变形检查 1e-30 (跨度 1e-31 失效) /
  平衡检查残差标签 — 修复并相对化

### 边界系统
- **微尺度圆弧误判闭合** (外部审查 Bug 1): `_segment_is_closed` 1.0 下限 →
  ULP (R=1e-16 开放圆弧曾判闭合)
- geometry.py 4 处绝对阈值 (1e-15/eps*10) → 坐标 ULP; `_validate_nodes`
  1.0 下限; `sharp_corner_indices` 闭合重复点 (八边形 7/8 角)
- `cad_boundary_complete` 只认 Physical Curves → curves or cad_curves

### 输入端
- **交互式建模向导新增** (P0-0): `fem2d/wizard.py` + `--wizard` + 无参数
  自动进入; 与 .txt 同内核 (spec dict → generate_geo → @FEM)
- 四套入口行为收敛: `parse_spec` 同名双实现改名消歧; `is_batch_mode` 统一
  (3 处); `.txt` 双分量面力; `.spec` 格式错误响亮化; 手写 .geo 覆盖保护;
  `{:.6f}` 精度塌缩 → `{:.17g}`
- **BC 公共 API 校验** (外部审查 Bug 2): 布尔掩码/负 DOF/重复异值/penalty
  NaN-Inf 全部拒绝

### 测试
- 374 → **496 passed** (+122): 12 个假通过测试判别化, 2 轮 8 agent 复扫
  23+27 项发现, fuzz 1150+ 轮 0 崩溃, Kirsch 经典验证 (K_t=3.04),
  Cook membrane 收敛, 单元验证路径, 向导 22 测试, 第四轮审查
  (全装配路径加权 +10 c / Z2 归一化 + BC 校验 + 标签 +7 d /
  6 个模块级边界测试改标准测试 +31 / 向导 n 文件污染修复),
  第五轮审查 (.geo 严格解析 / Ke 对称检查 / run_demo 孔压断言 /
  主应力稳定 / 高复杂度函数拆分 +7 e)

### 工程
- 死代码清理: `auto_classify_segments` / `_ask_str` / ULP 双实现合并
- 消毒逻辑去重 (gmsh_runner.sanitize_geo_source 唯一实现)
- Gmsh 失败测试契约 (外部审查 Bug 3): tmp_path + skip + None 防御
- Bandit 2 Low (故意吞异常) 加 nosec 标注; ILU-CG 限制文档化
## 包 5 — 架构与可维护性 (2026-08-03, 未发布, 版本号未变)

### 1. 库层 sys.exit() 迁移 (完成)

fem2d/ 包内最后的进程退出全部收敛为领域异常, CLI 层统一转换退出码:

- `runner.py` 7 处 → `CliError(exit_code=1)`: patch test 失败 /
  `--elem-type` 与网格不兼容 / 边界构建 ValueError /
  `--require-physical-groups` 三处 (无语义 / 未映射 / 存在错误) /
  平面材料验证失败。原 `[FATAL]` 文案进异常消息, main 捕获后输出不变。
- `wizard.py` 5 处 → `CliError`: EOF 干净退出 ×2 与取消建模 →
  `exit_code=0` (优雅中止非错误); 文件不存在 / 几何生成失败 → `exit_code=1`。
- `verification.py` 的 `__main__` 守卫保留 — 脚本独立运行语义, 与
  `scripts/` 工具一致。
- **退出码矩阵 (正常 0 / 用户错误 1 / 内部错误 2) 逐场景锁定**:
  `tests/test_exit_code_matrix.py` (10 场景) 在迁移前以 SystemExit 形态
  验证通过, 迁移后以 CliError→int 形态验证通过 — 进程退出码逐位不变。
- 嵌入方 (Jupyter/测试) 现在得到可捕获的 `CliError` 而非进程自杀。

### 2. 高复杂度函数重构 (行为逐字节锁定)

- `boundary/topology._point_in_loop` 环复杂度 **27 → 11**:
  拆为 5 个纯函数 (顶点去重 / 环边构建 / 边界命中 / 一般位置射线奇偶计数 /
  半开穿越兜底)。锁定测试含冻结的旧实现快照 + 固定/随机输入电池
  (`tests/test_refactor_point_in_loop_lock.py`), 覆盖边界/顶点/退化/
  微尺度 1e-150 / 大坐标 1e12。
- `solver.solve` 环复杂度 **22 → 5**: 拆出 6 个阶段函数
  (`_partition_dofs` / `_check_rigid_body_constraints` /
  `_q4r_aspect_ratio_warning` / `_compute_element_response` /
  `_hourglass_monitor` / `_condition_report`), solve 成为纯编排。
  锁定测试: 9 模型 (CST/Q4/Q4R × 消去/罚函数/CG/条件数 × 微尺度 1e-150 ×
  大坐标 1e12 × 纯 Dirichlet) 的 stdout 日志序列 + 完整 result dict
  逐值金标准 (`tests/test_solve_refactor_lock.py`) — 重构后逐位复现。

### 3. 参考装配导出收敛

`assemble_lil_reference` / `assemble_expand` (验证性冗余, 实现保留)
从 `fem2d/__init__` 顶层导出移除, 改从 `fem2d.assembly` 导入; 顶层保留
生产路径 `assemble_sparse` / `assemble_sparse_vectorized`。全项目引用
(含测试) 已核对, `test_assembly_recovery_solver.py` 新增
`test_reference_assemblies_not_reexported_at_top_level` 判别性测试。

### 4. 历史审计注释清理

~130 处 "审计 2026-08-03" / "第X轮外部审查" / "高强度审计 2026-08-02"
类叙事标记从代码注释与 docstring 移除 (对应修复历史均已在 9.17.0 小节),
注释只保留"为什么这样写"。逐文件 AST 逻辑签名校验 (注释/docstring 外
任何节点变化即拒绝), 未改任何逻辑; 全量测试 569 全绿。

## 包 6 — 测试与文档 (2026-08-03, 未发布, 版本号未变)

- **测试数实测修正**: 504 collected → 502 passed + 2 skipped (本机); 无 Gmsh
  环境 483 collected → 474 passed + 13 skipped (4 个模块级 skip 不计入
  collected)。文档不再写单一固定数字, 注明环境差异。
- **中文字体缺字警告收敛**: 绘图测试经 plot_three 渲染中文标签, 无 CJK
  字体机器上 matplotlib 发 "Glyph ... missing from font(s)" 噪音 —
  tests/conftest.py 经 pytest_configure 注册 filterwarnings 收敛
  (模块级 warnings.filterwarnings 会被 pytest 的 catch_warnings 绕过,
  必须用 ini 机制), 不动任何断言语义。
- **厚壁圆筒验证约束修正** (fem2d/verification.py): 旧版 fix_node(0, "both")
  把内边界 θ=0 节点径向位移强制为零, 与 Lame 自由膨胀 u_r(a) 冲突 —
  改为三个切向最小约束 (θ=0 处内外两点 uy + θ=π/2 处 ux), 刚体模态
  全消且不约束任何径向位移; σ_θ/σ_r 误差 6.7265%→6.7257% / 8.8287%→
  8.8273% (结论不变, 仍 PASS)。同文件 2 处 1e-30 绝对地板 → tiny。
- **判别性测试**: test_cylinder_constraint_leaves_inner_radial_free
  (旧约束 ux(0)=0, 相对 Lame 100% 误差必失败); 字体过滤注册守卫。
- **文档同步**: README / PROJECT_SUMMARY / NEXT_SESSION_HANDOFF 测试数、
  git 基线 (2026-08-03 起)、打包惯例
  (`FEM2D_project_Q4_YYYYMMDD_HHMMSS_maintainable.zip` 放 Downloads,
  不含 tools/、缓存、egg-info) 全部按实测更新。

