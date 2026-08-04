# CHANGELOG

> 历史修复里程碑汇总 (2026-08-03 起)。源码注释只保留"为什么必须这样做"；
> 修复历史与审计记录迁移至此。更早的历史散见于代码注释与知识库日志。

## 9.22.0 (2026-08-04) — pkg11_dedupe 重复实现/契约清理

### A 组 重复校验/重复实现 (判别性测试全部在场)
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

### B 组 死代码/无效防御
- **patch_test**: `bdy_nodes` 首次计算结果从未使用 → 删除; 模块内
  `sys.path.insert` 死操作 → 删除 (`python -m fem2d.patch_test` 是文档
  运行方式, 根目录已在 path)
- **bc_apply**: `if idx >= len(segs): continue` 死守卫删除 — resolver
  契约保证索引恒在界内, 静默 continue 曾把内部 bug 吞成"面力漏施加"

### C 组 API 形状/错误机制统一
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

### D 组 docstring 与实现对齐 (教学文档)
- error_est.estimate(): 算法描述改为实际积分点公式 (Σ_q w_q·(σ*−σ_h)ᵀ
  D⁻¹(σ*−σ_h), 无 A_e 项, 归一化积分权 + 尺度乘回)
- stress.principal_stresses(): 补第 4 返回 θ = 0.5·atan2(2τxy, σx−σy)
- preprocess.validate_mesh(): tol=None 语义同步为
  max(min(非零单元跨度)×1e-6, 坐标 ULP) (曾声称 min(edge_length)×1e-3)

### 验收
- 全量 pytest **0 失败** (含 A 组判别性测试); 回归对照
  (elimination/penalty) **相对差 0.0** (字符级一致)
- 冻结区零改动: solver 数值路径 / error_est 公式 / element/ 未触碰

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
