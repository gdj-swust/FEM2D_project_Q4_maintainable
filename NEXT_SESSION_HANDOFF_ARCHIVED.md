# 🔄 会话交接文档 (2026-08-03 夜间整修 + 两轮复扫 + 第三轮实跑验证) — 验收用

> 历史交接（2026-08-03），已被 COMMANDER_HANDOFF_20260804.md 取代。
>
> 前一个会话 (2026-08-03 凌晨) 因 token 预算中断; 本会话按待办 P0-0~P0-3 + 输入端整改 +
> 两轮 8 agent 全链路复扫 + 第三轮真实路径验证 完成整修。
> **验收第一件事**: 跑 `python -m pytest` (2026-08-04 pkg7 复核: 本机 **939 collected → 0 失败**;
> 缺捆绑 gmsh 的环境 collected/skip 以 `pytest --collect-only` 实测为准)
> → 跑 `python run.py models/test_spec.txt --no-plot` 冒烟
> → 试交互向导: `python run.py` (终端)。

---

## 一、项目概况

- **路径**: `C:\Users\35666\Downloads\FEM2D_project_Q4_maintainable\FEM2D_project_Q4_maintainable\`
- **性质**: 2D 有限元教学求解器 (CST/Q4/Q4R/Q4I)，~5.5 万行 Python，2026-08-03 起有 git 基线（主会话 init，含 @FEM 严格解析 / Physical Point 域外拒绝 / elem_type 只读 3 项已修）
- **测试**: `python -m pytest` 本机实测 **939 collected → 0 失败**（2026-08-04 pkg7 复核；
  轨迹 374 → 496 → 包6 502+2 → 939 见 CHANGELOG）；无 Gmsh 环境以实测为准；Ruff/Mypy 全干净
- **版本**: 9.16.0 → **9.21.1** (pyproject.toml；中间轨迹见 CHANGELOG)
- **输入链**: .txt 中文描述 → scripts/geo_spec.py → .geo → gmsh → .msh → gmsh_adapter.py → Mesh

## 二点五、第三轮真实路径验证 (2026-08-03 凌晨后半)

1. **models/ 资产批量冒烟 11/11**: 8 个 .spec + 3 个 .txt 全部跑通。
   - test_simple.txt (无 BC) 被刚体模态检查正确拒绝 — 报错清晰, 非 bug
   - hp_*.geo 无 left/right 几何边 → 用物理组名 (octagon/hole) 跑通
2. **run_demo.py 实跑**: 19447 节点完整求解 + EOF 优雅退出
3. **解析器 fuzz 800 轮** (随机畸形输入): 0 裸异常, 全部 ValueError 带行号
4. **大网格路径**: 14.5 万节点/29 万单元, weighted 恢复自动降级, eta 0.59%
5. **消去 vs 罚函数真实模型差分**: max|Δu|=1.6e-15 (相对 8e-11)
6. **README 全部文档命令验证**: --self-test / python -m fem2d.convergence (Richardson 收敛率 0.90) / --quad 全部可用
7. **修复**: convergence.py 打印标签过时 ("self-referenced vs finest" → Richardson ref)
8. **[重要] resolve_txt 手写 .geo 覆盖保护**: 实测发现 .txt 生成物会覆盖同名手写 .geo
   (今晚冒烟已覆盖 3 个文件 — 经备份对比确认原本就是生成物, 无真实损失)。
   修复: 检测 "Auto-generated" 标记, 手写 .geo 生成到临时副本, 原始文件不碰
   (+ test_resolve_txt_preserves_handwritten_geo 判别性测试)
9. **备份对比**: 被覆盖的 test_spec/test_complex/test_simple.geo 与
   FEM2D_project_Q4_20260802_v9.13_noinp_backup 内容语义一致 (同为生成物)

## 二、本会话完成的修复

### 二轮复扫总览（8 agent 并行审查, 30+ 项发现全部处理）
- **第一轮 4 agent** (核心数值/边界/后处理/输入端): 23 项 → 修 21 + 2 低风险保留
- **第二轮 4 agent** (单元内核/测试质量/输出报告/流程编排): 27 项 → 全部处理
- 冒烟测试: .txt/.geo/.spec/.msh 四路径 + 双分量面力 + 微尺度 + --save + --quad 全通过

### 第二轮新增修复（生产代码, 本文件下半部是第一轮记录）
1. **[P1] --band-* 末带静默加宽一倍**: floor((0.3-0.1)/0.1)=2 → 末带 0.2 (实测 31 组十进制组合复现)。修复: round(ratio) + 尾点归位
2. **[P2] _apply_fix_bcs 批处理不一致**: 仅 --body/--save 时 fix 仍交互提问阻塞 → 传入 batch_mode 统一
3. **[P2] --list-boundaries 交互选文件崩溃**: config.mesh=None → basename(None) TypeError (边界已列出却报错) → 兜底 "模型文件"
4. **[P2] Ctrl-C 裸 traceback**: interactive_plot input() 无 KeyboardInterrupt 处理 → 优雅退出
5. **[P2] --self-test 吞掉非法 BC 参数退出 0** → WARN 列表; --self-test+--list-boundaries 组合 WARN
6. **[P2] geo_spec.py main() 错误路径退出码 0** (脚本化无法感知失败) → sys.exit(1)/return 2 区分
7. **[P3] --force 数值错误风格分裂** (字段数 FATAL/1 码 vs float() 裸 ERROR/2 码) → 统一 FATAL
8. **[P3] .msh+--force 名称误报 "Gmsh API 不可用"** → no_geo_source 原因 + 针对性提示
9. **[P3] --elem-type 覆写后平面冲突误报** (CPS4+Q4R+--plane strain 矛盾提示) → 覆写内核跳过冲突检查
10. **[P3] Physical Point 失败原因区分** (not_found/ambiguous/outside_domain/too_far/no_geo_source) — 曾统一报"未找到"
11. **[P3] 零位移打印"变形放大 1000000x"** → "零位移 — 无变形放大"
12. **[P3] gmsh_runner subprocess locale 解码** (中文输出 UnicodeDecodeError 冒泡) → errors="replace"
13. **[P3] "云图已生成"先于绘图打印** (--save 失败时矛盾输出) → 移到 plot_three 之后
14. **[P3] reporting 残差标签与实际公式不符** (自由 DOF/修改系统残差) → 标签改为"求解系统残差"
15. **[P3] Q4/Q4I 标量路径大坐标 Jacobian** (1e12 偏移刚度偏差 ~1e-4) → 首个节点居中化 (实测 0 偏差)
16. **[P3] sharp_corner_indices 闭合重复点**: 闭合链首角点被静默跳过 (八边形只检出 7/8) → 与 curvature() 同款处理
17. **消毒逻辑去重**: gmsh_runner.sanitize_geo_source 唯一实现, gmsh_adapter._safe_geo_source 复用 (曾双实现 + 正则常量重复)
18. **physical_mapping.py 遗留路径标记** (生产中 edge_labels 恒 None 不可达, 保留 API 契约)

### 第二轮测试加强（10 条假通过测试）
- test_stress.py:64 空断言 → 均匀场精确恢复 (全零实现会失败)
- test_stress.py:100/112 载荷叠加 → 反力平衡断言 (发现期望符号修正, 叠加逻辑实际正确)
- test_plot_three.py:18/24/30 零断言 → collections/lines 非空断言
- test_extreme.py:9 占位 → 反力=施加力; :171 压缩 → 位移方向断言; :182 空间面力 → 反力积分
- test_boundary_extreme.py:112 嵌套 → 精确 4+4 计数
- test_boundary_robustness.py:117 → ==8 且噪声中点不误判 (暴露 sharp_corner_indices 真 bug)
- test_random.py:68 恒真区间 → 规则 vs 随机网格得分对比
- test_geo_spec.py 多孔/圆板/圆环 → 生成文本断言 (孔坐标写入)
- test_msh_import_audit_20260803.py 条件断言 → 保留段头清空条目强制 WARN 分支 (后调整为状态一致性断言 — 2.2 按 id 恢复物理组是正确行为, WARN 由 4.1 测试覆盖)
- **单元内核数学第二轮结论**: 无 P0/P1/P2 — patch test 机器精度 (4 单元 × 2 平面态), Q4 B 矩阵 vs 有限差分 1e-10, Q4I QM6 增强算子解析一致, Q4R 沙漏补空间精确, 消去/罚函数 1.9e-13, 悬臂梁 vs Timoshenko 偏差符合预期

### 第一轮记录（前半夜完成, 详见下半部）

### P0-1 — 回归测试补齐（+13 测试）

### P0-1 — 回归测试补齐（+13 测试）
- tests/test_geo_spec.py: NaN 孔/NaN 边界/NaN 体力 / 键缺值 / 多余 token / 固定带值 / 圆板边名 / 圆板 内孔N
- tests/test_config.py: .spec BOM / .spec float 转换带键名报错
- tests/test_regression_audit_20260802.py: parse_vec2 全角逗号 / 表达式语法错误带表达式

### P0-2 — 12 个假通过测试全部修复
- test_random.py:55 eta=0 恒真 → 载荷改 x 向 + 断言 eta>0
- test_random.py:85 rtol=0.1 → rtol=1e-12（实测 SPR 单元素精确）
- test_stress.py:168/185 → 真正常应力场 + 超界 levels 必须打印 isoband warning
- test_extreme.py:186 → 真实抛物线 profile 形状断言 (中点 1.0, 两端 0)
- test_plot_three.py:37 → 判别性 tag scoping (超界 levels 探针)
- test_regression_audit_20260802.py:598 OR 恒真 → `balance_ok is True`
- test_regression_issues.py:53 正则测生产 `_LC_PATTERN`（非内联副本）
- test_regression_issues.py:95 z 检查 → 行为化 fake gmsh 模块测试
- test_boundary_extreme.py:56 字面 True → 边界点=内部语义断言
- test_boundary_gmsh.py:157 → 段数精确断言 (8 段 = 4 直线 + 4 圆弧)
- test_q4.py:145 零断言 → ax.collections 断言

### P0-3 — scripts 工具脚本 6 项
- **convergence_study.py:117 P1**: 自参考误差收敛阶系统性高估 (h=2h_f 处斜率=2k，实测 k=1 报 1.40) → **Richardson 真参考** (实测恢复精确 k)
- check_dead_code.py: 单文件/不存在目录 WARN (曾静默"0 候选")；模块限定调用 (M.foo()) 计入 name_uses
- geo_spec.py main() 裸 traceback → 友好报错；同一边 固定+拉力 并存 → 拒绝（曾载荷被约束吞掉）
- check_imports_deep.py: `import fem2d.xxx` 断裂检查 (曾只查 ImportFrom)

### 输入端大整改（用户核心诉求）
1. **preprocess.parse_spec → parse_spec_config 改名** — 与 geo_spec.parse_spec (.txt) 同名分叉消除（__init__ 导出同步更新）
2. **is_batch_mode 统一** — _resolve_geo_lc 曾手写条件漏 fix_ux/fix_uy；_plot 曾内联重写 (交互终端 + CLI BC 参数时 interactive_plot 挂起)
3. **.txt 拉力/面力双分量** `边界 右 拉力 1e6,2e6` — 曾只能单值 (ty 恒 0)；物理曲线标签用下划线编码保证 @FEM 往返
4. **.spec 格式错误响亮化** — 漏写 `=` / 空值 → ValueError 带行号 (曾静默丢键)
5. **resolve_txt 覆盖同名 .geo → WARN**（曾静默摧毁手写 .geo）
6. **run_demo.py 标签未匹配 → FATAL**（曾 WARN 后继续求解, 欠约束静默错结果）
7. runner.py `inp_elem_type` 陈旧命名清理

### 全链路复扫（4 agent 并行, 23 项发现 → 修 21, 跳过 2）
**核心数值 (5)**: mesh.py nodes_on_edge 1.0 地板 / 刚体检查 scl 内层 1.0 / fix_node 警告绝对阈值；solver.py trivial 误标 (u 非平凡却标零解)；loads_core.py 压力 callable 无异常上下文
**边界 (6)**: geometry.py 4 处绝对阈值 (1e-15/eps*10 → 坐标 ULP，微尺度方形曾全判 arc)；segment_builder _validate_nodes max(...,1.0)；gmsh_adapter cad_boundary_complete 只认 curves → curves or cad_curves；naming.py auto_classify_segments 死代码删除；bc_apply 交互多匹配静默取第一个 → 逐段处理
**后处理 (5)**: visualize 体力箭头 1.0 地板；quality area_cv 1e-30 地板；convergence.py 1e-15 地板/1e-14 门槛；plot_three mesh/loads 重复子图；stress.py nodal_average 孤立节点静默填 0 → 与 L2 一致抛错
**输入端 (7)**: geo_spec.py `{:.6f}` → `{:.17g}` 全精度 (微尺度孔曾塌缩成 0.000000)；.spec 格式错误；runner._plot batch_mode；resolve_txt 覆写；run_demo FATAL；其余 2 项跳过
**自查 (2)**: patch_test.py 绝对分支 (u_ref<1e-15 时相对 50% 误差通过) → 无条件相对化；topology_core.py 零跨度轴 extent 绝对 1.0 → per-axis ULP

**跳过项 (低优先级, 明早验收不阻塞)**: physical_mapping.py 遗留 edge-label 路径 (生产中 edge_labels 恒 None, 测试直接调用)；gmsh_runner/gmsh_adapter 消毒逻辑双实现 (语义已对齐)

### 回归测试
- tests/test_regression_audit_20260803b.py: 9 个判别性测试（微尺度分段/精度/CV/nodes_on_edge/trivial 标签/压力上下文等）


## 二点七、第四轮 冗余/死代码清理 (2026-08-03)

1. **check_dead_code 全库扫描**: fem2d + scripts 全部 0 候选 (剔除保护名单 68 个 API 导出)
2. **wizard.py _ask_str 死函数删除** (定义后未使用)
3. **ULP 双实现合并**: boundary/geometry._coords_ulp 与 preprocess._coordinate_ulp 实现相同
   → geometry 改为别名转发 (唯一实现在 preprocess, 无循环 import)
4. **wizard._finite 内联** (函数内 import math 冗余 → 模块顶部 import)
5. **向导 CLI 优先级修复** (真实分叉): --wizard --E 5e7 曾被向导默认值覆盖
   → 材料/边界/体力在 CLI 显式 (_explicit) 时跳过提问 + 3 个判别性测试
6. **README 补向导用法**
7. **有意保留的复制** (评估后不合并): check_dead_code/check_imports_deep 的 _norm
   (工具脚本独立运行需自包含); 测试文件间的网格构造 helper (独立性取舍)


## 二点八、第五轮 性能剖析 + 数值对照 + 向导深测 (2026-08-03 清晨)

1. **大模型性能剖析 (29 万单元/58 万 DOF)**: 导入 2.2s / Mesh+几何缓存 0.4s /
   边界 1.1s / K 组装 1.4s / 误差估计 0.2s / **CG 求解 54s (4161 迭代) = 唯一热点**
2. **rtol 优化试验 (1e-10 → 1e-8) → 回退**: 位移差异 9.2e-10 (等效), 迭代 4161→3796
   (9% 收益), **但 CG 残差污染固定 DOF 反力 (~4.3e-3 N), 与平衡检查 tol_rel 同量级,
   合法大模型被 ΣF 检查误杀** → 回退 1e-10 并注释原因 (bc.py)。结论: Jacobi-PCG
   迭代数是本质, 无安全优化空间; ILU 在 58 万 DOF 内存/时间不可行 (超时)
3. **cook_membrane 收敛验证**: v = 444.16→445.80→446.40→446.60, 相邻差单调递减
   (1.64→0.60→0.20), 符合 CST 一阶收敛 ✓
4. **curved_beam 验证**: von Mises 峰值在内弧 (0.399, -40.17) — 物理正确位置, eta 4.1%
5. **向导深测 +4 测试 (22 个)**: 圆环(外边压力+内孔固定) / 多孔编号(内孔1/2) /
   圆板带孔 / 总览两次否定重来 — 全过
6. **裸 run.py 管道安全**: 无参数+非 tty → "[ERROR] 需要指定输入文件" 干净退出;
   --wizard+EOF → "[INFO] 未提供数值 — 退出向导" SystemExit(0) — 均无 traceback
7. 清理: cook_membrane 收敛测试产生的 models/tmp*.msh 残留已删


## 二点九、第六轮 覆盖率 + 经典验证 (2026-08-03 上午)

1. **覆盖率分析 (pytest-cov)**: TOTAL 83% (7498 语句)。最低为交互/错误路径
   (runner 50%/input_source 54% — 防御分支为主) 与数值核心部分路径
2. **单元验证路径补测** (verify_all_elements): 发现并修复真实缺陷 —
   cst._verify_element 对退化 (共线) 单元在 _B_matrix 的 1/(2A) 抛裸
   ZeroDivisionError (验证路径崩溃而非报告失败) → 零面积提前返回失败
   (+3 判别性测试)
3. **Kirsch 应力集中经典验证**: 带孔板 x 拉伸, K_t = 3.04 (理论 3.0)
   集中位置在孔上下缘 θ=±90°、右缘环向 -σ 压缩 — **全部正确**。
   (自查初期误判为"应力旋转 90° bug" — 实为验证脚本 Kirsch 公式
   σ_θ = σ(1-2cos2θ) 方向记忆错误; 代码无 bug。教训: 经典解预期
   必须先核对教材公式) → 已固化为 test_kirsch_stress_concentration
4. **对照验证**: 无孔板 σx=1.003 (理论 1.0) ✓; 固定端拐角奇异 (1.7) 物理合理;
   手动应变 vs solve 应力一致 (后处理链路正确); 远场 0.80 = 能量守恒
   (孔周 3σ 吸能) 合理

5. **fuzz 第六轮**: 边界系统 200 随机多边形 0 崩溃; CLI 随机参数组合
   150 组 0 裸异常 (argparse 孤儿参数 usage 退出 = 正确行为)

## 三、剩余待办（下个会话可选）

1. ~~P0-0 交互向导~~ — **已实现 (2026-08-03)**: fem2d/wizard.py + --wizard + 无参数自动进入 + 15 判别性测试 + 端到端验证通过 (详见 DESIGN_INPUT_WIZARD_20260803.md 实现状态)
2. physical_mapping.py 死路径清理；gmsh_runner/gmsh_adapter 消毒逻辑去重
3. Physical Point 映射失败原因区分 (P3)
4. models/_t_neglc.txt + _t_hand.msh 测试残留（删除被权限拒绝，留给用户决定）


## 三、外部审查修复 (2026-08-03 用户第三方审查, 4 bug + 2 风险 + 维护项)

**Bug 1 (中高) 微尺度圆弧误判闭合 — 已修**: geometry._segment_is_closed 的
coordinate_scale 曾 max(ptp, 1.0) — 1e-16 尺度开放圆弧被误判闭合。修复:
1.0 → np.finfo(float).tiny (R=1e-16/1e-20 复现通过, 真闭合圆仍正确)

**Bug 2 (中高) BC 公共 API 校验 — 已修**: apply_penalty/apply_elimination
曾接受 penalty=NaN/Inf (NaN 刚度)、布尔掩码 [True]→DOF 1、负 DOF 静默
约束最后一个、重复 DOF 静默覆盖。修复: _reject_duplicate_fixed_dofs 拒绝
布尔 dtype + apply_elimination 入口统一调用 + penalty 有限正数校验
(重复同值 = 幂等去重, 保留; 异值拒绝)。7 场景复现验证 + 判别性测试

**Bug 3 (中) Gmsh 失败掩盖异常 — 已修**: Kirsch 测试曾 mktemp + 失败后
unlink(None)/g.nodes 崩溃掩盖 libGLU 缺失。重构: pytest tmp_path +
Gmsh 异常/None 时 pytest.skip + 清理前检查路径存在

**Bug 4 (中低) 小变形检查 1e-30 — 已修**: model_span > 1e-30 绝对阈值,
跨度 1e-31 模型位移 20% 不告警。修复: span > tiny + u_range > 0 (纯刚体
平移不告警)。复现验证通过

**风险 1 ILU-CG**: 文档注明 CG 需 SPD 预条件器而 SuperLU ILU 不保证 —
ILU 仅显式选择, auto 默认 Jacobi (SPD), 失败时下游已有明确错误 (已注释)

**风险 2 spr.py 1e-30**: 3 处绝对地板 → np.finfo(float).tiny 清理 (0 剩余)

**维护项**: README 测试数 387 → 431; 根目录遗留文件 n (向导保存的模型)
移入 models/wizard_saved_demo.txt; AnalysisConfig.from_dict 未知键 WARN
(曾静默用默认值); geometry 627 区域 1e-30 经核实为曲率除零保护非尺度
地板, 未动


## 三点五、第二轮外部审查修复 (2026-08-03, 5 项全部处理)

1. **共享 DOF 验证器 (最高优先)**: 新增 _validate_dof_partition —
   free/fixed 重叠、遗漏 DOF (曾静默设 0)、布尔掩码、solver 名称
   提前到纯 Dirichlet 分支之前校验。报告 8 场景 + penalty 全拒绝,
   合法路径 (常规+全约束) 正常
2. **装配加权顺序 (B̃=√(t·detJ)·B)**: q4.py/q4i.py 刚度先加权再二次型 —
   微尺度几何 (1e-150) BᵀDB ~ E/L² 中间量曾溢出 Inf, 装配后误报
   "Factor is exactly singular"。修复后 1e-150 刚度有限, 正常尺度与
   标量路径相对差 2.3e-16 (数学等价)。**开发中踩坑**: einsum 下标
   "eai,ebj" 使应变分量独立求和 (数值全错, 15 测试失败) → "eai,eaj"
3. **_check_symmetry NaN/Inf 显式拒绝**: 装配溢出后误报奇异 →
   "contains NaN/Inf entries" 明确诊断
4. **error_est 能量二次型归一化**: 应力先除以 s_scale 在 O(1) 空间
   算能量再乘回 — 载荷 1e-152 时 eta 曾从 65.7% 塌缩到 0.07%,
   修复后全尺度 65.7320% (1e-155 次正规边界偏差 0.04%)。另修零应力
   场 (s_scale=0) 的 U_norm=0 除零警告
5. **椭圆轴比相对化 + 标签 :.6g**: 轴比分母 1e-30 → tiny (微尺度
   2:1 椭圆曾误判整圆); 标签曾 .3f 显示 a=0.000
6. **SPR 微尺度退化**: 3 种场景复现 (规则/边界/O(1) 场至 1e-45) 均
   线性精确 — 无法复现, 上轮已清 1e-30, 标记待提供复现脚本
7. **Q4R 措辞精确化**: q4r.py docstring 明确限制是"本实现 compact
   hourglass 公式的理论/数值特性, 非编码错误, 不能推广到所有 Q4R,
   商业软件更复杂沙漏控制可能更宽; Q4/Q4I 完全不受影响"
8. 测试: +4 判别性 (eta 1e-152 不变性 / 椭圆轴比 / BC 分区 / 装配 1e-150)


## 三点六、Q4R 工程落地 + Gmsh 链路加固 (2026-08-03 用户要求)

**Q4R 定位明确** (用户确认: 专用可选单元, 非主力):
1. **长宽比强警告** (solver.py): Q4R 求解前检查单元长宽比 —
   >10 WARN (compact 公式不可靠), >=50 更强 WARN (文档失效区, 建议 Q4I)
2. **沙漏能分级警告**: >90% 新增强警告 (沙漏主导, 结果不可靠);
   >30% 已有警告保留
3. **reporting Q4I 交叉验证提示**: Q4R 模型摘要附 [INFO] 提示
   (稳定性不如 Q4, 弯曲不如 Q4I — 建议 CPS4I 交叉验证)
4. **--elem-type help 批注**: 四单元用途一行的说明 (CST 教学基础/
   Q4 稳健保守/Q4I 综合最佳默认推荐/Q4R 专用)
5. **Gmsh 链路加固**: run_gmsh gmsh 缺失从 WARN 升级 ERROR +
   安装指引; generate_geo_with_topology 异常契约文档化 (生成失败→
   (None,None); 验证失败→抛异常; 调用方须同时处理)
6. 验证: 长宽比 50 网格实测警告触发; 436 全绿


## 三点七、回归审计 (2026-08-03 用户质疑"引入新 bug"后专项)

**抓到并修复 1 个我引入的真 bug**:
- **error_est elem_contrib 微尺度失真**: 能量归一化改动残留 — elem_contrib
  曾用乘回 s_scale 后的 elem_err² (~1e-308 次正规), 载荷 1e-150 时求和
  只到 0.46% (应 100%)。修复: 在归一化空间算贡献 (s_scale² 比值中抵消),
  sum=100% 且 worst_elem 稳定到 1e-155 (+判别性测试)

**未发现其他回归**:
1. 端到端: 9 资产 + run_demo + 向导全过
2. 3 模型 (test_spec/demo_complex/cook) × 消去/罚函数 + 非零位移差分
   max|du| ~1e-10 量级
3. 关键数值: error_est worst/contrib 一致; geometry 闭合 (真闭合+微尺度
   开放) 正确; 4 单元 × 2 平面态 patch test 全过
4. fuzz 重跑: .txt 400 + .spec 250 轮 0 崩溃
5. 历史踩坑记录 (今日): einsum 下标 "eai,ebj" 独立求和 (15 测试失败);
   nosec 替换吞 pass/continue; del mesh 与 Q4R 提示冲突 — 均已修复,
   教训: 批量替换后必须 AST 编译 + 全量测试


## 三点八、第三轮外部审查修复 (2026-08-03, 6 主要问题 + 维护项全部处理)

1. **椭圆完整链路**: topology.py 曾复制旧 1e-30 轴比逻辑 (geometry 修了
   但拓扑层重复) — 统一为 geometry._axis_ratio/_semi_axis_label 公共函数,
   完整 detect_boundaries 1e-40 端到端识别 ellipse + 科学计数标签
2. **elimination 统一校验** (_validate_elimination_inputs): K 方阵 /
   F 形状与 NaN / free 自身重复 (曾延迟到 SuperLU 奇异) — solve 的
   dirichlet_only 分支前也加 solver 名校验 (全约束 + bogus 曾静默成功)
3. **penalty 相对校验**: 曾绝对 ">1.0" (K=1e-12 时有效罚因子被误拒,
   K=1e12 时无效罚因子被接受) — 改为罚刚度 >= max|K_ii| 的相对判据
4. **Q4R 重复警告**: 删除求解时重复的长宽比警告, 保留装配前一套分级
   (>10 过刚或过柔 / >=50 文档失效区), 措辞与文档一致
5. **Z2 eta 彻底尺度无关**: 归一化空间 (frexp 指数分解防次正规 s_scale
   倒数溢出 inf; 零场 eta=0) — 载荷 1e-300 精确 65.7320%, 1e-310 有限
   (双精度极限 0 而非 NaN)
6. **全单元族装配加权**: CST 批量路径加权 (曾 BᵀDB~1/A² 微尺度溢出);
   Q4/Q4I 负 Jacobian 显式拒绝 (曾 sqrt(max(...,0)) 静默夹零);
   sqrt(t)·sqrt(detJ) 顺序防乘积下溢 — 全单元族 1e-150 刚度有限
7. 维护: test_geo_models skip 只针对 Gmsh 依赖不可用 (真实回归不掩盖);
   lil_reference 双 docstring 合并; 根目录 n 文件移入 models; 文档测试数
   统一 441


## 三点九、第四轮外部审查复核 (2026-08-03, 核对 6 条 + 修复 1 条真缺口)

**复核结论**: 审查报告 6 条主要问题中 5 条针对第三轮修复前版本 (已修,
判别性测试在场, 详见 3.8); **1 条部分为真** — 全装配路径加权一致性:

1. **真缺口: 标量/参考/Q4R批量路径仍旧乘法顺序** — 第三轮加权只覆盖
   Q4/Q4I/CST 生产批量路径; 标量 element_stiffness 四单元族
   (q4.py:76 / q4i.py:214 / q4r.py:120,127 / cst.py:106) 与 Q4R 批量
   (q4r.py:191-202, einsum 先算 BᵀDB 再乘 t·det) 仍是旧顺序, 参考装配
   (lil_reference/expand, 走标量路径) 连带溢出 — L=1e-150 复现
   (警告转错误模式: batch CPS4R 非有限 / 标量四族全部溢出 / 两参考
   路径溢出; 生产 sparse 有限)。修复: 全部统一 B̃=√t·√detJ·B 先加权
   再二次型; CST 批量 √(t·A) → √t·√A (乘积先下溢保护)。修复后
   1e-150 全路径有限, 正常尺度批量 vs 标量相对差 ≤2.3e-16
   (+10 判别性测试 tests/test_regression_audit_20260803c.py, 旧公式
   复现即溢出)
2. 维护: 根目录又出现残留文件 n (上一会话手动向导冒烟保存产物, 内容
   与 models/wizard_saved_demo2.txt 逐字节相同) — 已删; README/CHANGELOG
   测试数 437 → 451 (审查者指出 431/432/437 与实测不符, 属实)
3. 已核对为第三轮修复有效的项 (审查者断言仍坏, 实际已修, 测试在场):
   椭圆完整链路 (1e-16/1e-32 端到端识别 + 标签) / elimination 校验
   (free 重复/F NaN/K 方阵/bogus 全约束) / penalty 相对判据 /
   Q4R 单套长宽比警告 / eta 1e-160 精确 + 1e-310 有限 / test_geo_models
   只 skip Gmsh 依赖 / lil_reference 单 docstring

**教训**: 修复"一条路径"时必须 grep 同函数的全部调用链 (标量/批量/
参考装配共享数学内核, 只修批量会让审查者下一次抓到标量)。

## 四、工程约定（接手 AI 必须遵守）

1. **修复流程**: 最小复现 → 修复 → 判别性测试（放回旧实现必须失败）→ 全量 pytest
2. **绝对阈值 = 系统性病根**: `max(...,1.0)` 下限、固定 `1e-15`/`1e-30` 阈值会破坏微尺度模型 — 一律用相对尺度或 `np.finfo(float).tiny`（本会话又清了 ~15 处）
3. **DOF 约定**: x→`2n`、y→`2n+1`
4. **gmsh.exe 必须带 `-nt N` 和 `-2`**
5. **静默错误比崩溃危险**: 优先把静默忽略变成响亮 WARN/ERROR
6. 不要重复报告已修复问题（代码注释"审计 2026-08-03"标记 = 已修）

## 五、剩余风险（诚实清单）

| 区域 | 状态 |
|------|------|
| 核心数学 + 微尺度全家族 | 高置信（本轮 ~21 处相对化修复 + 判别性测试） |
| 输入端 4 入口 | 功能层加固 + 分叉收敛（is_batch_mode 统一/parse_spec 改名/双分量/格式响亮化） |
| physical_mapping / 消毒双实现 | 已知保留项（低风险） |
| 未来新缺陷 | 不可消除，换视角再查（教训: "审过 ≠ 干净"） |

## 六、打包

用户习惯: `FEM2D_project_Q4_YYYYMMDD_HHMMSS_maintainable.zip` 放 Downloads（不含 tools/、缓存、egg-info）
