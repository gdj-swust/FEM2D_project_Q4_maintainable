# CHANGELOG

> 历史修复里程碑汇总 (2026-08-03 起)。源码注释只保留"为什么必须这样做"；
> 修复历史与审计记录迁移至此。更早的历史散见于代码注释与知识库日志。

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
