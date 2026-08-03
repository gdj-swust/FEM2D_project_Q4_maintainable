# 包 6 — 测试与文档（评分 8.6 / 7.0）

> 本目录是 git worktree（分支 `pkg6_docs`）。你的改动只会影响本分支；
> **不要修改其他包负责的源码**（见文末文件边界）。改动后可自行
> `git add -A && git commit -m "..."`（可选，最终由主会话合并）。

## 角色与项目

你是 2D 有限元教学求解器 FEM2D 的维护工程师（Python，CST/Q4/Q4R/Q4I 单元，约 5.5 万行）。
先读：PROJECT_SUMMARY.md → NEXT_SESSION_HANDOFF.md（工程约定在"四"节）→ CHANGELOG.md。

## 任务（4 项）

### 1. README 测试数核实修正
README 宣称 489 个测试，实际收集数随环境变化（本机 Gmsh 齐全 ≈ 496 全过 /
无 Gmsh 环境 ≈ 468 collected + 11~12 skip）。
以 `python -m pytest --collect-only` 实测为准更新 README / PROJECT_SUMMARY 中的
测试数，并注明"无 Gmsh 环境 N skipped"的环境差异说明（不要写成单一固定数字误导）。

### 2. 绘图测试中文字体缺字警告
pytest 输出 `Glyph ... missing from font(s)` 类中文字体警告 — 找到来源测试，
用可渲染的方式处理（显式指定字体 / 符号替换 / 收敛警告过滤），消除警告噪音，
**不改断言语义**。

### 3. fem2d/verification.py（或验证脚本）厚壁圆筒约束
固定内边界一个节点的两个方向与解析径向位移不完全兼容 —
改用切向最小约束或四分之一对称模型。**先核实现有验证的物理含义再改**，
保持既有数值结论有效（改的是约束方式，不是结论数字本身）。

### 4. 文档同步
PROJECT_SUMMARY.md / NEXT_SESSION_HANDOFF.md / CHANGELOG.md 的：
- 版本号、测试数、已知边界清单（Q4R 长宽比 ≥50 不可靠、真实 Gmsh CI 未搭、
  高复杂度函数 `_point_in_loop`/`solve` 等）
- 打包说明按"`FEM2D_project_Q4_YYYYMMDD_HHMMSS_maintainable.zip` 放 Downloads
  （不含 tools/、缓存、egg-info）"惯例更新
- 本项目已建立 git（主会话今日已 init，baseline 含 3 项已修：
  @FEM 严格解析 / Physical Point 域外拒绝 / elem_type 只读）— 相关文档可提一句
  "无 git 历史"需改为"2026-08-03 起有 git 基线"

## 文件边界（只许改这些）

- `README.md` / `PROJECT_SUMMARY.md` / `NEXT_SESSION_HANDOFF.md` / `CHANGELOG.md`
  / `ARCHITECTURE.md`（如有数字不一致）
- `fem2d/verification.py`（仅厚壁圆筒约束一处）
- `tests/` 下与字体警告相关的测试、新增文档一致性检查测试（如有必要）

**禁止触碰**：`fem2d/mesh.py`（包1）、`fem2d/element/`（包2）、`fem2d/error_est.py`（包3）、
`scripts/geo_spec.py`、`fem2d/gmsh_*.py`、`fem2d/input_source.py`、`fem2d/bc_apply.py`、
`fem2d/boundary/physical_mapping.py`（包4）、其余所有数值逻辑。

## 工程约定（必须遵守）

1. 修复流程：最小复现 → 修复 → 判别性测试 → 全量 pytest
2. 禁止绝对阈值：`max(...,1.0)`/固定 `1e-15`/`1e-30` — 用相对尺度或 `np.finfo(float).tiny`
3. 注释不写历史叙事；静默错误比崩溃危险
4. **不改数值逻辑**，除非有明确 bug（本包以核实/修正为主）
5. 本 worktree 无 gmsh 工具 — Gmsh 依赖测试会 skip，属正常

## 验收

1. `python -m pytest` 全量全绿（无新增 warning 类噪音）
2. 文档数字与 pytest 实测一致
3. `python run.py models/test_spec.txt --no-plot` 冒烟通过

## 🔥 高强度要求（用户明确要求）

不要停留在最小修复。修完后做多轮对抗性自查：
1. **重读你改动的完整函数/文件**，找隐藏问题（边界、畸形输入、微尺度、大坐标）
2. **数值对照**：与你改动相关的量做交叉验证（参考实现/理论解/有限差分/前后差分）
3. **微尺度+大坐标**：用 1e-150 几何 / 1e12 坐标 / 1e-310 载荷各跑一遍你改动的路径
4. **绝对阈值扫描**：你新增的代码 grep `max(...,1.0)`、`1e-15`、`1e-30` 类字面量 — 必须相对化
5. 发现的新问题一并修掉（不越界，遵守文件边界）

## ✅ 自检清单（交付前逐条核对）

- [ ] 判别性测试已放回旧实现验证过（确实失败）
- [ ] `python -m pytest` 全量全绿
- [ ] 改动文件无遗留 TODO / print 调试 / 死代码
- [ ] `ruff` 不新增错误（如可用）
- [ ] `python run.py models/test_spec.txt --no-plot` 冒烟通过
- [ ] 回复里列出：改动文件清单 + pytest 结果 + 自查发现与处理
