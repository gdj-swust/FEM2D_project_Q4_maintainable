# 包 5 — 架构与可维护性（评分 7.5）

> 本目录是 git 主仓库（分支 `main`）。**本包在最后做**（等 pkg1/2/3/4/6 合并完成后）。
> 改动前先 `git pull` 等价操作：`git checkout main && git merge pkg1_loads pkg2_q4inv pkg3_z2 pkg4_gmsh pkg6_docs`
> （或由主会话先完成合并）。改动可自行 `git add -A && git commit -m "..."`。

## 角色与项目

你是 2D 有限元教学求解器 FEM2D 的维护工程师（Python，CST/Q4/Q4R/Q4I 单元，约 5.5 万行，有 git）。
先读：PROJECT_SUMMARY.md → NEXT_SESSION_HANDOFF.md（工程约定在"四"节）→ CHANGELOG.md。

## 任务（4 项，按依赖顺序）

### 1. 库层 sys.exit() 迁移
`fem2d/` 包内 ~27 处 `sys.exit()` → 抛领域异常（`fem2d/errors.py` 的 `CliError` 等），
CLI 层（`run.py`/`run_demo.py`）统一捕获转换退出码。
`scripts/` 工具脚本的 `sys.exit` 保留（脚本独立运行语义）。
逐处 grep 确认调用链；**迁移后 CLI 退出码矩阵（正常 0 / 用户错误 1 / 内部错误 2）必须不变**
—— 先写退出码行为锁定测试再迁移。

### 2. 高复杂度函数重构
- `fem2d/boundary/geometry.py` 的 `_point_in_loop`（复杂度 27）
- `fem2d/solver.py` 的 `solve`（复杂度 22）
拆分/简化，**行为必须逐字节一致**（先写行为锁定测试再动）。

### 3. 参考装配导出收敛
`assemble_lil_reference()` / `assemble_expand()`（验证冗余，**保留实现**）
从 `fem2d/__init__.py` 顶层导出移除，改从 `fem2d/assembly` 模块导入；
全项目 grep 修正引用（含测试）。

### 4. 历史审计注释清理
代码里 "审计 2026-08-03"/"第X轮外部审查" 类叙事 → 迁移到 CHANGELOG.md 对应小节，
代码注释只保留"为什么这样写"。（注意：已有 `审计 2026-08-03` 标记 = 已修，**清注释时不得改逻辑**）

## 工程约定（必须遵守）

1. 修复流程：最小复现 → 修复 → 判别性测试 → 全量 pytest；重构类改动必须有行为锁定测试在前
2. 禁止绝对阈值：`max(...,1.0)`/固定 `1e-15`/`1e-30` — 用相对尺度或 `np.finfo(float).tiny`
3. DOF 约定：x→`2n`、y→`2n+1`；静默错误比崩溃危险
4. 不要重复修复已合并分支中的内容（本包只做上述 4 项）

## 验收

1. CLI 退出码矩阵测试：迁移前后一致（正常 0 / 用户错误 1 / 内部错误 2）
2. `python -m pytest` 全量全绿（应含 pkg1-6 全部新增测试）
3. `python run.py models/test_spec.txt --no-plot` 冒烟通过
4. `ruff`/`mypy` 不新增错误；`python scripts/check_dead_code.py` 仍 0 候选

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
