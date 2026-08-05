# 🎖️ 指挥官交接文档 (2026-08-05, 9.24.0)

> 本会话（指挥官）token 接近上限，移交指挥权。你接手后职责：
> **派发并行任务 → 合并验收 → 版本/打包**。项目 9.24.0 已稳定，所有已知缺陷闭环。

## 一、项目现状速览（接手即确认）

| 项 | 值 |
|---|---|
| 路径 | `C:\Users\35666\Downloads\FEM2D_project_Q4_maintainable\FEM2D_project_Q4_maintainable\`（**两层**，代码所在地） |
| 版本 | 9.24.0（pyproject.toml） |
| 测试 | **1022 个测试函数全过**（本地 Windows 3.13 含 gmsh；覆盖率 93% ≥ 90 门） |
| git | main 最新 = `d2d9e6e`，已推送 origin，**GitHub CI 全绿** |
| 远程 | `https://github.com/gdj-swust/FEM2D_project_Q4_maintainable`（**私有**，CI 已托管，push 需 GCM 授权弹窗） |
| 性质 | 2D 线弹性 FEM 教学求解器（CST/Q4/Q4R/Q4I），~5.5 万行 Python |
| 外部审查 | 8.7/10（修复前）；5 项发现已全部闭环 + 判别性测试锁死 |

## 二、目录地图（磁盘布局有历史嵌套，别迷路）

```
Downloads\FEM2D_project_Q4_maintainable\          ← ① 外层 = git 仓库根（.git 在此）
  └─ FEM2D_project_Q4_maintainable\              ← ② 两层 = 项目代码（干活在这）
       ├─ fem2d/ scripts/ tests/ docs/ models/ run.py ...
       └─ wt_pkg1_loads ~ wt_pkg6_docs           ←    旧 worktree（5 个，已合并可清）
```

- **python/pytest 必须在两层跑**；打包源必须是两层
- 另注意：Downloads 里有大量历史快照目录/zip（`FEM2D_project_Q4_*_20260729~31` 等，几十个）——**都不是当前代码**，别拿它们当源

## 三、git 状态与清理

- 可清理（已合并进 main）：`pkg1_loads pkg2_q4inv pkg3_z2 pkg4_gmsh pkg6_docs`（5 个 worktree + 分支）
- 保留：main（当前工作分支）

## 四、指挥工作流（标准模板）

1. 建 worktree：`git worktree add -b <pkg_name> wt_<pkg_name> main`
2. 写 PROMPT.md（worktree 根）：任务 / **文件边界（并行零冲突关键）** / 行为冻结区 / 工程约定 / 验收 + 🔥 高强度 + ✅ 自检清单 + **另眼审查（陌生评审挑 3 问）**
3. 派发消息模板：路径 → 先读 PROMPT.md → 每提交必全量 pytest → 完成后 push + 等 CI 全绿 → 汇报（改动/判别性/验证数字/CI/另眼审查）
4. 验收四件套：
```bash
python -m pytest -q -p no:warnings        # 1. 全量 1022 全过
python scripts/audit_contract_probe.py    # 2. 探针 0 FAIL
python scripts/regression_compare.py _ci_drift.msh 2.1e11 0.3 0.01 1e6 0.0 elimination   # 3. 漂移门 (需自建 _ci_drift.msh, 见 ci.yml 内联脚本)
python scripts/fuzz_api.py 500            # 4. fuzz 0 problems
```
5. 版本 + CHANGELOG + 打包（python 脚本 zipfile；排除 .git/.coverage/tools(非full)/wt_*/缓存/*.zip/tmp*/PROMPT_*；**⚠️ .github 必须包含——隐藏排除规则不能无脑排除点开头路径**；放 Downloads + 复制两个桌面）

## 五、工程约定（派发时写进每个 PROMPT）

0. 每提交必全量 pytest 0 失败；任何一步全红 = 立即回滚该步
1. 修复流程：最小复现 → 修复 → **判别性测试（放回旧实现必须失败）** → 全量 pytest
2. 禁绝对阈值：`max(...,1.0)` / 固定 `1e-15` — 用相对尺度或 `np.finfo(float).tiny`（微尺度 1e-150 模型是硬指标）
3. DOF 约定：x→`2n`、y→`2n+1`
4. gmsh.exe 必须带 `-nt N` 和 `-2`
5. 静默错误比崩溃危险；注释不写历史叙事（只留"为什么"）
6. 不要重复修复已有注释标记 "审计" 的已修项
7. 无 gmsh 环境的测试必须 skip 而非失败

## 六、行为冻结区（改动需特殊程序：先金标准 → 逐位一致 → 回归 0.0）

- `fem2d/element/` 内核公式、`solver.py` 数值逻辑、`error_est.py` 公式——只改入口校验
- 边界识别管线（`fem2d/boundary/` 探测器/拓扑/载荷路径）——只加插件/标签/输入；识别算法本身冻结
- 金标准快照 `tests/boundary_golden/` 锁边界行为；golden 锁求解；漂移门锁数值

## 七、已完成的轮次（2026-08-04 ~ 08-05，本指挥官任期）

| 轮 | 内容 | 结果 |
|---|---|---|
| 轮0 | **CI 机制建立**：GitHub 私有仓托管 + 覆盖率硬门 90% + vulture + 漂移门 | 已锁 |
| 轮1 | **边界层插件化重构**：金标准快照 → 探测器注册表 + 显式管线 + 原生实体信息一等公民 + 示例插件 | 已锁 |
| 轮2 | **三识别插件**：组级椭圆标签（原生直读+2% 严格门）/ @组名批量选择 / 曲率分段（ρ+圆心，短弧不硬拟合）| 已锁，9.24.0 |
| 轮3 | **审查修复包**：hypot 溢出 / 警告吞噬 / 退出码泄漏 / 能量钳 0（问题规模相对容差）/ .geo SystemCall 拦截 | 已锁 |
| 轮4 | **--keep-open**：批处理命令可选保留交互窗口（与 --no-plot 互斥 exit 1）| 已锁 |
| 发布 | 9.24.0 打包（标准 1.2MB + full 35.2MB，双桌面） | 完成 |

## 八、待办（诚实清单，按建议优先级）

| 项 | 状态/建议 |
|---|---|
| **解析解对照测试包** | **最高优先**：悬臂梁精确解 / Kirsch 圆孔应力集中（→3.0）/ **Z2 效应指数 θ∈[0.8,1.2]**（估计误差 vs 真实误差）——当前唯一理论盲区：Z2/SPR 实现被锁但"理论正确性"无权威测试 |
| **新能力（模态/热应力）** | 教学/面试价值最大（用户目标：保研大工计算力学）；走"并列新增"不碰冻结区 |
| **Windows runner 补 CI** | 半小时；GBK 控制台乱码是活证据（用户终端中文输出在部分环境乱码） |
| **7% 覆盖（551 行）** | 按覆盖率报告补复杂分支测试 |
| Q4R 长宽比 ≥50 不可靠 | 已分级警告；生产 opt-in 标记待用户拍板 |
| ILU-CG 理论风险 | 已文档化（auto 默认 Jacobi/SPD） |
| 架构评估报告 | 曾答应用户的 docs 路线图（维护/性能/演进），未写 |
| 旧 worktree 清理 | 5 个已合并，`git worktree remove` + `git branch -d` |
| fuzz 扩面 | 加输入类型/API 面（已证明能抓真 bug） |

## 九、关键资产

```
docs/api_contract.md           契约表（含 G2 边界契约行）
docs/boundary_plugins.md       五步插件接入手册（轮 1 产物）
docs/ci.md                     CI 行为说明
tests/boundary_golden/         边界金标准快照（13 JSON + 规范化 12g）
tests/test_solve_refactor_lock.py  求解 golden（数值逐位锁）
.github/workflows/ci.yml       lint(ruff/mypy/vulture) + test-core(3.12/3.13) + test-full(覆盖率90+漂移门)
scripts/audit_contract_probe.py    探针（129+ 项）
scripts/fuzz_api.py            误用 fuzz（固定种子，500/2000 轮）
scripts/regression_compare.py  数值漂移对照（带参运行）
```

## 九·补、任务书档案（磁盘存档，gitignore 不入库）

仓库根目录有 **9 份 PROMPT_*.md**（`.gitignore` 第 22 行 `PROMPT*.md` 刻意排除——任务书是临时文档；**需要时直接磁盘读取，别 git add**）：

| 文件 | 轮次 | 用途 |
|---|---|---|
| PROMPT_CI_GREEN_20260804.md | 轮0 | CI 全绿修复 + 机制补漏（golden 容差/fuzz 种子/漂移门/极端拓扑） |
| PROMPT_BOUNDARY_REFACTOR_20260804.md | 轮1 | 边界层插件化重构（三阶段：快照→注册表→示例插件） |
| PROMPT_BOUNDARY_PLUGINS_20260805.md | 轮2 | 三识别插件（椭圆标签/@组名/曲率分段）+ 3 挑剔点硬约束 |
| PROMPT_REVIEW_FIXES_20260805.md | 轮3 | 审查修复包 5 项（hypot/警告/退出码/能量钳/.geo） |
| PROMPT_KEEP_OPEN_20260805.md | 轮4 | --keep-open 选项（判别性测试 4 场景） |
| PROMPT_API_CONTRACT*.md / PROMPT_FINAL_REGRESSION.md / PROMPT_PKG5.md | 前任轮次 | 契约清账/终轮回归/pkg5（早期任务书，可当结构参考） |

**新任务书直接照抄结构**：背景（已定位根因）→ 任务（可验收）→ 文件边界 → 行为冻结区 → 工程约定 → 验收（判别性/四件套/CI 全绿/另眼审查）→ 汇报格式。

## 十、本任期教训（照做，别重蹈）

1. **总指挥先获批再动源码**：派发前必须获得用户明确批准；用户说"你来做"≠ 授权改源码（曾因此挨批）
2. **任务书先理清再派**：任务/文件边界/验收三件套，行为冻结区写清；执行者要"另眼审查"环节
3. **打包排除规则**：.github 必须保留（曾误排除 CI 配置）；用 python zipfile 不用 bash mv
4. **用户实测发现**：段编号易选错（用户原命令段 11 实为外角圆角）——施力一律推荐 @组名
5. **验收要自己跑**：四件套独立复跑 + 抽查关键 diff（golden 单键变化、快照正当性、插件真实性）

---

**最后一句**：项目处于"核心锁死 + 机制常驻 + 新能力待开"状态。未来方向是**加法**（解析解验证包 → 模态/热应力 → 教学演示），不是再查旧的。祝接手顺利。
