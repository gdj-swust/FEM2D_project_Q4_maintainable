# 🎖️ 指挥官交接文档 (2026-08-04, 9.22.0)

> 本会话（指挥官）token 接近上限，移交指挥权。你接手后职责：
> **派发并行任务 → 合并验收 → 版本/打包**。项目本身 9.22.0 已稳定。

---

## 一、项目现状速览（接手即确认）

| 项 | 值 |
|---|---|
| 路径 | `C:\Users\35666\Downloads\FEM2D_project_Q4_maintainable\FEM2D_project_Q4_maintainable\`（**两层**，代码所在地） |
| 版本 | 9.22.0（pyproject.toml） |
| 测试 | ~991 passed，0 失败（2 skip = 无 gmsh 环境项；仓库外 cwd 亦 0 失败） |
| git | main 分支最新 = `b3ee63a` 之后若干（`git log --oneline -5` 看） |
| 性质 | 2D 线弹性 FEM 教学求解器（CST/Q4/Q4I/Q4R），~5.5 万行 Python |
| 审查评分轨迹 | 8.0 → 8.8 → 9.0（当前所有可落地项已闭环） |

## 二、目录地图（重要——磁盘布局有历史嵌套，别迷路）

```
Downloads\FEM2D_project_Q4_maintainable\          ← ① 外层 = git 仓库根（.git 在此）
  └─ FEM2D_project_Q4_maintainable\              ← ② 两层 = 项目代码（干活在这）
       ├─ fem2d/ scripts/ tests/ docs/ models/ run.py ...
       └─ FEM2D_project_Q4_maintainable\         ← ③ 三层 = 历史 worktree 挂载处
            └─ wt_opt2_xxx\ 等                   ←     （旧 worktree，可清理）
```

- **git 命令**：从两层或外层跑都行（git 向上找 .git）
- **python/pytest**：必须在**两层**跑（`cd ...\FEM2D_project_Q4_maintainable\FEM2D_project_Q4_maintainable`）
- **打包源**：必须是**两层**（用外层会把嵌套目录打进包——历史教训，出过 716MB 的包）

## 三、git 状态与清理

```bash
git worktree list          # 看所有挂载
git branch -a              # 看所有分支
```

**可清理**（已合并进 main 的分支/worktree）：
`pkg1_loads pkg2_q4inv pkg3_z2 pkg4_gmsh pkg6_docs opt_ci opt_coverage opt_geo_out opt_perf opt2_loads opt2_meshbc opt2_q4rest opt2_runner opt2_scripts`（15 个，全部已合并）。
清理命令：`git worktree remove <dir>` + `git branch -d <name>`。
**保留**：main（当前工作分支）。

## 四、指挥工作流（标准模板）

### 1. 建 worktree（每包一个隔离分支）
```bash
git worktree add -b <pkg_name> wt_<pkg_name> main
```
⚠️ worktree 路径相对仓库根解析——注意最终落点，写 PROMPT 时用实际路径。

### 2. 写 PROMPT.md（每个 worktree 根目录）
必需段落：
- 角色与项目（路径、先读 PROJECT_SUMMARY → CHANGELOG → 本文件）
- 任务（具体、可验收）
- **文件边界（只许改这些）**——并行零冲突的关键
- **行为冻结区**：fem2d/element/ 内核公式、solver.py 数值逻辑、error_est.py 公式——只改入口校验
- 工程约定（见第五节）
- 验收 + 🔥 高强度要求 + ✅ 自检清单（模板见下）

### 3. 派发消息模板
```
接手 FEM2D 项目（二维线弹性有限元教学求解器，Python，约5.5万行，有 git）：
<worktree 完整路径>
先读该目录的 PROMPT.md —— 任务/文件边界/工程约定/验收标准都在里面。
每提交必全量 pytest（基线 903 passed 0 失败）。按高强度要求执行，
完成后 commit 并汇报（改动清单 + 判别性测试 + pytest 结果 + 自检清单）。
```

### 4. 合并 + 验收四件套（每轮并行完成后的必做动作）
```bash
# 合并（一般零冲突——文件边界设计保证）
for b in <分支列表>; do git merge $b -m "merge $b"; done
# 验收四件套
python -m pytest -q -p no:warnings              # 1. 全量 0 失败
python run.py models/test_spec.txt --no-plot     # 2. 冒烟 .txt
python run.py models/l_bracket.spec --no-plot    # 3. 冒烟 .spec
python scripts/audit_contract_probe.py           # 4a. 契约探针 100 项
python scripts/fuzz_api.py 500                   # 4b. fuzz（收紧后含静默成功判定）
# 数值漂移对照（大改动后必跑）
python scripts/regression_compare.py             # 早期 vs 当前，要求相对差 0.0
```

### 5. 版本 + CHANGELOG + 打包
- 升版本：`sed -i 's/^version = "X"/version = "Y"/' pyproject.toml`
- CHANGELOG 顶部插入新版本节（含修复清单/验收数字）
- 打包：**python 脚本**（不要用 bash mv 带尾反斜杠的路径——历史教训：`"C:\...\"` 会把引号转义，mv 出畸形文件名）
  - 源 = 两层项目
  - 排除：`.git`（18MB）、`.coverage`、`tools/`（无 gmsh 版）、所有 `wt_*` worktree、缓存、`*.zip`、`tmp*`
  - full 版 = 含 `tools/`（gmsh.exe 43MB）
  - 命名：`FEM2D_project_Q4_YYYYMMDD_HHMMSS_maintainable[_full].zip`
  - 放 `C:\Users\35666\Downloads\`，并复制到桌面（本地 + OneDrive 两个都放——用户两个都可能看）

## 五、工程约定（派发时写进每个 PROMPT）

0. **每提交必全量 pytest 0 失败**；任何一步全红 = 立即回滚该步
1. 修复流程：最小复现 → 修复 → **判别性测试（放回旧实现必须失败）** → 全量 pytest
2. **禁止绝对阈值**：`max(...,1.0)` / 固定 `1e-15`/`1e-30` — 用相对尺度或 `np.finfo(float).tiny`（微尺度 1e-150 模型是硬指标）
3. DOF 约定：x→`2n`、y→`2n+1`
4. gmsh.exe 必须带 `-nt N` 和 `-2`
5. 静默错误比崩溃危险；注释不写历史叙事（只留"为什么"）
6. 不要重复修复已有注释标记 "审计 2026-08-03" 的已修项
7. 无 gmsh 环境的测试必须 skip 而非失败（参考 test_msh_import_audit 写法）

## 六、已知边界 / 待办（诚实清单）

| 项 | 状态 |
|---|---|
| CI 已配置未托管 | .github/workflows/ci.yml + docs/ci.md 就绪；要真跑需推 GitHub 私有仓库（用户暂不急） |
| Q4R 长宽比 ≥50 不可靠 | 已文档化+分级警告；生产 opt-in 标记待用户拍板 |
| ILU-CG 理论风险 | 已文档化（auto 默认 Jacobi/SPD） |
| 磁盘嵌套布局 | 功能正常，建议不动 |
| 旧 worktree（15 个） | 可清理（第三节命令） |
| models/ 里的生成 .msh | gitignore 已排除，不入库 |

## 七、历史轮次摘要（供引用）

| 轮 | 内容 | 结果 |
|---|---|---|
| 接手轮 | 3 项 P1（@FEM 严格解析/Physical Point 域外拒绝/elem_type 只读） | 已锁 |
| 6 包并行 | 载荷 schema/Q4 逆映射/Z2 尺度/Gmsh 输入/文档/架构 | 零冲突合并 |
| 9.18.0 | 第十轮审查 7 项（含 .spec 回归修复 9.18.1） | 已锁 |
| 契约清账 3 轮 | docs/api_contract.md + 校验收敛 + 复查 + 终轮回归对照 | 676 测试 |
| 9.20.0 | 四路并行（CI/覆盖率 93%/--output-dir/性能 3×） | 已锁 |
| 9.21.0 | 五路并行（压力 callable/标量入口/缓存顺序/审计脚本/fuzz 收紧） | 已锁 |
| 9.21.1 | 单元素数组版本无关 + 探针/fuzz 加固 + 打包卫生 | 已锁 |
| 9.22.0 | 六包并行两批：pkg7 文档统刷 / pkg8 探针 129 项+smoke / pkg9 尺度不变 / pkg10 CLI 退出码+交互 / pkg11 去重 20 项 / pkg12 conftest+路径卫生 | **当前版** |

## 八、关键资产清单

```
docs/api_contract.md          契约表（新代码照着写校验）
docs/regression_comparison.md 数值漂移对照（10 组合逐位一致）
docs/performance.md           性能预期（CG 热点已剖析）
docs/ci.md                    CI 行为说明
scripts/audit_contract_probe.py   100 项探针
scripts/fuzz_api.py               误用 fuzz（500/2000 轮）
scripts/combo_fuzz.py             组合 fuzz
scripts/regression_compare.py     漂移对照
scripts/perf_benchmark.py         性能基准
```

---

**最后一句**：项目处于审查 9.0 且可落地项全闭环的状态。未来工作方向应是**新东西**（新功能/新算例/教学演示/CI 托管），而不是"再查一遍旧的"。祝接手顺利。
