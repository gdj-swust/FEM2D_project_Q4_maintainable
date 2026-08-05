# 🎖️ 指挥官交接文档 (2026-08-05c, 大优化中继)

> 本会话（崩溃后重建的新会话）完成：大优化进度摸底 + 代码结构地图 + **崩溃真因定位**（见第十节）。
> 接手后职责：**验收合并 B/D/E 三分支 → 续做 C 覆盖轮 → 收尾**。
> ⚠️ 与历史版本的区别：本次经历了会话 32MB 超限崩溃 + 重置，原始对话丢失，本文档即唯一跨会话保险。

## 一、项目现状速览（接手即确认）

| 项 | 值 |
|---|---|
| 路径 | `C:\Users\35666\Downloads\FEM2D_project_Q4_maintainable\FEM2D_project_Q4_maintainable\`（**两层**，git 根=内层） |
| 版本 | **9.25.0**（pyproject.toml 单一版本源；A 轮合并后**未升版**，下轮应升 9.26.0 + 补 CHANGELOG） |
| main | `788fce1` = Merge pkg_cleanA（A 清理轮已合入） |
| 测试 | 91 文件 / 1118 个 test 函数收集，全量 pytest 实测 0 失败（20260805c 复跑；9.25.0 基线 1097 passed） |
| 远程 | `https://github.com/gdj-swust/FEM2D_project_Q4_maintainable`（私有，CI 四绿 = 9.25.0 时状态） |
| 性质 | 2D 线弹性 FEM 教学求解器（CST/Q4/Q4R/Q4I），~1.6 万行源码 / 2 万行测试 |
| 审查评分 | 8.4/10（9.25.0 发布链路补洞后，4 项实锤缺陷全修复锁死） |
| 用户方向 | **不开发新功能，优化完善现有源码**（A 清理 → B 结构 → C 覆盖 → D 收尾） |

## 二、大优化进度总表（2026-08-05 崩溃中断点）

| 轮 | 分支/worktree | 状态 | 内容 |
|---|---|---|---|
| **A 清理轮** | pkg_cleanA / wt_cleanA | ✅ **已合并 main** | 死参数×2（`_fit_closed_conic.prefix`/`_arc_algebraic.scale`）+ PhysicalEdgeMapper 弃用（警告+工厂静默）+ 叙事注释 190→0（只留 rationale）。T1-T4 四提交 |
| **B 结构轮** | pkg_structure / wt_structure | ✅ 完成，**待验收合并** | S1-S7 七个长函数按职责拆分 ≤60 行 + S8 顺手闭环 9.25.0 遗留项（`_safe_geo_source` Include 递归，API 路径） |
| **C 覆盖轮** | pkg_coverage / wt_coverage | 🔶 **半途** | 分析产物已齐（见第四节），**代码零提交**——最可能是崩溃中断处 |
| **D 易用性** | pkg_usability / wt_usability | ✅ 完成，**待验收合并** | D1 CLI @段名引用 + D2 输入入口引导 + D3 Windows GBK 乱码修复 |
| **E 解析解验证** | pkg_analytic / wt_analytic | ✅ 完成，**待验收合并** | E1 悬臂梁 CST/Q4 vs Euler-Bernoulli + E2 Kirsch 圆孔 Kt→3 + E3/E4 **Z2 效应指数 θ∈[0.8,1.2]**（原"最高优先待办"已落地） |

**接手第一件事 = 验收合并 B/D/E 三分支**（验收四件套 + wheel 冒烟 + 合并 + 升版 9.26.0 + CHANGELOG 补 A/B/D/E 轮记录）。

### CI/推送中断点状态（20260805c 核实）

| 分支 | 本地提交 | 推送 origin | CI 门禁 |
|---|---|---|---|
| main=788fce1（A 合并） | ✅ | ✅ 已推送，与 origin/main 同步 | 已触发（lint+test-core×4+test-full+test-wheel，**绿/红需 GitHub 复核**） |
| pkg_structure / pkg_usability / pkg_analytic | ✅ | ❌ **均未推送** | ❌ **从未跑过** |
| pkg_coverage | 零提交 | — | — |

- origin 仅 4 分支：`main / pkg_numeric / pkg_packaging / pkg_security`（9.25.0 时代遗留）
- B/D/E 的测试验证只存在于执行者本地汇报，**CI 从未背书**——验收时必须在本地独立复跑四件套，合并后 push main 触发 CI 四 job（工程约定：红一个都不算完成）
- C 轮中断点 = 覆盖率分析完成（_gap 产物齐）、测试编写未开始、无任何提交

## 三、待验收分支详情

### pkg_structure（B 结构轮，2 提交，8 文件 +1027/-629）
- `6e6bbb6` S1-S7 长函数拆分：`bc.apply_elimination`(160→拆分)、`boundary.naming`、`convergence.run_cantilever_convergence`(204)、`error_est.element_refinement_indicator`(181)、`gmsh_adapter._extract_regions`(198)、`runner.main`(125)、`verification.run_plane_verification`(155)
- `f5e5fe5` S8：`_safe_geo_source` Include 递归闭环（9.25.0 遗留第 1 项）
- 新测试：`tests/test_structure_split.py`(123 行)
- ⚠️ 验收要点：行为冻结（数值逐位不变）——拆分轮必须 golden 零变化

### pkg_usability（D 易用性，3 提交，7 文件 +458/-3）
- `3127c83` D1：CLI @段名引用（段标签通道，bc_apply 引用层）
- `2fc3f16` D2：输入入口引导（run.py --help 指南 + input_source 报错引导 + docs/input_entries.md 对照表）
- `3fd9454` D3：Windows GBK 乱码（输出统一 UTF-8，run.py 入口强制）
- 新测试：test_usability_round_d1/d2/d3.py（共 289 行）

### pkg_analytic（E 解析解验证，3 提交，4 文件 +467）
- `8db9978` E1：悬臂梁端部集中力 CST/Q4 vs Euler-Bernoulli
- `3732cab` E2：Kirsch 圆孔 Kt→3 收敛 + 网格方向不敏感
- `dd3ea50` E3+E4：Z2 效应指数 θ∈[0.8,1.2]（估计误差 vs 真实误差）
- 文档：`docs/analytic_verification.md`(186 行) —— 理论盲区（Z2/SPR 理论正确性无权威测试）就此闭环

## 四、C 覆盖轮现状（wt_coverage/ 产物齐，代码零提交）

```
wt_coverage/_gap_summary.txt   逐文件覆盖率 + 缺失行号（文本）
wt_coverage/_gap.json          同上（结构化，pct + miss 行号）
wt_coverage/_gaps.json         纯缺失行号列表（补测直接用）
wt_coverage/.coverage          coverage 数据文件
```

- 现状：boundary/ 子包与 element/registry 等普遍 **100%**；低区：`bc_apply.py` 84%、`boundary/topology.py` 89%、`boundary/geometry.py` 89%、`boundary/selectors.py` 87%、`bc.py` 89%、`element/base.py` 86%、`assembly.py` 90%
- 下一步：按 `_gaps.json` 缺失行号补测试，达标后跑覆盖率确认（hard gate 90%，CI test-full 背书）
- 注意：gap 产物基于哪个 commit 生成需验证（worktree 停在 788fce1，若 B/D/E 合并后行号会漂移，**建议合并后再重跑覆盖率分析**，别直接用旧 gap 补）

## 五、遗留事项（诚实清单，按优先级）

| 项 | 状态/建议 |
|---|---|
| **验收合并 B/D/E** | 第一优先。四件套（pytest 全量/探针 0 FAIL/漂移门/fuzz 500 0 problems）+ wheel 安装冒烟；合并后升版 9.26.0 + CHANGELOG（A 轮也欠着 CHANGELOG 记录） |
| **C 覆盖轮续做** | 合并后重跑覆盖率 → 按缺口补测 → 90% 硬门 |
| **docs/api_contract.md 单向量条目** | 9.25.0 遗留：`(3,)` 单向量为合法输入（6c 契约扩展的伴随更新） |
| **架构评估报告** | 曾答应用户的 docs 路线图（维护/性能/演进），未写 |
| **旧 worktree 清理** | pkg1-6 / cleanA / structure / coverage / usability / analytic 合并后可清（`git worktree remove --force` + `git branch -d`）；wt_drift_check 是空目录 |
| **系统 Python 残留** | `fem2d-q4 9.14.1` editable 安装指向主目录（不在项目边界，可提示用户卸载） |
| **解析解验证包** | E1-E4 已做（见第三节）——原"最高优先待办"已落地，若验收通过即闭环 |

## 六、指挥工作流（标准模板，按用户实际工作方式）

1. 建 worktree：`git worktree add -b <pkg_name> wt_<pkg_name> main`（放内层目录下）
2. 写 PROMPT.md（worktree 根）：背景（已定位根因）/ 任务（可验收）/ **文件边界（并行零冲突关键）** / 行为冻结区 / 工程约定 / 验收（判别性 + 四件套 + CI 全绿 + 另眼审查）
3. **通知用户"任务书就绪，请分发"——用户亲自分发到各 Claude Code 项目栏执行，执行结果由用户回报（不要自己派发 Agent！）**
4. 验收四件套（独立复跑，不采信汇报）：
```bash
python -m pytest -q -p no:warnings        # 全量
python scripts/audit_contract_probe.py    # 探针 0 FAIL
python scripts/regression_compare.py ...  # 漂移门（CI test-full 背书）
python scripts/fuzz_api.py 500            # fuzz 0 problems
```
5. 验收加 wheel 安装冒烟：`python -m build` → venv 安装 → 源码树外 import + 最小求解
6. 版本 + CHANGELOG + 打包：`python scripts/make_release_zip.py --full --out-dir <Downloads>`（标准 + full 两个），复制双桌面
7. 抽查关键 diff（越界改动、判别性真实性、调试残留 `_dbg_*`/`_st_*`/`_smoke_*`）

## 七、工程约定（派发时写进每个 PROMPT）

0. 每提交必全量 pytest 0 失败；任何一步全红 = 立即回滚该步
1. 修复流程：最小复现 → 修复 → **判别性测试（放回旧实现必须失败）** → 全量 pytest
2. 禁绝对阈值：`max(...,1.0)` / 固定 `1e-15` — 用相对尺度或 `np.finfo(float).tiny`（微尺度 1e-150 模型是硬指标）
3. DOF 约定：x→`2n`、y→`2n+1`
4. gmsh.exe 必须带 `-nt N` 和 `-2`
5. 静默错误比崩溃危险；注释不写历史叙事（只留"为什么"）
6. 不要重复修复已有注释标记 "审计" 的已修项
7. 无 gmsh 环境的测试必须 skip 而非失败

## 八、行为冻结区（改动需特殊程序：先金标准 → 逐位一致 → 回归 0.0）

- `fem2d/element/` 内核公式、`solver.py` 数值逻辑、`error_est.py` 公式——只改入口校验
- 边界识别管线（`fem2d/boundary/`）——只加插件/标签/输入；识别算法冻结
- 金标准快照 `tests/boundary_golden/` 锁边界行为；`tests/test_solve_refactor_lock.py` 锁求解；漂移门锁数值
- `von_mises`/`principal_stresses` 对 `(n,3)` 输入的输出逐位不变；`apply_penalty` 对有限输入数值逐位不变

## 八·补、全局代码地图（20260805c 摸底产物）

**`docs/CODE_MAP.md` = 本轮全局摸底的结构化产物**，含：顶层结构 / 主线数据流（5 阶段带函数行号）/ 全部 48+ 模块一句话职责 / solver.solve 阶段细节 / 测试体系 / 冻结区与契约。接手前先读它，20 分钟进入状态。

摸底要点（速览）：
- 数据流 5 阶段：`_resolve_input → _build_model → _apply_conditions → _analyze_and_report → _plot`（runner.py）
- 边界管线优先级：RegionRegistry → edge_labels → 纯拓扑 detect（boundary/naming.py:484）
- 内核：ElementKernel ABC + 注册表；Q4I=QM6 静力凝聚、Q4R=单点积分+沙漏稳定；Mesh 浅拷贝 kernel 防可变类属性污染
- 装配 4 路径（sparse/vectorized/lil 参考/expand）；载荷 = loads_core.assemble（F=Rc+ΣRs+ΣRb），loads.py 是纯 facade
- 求解：消去（默认）vs 罚；奇异守卫把秩警告转异常、非秩警告转发；刚体模态逐连通分量 rank 检查；全局平衡相对尺度判据
- 测试：金标准（boundary_golden + solve_refactor_lock）+ 漂移门（基线 7ee65fc）+ 探针 + fuzz + 分支锁

## 九、关键资产（20260805c 更新）

```
COMMANDER_HANDOFF_20260804/05/05b/05c.md   历史交接（勿改）；05c = 当前
docs/CODE_MAP.md                           全局代码地图（20260805c 摸底产物，接手先读）
PROMPT_*.md (仓库根, gitignore 不入库)      历史任务书，写新任务书时照抄结构
docs/api_contract.md                   契约表（含 G2 边界契约行；待补 6c 单向量条目）
docs/boundary_plugins.md / ci.md / analytic_verification.md / input_entries.md
tests/boundary_golden/                 边界金标准快照
tests/test_structure_split.py          结构轮拆分锁（B 轮新增，未合并）
tests/test_usability_round_d1~d3.py    易用性轮（D 轮新增，未合并）
tests/test_analytic_e1~e3.py           解析解验证（E 轮新增，未合并）
scripts/make_release_zip.py / audit_contract_probe.py / fuzz_api.py / regression_compare.py
.github/workflows/ci.yml               lint + test-core×4 + test-full + test-wheel
```

## 十、本任期教训（照做，别重蹈）

1. **🚨 崩溃真因已定位（2026-08-05 下午的"崩溃"）**：
   报错为 `Request too large (max 32MB). Accumulated images and attachments...`——**会话请求体超 Anthropic 协议 32MB 上限**。来源：长会话（xhigh）累积 + 附件（>1MB 文件、云图 PNG、二进制）进上下文。**与代理/DeepSeek 无关**，证据：①代理只注入几十字节字段，不生成体积 ②报错是协议层 413 文案 ③Windows 事件日志无进程崩溃记录（18:39 的 ConnectionRefused 只是重启窗口的正常间隙）。预防：**禁 Read/cat >1MB 文件（.inp/.msh/PNG/zip/gmsh.exe），验证用脚本摘要；模型无视觉禁读图（用 vision.py）；一个任务一个会话；膨胀即 /compact；出现反复自动压缩立即开新会话**。
2. **重置后历史恢复路径**：memory/ 目录原本为空 + 会话 jsonl 被清 → 唯一保险 = 本文档体系 + git + CHANGELOG + 知识库 lessons（日志只到 08-01，08-02~05 靠 handoff 覆盖）。本会话已把进度/结构/崩溃真因全部固化进本文档。
3. **摸底最快路径**：`git log --oneline` + `git worktree list` + `git log 788fce1..<branch>` + 分支 diff --stat —— 比读文档快且真。PROMPT_* 任务书在磁盘（gitignore 不入库），需要时磁盘直读。
4. 沿用 20260805b 教训：**指挥官只写任务书不亲自派 Agent**；验收自己跑；判别性红侧无法复现时诚实披露；curl 本机不可用（HTTP 000），GitHub API 用 python urllib + `git credential fill`；worktree 清理遇 CRLF 伪变更用 `--force`（内容为空时安全）。
5. 用户当前意图：**不开发新功能，优化完善**。新能力（模态/热应力）只做待办备选，等用户拍板，别主动推。

---

**最后一句**：项目处于"A 已合 + B/D/E 待验收 + C 半途"状态，原始对话已随重置丢失，本文件是唯一真相源。接手先验收合并三分支，再续 C 轮。祝顺利。
