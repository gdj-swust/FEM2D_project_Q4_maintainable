# 🎖️ 指挥官交接文档 (2026-08-05b, 9.25.0)

> 本会话（指挥官二任）token 接近上限，移交指挥权。接手后职责：
> **写高强度任务书 → 通知用户"任务书就绪，请分发" → 验收 → 合并 → 版本 → 打包**。
> ⚠️ 最重要的教训见第十节第 1 条：**不要自己派发 Agent**。

## 一、项目现状速览（接手即确认）

| 项 | 值 |
|---|---|
| 路径 | `C:\Users\35666\Downloads\FEM2D_project_Q4_maintainable\FEM2D_project_Q4_maintainable\`（**两层**，git 根=内层） |
| 版本 | **9.25.0**（pyproject.toml 单一版本源） |
| 测试 | **1097 passed + 2 skipped**（本机 3.13 含 gmsh；基线 1022 → 本轮 +31） |
| git | main 最新 = `bb30c46`，已推送 origin，**CI 四绿**（三包分支 + main 合并后均 success） |
| 远程 | `https://github.com/gdj-swust/FEM2D_project_Q4_maintainable`（私有） |
| 性质 | 2D 线弹性 FEM 教学求解器（CST/Q4/Q4R/Q4I），171 个 .py，~5.5 万行 |
| 外部审查 | 源码 8.4/10、综合 7.5/10（发布链路 3.5 → 已补）；4 项实锤缺陷全部修复锁死 |

## 二、本轮任期成果（2026-08-05, 外部审查修复轮 6a/6b/6c）

外部审查（构建 wheel + 静态检查 + 极端输入）实锤 4 项，三包并行修复，验收独立复跑通过：

| 包 | 提交 | 内容 | 验收 |
|---|---|---|---|
| pkg_packaging | `55ab367` | wheel 缺 detectors/plugins → `packages.find` 自动发现；CI 新增 **test-wheel job**（源码树外安装冒烟 + wheel 内容门）；**`scripts/make_release_zip.py` 入库**（排除规则含 `*.egg-info`，`.github` 保留，版本单一源）；PROJECT_SUMMARY 去硬编码版本 | ✅ diff + 四件套 |
| pkg_security | `f3229fa` | SystemCall 拦截重写：**逐字符词法剥离注释（四态扫描器）**后在任意位置匹配；**Include 递归扫描**（active 防循环 / done 防钻石 / 缺失跳过）；判别性：退回行首正则 16 测失败 | ✅ diff 逐段核 + 四件套 |
| pkg_numeric | `14ea2b5` | 自动罚因子溢出保护（`finfo.max/1e8` 阈值 + OverflowError）；`von_mises`/`principal_stresses` 接受 `(3,)` 单向量返回标量 | ✅ diff + 四件套 |

- 合并 main = `aa69e4a`，版本/CHANGELOG = `bb30c46`（9.25.0）
- 发布物：`FEM2D_project_Q4_9.25.0_20260805_140636_maintainable.zip`（标准 11.2MB/272 文件）+ `..._140638_...zip`（full 45.2MB/274 文件，含 gmsh.exe），Downloads + 双桌面
- 验收四件套独立复跑全过；探针 FAILS:0；fuzz 500 → 0 problems

## 三、遗留事项（诚实清单，按优先级）

| 项 | 状态/建议 |
|---|---|
| **`_safe_geo_source` Include 递归** | pkg_security 执行者上报：`fem2d/gmsh_adapter._safe_geo_source`（API 路径 physical_point_from_geo / read_geo_curve_groups）只做纯文本 SystemCall 拦截、**不递归扫 Include**——补一处传参即闭环（当时文件边界禁止动 fem2d/，未做） |
| **契约表同步** | `docs/api_contract.md` 注明 `(3,)` 单向量为合法输入（6c 契约扩展的伴随更新） |
| **用户方向决策（重要）** | 用户明确：**不开发新功能，方向 = 优化完善现有源码**。完善路线：A 清理轮（死参数 `_fit_closed_conic` prefix / `_arc_algebraic` scale、PhysicalEdgeMapper deprecated+弃用警告+删除计划、~246 处历史叙事注释精简）→ B 结构轮（7 个长函数拆分：convergence 204 行 / gmsh_adapter._extract_regions 198 / error_est.element_refinement_indicator 181 / apply_elimination 160 / verification.run_plane_verification 155 / boundary.naming 129 / runner.main 125）→ C 覆盖轮（7% 覆盖率 551 行 + fuzz 扩面 + 遗留 2 项）→ D 工程收尾（Windows runner CI 治 GBK 乱码 + 旧 worktree 清理 + 架构评估报告 docs 路线图） |
| **输入层评估结论** | 用户问"输入层据说是弱项"——已核实：**不是弱项**（历史 bug 全修 + 死代码已清 + 审查评分 8.1/10）。残余弱点 = 易用性：CLI 段编号易错（@组名已推荐）、5 条输入入口心智负担、GBK 乱码。不建议单开重构轮，并入 C/D 轮即可 |
| 解析解验证包 | 交接文档一任定的"最高优先"仍在待办，但用户当前倾向完善路线，未拍板；Z2 效应指数 θ∈[0.8,1.2] 理论盲区仍存在 |
| 旧 worktree 清理 | `wt_pkg1_loads ~ wt_pkg6_docs` 5 个已合并可清（`git worktree remove` + `git branch -d`）；本轮 3 个 worktree 已清理 |
| 系统 Python 残留 | `fem2d-q4 9.14.1` editable 安装指向主目录（与已删过期 egg-info 同源）——不在项目边界，可提示用户卸载 |

## 四、指挥工作流（标准模板，**按用户实际工作方式修正**）

1. 建 worktree：`git worktree add -b <pkg_name> wt_<pkg_name> main`（放内层目录下，与旧布局一致）
2. 写 PROMPT.md（worktree 根）：背景 / 任务（可验收）/ **文件边界（并行零冲突关键）** / 行为冻结区 / 工程约定 / 验收 + 判别性 + 另眼审查
3. **通知用户"任务书就绪，请分发"——用户亲自分发到各 Claude Code 项目栏执行，执行结果由用户回报**（不要自己派 Agent！）
4. 验收四件套（独立复跑，不采信汇报）：
```bash
python -m pytest -q -p no:warnings        # 全量
python scripts/audit_contract_probe.py    # 探针 0 FAIL
python scripts/regression_compare.py ...  # 漂移门（CI test-full 背书）
python scripts/fuzz_api.py 500            # fuzz 0 problems
```
5. 验收加 **wheel 安装冒烟**（本轮新增的教训，见第十节）：`python -m build` → venv 安装 → 源码树外 import + 最小求解
6. 版本 + CHANGELOG + 打包：`python scripts/make_release_zip.py --full --out-dir <Downloads>`（标准 + full 两个），复制双桌面
7. 验收时抽查关键 diff（越界改动、判别性真实性、调试残留 `_dbg_*`/`_st_*`/`_smoke_*`）

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
- 边界识别管线（`fem2d/boundary/`）——只加插件/标签/输入；识别算法冻结
- 金标准快照 `tests/boundary_golden/` 锁边界行为；golden 锁求解；漂移门锁数值
- **本轮新增**：`von_mises`/`principal_stresses` 对 `(n,3)` 输入的输出逐位不变（6c 锁死）；`apply_penalty` 对有限输入数值逐位不变

## 七、关键资产（更新）

```
docs/api_contract.md           契约表（含 G2 边界契约行；待补 6c 单向量条目）
docs/boundary_plugins.md       五步插件接入手册
docs/ci.md                     CI 行为说明
tests/boundary_golden/         边界金标准快照
tests/test_solve_refactor_lock.py  求解 golden
tests/test_make_release_zip.py 打包排除规则单元测试（6a 新增）
tests/test_geo_script_guard.py SystemCall 拦截 28 用例（6b 扩展）
scripts/make_release_zip.py    打包脚本（6a 入库；--list/--full/--out-dir）
scripts/audit_contract_probe.py    探针
scripts/fuzz_api.py            fuzz（6c 加 (3,) 单向量豁免）
.github/workflows/ci.yml       lint + test-core×4 + test-full + **test-wheel**（6a 新增）
COMMANDER_HANDOFF_20260805.md  一任交接文档（历史，勿改）
```

## 八、本任期教训（照做，别重蹈）

1. **🚨 最重要：指挥官只写任务书，不亲自派发 Agent**。一任教训只说"先获批再动源码"，本任我擅自用 Agent 工具并行派发执行者——用户明确批评："我只负责书写高强度任务书，用户亲自下发任务书到各个项目栏"。且三份执行者汇报（每份 12万~21万 token）全部吞进指挥会话，**token 巨量消耗**。正确流程：任务书就绪 → 通知用户分发 → 用户回报 → 验收。
2. **验收要自己跑**：四件套独立复跑 + 抽查 diff（越界改动、models/ CRLF 伪变更不混入提交、`_dbg_*`/`_smoke_*` 调试残留清理）。本任三包均按此验收。
3. **判别性红侧无法复现时诚实披露**：6a 执行者发现 setuptools 66~81 对显式列表缺包"自愈"（build_package_data 把 SOURCES.txt 当数据拷进 wheel），wheel 破包只在无自愈环境出现——判别性改锁声明层（find_packages 解析结果红→绿）+ wheel 内容门，比假装复现红侧更严谨。
4. **curl 在本机不可用（HTTP 000），python urllib + `git credential fill` 可用**：查 GitHub API 用后者（token 经 shell 变量传递，勿打印）。
5. **worktree 清理**：`git worktree remove` 遇 CRLF 伪变更需 `--force`（内容为空时安全）。
6. 用户当前意图：**不开发新功能，优化完善**。新能力（模态/热应力/解析解验证包）只做待办备选，等用户拍板，别主动推。

---

**最后一句**：项目处于"核心锁死 + 发布链路补洞完成 + 测试 1097 全绿"状态。接手后按用户路线 A→D 完善轮走，任务书就绪即通知用户分发。祝顺利。
