# 🎖️ 指挥官交接文档 (2026-08-06, 审查轮中继)

> 前序: COMMANDER_HANDOFF_20260805c.md (9.25.0 大优化轮起点)。本文档 = 当前唯一真相源。

## 一、项目现状 (2026-08-06 凌晨)

| 项 | 值 |
|---|---|
| 版本 | **9.27.0** (main = 4e5a40d, CI run #28 七 job 全绿) |
| 发布 | 9.26.0 (B/D/E 合并) → 9.27.0 (C 覆盖轮三包: 98.4% 覆盖率 / fuzz 41 分支 / 架构文档) |
| 测试 | 1118+ 全绿, 覆盖率 98.4%, 漂移门 0.000e+00 |
| 用户方向 | 不开发新功能, 优化完善 |

## 二、审查轮 (已完成) — 29 条发现

- 审查报告: `wt_review/docs/review_20260805.md` (355 行, 两轮: 第一轮 16 条 +
  复查轮 13 条)。审查提交: wt_review 分支 pkg_review (3305fdf + c08896c)。
- 统计: **P0=0, P1=0, P2=3, P3=26** + 可优化点 12 + 体系弱势 5。
- 数值内核零公式错误 (8 组差分全过, Bathe 对照逐公式核对); 发现集中在:
  ①输入校验 dtype 盲区 (P2 complex 静默错值) ②安全清洗器 Merge 绕过 (P2, 已实证)
  ③边界识别 _merge_adjacent_lines 共线判据 (P2 真 bug, 已复现, 冻结区)
  ④交互/契约/文档/测试强度 (P3 群)。

## 三、修复五包任务书 (已就绪, 待分发 — 这是接手第一件事)

| 包 | 工作台/分支 | 内容 | 条数 | 风险 |
|---|---|---|---|---|
| R-α | wt_fix_a / pkg_fix_a | 输入校验 dtype 家族 + mesh 状态 + element 冻结区入口校验×2 | 10 (P2×1+P3×9) | 中 |
| R-β | wt_fix_b / pkg_fix_b | 清洗器: Merge 绕过 SystemCall + Mesh.Format 锚点 | 2 (P2×1+P3×1) | 低 |
| R-γ | wt_fix_c / pkg_fix_c | 边界冻结区: merge 共线判据 (金标准会变, 特殊程序) + 角弧悬崖确认 | 2 (P2×1+P3×1) | 最高 |
| R-δ | wt_fix_d / pkg_fix_d | 交互/向导 (GBK/裸 input/内孔重叠) + 契约表 5 条 + 探针增强 + CODE_MAP + 死参数 | 12 | 低 |
| R-ε | wt_fix_e / pkg_fix_e | 测试强度: Q4R 物理锚点 / Q4 积分点 / 冒烟断言 | 3 | 低 |

- 每包 PROMPT.md 含: 逐条发现 (文件:行号/现象/复现/修复/判别性) + 文件边界 +
  冻结区 + 工程约定 + 验收 (四件套 + 无 gmsh 模拟 + lint + 判别性红侧自证)。
- **合并顺序 (验收通过后)**: R-β → R-ε → R-α → R-δ → R-γ (γ 最后, 金标准独立收尾;
  δ 与 α 在 bc.py/convergence.py 有同文件不同区域改动, 顺序合并)。

## 四、工作流铁律 (沿用)

1. 写任务书 → 通知用户分发到各项目栏 → 执行者回报 → **总指挥独立复跑验收** (不采信汇报)
2. 验收四件套 + **新增两项 (2026-08-05 教训)**: ① lint 全命令复跑 (ruff/mypy/
   vulture/self-test) ② 无 gmsh 模拟全量 (PYTHONPATH 假 gmsh 模块法, 0 失败)
3. CI 查询用短间隔轮询 (30-60s), 不设长 sleep
4. CI 日志读取: logs 端点 302 后丢 Authorization 跟随 (凭据 scope 本来够) — 见记忆
5. 防崩溃: 不 Read >1MB; 输出重定向; 膨胀即 /compact; 反复压缩开新会话

## 五、待办

- **第一优先**: 分发五包修复任务书 → 验收 → 合并 → 升版 9.28.0 + CHANGELOG → CI 全绿
- 收尾: 9 个旧 worktree 清理 (wt_pkg1~4/6, wt_cleanA, wt_structure, wt_usability,
  wt_analytic — 全部已合并可清; wt_coverage/wt_cov_b/wt_cov_c 已合并可清;
  wt_review 保留至报告归档), 系统 Python 9.14.1 editable 残留提示
- 发布 zip 打包 (make_release_zip.py --full) — 9.27.0 发布时未打包, 补
- 审查轮遗留决策: R-γ C2 角弧悬崖 (接受现状文档化 vs 修复) — 执行者给方案, 总指挥拍板

## 六、本任期教训 (2026-08-05 晚)

1. **三轮 CI 红灯链**: lint (验收漏跑) → 网络 (GitHub runner→PyPI, 加固 ci.yml
   PIP_DEFAULT_TIMEOUT=120 + --retries 10) → 无 gmsh skip 缺陷 (验收漏模拟)。
   教训: 验收必须补 lint + 无 gmsh 模拟 (已入记忆 acceptance-must-cover-ci-scenarios)
2. **CI 日志 403 真相**: 不是凭据权限不足, 是 urllib 自动跟随 302 带错 header —
   手工跟随丢 Authorization 即可读 (已入记忆 github-ci-logs-access)
3. 审查报告质量: 本轮审查 (pkg_review) 每条发现带复现/冻结区标注/自查, 判别性
   验证充分 — 报告可作为后续修复的验收基准
