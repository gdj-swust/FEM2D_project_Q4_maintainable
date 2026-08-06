# 🎖️ 指挥官交接文档 (2026-08-06b, 最终版 — 9.28.0 发布完成)

> 前序: 20260805c (9.25.0 起点) / 20260806 (9.27.0 中继)。**本文件 = 当前唯一真相源**。
> 大优化轮 A/B/C/D/E + 审查轮 R 全部闭环。下一任接手: 先读本文件 + docs/CODE_MAP.md
> + wt_review/docs/review_20260805.md (审查报告, 后续工作的基准)。

## 一、项目现状 (2026-08-06)

| 项 | 值 |
|---|---|
| 版本 | **9.28.0** (main = 617c002, CI run #31081427091 七 job 全绿) |
| 发布链 | 9.26.0 (B/D/E) → 9.27.0 (C 覆盖轮) → 9.28.0 (审查轮五包修复 + **P轮Ⅰ 四包**) — 已推送 + CI 全绿 + zip 已打包 (标准 319/full 321 文件) |
| 测试 | **1552+ 函数 (134 个 test_ 文件, pytest 收集 1628)**, 覆盖率 **98.3%** (CI 官方命令 --cov=fem2d 实测 98.25%), 漂移门 0.000e+00 |
| 审查评分 | 29 条发现全部闭环 (P0/P1 = 0), 剩余为"已知取舍" (见第五节) |
| 用户方向 | 不开发新功能, 优化完善 |

## 二、审查轮结果 (已完成 — 下一任工作的基准)

- 报告: `wt_review/docs/review_20260805.md` (355 行, 两轮 29 条: P2×3 + P3×26)。
  审查分支 pkg_review 已合入? **否** — 报告文件随 main 的 worktree 保留, 内容已入库? **否**
  (报告是 wt_review 的未合并工作区产物, docs/review_20260805.md 尚未进 main — 需要时
  git add 到 main 或从 wt_review 直接读)。
- **数值内核零公式错误** (8 组差分 + Bathe 对照); 29 条全部修复于 9.28.0:
  - R-α (10 条): complex 静默错值(P2)/bool/str 索引/float32 精度/K 形状/重复 traction
    双倍/fix_node O(n²)/fuzz 分支重复/element 冻结区入口守卫×2
  - R-β (2 条): **Merge 绕过 SystemCall(P2, 实证)**/行首锚点
  - R-γ (2 条): merge 共线判据(P2 真 bug, 金标准 21/21 零 diff)/角弧悬崖文档化
  - R-δ (12 项): 交互 GBK/裸 input/内孔重叠/契约表 5 条/探针增强/CODE_MAP 符号引用/死参数
  - R-ε (3 条): Q4R 物理锚点/Q4 积分点锁定/冒烟断言

## 三、AI 错误教训 (本任期踩坑记录 — 照做别重蹈)

1. **三轮 CI 红灯链** (2026-08-05):
   - lint×2: C 轮新测试未使用 import — 根因: **验收没跑 lint** → 验收必须补跑
     `ruff check fem2d/ scripts/ tests/ run.py run_demo.py` + mypy + vulture + self-test
   - test-core Install×4: GitHub runner→PyPI 网络故障 — 本地解析全成功 (Windows/Linux/
     pip 版本矩阵), 连续 4 次重跑全挂 → 加固 ci.yml (PIP_DEFAULT_TIMEOUT=120 +
     --retries 10) → 恢复。**判断网络问题的证据**: test-full (无 numpy 约束) 每轮成功
     而 test-core (带 numpy 约束) 每轮失败 = 非随机抖动
   - 无 gmsh skip 缺陷: C-α 1 个测试在无 gmsh 环境不 skip — 根因: **验收只在有 gmsh
     的本地跑** → 验收必须补"无 gmsh 模拟全量" (假 gmsh 模块法: 临时目录放
     `gmsh.py` 内容 `raise ImportError(...)`, PYTHONPATH 前置跑 pytest)
2. **CI 日志 403 误判** (2026-08-06): 以为凭据权限不足 (绕了用户从网页复制日志),
   实际是 **urllib 自动跟随 302 重定向时带错 Authorization header** — logs 端点
   返回 302 到 blob 存储, 手工跟随 (NoRedirect opener 拿 Location) + 丢弃
   Authorization 请求 → 成功。凭据 scope 含 repo 本来就够。
3. **任务书设计失误**: C-α 任务书沿用旧任务书 C2/C3 段, 与 C-β 任务书重复 →
   两个执行者并行改了 fuzz_api.py/api_contract.md → 合并冲突。教训: **并行拆包时
   每个文件必须只属于一个包**, 任务书里"文件边界"写死, 交叉文件用"区域不重叠 +
   顺序合并"处理 (bc.py 的 R-α 校验 vs R-δ 死参数 = 不同区域, git 自动合并)。
4. **A9 测试跨平台位级断言** (2026-08-06): test_solve_bit_exact_gold 用硬编码浮点
   字面量逐位断言 → Linux OpenBLAS vs Windows MKL 最后 1-2 位不同 → CI 红。教训:
   **涉及稀疏求解/BLAS 的断言必须用相对容差** (rtol=1e-13); 零值附近 (支反力残差
   噪声 ±3.6e-12 符号随平台翻转) 加 atol (1e-9) 兜底 — 位级锁只适合同平台
   (test_solve_refactor_lock 是本地录制的, 注意它的 CI 可移植性风险)。
5. **无 gmsh 模拟的本地假象**: main 主目录 tools/gmsh.exe (gitignore 未入库, 手动
   放置) → 无 gmsh 模拟下 .txt 生成成功→API 回读失败→exit 2; worktree 无 gmsh.exe
   → 走"不可用"路径→exit 1 (与 CI 一致)。**验收无 gmsh 模拟必须在干净 worktree 跑**,
   或临时移走 tools/。
6. **32MB 崩溃点** (2026-08-05, 前任遗留): 会话请求体超 32MB (Anthropic 协议 413),
   元凶 = 长会话累积 + 附件 (>1MB 文件/图)。预防: 禁 Read/cat >1MB; 输出重定向;
   膨胀即 /compact; 反复压缩开新会话。**本任期靠此纪律撑过 100+ 轮后台任务未崩**。

## 四、提示词细节 (下一任写任务书/分发时照抄)

### 工作流铁律
1. **指挥官只写任务书, 绝不自己派发 Agent** — 用户亲自分发到各项目栏, 执行者
   回报, 指挥官独立复跑验收 (不采信汇报)
2. 用户偶尔主动问"你能不能自己派" — 需诚实说明 Agent 与项目栏差别 (监督缺失),
   给方案 A/B/C 让用户拍板
3. 验收四件套 + **新增两项**: ① lint 全命令 ② 无 gmsh 模拟全量 (干净 worktree)
4. CI 查询**短间隔轮询 (30-60s), 不设长 sleep** (用户明确指导: 反复轮询不浪费)

### 任务书模板 (PROMPT.md, 每包一份, 高强度)
```
# PROMPT — <包名>: <一句话任务>
## 背景 (基线版本/分支/审查报告出处/现状数字)
## 任务 (逐条: 发现编号/位置(文件:行号)/现象/复现/修复方向/判别性要求)
## 文件边界 (本包独占文件清单 + 禁碰清单 + 交叉区域注明) ← 并行零冲突关键
## 行为冻结区 (element内核/solver数值/error_est公式/边界识别 — 特殊程序)
## 工程约定 (0-7 条 + 无 gmsh skip + lint + 禁绝对阈值)
## 验收 (判别性红侧自证 + 四件套 + 无 gmsh + lint + 提交拆分 + 回报格式)
```

### 分发格式 (粘贴文本, 一段话可复制)
```
你是 FEM2D <轮次>执行者 (<包名>)。工作台: <绝对路径> (分支 <branch>).
先读 <PROMPT.md 路径> 全文。任务: <精炼 3-5 行>。
文件领地: <独占文件>。纪律: <验收项>。回报: <格式>。
```

### 合并顺序原则
- 独立低风险先合; 大包居中; **冻结区/金标准变更最后** (金标准独立收尾)
- 同文件不同区域改动的包: 顺序合并让 git 自动处理, 提交注明改动区域

## 五、遗留事项 (诚实清单, 按优先级)

| 项 | 状态/建议 |
|---|---|
| **审查报告入库** | wt_review/docs/review_20260805.md 未进 main — 建议 git add 到 main (归档) |
| **R-γ C2 角弧悬崖决策** | 执行者已文档化方案 (docs/boundary_plugins.md 标注) — 用户拍板"接受现状"即可闭环 |
| **旧 worktree 清理** | 12 个 (pkg1~4/6, cleanA, structure, usability, analytic, coverage, cov_b, cov_c, fix_a~e, review) — 已全部合并可清; `git worktree remove --force` + `git branch -d`; 保留 wt_review 至报告归档 |
| **系统 Python 残留** | fem2d-q4 9.14.1 editable 指向主目录 — 提示用户卸载 |
| **性能回归门** | perf_benchmark 未绑 CI (审查弱势 5) — 可选增强 |
| **清洗器模型** | 黑名单 → 白名单重构 (审查弱势 3) — 设计取舍, 用户拍板 |
| **文档漂移守卫** | CODE_MAP 已改符号引用, 契约表数字人工维护 (弱势 4 部分落实); P-λ 建议采纳: 文档数字判别性测试 (test_fix_d11_doc_map.py) 的锁定值纳入"docs 数字同步"例行范围 |

## 六、关键资产地图

```
COMMANDER_HANDOFF_20260805c.md / 20260806.md / 20260806b.md  历史交接 (06b = 当前)
docs/CODE_MAP.md          代码地图 (main 内, D11 已改符号引用)
wt_review/docs/review_20260805.md  审查报告 (355 行, 29 条发现 + 复现 + 自查)
wt_fix_a~e/PROMPT.md      五包任务书 (已完成, 存档参考)
scripts/make_release_zip.py / audit_contract_probe.py / fuzz_api.py / regression_compare.py
.github/workflows/ci.yml  lint + test-core×4 + test-full + test-wheel (pip 加固已入)
tests/boundary_golden/ + test_solve_refactor_lock.py  金标准 (同平台位级锁)
```

## 七、P 轮Ⅰ 记录 (2026-08-06, 四包闭环 — 基线 main=617c002)

### 验收结果 (指挥官独立复跑, 不采信汇报)

| 包 | 内容 | 提交 | 验收 |
|---|---|---|---|
| P-θ | bc_apply 整体 callable 体力每积分点双重求值 → 分量 lambda 精确缓存 (单次求值) | 93d2d10 | 四件套/lint/无 gmsh/漂移门 0.000e+00 ✅ |
| P-ι | API 入口校验: input_source 类型守卫 (bytes 排除) / visualize tag 白名单从 PLOTS 派生 / run_demo EOFError 移除 | 5f328a3 | 同上 ✅ |
| P-η | error_est: 统一 NaN/Inf 入口防护 + _logaddexp_scatter 排序+reduceat + 常数体力/面力批量向量化 + 批量边界法向 ULP 判据 | a3b90a5 | diff 全文审查 + 同上 ✅ |
| P-λ | README/docs 数字统一 9.28.0 + make_release_zip --split 拆 4 包 (source/runtime-win64/models/testdata) + 守恒判别性测试 | e118a1c + e1f42a0 | 同上 ✅ |

- 三包 (θ/ι/η) 已按 θ/ι→η 顺序合入 main=617c002, 推送后 CI 七 job 全绿; P-λ 最后合入 (发布链)。
- **pytest 9.1.1 注意**: `-q` 模式抑制最终 "N passed" 汇总行 (pytest 9 行为, `-v` 才显示)。
  验收判据改用「进度到 [100%] + 0 FAILED/ERROR + exit 0」三重证据。
- **P-η 已知非阻塞事项**: 常数边与 callable 边分组后 `.at` 累加顺序与逐边字典序不同,
  同 eid 双类贡献时 logaddexp 理论 1 ULP 级差异 (仅 error_est 细化指标对数值, 不在钉死漂移值内)。

### P-λ 执行者回报 — d11 越界决策 (指挥官已验收拍板)

> ⚠️ 执行者如实上报的越界: test_fix_d11_doc_map.py 在任务书"禁碰测试文件"名单内, 但它是
> R-δ 轮的 CODE_MAP 数字判别性测试, 硬编码锁定旧值 (116 文件/1401/~1400 函数)。任务 2 强制
> CODE_MAP 数字前进到 9.28.0 后该测试必红, 与"pytest 全绿 + 无 gmsh 全绿"验收直接冲突。
> 处理: 仅同步锁定值到新实测 (保持"回退必红"的判别性语义, 未削弱测试), 随 docs 提交合入。

指挥官核验: 改动恰为 4 处锁定字符串 (116→133 文件 / 1401→1546 函数), "回退必红"语义保留 →
**决策: 接受**。执行者建议 (已采纳入规): 后续轮次将文档数字判别性测试的锁定值纳入
"docs 数字同步"例行范围。

### 数字同步提交 (任务书"顺序注意"规则)

P-λ 分支新增 test_p_lambda_release_manifest.py (+1 文件 +6 函数 +165 行) 后, 执行者按任务书
口径测的 main 基线数字 (133/1546/27037) 与分支实测 (134/1552/27202) 差一, 按规则在最终合入前
追加**纯数字同步提交** e1f42a0 (README/CODE_MAP/d11 锁/ARCH 同步, 提交信息注明)。
指挥官独立复测数字: 收集 1628 / 覆盖率 98.25% / 探针 151 PASS / fem2d 18379 行 — 全部吻合。

### P 轮Ⅱ 待发 (7 个性能包, P轮Ⅰ 全闭后发)

P-α solver K·u / P-β mesh 刚体模态缓存 / P-γ topology O(L²) / P-δ geometry 向量化 /
P-ε spr+stress / P-ζ perf_benchmark 绑 CI / P-κ verify_all.py

## 八、给下一任的三句话

1. 代码处于"数值可信、门禁严密、缺陷受控"状态 — 29 条审查发现全部闭环,
   剩的是已知取舍, 别把已知当未知去翻。
2. 先归档审查报告 + 清 worktree + 让用户拍板 C2, 然后项目进入"维护模式"
   (用户方向: 不开发新功能, 优化完善)。
3. 防崩溃纪律 + 短轮询 + 验收补 lint/无 gmsh — 这三条是本任期用血泪换的,
   别重蹈。
