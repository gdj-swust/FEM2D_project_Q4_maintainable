# 性能预期与实测 (2026-08-06)

> 本文件是 FEM2D 的性能基线: 各规模模型的时间/内存实测表、瓶颈解释、
> 用户建议、性能回归保护方法。数据由 `scripts/perf_benchmark.py` 生成
> (规则方形网格, 4 角节点全约束 + 右下角 x 向集中力)。
> **CI 回归门基线见 `scripts/perf_benchmark.py` 的 CI_BASELINE 常量** —
> 本文件表格只是人工可读记录, 不作为门的数据源 (解析 md 脆)。

## 1. 测量环境与方法

- 机器: Windows 11 (10.0.26200), AMD64, Python 3.13.5, numpy 2.5.1, scipy 1.18.0
  (基线录制 commit 1902d6a, 2026-08-06)
- 方法: `perf_counter` 单次测量; **本表数值 = 每档两遍独立运行取中位数**
  (与 CI_BASELINE 录制方法一致); 时间与内存分两次独立运行
  (`--mem` 走 tracemalloc, 时间数据偏大 ~2-4×, 仅用于内存量级判断)。
- 网格: 规则方形 (Q4: nx×nx 四边形; CST: 2×nx×nx 三角形), 单元数四舍五入到档位。
- solver: `auto` — ≤10 万 DOF 用稀疏直接 (direct), ≥20 万 DOF 用 Jacobi-PCG。
- 误差: 本表是单机双次实测, 波动 ±20% 属正常; 跨版本对比请用脚本重跑。

## 2. 规模-时间表 (实测, ms)

### CPS4 (Q4 四边形, 全积分)

| 档位 | 单元 | 节点 | DOF | nnz | 连接/几何 | K 组装 | 求解 | 应力 | L2 恢复 | Z2 估计 |
|------|------|------|-----|-----|-----------|--------|------|------|---------|---------|
| 1k   | 1,024 | 1,089 | 2,178 | 35.5k | 3.5 | 5.4 | 24.8 (direct) | 0.4 | 3.4 | 5.3 |
| 10k  | 10,000 | 10,201 | 20,402 | 331k | 29.1 | 47.6 | 242 (direct) | 4.5 | 39.1 | 43.8 |
| 100k | 100,489 | 101,124 | 202,248 | 3.34M | 332 | 477 | 919 (cg) | 44.4 | 710 | 513 |
| 300k | 300,304 | 301,401 | 602,802 | 10.2M | 1,007 | 1,541 | 3,353 (cg) | 154 | 3,388 | 1,548 |

### CPS3 (CST 三角形, 常应变)

| 档位 | 单元 | 节点 | DOF | nnz | 连接/几何 | K 组装 | 求解 | 应力 | L2 恢复 | Z2 估计 |
|------|------|------|-----|-----|-----------|--------|------|------|---------|---------|
| 1k   | 1,058 | 576 | 1,152 | 13.2k | 1.1 | 1.9 | 7.4 (direct) | 0.2 | 4.9 | 2.7 |
| 10k  | 10,082 | 5,184 | 10,368 | 122k | 8.0 | 11.7 | 75.6 (direct) | 2.2 | 50.9 | 14.5 |
| 100k | 100,352 | 50,625 | 101,250 | 1.21M | 94 | 116 | 295 (cg) | 24.6 | 647 | 158 |
| 300k | 301,088 | 151,321 | 302,642 | 3.62M | 293 | 359 | 1,238 (cg) | 81.5 | 2,641 | 419 |

**读法**:
- 组装/连接/应力/后处理全部随规模近线性 (向量化路径, 无 Python 层循环热点)。
- 求解 (direct) 是超线性 (稀疏分解 fill-in); 求解 (CG) 依赖网格条件数 —
  规则网格收敛快, 条件差网格可能慢 1-2 个数量级 (见 §3)。
- **L2 恢复在 CST 上曾是最慢的后处理项**: 200k 单元 8-13 s 的 Python 三层
  循环已向量化 (2026-08-04, 见 §5), 300k 单元 12.8 s → 4.2 s。

## 3. 瓶颈解释

### 3.1 CG 迭代数是本质 (唯一热点)

29 万单元 / 58 万 DOF 的真实模型剖析 (2026-08-03 交接): 导入 2.2 s /
边界 1.1 s / K 组装 1.4 s / 误差估计 0.2 s / **CG 求解 54 s (4161 迭代)**。
Jacobi-PCG 的迭代数由刚度矩阵的谱性质决定 (条件数 ∝ 网格质量/长宽比),
**不是 Python 层可以微优化的对象** — 同一模型在规则网格上 (上表) 相同
DOF 数只需 1.5-4 s。迭代数是算法本质, 微优化无法改变。

### 3.2 为什么 ILU 预条件不可行

58 万 DOF 下 SuperLU ILU 预条件的 fill-in 内存与分解时间实测不可行
(交接文档: 超时)。且 SuperLU ILU 不保证 SPD, 与 CG 的 SPD 前提冲突
(solver.py 已注释) — auto 默认 Jacobi (SPD), ILU 仅显式选择。

### 3.3 为什么 rtol 不能放松

已试验 rtol 1e-10 → 1e-8: 迭代 4161 → 3796 (9% 收益), 但 CG 残差
污染固定 DOF 反力 (~4.3e-3 N), 与平衡检查 `tol_rel` 同量级, 合法大型模型
被 ΣF 检查误杀。已回退 1e-10 并在 bc.py 注释原因。**收敛判据是正确性
的一部分, 不是性能旋钮。**

## 4. 用户建议 (什么规模用什么求解器)

| 模型规模 | 建议 |
|----------|------|
| ≤ 10 万 DOF | 默认 `direct` (消去法) — 快且精确, 反力干净 |
| 10 万 ~ 50 万 DOF | `auto` (自动选 Jacobi-PCG) — 内存友好 |
| > 50 万 DOF | 必须 CG (`--linear-solver cg`); 若迭代数爆炸, **优先检查网格质量** (长宽比/扭歪), 或粗网格先行 |
| 教学/粗网格 | 1k-10k 单元一切瞬发, 无需关心性能 |

迭代数预警信号: 求解报告里的迭代数 (如 4000+) — 与同规模规则网格
(几十次) 对比, 差距就是网格条件差的代价。**粗网格 + 高质量单元
(CST→Q4I) 通常比细网格 + CG 迭代爆炸更省时。**

## 5. 微优化记录 (2026-08-04, 包 4)

| 项 | 位置 | 改前 | 改后 | 结果 | 验证 |
|----|------|------|------|------|------|
| L2 fallback 批量堆叠 | stress.py `nodal_L2_projection` | 300k CST 12.8 s | 4.2 s (3×) | 逐位一致 | 7 场景 bitwise (CST/Q4R × 3 输入形态 + 非均匀 nqp else 分支) |

- **改了**: 逐单元兼容路径在 nqp/N 形状逐单元一致时 (CST/Q4R 的 L2 质量阵
  规则逐单元相同, 仅 dA 随单元面积变化) 堆叠为向量化 einsum + COO
  scatter。收缩序与散点顺序均不变 → 局部质量/右端/全局散点**逐位一致**
  (验证脚本 7/7 bitwise=True, 含第三方非均匀 nqp 内核走 else 分支与旧实现
  逐位一致)。
- **没改 (已 profile, 无安全空间)**:
  - Q4/Q4I/Q4R 批量组装的热点在 element 内核 einsum — 文件边界禁止
    触碰 element/ (内核公式)。
  - `assemble_sparse_vectorized` 的 rows/cols 每次调用重建 (~3% 收益,
    需引入 mesh 缓存失效管理, 风险/收益不划算, 不做)。
  - `error_est._element_energy_errors` 已按 5 万单元分块向量化, 100k 单元
    111 ms 由 einsum 主导, 无多余复制/转置。
  - `spr_recovery` 占 estimate 时间 ~78%, 但 spr.py 不在本包文件边界内。
  - L2 恢复剩余时间构成: splu 质量阵分解 (算法级, 不可省) + 内核
    `recovery_quadrature` 的逐单元 API 调用 (element/ 边界外)。

## 6. 性能回归保护 (scripts/perf_benchmark.py)

```bash
python scripts/perf_benchmark.py            # 默认 1k/10k/100k 档, 输出 perf_results.json
python scripts/perf_benchmark.py --all      # 1k/10k/100k/300k 全档
python scripts/perf_benchmark.py --mem      # 附加每阶段内存峰值 (tracemalloc)
python scripts/perf_benchmark.py --scale 300000
python scripts/perf_benchmark.py --ci       # CI 门: 1k/10k 与基线比较, 超阈值退出非 0
python scripts/perf_benchmark.py --ci --all # CI 门加 100k 档
python scripts/perf_benchmark.py --update-baseline  # 重测基线并打印可粘贴常量段
```

- JSON 含环境指纹 (platform/python/numpy/scipy) 与每档每阶段毫秒数 —
  未来改动直接重跑对比, 组装/求解时间不退化即可量化检查。
- 每次运行最后做 1e-150 微尺度冒烟: 全链路 (连接/组装/求解/L2/误差估计)
  必须全有限 — 性能路径存在绝对阈值时会在此暴露。
- 脚本把项目根显式加入 `sys.path`: editable install 指向其他 worktree 时,
  `python scripts/perf_benchmark.py` 会静默 import 到外部 fem2d 副本
  (实测发生过 — 基准数据全部失真, 修复后重测), 此防护防止再犯。

### 6.1 CI 性能门 (workflow test-perf job, 2026-08-06)

- `.github/workflows/ci.yml` 的 `test-perf` job 跑 `python scripts/perf_benchmark.py
  --ci` (1k/10k 两档; `--ci --all` 才加 100k), 任一阶段超阈值 → job 红;
  `--out` JSON (perf_results.json) 以 artifact 上传, 失败也传 (超阈值诊断用)。
- **无 gmsh**: perf_benchmark 自建规则网格, 全路径无 gmsh import (fem2d 顶层
  无急切 gmsh/matplotlib — gmsh 经 gmsh_adapter `_load_gmsh_module` 惰性加载),
  job 安装段只装 numpy/scipy (与 pyproject 地板版一致)。
- **阈值公式**: 每阶段阈值 = max(基线 × 4.0, 基线 + 200ms); 实测 ≤ 阈值 → 绿,
  **实测 > 阈值才红** (边界值不红, 防 flaky)。系数论证:
  - `×4.0`: 本地开发机 → ubuntu-latest 2 vCPU 标准 runner, BLAS 单线程路径
    典型 1.5-3×, 叠加 co-tenant 争用后 4× 以下均属"正常波动"; 门只拦量级级
    回归 (历史优化回退, 如 L2 向量化回退、组装复杂度退化, 均 ≥4-6×)。
  - `+200ms`: 微阶段 (1k 应力 ~0.4ms) 的固定噪声地板 — 乘性门在绝对毫秒级
    阶段上对 runner 启动/调度抖动过敏感; 两式在基线 66.7ms 处相等, 大阶段
    乘性项主导, 小阶段加性项主导。代价: 1k 档微阶段退化数百倍 (<200ms) 不红
    — 1k 是冒烟档, 10k 档才是结构回归的检测面 (10k 求解阈值 ≈ 968ms)。
- **基线**: 脚本内 CI_BASELINE 常量 (2026-08-06 录制, 每档两遍取中位数,
  meta 记录日期/commit/环境指纹)。刷新:
  `python scripts/perf_benchmark.py --update-baseline` → 输出可粘贴常量段,
  替换 CI_BASELINE 后提交。基线以常量为准, 本 md 表不是门的数据源。
- **防 flaky 兜底**: 若 runner 实测与本地基线比持续 >4× (环境变更/跑分机
  换代), 走刷新流程重录基线, 而不是放宽系数 — 系数放宽会同时放过真回归。
- `--ci` 禁止 `--mem` (tracemalloc 与计时语义冲突), 禁止默认 100k (耗时过重);
  基线损坏 (缺阶段键/非正数) 启动即明确报错并非零退出, 不静默当全绿。
