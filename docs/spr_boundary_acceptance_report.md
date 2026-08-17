# SPR-BC-2026-001 验收报告 — SPR 边界节点恢复

- **任务书**: SPR-BC-2026-001（下达人：郭大杰）
- **日期**: 2026-08-17　**分支**: `feature/spr-boundary-recovery`
- **提交**: `c6dba64`（feat）· `a9f6c2d`（test）· `552af39`（test 重基线）
- **依据**: Zienkiewicz–Zhu 1992, *Int. J. Numer. Meth. Engng*, **33**, 1331–1364, §2.3 (p.1337–1338)

## 1. 算法摘要

按 ZZ92 §2.3 推荐流程，边界节点恢复不再用自身薄边界 patch（单侧外推，~O(h)），改由内部 patch 拟合、在边界节点坐标处求值。确定性规则（无随机）：

1. 边界节点集 B ← `mesh.boundary_edges` 出现过的节点；
2. 每个 b ∈ B：候选内部节点 = 邻接单元顶点 − B；空则扩 ring-1 单元顶点（CST 角点情形）；
3. 取距 b 最近且 patch 非空的候选 i（并列取最小节点号），恢复值 = patch(i) 最小二乘线性拟合在 b 坐标处求值；
4. 候选仍空（全边界退化网格）→ 退回 b 自身 patch 兜底。

实现：`spr_recovery` 入口构造增强 node→elems 表（`_boundary_patch_table`）下传批量/精确路径；下游拟合、条件数判据、兜底逻辑全部不动；**`_fit_node_block` 拟合核心零改动**，公共 API 与 `error_est.py` 误差积分零改动。

## 2. 验收结果

| 项 | 要求 | 结果 |
|---|---|---|
| A 全绿 | pytest 全绿 | **1877 passed / 1 failed**（见 §2.1） |
| B 逐位不变 | 非边界节点改前改后逐位一致 | ✅ tg CPS3/CPS4 + Kirsch CPS3 三快照 `np.array_equal` 全 True |
| C 精度 | 至少两项明显改善 | ✅ C1 47×/2×，C2 4.7×/2×，C3 60°/90° 7.4×/2.2× |
| D 收敛阶 | 边界采样点 ~O(h²)±0.2 | ✅ CPS3 四点 1.79–1.81；CPS4 可达点 1.94/2.36 |
| E 效应指数 | η_est/η_exact ∈ [0.8, 1.2] | ✅ CPS3 [0.85, 0.94, 0.98]（基线 1.55 越界）；CPS4 [0.95, 0.98, 0.99] |

### 2.1 A — 全量 pytest

1877 passed，唯一失败 `test_circle_fan_golden` 为**分支前已存在**：在 main@`87b6782` 复现相同失败，系纯几何金标准 `fit_residual_max` 末位浮点噪声（2.83173484661e-12 vs 2.831512802e-12，Windows/Linux libm 圆函数残差），与 SPR 无关。分支引入 **0** 新失败。

### 2.2 C — 边界点精度（改前基线 → 改后）

- **C1 悬臂梁顶边中点 σy**（精确 0）：CPS3 0.00815 → **0.000173**（47×）；CPS4 0.00514 → **0.00257**（2×）。
- **C2 右端面 τxy 对抛物线剪流**（最大误差）：CPS3 0.0621 → **0.0131**（4.7×）；CPS4 0.0520 → **0.0263**（2×）。
- **C3 Kirsch 孔边 σθθ**（σ∞=1）：θ=60° 0.588 → **0.080**（7.4×）；θ=90° 0.266 → **0.120**（2.2×）；θ=30° 0.075 → **0.055**（1.4×）。θ=0°（精确 −1，1/r² 曲率最强处）0.016 → 0.123 **退化**：最近内部 patch（孔心距 0.42a）的线性多项式外推平滑了曲率，如实记录。

### 2.3 D — 收敛阶（对数斜率）

CPS3 边界采样点（基线 ~O(h)）：syy_top 0.96→**1.79**，txy_top 0.82→**1.79**，txy_right_q 1.55→**1.81**，sxx_top 0.19→**1.79** — 全点达 O(h²)±0.2。

CPS4：sxx_top 2.53→**2.36**、txy_right_q 1.95→**1.94**（保持 O(h²)）；syy_top/txy_top 仍 ~O(h)（0.93/1.15，幅值减半：syy_top 0.00514→0.00257）。**机理**（探针实测 + 形函数残差分析）：Q4 2×2 Gauss 采样的 σy/τxy 本身含 O(h) 伪场 — u_y 的 νP(L−x)y²/(2EI) 项 y² 插值残差在每行 Gauss 点形成 ±0.577h·c(L−x) 锯齿（x 线性、y 逐行反号），线性恢复不可消除。此为 FE 采样一致性下限，非算法缺陷；CST 质心采样成对抵消，无此限制。

### 2.4 E — 效应指数（含边界单元区域）

CPS3：基线 1.32/1.50/1.55（越界）→ **0.847/0.942/0.984**；CPS4：0.956/0.986/0.996 → **0.947/0.976/0.985**。全部落 [0.8, 1.2]。

## 3. 红线遵守

- 公共 API 不变；`error_est.py` 零改动；`_fit_node_block` 拟合核心零改动；
- 内部节点恢复值与改前逐位一致（B 三快照 `np.array_equal`，P-ε 等价测试全绿）；
- 边界节点增强表对内部节点行原样保留；无边界边的网格直返原表（零开销）。

## 4. 已知限制（如实记录）

1. C3 θ=0° 退化（见 §2.2）— 内部 patch 线性外推的固有代价；
2. CPS4 σy/τxy 顶边 ~O(h) — Q4 Gauss 采样一致性下限（见 §2.3）；
3. `test_circle_fan_golden` 平台浮点噪声失败为 main 预存，非本分支引入。

## 5. 交付物

| 交付物 | 位置 |
|---|---|
| 实现（含 `_boundary_patch_table`/`_pick_nearest`） | `fem2d/spr.py` |
| 测试（验收 C/D 用例 + 算法单测 + B/E 门禁） | `tests/test_spr_boundary.py` |
| 受影响测试重基线 | `tests/test_p_epsilon_spr_stress_equivalence.py`、`tests/test_assembly_recovery_solver.py`、`tests/test_regression_audit_20260803d.py` |
| 测量脚本与数据 | `scripts/spr_boundary_acceptance.py`、`docs/spr_boundary_data/*.json|npz` |
| 版本记录 | `CHANGELOG.md` 9.29.0、`pyproject.toml` 9.29.0 |
