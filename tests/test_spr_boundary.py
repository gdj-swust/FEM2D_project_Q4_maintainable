"""SPR-BC-2026-001 边界节点恢复 — 确定性算法单测 + 验收 B/C/D/E 门禁.

算法 (Zienkiewicz-Zhu 1992, Int. J. Numer. Meth. Engng, 33, 1331–1364,
§2.3): 边界节点 b 的恢复值由最近内部候选节点 i 的 patch 拟合、在 b 坐标
处求值 — 候选 = b 邻接单元顶点 − 边界集, 为空时扩 ring-1 (CST 角点
情形); 并列取最小节点号; 候选仍空 (全边界退化网格) 退回自身 patch。
内部节点路径逐位不变 (红线: _fit_node_block 拟合核心零改动)。

判别性 (回滚必红):
* 算法单测 — 增强表行替换 / ring-1 并列裁决 / 兜底 / 内部行不变
* B  内部不变性 — 旧管线 (原始表) 与新产品在内部节点逐位一致
* C1/C2/C3 阈值 — 边界点恢复误差相对基线明显下降 (基线数值见注释)
* D  收敛阶 — CPS3 边界采样点斜率 ~0.2-1.6 (基线) → 1.7-2.1 (改后);
    CPS4 σy/τxy 顶边点受 Q4 Gauss 采样一致性下限限制 (~O(h)), 只对
    σx/τxy 可达 O(h²) 的点断言 (机理见验收报告)
* E  效应指数 — 含边界单元区域 θ ∈ [0.8, 1.2] (基线 CPS3 ≈ 1.55 越界)

阈值纪律: 全部来自 scripts/spr_boundary_acceptance.py 实测
(docs/spr_boundary_data/*.json), 留 ≥15% 裕度; 若实测漂移如实报告,
禁止调阈值凑区间。
"""
import numpy as np
import pytest

from fem2d import Mesh, solve
from fem2d.convergence import _gen_cantilever_mesh, _parabolic_shear_traction
from fem2d.material import D_matrix
from fem2d.spr import (spr_recovery, _boundary_patch_table, _node_patch_csr,
                       _pick_nearest)

# ── TG 悬臂梁模型 (与 tests/test_analytic_e3_z2_rate.py 同款) ──
L, H, T = 5.0, 1.0, 0.1
E_MOD, NU = 210e9, 0.3
P_SHEAR = 10000.0
I_BEAM = T * H**3 / 12.0
SXX_MAX = P_SHEAR * L * (H / 2.0) / I_BEAM
TAU_MAX = P_SHEAR / (2.0 * I_BEAM) * (H**2 / 4.0)

# ── Kirsch 模型 (与 tests/test_analytic_e2_kirsch.py 同款) ──
A_HOLE, W_PLATE = 1.0, 10.0
SIGMA_INF = 1e6


def tg_exact_stress(x, y):
    """TG 闭式解 (P>0 向下, 平面应力, 网格 y ∈ [-H/2, H/2]).

    符号以 FE 实测为准: 下弯梁顶边受拉 σx = +P(L-x)y/I > 0 (y>0);
    τxy = -P/(2I)(H²/4-y²) 与 _parabolic_shear_traction 同约定。
    """
    return np.array([
        P_SHEAR * (L - x) * y / I_BEAM,
        0.0,
        -P_SHEAR / (2.0 * I_BEAM) * (H**2 / 4.0 - y**2),
    ])


def _tg_solve(elem_type, nx):
    """TG 抛物线剪流悬臂梁: 构建/求解/恢复 (estimate 同源输入)."""
    ny = nx // 2
    nodes, elements = _gen_cantilever_mesh(L, H, nx, ny,
                                           elem_type=elem_type)
    m = Mesh(nodes=nodes, elements=elements, E=E_MOD, nu=NU, thickness=T,
             plane_type="stress", elem_type=elem_type)
    for n in m.nodes_on_edge("x", "min", tol=1e-6):
        m.fix_node(int(n), "both", 0.0)
    right = sorted(m.nodes_on_edge("x", "max", tol=1e-6),
                   key=lambda n: m.nodes[int(n), 1])
    for a_, b_ in zip(right, right[1:]):
        m.add_traction(int(a_), int(b_), 0.0,
                       lambda x, y: _parabolic_shear_traction(y, H, T,
                                                              P_SHEAR))
    result = solve(m, method="elimination", verbose=False)
    qp = result["stress_qp"]
    recovered = spr_recovery(m, qp if qp is not None else result["stress"])
    return m, result, recovered


def _old_pipeline_recovery(mesh, elem_stress):
    """旧管线: 原始 node_to_elems 表 (无边界替换) + 批量/精确路径.

    用生产内部函数以原始表重放改前管线 — 内部节点结果与改前逐位一致,
    用于 B 不变性与 D/E 基线判别。
    """
    from fem2d.spr import (_fit_node_block, _fit_nodes_exact, _prepare_samples,
                           _SAMPLE_BLOCK)
    sample_xy, sample_values = _prepare_samples(mesh, elem_stress)
    n_comp = sample_values.shape[-1]
    recovered = np.zeros((mesh.n_nodes, n_comp))
    ptr, flat = _node_patch_csr(mesh)
    per_node = np.diff(ptr) * sample_xy.shape[1]
    n_nodes = mesh.n_nodes
    unresolved = []
    node_lo = 0
    while node_lo < n_nodes:
        node_hi = node_lo + 1
        budget = int(per_node[node_lo])
        while (node_hi < n_nodes
               and budget + per_node[node_hi] <= _SAMPLE_BLOCK):
            budget += int(per_node[node_hi])
            node_hi += 1
        unresolved.append(_fit_node_block(
            mesh, sample_xy, sample_values, ptr, flat,
            node_lo, node_hi, recovered))
        node_lo = node_hi
    pending = (np.unique(np.concatenate(unresolved))
               if unresolved else np.empty(0, dtype=np.int64))
    if pending.size:
        _fit_nodes_exact(mesh, sample_xy, sample_values, pending, recovered)
    return recovered


def _node_at(mesh, x, y, tol=1e-9):
    d = np.linalg.norm(mesh.nodes - np.array([x, y]), axis=1)
    nid = int(np.argmin(d))
    assert d[nid] <= tol, f"节点 ({x},{y}) 不存在"
    return nid


def _boundary_mask(mesh):
    mask = np.zeros(mesh.n_nodes, dtype=bool)
    for lo, hi in mesh.boundary_edges:
        mask[int(lo)] = mask[int(hi)] = True
    return mask


def _table_rows(table):
    return [list(table.ids(n)) for n in range(len(table))]


def _kirsch_mesh(n_ang, n_rad):
    """1/4 板径向网格 (复制自 tests/test_analytic_e2_kirsch.py)."""
    nodes = []
    for i in range(n_ang + 1):
        th = 0.5 * np.pi * i / n_ang
        c, s = np.cos(th), np.sin(th)
        rmax = W_PLATE / max(abs(c), abs(s))
        for j in range(n_rad + 1):
            r = A_HOLE + (rmax - A_HOLE) * j / n_rad
            nodes.append([r * c, r * s])
    nodes = np.array(nodes)
    elems = []
    for i in range(n_ang):
        for j in range(n_rad):
            n0 = i * (n_rad + 1) + j
            n1 = n0 + 1
            n2 = (i + 1) * (n_rad + 1) + j + 1
            n3 = n2 - 1
            elems.append([n0, n1, n2])
            elems.append([n0, n2, n3])
    elems = np.array(elems, dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=1.0,
             plane_type="stress", elem_type="CST")
    for j in range(n_rad + 1):
        m.fix_node(j, "y")                        # θ=0 轴: uy=0
        m.fix_node(n_ang * (n_rad + 1) + j, "x")  # θ=π/2 轴: ux=0
    return m, nodes


# ────────────────────────────────────────────────────────────────────────
# 确定性算法单测
# ────────────────────────────────────────────────────────────────────────

def test_recovery_is_deterministic():
    """同一网格两次恢复逐位一致 (确定性算法, 无随机)."""
    mesh, result, _rec = _tg_solve("CPS4", 8)
    qp = result["stress_qp"]
    a = spr_recovery(mesh, qp)
    b = spr_recovery(mesh, qp)
    assert np.array_equal(a, b)


def test_boundary_row_replaced_by_nearest_interior_patch():
    """顶边中点行替换为直接下方内部节点 (最近候选) 的 patch 行."""
    nodes, elements = _gen_cantilever_mesh(2.0, 1.0, 4, 2, elem_type="CPS4")
    m = Mesh(nodes=nodes, elements=elements, E=1e9, nu=0.3, thickness=0.1,
             plane_type="stress", elem_type="CPS4")
    table = _boundary_patch_table(m)
    rows = _table_rows(table)
    orig = m.node_to_elems
    top_mid = _node_at(m, 1.0, 0.5)
    below = _node_at(m, 1.0, 0.0)
    assert top_mid in {int(lo) for lo, _ in m.boundary_edges} | {
        int(hi) for _, hi in m.boundary_edges}
    assert rows[top_mid] != list(orig.ids(top_mid))   # 行确实被替换
    assert rows[top_mid] == list(orig.ids(below))     # 最近内部候选 patch


def test_pick_nearest_prefers_nearest_then_smallest_id():
    """_pick_nearest 裁决规则: 最近者胜; 等距并列取最小节点号.

    结构化网格上直接下方节点恒存在、等距并列几乎无法用合法网格构造
    (候选须为内部节点, 即完全被单元环绕), 因此并列分支直接对生产裁决
    函数单测 — 该函数即 _boundary_patch_table 的选择逻辑。
    """
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0], [0.0, 1.0]])
    b = 0
    # 非并列: 最近者胜, 与节点号无关 (1 与 2 距 b 0.5/1.0)
    assert _pick_nearest({1, 2}, nodes, b) == 2
    # 并列: 1 与 3 距 b 均为 1.0 → 取最小节点号 1
    assert _pick_nearest({1, 3}, nodes, b) == 1
    # 三候选并列 (2 与 3 距 b 均为 1.0, 1 更近) → 仍取最近者
    assert _pick_nearest({1, 2, 3}, nodes, b) == 2


def test_cst_corner_ring1_expansion_falls_back_on_all_boundary():
    """CST 角点 ring-0 候选全为边界 → 扩 ring-1 仍全边界 → 兜底保留原行.

    2×1 高宽单元网格 (nx=2, ny=1): 左下角 (0,-1) 邻接三角形顶点
    (1,-1)/(1,1)/(0,1) 全为边界节点 → ring-0 空; ring-1 邻居单元顶点
    (2,-1)/(2,1) 亦全为边界 → 候选仍空 → 行原样保留 (兜底, 与退化
    全边界网格同路径)。该构型验证 ring-1 扩展开销与空结果处理。
    """
    nodes, elements = _gen_cantilever_mesh(2.0, 1.0, 2, 1, elem_type="CPS3")
    m = Mesh(nodes=nodes, elements=elements, E=1e9, nu=0.3, thickness=0.1,
             plane_type="stress", elem_type="CST")
    table = _boundary_patch_table(m)
    rows = _table_rows(table)
    orig = m.node_to_elems
    corner = _node_at(m, 0.0, -0.5)
    assert corner in {int(lo) for lo, _ in m.boundary_edges} | {
        int(hi) for _, hi in m.boundary_edges}
    assert rows[corner] == list(orig.ids(corner))  # 兜底: 原行保留


def test_degenerate_all_boundary_mesh_falls_back_to_own_patch():
    """单三角形网格全边界节点 → 候选恒空 → 行原样保留 (兜底)."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=1e9, nu=0.3, thickness=1.0,
             plane_type="stress", elem_type="CST")
    table = _boundary_patch_table(m)
    assert table is m.node_to_elems  # 无替换 → 原表零开销直返


def test_interior_rows_unchanged():
    """内部节点行逐位不变 (红线: 内部节点恢复结果不受影响)."""
    nodes, elements = _gen_cantilever_mesh(2.0, 1.0, 4, 2, elem_type="CPS4")
    m = Mesh(nodes=nodes, elements=elements, E=1e9, nu=0.3, thickness=0.1,
             plane_type="stress", elem_type="CPS4")
    table = _boundary_patch_table(m)
    rows = _table_rows(table)
    orig = m.node_to_elems
    bmask = _boundary_mask(m)
    for n in range(m.n_nodes):
        if not bmask[n]:
            assert rows[n] == list(orig.ids(n))


def test_all_boundary_nodes_replaced_on_structured_q4():
    """Q4 结构化网格上每个边界节点都有内部候选 → 行全部被替换."""
    nodes, elements = _gen_cantilever_mesh(2.0, 1.0, 4, 2, elem_type="CPS4")
    m = Mesh(nodes=nodes, elements=elements, E=1e9, nu=0.3, thickness=0.1,
             plane_type="stress", elem_type="CPS4")
    table = _boundary_patch_table(m)
    rows = _table_rows(table)
    orig = m.node_to_elems
    bmask = _boundary_mask(m)
    assert bmask.sum() > 0
    for n in range(m.n_nodes):
        if bmask[n]:
            assert rows[n] != list(orig.ids(n))


# ────────────────────────────────────────────────────────────────────────
# B 内部不变性 (验收 B: 非边界节点恢复值改前改后逐位一致)
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("elem_type", ["CPS3", "CPS4"])
def test_b_interior_nodes_bitwise_invariant(elem_type):
    mesh, result, rec_new = _tg_solve(elem_type, 16)
    qp = result["stress_qp"]
    rec_old = _old_pipeline_recovery(mesh, qp)
    interior = ~_boundary_mask(mesh)
    assert interior.sum() > 0
    assert np.array_equal(rec_new[interior], rec_old[interior]), elem_type


# ────────────────────────────────────────────────────────────────────────
# C 边界点精度 (验收 C1/C2/C3; 阈值 = 基线实测 × 裕度)
# ────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("elem_type,threshold,base", [
    # 基线 (nx=64): CPS3 0.00815 / CPS4 0.00514; 改后实测 0.000173 / 0.00257
    ("CPS3", 0.001, 0.00815),
    ("CPS4", 0.003, 0.00514),
])
def test_c1_top_edge_syy_drops_below_threshold(elem_type, threshold, base):
    """C1: 顶边中点 σy=0 → |σy|/σx_max 较基线明显下降."""
    mesh, _result, rec = _tg_solve(elem_type, 64)
    nid = _node_at(mesh, L / 2.0, H / 2.0)
    ratio = float(abs(rec[nid, 1])) / SXX_MAX
    assert ratio < threshold, f"{elem_type} C1 ratio={ratio:.5f}"
    assert ratio < base / 1.7, f"{elem_type} C1 较基线降幅不足: {ratio:.5f}"


@pytest.mark.parametrize("elem_type,threshold,base", [
    # 基线 (nx=64): CPS3 max 0.062 / CPS4 max 0.052; 改后 0.0131 / 0.0263
    ("CPS3", 0.016, 0.062),
    ("CPS4", 0.031, 0.052),
])
def test_c2_right_face_txy_max_error_drops(elem_type, threshold, base):
    """C2: 右端面节点恢复 τxy 对抛物线剪流最大误差明显下降."""
    mesh, _result, rec = _tg_solve(elem_type, 64)
    face = [int(n) for n in mesh.nodes_on_edge("x", "max", tol=1e-9)]
    errs = [abs(rec[n, 2] - tg_exact_stress(*mesh.nodes[n])[2]) / TAU_MAX
            for n in face]
    err_max = float(np.max(errs))
    assert err_max < threshold, f"{elem_type} C2 max={err_max:.5f}"
    assert err_max < base / 1.7, f"{elem_type} C2 较基线降幅不足: {err_max:.5f}"


def test_c3_kirsch_hole_edge_improves_on_two_of_four_points():
    """C3: Kirsch 孔边 σθθ (θ=0/30/60/90°) 至少两点明显改善.

    基线 (96,48): abs_err = [0.0159, 0.0747, 0.588, 0.266];
    改后实测 = [0.123, 0.0545, 0.0796, 0.120]。θ=0 (场曲率最大的孔边
    点) 退化 — 内部 patch 外推平滑了 1/r² 曲率, ZZ92 §2.3 对光滑场结论
    的已知边界效应, 如实记录不设阈; θ=60/90 大幅改善。
    """
    m, nodes = _kirsch_mesh(96, 48)
    right = sorted(np.flatnonzero(np.abs(nodes[:, 0] - W_PLATE) < 1e-9),
                   key=lambda n: nodes[n, 1])
    for a_, b_ in zip(right, right[1:]):
        m.add_traction(int(a_), int(b_), SIGMA_INF, 0.0)
    result = solve(m, method="elimination", verbose=False)
    qp = result["stress_qp"]
    rec = spr_recovery(m, qp if qp is not None else result["stress"])
    errs = {}
    for th_deg in (30.0, 60.0, 90.0):
        th = np.radians(th_deg)
        i = int(round(th / (0.5 * np.pi) * 96))
        nid = i * (48 + 1)
        sx, sy, txy = rec[nid]
        s_theta = (sx * np.sin(th)**2 + sy * np.cos(th)**2
                   - 2.0 * txy * np.sin(th) * np.cos(th))
        exact = SIGMA_INF * (1.0 - 2.0 * np.cos(2.0 * th))
        errs[th_deg] = float(abs(s_theta - exact)) / SIGMA_INF
    assert errs[60.0] < 0.15, f"θ=60° err={errs[60.0]:.4f} (基线 0.588)"
    assert errs[90.0] < 0.15, f"θ=90° err={errs[90.0]:.4f} (基线 0.266)"
    assert errs[30.0] < 0.075, f"θ=30° err={errs[30.0]:.4f} (基线 0.0747)"


# ────────────────────────────────────────────────────────────────────────
# D 收敛阶 (验收 D; CPS3 基线 ~O(h) → 改后 O(h²), CPS4 达限点断言)
# ────────────────────────────────────────────────────────────────────────

D_COMP = {"syy_top": 1, "txy_top": 2, "txy_right_q": 2, "sxx_top": 0}
D_NORM = {"syy_top": SXX_MAX, "txy_top": TAU_MAX,
          "txy_right_q": TAU_MAX, "sxx_top": SXX_MAX}


def _d_sequences(elem_type, levels, old=False):
    """各 D 采样点的 (h, err) 序列; old=True 时用改前管线."""
    seq = {k: [] for k in D_COMP}
    hs = {k: [] for k in D_COMP}
    for nx in levels:
        mesh, result, rec = _tg_solve(elem_type, nx)
        if old:
            qp = result["stress_qp"]
            rec = _old_pipeline_recovery(mesh, qp)
        for key in D_COMP:
            if key == "txy_right_q":
                if nx // 2 < 4:
                    continue  # ny<4 无 y=H/4 节点
                nid = _node_at(mesh, L, H / 4.0)
                exact = tg_exact_stress(L, H / 4.0)[2]
            else:
                nid = _node_at(mesh, L / 2.0, H / 2.0)
                if key == "sxx_top":
                    exact = tg_exact_stress(L / 2.0, H / 2.0)[0]
                else:
                    exact = 0.0
            err = abs(rec[nid, D_COMP[key]] - exact) / D_NORM[key]
            seq[key].append(err)
            hs[key].append(L / nx)
    return hs, seq


def _slope(h_arr, errs):
    """ln|e| ~ ln h 最小二乘斜率 (跳过最粗非渐近层)."""
    h = np.array(h_arr[1:])
    e = np.array(errs[1:])
    return float(np.polyfit(np.log(h),
                            np.log(np.maximum(e, np.finfo(float).tiny)),
                            1)[0])


@pytest.mark.parametrize("key", sorted(D_COMP))
def test_d_cps3_boundary_slopes_reach_oh2(key):
    """CPS3 边界采样点: 基线 ~O(h) → 改后 O(h²).

    实测 (levels 8-64): syy_top 0.957→1.793, txy_top 0.822→1.788,
    txy_right_q 1.550→1.806, sxx_top 0.189→1.787 — 全进入 [1.6, 2.2]。
    """
    levels = (8, 16, 32, 64)
    hs_old, seq_old = _d_sequences("CPS3", levels, old=True)
    hs_new, seq_new = _d_sequences("CPS3", levels)
    slope_old = _slope(hs_old[key], seq_old[key])
    slope_new = _slope(hs_new[key], seq_new[key])
    assert 1.6 <= slope_new <= 2.2, \
        f"CPS3 {key} 改后斜率 {slope_new:.3f} 不在 [1.6, 2.2]"
    assert slope_new - slope_old >= 0.2, \
        f"CPS3 {key} 斜率改善不足: {slope_old:.3f} → {slope_new:.3f}"


@pytest.mark.parametrize("key,band", [
    ("sxx_top", (1.8, 2.8)),       # 实测 2.36 (基线 2.53, 本已超收敛)
    ("txy_right_q", (1.7, 2.2)),   # 实测 1.94 (基线 1.95, 保持)
])
def test_d_cps4_attainable_points_stay_oh2(key, band):
    """CPS4: σx/τxy 可达 O(h²) 的点在改后保持 O(h²).

    CPS4 σy/τxy 顶边点受 Q4 Gauss 采样一致性下限限制 (~O(h), 机理:
    νP(L-x)y²/(2EI) 项的 y² 形函数插值残差在每行 Gauss 点形成
    ±0.289h·(L-x) 伪场, 线性恢复不可消除) — 不设阶断言, 见验收报告。
    """
    levels = (8, 16, 32, 64)
    hs_new, seq_new = _d_sequences("CPS4", levels)
    slope_new = _slope(hs_new[key], seq_new[key])
    assert band[0] <= slope_new <= band[1], \
        f"CPS4 {key} 斜率 {slope_new:.3f} 不在 {band}"


def test_d_cps4_syy_top_magnitude_halved():
    """CPS4 顶边 σy 阶不改善但幅值减半 (基线 0.00514 → 实测 0.00257)."""
    levels = (8, 16, 32, 64)
    hs_old, seq_old = _d_sequences("CPS4", levels, old=True)
    hs_new, seq_new = _d_sequences("CPS4", levels)
    assert seq_new["syy_top"][-1] < seq_old["syy_top"][-1] / 1.7
    assert len(hs_new["syy_top"]) == len(levels)


# ────────────────────────────────────────────────────────────────────────
# E 效应指数 (验收 E: 含边界单元区域 θ ∈ [0.8, 1.2])
# ────────────────────────────────────────────────────────────────────────

def _region_effectivity(mesh, result, recovered):
    """含边界单元区域 (顶边/右端面相邻且质心 x ≥ H) 的 θ=||e_est||/||e_exact||.

    与 scripts/spr_boundary_acceptance.py 同定义: 误差在积分点离散,
    e_est = σ*(q) − σh(q), e_exact = σ_exact(q) − σh(q), D⁻¹ 能量内积。
    """
    kernel = mesh.element_kernel
    shape = np.asarray(kernel.recovery_shape_matrix(mesh), dtype=float)
    weights = np.asarray(kernel.recovery_weights(mesh), dtype=float)
    stress_qp = result["stress_qp"]
    bmask = _boundary_mask(mesh)
    on_top = np.any(np.abs(mesh.nodes[mesh.elements, 1] - H / 2.0) < 1e-9,
                    axis=1)
    on_right = np.any(np.abs(mesh.nodes[mesh.elements, 0] - L) < 1e-9,
                      axis=1)
    region = ((on_top | on_right) & (mesh.centroids[:, 0] >= H)
              & np.any(~bmask[mesh.elements], axis=1)
              & np.any(bmask[mesh.elements], axis=1))
    assert np.any(region), "效应指数区域为空 — 检查网格/区域定义"
    d_inv = np.linalg.inv(D_matrix(mesh.E, mesh.nu, mesh.plane_type))
    e_est_sq = 0.0
    e_exact_sq = 0.0
    for eid in np.flatnonzero(region):
        conn = mesh.elements[eid]
        s_star = shape @ recovered[conn]
        s_h = stress_qp[eid]
        s_exact = np.array([tg_exact_stress(*p)
                            for p in shape @ mesh.nodes[conn]])
        d_est = s_star - s_h
        d_exact = s_exact - s_h
        w = weights[eid]
        e_est_sq += float(np.sum(
            w * np.einsum("qi,ij,qj->q", d_est, d_inv, d_est)))
        e_exact_sq += float(np.sum(
            w * np.einsum("qi,ij,qj->q", d_exact, d_inv, d_exact)))
    return float(np.sqrt(e_est_sq / e_exact_sq))


@pytest.mark.parametrize("elem_type", ["CPS3", "CPS4"])
def test_e_region_effectivity_in_range(elem_type):
    """含边界单元区域效应指数 θ ∈ [0.8, 1.2] (实测 CPS3 0.94 / CPS4 0.98)."""
    mesh, result, rec = _tg_solve(elem_type, 32)
    theta = _region_effectivity(mesh, result, rec)
    assert 0.8 <= theta <= 1.2, f"{elem_type} θ={theta:.3f} 越出 [0.8, 1.2]"


def test_e_cps3_baseline_was_out_of_range():
    """判别: 改前管线 CPS3 效应指数 ≈ 1.55 越界 → 改后拉回区间."""
    mesh, result, _rec = _tg_solve("CPS3", 64)
    qp = result["stress_qp"]
    rec_old = _old_pipeline_recovery(mesh, qp)
    theta_old = _region_effectivity(mesh, result, rec_old)
    assert theta_old > 1.2, f"基线不再越界 (θ={theta_old:.3f}) — 检查判据"
    rec_new = spr_recovery(mesh, qp)
    theta_new = _region_effectivity(mesh, result, rec_new)
    assert 0.8 <= theta_new <= 1.2
