"""P-η: error_est 统一 NaN/Inf 入口防护 + 批量归约/向量化重构等价性.

任务1 (外部审查 2026-08-06): _traction_jump_arrays 只查 complex 与形状,
不查有限性 — NaN 应力被 L366 np.where 静默归零成 jump_rel=0.0 (当作
"无跳跃"), 单单元空数据路径直接返回空列表吞掉非法数据. 本文件锁定
修复后行为: 任何 NaN/Inf 应力入口 → ValueError, 非法数据不得静默.

任务2/3: logaddexp 归约 (排序+reduceat) 与残差循环向量化重构 —
与基线逐单元/逐边实现逐位一致 (随机网格自对比).
"""
import math

import numpy as np
import pytest

from fem2d.element import evaluate_vector_field
from fem2d.error_est import (
    _body_force_residual_logs,
    _boundary_edge_residuals,
    _collect_loaded_edges,
    _element_sigma_tensors,
    _logaddexp_scatter,
    _neumann_edge_residuals,
    _traction_jump_arrays,
    compute_traction_jumps,
    element_refinement_indicator,
    estimate,
)
from fem2d.loads_core import LINE_GAUSS
from fem2d.mesh import Mesh


def _two_tri(body_force=None, surface_tractions=None):
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        elements=np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        elem_type="CPS3",
        body_force=body_force,
        surface_tractions=surface_tractions)


def _single_tri():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
        elements=np.array([[0, 1, 2]], dtype=int),
        elem_type="CPS3")


# ═══════════════════════════════════════════════════════════════
# 任务1: NaN/Inf 入口防护 (红侧: 修复前静默接受/静默忽略)
# ═══════════════════════════════════════════════════════════════

def test_traction_jumps_nan_stress_rejected():
    """单元应力含 NaN → ValueError (红侧: 返回 jump_abs=nan, jump_rel=0.0)."""
    mesh = _two_tri()
    stress = np.array([[1e6, 0.0, np.nan], [2e6, 0.0, 0.0]])
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_traction_jumps(mesh, stress)


def test_traction_jumps_inf_stress_rejected():
    """单元应力含 Inf → ValueError."""
    mesh = _two_tri()
    stress = np.array([[1e6, np.inf, 0.0], [2e6, 0.0, 0.0]])
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_traction_jumps(mesh, stress)


def test_traction_jumps_single_element_nan_rejected():
    """单单元网格 (无内部边, 空数据路径) + NaN → ValueError —
    有限性校验必须先于空数据提前返回 (红侧: 静默返回 [])."""
    mesh = _single_tri()
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_traction_jumps(mesh, np.array([[1e6, np.nan, 0.0]]))


def test_refinement_indicator_nan_stress_rejected():
    """两单元网格 + NaN 单元应力 → ValueError (红侧: 返回 [nan nan])."""
    mesh = _two_tri()
    result = {"stress": np.array([[1e6, np.nan, 0.0], [1e6, 0.0, 0.0]])}
    with pytest.raises(ValueError, match="NaN/Inf"):
        element_refinement_indicator(mesh, result)


def test_refinement_indicator_nan_stress_qp_rejected():
    """result['stress_qp'] 含 NaN → ValueError (红侧: 被静默忽略)."""
    mesh = _two_tri()
    result = {"stress": np.ones((2, 3)),
              "stress_qp": np.full((2, 2, 3), np.nan)}
    with pytest.raises(ValueError, match="stress_qp"):
        element_refinement_indicator(mesh, result)


def test_refinement_indicator_finite_stress_qp_ok():
    """有限 stress_qp 不受新入口防护影响 (solve() 正常结果不被误拒)."""
    mesh = _two_tri()
    result = {"stress": np.ones((2, 3)),
              "stress_qp": np.ones((2, 2, 3))}
    eta = element_refinement_indicator(mesh, result)
    assert eta.shape == (2,) and np.all(np.isfinite(eta))


def test_estimate_nan_stress_qp_rejected():
    """estimate 的 stress_qp 入口含 NaN → ValueError (统一防护覆盖)."""
    mesh = _two_tri()
    result = {"stress": np.ones((2, 3)),
              "stress_qp": np.full((2, 2, 3), np.nan)}
    with pytest.raises(ValueError, match="NaN/Inf"):
        estimate(mesh, result, verbose=False)


# ═══════════════════════════════════════════════════════════════
# 任务2: logaddexp 归约等价性 (排序+reduceat vs ufunc.at)
# ═══════════════════════════════════════════════════════════════

def test_logaddexp_scatter_matches_ufunc_at():
    """排序+reduceat 归约与 np.logaddexp.at 逐位一致 (随机含重复 eid/-inf).

    内部调用场景: eta_log 基恒为 -inf (element_refinement_indicator
    用 np.full(n_elem, -inf) 初始化), logaddexp(-inf, x) ≡ x, 归约
    结合序重排不引入任何 ulp 差异.
    """
    rng = np.random.default_rng(20260806)
    for _ in range(50):
        n = 200
        eids = rng.integers(0, 40, size=rng.integers(1, 600))
        terms = rng.normal(0.0, 10.0, size=len(eids))
        terms[rng.random(len(eids)) < 0.3] = -np.inf   # 零跳跃边
        got = np.full(n, -np.inf)
        _logaddexp_scatter(got, eids, terms)
        ref = np.full(n, -np.inf)
        np.logaddexp.at(ref, eids, terms)
        assert np.array_equal(got, ref)


# ═══════════════════════════════════════════════════════════════
# 任务3: 残差循环向量化等价性 (基线逐单元/逐边循环 vs 批量实现)
# 判定: 逐位一致 或 相对差 ≤ 1e-12 (任务书验收标准)
# ═══════════════════════════════════════════════════════════════

def _random_cps3_mesh(rng, trial, with_loads=True):
    """4×4 三角网格 (随机节点坐标), 与自对比脚本同构."""
    nx = ny = 4
    pts = rng.uniform(-2.0, 2.0, size=(ny + 1, nx + 1, 2))
    nodes = pts.reshape(-1, 2)
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = a + nx + 1
            d = c + 1
            elems.extend([[a, b, c], [b, d, c]])
    surface_tractions = []
    if with_loads:
        for i in range(nx):
            a = i
            b = i + 1
            if i % 2 == 0:
                surface_tractions.append(
                    {"nodes": (a, b), "traction": (1e6, -2e6)})
            else:
                surface_tractions.append(
                    {"nodes": (a, b), "traction": (5e5,), "is_pressure": True})
    body_force = ((trial + 1) * 1e3, -(trial + 1) * 2e3)
    return Mesh(nodes=nodes, elements=np.array(elems, dtype=int),
                elem_type="CPS3", body_force=body_force,
                surface_tractions=surface_tractions)


def _assert_close_enough(got, ref, what):
    """逐位一致 或 相对差 ≤ 1e-12 — 任务书验收标准.

    无贡献边在两边同为 -inf (跳过), 不参与相对差计算.
    """
    if np.array_equal(got, ref):
        return
    fin = np.isfinite(got) & np.isfinite(ref)
    if not np.any(fin):
        return   # 全部无贡献且逐位相等 — 无值可比
    rel = np.abs(got[fin] - ref[fin]) / np.maximum(np.abs(ref[fin]), 1e-300)
    max_rel = float(np.max(rel))
    assert max_rel <= 1e-12, (
        f"{what}: 相对差 {max_rel:.3e} 超出 1e-12 判定线 "
        f"(逐位不一致, 见 vectorized vs 基线)")


def _body_force_residuals_ref(mesh, nodes, n_elem, eta_log):
    """基线: 逐单元循环 (修复前实现)."""
    if mesh.body_force is None:
        return
    for eid in range(n_elem):
        xc, yc = mesh.centroids[eid]
        bx, by = evaluate_vector_field(mesh.body_force, xc, yc)
        f_norm2 = bx**2 + by**2
        if f_norm2 == 0.0:
            continue
        conn = mesh.elements[eid]
        h_K = max(np.linalg.norm(nodes[conn[ib]] - nodes[conn[ia]])
                  for ia, ib in mesh.element_kernel.local_edges)
        A_K = abs(mesh.areas[eid])
        if h_K == 0.0 or A_K == 0.0:
            continue
        eta_log[eid] = np.logaddexp(
            eta_log[eid],
            2.0 * math.log(h_K) + math.log(A_K) + math.log(f_norm2))


def _neumann_residuals_ref(mesh, nodes, sigma_e, loaded_by_edge, eta_log):
    """基线: 逐加载边循环 (3 点 Gauss, 修复前实现)."""
    loaded_edges = set()
    for key, st_list in loaded_by_edge.items():
        eid = mesh.edge_to_elems[key][0]
        loaded_edges.add(key)
        ni, nj = key
        n = np.array(mesh.boundary_outward_normal(ni, nj), dtype=float)
        t_fe = sigma_e[eid] @ n
        xa, ya = nodes[ni]
        xb, yb = nodes[nj]
        edge_vec = nodes[nj] - nodes[ni]
        h_e = float(np.linalg.norm(edge_vec))
        if h_e <= 64.0 * np.finfo(float).eps * max(
                float(np.max(np.abs(nodes))), np.finfo(float).tiny):
            continue
        integral = 0.0
        for w, xi_g in LINE_GAUSS:
            Ni = 0.5 * (1.0 - xi_g)
            Nj = 0.5 * (1.0 + xi_g)
            xg = Ni * xa + Nj * xb
            yg = Ni * ya + Nj * yb
            t_exact = np.zeros(2)
            for st in st_list:
                if st.get("is_pressure"):
                    p_val = st["traction"][0]
                    p = p_val(xg, yg) if callable(p_val) else p_val
                    t_exact += np.array([-p * n[0], -p * n[1]])
                else:
                    tx, ty = evaluate_vector_field(st["traction"], xg, yg)
                    t_exact += np.array([tx, ty])
            residual = t_fe - t_exact
            integral += 0.5 * w * np.dot(residual, residual)
        if integral > 0.0:
            eta_log[eid] = np.logaddexp(
                eta_log[eid], 2.0 * math.log(h_e) + math.log(integral))
    return loaded_edges


def _boundary_residuals_ref(mesh, nodes, sigma_e, loaded_edges, eta_log):
    """基线: 逐边界边循环 (修复前实现)."""
    fixed_dofs_set = set(mesh.fixed_dofs.tolist())
    all_boundary_edges = set(mesh.boundary_edges)
    for (a, b) in all_boundary_edges:
        key = (min(a, b), max(a, b))
        if key in loaded_edges:
            continue
        if key not in mesh.edge_to_elems:
            continue
        eids = mesh.edge_to_elems[key]
        if len(eids) == 0:
            continue
        eid = eids[0]
        dof_a = (2 * a in fixed_dofs_set, 2 * a + 1 in fixed_dofs_set)
        dof_b = (2 * b in fixed_dofs_set, 2 * b + 1 in fixed_dofs_set)
        if dof_a[0] and dof_a[1] and dof_b[0] and dof_b[1]:
            continue
        try:
            n = np.asarray(mesh.boundary_outward_normal(a, b), dtype=float)
        except (ValueError, RuntimeError):
            continue
        t_fe = sigma_e[eid] @ n
        edge_vec = nodes[b] - nodes[a]
        h_e = float(np.linalg.norm(edge_vec))
        if h_e <= 64.0 * np.finfo(float).eps * max(
                float(np.max(np.abs(nodes))), np.finfo(float).tiny):
            continue
        skip_x = dof_a[0] and dof_b[0]
        skip_y = dof_a[1] and dof_b[1]
        res_x = 0.0 if skip_x else t_fe[0]
        res_y = 0.0 if skip_y else t_fe[1]
        res2 = res_x**2 + res_y**2
        if res2 > 0.0:
            eta_log[eid] = np.logaddexp(
                eta_log[eid], 2.0 * math.log(h_e) + math.log(res2))


def _indicator_ref(mesh, result):
    """基线整链: 4 个被重构函数替换为基线循环, 其余与现实现共用."""
    mesh.build_connectivity()
    stress = result["stress"]
    n_elem = mesh.n_elements
    nodes = mesh.nodes
    eta_log = np.full(n_elem, -np.inf)
    edge_data, edge_lengths, jump_abs, _ = _traction_jump_arrays(
        mesh, stress)
    if len(edge_data):
        with np.errstate(divide="ignore"):
            log_term = (2.0 * np.log(edge_lengths)
                        + 2.0 * np.log(jump_abs))
        np.logaddexp.at(eta_log, edge_data[:, 2], log_term)
        np.logaddexp.at(eta_log, edge_data[:, 3], log_term)
    eta_log += math.log(0.5)
    _body_force_residuals_ref(mesh, nodes, n_elem, eta_log)
    sigma_e = _element_sigma_tensors(stress, n_elem)
    loaded_by_edge = _collect_loaded_edges(mesh)
    loaded_edges = _neumann_residuals_ref(
        mesh, nodes, sigma_e, loaded_by_edge, eta_log)
    _boundary_residuals_ref(mesh, nodes, sigma_e, loaded_edges, eta_log)
    return np.exp(0.5 * eta_log)


def test_body_force_vectorized_matches_baseline():
    """常数体力批量路径与逐单元基线等价 (逐位或 ≤1e-12).

    批量 h_K/A_K/单点求值 vs 逐单元循环 — 审查报告可优化点 8.
    """
    rng = np.random.default_rng(11)
    for trial in range(3):
        mesh = _random_cps3_mesh(rng, trial, with_loads=False)
        mesh.build_connectivity()
        n_elem = mesh.n_elements
        got = np.full(n_elem, -np.inf)
        _body_force_residual_logs(mesh, mesh.nodes, n_elem, got)
        ref = np.full(n_elem, -np.inf)
        _body_force_residuals_ref(mesh, mesh.nodes, n_elem, ref)
        _assert_close_enough(got, ref, f"体力残差 trial {trial}")


def test_neumann_vectorized_matches_baseline():
    """常数面力/压力边批量 3 点 Gauss 与逐边基线等价.

    含 callable 压力边 → 保持逐边求值路径 (值随坐标变化).
    """
    rng = np.random.default_rng(11)
    for trial in range(3):
        mesh = _random_cps3_mesh(rng, trial, with_loads=True)
        mesh.build_connectivity()
        n_elem = mesh.n_elements
        stress = rng.normal(0.0, 2e6, size=(n_elem, 3))
        sigma_e = _element_sigma_tensors(stress, n_elem)
        loaded_by_edge = _collect_loaded_edges(mesh)
        got = np.full(n_elem, -np.inf)
        _neumann_edge_residuals(
            mesh, mesh.nodes, sigma_e, loaded_by_edge, got)
        ref = np.full(n_elem, -np.inf)
        _neumann_residuals_ref(
            mesh, mesh.nodes, sigma_e, loaded_by_edge, ref)
        _assert_close_enough(got, ref, f"加载边残差 trial {trial}")


def test_boundary_vectorized_matches_baseline():
    """自由边/部分约束边批量路径与逐边基线等价.

    固支边跳过、仅约束单方向时只保留未约束方向 — 分类必须一致.
    """
    rng = np.random.default_rng(11)
    for trial in range(3):
        mesh = _random_cps3_mesh(rng, trial, with_loads=True)
        mesh.fix_node(0, "x", 0.0)
        mesh.fix_node(4, "both", 0.0)
        mesh.build_connectivity()
        n_elem = mesh.n_elements
        stress = rng.normal(0.0, 2e6, size=(n_elem, 3))
        sigma_e = _element_sigma_tensors(stress, n_elem)
        loaded_by_edge = _collect_loaded_edges(mesh)
        loaded_edges = set(loaded_by_edge)
        got = np.full(n_elem, -np.inf)
        _boundary_edge_residuals(
            mesh, mesh.nodes, sigma_e, loaded_edges, got)
        ref = np.full(n_elem, -np.inf)
        _boundary_residuals_ref(
            mesh, mesh.nodes, sigma_e, loaded_edges, ref)
        _assert_close_enough(got, ref, f"自由边残差 trial {trial}")


def test_indicator_vectorized_full_chain_matches_baseline():
    """element_refinement_indicator 整链与基线整链等价 (随机网格).

    覆盖: 无载荷 / 常数体力+面力+压力+部分约束 / callable 载荷 /
    Q4 — 与开发期自对比同构, 固化为回归.
    """
    rng = np.random.default_rng(20260806)
    # CPS3 无载荷 + 载荷约束
    for trial in range(3):
        mesh = _random_cps3_mesh(rng, trial, with_loads=(trial % 2 == 1))
        if trial % 2 == 1:
            mesh.fix_node(0, "x", 0.0)
            mesh.fix_node(4, "both", 0.0)
        stress = rng.normal(0.0, 2e6, size=(mesh.n_elements, 3))
        got = element_refinement_indicator(mesh, {"stress": stress})
        ref = _indicator_ref(mesh, {"stress": stress})
        _assert_close_enough(got, ref, f"CPS3 整链 trial {trial}")
    # callable 压力 + callable 体力 (逐边求值路径)
    nx = ny = 4
    pts = rng.uniform(-2.0, 2.0, size=(ny + 1, nx + 1, 2))
    nodes = pts.reshape(-1, 2)
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = a + nx + 1
            d = c + 1
            elems.extend([[a, b, c], [b, d, c]])

    def _p(x, y):
        return 1e6 * (1.0 - x)

    def _bf(x, y):
        return (1e3 * x, -2e3 * y)

    mesh = Mesh(
        nodes=nodes, elements=np.array(elems, dtype=int),
        elem_type="CPS3", body_force=_bf,
        surface_tractions=[{"nodes": (0, 1), "traction": (_p,),
                            "is_pressure": True},
                           {"nodes": (20, 21), "traction": (1e6, 0.0)}])
    stress = rng.normal(0.0, 2e6, size=(mesh.n_elements, 3))
    got = element_refinement_indicator(mesh, {"stress": stress})
    ref = _indicator_ref(mesh, {"stress": stress})
    _assert_close_enough(got, ref, "callable 载荷整链")
    # Q4
    q4_nodes = np.zeros((5, 5, 2))
    for j in range(5):
        for i in range(5):
            q4_nodes[j, i] = [i / 4, j / 4]
    q4_nodes = (q4_nodes + rng.uniform(-0.02, 0.02, size=(5, 5, 2))
                ).reshape(-1, 2)
    q4_elems = []
    for j in range(4):
        for i in range(4):
            a = j * 5 + i
            b = a + 1
            c = a + 5
            d = c + 1
            q4_elems.extend([[a, b, d, c]])
    mesh = Mesh(nodes=q4_nodes, elements=np.array(q4_elems, dtype=int),
                elem_type="Q4", body_force=(2e2, 0.0),
                surface_tractions=[{"nodes": (0, 1), "traction": (2e6, 1e6)}])
    mesh.fix_node(0, "both", 0.0)
    stress = rng.normal(0.0, 2e6, size=(mesh.n_elements, 3))
    got = element_refinement_indicator(mesh, {"stress": stress})
    ref = _indicator_ref(mesh, {"stress": stress})
    _assert_close_enough(got, ref, "Q4 整链")
