"""Zienkiewicz-Zhu 误差估计 — Bathe §4.3.6 Eq 4.106-4.108

Z2 估计器核心思想:
  σ_h  — 原始 FEM 积分点应力
  σ*  — 改进/光滑应力 (L2 投影到 C0 连续空间)
  误差 e_σ = σ* - σ_h 用于估计网格充分性

Bathe §4.3.6 Eq (4.106)-(4.108):
  应力不平衡:  ∇·σ_h + f^B ≠ 0           (域内)
               σ_h·n - t ≠ 0             (边界)
  Z2 估计: Δτ = (τ*)improved - τ_h       (Eq 4.108)

改进 (相对原始实现):
  - 使用 L2 投影 (Bathe Example 4.27) 替代简单面积加权平均
  - 误差能量范数含柔度矩阵 D⁻¹ 加权:  ||e||² = ∫ e_σᵀ·D⁻¹·e_σ dΩ
  - 输出每个单元的误差贡献百分比 (指导 h-adaptivity)
"""
import numpy as np

from .element import evaluate_vector_field
from .loads_core import LINE_GAUSS
from .material import D_matrix
from .spr import spr_recovery
from .stress import nodal_L2_projection, nodal_weighted


def _energy_density(s_star_pt, s_h, D_inv, use_scale, s_scale):
    """归一化应力空间的能量密度 (误差 + 恢复) — 微尺度载荷下
    σ²/E ~1e-320 次正规下溢为 0, eta 静默塌缩; 归一化后 O(1)。
    einsum 椭圆下标统一批量 (ne,q,3) 与标量 (q,3) 两种形态。
    """
    if use_scale:
        s_n = s_star_pt / s_scale
        diff_n = (s_star_pt - s_h) / s_scale
    else:
        s_n = s_star_pt
        diff_n = s_star_pt - s_h
    error_density = np.einsum("...i,ij,...j->...", diff_n, D_inv, diff_n)
    recovered_density = np.einsum("...i,ij,...j->...", s_n, D_inv, s_n)
    return error_density, recovered_density


def _element_energy_errors(mesh, method, stress, stress_qp, s_star, D_inv):
    """逐单元能量误差 (elem_err_sq) 与恢复能量 (recovered_energy).

    批量路径 (shape/weights 缓存) + 兼容路径 (逐单元 recovery_quadrature);
    返回 (elem_err_sq, recovered_energy, s_scale, use_scale)。s_scale
    覆盖全部应力源 (单元代表应力可因积分点正负抵消而≈0, 弯曲型场
    归一化会爆炸)。
    """
    n_elem = mesh.n_elements
    elem_err_sq = np.zeros(n_elem)
    recovered_energy = np.zeros(n_elem)
    kernel = mesh.element_kernel
    s_scale = 0.0
    for _src in (stress, stress_qp, s_star):
        if _src is None:
            continue
        _arr = np.asarray(_src, dtype=float)
        if _arr.size:
            s_scale = max(s_scale, float(np.max(np.abs(_arr))))
    use_scale = s_scale > 0.0

    shape = kernel.recovery_shape_matrix(mesh)
    weights = kernel.recovery_weights(mesh)
    if shape is not None and weights is not None:
        shape = np.asarray(shape, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if shape.ndim != 2 or shape.shape[1] != mesh.elements.shape[1]:
            raise ValueError(
                f"{kernel.name} recovery shape matrix has invalid shape "
                f"{shape.shape}; expected (n_sample, "
                f"{mesh.elements.shape[1]}).")
        expected_weights = (n_elem, shape.shape[0])
        if weights.shape != expected_weights:
            raise ValueError(
                f"{kernel.name} recovery weights have shape {weights.shape}; "
                f"expected {expected_weights}.")

        for start in range(0, n_elem, 50000):
            stop = min(start + 50000, n_elem)
            conn = mesh.elements[start:stop]
            s_star_pt = np.einsum("qn,enc->eqc", shape, s_star[conn])
            if stress_qp is None or method == "weighted":
                # weighted 的 σ* 来自单元均值恢复 — 与积分点原始应力
                # 量纲语义不可比, 曾混用导致精确线性场报 ~14% 虚假误差
                s_h = stress[start:stop, None, :]
            else:
                s_h = stress_qp[start:stop]
                if s_h.shape != s_star_pt.shape:
                    raise ValueError(
                        "Recovered quadrature has shape "
                        f"{s_star_pt.shape}, but stress_qp has shape "
                        f"{s_h.shape}")
            dA = weights[start:stop]
            error_density, recovered_density = _energy_density(
                s_star_pt, s_h, D_inv, use_scale, s_scale)
            elem_err_sq[start:stop] = (
                mesh.thickness * np.sum(dA * error_density, axis=1))
            recovered_energy[start:stop] = (
                mesh.thickness * np.sum(dA * recovered_density, axis=1))
    else:
        # Compatibility path for external kernels with custom recovery rules.
        for eid, conn in enumerate(mesh.elements):
            shape, dA = kernel.recovery_quadrature(mesh, eid)
            s_star_pt = shape @ s_star[conn]
            if stress_qp is None or method == "weighted":
                s_h = np.broadcast_to(stress[eid], s_star_pt.shape)
            else:
                s_h = stress_qp[eid]
                if s_h.shape != s_star_pt.shape:
                    raise ValueError(
                        f"Element {eid}: recovered quadrature has shape "
                        f"{s_star_pt.shape}, but stress_qp has shape "
                        f"{s_h.shape}")
            error_density, recovered_density = _energy_density(
                s_star_pt, s_h, D_inv, use_scale, s_scale)
            elem_err_sq[eid] = (
                mesh.thickness * float(np.dot(dA, error_density)))
            recovered_energy[eid] = (
                mesh.thickness * float(np.dot(dA, recovered_density)))
    return elem_err_sq, recovered_energy, s_scale, use_scale


def estimate(mesh, result, method="SPR", verbose=True):
    """Z2 误差估计 — 基于应力恢复评估网格质量

    Bathe §4.3.6: 通过比较原始应力 σ_h 和改进应力 σ* 来估计误差。

    算法:
      1. 计算改进应力 σ* (L2 投影 或 面积加权平均)
      2. 对每个单元: ||e_e||²_energy = A_e · (σ* - σ_e)ᵀ·D⁻¹·(σ* - σ_e)
      3. ||e||² = Σ||e_e||²_energy
      4. ||U||² = Σ A_e · σ*_avgᵀ·D⁻¹·σ*_avg
      5. η = ||e|| / ||U|| × 100%

    参数
    ----
    mesh : Mesh — 网格对象
    result : dict — solver.solve() 的返回结果
    method : str — "SPR" (默认), "L2" (Bathe Ex 4.27), 或 "weighted" (传统)
    verbose : bool — 是否打印详细报告

    返回
    ----
    dict:
        elem_error  : (n_elem,) ndarray — 各单元的能量误差
        total_error : float — 总能量误差
        energy_norm : float — 应变能范数
        eta         : float — 相对误差百分比
        worst_elem  : int — 最差单元索引
        elem_contrib: (n_elem,) ndarray — 各单元误差贡献百分比 (%)
    """
    stress = np.asarray(result["stress"], dtype=float)
    stress_qp = result.get("stress_qp")
    if stress_qp is not None:
        stress_qp = np.asarray(stress_qp, dtype=float)

    # ── 1. 改进应力 σ* ──
    if method == "SPR":
        s_star = spr_recovery(
            mesh, stress_qp if stress_qp is not None else stress)
    elif method == "L2":
        s_star = nodal_L2_projection(
            mesh, stress_qp if stress_qp is not None else stress)
    elif method == "weighted":
        s_star = nodal_weighted(mesh, stress)
    else:
        raise ValueError(
            f"Unknown error estimation method '{method}'. "
            f"Expected: 'SPR', 'L2', or 'weighted'.")

    # ── 2. 柔度矩阵 D⁻¹ (用于能量内积) ──
    D = D_matrix(mesh.E, mesh.nu, mesh.plane_type)
    D_inv = np.linalg.inv(D)  # D 是对称正定的 3×3 矩阵

    elem_err_sq, recovered_energy, s_scale, use_scale = \
        _element_energy_errors(mesh, method, stress, stress_qp, s_star, D_inv)

    elem_err = np.sqrt(np.maximum(elem_err_sq, 0.0))
    if use_scale:
        elem_err = elem_err * s_scale
    U_norm2 = float(np.sum(np.maximum(recovered_energy, 0.0)))

    # ── 3. 全局量 ──
    total_err = np.sqrt(float(np.sum(np.maximum(elem_err_sq, 0.0))))
    if use_scale:
        total_err *= s_scale
    # eta 在归一化空间直接计算 (elem_err_sq/recovered_energy 未乘回
    # s_scale) — 曾对乘回后的绝对量求比, 载荷 1e-160 时 elem_err² 次正规
    # 下溢, eta 塌缩到 0、1e-310 时 NaN (第三轮外部审查复现)。归一化
    # 空间的比值完全尺度无关。绝对量 (total_err/U_norm) 乘回供报告。
    _eta_num = np.sqrt(max(float(np.sum(np.maximum(
        elem_err_sq, 0.0))), np.finfo(float).tiny))
    _eta_den = np.sqrt(max(U_norm2, np.finfo(float).tiny))
    if float(np.sum(np.maximum(elem_err_sq, 0.0))) == 0.0:
        # 真零误差场 (含次正规下溢到 0): eta=0, 曾 0/0=NaN
        eta = 0.0
    else:
        eta = _eta_num / _eta_den * 100.0
    # 下限仅保护零能量场 (eta=0 而非 0/0=nan); s_scale=0 时不乘回 —
    # 曾 U_norm=0 产生 0/0 NaN 警告
    U_norm = np.sqrt(max(U_norm2, np.finfo(float).tiny)
                     * (1.0 + 1e-12))
    if s_scale > 0.0:
        U_norm *= s_scale

    worst = int(np.argmax(elem_err))
    # 贡献百分比在归一化空间计算 (elem_err_sq 未乘 s_scale) — 曾用乘回
    # 后的 elem_err² (~1e-308 次正规), 微尺度载荷下求和失真, sum 只到
    # 0.46% (回归审计 2026-08-03 发现)。s_scale² 在比值中自动抵消。
    elem_contrib = (elem_err_sq
                    / max(float(np.sum(elem_err_sq)),
                          np.finfo(float).tiny)) * 100.0

    # ── 4. 应力跳跃度量 (Bathe §4.3.6 等应力带法) ──
    stress_jumps = _estimate_stress_jumps(mesh, result)

    # ── 5. 报告 ──
    if verbose:
        print(f"\n{'='*55}")
        print("  Z2 Error Estimate (Zienkiewicz-Zhu) — Bathe §4.3.6")
        print(f"  Method: {method}")
        print(f"{'='*55}")
        print(f"  Energy error:   {total_err:.3e}")
        print(f"  Energy norm:    {U_norm:.3e}")
        print(f"  Relative eta:   {eta:.2f}%")
        print(f"  Stress jumps:   avg={stress_jumps['avg_jump']:.3e}, "
              f"rms={stress_jumps['rms_jump']:.3e}, "
              f"max={stress_jumps['max_jump']:.3e}")
        print(f"  Worst elem #{worst}: e = {elem_err[worst]:.3e} "
              f"({elem_contrib[worst]:.1f}% of total)")
        if eta > 20:
            print("  [ADVICE] η > 20% → refine mesh globally")
        elif eta > 10:
            print("  [NOTE] η > 10% → consider local refinement")
        else:
            print("  [OK] η < 10% → recovered-error indicator is low; "
                  "still verify near singularities, concentrated loads, "
                  "restraint corners, and local stress peaks")
        print(f"{'='*55}")

    return {
        "elem_error": elem_err,
        "total_error": total_err,
        "energy_norm": U_norm,
        "eta": eta,
        "worst_elem": worst,
        "elem_contrib": elem_contrib,
        "stress_jumps": stress_jumps,
    }


def _traction_jump_arrays(mesh, elem_stress, sigma_ref=None):
    """Vectorized internal-edge traction jumps.

    Returns ``(edge_data, edge_lengths, jump_abs, jump_rel)`` where
    ``edge_data`` columns are ``node_a,node_b,eid1,eid2``.
    """
    mesh.build_connectivity()
    stress = np.asarray(elem_stress, dtype=float)
    edge_data = mesh.internal_edge_data
    if edge_data is None or len(edge_data) == 0:
        empty = np.empty(0, dtype=float)
        return np.empty((0, 4), dtype=np.int64), empty, empty, empty

    a, b, e1, e2 = edge_data.T
    edge_vec = mesh.nodes[b] - mesh.nodes[a]
    edge_lengths = np.linalg.norm(edge_vec, axis=1)
    valid = edge_lengths > np.finfo(float).tiny
    if not np.all(valid):
        edge_data = edge_data[valid]
        a, b, e1, e2 = edge_data.T
        edge_vec = edge_vec[valid]
        edge_lengths = edge_lengths[valid]

    normals = np.column_stack(
        (edge_vec[:, 1], -edge_vec[:, 0])) / edge_lengths[:, None]
    delta = stress[e1] - stress[e2]
    jump_x = delta[:, 0] * normals[:, 0] + delta[:, 2] * normals[:, 1]
    jump_y = delta[:, 2] * normals[:, 0] + delta[:, 1] * normals[:, 1]
    jump_abs = np.hypot(jump_x, jump_y)

    frob = np.sqrt(
        stress[:, 0]**2 + stress[:, 1]**2 + 2.0 * stress[:, 2]**2)
    if sigma_ref is not None:
        # 显式参考应力必须正且有限 — 曾 max(...,1e-30) 覆盖用户传入的
        # 微尺度参考值, jump_rel 随应力幅值静默衰减 (审计 2026-08-03)
        if not (np.isfinite(sigma_ref) and sigma_ref > 0.0):
            raise ValueError(
                f"sigma_ref must be finite and positive, got {sigma_ref!r}")
        denom = np.full(len(edge_data), float(sigma_ref))
    else:
        # 归一化纯相对: local/global 尺度本身随应力幅值缩放, 无绝对下限
        global_scale = float(np.percentile(frob, 95))
        local_scale = 0.5 * (frob[e1] + frob[e2])
        denom = np.maximum(local_scale, 0.05 * global_scale)
        with np.errstate(divide="ignore", invalid="ignore"):
            jump_rel = jump_abs / denom
        # 零应力场: denom=0 且 jump_abs=0 → 0/0; 相对跳跃无定义 → 0
        jump_rel = np.where(np.isfinite(jump_rel), jump_rel, 0.0)
        return edge_data, edge_lengths, jump_abs, jump_rel
    return edge_data, edge_lengths, jump_abs, jump_abs / denom


def compute_traction_jumps(mesh, elem_stress, sigma_ref=None):
    """计算每条内部边的牵引跳跃 — Bathe §4.3.6 Eq (4.107) 衍生

    归一化:
      sigma_ref=None → j_rel = j_abs / max((‖σ⁺‖_F+‖σ⁻‖_F)/2, 0.05·σ_95)  (局部+全局混合)
      sigma_ref=1e6  → j_rel = j_abs / 1e6  (固定名义应力, 跨网格可比)
      其中 σ_95 = 全局 Frobenius 范数 95% 分位数

    返回 list of dict:
      {node_a, node_b, eid1, eid2, edge_length, jump_abs, jump_rel}
    """
    edge_data, edge_lengths, jump_abs, jump_rel = _traction_jump_arrays(
        mesh, elem_stress, sigma_ref=sigma_ref)
    return [
        {
            'node_a': int(edge[0]), 'node_b': int(edge[1]),
            'eid1': int(edge[2]), 'eid2': int(edge[3]),
            'edge_length': float(length),
            'jump_abs': float(absolute),
            'jump_rel': float(relative),
        }
        for edge, length, absolute, relative in zip(
            edge_data, edge_lengths, jump_abs, jump_rel)
    ]


def _estimate_stress_jumps(mesh, result):
    """内部边牵引不连续度统计 — 基于 Bathe §4.3.6 Eq (4.107) 衍生

    Returns dict: {avg_jump, rms_jump, max_jump, n_jumps}
      rms_jump = sqrt( Σ h_e·j_rel² / Σ h_e ) (edge-length-weighted)
    """
    _, edge_lengths_arr, _, jump_vals = _traction_jump_arrays(
        mesh, result["stress"])
    if len(jump_vals) == 0:
        return {"avg_jump": 0.0, "rms_jump": 0.0, "max_jump": 0.0, "n_jumps": 0}

    # 边长加权 RMS — 网格加密后边长变化时比普通平均更公平
    # 分母 1e-30 绝对地板与 eta/contrib 同族 (审计 2026-08-03 已清除)
    # — 改为仅防零总长除零
    j_rms = np.sqrt(
        np.sum(edge_lengths_arr * jump_vals**2)
        / max(np.sum(edge_lengths_arr), np.finfo(float).tiny))
    return {
        "avg_jump": float(np.mean(jump_vals)),
        "rms_jump": float(j_rms),
        "max_jump": float(np.max(jump_vals)),
        "n_jumps": len(jump_vals),
    }


def element_refinement_indicator(mesh, result):
    """显式残差型后验误差估计器 (Verfürth 1996, §3.2).

    来源: 弱形式 a(e,v) = L(v) − a(u_h,v), 分部积分:
      Σ_K ∫_K (f + ∇·σ^h)·v dx + Σ_e ∫_e [[σ^h n]]·v ds + Σ_{e∈Γ_t} ∫_e (t̄ − σ^h n)·v ds
    取局部 bubble 函数为检验函数 v, 用 Cauchy-Schwarz + bubble 的 h-缩放得:
      η_K² = h_K²·∫_K|f+∇·σ^h|²dx + ½ Σ h_e·∫_e|[[σ^h n]]|²ds + Σ h_e·∫_e|t̄−σ^h n|²ds

    CST 中 σ^h 为常数，下面的边/域残差缩放是精确的。Q4 使用单元
    Gauss 平均应力形成稳健的排序型指标；Q4 的能量误差百分比应以
    :func:`estimate` 的积分点 Z2 结果为准。
      ① 内部: h_K² × A_K × |f|²     (A_K = 单元面积)
      ② 边跳跃: h_e² × |(σ⁺−σ⁻)n|²  (∫_e|J|²ds = h_e·|J|², 再×h_e = h_e²·|J|²)
      ③ 加载边: h_e² × |σ^h n − t̄|²
      ④ Dirichlet边: 跳过 (反力未知, 不能当自由边)

    返回 (n_elem,) ndarray — η_K [Pa·m], 值越高优先加密.
    用于单元排序和 Dörfler 标记, 本身不是误差百分比.
    """
    mesh.build_connectivity()
    stress = result["stress"]
    n_elem = mesh.n_elements
    nodes = mesh.nodes

    # ── 1. 内部边牵引跳跃 (数组路径 — 免去每条边一个 dict) ──
    edge_data, edge_lengths, jump_abs, _ = _traction_jump_arrays(
        mesh, stress)
    eta_sq = np.zeros(n_elem)
    if len(edge_data):
        # h_e × ‖J‖²_{L²(e)} = h_e × (h_e × jump²) = h_e² × jump²
        contrib = edge_lengths**2 * jump_abs**2
        np.add.at(eta_sq, edge_data[:, 2], contrib)
        np.add.at(eta_sq, edge_data[:, 3], contrib)
    eta_sq *= 0.5

    # ── 2. 体力残差: h_K²·∫_K|f^B|²dx = h_K²·A_K·|f|² ──
    if mesh.body_force is not None:
        for eid in range(n_elem):
            xc, yc = mesh.centroids[eid]
            bx, by = evaluate_vector_field(mesh.body_force, xc, yc)
            f_norm2 = bx**2 + by**2
            conn = mesh.elements[eid]
            h_K = max(
                np.linalg.norm(nodes[conn[ib]] - nodes[conn[ia]])
                for ia, ib in mesh.element_kernel.local_edges)
            A_K = abs(mesh.areas[eid])
            eta_sq[eid] += h_K**2 * A_K * f_norm2

    # ── 3. Neumann 边界牵引残差 + 自由边界残差 ──
    # 加载边: h_e·∫_e|σ^h·n−t̄|²ds = h_e²·|σ^h·n−t̄|² (常数)
    # 自由边: h_e·∫_e|σ^h·n−0|²ds = h_e²·|σ^h·n|²
    sigma_e = np.zeros((n_elem, 2, 2))
    sigma_e[:, 0, 0] = stress[:, 0]
    sigma_e[:, 1, 1] = stress[:, 1]
    sigma_e[:, 0, 1] = stress[:, 2]
    sigma_e[:, 1, 0] = stress[:, 2]

    # 收集已施加面力的边并按边合并 — 同一条边被拆成多个面力记录时,
    # 残差必须用合并后的 t̄ (拆分方式不应改变误差指标)。
    loaded_edges = set()
    loaded_by_edge = {}  # key → [st, ...]
    for st in mesh.surface_tractions:
        ni, nj = st["nodes"]
        key = (min(ni, nj), max(ni, nj))
        if key not in mesh.edge_to_elems:
            continue
        if len(mesh.edge_to_elems[key]) == 0:
            continue
        loaded_by_edge.setdefault(key, []).append(st)

    for key, st_list in loaded_by_edge.items():
        eid = mesh.edge_to_elems[key][0]
        loaded_edges.add(key)
        ni, nj = key

        n = np.array(
            mesh.boundary_outward_normal(ni, nj), dtype=float)
        t_fe = sigma_e[eid] @ n

        # 3 点 Gauss-Legendre 线积分 (与 loads_core 同规则): 边中点采样
        # 会低估线性/抛物线面力的残差 (ty=y-0.5 中点值为 0, 真实残差
        # √(1/12)≠0)。integral = (1/h_e)·∫|σn−t̄|²ds (Gauss: 0.5·Σw·f)。
        xa, ya = nodes[ni]
        xb, yb = nodes[nj]
        edge_vec = nodes[nj] - nodes[ni]
        h_e = float(np.linalg.norm(edge_vec))
        # 退化边判据与 loads_core 的坐标 ULP 相对判据统一 —
        # 绝对 1e-30 与文件内已确立的约定不一致 (审计 2026-08-03)
        if h_e <= 64.0 * np.finfo(float).eps * max(
                float(np.max(np.abs(nodes))), np.finfo(float).tiny):
            continue
        integral = 0.0
        for w, xi_g in LINE_GAUSS:
            Ni = 0.5*(1.0 - xi_g)
            Nj = 0.5*(1.0 + xi_g)
            xg = Ni*xa + Nj*xb
            yg = Ni*ya + Nj*yb
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
        # 目标项 = h_e·∫|σn−t̄|²ds = h_e²·integral (常数面力 → h_e²·|r|²)
        eta_sq[eid] += h_e**2 * integral

    # Dirichlet / 自由边界边
    # Dirichlet边(固支): 反力未知, 不能算 σ^h·n − 0, 必须跳过
    # 仅-Ux/Uy边: 保留未约束方向的残差
    # 自由边: t̄=0, 全量残差 ‖σ^h·n‖
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

        # ── Dirichlet分类: 检查边两端DOF ──
        dof_a = (2*a in fixed_dofs_set, 2*a+1 in fixed_dofs_set)  # (ux, uy)
        dof_b = (2*b in fixed_dofs_set, 2*b+1 in fixed_dofs_set)
        both_fixed = lambda d1,d2: d1 and d2
        # 两边节点都是全约束(ux+uy) → 固支边 → 跳过
        if both_fixed(*dof_a) and both_fixed(*dof_b):
            continue

        try:
            n = np.asarray(mesh.boundary_outward_normal(a, b), dtype=float)
        except (ValueError, RuntimeError):
            continue

        t_fe = sigma_e[eid] @ n  # [tx, ty]
        edge_vec = nodes[b] - nodes[a]
        h_e = float(np.linalg.norm(edge_vec))
        # 退化边判据与 loads_core 的坐标 ULP 相对判据统一 —
        # 绝对 1e-30 与文件内已确立的约定不一致 (审计 2026-08-03)
        if h_e <= 64.0 * np.finfo(float).eps * max(
                float(np.max(np.abs(nodes))), np.finfo(float).tiny):
            continue

        # 部分约束: 只保留未约束方向
        # 两边都仅ux约束 → tx方向跳过, ty方向保留
        # 两边都仅uy约束 → ty方向跳过, tx方向保留
        skip_x = dof_a[0] and dof_b[0]
        skip_y = dof_a[1] and dof_b[1]
        res_x = 0.0 if skip_x else t_fe[0]
        res_y = 0.0 if skip_y else t_fe[1]
        eta_sq[eid] += h_e**2 * (res_x**2 + res_y**2)

    return np.sqrt(np.maximum(eta_sq, 0.0))
