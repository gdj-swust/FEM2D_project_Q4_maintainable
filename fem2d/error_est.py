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
import math

import numpy as np

from .assembly import ASSEMBLY_BATCH_ELEMENTS
from .checks import require_finite_positive
from .element import evaluate_vector_field
from .loads_core import LINE_GAUSS
from .material import D_matrix
from .spr import spr_recovery
from .stress import nodal_L2_projection, nodal_weighted


def _energy_density(s_star_pt, s_h, D_inv, use_scale, s_scale):
    """归一化空间的能量密度 (误差 + 恢复) — 应力已除 s_scale,
    柔度已除 d_scale (调用方完成), 三项乘积均 O(1), 不溢出/不下溢。
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

    纯无量纲空间: 应力已除 s_scale, 柔度已除 d_scale (estimate 内完成),
    积分权已除全局 maxw — t·dA·D_inv·σ² 不再作为中间量出现, 中间量
    全部 O(1), eta/contrib 对材料与几何尺度完全不变。乘回物理尺度所需
    的 sqrt(|t|·maxw) 拆成 sqrt(|t|)·sqrt(maxw) 返回 (先开方再相乘,
    防乘积先下溢为 0 后开方失真)。

    批量路径 (shape/weights 缓存) + 兼容路径 (逐单元 recovery_quadrature);
    返回 (elem_err_sq, recovered_energy, s_scale, use_scale, vol_sqrt)。
    s_scale 覆盖全部应力源 (单元代表应力可因积分点正负抵消而≈0,
    弯曲型场归一化会爆炸)。
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
        maxw = float(np.max(np.abs(weights)))
        vol_sqrt = math.sqrt(abs(mesh.thickness)) * math.sqrt(maxw)
        if maxw > 0.0:
            norm_w = weights / maxw  # 全局尺度下的 O(1) 相对权重
            for start in range(0, n_elem, ASSEMBLY_BATCH_ELEMENTS):
                stop = min(start + ASSEMBLY_BATCH_ELEMENTS, n_elem)
                conn = mesh.elements[start:stop]
                s_star_pt = np.einsum("qn,enc->eqc", shape, s_star[conn])
                if stress_qp is None or method == "weighted":
                    # weighted 的 σ* 来自单元均值恢复 — 与积分点原始应力
                    # 量纲语义不可比, 混用会导致精确线性场报 ~14% 虚假误差
                    s_h = stress[start:stop, None, :]
                else:
                    s_h = stress_qp[start:stop]
                    if s_h.shape != s_star_pt.shape:
                        raise ValueError(
                            "Recovered quadrature has shape "
                            f"{s_star_pt.shape}, but stress_qp has shape "
                            f"{s_h.shape}")
                error_density, recovered_density = _energy_density(
                    s_star_pt, s_h, D_inv, use_scale, s_scale)
                elem_err_sq[start:stop] = np.sum(
                    norm_w[start:stop] * error_density, axis=1)
                recovered_energy[start:stop] = np.sum(
                    norm_w[start:stop] * recovered_density, axis=1)
        # maxw == 0: 全零积分权 → 能量全零, vol_sqrt=0 乘回得 0
    else:
        # Compatibility path for external kernels with custom recovery rules.
        quads = [kernel.recovery_quadrature(mesh, eid)
                 for eid in range(n_elem)]
        maxw = max((float(np.max(np.abs(dA))) for _, dA in quads),
                   default=0.0)
        vol_sqrt = math.sqrt(abs(mesh.thickness)) * math.sqrt(maxw)
        if maxw > 0.0:
            for eid, (shape_e, dA) in enumerate(quads):
                s_star_pt = shape_e @ s_star[mesh.elements[eid]]
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
                norm_w = dA / maxw
                elem_err_sq[eid] = float(np.dot(norm_w, error_density))
                recovered_energy[eid] = float(
                    np.dot(norm_w, recovered_density))
    return elem_err_sq, recovered_energy, s_scale, use_scale, vol_sqrt


def estimate(mesh, result, method="SPR", verbose=True):
    """Z2 误差估计 — 基于应力恢复评估网格质量

    Bathe §4.3.6: 通过比较原始应力 σ_h 和改进应力 σ* 来估计误差。

    算法 (积分点离散, 无 A_e 项 — 面积权重由积分权 dA 承载):
      1. 计算改进应力 σ* (L2 投影 或 面积加权平均)
      2. 对每个单元: ||e_e||²_energy = Σ_q w_e,q · (σ*_q - σ_h,q)ᵀ·D⁻¹·(σ*_q - σ_h,q)
         其中 w_e,q = dA_e,q / max_w (积分权全局归一化; 单元厚度 t 与
         应力/柔度尺度一并进入物理量乘回因子, 不影响 η 比值)
      3. ||e||² = Σ_e ||e_e||²_energy
      4. ||U||² = Σ_e Σ_q w_e,q · σ*_qᵀ·D⁻¹·σ*_q
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
    if not isinstance(result, dict) or "stress" not in result:
        # 非 solve() 输出的 dict 会冒裸 KeyError — 契约前置校验
        raise ValueError(
            "estimate_error: result 必须是 solve() 的返回 dict 且含 "
            f"'stress' 键, got {type(result).__name__}")
    stress = np.asarray(result["stress"], dtype=float)
    if not np.all(np.isfinite(stress)):
        # 统一 NaN/Inf 入口防护 (审查 2026-08-06): stress 虽经
        # _estimate_stress_jumps → _traction_jump_arrays 兜底校验,
        # 但前置拒绝可避免 SPR 恢复在 NaN 上白跑 — 非法数据不得静默
        raise ValueError(
            f"estimate: result['stress'] 包含 NaN/Inf — 误差指标无法计算 "
            f"(形状 {stress.shape})")
    stress_qp = _validate_stress_qp_entry(
        "estimate: result['stress_qp']", result.get("stress_qp"))

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
    # 柔度与应力、积分权同尺度归一化 — 不归一化时 t·dA·D_inv·σ² 的
    # 中间量在极端材料/几何尺度下溢出/下溢, eta 静默跳到 100%/0%
    d_scale = max(float(np.max(np.abs(D_inv))), np.finfo(float).tiny)
    D_inv = D_inv / d_scale

    elem_err_sq, recovered_energy, s_scale, use_scale, vol_sqrt = \
        _element_energy_errors(mesh, method, stress, stress_qp, s_star,
                               D_inv)

    elem_err = np.sqrt(np.maximum(elem_err_sq, 0.0))
    if use_scale:
        # 乘回物理尺度 (报告用): s_scale·sqrt(d_scale)·sqrt(|t|·maxw)。
        # 三因子跨度可达 ±300 阶, 连乘必存在一种先溢出/下溢的顺序 —
        # 对数空间求和: 可表示时精确, 超出双精度时诚实 0/inf。
        # math.exp 在 log 和 > log(float_max) 时抛 OverflowError (全有限
        # 输入崩溃, E=1e-150/t=1e150/σ=1e308 复现) — 显式分支代替
        log_factor = (np.log(s_scale) + np.log(vol_sqrt)
                      + 0.5 * np.log(d_scale))
        if log_factor > np.log(np.finfo(float).max):
            factor = np.inf
        elif log_factor < np.log(np.finfo(float).tiny):
            factor = 0.0
        else:
            factor = float(np.exp(log_factor))
        with np.errstate(invalid="ignore", over="ignore"):
            elem_err = elem_err * factor
            elem_err = np.where(elem_err_sq > 0.0, elem_err, 0.0)
    else:
        factor = 0.0  # 零应力场: 绝对量全零
    err_sq_sum = float(np.sum(np.maximum(elem_err_sq, 0.0)))
    U_norm2 = float(np.sum(np.maximum(recovered_energy, 0.0)))

    # ── 3. 全局量 ──
    # eta 在归一化空间直接计算 (elem_err_sq/recovered_energy 未乘回
    # 尺度因子) — 归一化空间的比值完全尺度无关。绝对量 (total_err/
    # U_norm) 乘回供报告; factor=0 (零应力场) 时绝对量全零。
    if err_sq_sum == 0.0:
        # 真零误差场: eta=0 (0/0=NaN)
        total_err = 0.0
        eta = 0.0
    else:
        total_err = math.sqrt(err_sq_sum) * factor
        eta = (math.sqrt(err_sq_sum)
               / math.sqrt(max(U_norm2, np.finfo(float).tiny)) * 100.0)
    if U_norm2 > 0.0:
        U_norm = math.sqrt(U_norm2) * factor
    else:
        # 零恢复能量场: 范数 0
        U_norm = 0.0

    worst = int(np.argmax(elem_err))
    # 贡献百分比在归一化空间计算 (elem_err_sq 未乘回) — 乘回后的
    # elem_err² 在微尺度下为次正规 (~1e-308), 求和失真; 归一化空间
    # 的尺度因子在比值中自动抵消。
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


def _validate_sigma_ref(sigma_ref):
    """sigma_ref 必须为有限正数或 None (显式参考应力).

    统一标量校验 (checks.require_finite_positive): 字符串/容器/None
    带参数名 TypeError, NaN/Inf/0/负值 ValueError — 裸 np.isfinite 对
    字符串冒无上下文的 TypeError, 且 0/负值曾覆盖于 max(...,1e-30) 地板.
    """
    if sigma_ref is not None:
        require_finite_positive(sigma_ref, "sigma_ref")


def _validate_stress_qp_entry(entry_name, stress_qp):
    """stress_qp 入口统一防护 — complex 拒绝 + 有限性校验.

    审查 (2026-08-06): stress_qp 不进 _traction_jump_arrays (直接进
    SPR 恢复 / 不参与计算), 在 estimate / element_refinement_indicator
    入口补同款防护 — 非法数据不得静默进入恢复或被忽略. complex 拒绝
    与 _traction_jump_arrays 同族 (numpy≥2 astype 静默丢虚部).
    """
    if stress_qp is None:
        return None
    raw_qp = np.asarray(stress_qp)
    if np.iscomplexobj(raw_qp):
        raise ValueError(
            f"{entry_name} 必须为实数 — complex 虚部会被静默丢弃")
    qp_arr = np.asarray(raw_qp, dtype=float)
    if not np.all(np.isfinite(qp_arr)):
        raise ValueError(
            f"{entry_name} 包含 NaN/Inf — 误差指标无法计算 "
            f"(形状 {qp_arr.shape})")
    return qp_arr


def _traction_jump_arrays(mesh, elem_stress, sigma_ref=None):
    """Vectorized internal-edge traction jumps.

    Returns ``(edge_data, edge_lengths, jump_abs, jump_rel)`` where
    ``edge_data`` columns are ``node_a,node_b,eid1,eid2``.
    """
    # 公共参数先校验, 再走空数据快速返回 — 单单元网格 (无内部边) 时
    # 校验位于提前返回之后会让非法 sigma_ref 被静默接受并返回空列表.
    _validate_sigma_ref(sigma_ref)
    mesh.build_connectivity()
    raw_stress = np.asarray(elem_stress)
    if np.iscomplexobj(raw_stress):
        # numpy≥2 astype(float) 对 complex 静默丢虚部 (ComplexWarning) —
        # 与 R-α A1 同族拒绝 (fuzz 值池补 complex 数组后暴露)
        raise ValueError(
            "compute_traction_jumps: elem_stress 必须为实数 — "
            "complex 虚部会被静默丢弃")
    stress = np.asarray(raw_stress, dtype=float)
    if stress.ndim != 2 or stress.shape[1] != 3:
        # 裸 IndexError (stress[e1]) 会把用户引向错误的数组索引方向 —
        # 形状契约 (n_elem, 3) 前置校验
        raise ValueError(
            f"elem_stress 必须为 (n_elem, 3) 数组, 得到形状 {stress.shape}")
    if not np.all(np.isfinite(stress)):
        # 审查 (2026-08-06): 缺少有限性校验 — NaN 应力在 np.where 处被
        # 静默归零成 jump_rel=0.0 (当作"无跳跃"), 单单元空数据路径
        # 直接返回空列表吞掉非法数据. 统一入口拒绝 (与 R-α complex
        # 拒绝同族), 校验必须先于空数据提前返回.
        raise ValueError(
            f"elem_stress 包含 NaN/Inf — 误差指标无法计算 "
            f"(形状 {stress.shape})")
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
        # 已由 _validate_sigma_ref 保证有限正数
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
    # 分母 1e-30 绝对地板与 eta/contrib 同族 (已清除)
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


def _logaddexp_scatter(eta_log, eids, terms):
    """logaddexp 散点归约 — 按 eid 排序 + reduceat (替代 ufunc.at).

    at() 无缓冲逐元素散点 (审查报告可优化点 8): 每个 (eid, term) 触发
    一次 Python 级循环。本实现: 稳定排序 (组内保持原顺序) → 组首索引
    → reduceat 左结合逐项累加 — 与 at() 的逐项累加序列逐位一致。
    """
    order = np.argsort(eids, kind="stable")
    sorted_eids = eids[order]
    sorted_terms = terms[order]
    first = np.flatnonzero(
        np.concatenate(([True], sorted_eids[1:] != sorted_eids[:-1])))
    reduced = np.logaddexp.reduceat(sorted_terms, first)
    eta_log[sorted_eids[first]] = np.logaddexp(
        eta_log[sorted_eids[first]], reduced)


def _internal_edge_jump_logs(mesh, stress, eta_log):
    """内部边牵引跳跃 (数组路径 — 免去每条边一个 dict)."""
    edge_data, edge_lengths, jump_abs, _ = _traction_jump_arrays(
        mesh, stress)
    if len(edge_data):
        # h_e × ‖J‖²_{L²(e)} = h_e × (h_e × jump²) = h_e² × jump²
        # jump=0 (零应力边) → log(0)=-inf, logaddexp 视作无该项
        with np.errstate(divide="ignore"):
            log_term = 2.0 * np.log(edge_lengths) + 2.0 * np.log(jump_abs)
        # 两列 (eid1, eid2) 拼接后归约 — 稳定排序保证组内顺序与 at()
        # 的两次逐项累加 (eid1 列全部先于 eid2 列) 完全一致
        eids = np.concatenate((edge_data[:, 2], edge_data[:, 3]))
        terms = np.concatenate((log_term, log_term))
        _logaddexp_scatter(eta_log, eids, terms)
    eta_log += math.log(0.5)


def _body_force_residual_logs(mesh, nodes, n_elem, eta_log):
    """体力残差: h_K²·∫_K|f^B|²dx = h_K²·A_K·|f|².

    向量化 (审查报告可优化点 8): 常数体力整网一次求值 + 批量 h_K/A_K,
    与逐单元路径代数等价 (逐位一致); callable/混合体力逐单元求值
    (值随坐标变化, 无法批量)。
    """
    if mesh.body_force is None:
        return
    bf = mesh.body_force
    if callable(bf) or (isinstance(bf, (tuple, list))
                        and any(callable(c) for c in bf)):
        # 逐单元求值 (原路径)
        for eid in range(n_elem):
            xc, yc = mesh.centroids[eid]
            bx, by = evaluate_vector_field(mesh.body_force, xc, yc)
            f_norm2 = bx**2 + by**2
            if f_norm2 == 0.0:
                continue
            conn = mesh.elements[eid]
            h_K = max(
                np.linalg.norm(nodes[conn[ib]] - nodes[conn[ia]])
                for ia, ib in mesh.element_kernel.local_edges)
            A_K = abs(mesh.areas[eid])
            if h_K == 0.0 or A_K == 0.0:
                continue
            eta_log[eid] = np.logaddexp(
                eta_log[eid],
                2.0 * math.log(h_K) + math.log(A_K) + math.log(f_norm2))
        return
    # 常数体力: 对任何求值点同值 — 一次求值 (校验语义同逐单元首元素)
    bx, by = evaluate_vector_field(bf, *mesh.centroids[0])
    f_norm2 = bx**2 + by**2
    if f_norm2 == 0.0:
        return
    conn = mesh.elements
    # h_K = 各单元最长局部边 (local_edges 顺序与逐单元 max 同序)
    h_K = np.zeros(n_elem)
    for ia, ib in mesh.element_kernel.local_edges:
        edge_len = np.linalg.norm(
            nodes[conn[:, ia]] - nodes[conn[:, ib]], axis=1)
        h_K = np.maximum(h_K, edge_len)
    A_K = np.abs(mesh.areas)
    valid = (h_K > 0.0) & (A_K > 0.0)
    if not np.any(valid):
        return
    terms = (2.0 * np.log(h_K[valid]) + np.log(A_K[valid])
             + math.log(f_norm2))
    # 每单元恰贡献一次 — 与逐单元循环同一逐项累加 (无重复目标, 不需 .at)
    eta_log[valid] = np.logaddexp(eta_log[valid], terms)


def _element_sigma_tensors(stress, n_elem):
    """应力向量 → (n_elem, 2, 2) 对称张量场 (σ^h 逐单元常数)."""
    sigma_e = np.zeros((n_elem, 2, 2))
    sigma_e[:, 0, 0] = stress[:, 0]
    sigma_e[:, 1, 1] = stress[:, 1]
    sigma_e[:, 0, 1] = stress[:, 2]
    sigma_e[:, 1, 0] = stress[:, 2]
    return sigma_e


def _collect_loaded_edges(mesh):
    """收集已施加面力的边并按边合并 — 同一条边被拆成多个面力记录时,
    残差必须用合并后的 t̄ (拆分方式不应改变误差指标)."""
    loaded_by_edge = {}  # key → [st, ...]
    for st in mesh.surface_tractions:
        ni, nj = st["nodes"]
        key = (min(ni, nj), max(ni, nj))
        if key not in mesh.edge_to_elems:
            continue
        if len(mesh.edge_to_elems[key]) == 0:
            continue
        loaded_by_edge.setdefault(key, []).append(st)
    return loaded_by_edge


def _batch_boundary_normals(mesh, lo, hi, eids):
    """批量外法向 — 与 Mesh.boundary_outward_normal 逐边算法代数等价.

    对每条边 (lo,hi) 取其唯一相邻单元 eid, 在单元局部 CCW 边
    (local_edges) 中匹配, 外法向 n = (dy/L, -dx/L) (Bathe §5.3.2:
    CCW 域外法向 = 切向顺时针 90°)。失败边写入对应 fail 掩码, 由
    调用方决定 抛出或跳过 — 与逐边 try/except 语义一致; 零长边
    同时返回 L/ulp 供复现 boundary_outward_normal 的报错文案。

    返回 (normals, fail_value, fail_runtime, L_arr, ulp_arr);
    fail_* 为 True 时对应 normals 行未定义。
    """
    n = len(eids)
    normals = np.zeros((n, 2))
    fail_value = np.zeros(n, dtype=bool)
    fail_runtime = np.zeros(n, dtype=bool)
    L_arr = np.zeros(n)
    ulp_arr = np.zeros(n)
    matched = np.zeros(n, dtype=bool)
    conn = mesh.elements[eids]
    for ia, ib in mesh.element_kernel.local_edges:
        ca, cb = conn[:, ia], conn[:, ib]
        slot = ((ca == lo) & (cb == hi)) | ((ca == hi) & (cb == lo))
        sel = slot & ~matched
        idx_sel = np.flatnonzero(sel)
        if len(idx_sel) == 0:
            continue
        xa, ya = mesh.nodes[ca[idx_sel], 0], mesh.nodes[ca[idx_sel], 1]
        xb, yb = mesh.nodes[cb[idx_sel], 0], mesh.nodes[cb[idx_sel], 1]
        dx, dy = xb - xa, yb - ya
        L = np.hypot(dx, dy)
        # 退化边 ULP 判据同 boundary_outward_normal (逐边坐标, 非全局)
        edge_ulp = 64.0 * np.finfo(float).eps * np.maximum(
            np.maximum(np.abs(xa), np.abs(xb)),
            np.maximum(np.abs(ya), np.abs(yb)))
        edge_ulp = np.maximum(edge_ulp, np.finfo(float).tiny)
        bad = L <= edge_ulp
        if np.any(bad):
            fail_idx = idx_sel[bad]
            fail_value[fail_idx] = True
            L_arr[fail_idx] = L[bad]
            ulp_arr[fail_idx] = edge_ulp[bad]
        ok_idx = idx_sel[~bad]
        normals[ok_idx, 0] = dy[~bad] / L[~bad]
        normals[ok_idx, 1] = -dx[~bad] / L[~bad]
        matched[sel] = True
    fail_runtime = ~matched
    return normals, fail_value, fail_runtime, L_arr, ulp_arr


def _record_is_constant(st):
    """载荷记录是否为处处常数 (值不随 Gauss 点坐标变化).

    常数记录 → 批量 3 点 Gauss 积分; 含 callable 的记录保持逐边求值.
    """
    if st.get("is_pressure"):
        return not callable(st["traction"][0])
    t = st["traction"]
    if callable(t):
        return False
    if isinstance(t, (tuple, list)):
        return not any(callable(c) for c in t)
    return True


def _neumann_edge_residuals(mesh, nodes, sigma_e, loaded_by_edge, eta_log):
    """加载边: h_e·∫_e|σ^h·n−t̄|²ds = h_e²·|σ^h·n−t̄|² (常数).

    向量化 (审查报告可优化点 8): 常数载荷边批量取法向/边长 + 3 点
    Gauss 线积分, 与逐边路径代数等价 (逐位一致); 含 callable 载荷
    的边保持逐边求值 (值随坐标变化, 无法批量)。累加顺序与逐边路径
    一致: 按 loaded_by_edge 字典序收集 (eid, term) 后一次 .at。
    """
    if not loaded_by_edge:
        return set()
    keys = list(loaded_by_edge)
    loaded_edges = set(keys)
    n = len(keys)
    lo = np.array([k[0] for k in keys], dtype=np.int64)
    hi = np.array([k[1] for k in keys], dtype=np.int64)
    eids = np.array([mesh.edge_to_elems[k][0] for k in keys],
                    dtype=np.int64)

    normals, fail_value, fail_runtime, L_arr, ulp_arr = \
        _batch_boundary_normals(mesh, lo, hi, eids)
    if np.any(fail_value) or np.any(fail_runtime):
        # 与逐边路径一致: 加载边法向异常向上抛 (add 时已验证为边界边,
        # 仅退化网格/外部修改可达) — 按边序抛第一个失败, 文案同
        # Mesh.boundary_outward_normal
        for i in range(n):
            if fail_value[i]:
                raise ValueError(
                    f"Zero-length edge ({lo[i]},{hi[i]}) "
                    f"(L={L_arr[i]:.3e} <= ULP {ulp_arr[i]:.3e}).")
            if fail_runtime[i]:
                raise RuntimeError(
                    f"Boundary edge ({lo[i]},{hi[i]}) not found in "
                    f"adjacent element {eids[i]}. This should not happen "
                    f"— check mesh consistency.")
    # σ^h·n — 批量 matmul, 与逐边 (2,2)@(2,) 同一内核 (逐位一致)
    t_fe = np.matmul(sigma_e[eids], normals[:, :, None])[:, :, 0]
    edge_vec = nodes[hi] - nodes[lo]
    h_e = np.linalg.norm(edge_vec, axis=1)
    # 退化边判据与 loads_core 的坐标 ULP 相对判据统一 (全局阈值, 同逐边)
    deg = h_e <= 64.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(nodes))), np.finfo(float).tiny)

    const_mask = np.zeros(n, dtype=bool)
    for i, k in enumerate(keys):
        st_list = loaded_by_edge[k]
        if all(_record_is_constant(st) for st in st_list):
            const_mask[i] = True
    const_idx = np.flatnonzero(const_mask)

    term = np.full(n, -np.inf)   # -inf = 无贡献 (跳过/零积分语义一致)
    if len(const_idx):
        # ── 常数边: 3 点 Gauss 线积分向量化 (t̄ 处处相同) ──
        # integral = (1/h_e)·∫|σn−t̄|²ds (Gauss: 0.5·Σw·f); 目标项
        # = h_e·∫|σn−t̄|²ds = h_e²·integral。逐记录累加顺序同逐边路径。
        t_exact = np.zeros((len(const_idx), 2))
        for r_i, i in enumerate(const_idx):
            for st in loaded_by_edge[keys[i]]:
                if st.get("is_pressure"):
                    p = float(st["traction"][0])
                    t_exact[r_i, 0] -= p * normals[i, 0]
                    t_exact[r_i, 1] -= p * normals[i, 1]
                else:
                    t_exact[r_i, 0] += float(st["traction"][0])
                    t_exact[r_i, 1] += float(st["traction"][1])
        r = t_fe[const_idx] - t_exact
        r2 = r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1]
        integral = np.zeros(len(const_idx))
        integral += 0.5 * LINE_GAUSS[0][0] * r2
        integral += 0.5 * LINE_GAUSS[1][0] * r2
        integral += 0.5 * LINE_GAUSS[2][0] * r2
        ok = (~deg[const_idx]) & (integral > 0.0)
        if np.any(ok):
            term[const_idx[ok]] = (
                2.0 * np.log(h_e[const_idx[ok]]) + np.log(integral[ok]))
    # ── callable 边: 逐边 3 点 Gauss (值随坐标变化, 无法批量) ──
    for i in np.flatnonzero(~const_mask):
        st_list = loaded_by_edge[keys[i]]
        if deg[i]:
            continue
        integral = 0.0
        for w, xi_g in LINE_GAUSS:
            Ni = 0.5 * (1.0 - xi_g)
            Nj = 0.5 * (1.0 + xi_g)
            xg = Ni * nodes[lo[i], 0] + Nj * nodes[hi[i], 0]
            yg = Ni * nodes[lo[i], 1] + Nj * nodes[hi[i], 1]
            t_exact = np.zeros(2)
            for st in st_list:
                if st.get("is_pressure"):
                    p_val = st["traction"][0]
                    p = p_val(xg, yg) if callable(p_val) else p_val
                    t_exact += np.array([-p * normals[i, 0],
                                         -p * normals[i, 1]])
                else:
                    tx, ty = evaluate_vector_field(st["traction"], xg, yg)
                    t_exact += np.array([tx, ty])
            residual = t_fe[i] - t_exact
            integral += 0.5 * w * np.dot(residual, residual)
        if integral > 0.0:
            term[i] = 2.0 * math.log(h_e[i]) + math.log(integral)
    valid = np.isfinite(term)
    if np.any(valid):
        np.logaddexp.at(eta_log, eids[valid], term[valid])
    return loaded_edges


def _boundary_edge_residuals(mesh, nodes, sigma_e, loaded_edges, eta_log):
    """Dirichlet / 自由边界边残差 — 固支跳过, 部分约束只保留未约束方向.

    向量化 (审查报告可优化点 8): 批量取法向/边长 + DOF 分类, 与逐边
    路径代数等价 (逐位一致)。累加顺序与逐边路径一致: 按边界边集合
    迭代序收集 (eid, term) 后一次 .at。
      Dirichlet边(固支): 反力未知, 不能算 σ^h·n − 0, 必须跳过
      仅-Ux/Uy边: 保留未约束方向的残差
      自由边: t̄=0, 全量残差 ‖σ^h·n‖
    """
    all_boundary_edges = list(set(mesh.boundary_edges))  # set 迭代序同逐边
    if not all_boundary_edges:
        return
    n = len(all_boundary_edges)
    lo = np.array([e[0] for e in all_boundary_edges], dtype=np.int64)
    hi = np.array([e[1] for e in all_boundary_edges], dtype=np.int64)
    # 与逐边路径同判据: 已加载边/不在边表/空邻接 → 跳过 (边界边必在
    # 边表且邻接非空, 判据保留以防御网格被外部修改)
    eids = np.full(n, -1, dtype=np.int64)
    for i, k in enumerate(all_boundary_edges):
        if k in loaded_edges:
            continue
        el = mesh.edge_to_elems.get(k)
        if not el:
            continue
        eids[i] = el[0]
    sel = eids >= 0
    if not np.any(sel):
        return
    lo, hi, eids = lo[sel], hi[sel], eids[sel]

    normals, fail_value, fail_runtime, _L, _ulp = \
        _batch_boundary_normals(mesh, lo, hi, eids)
    # 法向失败 (零长/匹配失败) → 跳过 — 与逐边 except continue 一致
    ok_normal = ~(fail_value | fail_runtime)
    if not np.any(ok_normal):
        return
    lo, hi, eids = lo[ok_normal], hi[ok_normal], eids[ok_normal]
    normals = normals[ok_normal]

    # ── Dirichlet 分类: 检查边两端 DOF (逐边 set 成员查询向量化) ──
    fixed = mesh.fixed_dofs
    skip_x = np.isin(2 * lo, fixed) & np.isin(2 * hi, fixed)
    skip_y = np.isin(2 * lo + 1, fixed) & np.isin(2 * hi + 1, fixed)
    # 两边节点都是全约束 (ux+uy) → 固支边 → 跳过
    free = ~(skip_x & skip_y)
    if not np.any(free):
        return
    lo, hi, eids = lo[free], hi[free], eids[free]
    normals = normals[free]
    skip_x, skip_y = skip_x[free], skip_y[free]

    # σ^h·n — 批量 matmul, 与逐边 (2,2)@(2,) 同一内核 (逐位一致)
    t_fe = np.matmul(sigma_e[eids], normals[:, :, None])[:, :, 0]
    edge_vec = nodes[hi] - nodes[lo]
    h_e = np.linalg.norm(edge_vec, axis=1)
    # 退化边判据与 loads_core 的坐标 ULP 相对判据统一 (全局阈值, 同逐边)
    deg = h_e <= 64.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(nodes))), np.finfo(float).tiny)

    # 部分约束: 只保留未约束方向 (两边都仅ux约束 → tx跳过, ty保留)
    res_x = np.where(skip_x, 0.0, t_fe[:, 0])
    res_y = np.where(skip_y, 0.0, t_fe[:, 1])
    res2 = res_x**2 + res_y**2
    ok = (~deg) & (res2 > 0.0)
    if np.any(ok):
        term = 2.0 * np.log(h_e[ok]) + np.log(res2[ok])
        np.logaddexp.at(eta_log, eids[ok], term)


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

    全部中间量在对数空间累加 (logsumexp) — 几何/应力尺度极端时
    h²·|J|² 等项直接相乘会溢出为 inf (排序静默失效) 或下溢为 0
    (项丢失); 对数空间两项均不发生, 最终开方乘回。
    """
    mesh.build_connectivity()
    if not isinstance(result, dict) or "stress" not in result:
        # 与 estimate 同契约 — 非 solve() 输出会冒裸 KeyError
        raise ValueError(
            "element_refinement_indicator: result 必须是 solve() 的返回 "
            f"dict 且含 'stress' 键, got {type(result).__name__}")
    # result["stress"] 经 _internal_edge_jump_logs → _traction_jump_arrays
    # 统一有限性校验; stress_qp 为本函数不使用但 solve() 契约携带的字段 —
    # 同族防护: 非法数据不得被静默忽略 (审查 2026-08-06)
    _validate_stress_qp_entry(
        "element_refinement_indicator: result['stress_qp']",
        result.get("stress_qp"))
    stress = result["stress"]
    n_elem = mesh.n_elements
    nodes = mesh.nodes

    # 对数空间累加 (logsumexp) — 几何/应力尺度极端时直接相乘会
    # 溢出为 inf 或下溢为 0; 对数空间两项均不发生, 最终开方乘回。
    eta_log = np.full(n_elem, -np.inf)
    _internal_edge_jump_logs(mesh, stress, eta_log)
    _body_force_residual_logs(mesh, nodes, n_elem, eta_log)

    # ── 3. Neumann 边界牵引残差 + 自由边界残差 ──
    # 加载边: h_e·∫_e|σ^h·n−t̄|²ds = h_e²·|σ^h·n−t̄|² (常数)
    # 自由边: h_e·∫_e|σ^h·n−0|²ds = h_e²·|σ^h·n|²
    sigma_e = _element_sigma_tensors(stress, n_elem)
    loaded_by_edge = _collect_loaded_edges(mesh)
    loaded_edges = _neumann_edge_residuals(
        mesh, nodes, sigma_e, loaded_by_edge, eta_log)
    _boundary_edge_residuals(mesh, nodes, sigma_e, loaded_edges, eta_log)

    return np.exp(0.5 * eta_log)
