"""求解器 — Bathe §8.2: 对称正定系统的直接求解

默认使用消去法 (Bathe §4.2.2) + 稀疏直接求解器 (SuperLU)。
乘大数法作为备选方案保留。

求解流程:
  1. 组装全局 K (稀疏 CSR) + F
  2. 划分自由/约束 DOF
  3. 消去法: K_aa · U_a = R_a - K_ab · U_b → spsolve
     或 乘大数法: (K + ΔK)·U = F_mod → spsolve
  4. 残差检查 + 条件数估计 (Bathe §8.2.6)
  5. 应力计算

Bathe §8.2.6 Eq (8.65): 相对误差 ≈ cond(K) × ε_machine
  条件数 κ(K) = λ_max / λ_min → 评估解的精度损失位数
  κ ≈ 1e12 → 约损失 12 位十进制精度, 剩余约 4 位有效数字 → 警告
"""
import warnings

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import MatrixRankWarning, eigsh, spsolve

from .assembly import assemble_sparse
from .bc import (
    _validate_system_inputs,
    apply_elimination,
    apply_penalty,
)
from .errors import UnderconstrainedError
from .loads import assemble as assemble_loads
from .material import von_mises


def estimate_condition(K, method="auto"):
    """刚度矩阵条件数估计 (Bathe §8.2.6)

    Bathe Eq (8.65): ||δU||/||U|| ≈ cond(K) × ε_machine
    - ε_machine (float64) ≈ 2.2e-16, ≈ 15.6 位十进制精度
    - cond(K) = 10^12 → 最坏损失 ~12 位, 典型损失 ~10 位 (log10(κ)-2)
    - digits_lost = max(log10(κ) - 2, 0): 经验估计, 非最坏界

    参数
    ----
    K : csr_matrix — 刚度矩阵 (子矩阵, 如 K_aa)
    method : str — "auto" (默认: n_dof<500→dense, >=500→sparse)

    返回
    ----
    dict (7 键, 成功/失败形状一致):
        condition_number : float or None — cond(K); 特征值求解失败时为 None
        lambda_min / lambda_max : float or None — 最小/最大特征值
        digits_lost : float or None — 估计的精度损失位数
        status : str — "GOOD"/"OK"/"WARN"/"CRITICAL"/"SINGULAR?"/"SKIP"
        error : str or None — 失败原因 (成功时为 None)
    """
    if not issparse(K):
        K = np.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        # tuple/list/标量/非方阵曾冒裸 AttributeError ('.shape') —
        # 契约: 形状错误带上下文
        raise ValueError(
            f"estimate_condition: K 必须为方阵 (n×n), "
            f"got {getattr(K, 'shape', type(K).__name__)}")
    n = K.shape[0]
    method = str(method).strip().lower()
    if method not in ("auto", "dense", "sparse"):
        # 非法 method 曾静默降级到 sparse 路径并成功返回 (拼错无察觉) —
        # 契约: 用户可控字符串非法值必须响亮失败
        raise ValueError(
            f"estimate_condition: 未知 method {method!r} — "
            "仅支持 auto/dense/sparse")

    # Bathe §8.2.6: λ_min 和 λ_max 的最可靠估计来自刚度矩阵本身
    # 对于 SPD 矩阵, cond(K) = λ_max/λ_min
    try:
        if method == "auto":
            method = "dense" if n < 500 else "sparse"

        if method == "dense":
            Kd = K.toarray() if hasattr(K, 'toarray') else K
            eigs = np.linalg.eigvalsh(Kd)
            lam_min, lam_max = eigs[0], eigs[-1]
        else:
            # 稀疏特征值: 求最大和最小特征值
            # λ_max: which='LM' (Largest Magnitude) — 始终稳健
            lam_max = eigsh(K, k=1, which='LM', return_eigenvectors=False)[0]
            # λ_min: shift-invert mode (sigma=0 → K⁻¹ 的最大特征值)
            #   比 which='SM' 更稳健: 对接近奇异的矩阵 (约束不足/孤立自由度)
            #   which='SM' 可能不收敛或返回伪零, shift-invert 显式求解 K·x = y
            #   Bathe §8.2.6: λ_min → 0 时条件数 → ∞, 需可靠估计
            lam_min = eigsh(K, k=1, sigma=0.0, which='LM',
                           return_eigenvectors=False)[0]

        cond = lam_max / lam_min if lam_min > 0 else float('inf')
        digits_lost = max(np.log10(max(cond, 1.0)) - 2, 0.0)  # log10(κ)-2 ≈ 有效位数损失(Bathe §8.2.6)

        if cond > 1e14:
            status = "CRITICAL"
        elif cond > 1e12:
            status = "WARN"
        elif cond > 1e8:
            status = "OK"
        else:
            status = "GOOD"

        return {
            "condition_number": cond,
            "lambda_min": lam_min,
            "lambda_max": lam_max,
            "digits_lost": digits_lost,
            "status": status,
            "error": None,
        }
    except Exception as e:
        # shift-invert 在奇异矩阵上可靠失败 — 正是最需要报警的场景。
        # 特征值求解器的数值失败 (奇异/秩亏矩阵) 单独归为 "SINGULAR?",
        # 与 "SKIP" (规模过大/其他原因) 区分, 调用方据此给出不同提示。
        # 失败路径补齐成功路径的全部键 (None 填充) — 曾缺
        # lambda_min/lambda_max/digits_lost 四键, 调用方逐键访问崩溃
        from numpy.linalg import LinAlgError
        try:
            from scipy.sparse.linalg import ArpackError
        except ImportError:
            ArpackError = RuntimeError  # 老 scipy 无 ArpackError
        if isinstance(e, (LinAlgError, ArpackError, RuntimeError)):
            status = "SINGULAR?"
        else:
            status = "SKIP"
        return {
            "condition_number": None,
            "lambda_min": None,
            "lambda_max": None,
            "digits_lost": None,
            "status": status,
            "error": f"{type(e).__name__}: {e}",
        }


def _solve_with_singular_guard(fn, *args, **kwargs):
    """Run a sparse solve, converting MatrixRankWarning into a loud error."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fn(*args, **kwargs)
    # catch_warnings 捕获即吞 — 非秩警告 (如 spsolve 的近奇异 RuntimeWarning)
    # 必须在块外原样转发 (块内转发会被同一记录器重新捕获, 列表自增长);
    # 守卫只把秩警告转成异常, 不选择性丢弃其他警告
    for wi in w:
        if not issubclass(wi.category, MatrixRankWarning):
            warnings.warn_explicit(
                wi.message, wi.category, wi.filename, wi.lineno)
    for wi in w:
        if issubclass(wi.category, MatrixRankWarning):
            raise RuntimeError(
                "Linear system is singular or ill-conditioned. "
                "Check boundary constraints, isolated nodes, "
                "degenerate elements, and disconnected components."
            ) from None
    return result


def _balance_failure_message(sum_F, tol_rel, tol_noise, sum_M, m_scale):
    """平衡检查失败消息 — 独立函数使判据与消息共享同一分母."""
    return (
        f"Global equilibrium NOT satisfied: "
        f"ΣF rel={np.linalg.norm(sum_F)/max(tol_rel, tol_noise):.2e}, "
        f"ΣM rel={abs(sum_M)/m_scale:.2e}. "
        f"System may be singular or near-singular — "
        f"check boundary constraints and load paths.")


# ── solve 的独立阶段 (每阶段可单测, 2026-08 拆分) ──

def _solve_linear_system(K, F, free_dofs, fixed_dofs, prescribed,
                         method, linear_solver, n_dof, log):
    """求解线性系统 — 纯 Dirichlet / 消去法 / 乘大数法 三分支.

    返回 (u, reactions, K_mod, F_mod, linear_solver_info, dirichlet_only).
    """
    K_mod, F_mod = None, None  # penalty 方法使用, 残差检查需要
    linear_solver_info = {"name": "none", "iterations": 0}
    _validate_system_inputs(K, F)
    # solver 名称在任何分支前统一校验一次 — 曾纯 Dirichlet 分支 (全约束)
    # 在名称检查前返回, linear_solver="bogus" 静默成功; 且 elimination
    # 分支内逐字重复第二遍校验, 两处必须保持同步
    solver_key = str(linear_solver).strip().lower()
    if solver_key == "cg-block":
        solver_key = "cg"  # 兼容别名 — 归一后各分支只认 4 种键
    if solver_key not in {"direct", "cg", "ilu", "auto"}:
        raise ValueError(
            f"Unknown linear_solver '{linear_solver}'; "
            "expected auto, direct, cg, cg-block or ilu.")
    dirichlet_only = (method == "elimination" and len(free_dofs) == 0)
    if dirichlet_only:
        # 纯 Dirichlet: u 全部已知, 直接算反力 (Bathe §4.2.2)
        u = np.zeros(n_dof)
        u[fixed_dofs] = prescribed
        R_full = K.dot(u) - F
        reactions = R_full[fixed_dofs]
    elif method == "elimination":
        if solver_key == "auto":
            solver_key = "cg" if len(free_dofs) >= 100000 else "direct"
        solver_label = {
            "direct": "SuperLU",
            "cg": "Jacobi-PCG",
            "ilu": "ILU-PCG",
        }[solver_key]
        log(f"[Solver] 消去法 + {solver_label} ...")
        u, reactions, linear_solver_info = _solve_with_singular_guard(
            apply_elimination,
            K, F, free_dofs, fixed_dofs, prescribed,
            linear_solver=solver_key, return_info=True,
            _system_validated=True)
        if linear_solver_info["name"] == "cg":
            log(
                f"  PCG ({linear_solver_info['preconditioner']}) iterations = "
                f"{linear_solver_info['iterations']} "
                f"(rtol={linear_solver_info['rtol']:.1e})")
    elif method == "penalty":
        if solver_key not in {"auto", "direct"}:
            raise ValueError(
                "Penalty constraints currently require the direct solver.")
        log("[Solver] 乘大数法 + spsolve ...")
        K_mod, F_mod, penalty = apply_penalty(
            K, F, fixed_dofs, prescribed, _system_validated=True)
        log(f"  penalty = {penalty:.2e} "
            "(k_penalty = max(|K_ii|) × 1e8, additive)")
        u = _solve_with_singular_guard(spsolve, K_mod, F_mod)
        # penalty: 计算等效反力 R = K·u - F (用于平衡检查)
        reactions = (K.dot(u) - F)[fixed_dofs]
        linear_solver_info = {"name": "direct", "iterations": 1}
    else:
        raise ValueError(f"Unknown method: {method}")
    return (u, reactions, K_mod, F_mod,
            linear_solver_info, dirichlet_only)


def _check_solution_finite(u, reactions, method):
    """解与反力的有限性 — NaN/Inf 快速失败."""
    if not np.all(np.isfinite(u)):
        raise RuntimeError(
            "Solution contains NaN or Inf. "
            "Likely cause: singular stiffness matrix, insufficient "
            "constraints, or degenerate elements. Check mesh quality "
            "and boundary conditions.")
    if method == "elimination" and reactions is not None:
        if not np.all(np.isfinite(reactions)):
            raise RuntimeError(
                "Reaction forces contain NaN or Inf — matrix may be "
                "ill-conditioned.")


def _compute_residual(K, F, K_mod, F_mod, u, free_dofs,
                      method, dirichlet_only, log):
    """Backward error 残差 (Bathe §8.2.6, ∞-范数) + 报告 + 阈值拒绝.

    ||r||_∞ / (||K||_∞·||u||_∞ + ||F||_∞) — 矩阵/向量 ∞-范数相容,
    避免 Frobenius 范数放大分母。返回 (residual, residual_abs)。
    """
    if dirichlet_only:
        residual = 0.0
        residual_abs = 0.0
    elif method == "elimination":
        # 自由行上的原方程 K_a·u = F_a。使用完整自由行避免再次构造
        # K_aa/K_ab; 也是含非零给定位移时的一致 backward error。
        rhs = F[free_dofs]
        r = (K.dot(u) - F)[free_dofs]
        r_inf = np.linalg.norm(r, ord=np.inf)
        row_counts = np.diff(K.indptr)
        row_sums = np.zeros(K.shape[0], dtype=float)
        nonempty = np.flatnonzero(row_counts)
        row_sums[nonempty] = np.add.reduceat(
            np.abs(K.data), K.indptr[nonempty])
        K_inf = row_sums[free_dofs].max()
        u_inf = np.linalg.norm(u, ord=np.inf)
        f_inf = np.linalg.norm(rhs, ord=np.inf)
    else:
        r = K_mod.dot(u) - F_mod
        r_inf = np.linalg.norm(r, ord=np.inf)
        K_inf = np.asarray(np.abs(K_mod).sum(axis=1)).ravel().max()
        u_inf = np.linalg.norm(u, ord=np.inf)
        f_inf = np.linalg.norm(F_mod, ord=np.inf)
    if not dirichlet_only:
        denom = K_inf * u_inf + f_inf + np.finfo(float).tiny
        if not np.isfinite(denom) or not np.isfinite(r_inf):
            # 极端 E/thickness 使刚度行和/残差溢出 — 相对残差无法定义,
            # 显式置 1.0 (必被残差检查拒绝, 不会显示"残差 0"假象)
            residual = 1.0
        else:
            residual = r_inf / denom
        residual_abs = r_inf
        # 曾 denom<1e-15 绝对判据: 微尺度载荷的非平凡解 (max|u|=1.87e-9)
        # 被误标 "trivial solution"。"平凡解"标签只
        # 属于真正零解 (u≡0), 非平凡解一律走相对残差日志。
        if u_inf == 0.0:
            if residual_abs > 1e-10:
                log(f"  [WARN] ||K·u - F|| = {residual_abs:.3e} "
                    "(zero load, zero displacement)")
            else:
                log(f"  [OK] ||K·u - F|| = {residual_abs:.2e} "
                    "(trivial solution)")
        elif residual > 1e-8:
            log(f"  [WARN] Residual = {residual:.3e} "
                "(backward error, large)")
        else:
            log(f"  [OK] Residual = {residual:.2e} (backward error)")

    log(f"[Solver] max|u| = {np.max(np.abs(u)):.6e}")

    # 残差异常: 近奇异矩阵可能被 splu 勉强分解, 位移有限但物理上无意义
    if not dirichlet_only and (not np.isfinite(residual)
                               or residual > 1e-3):
        raise RuntimeError(
            f"Residual {residual:.3e} is too large "
            f"({residual:.1e} > 1e-3). "
            "System is likely near-singular or ill-conditioned. "
            "Check boundary constraints, mesh quality, and load balance.")
    return residual, residual_abs


def _global_balance_check(mesh, K, F, u, reactions, fixed_dofs,
                          dirichlet_only, log):
    """全局力/力矩平衡检查 (Bathe §4.2.2).

    残差小只说明线性方程解得准, 不代表载荷方向/厚度/边界正确。
    失衡时抛 RuntimeError (含 ΣF/ΣM 相对值)。
    返回 (R_full, sum_F, sum_M, balance_ok) 供 result 组装; 无反力时 None。
    """
    if reactions is None or len(reactions) == 0:
        return None
    # 支反力向量 (n_dof,)
    R_full = np.zeros(2 * mesh.n_nodes)
    R_full[fixed_dofs] = reactions
    R_2d = R_full.reshape(-1, 2)
    F_2d = F.reshape(-1, 2)

    sum_F = R_2d.sum(axis=0) + F_2d.sum(axis=0)  # ΣR + ΣF
    # 力矩对形心取矩: ΣF ≈ 0 时力矩与取矩点无关 (物理等价), 但对全局
    # 原点取矩时单项量级含坐标偏移, 浮点抵消误差按 ε·|x_0|·|F| 增长 —
    # 远距坐标系下会误杀合法算例; 形心取矩消除该放大因子。
    centroid = mesh.nodes.mean(axis=0)
    rel = mesh.nodes - centroid
    mx_R = rel[:, 0] * R_2d[:, 1] - rel[:, 1] * R_2d[:, 0]
    mx_F = rel[:, 0] * F_2d[:, 1] - rel[:, 1] * F_2d[:, 0]
    sum_M = float(mx_R.sum() + mx_F.sum())

    # 内部力尺度 = max 行和(K) × max|u| — 位移驱动/近刚体工况下
    # 反力退化为舍入噪声, 需以它为噪声判据的尺度基准
    row_sums = np.asarray(np.abs(K).sum(axis=1)).ravel()
    internal_scale = float(np.max(row_sums)) * float(np.max(np.abs(u)))
    # ΣF 双判据: ① 相对载荷判据 (有外载时保持敏感度);
    # ② 内部力尺度噪声判据 (无外载时以浮点舍入噪声界兜底)
    tol_rel = 1e-8 * max(float(np.linalg.norm(F)),
                         float(np.linalg.norm(R_full)),
                         np.finfo(float).tiny)
    tol_noise = 1e-13 * internal_scale
    # 分母用未抵消的单项绝对值之和 — 对自平衡体系总力矩 ≈ 0,
    # 若拿它做分母会把相对判据退化成绝对判据
    m_scale = max(
        float(np.abs(mx_R).sum() + np.abs(mx_F).sum()),
        np.finfo(float).tiny)

    log(f"[Solver] ΣF = ({sum_F[0]:.3e}, {sum_F[1]:.3e}) N  "
        f"(rel: {np.linalg.norm(sum_F)/max(tol_rel, tol_noise):.2e})")
    log(f"[Solver] ΣM = {sum_M:.3e} N·m  "
        f"(rel: {abs(sum_M)/m_scale:.2e})")

    # 纯 Dirichlet (全约束) 跳过平衡检查 — ΣR = K·u - F 为反力定义
    balance_ok = True
    if not dirichlet_only:
        # 位移驱动/近刚体工况: 反力退化为舍入噪声, 力矩相对判据的
        # 分母同量级退化 → 误报合法算例。反力低于 1e-10 倍内部力尺度
        # 时力矩检查无物理意义, 跳过 (ΣF 检查保留)。
        true_f_scale = max(float(np.linalg.norm(F)),
                           float(np.linalg.norm(R_full)))
        balance_ok = np.linalg.norm(sum_F) <= max(tol_rel, tol_noise)
        if true_f_scale >= 1e-10 * internal_scale:
            # 力矩判据: 相对判据 (5e-8×m_scale) + 噪声兜底。对称网格 +
            # 自平衡载荷 (完美对称压力环/对拉) 下 Σ|单项力矩| 也互相
            # 抵消到浮点噪声级, 相对分母退化 → 误杀合法算例; 绝对噪声
            # 界 = 内部力尺度 × 特征长度 × 1e-13 (与 ΣF 的 tol_noise 同源)。
            char_length = float(np.max(np.linalg.norm(
                mesh.nodes - mesh.nodes.mean(axis=0), axis=1)))
            moment_noise = 1e-13 * internal_scale * char_length
            balance_ok = balance_ok and (
                abs(sum_M) <= max(5e-8 * m_scale, moment_noise))
        else:
            log("  [skip] reactions at round-off level — "
                "moment balance check skipped")
        if not balance_ok:
            raise RuntimeError(_balance_failure_message(
                sum_F, tol_rel, tol_noise, sum_M, m_scale))
    return R_full, sum_F, sum_M, balance_ok


def _small_deformation_check(mesh, u):
    """位移量级检查 — 大变形假设失效时警告."""
    # np.hypot: 平方和再开方在跨度 ~1e308 时溢出成 inf, 位移比恒 0
    model_span = float(np.hypot(
        np.ptp(mesh.nodes[:, 0]), np.ptp(mesh.nodes[:, 1])))
    u2 = u.reshape(-1, 2)
    u_range = max(np.ptp(u2[:, 0]), np.ptp(u2[:, 1]))
    # 曾 model_span > 1e-30 绝对阈值: 跨度 1e-31 的模型位移比 20% 也不
    # 告警。判据相对化: span 仅需可表示
    # (> 0), 位移比为 0 时 (纯刚体平移) 无意义。
    span_pos = model_span > np.finfo(float).tiny
    ok = not (span_pos and u_range > 0.0
              and u_range / model_span > 0.1)
    if not ok:
        warnings.warn(
            f"Displacement variation = {u_range:.3e} m "
            f"({u_range/model_span:.2f} × model span). "
            f"Small-deformation assumption may not hold.",
            RuntimeWarning, stacklevel=2)
    return ok


def _check_result_finite(stress, strain, vm):
    """结果有限性 — 极端材料/载荷下应力等溢出为 NaN/Inf 时快速失败."""
    for name, arr in (("stress", stress), ("strain", strain),
                      ("vm_stress", vm)):
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(
                f"Result {name} contains NaN/Inf — 材料参数或载荷可能"
                "超出数值范围 (极端 E/thickness/应力), 检查模型设置。")


# ── solve 的其余独立阶段 (2026-08-03 拆分为纯编排层) ──

def _partition_dofs(mesh, method, n_dof, log):
    """自由/约束 DOF 划分; 纯 Dirichlet (全约束) 问题提示."""
    fixed_dofs = mesh.fixed_dofs.astype(int)
    prescribed = np.array(
        [mesh.prescribed_vals.get(d, 0.0) for d in fixed_dofs])
    free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)
    log(f"[Solver] 约束: {len(fixed_dofs)} DOFs, {len(free_dofs)} free DOFs")
    if method == "elimination" and len(free_dofs) == 0:
        # 纯 Dirichlet 问题合法 — 组装 K 后直接算反力 (无需求解线性系统)
        log(f"[Solver] 全部 {n_dof} DOF 给定位移 — 直接计算反力")
    return fixed_dofs, prescribed, free_dofs


def _check_rigid_body_constraints(mesh):
    """逐连通分量刚体模态检查 (Bathe §4.2.2).

    任何分量 rank < 3 都意味着存在未约束的刚体模态 → 系统奇异 → 禁止求解.
    """
    rb_issues = mesh.check_rigid_body_constraints()
    if not rb_issues:
        return
    lines = ["Rigid-body modes are not fully constrained:"]
    for iss in rb_issues:
        nodes_str = (f"nodes {iss['nodes'][0]}~{iss['nodes'][-1]}"
                     if len(iss['nodes']) > 3
                     else f"nodes {iss['nodes']}")
        lines.append(
            f"  Component {iss['component']} ({nodes_str}): {iss['issue']}")
    lines.append(
        "Fix boundary conditions on all disconnected parts before solving.")
    raise UnderconstrainedError("\n".join(lines))


def _q4r_aspect_ratio_warning(mesh):
    """Q4R 稳定性预处理告警: 按单元长宽比而非沙漏能占比.

    长宽比 ≥50 时过刚 (解只有解析值 ~2%), L/h≈10 少行时过柔 (~5倍),
    且沙漏能占比在两种情况下都 >90% — 占比不是可靠性指标, 预处理
    阶段按长宽比直接告警 (实测长宽比 12 的 8×2 网格已偏软 32%)。
    """
    if not hasattr(mesh.element_kernel, "hourglass_energy"):
        return
    q4r_coords = mesh.nodes[mesh.elements]
    # 4 条局部边含闭合边 3→0 (漏算会低估长宽比告警)
    q4r_edge = np.roll(q4r_coords, -1, axis=1) - q4r_coords
    q4r_len = np.linalg.norm(q4r_edge, axis=2)
    q4r_ar = q4r_len.max(axis=1) / np.maximum(
        q4r_len.min(axis=1), np.finfo(float).tiny)
    _q4r_ar = float(q4r_ar.max())
    if _q4r_ar >= 50.0:
        # 文档失效区 (q4r.py): 稳定刚度不随长宽比衰减, 过刚,
        # 解可能只有解析值 ~2%
        warnings.warn(
            f"Q4R (CPS4R/CPE4R) 网格单元长宽比最大 {_q4r_ar:.0f} >= 50 "
            "-- compact hourglass 稳定公式的文档失效区 (解可能只有 "
            "解析值 ~2%, 偏硬)。强烈建议换用 Q4I (CPS4I/CPE4I) 或"
            "细化网格。",
            RuntimeWarning)
    elif _q4r_ar > 10.0:
        # 中等长宽比可能过柔也可能过刚 (尺度特性, 见 q4r.py)
        warnings.warn(
            f"Q4R (CPS4R/CPE4R) 网格单元长宽比最大 {_q4r_ar:.1f} > 10 "
            "-- 减缩积分单元在薄板/少行网格上不稳定 (过刚或过柔, "
            "沙漏能占比不可靠)。建议换用 Q4I (CPS4I/CPE4I) 或规则网格。",
            RuntimeWarning)


def _compute_element_response(mesh, u_e):
    """积分点响应 → 单元平均应力/应变/von Mises + 原始积分点数据."""
    stress_qp, strain_qp, dA_qp = (
        mesh.element_kernel.response_at_quadrature(mesh, u_e))
    area_qp = np.sum(dA_qp, axis=1)
    if np.any(area_qp <= 0.0):
        raise RuntimeError(
            "Element response quadrature returned non-positive area weights.")
    stress = np.sum(
        stress_qp * dA_qp[:, :, None], axis=1) / area_qp[:, None]
    strain = np.sum(
        strain_qp * dA_qp[:, :, None], axis=1) / area_qp[:, None]
    vm = von_mises(stress, mesh.plane_type, mesh.nu)
    return stress, strain, vm, stress_qp, strain_qp, dA_qp


def _hourglass_monitor(mesh, K, u, u_e, log):
    """内能有限性检查 + Q4R 沙漏能分级监控.

    沙漏能占比本身不是精度指标 (过刚/过柔都会 >90%) — 只作可靠性
    提示; 长宽比警告由 _q4r_aspect_ratio_warning 在装配前给出.
    """
    hourglass_energy_elem = None
    hourglass_energy = 0.0
    internal_energy = float(0.5 * u @ K.dot(u))
    if not np.isfinite(internal_energy):
        # 极端但有限的载荷/材料下内能可能溢出 — 不返回"成功"的 inf 结果
        raise RuntimeError(
            f"Internal energy = {internal_energy:.3e} is not finite — "
            "载荷或材料参数超出数值范围, 检查模型设置")
    hourglass_energy_ratio = 0.0
    if hasattr(mesh.element_kernel, "hourglass_energy"):
        hourglass_energy_elem = np.asarray(
            mesh.element_kernel.hourglass_energy(mesh, u_e), dtype=float)
        hourglass_energy = float(np.sum(hourglass_energy_elem))
        if internal_energy > np.finfo(float).tiny:
            hourglass_energy_ratio = hourglass_energy / internal_energy
        log(f"[Solver] Q4R hourglass energy = {hourglass_energy:.6e} "
            f"({hourglass_energy_ratio:.2%} of internal energy)")
        if hourglass_energy_ratio > 0.90:
            # 沙漏能主导 — compact 公式已知失效区 (q4r.py 文档):
            # 结果可能过柔或过刚, 沙漏能占比本身不是精度指标
            warnings.warn(
                f"Q4R hourglass energy ratio = {hourglass_energy_ratio:.0%} "
                "(> 90%) — hourglass modes dominate; result is unreliable. "
                "Verify with Q4I (CPS4I).",
                RuntimeWarning, stacklevel=2)
        elif hourglass_energy_ratio > 0.30:
            # 高沙漏能对 Q4R compact 公式在薄板/少行网格上不可靠:
            # 解可能显著过柔或过刚 (见 q4r.py 模块文档)。沙漏能占比
            # 本身不是精度指标 — 建议改用 Q4I 交叉验证。
            warnings.warn(
                f"Q4R hourglass energy ratio is "
                f"{hourglass_energy_ratio:.1%} (> 30%). "
                "The compact Q4R stabilization is unreliable on thin/"
                "few-row meshes (solution may be too soft or too stiff). "
                "Refine the mesh, improve aspect ratio, or verify with "
                "Q4I (CPS4I).",
                RuntimeWarning, stacklevel=2)
    return internal_energy, hourglass_energy_elem, \
        hourglass_energy, hourglass_energy_ratio


def _condition_report(K, free_dofs, check_condition, log):
    """条件数估计 (Bathe §8.2.6) — 默认关闭; 返回 info 或 None."""
    if not (check_condition and len(free_dofs) > 0):
        return None
    K_aa = K[free_dofs][:, free_dofs].tocsr()
    if K_aa.shape[0] > 20000:
        log(f"[Solver] cond(K_aa): {K_aa.shape[0]} DOF — "
            f"稀疏特征值估计可能较慢, 请耐心等待 ...")
    cond_info = estimate_condition(K_aa)
    if cond_info["status"] == "SINGULAR?":
        # condition_number=None — 格式化 :.2e 会二次崩溃 (曾静默)
        log(f"[Solver] cond(K_aa): [SINGULAR?] 特征值求解失败 — "
            f"刚度矩阵疑似奇异: {cond_info.get('error', '')}")
    elif cond_info["status"] != "SKIP":
        log(f"[Solver] cond(K_aa) = {cond_info['condition_number']:.2e} "
            f"-> ~{cond_info['digits_lost']:.1f} digits lost "
            f"[{cond_info['status']}]")
    return cond_info


def solve(
        mesh, method="elimination", verbose=True, check_condition=False,
        linear_solver="auto"):
    """二维位移有限元求解 (Bathe §4.2 完整流程)

    参数
    ----
    mesh : Mesh
        包含节点、单元、材料、边界条件的网格对象
    method : str
        "elimination" (默认, Bathe Eq 4.42) 或 "penalty" (Bathe §4.2.2 p.190)
    verbose : bool
        是否打印求解进度和残差
    check_condition : bool
        是否估计条件数 (稠密求解 O(n³), 大网格建议关闭)
    linear_solver : str
        ``"auto"``、``"direct"``、``"cg"``、``"cg-block"`` 或
        ``"ilu"``。``cg-block`` 是 Jacobi-PCG 的兼容别名；``ilu``
        使用不完全 LU 预条件的 CG。auto 对十万以上自由 DOF 使用
        Jacobi-PCG，以避免稀疏直接分解的 fill-in 内存峰值。

    返回
    ----
    dict (消去法与乘大数法均返回相同键集; 条件数/平衡数据按参数条件附加):
        u         : (n_dof,) ndarray — 节点位移
        stress    : (n_elem, 3) ndarray — 单元应力 [σ_x, σ_y, τ_xy]
        strain    : (n_elem, 3) ndarray — 单元应变 [ε_x, ε_y, γ_xy]
        vm_stress : (n_elem,) ndarray — von Mises 等效应力
        stress_qp : (n_elem, n_qp, 3) ndarray — 积分点应力
        strain_qp : (n_elem, n_qp, 3) ndarray — 积分点应变
        dA_qp     : (n_elem, n_qp) ndarray — 积分点面积权重
        reactions : (n_fixed,) ndarray — 支反力 (两种方法均计算返回)
        residual  : float — 后向误差相对残差
                   ||r||_∞/(||K||_∞·||u||_∞ + ||F||_∞) (Bathe §8.2.6)
        small_deformation_ok : bool — 小变形假设是否成立
        internal_energy : float — 应变能 ½uᵀKu
        hourglass_energy / hourglass_energy_ratio : float — Q4R 沙漏能
        hourglass_energy_elem : (n_elem,) ndarray — 仅 Q4R 内核存在
        linear_solver : dict — {name, iterations[, preconditioner, rtol]};
                    name ∈ {"none", "direct", "cg"} ("none" = 纯 Dirichlet)
        reaction_dofs : (n_fixed,) ndarray — 支反力对应 DOF
        reaction_vector : (n_dof,) ndarray — 全 DOF 支反力 (零填充)
        external_force_vector : (n_dof,) ndarray — 等效节点载荷 F
        force_balance / moment_balance : ndarray / float — ΣR+ΣF, ΣM
        balance_ok : bool — 全局平衡检查是否通过
                    (reaction_dofs..balance_ok 仅在存在支反力时附加)
        condition_info : dict — estimate_condition 输出 (check_condition 时)
    """
    log = print if verbose else (lambda *a, **k: None)
    if not (hasattr(mesh, "validate_state") and hasattr(mesh, "n_dof")):
        # 非 Mesh (dict/None/标量) 曾冒裸 AttributeError — 类型契约前置
        raise TypeError(
            f"solve: mesh 必须是 fem2d.Mesh 实例, "
            f"got {type(mesh).__name__}: {mesh!r}")
    n_dof = mesh.n_dof

    # ── 0. 状态校验: 构造后字段可被重写, 求解前快速失败而非静默错解 ──
    mesh.validate_state()

    # ── 1. 预处理: 拓扑 + 单元验证 + Jacobian (组装前检查, 快速失败) ──
    mesh.build_connectivity()

    # Jacobian 检查 (先于组装, 避免无效单元导致 element_stiffness 崩溃)
    # 完备性/刚体模态验证移至 --self-test (开发/CI 时显式调用)
    jac = mesh.element_kernel.jacobian_report(mesh)
    if not jac.ok:
        raise RuntimeError(
            f"Mesh contains {len(jac.bad)} invalid "
            f"{mesh.element_kernel.name} element(s): "
            f"{jac.inverted} inverted (negative Jacobian), "
            f"{jac.degenerate} degenerate (zero Jacobian). "
            f"First bad element: #{jac.bad[0]}. "
            f"Fix mesh before solving."
        )

    # ── 2. 约束检查 (组装前, 避免无效 BC 导致无意义求解) ──
    fixed_dofs, prescribed, free_dofs = _partition_dofs(
        mesh, method, n_dof, log)

    # ── 逐连通分量刚体模态检查 (Bathe §4.2.2) ──
    # 任何分量 rank < 3 都意味着存在未约束的刚体模态 → 系统奇异 → 禁止求解
    _check_rigid_body_constraints(mesh)

    # ── Q4R 稳定性预处理告警: 按单元长宽比而非沙漏能占比 ──
    _q4r_aspect_ratio_warning(mesh)

    # ── 3. 组装 (验证全部通过后才进行) ──
    log(f"[Solver] 组装总刚 K ({n_dof}×{n_dof}) ...")
    K = assemble_sparse(mesh)

    log("[Solver] 组装等效载荷 F ...")
    F = assemble_loads(mesh, n_dof)

    # ── 4. 求解 + 解的有限性检查 ──
    (u, reactions, K_mod, F_mod,
     linear_solver_info, _dirichlet_only) = _solve_linear_system(
        K, F, free_dofs, fixed_dofs, prescribed,
        method, linear_solver, n_dof, log)
    _check_solution_finite(u, reactions, method)

    # ── 4. 残差检查 (Bathe §8.2.6: backward error, 一致范数) ──
    residual, _residual_abs = _compute_residual(
        K, F, K_mod, F_mod, u, free_dofs,
        method, _dirichlet_only, log)

    # ── 5. 应力计算 ──
    u_e = u[mesh.element_dofs]
    stress, strain, vm, stress_qp, strain_qp, dA_qp = (
        _compute_element_response(mesh, u_e))

    # ── Q4R 沙漏能量监控 (含内能有限性检查) ──
    internal_energy, hourglass_energy_elem, hourglass_energy, \
        hourglass_energy_ratio = _hourglass_monitor(mesh, K, u, u_e, log)

    # ── 5.5 全局力/力矩平衡检查 (Bathe §4.2.2) ──
    # 残差小只说明线性方程解得准, 不代表载荷方向/厚度/边界正确
    balance_data = _global_balance_check(
        mesh, K, F, u, reactions, fixed_dofs, _dirichlet_only, log)

    # ── 位移量级 + 结果有限性检查 ──
    small_deformation_ok = _small_deformation_check(mesh, u)
    _check_result_finite(stress, strain, vm)

    result = {
        "u": u,
        "stress": stress,
        "strain": strain,
        "vm_stress": vm,
        "stress_qp": stress_qp,
        "strain_qp": strain_qp,
        "dA_qp": dA_qp,
        "reactions": reactions,
        "residual": residual,
        "small_deformation_ok": small_deformation_ok,
        "internal_energy": internal_energy,
        "hourglass_energy": hourglass_energy,
        "hourglass_energy_ratio": hourglass_energy_ratio,
        "linear_solver": linear_solver_info,
    }
    if hourglass_energy_elem is not None:
        result["hourglass_energy_elem"] = hourglass_energy_elem

    # ── 完整支反力 & 平衡数据 (Bathe §4.2.2) ──
    if balance_data is not None:
        R_full, sum_F, sum_M, balance_ok = balance_data
        result["reaction_dofs"] = fixed_dofs.copy()
        result["reaction_vector"] = R_full
        result["external_force_vector"] = F.copy()
        result["force_balance"] = sum_F
        result["moment_balance"] = sum_M
        result["balance_ok"] = balance_ok
        log(f"[Solver] max|reaction| = {np.max(np.abs(reactions)):.6e}")

    # ── 6. 条件数估计 (默认关闭, Bathe §8.2.6) ──
    cond_info = _condition_report(K, free_dofs, check_condition, log)
    if cond_info is not None:
        result["condition_info"] = cond_info

    return result
