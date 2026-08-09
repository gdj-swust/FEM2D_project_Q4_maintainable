"""位移边界条件施加 — Bathe §4.2.2

提供两种方法:
  1. 消去法 (elimination) — Bathe Eq 4.42-4.45, 精确, 默认推荐
  2. 乘大数法 (penalty)    — Bathe §4.2.2 p.190, 简单快速

Bathe §4.2.2 Eq 4.42-4.45 (消去法):
  将 DOF 分为自由 (a) 和约束 (b):
    K_aa · U_a = R_a - K_ab · U_b   → 求解 U_a
    R_b = K_ba · U_a + K_bb · U_b   → 计算支反力

Bathe §4.2.2 p.190 (乘大数法 / 罚函数法):
  K_ii += k_penalty;  F_i += k_penalty · d_i  (加法式, 避免刚度量纲平方)
  其中 k_penalty = max(|K_ii|) × 10⁸  (确保约束残差 < 10⁻⁸ 量级)
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg, spilu, splu

from .checks import require_dof_index_array, require_finite_scalar

# ═══════════════════════════════════════════════════════════════
# 1. 消去法 — Bathe §4.2.2 Eq 4.42-4.45 (推荐)
# ═══════════════════════════════════════════════════════════════

def _normalize_solver_key(linear_solver):
    """solver 名称归一 + 校验 — 在任何分支前检查.

    纯 Dirichlet 分支在名称检查前返回会让 linear_solver="bogus"
    静默成功, 因此校验必须先于空自由集分支。
    """
    solver_key = str(linear_solver).strip().lower()
    if solver_key == "cg-block":
        solver_key = "cg"
    if solver_key not in {"direct", "cg", "ilu"}:
        raise ValueError(
            f"Unknown linear_solver '{linear_solver}'; "
            "expected direct, cg, cg-block or ilu.")
    return solver_key


def _solve_direct(K_aa, rhs):
    """SuperLU: 稳健的默认路径, 适合中小规模模型."""
    lu = splu(K_aa.tocsc())
    return lu.solve(rhs), {"name": "direct", "iterations": 1}


def _cg_preconditioner(K_aa, solver_key):
    """PCG 预条件器: Jacobi (SPD, 默认) 或显式 ILU.

    K_aa 对线弹性充分约束问题为 SPD; Jacobi 是低内存默认预条件器,
    显式 ``ilu`` 使用 SuperLU 的不完全 LU 因子。
    """
    diagonal = K_aa.diagonal()
    if (
            not np.all(np.isfinite(diagonal))
            or np.any(diagonal <= 0.0)):
        raise RuntimeError(
            "CG requires a positive finite stiffness diagonal. "
            "Check element Jacobians and boundary constraints.")
    if solver_key == "ilu":
        # ⚠️ 数值方法风险: CG 理论上要求预条件器
        # 对称正定, 而 SuperLU 的 ILU 不保证 SPD — 病态网格/畸形单元下
        # 可能异常停滞 (rho 崩溃, 已有下游检查)。工程上 ILU-PCG 是常见
        # 近似, 本项目限定: ILU 仅作显式选择, 默认 auto 走 Jacobi (SPD);
        # 若 CG 收敛失败 (info != 0) 下游会给出明确错误建议 direct。
        try:
            # drop_tol=1e-4 的 ILU 因子对中等网格可能非正定 (预条件
            # CG 的 rho 崩溃为 0, 与 maxiter 无关地永不收敛) —
            # 实测 306 DOF 悬臂即复现, 1e-6 收敛。保持更精确的因子。
            ilu = spilu(
                K_aa.tocsc(),
                drop_tol=1.0e-6,
                fill_factor=10.0,
                permc_spec="MMD_AT_PLUS_A",
                diag_pivot_thresh=0.0,
            )
        except RuntimeError as error:
            raise RuntimeError(
                "ILU preconditioner factorization failed. "
                "Try linear_solver='direct' or 'cg', improve mesh "
                "quality, or check boundary constraints."
            ) from error
        preconditioner = LinearOperator(
            K_aa.shape, matvec=ilu.solve, dtype=K_aa.dtype)
        preconditioner_name = "ilu"
    else:
        preconditioner = diags(
            1.0 / diagonal, offsets=0, shape=K_aa.shape, format="csr")
        preconditioner_name = "jacobi"
    return preconditioner, preconditioner_name


def _solve_cg(K_aa, rhs, preconditioner, preconditioner_name,
              cg_rtol, cg_maxiter):
    """PCG 求解 + 收敛失败报错 (maxiter 自适应于 DOF 数)."""
    if cg_maxiter is None:
        cg_maxiter = min(
            20000,
            max(1000, int(20.0 * np.sqrt(K_aa.shape[0]))),
        )
    iterations = [0]

    def count_iteration(_):
        iterations[0] += 1

    U_a, info = cg(
        K_aa,
        rhs,
        M=preconditioner,
        rtol=float(cg_rtol),
        atol=0.0,
        maxiter=int(cg_maxiter),
        callback=count_iteration,
    )
    if info != 0:
        detail = (
            f"did not converge within {info} iterations"
            if info > 0 else f"failed with status {info}")
        raise RuntimeError(
            f"{preconditioner_name.upper()}-preconditioned CG {detail}. "
            "Try linear_solver='direct', improve mesh quality, or check "
            "boundary constraints.")
    return U_a, {
        "name": "cg",
        "iterations": int(iterations[0]),
        "rtol": float(cg_rtol),
        "preconditioner": preconditioner_name,
    }


def _pure_dirichlet_solution(K, F, n_dof, fixed_dofs,
                             prescribed_vals, return_info):
    """纯 Dirichlet 问题: K_aa 是 0×0, splu 不支持空矩阵.

    solve() 在上游拦截该情形, 但 apply_elimination 是公开 API —
    直接调用时空自由度集必须可正常工作 (Bathe Eq 4.45)。
    """
    u = np.zeros(n_dof)
    u[fixed_dofs] = prescribed_vals
    reactions = (K.dot(u) - F)[fixed_dofs]
    if return_info:
        return u, reactions, {"name": "direct", "iterations": 0}
    return u, reactions


def _reduced_system(K, F, free_dofs, fixed_dofs, prescribed_vals):
    """(1) 提取子矩阵 K_aa + (2) 修正右端项 R_a' = R_a − K_ab·U_b
    (Bathe Eq 4.43)."""
    K_aa = K[free_dofs][:, free_dofs].tocsr()
    rhs = F[free_dofs].copy()
    if len(fixed_dofs) and np.any(prescribed_vals != 0.0):
        K_ab = K[free_dofs][:, fixed_dofs].tocsr()
        rhs -= K_ab.dot(prescribed_vals)
    return K_aa, rhs


def _assemble_solution(K, F, n_dof, free_dofs, fixed_dofs, U_a,
                       prescribed_vals, return_info, solver_info):
    """(4) 组装全位移向量 + (5) 支反力 R_b = (K·U − F)_b (Bathe Eq 4.45).

    直接使用完整残差避免额外构造 K_ba/K_bb 两个稀疏子矩阵。
    """
    u = np.zeros(n_dof)
    u[free_dofs] = U_a
    u[fixed_dofs] = prescribed_vals
    reactions = (K.dot(u) - F)[fixed_dofs]
    if return_info:
        return u, reactions, solver_info
    return u, reactions


def apply_elimination(
        K, F, free_dofs, fixed_dofs, prescribed_vals,
        linear_solver="direct", cg_rtol=1e-10, cg_maxiter=None,
        return_info=False, *, _system_validated=False):
    # cg_rtol 必须保持 1e-10: 实测 1e-8 时 CG 残差污染固定 DOF 反力
    # (~4.3e-3 N @ 29 万单元), 与平衡检查 tol_rel=1e-8·|F| 同量级,
    # 合法大型模型被 ΣF 检查误杀 (性能优化试验后回退)
    """用消去法施加位移约束 (Bathe §4.2.2 Eq 4.42-4.45)

    将 DOF 划分为自由 (a) 和约束 (b) 两组，仅求解自由 DOF，
    然后计算支反力。不引入任何数值近似，是"精确"的约束施加法。

    参数
    ----
    K : csr_matrix (n_dof, n_dof)
        全局刚度矩阵 (稀疏 CSR)
    F : (n_dof,) ndarray
        等效节点力向量 (包含体力和面力)
    free_dofs : (n_free,) ndarray of int
        自由 DOF 索引 (已排序)
    fixed_dofs : (n_fixed,) ndarray of int
        约束 DOF 索引
    prescribed_vals : (n_fixed,) ndarray
        指定位移值 U_b

    返回
    ----
    u : (n_dof,) ndarray — 全位移向量
    reactions : (n_fixed,) ndarray — 支反力 (Bathe Eq 4.45)
    """
    free_dofs, fixed_dofs, prescribed_vals = _validate_elimination_inputs(
        K, F, free_dofs, fixed_dofs, prescribed_vals,
        system_validated=_system_validated)  # K 形状 (含 0-D 前置) 在此校验
    n_dof = len(free_dofs) + len(fixed_dofs)  # 分区已校验覆盖 [0,n_dof)
    solver_key = _normalize_solver_key(linear_solver)

    if len(free_dofs) == 0:
        return _pure_dirichlet_solution(
            K, F, n_dof, fixed_dofs,
            prescribed_vals, return_info)

    K_aa, rhs = _reduced_system(
        K, F, free_dofs, fixed_dofs, prescribed_vals)

    if solver_key == "direct":
        U_a, solver_info = _solve_direct(K_aa, rhs)
    elif solver_key in {"cg", "ilu"}:
        preconditioner, preconditioner_name = _cg_preconditioner(
            K_aa, solver_key)
        U_a, solver_info = _solve_cg(
            K_aa, rhs, preconditioner, preconditioner_name,
            cg_rtol, cg_maxiter)
    else:  # _normalize_solver_key 已拦截非法名称 — 仅防御
        raise ValueError(
            f"Unknown linear_solver '{linear_solver}'; "
            "expected direct, cg or ilu.")

    return _assemble_solution(
        K, F, n_dof, free_dofs, fixed_dofs, U_a,
        prescribed_vals, return_info, solver_info)


# ═══════════════════════════════════════════════════════════════
# 2. 乘大数法 — Bathe §4.2.2 p.190 (备选)
# ═══════════════════════════════════════════════════════════════

def apply_penalty(K, F, fixed_dofs, prescribed_vals=None, penalty=None,
                  *, _system_validated=False):
    """用乘大数法施加位移约束 (Bathe §4.2.2 p.190)

    加法式: K_ii += k_penalty,  F_i += k_penalty * d_i
      → u_i ≈ d_i  (k_penalty >> max|K_ij|)

    Bathe 推荐: k_penalty = max(|K_ii|) × 10⁸

    代价: 条件数 κ(K) 被抬高约 8 个量级 (k_penalty / min_eigenvalue).
    对良态网格, 消去法 vs 罚函数位移相对差 ~2.8e-10, 约束残差 ~1.4e-15,
    精度完全可用。但在病态网格上, 额外 8 个量级的条件数恶化可能使 splu
    的误差边界被击穿。推荐: 默认使用消去法 (精确, 无此代价);
    罚函数法作为备选, 仅在消去法因自由度为空的纯 Dirichlet 问题不适用时使用。

    参数
    ----
    K : csr_matrix (n_dof, n_dof)
    F : (n_dof,) ndarray
    fixed_dofs : (n_fixed,) ndarray of int
    prescribed_vals : (n_fixed,) ndarray or None
    penalty : float or None
        罚因子; None 则自动 k_penalty = max(|K_ii|)×1e8

    返回
    ----
    K_mod : csr_matrix
    F_mod : (n_dof,) ndarray
    penalty : float — 实际使用的罚刚度值 [N/m]
    """
    if not _system_validated:
        _validate_system_inputs(K, F)
    n_dof = K.shape[0]
    fixed_dofs, prescribed_vals = _reject_duplicate_fixed_dofs(
        fixed_dofs, prescribed_vals, n_dof=n_dof)

    # 自动罚刚度: k_penalty = max(|K_ii|) × 1e8
    diag = np.abs(K.diagonal())
    max_diag = diag.max() if diag.max() > 0 else 1.0
    if penalty is None:
        factor = 1e8
        max_allowed = np.finfo(float).max / factor
        if max_diag > max_allowed:
            raise OverflowError(
                "apply_penalty: 自动罚因子将溢出 "
                f"(max|K_ii|={max_diag:.3e} × {factor} → inf); "
                "请改用 elimination 约束或缩放模型量纲")
        penalty = max_diag * factor
        if not np.isfinite(penalty):
            raise OverflowError(
                "apply_penalty: 自动罚因子非有限, 请改用 elimination 约束")
    else:
        penalty = require_finite_scalar(penalty, "apply_penalty: penalty")
        if penalty < max_diag * 1e4:
            raise ValueError(
                f"apply_penalty: penalty={penalty!r} — must be >= "
                f"1e4 * max|K_ii| = {max_diag*1e4:.3e} "
                f"(max|K_ii| = {max_diag:.3e}; 建议自动值 {max_diag*1e8:.3e})")

    # 加法式: K_ii += k_penalty (量纲一致, 不平方)
    penalty_vals = np.zeros(n_dof)
    for dof in fixed_dofs:
        penalty_vals[dof] = penalty

    K_mod = K + diags(penalty_vals, 0, format='csr')

    # F_mod: 累加罚项 (不覆盖已有外载)
    F_mod = F.copy()
    for k, dof in enumerate(fixed_dofs):
        F_mod[dof] += penalty * prescribed_vals[k]

    return K_mod, F_mod, penalty


def _validate_system_inputs(K, F):
    """K 方阵 + 数据有限 / F 形状与有限性 (elimination/penalty 共用)."""
    from scipy.sparse import issparse
    if not issparse(K):
        K = np.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        # 标量/0-D 数组会冒裸 IndexError (0 维 shape[0] 越界), 非数组
        # 冒裸 AttributeError ('.shape') — 与 estimate_condition 同款
        # 前置, ValueError 带上下文
        raise ValueError(
            f"K must be a square matrix (n×n), got "
            f"{getattr(K, 'shape', type(K).__name__)}")
    k_data = K.data if issparse(K) else np.asarray(K).ravel()
    if k_data.size and not np.all(np.isfinite(k_data)):
        n_bad = int(np.count_nonzero(~np.isfinite(k_data)))
        raise ValueError(
            f"K contains {n_bad} NaN/Inf entries — 刚度矩阵非法 "
            "(element stiffness overflow or corrupted assembly)")
    F = np.asarray(F)
    if F.shape != (K.shape[0],):
        raise ValueError(
            f"F must have shape ({K.shape[0]},), got {F.shape}")
    if not np.all(np.isfinite(F)):
        raise ValueError("F contains NaN/Inf — 载荷向量非法")
    return F


def _validate_elimination_inputs(K, F, free_dofs, fixed_dofs,
                                 prescribed_vals, system_validated=False):
    """apply_elimination 统一输入校验:
    K 方阵/有限 / F 形状与有限性 / DOF 分区 (重复、重叠、覆盖) / 给定位移。
    返回 (free_dofs, fixed_dofs, prescribed_vals) 规范化后的分区。
    """
    if not system_validated:
        F = _validate_system_inputs(K, F)
    n_dof = K.shape[0]
    fixed_dofs, prescribed_vals = _reject_duplicate_fixed_dofs(
        fixed_dofs, prescribed_vals, n_dof)
    free_dofs = _validate_dof_partition(free_dofs, fixed_dofs, n_dof)
    return free_dofs, fixed_dofs, prescribed_vals


def _validate_dof_partition(free_dofs, fixed_dofs, n_dof):
    """自由/约束 DOF 分区校验 — 曾缺省:
    free/fixed 重叠时约束值覆盖自由解, 遗漏 DOF 静默设 0。
    要求: free 合法整数、无重叠、free ∪ fixed 完整覆盖 [0, n_dof)。
    """
    free = require_dof_index_array(
        free_dofs, "free_dofs", n_dof=n_dof, bool_error=ValueError)
    if np.unique(free).size != free.size:
        # 自身重复会延迟到 SuperLU 奇异才报错
        dup = free[np.flatnonzero(
            np.diff(np.sort(free)) == 0)]
        raise ValueError(
            f"free_dofs contain duplicates: {dup[:10].tolist()}")

    overlap = np.intersect1d(free, np.asarray(fixed_dofs))
    if overlap.size:
        raise ValueError(
            f"free_dofs 与 fixed_dofs 重叠: {overlap[:10].tolist()} — "
            "同一 DOF 不能既是自由又是约束")

    covered = np.unique(np.concatenate([free, np.asarray(fixed_dofs)]))
    missing = n_dof - covered.size
    if missing:
        raise ValueError(
            f"DOF 分区未覆盖全部 {n_dof} 个自由度, 遗漏 {missing} 个 "
            f"(如 {np.setdiff1d(np.arange(n_dof), covered)[:5].tolist()}) — "
            "遗漏 DOF 曾静默设为 0")
    return free


def _reject_duplicate_fixed_dofs(fixed_dofs, prescribed_vals, n_dof=None):
    """罚函数参数统一验证 + 重复约束防御.

    公开 API 曾对长度不等静默丢约束 (zip 截断)、负 DOF 静默约束最后
    一个 — 主 solve() 路径已由 validate_state 防护, 此处兜底直接调用。
    返回去重后的 (fixed_dofs, prescribed_vals)。
    """
    fixed = require_dof_index_array(
        fixed_dofs, "fixed_dofs", n_dof=n_dof, bool_error=ValueError)
    if prescribed_vals is None:
        prescribed = np.zeros(len(fixed))
    else:
        prescribed = np.asarray(prescribed_vals)
        if len(prescribed) != len(fixed):
            raise ValueError(
                f"fixed_dofs ({len(fixed)}) 与 prescribed_vals "
                f"({len(prescribed)}) 长度必须相等 — 曾静默丢约束")
        if not np.all(np.isfinite(prescribed)):
            raise ValueError("prescribed_vals contain NaN/Inf")
    seen = {}
    for dof, value in zip(fixed, prescribed):
        if dof in seen and seen[dof] != value:
            raise ValueError(
                f"DOF {dof} 被重复约束为不同值 {seen[dof]} 与 {value} — "
                "同一 DOF 只能约束一次")
        seen[dof] = value
    return list(seen.keys()), list(seen.values())
