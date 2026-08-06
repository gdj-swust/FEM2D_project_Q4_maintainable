"""P-α 判别性测试 — solve() 主流程全量 K·u 只算一次 (共享复用).

缺陷 (外部审查 2026-08-06): 一次 solve() 中全量稀疏矩阵向量积
K·u (shape=(n_dof,)) 被重复计算 3 次: apply_elimination 支反力
(固有计算, 不动)、_compute_residual 残差、_hourglass_monitor 内能。
修复: solve() 主流程计算一次 Ku = K.dot(u), 残差/内能/penalty 与
纯 Dirichlet 分支支反力对同一数组切片/点积。

判别性 (红侧 → 绿侧):
- elimination: 全量 K.dot 3 次 (红) → ≤2 次 (固有支反力 1 + 主流程 1)
- penalty / 纯 Dirichlet: 2 次 (红) → ≤1 次 (主流程 1)

计数只统计 solve() 顶层 K (assemble_sparse 返回值) 上结果 shape
== (n_dof,) 的 dot 调用 — K_mod (乘大数法残差, 修改阵独立计算) 与
子矩阵 (K_aa/K_ab) 上的 dot 天然排除。

数值冻结: 共享 Ku 与未包装 K 独立直算逐位相等 (零容差 ==),
残差与 _compute_residual 的 Ku=None 回退路径 (即重构前原公式)
逐位相等 — 复用只改变 K·u 的取得位置, 不改变任何数值。
"""
import numpy as np

import fem2d.solver as solver_mod
from fem2d import Mesh
from fem2d.assembly import assemble_sparse

# 2×2 单位方板 CPS4 规则网格 (CCW 节点序, Jacobian 恒正) — 4 单元
_NODES = np.array([
    [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
    [0.0, 1.0], [1.0, 1.0], [2.0, 1.0],
    [0.0, 2.0], [1.0, 2.0], [2.0, 2.0],
])
_ELEMS = np.array([
    [0, 1, 4, 3], [1, 2, 5, 4],
    [3, 4, 7, 6], [4, 5, 8, 7],
], dtype=int)


def _quad_mesh():
    """2×2 单位方板 CPS4 网格 (无 gmsh 依赖)."""
    return Mesh(nodes=_NODES.copy(), elements=_ELEMS.copy(),
                E=210e9, nu=0.3, thickness=1.0,
                plane_type="stress", elem_type="CPS4")


def _solve_counting(mesh, method, linear_solver, monkeypatch):
    """solve() 一次, 统计顶层 K 上结果 shape==(n_dof,) 的全量 dot 调用.

    solve() 内部经 assemble_sparse 构造 K — 先按真实路径组装一次,
    给实例的 dot 挂计数包装 (scipy csr 允许实例属性赋值), 再让
    solve() 的 assemble_sparse 返回该包装 K。包装只计数不改结果。
    """
    K_real = assemble_sparse(mesh)
    n_dof = mesh.n_dof
    calls = []
    original_dot = K_real.dot

    def counting_dot(other, *args, **kwargs):
        out = original_dot(other, *args, **kwargs)
        calls.append(tuple(out.shape))
        return out

    K_real.dot = counting_dot
    monkeypatch.setattr(solver_mod, "assemble_sparse", lambda m: K_real)
    result = solver_mod.solve(mesh, method=method, verbose=False,
                              linear_solver=linear_solver)
    full = [shape for shape in calls if shape == (n_dof,)]
    return result, full


def _constrained_mesh():
    """固定底边 2 角节点 + 右上角竖向集中力 (刚体模态约束充分)."""
    mesh = _quad_mesh()
    mesh.fix_node(0)
    mesh.fix_node(1)
    mesh.add_force(8, 0.0, 1e6)
    return mesh


def test_elimination_full_ku_matvec_count(monkeypatch):
    """elimination: 全量 K·u ≤ 2 次 (红侧 3 → 绿侧 2).

    3 次中的支反力计算 (apply_elimination._assemble_solution) 是消除法
    固有计算 (K_aa 已提取, 全量 K·u 无法避免), 不在复用范围; 复用
    消除的是主流程中残差 + 内能两处重复 matvec。
    """
    mesh = _constrained_mesh()
    result, full = _solve_counting(
        mesh, "elimination", "direct", monkeypatch)
    assert np.all(np.isfinite(result["u"])), "elimination 解非有限"
    assert len(full) <= 2, (
        f"elimination 全量 K.dot 调用 {len(full)} 次 (>2): "
        f"修复前 3 次 (支反力+残差+内能), 修复后 ≤2 次 "
        f"(固有支反力 1 + 主流程共享 1) — 残差/内能必须复用同一 Ku")


def test_penalty_full_ku_matvec_count(monkeypatch):
    """penalty: 全量 K·u ≤ 1 次 (红侧 2 → 绿侧 1).

    K_mod.dot(u) (乘大数法残差) 是修改阵上的独立计算, 不在复用范围 —
    计数只统计顶层 K, 天然排除。
    """
    mesh = _constrained_mesh()
    result, full = _solve_counting(
        mesh, "penalty", "direct", monkeypatch)
    assert np.all(np.isfinite(result["u"])), "penalty 解非有限"
    assert len(full) <= 1, (
        f"penalty 全量 K.dot 调用 {len(full)} 次 (>1): "
        f"修复前 2 次 (支反力+内能), 修复后 1 次 (主流程共享) — "
        f"支反力/内能必须复用同一 Ku")


def test_pure_dirichlet_full_ku_matvec_count(monkeypatch):
    """纯 Dirichlet (全约束): 全量 K·u ≤ 1 次 (红侧 2 → 绿侧 1)."""
    mesh = _quad_mesh()
    for n in range(mesh.n_nodes):
        mesh.fix_node(n)
    result, full = _solve_counting(
        mesh, "elimination", "direct", monkeypatch)
    assert np.all(np.isfinite(result["reactions"])), "支反力非有限"
    assert len(full) <= 1, (
        f"纯 Dirichlet 全量 K.dot 调用 {len(full)} 次 (>1): "
        f"修复前 2 次 (支反力+内能), 修复后 1 次 (主流程共享) — "
        f"支反力/内能必须复用同一 Ku")


def _assert_bit_identical(mesh, method):
    """共享 Ku 与未包装 K 独立直算逐位一致 (零容差 ==)."""
    K_ref = assemble_sparse(mesh)
    fixed = np.asarray(mesh.fixed_dofs, dtype=int)
    res = solver_mod.solve(mesh, method=method, verbose=False,
                           linear_solver="direct")
    u = res["u"]
    Ku = K_ref.dot(u)
    F = res["external_force_vector"]
    assert res["internal_energy"] == float(0.5 * u @ Ku), (
        f"{method}: 内能不是共享 Ku 的 ½uᵀKu (逐位必须相等)")
    assert np.array_equal(res["reactions"], (Ku - F)[fixed]), (
        f"{method}: 支反力不是 (Ku−F)[fixed] (逐位必须相等)")


def test_ku_reuse_values_bit_identical():
    """三种分支: 共享 Ku 与独立直算逐位相等 (数值冻结, 零容差).

    复用只改变 K·u 的取得位置 — 内能/支反力对同一数组切片/点积,
    与独立重算的 K.dot(u) 结果必须逐位相等 (==)。
    """
    _assert_bit_identical(_constrained_mesh(), "elimination")
    _assert_bit_identical(_constrained_mesh(), "penalty")
    mesh = _quad_mesh()
    for n in range(mesh.n_nodes):
        mesh.fix_node(n)
    _assert_bit_identical(mesh, "elimination")


def test_elimination_residual_bit_identical_to_fallback():
    """elimination 残差与 Ku=None 回退路径 (重构前原公式) 逐位相等.

    _compute_residual 直接调用 (Ku=None) 走自身 K.dot(u) — 即重构前
    的计算路径; solve() 传 Ku 后必须与之逐位一致 (共享数组与直算
    同值)。顺带锁定残差公式 (Bathe §8.2.6) 未动。
    """
    mesh = _constrained_mesh()
    K_ref = assemble_sparse(mesh)
    fixed = np.asarray(mesh.fixed_dofs, dtype=int)
    free = np.setdiff1d(np.arange(mesh.n_dof), fixed)
    res = solver_mod.solve(mesh, method="elimination", verbose=False,
                           linear_solver="direct")
    residual_ref, _ = solver_mod._compute_residual(
        K_ref, res["external_force_vector"], None, None, res["u"], free,
        "elimination", False, lambda *a, **k: None, Ku=None)
    assert res["residual"] == residual_ref, (
        f"elimination 残差与回退路径不一致: "
        f"{res['residual']!r} vs {residual_ref!r} (逐位必须相等)")
