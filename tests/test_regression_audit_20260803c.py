"""2026-08-03 第四轮外部审查回归测试 — 全装配路径加权一致性.

覆盖: 标量 stiffness() 四单元族微尺度有限 / Q4R 批量路径微尺度有限 /
assemble_lil_reference + assemble_expand 微尺度有限 / 正常尺度
批量 vs 标量数学等价。

背景 (第四轮外部审查, 2026-08-03): 第三轮只修了 Q4/Q4I/CST 的
生产批量路径; 标量 element_stiffness 全四族、Q4R 批量路径仍用旧
乘法顺序 (先 BᵀDB ~ E/L² 再乘 t·detJ), L~1e-150 时溢出 Inf→NaN。
"""
import numpy as np
import pytest


def _micro_mesh(scale, elem_type):
    from fem2d import Mesh
    if elem_type == "CPS3":
        nodes = np.array([[0.0, 0.0], [scale, 0.0], [0.0, scale]])
        elements = np.array([[0, 1, 2]])
    else:
        nodes = np.array(
            [[0.0, 0.0], [scale, 0.0], [scale, scale], [0.0, scale]])
        elements = np.array([[0, 1, 2, 3]])
    m = Mesh(nodes=nodes, elements=elements, E=1e9, nu=0.3,
             thickness=1e-9, elem_type=elem_type)
    m.build_connectivity()
    return m


# ═══════════════════════════════════════════════════════════════
# 标量路径 — 曾全四族旧乘法顺序溢出 (第四轮外部审查复现)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("elem_type", ["CPS3", "CPS4", "CPS4R", "CPS4I"])
def test_scalar_stiffness_extreme_scale_all_families(elem_type):
    """标量 element_stiffness 在 1e-150 几何必须有限 —
    曾 Ke += t·detJ·(BᵀDB): BᵀDB ~ E/L² = 1e309 溢出 Inf,
    再乘 t·detJ ~ 1e-309 得 NaN (参考路径 lil/expand 连带崩溃)."""
    m = _micro_mesh(1e-150, elem_type)
    Ke = m.element_kernel.stiffness(m, 0)
    assert np.all(np.isfinite(Ke)), \
        f"{elem_type} 标量刚度 @1e-150 非有限 (max={np.abs(Ke).max():.3e})"


def test_q4r_batch_path_extreme_scale_no_overflow():
    """Q4R 批量装配在 1e-150 必须有限 —
    曾 einsum 先算 BᵀDB 再乘 t·det (第三轮修 Q4/Q4I 时遗漏 Q4R)."""
    m = _micro_mesh(1e-150, "CPS4R")
    Kb = m.element_kernel.stiffness_batch(m)[0]
    assert np.all(np.isfinite(Kb)), "Q4R 批量刚度 @1e-150 非有限"


def test_reference_assembly_paths_extreme_scale():
    """assemble_lil_reference / assemble_expand 在 1e-150 必须有限 —
    两参考路径都走标量 stiffness(), 曾随标量路径溢出 (审查复现)."""
    from fem2d import assembly
    m = _micro_mesh(1e-150, "CPS4")
    for name, fn in (
            ("lil_reference", assembly.assemble_lil_reference),
            ("expand", assembly.assemble_expand),
            ("sparse", assembly.assemble_sparse_vectorized)):
        K = fn(m)
        assert np.all(np.isfinite(K.data)), \
            f"{name} @1e-150 刚度含 NaN/Inf"


# ═══════════════════════════════════════════════════════════════
# 数学等价守护 — 加权改动不得改变正常尺度结果
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("elem_type", ["CPS3", "CPS4", "CPS4R", "CPS4I"])
def test_scalar_batch_consistency_normal_scale(elem_type):
    """正常尺度下批量路径与标量路径必须一致 (相对差 < 1e-10) —
    加权顺序是数学等价变换, 任何实现分叉都会在这里暴露."""
    m = _micro_mesh(1.0, elem_type)
    K_batch = m.element_kernel.stiffness_batch(m)[0]
    K_scalar = m.element_kernel.stiffness(m, 0)
    rel = np.abs(K_batch - K_scalar).max() / max(
        np.abs(K_scalar).max(), 1e-300)
    assert rel < 1e-10, \
        f"{elem_type} 批量 vs 标量 相对差 {rel:.2e} (应机器精度)"
