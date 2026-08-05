"""E3 解析解验证 — Z2 效应指数 θ (Z2 估计器能量误差收敛速率).

Z2 估计器 (fem2d.error_est.estimate, Bathe §4.3.6 / Zienkiewicz-Zhu
1987): 线性单元 (CST/Q4) 的应变能范数误差 ||e|| ~ O(h), 即估计量 η
在对数坐标的斜率 θ = d(ln η)/d(ln h) ≈ 1。

加密序列: Timoshenko 抛物线剪流悬臂梁 (fem2d.convergence 同款 —
光滑解、无点载荷奇异性, 收敛速率干净), CST/Q4 各 5 层 (nx=4..64,
ny=2..32, 每层双向 2× 加密)。拟合: 跳过最粗层 (非渐近区), 对
ln η ~ ln h 线性回归 (与 convergence.py 的全局拟合同模式)。

可复现性: 结构化网格 + elimination 直接求解, 全流程确定性 (无随机
数); 拟合过程即上文, 数据可在 docs/analytic_verification.md 复现。

判别性 (回滚必红): 区间断言 θ ∈ [0.8, 1.2]。⚠️ 纪律: 若实测偏离
区间, 如实报告 + 记录数据, 禁止调公式/阈值凑区间。
"""
import numpy as np
import pytest

from fem2d import Mesh, solve
from fem2d.convergence import _gen_cantilever_mesh, _parabolic_shear_traction
from fem2d.error_est import estimate

L, H, T = 5.0, 1.0, 0.1
E_MOD, NU = 210e9, 0.3
P_SHEAR = 10000.0
THETA_MIN, THETA_MAX = 0.8, 1.2


def _eta_sequence(elem_type):
    """抛物线剪流悬臂梁 5 层加密的 (h, η) 序列."""
    hs, etas = [], []
    for level in range(5):
        nx, ny = 4 * 2**level, 2 * 2**level
        nodes, elements = _gen_cantilever_mesh(L, H, nx, ny,
                                               elem_type=elem_type)
        m = Mesh(nodes=nodes, elements=elements, E=E_MOD, nu=NU,
                 thickness=T, plane_type="stress", elem_type=elem_type)
        for n in m.nodes_on_edge("x", "min", tol=1e-6):
            m.fix_node(int(n), "both", 0.0)
        right = sorted(m.nodes_on_edge("x", "max", tol=1e-6),
                       key=lambda n: m.nodes[int(n), 1])

        def shear(x, y):
            return _parabolic_shear_traction(y, H, T, P_SHEAR)

        for a_, b_ in zip(right, right[1:]):
            m.add_traction(int(a_), int(b_), 0.0, shear)
        result = solve(m, method="elimination", verbose=False)
        etas.append(estimate(m, result, method="SPR", verbose=False)["eta"])
        hs.append(L / nx)
    return np.array(hs), np.array(etas)


def _z2_effect_index(elem_type):
    """ln η ~ ln h 线性回归斜率 (跳过最粗非渐近层)."""
    h, eta = _eta_sequence(elem_type)
    # tiny 地板只防 log(0) — η 恒正, 不会截平真实速率
    slope = np.polyfit(np.log(h[1:]),
                       np.log(np.maximum(eta[1:], np.finfo(float).tiny)), 1)[0]
    return float(slope)


@pytest.mark.parametrize("elem_type", ["CPS3", "CPS4"])
def test_z2_effect_index_in_range(elem_type):
    """Z2 效应指数 θ ∈ [0.8, 1.2] (判别: 区间断言, 回滚必红)."""
    theta = _z2_effect_index(elem_type)
    assert THETA_MIN <= theta <= THETA_MAX, \
        f"{elem_type} Z2 效应指数 θ={theta:.3f} 越出 [{THETA_MIN}, {THETA_MAX}]"
