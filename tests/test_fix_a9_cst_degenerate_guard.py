"""A9 (P3, R-α, 冻结区): CST 批量刚度路径退化单元显式拒绝.

审查现象 (2026-08-05): 零面积 (共线) 单元 — 批量路径静默返回
NaN/Inf (仅 RuntimeWarning), 单元素路径抛 ValueError "Element area
≈ 0 — degenerate triangle", 行为分叉。

修复 (冻结区特殊程序): stiffness_batch 入口补显式面积守卫, 与
单元素路径 _element_stiffness 同判据 (|A| ≤ 64·eps·scl², scl =
最大边长)。**只改入口守卫, 不触碰 B 矩阵/刚度公式** — 有限输入
输出必须逐位不变 (本文件金标准 = 修复前基线录制; 另有
test_solve_refactor_lock + 漂移门三重验证)。

判别性: 回滚本提交 → 批量退化单元静默 NaN 回归。
"""
import hashlib

import numpy as np
import pytest

from fem2d.mesh import Mesh
from fem2d.solver import solve

# ── 金标准 (修复前基线录制): 有限输入必须逐位不变 ──

GOLD_K0 = [[1153846153.8461542, 0.0, -1153846153.8461542, 346153846.15384626,
            0.0, -346153846.15384626],
           [0.0, 403846153.8461539, 403846153.8461539, -403846153.8461539,
            -403846153.8461539, 0.0],
           [-1153846153.8461542, 403846153.8461539, 1557692307.6923082,
            -750000000.0000002, -403846153.8461539, 346153846.15384626],
           [346153846.15384626, -403846153.8461539, -750000000.0000002,
            1557692307.6923082, 403846153.8461539, -1153846153.8461542],
           [0.0, -403846153.8461539, -403846153.8461539, 403846153.8461539,
            403846153.8461539, 0.0],
           [-346153846.15384626, 0.0, 346153846.15384626, -1153846153.8461542,
            0.0, 1153846153.8461542]]
GOLD_K1 = [[403846153.8461539, 0.0, 0.0, -403846153.8461539, -403846153.8461539,
            403846153.8461539],
           [0.0, 1153846153.8461542, -346153846.15384626, 0.0,
            346153846.15384626, -1153846153.8461542],
           [0.0, -346153846.15384626, 1153846153.8461542, 0.0,
            -1153846153.8461542, 346153846.15384626],
           [-403846153.8461539, 0.0, 0.0, 403846153.8461539, 403846153.8461539,
            -403846153.8461539],
           [-403846153.8461539, 346153846.15384626, -1153846153.8461542,
            403846153.8461539, 1557692307.6923082, -750000000.0000002],
           [403846153.8461539, -1153846153.8461542, 346153846.15384626,
            -403846153.8461539, -750000000.0000002, 1557692307.6923082]]

GOLD_U = [0.0, 0.0, 8.078118064802482e-06, -3.8082556591211684e-05,
          0.0, 0.0, 7.61651131824234e-05, -3.0004438526409205e-05]
GOLD_REACTIONS = [-3.637978807091713e-12, -38482.02396804262,
                  -99999.99999999997, 38482.023968042595]
GOLD_STRESS = [[2423435.4194407444, 2423435.4194407435, 2423435.419440745],
               [17576564.580559246, 5272969.374167773, -2423435.419440743]]

NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])


def _healthy_mesh():
    m = Mesh(NODES, ELEMS, E=2.1e11, nu=0.3, thickness=0.01)
    m.build_connectivity()
    return m


# ── 判别性红侧: 退化单元批量路径必须拒绝 ──

def test_degenerate_batch_raises_valueerror():
    # 回滚 → 批量路径静默返回 NaN (仅 RuntimeWarning)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 3], [1, 2, 3], [0, 1, 2]])  # elem 2 共线零面积
    m = Mesh(nodes, elems, E=2.1e11, nu=0.3, thickness=0.01)
    m.build_connectivity()
    with pytest.raises(ValueError, match="degenerate triangle"):
        m.element_kernel.stiffness_batch(m)


def test_degenerate_solve_rejected_upstream():
    # solve 全路径在 jacobian_report 即拦截 (RuntimeError, 早于刚度
    # 装配) — A9 守卫针对的是直接批量调用路径 (曾静默 NaN 的唯一入口)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 3], [1, 2, 3], [0, 1, 2]])
    m = Mesh(nodes, elems, E=2.1e11, nu=0.3, thickness=0.01)
    with pytest.raises(RuntimeError, match="degenerate"):
        solve(m, verbose=False)


def test_degenerate_slice_rejected():
    # element_slice 子集路径同样守卫
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3]])
    m = Mesh(nodes, elems, E=2.1e11, nu=0.3, thickness=0.01)
    m.build_connectivity()
    with pytest.raises(ValueError, match="degenerate triangle"):
        m.element_kernel.stiffness_batch(m, slice(0, 2))  # 含退化单元 0


# ── 冻结区: 有限输入逐位不变 (金标准 + 锁) ──

def test_batch_stiffness_bit_exact_gold():
    K = _healthy_mesh().element_kernel.stiffness_batch(_healthy_mesh())
    assert K.shape == (2, 6, 6)
    np.testing.assert_array_equal(K[0], np.array(GOLD_K0))
    np.testing.assert_array_equal(K[1], np.array(GOLD_K1))


def test_solve_bit_exact_gold():
    # 求解路径含稀疏消去 (BLAS) — 跨平台 (Win MKL vs Linux OpenBLAS) 最后
    # 1-2 位浮点不同, 逐位断言不可移植 (CI 实测 8.07811806480248e-06 vs
    # 8.078118064802482e-06). 位级锁由 test_solve_refactor_lock 同平台承担;
    # 此处断言"接近金标准" (相对 1e-13) 锁 A9 修复不改变求解结果.
    m = Mesh(NODES, ELEMS, E=2.1e11, nu=0.3, thickness=0.01)
    for n in (0, 2):
        m.fix_node(n, "both", 0.0)
    m.add_force(3, 1e5, 0.0)
    r = solve(m, verbose=False)
    np.testing.assert_allclose(r["u"], GOLD_U, rtol=1e-13, atol=0.0)
    # reactions[0] 是固定 DOF 支反力, 理论精确 0 — 平台残差 ±3.6e-12 且符号
    # 随 BLAS 消去顺序翻转 (CI 实测 +3.6e-12 vs 本地 -3.6e-12), 相对误差在
    # 零附近数学上失效. atol=1e-9 只放行残差噪声, 真实支反力 (~4e4) 仍由 rtol 锁.
    np.testing.assert_allclose(r["reactions"], GOLD_REACTIONS, rtol=1e-13, atol=1e-9)
    np.testing.assert_allclose(r["stress"], GOLD_STRESS, rtol=1e-13, atol=0.0)


def test_grid_batch_stiffness_lock():
    # 6×4 网格批量刚度 (48 单元) — 修复前后哈希逐位锁定
    nx, ny = 6, 4
    pts = np.array([[i / 5, j / 3] for j in range(ny + 1)
                    for i in range(nx + 1)])
    tri = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            b = a + 1
            c = a + (nx + 1)
            d = c + 1
            tri += [[a, b, c], [b, d, c]]
    m = Mesh(pts, np.array(tri), E=2.1e11, nu=0.3, thickness=0.01)
    m.build_connectivity()
    K = m.element_kernel.stiffness_batch(m)
    assert K.shape == (48, 6, 6)
    digest = hashlib.sha256(K.tobytes()).hexdigest()
    assert digest == "297ea0455fd3e7e730a3084e480ee9b9f8007b358a64d2e6759c0588be5dae9b"
