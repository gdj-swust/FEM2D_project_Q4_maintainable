"""E1 解析解验证 — 悬臂梁端部集中力, CST/Q4 对 Euler-Bernoulli 梁理论.

对标公式 (平面应力, 标准梁理论):
    w_max = P·L³/(3EI),   I = b·h³/12        (端部挠度)
    σx(x,y) = M(x)·y/I = P(L−x)·y/I          (弯曲正应力)

模型: L/H = 10 细长梁 (L=5, H=0.5, t=0.1), 左端全固支, 端部中点
集中力 P 向下 (P=1000 N)。平面应力, E=210 GPa, ν=0.3。

比较量:
  - 端部挠度: 端部中点节点 uy 对 w_max 的相对误差
  - 根部应力: 根列上半单元 σx 平均对梁公式在同质心处平均的相对误差。
    排除角点邻接单元 — 固支角点处应力奇异性随加密发散 (质心随加密
    逼近角点), 与 convergence.py 采样避开固定端角点的注释同理。

判别性 (回滚必红): 误差随加密严格单调下降 + 最细网格相对误差 < 5%。
CST/Q4 同门槛 5% — 实测最细层: 挠度 0.17%/0.38%, 根部应力 3.08%/
2.89%, 余量 ≥ 2 个百分点; 门槛与实测依据见 docs/analytic_verification.md。
"""
import numpy as np
import pytest

from fem2d import Mesh, solve
from fem2d.convergence import _find_tip_node, _gen_cantilever_mesh

L, H, T = 5.0, 0.5, 0.1
E_MOD, NU = 210e9, 0.3
P_FORCE = 1000.0
MOMENT_I = T * H**3 / 12.0
W_EB = P_FORCE * L**3 / (3.0 * E_MOD * MOMENT_I)
LEVELS = (16, 32, 64, 128)
ERR_THRESHOLD = 0.05  # 5% 相对误差门槛 (CST/Q4 同门槛, 依据见模块 docstring)


def _cantilever_solve(elem_type, nx):
    """构建 L×H 悬臂梁 (左端固支 + 端部中点集中力) 并求解.

    只读调用 fem2d API — 网格生成复用 fem2d.convergence._gen_cantilever_mesh,
    不复制实现 (重复实现是静默分歧的温床).
    """
    nodes, elements = _gen_cantilever_mesh(L, H, nx, nx // 2,
                                           elem_type=elem_type)
    m = Mesh(nodes=nodes, elements=elements, E=E_MOD, nu=NU, thickness=T,
             plane_type="stress", elem_type=elem_type)
    for n in m.nodes_on_edge("x", "min", tol=1e-6):
        m.fix_node(int(n), "both", 0.0)
    tip = _find_tip_node(m)
    m.add_force(tip, 0.0, -P_FORCE)  # P>0 向下 (与 convergence 约定一致)
    return m, solve(m, method="elimination", verbose=False)


def _tip_deflection_error(mesh, result):
    tip = _find_tip_node(mesh)
    uy = -result["u"].reshape(-1, 2)[tip, 1]  # 向下为正
    return abs(uy - W_EB) / (abs(W_EB) + np.finfo(float).tiny)


def _root_stress_error(mesh, result, nx):
    """根列上半单元平均 σx 对梁公式的误差 (排除角点邻接单元).

    角点邻接单元的质心随加密逼近角点, 应力被奇异性主导而发散;
    上半列其余单元的奇异项平均后随 h 衰减, 得到平滑收敛的度量.
    """
    hx = L / nx
    corner = np.array([n for n in range(mesh.n_nodes)
                       if abs(mesh.nodes[n, 0]) < 1e-9
                       and abs(abs(mesh.nodes[n, 1]) - H / 2) < 1e-9])
    corner_touch = np.zeros(mesh.n_elements, dtype=bool)
    for eid, conn in enumerate(mesh.elements):
        if np.any(np.isin(conn, corner)):
            corner_touch[eid] = True
    sel = np.where((mesh.centroids[:, 0] < hx)
                   & (mesh.centroids[:, 1] > 0.0) & ~corner_touch)[0]
    if len(sel) == 0:
        raise AssertionError("根列上半无可用单元")
    fe = np.mean(result["stress"][sel, 0])
    th = np.mean(P_FORCE * (L - mesh.centroids[sel, 0])
                 * mesh.centroids[sel, 1] / MOMENT_I)
    return abs(fe - th) / (abs(th) + np.finfo(float).tiny)


@pytest.mark.parametrize("elem_type", ["CPS3", "CPS4"])
def test_cantilever_tip_deflection_converges(elem_type):
    """端部挠度: 误差随加密严格单调下降, 最细层 < 5% (判别: 回滚必红)."""
    errors = [_tip_deflection_error(*_cantilever_solve(elem_type, nx))
              for nx in LEVELS]
    assert all(b < a for a, b in zip(errors, errors[1:])), \
        f"{elem_type} 挠度误差未单调下降: {errors}"
    assert errors[-1] < ERR_THRESHOLD, \
        f"{elem_type} 最细层挠度误差 {errors[-1]:.3%} ≥ 5%"


@pytest.mark.parametrize("elem_type", ["CPS3", "CPS4"])
def test_cantilever_root_stress_converges(elem_type):
    """根部应力: 误差随加密严格单调下降, 最细层 < 5% (判别: 回滚必红)."""
    errors = []
    for nx in LEVELS:
        mesh, result = _cantilever_solve(elem_type, nx)
        errors.append(_root_stress_error(mesh, result, nx))
    assert all(b < a for a, b in zip(errors, errors[1:])), \
        f"{elem_type} 根部应力误差未单调下降: {errors}"
    assert errors[-1] < ERR_THRESHOLD, \
        f"{elem_type} 最细层根部应力误差 {errors[-1]:.3%} ≥ 5%"
