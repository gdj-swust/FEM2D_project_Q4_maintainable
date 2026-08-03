"""深度验证: 材料/单元/应力恢复/载荷组合."""
import numpy as np
import pytest

# ═══════════════════════════════════════════════════════════════
# 1. 材料参数校验
# ═══════════════════════════════════════════════════════════════

def test_negative_E_rejected_by_D_matrix():
    """负弹性模量必须被拒绝 (2026-08 起在 solve 入口校验, 早于 D_matrix)."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=-210e9, nu=0.3, thickness=0.01)
    for i in range(3): mesh.fix_node(i, 'both', 0.0)
    with pytest.raises(ValueError, match='must be finite and > 0'):
        solve(mesh, verbose=False)

def test_nu_09_rejected_by_D_matrix():
    """nu=0.9 必须被拒绝 (2026-08 起在 solve 入口校验, 早于 D_matrix)."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.9, thickness=0.01)
    for i in range(3): mesh.fix_node(i, 'both', 0.0)
    with pytest.raises(ValueError, match='must be in'):
        solve(mesh, verbose=False)

def test_invalid_plane_type_mesh_accepts_but_D_matrix_rejects():
    """Mesh 接受任意 plane_type 字符串, D_matrix 校验."""
    from fem2d import Mesh
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.]], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    # Mesh 构造不校验 plane_type (延迟到 D_matrix)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01, plane_type='bogus')
    from fem2d import solve
    for i in range(3): mesh.fix_node(i, 'both', 0.0)
    with pytest.raises(ValueError):
        solve(mesh, verbose=False)


# ═══════════════════════════════════════════════════════════════
# 2. 应变能/应力恢复一致性
# ═══════════════════════════════════════════════════════════════

def test_stress_recovery_methods_agree():
    """三种应力恢复方法对均匀应力场必须精确恢复.

    曾只断言 s_simple≈s_weighted (rtol=0.05): 两个恢复函数同时返回全零
    也通过 — 空断言. 直接传均匀单元应力场, 恢复值
    必须等于该值 (机器精度).
    """
    from fem2d import Mesh
    from fem2d.stress import nodal_L2_projection, nodal_simple, nodal_weighted
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    mesh.build_connectivity()
    uniform = np.tile(np.array([1e6, 2e6, 3e6]), (mesh.n_elements, 1))
    for name, rec in (("simple", nodal_simple(mesh, uniform)),
                      ("weighted", nodal_weighted(mesh, uniform)),
                      ("L2", nodal_L2_projection(mesh, uniform))):
        assert np.allclose(rec, uniform[0], rtol=1e-12, atol=0.0), \
            f"{name} 对均匀场未精确恢复: {rec[0]} vs {uniform[0]}"

def test_plane_stress_vs_strain():
    """平面应力和平面应变对纯拉伸应不同."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    # 平面应力
    m_stress = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01, plane_type='stress')
    m_stress.fix_node(0, 'both', 0.0); m_stress.fix_node(2, 'both', 0.0)
    m_stress.fix_node(1, 'x', 0.001); m_stress.fix_node(3, 'x', 0.001)
    r_stress = solve(m_stress, verbose=False)
    # 平面应变
    m_strain = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01, plane_type='strain')
    m_strain.fix_node(0, 'both', 0.0); m_strain.fix_node(2, 'both', 0.0)
    m_strain.fix_node(1, 'x', 0.001); m_strain.fix_node(3, 'x', 0.001)
    r_strain = solve(m_strain, verbose=False)
    # 平面应变 σ_xx 应更大 (ε_z=0 → σ_xx = E/(1-ν²)·ε_xx)
    assert r_strain['stress'][0, 0] > r_stress['stress'][0, 0]


# ═══════════════════════════════════════════════════════════════
# 3. 多载荷叠加
# ═══════════════════════════════════════════════════════════════

def test_multiple_load_types():
    """体力+面力+集中力同时作用 — 反力必须平衡总外力.

    曾只断言 isfinite(u): 全部载荷静默丢弃 (u=0 有限) 也通过 (审计
    2026-08-03). 反力 = -外力. 面力 (1e6,0)×1m×0.01 = +10000 N (x),
    集中力 (0,-1000), 体力 78000×2×0.01 = 1560 N (y 向下).
    """
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [2., 0.], [0., 1.], [2., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.body_force = (0., -78000.)  # 重力
    mesh.add_traction(1, 3, 1e6, 0.)  # 右端面力
    mesh.add_force(1, 0., -1000.)  # 集中力
    result = solve(mesh, verbose=False)
    rx = result['reaction_vector'].reshape(-1, 2)[:, 0].sum()
    ry = result['reaction_vector'].reshape(-1, 2)[:, 1].sum()
    assert abs(rx + 10000.0) < 1.0, f"ΣRx={rx} ≠ -10000"
    assert abs(ry - (1000.0 + 1560.0)) < 1.0, f"ΣRy={ry} ≠ 2560"

def test_pressure_and_traction_same_edge():
    """同一边上压力和面力叠加 — 反力 = 两者合力.

    曾只断言 isfinite(u): 载荷静默丢弃也通过.
    压力 1e6 法向 (压向域内, t=-p·n = -1e6 x) ×1m×0.01 = -10000 N,
    面力 (5e5,0) ×1m×0.01 = +5000 N → ΣFx = -5000 → ΣRx = +5000.
    """
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.add_pressure(1, 3, 1e6)
    mesh.add_traction(1, 3, 5e5, 0.)
    result = solve(mesh, verbose=False)
    rx = result['reaction_vector'].reshape(-1, 2)[:, 0].sum()
    assert abs(rx - 5000.0) < 1.0, f"ΣRx={rx} ≠ +5000"


# ═══════════════════════════════════════════════════════════════
# 4. mesh/element API
# ═══════════════════════════════════════════════════════════════

def test_fix_node_separate_axes():
    """分步约束 x 和 y 应等价于同时约束."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    # 同时约束
    m1 = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    m1.fix_node(0, 'both', 0.0); m1.fix_node(2, 'both', 0.0)
    m1.add_force(3, 1000., 0.)
    r1 = solve(m1, verbose=False)
    # 分步约束
    m2 = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    m2.fix_node(0, 'x', 0.0); m2.fix_node(0, 'y', 0.0)
    m2.fix_node(2, 'x', 0.0); m2.fix_node(2, 'y', 0.0)
    m2.add_force(3, 1000., 0.)
    r2 = solve(m2, verbose=False)
    assert np.allclose(r1['u'], r2['u'])

def test_add_force_accumulates():
    """对同一节点多次 add_force 应累加."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    m1 = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: m1.fix_node(n, 'both', 0.0)
    m1.add_force(3, 500., 0.); m1.add_force(3, 500., 0.)
    r1 = solve(m1, verbose=False)
    m2 = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: m2.fix_node(n, 'both', 0.0)
    m2.add_force(3, 1000., 0.)
    r2 = solve(m2, verbose=False)
    assert np.allclose(r1['u'], r2['u'])

def test_check_condition_runs():
    """条件数估计应可运行."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.add_force(3, 1000., 0.)
    result = solve(mesh, verbose=False, check_condition=True)
    assert 'condition_info' in result


# ═══════════════════════════════════════════════════════════════
# 5. Isoband 边缘情况
# ═══════════════════════════════════════════════════════════════

def test_isoband_constant_stress_single_band(capsys):
    """真正常应力场 (所有单元值相同) 应能生成 isoband — 自动 levels
    走常应力保护分支 (曾 np.linspace(x,x,n) 使 BoundaryNorm 崩溃)."""
    from fem2d import Mesh
    import matplotlib; matplotlib.use('Agg')
    from fem2d.visualize import plot_contour
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    values = np.full(mesh.n_elements, 5e6)   # 真正常应力场 (曾用非恒定解)
    fig, ax = matplotlib.pyplot.subplots()
    try:
        plot_contour(mesh, values, shading='isoband', levels=None, ax=ax)
        out = capsys.readouterr().out
        assert "isoband warning" not in out, f"常应力场不应有超界警告: {out!r}"
    finally:
        matplotlib.pyplot.close('all')

def test_isoband_user_levels_out_of_range_warns(capsys):
    """用户指定的 levels 范围不完全覆盖应力值时应打印 warning 而非静默
    clip — 曾只测"不崩溃", 超界静默无提示."""
    from fem2d import Mesh, solve
    import matplotlib; matplotlib.use('Agg')
    from fem2d.visualize import plot_contour
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]: mesh.fix_node(n, 'both', 0.0)
    mesh.fix_node(1, 'x', 0.001); mesh.fix_node(3, 'x', 0.001)
    result = solve(mesh, verbose=False)
    vm = result['vm_stress']
    # 故意取小范围: 所有单元值都在 levels 之上 → 必须打印超界警告
    levels = np.array([0.0, float(vm.max()) * 0.1])
    fig, ax = matplotlib.pyplot.subplots()
    try:
        plot_contour(mesh, vm, shading='isoband', levels=levels, ax=ax)
        out = capsys.readouterr().out
        assert "isoband warning" in out, f"超界 levels 未警告: {out!r}"
    finally:
        matplotlib.pyplot.close('all')