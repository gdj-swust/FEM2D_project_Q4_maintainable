"""plot_three / interactive_plot display tests"""
import numpy as np
from fem2d import Mesh, solve
from fem2d.visualize import plot_three
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _make_result():
    nodes = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=float)
    elems = np.array([[0,1,2],[1,3,2]], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    mesh.fix_node(0,'both'); mesh.fix_node(1,'both')
    mesh.add_force(3, 1e6, 0)
    return mesh, solve(mesh)


def _current_axes():
    """plot_three 返回 None — 从 matplotlib 当前 figure 取子图."""
    figs = [f for f in plt.get_fignums()]
    assert figs, "未创建任何 figure"
    return plt.figure(figs[-1]).axes


def test_plot_three_mesh_draws_content():
    """tag='mesh' 必须画出网格 — 曾零断言, plot_three 改成 pass 也通过
    (审计 2026-08-03)."""
    mesh, r = _make_result()
    plot_three(mesh, r, tag='mesh')
    axes = _current_axes()
    plt.close('all')
    assert len(axes) == 3, f"期望 3 张子图, 得到 {len(axes)}"
    assert any(
        len(ax.collections) + len(ax.lines) > 0 for ax in axes), \
        "mesh 模式未画出任何内容"


def test_plot_three_loads_draws_content():
    """tag='loads' 必须画出载荷/网格 — 曾零断言 (审计 2026-08-03)."""
    mesh, r = _make_result()
    plot_three(mesh, r, tag='loads')
    axes = _current_axes()
    plt.close('all')
    assert any(
        len(ax.collections) + len(ax.lines) > 0 for ax in axes), \
        "loads 模式未画出任何内容"


def test_plot_three_displacement_draws_content():
    """tag='ux'/'uy'/'umag' 必须画出变形图 — 曾零断言 (审计 2026-08-03)."""
    mesh, r = _make_result()
    for tag in ('ux', 'uy', 'umag'):
        plot_three(mesh, r, tag=tag)
        axes = _current_axes()
        assert any(
            len(ax.collections) + len(ax.lines) > 0 for ax in axes), \
            f"tag={tag} 未画出任何内容"
    plt.close('all')


def test_isoband_tag_scoping(capsys):
    """Fixed isoband levels should only apply when tag matches isoband_tag.

    用远小于应力量级的 levels 作探针: tag 匹配 → levels 应用 → 超界
    警告; tag 不匹配 → 回退 auto levels → 无警告. 曾只测"不崩溃"
    (审计 2026-08-03).
    """
    mesh, r = _make_result()
    levels = np.array([0.0, 1.0])   # 远小于应力量级 → 必然超界
    plot_three(mesh, r, tag='vm', isoband_levels=levels, isoband_tag='vm')
    out1 = capsys.readouterr().out
    assert "isoband warning" in out1, \
        f"tag 匹配时固定 levels 未应用: {out1!r}"
    plot_three(mesh, r, tag='sx', isoband_levels=levels, isoband_tag='vm')
    out2 = capsys.readouterr().out
    assert "isoband warning" not in out2, \
        f"tag 不匹配时不应应用固定 levels: {out2!r}"
    plt.close('all')
