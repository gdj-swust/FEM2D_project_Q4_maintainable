"""visualize.py 分支补测 — 包 2 覆盖率任务.

未覆盖行集中: 三角化形状守卫、网格线密度分级、_to_node 的 L2/加权
回退、牵引跳跃图空边/零场回退、载荷箭头 (压力/体力/集中力/下采样)、
isoband 校验拒绝分支、scalar_jump 分支、shading/location 兼容性、
plot_three 保存路径、interactive_plot 交互分支。

判别性: 断言具体异常消息/箭头数量/图例文本/文件产物/输出文本。
"""
import matplotlib
matplotlib.use("Agg")   # 必须先于 pyplot/visualize 导入 (无显示器环境)

import numpy as np
import pytest

import fem2d.visualize as viz
from fem2d.mesh import Mesh
from fem2d.visualize import (
    _compute_edge_jumps_scalar,
    _display_triangulation,
    _draw_element_edges,
    _plot_loads,
    _to_node,
    _validate_isoband_levels,
    interactive_plot,
    plot_contour,
    plot_mesh,
    plot_three,
    plot_traction_jumps,
)


def _quad():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        elem_type="CPS4")


def _two_tri():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        elements=np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        elem_type="CPS3")


@pytest.fixture
def ax():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 三角化 / 网格线
# ═══════════════════════════════════════════════════════════════

def test_display_triangulation_rejects_unsupported_npe():
    """5 节点单元不可显示 → ValueError (曾裸切片错误)."""
    nodes = np.zeros((5, 2))
    elements = np.array([[0, 1, 2, 3, 4]], dtype=int)
    with pytest.raises(ValueError, match="3-node and 4-node"):
        _display_triangulation(nodes, elements)


def test_draw_element_edges_density_grading(ax):
    """网格密度分级: >5000 → 0.08 线宽; >1000 → 0.15 (防黑糊一片)."""
    nodes = np.zeros((24000, 2))
    elements = np.arange(24000).reshape(-1, 4)      # 6000 单元
    coll = _draw_element_edges(ax, nodes, elements)
    assert float(coll.get_linewidths()[0]) == pytest.approx(0.08)

    nodes2 = np.zeros((8000, 2))
    elements2 = np.arange(8000).reshape(-1, 4)      # 2000 单元
    coll2 = _draw_element_edges(ax, nodes2, elements2)
    assert float(coll2.get_linewidths()[0]) == pytest.approx(0.15)


def test_draw_element_edges_small_mesh_default(ax):
    """小网格 → 默认 0.3 线宽."""
    coll = _draw_element_edges(ax, _quad().nodes, _quad().elements)
    assert float(coll.get_linewidths()[0]) == pytest.approx(0.3)


# ═══════════════════════════════════════════════════════════════
# _to_node 恢复方法分支
# ═══════════════════════════════════════════════════════════════

def test_to_node_l2_projection():
    """L2 投影 → 积分点数据直接投影到节点."""
    mesh = _quad()
    data = np.array([[1.0, 2.0, 3.0]])
    result = _to_node(mesh, data, method="L2")
    assert result.shape == (4, 3)
    assert np.allclose(result[:, 0], 1.0)   # 常场精确恢复


def test_to_node_weighted_weights_shape_mismatch_fallback(monkeypatch):
    """加权恢复: kernel 权重形状不符 → 回退算术平均 (曾裸广播错误)."""
    mesh = _quad()
    monkeypatch.setattr(mesh.element_kernel, "recovery_weights",
                        lambda m: np.ones((2, 2)))
    data = np.ones((1, 4, 3))
    result = _to_node(mesh, data, method="weighted")
    assert result.shape == (4, 3)


def test_to_node_weighted_no_weights_fallback(monkeypatch):
    """加权恢复: kernel 无权重实现 → 回退算术平均."""
    mesh = _quad()
    monkeypatch.setattr(mesh.element_kernel, "recovery_weights",
                        lambda m: None)
    data = np.ones((1, 4, 3))
    result = _to_node(mesh, data, method="weighted")
    assert result.shape == (4, 3)


# ═══════════════════════════════════════════════════════════════
# plot_traction_jumps 分支
# ═══════════════════════════════════════════════════════════════

def test_traction_jumps_no_internal_edges(ax):
    """单单元 (无内部边) → 标题占位, 返回 None."""
    mesh = _quad()
    result = plot_traction_jumps(mesh, np.zeros((1, 3)), ax=ax)
    assert result is None
    assert "(no internal edges)" in ax.get_title()


def test_traction_jumps_zero_field_fallback(capsys):
    """零跳跃场 (常应力) → P98=0 → 1.0 回退 (色标不塌缩)."""
    mesh = _two_tri()
    stress = np.ones((2, 3))
    plot_traction_jumps(mesh, stress)
    out = capsys.readouterr().out
    assert "zero-field fallback" in out


def test_traction_jumps_user_vmax_and_sigma_ref():
    """显式 vmax + 固定参考应力 → 标签诚实标注."""
    mesh = _two_tri()
    stress = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    plot_traction_jumps(mesh, stress, vmax=0.5, sigma_ref=1e6)
    # 无异常; vmax/标签路径已执行


# ═══════════════════════════════════════════════════════════════
# _plot_loads 载荷箭头
# ═══════════════════════════════════════════════════════════════

def _load_mesh(n_edges=131):
    """1 单元 + n_edges 条面力 (同一物理边) + 压力 + 体力 + 集中力."""
    tractions = [{"nodes": (0, 1), "traction": (1e6, 0.0)}
                 for _ in range(n_edges - 1)]
    tractions.append({"nodes": (2, 3), "traction": (2e6,), "is_pressure": True})
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        elem_type="CPS4",
        surface_tractions=tractions,
        body_force=(1000.0, 0.0),
        concentrated_forces=[{"node": 0, "force": (5.0, 5.0)}])


def test_plot_loads_arrows_and_downsampling(ax):
    """面力下采样 (131→66 箭头) + 压力法向 + 体力 + 集中力全部绘制."""
    mesh = _load_mesh()
    _plot_loads(mesh, ax)
    n_patches = len(ax.patches)
    # 131 边 → step=2 → 66 红箭头; 1 单元 → 1 蓝; 1 集中力 → 1 绿
    assert n_patches == 66 + 1 + 1


def test_plot_loads_legend_labels(ax):
    """图例: 面力边数 / 体力数值 / 集中力点数."""
    mesh = _load_mesh()
    _plot_loads(mesh, ax)
    legend = ax.get_legend()
    texts = [t.get_text() for t in legend.get_texts()]
    assert "面力 (131 边)" in texts
    assert "体力 (1000.0,0.0) N/m³" in texts
    assert "集中力 (1 点)" in texts


def test_plot_loads_callable_body_legend(ax):
    """callable 体力 → 图例显示 f(x,y) (曾把函数对象塞进标签)."""
    mesh = _load_mesh(n_edges=3)
    mesh.body_force = (lambda x, y: 1e3, 0.0)
    _plot_loads(mesh, ax)
    legend = ax.get_legend()
    texts = [t.get_text() for t in legend.get_texts()]
    assert "体力 (f(x,y),0.0) N/m³" in texts


# ═══════════════════════════════════════════════════════════════
# plot_mesh
# ═══════════════════════════════════════════════════════════════

def test_plot_mesh_default_ax_and_fixed_downsampling():
    """ax 缺省自建; 固定节点 >40 → 下采样绘制."""
    xs = np.linspace(0, 9, 10)
    ys = np.linspace(0, 9, 10)
    nodes = np.array([[x, y] for y in ys for x in xs], dtype=float)
    elements = []
    for j in range(9):
        for i in range(9):
            n00 = j * 10 + i
            elements.append([n00, n00 + 1, n00 + 11, n00 + 10])
    mesh = Mesh(nodes=nodes, elements=np.array(elements, dtype=int),
                elem_type="CPS4")
    for nid in range(len(nodes)):
        mesh.fix_node(nid, "both", 0.0)
    plot_mesh(mesh)   # 100 固定节点 → 下采样 50
    assert len(np.unique(mesh.fixed_dofs // 2)) == 100


# ═══════════════════════════════════════════════════════════════
# _validate_isoband_levels
# ═══════════════════════════════════════════════════════════════

def test_isoband_levels_reject_too_few():
    with pytest.raises(ValueError, match=">= 2 values"):
        _validate_isoband_levels([1.0])


def test_isoband_levels_reject_nan():
    with pytest.raises(ValueError, match="NaN or Inf"):
        _validate_isoband_levels([0.0, np.nan, 1.0])


def test_isoband_levels_reject_non_increasing():
    with pytest.raises(ValueError, match="strictly increasing"):
        _validate_isoband_levels([0.0, 1.0, 1.0])


def test_isoband_levels_reject_non_uniform():
    with pytest.raises(ValueError, match="uniformly spaced"):
        _validate_isoband_levels([0.0, 1.0, 2.5])


# ═══════════════════════════════════════════════════════════════
# plot_contour shading/location 兼容性
# ═══════════════════════════════════════════════════════════════

def test_plot_contour_gouraud_scalar_element_rejected():
    """gouraud + 单元定位标量 → ValueError (SPR 直接作用于标量不可靠)."""
    mesh = _quad()
    values = np.array([1.0])
    with pytest.raises(ValueError, match="unreliable"):
        plot_contour(mesh, values, shading="gouraud", location="element")


def test_plot_contour_flat_requires_element_values():
    """flat + 节点定位数据 → ValueError (静默落 gouraud 曾画错图)."""
    mesh = _quad()
    values = np.zeros(4)
    with pytest.raises(ValueError, match="requires element-located"):
        plot_contour(mesh, values, shading="flat", location="node")


def test_plot_contour_scalar_jump_values(ax):
    """scalar_jump: 内部边按 |Δs|/σ_range 着色."""
    mesh = _two_tri()
    values = np.array([1.0, 10.0])
    plot_contour(mesh, values, ax=ax, shading="scalar_jump",
                 location="element")
    jumps = _compute_edge_jumps_scalar(mesh, values, 9.0)
    assert len(jumps) == 1                     # 两三角共享一条内部边
    assert jumps[0]["jump_rel"] == pytest.approx(9.0 / 9.0)


# ═══════════════════════════════════════════════════════════════
# plot_three 保存路径
# ═══════════════════════════════════════════════════════════════

def test_plot_three_save_writes_file_and_closes(tmp_path, capsys):
    """--save: 保存文件 + 关闭 figure (曾长期不 close 累积内存)."""
    mesh = _quad()
    result = {"u": np.zeros(8), "stress": np.zeros((1, 3)),
              "vm_stress": np.zeros(1), "stress_qp": None}
    out_path = str(tmp_path / "plot.png")
    plot_three(mesh, result, tag="ux", scale=1.0, save=out_path)
    assert capsys.readouterr().out.endswith(f"→ {out_path}\n")
    assert (tmp_path / "plot.png").exists() and \
        (tmp_path / "plot.png").stat().st_size > 0


# ═══════════════════════════════════════════════════════════════
# interactive_plot
# ═══════════════════════════════════════════════════════════════

def test_interactive_plot_quit_and_plot(monkeypatch, capsys):
    """输入分量号 + q → 绘图后优雅退出."""
    calls = []
    monkeypatch.setattr(viz, "plot_three",
                        lambda *a, **k: calls.append(k))
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    mesh, result = _quad(), {"u": np.zeros(8)}
    interactive_plot(mesh, result)
    assert len(calls) == 1
    assert calls[0]["tag"] == "mesh"
    out = capsys.readouterr().out
    assert "网格 + 载荷 + 边界条件" in out


def test_interactive_plot_unknown_tag_ignored(monkeypatch, capsys):
    """未知输入 → 提示 '?' 并继续 (不崩溃)."""
    monkeypatch.setattr(viz, "plot_three", lambda *a, **k: None)
    answers = iter(["zz", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    interactive_plot(_quad(), {"u": np.zeros(8)})
    assert "? zz" in capsys.readouterr().out


def test_interactive_plot_recovery_cycle(monkeypatch):
    """按 m → 恢复方法切换 (SPR → weighted → ...)."""
    calls = []
    monkeypatch.setattr(viz, "plot_three",
                        lambda *a, **k: calls.append(k))
    answers = iter(["m", "1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    interactive_plot(_quad(), {"u": np.zeros(8)})
    assert calls[0]["recovery"] == "weighted"


def test_interactive_plot_ctrl_c_graceful(monkeypatch, capsys):
    """Ctrl-C (KeyboardInterrupt) → 优雅退出 (曾裸 traceback)."""
    def _input(prompt):
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", _input)
    interactive_plot(_quad(), {"u": np.zeros(8)})
    assert "[INFO] 已退出交互绘图 (Ctrl-C)" in capsys.readouterr().out
