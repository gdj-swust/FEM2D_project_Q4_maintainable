"""visualize.py 分支补测 — 包 2 覆盖率任务.

未覆盖行集中: 三角化形状守卫、网格线密度分级、_to_node 的 L2/加权
回退、牵引跳跃图空边/零场回退、载荷箭头 (压力/体力/集中力/下采样)、
isoband 校验拒绝分支、scalar_jump 分支、shading/location 兼容性、
plot_three 保存路径、interactive_plot 交互分支。

判别性: 断言具体异常消息/箭头数量/图例文本/文件产物/输出文本。
"""
import matplotlib
matplotlib.use("Agg")   # 必须先于 pyplot/visualize 导入 (无显示器环境)

import time

import matplotlib.pyplot as plt
import numpy as np
import pytest

import fem2d.visualize as viz
from fem2d.mesh import Mesh
from fem2d.stress import stress_probe
from fem2d.visualize import (
    _DEFAULT_READOUT,
    _READOUT_MODES,
    _compute_edge_jumps_scalar,
    _display_triangulation,
    _draw_element_edges,
    _pad_near_constant_range,
    _p98_vmax,
    _parse_coord,
    _plot_loads,
    _readout_line,
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


def test_to_node_weighted_weights_shape_mismatch_fallback(monkeypatch,
                                                          capsys):
    """加权恢复: kernel 权重形状不符 → WARN + 回退算术平均.

    静默替换恢复方法违反"静默错误比崩溃危险" — 必须响亮 (曾无任何提示).
    """
    mesh = _quad()
    monkeypatch.setattr(mesh.element_kernel, "recovery_weights",
                        lambda m: np.ones((2, 2)))
    data = np.ones((1, 4, 3))
    result = _to_node(mesh, data, method="weighted")
    assert result.shape == (4, 3)
    out = capsys.readouterr().out
    assert "[WARN]" in out and "回退算术平均" in out


def test_to_node_weighted_no_weights_fallback(monkeypatch):
    """加权恢复: kernel 无权重实现 → 回退算术平均."""
    mesh = _quad()
    monkeypatch.setattr(mesh.element_kernel, "recovery_weights",
                        lambda m: None)
    data = np.ones((1, 4, 3))
    result = _to_node(mesh, data, method="weighted")
    assert result.shape == (4, 3)


def test_to_node_spr_memoized(monkeypatch):
    """同一 (mesh, 应力数组) 的 SPR 恢复只算一次 — 切分量曾每次 3 遍重算.

    判别性: spr_recovery 调用计数 — 备忘命中后重算即红; 不同数组
    对象 (不同 id) 须重算 (备忘键含 id, 值强引用防 GC 错配).
    """
    mesh = _two_tri()
    s = np.ones((2, 3))
    viz._TO_NODE_CACHE.clear()
    calls = {"n": 0}
    real = viz.spr_recovery

    def _count(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(viz, "spr_recovery", _count)
    a1 = _to_node(mesh, s, None, "SPR")
    a2 = _to_node(mesh, s, None, "SPR")
    assert calls["n"] == 1
    assert a1 is a2  # 备忘命中 — 同一数组, 无重算
    _to_node(mesh, s * 2.0, None, "SPR")  # 新数组对象 → 必须重算
    assert calls["n"] == 2


def test_to_node_spr_stale_cache_recomputed(monkeypatch):
    """就地改应力数组 → 内容指纹变 → 备忘失效重算.

    判别性: 同一数组对象 (同 id) 就地改值后 spr_recovery 必须再跑 —
    曾只比对象 id, 就地修改静默吃旧缓存 (应力值变了云图不变).
    """
    mesh = _two_tri()
    s = np.ones((2, 3))
    viz._TO_NODE_CACHE.clear()
    calls = {"n": 0}
    real = viz.spr_recovery

    def _count(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(viz, "spr_recovery", _count)
    _to_node(mesh, s, None, "SPR")
    s[0, 0] = 5.0                      # 同一对象就地改 → id 不变
    _to_node(mesh, s, None, "SPR")
    assert calls["n"] == 2


def test_to_node_spr_cache_evicts_at_capacity(monkeypatch):
    """容量 8 满 → 插入前全清 (整体清空防 GC 后 id 复用错配)."""
    mesh = _two_tri()
    viz._TO_NODE_CACHE.clear()
    calls = {"n": 0}
    real = viz.spr_recovery

    def _count(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(viz, "spr_recovery", _count)
    for k in range(9):                 # 9 个不同数组对象 → 9 个不同 id 键
        _to_node(mesh, np.ones((2, 3)) * (k + 1), None, "SPR")
    assert len(viz._TO_NODE_CACHE) == 1    # 第 9 个触发全清后只存自己
    assert calls["n"] == 9


# ═══════════════════════════════════════════════════════════════
# plot_traction_jumps 分支
# ═══════════════════════════════════════════════════════════════

def test_traction_jumps_no_internal_edges(ax):
    """单单元 (无内部边) → 标题占位, 返回 None."""
    mesh = _quad()
    result = plot_traction_jumps(mesh, np.zeros((1, 3)), ax=ax)
    assert result is None
    assert "(no internal edges)" in ax.get_title()


def test_traction_jumps_memoized(monkeypatch):
    """同一 (mesh, 应力) 的跳跃计算 + 段构建只做一次 — 切分量重算浪费.

    判别性: compute_traction_jumps 调用计数 — 备忘命中后重算即红.
    """
    mesh = _two_tri()
    s = np.ones((2, 3))
    viz._TRACTION_JUMP_CACHE.clear()
    calls = {"n": 0}
    real = viz.compute_traction_jumps

    def _count(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(viz, "compute_traction_jumps", _count)
    fig, ax = plt.subplots()
    plot_traction_jumps(mesh, s, ax=ax)
    plot_traction_jumps(mesh, s, ax=ax)
    assert calls["n"] == 1
    plt.close(fig)


def test_traction_jumps_stale_cache_recomputed(monkeypatch):
    """就地改应力数组 → 指纹变 → 跳跃重算 (同 _TO_NODE_CACHE 口径)."""
    mesh = _two_tri()
    s = np.ones((2, 3))
    viz._TRACTION_JUMP_CACHE.clear()
    calls = {"n": 0}
    real = viz.compute_traction_jumps

    def _count(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(viz, "compute_traction_jumps", _count)
    fig, ax = plt.subplots()
    plot_traction_jumps(mesh, s, ax=ax)
    s[0, 0] = 3.0                      # 同一对象就地改 → id 不变
    plot_traction_jumps(mesh, s, ax=ax)
    assert calls["n"] == 2
    plt.close(fig)


def test_traction_jumps_sigma_ref_split_and_eviction(monkeypatch):
    """sigma_ref 在备忘键内 — 换参考应力必须重算 (跨网格收敛对比时传
    不同 sigma_ref, 旧图段不得复用); 容量 8 满 → 全清."""
    mesh = _two_tri()
    s = np.ones((2, 3))
    viz._TRACTION_JUMP_CACHE.clear()
    calls = {"n": 0}
    real = viz.compute_traction_jumps

    def _count(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(viz, "compute_traction_jumps", _count)
    fig, ax = plt.subplots()
    plot_traction_jumps(mesh, s, ax=ax)
    plot_traction_jumps(mesh, s, ax=ax)                  # 命中
    plot_traction_jumps(mesh, s, ax=ax, sigma_ref=1e6)   # 键不同 → 重算
    assert calls["n"] == 2
    for k in range(7):                                   # 再凑 7 个 → 第 9 键全清
        plot_traction_jumps(mesh, s * (k + 2), ax=ax)
    assert len(viz._TRACTION_JUMP_CACHE) == 1
    plt.close(fig)


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


def test_plot_contour_node_location_when_counts_equal():
    """location='node' 显式 + n_nodes==n_elements 网格 → 不按长度猜位置.

    K4 网格 (4 节点 4 三角) 上节点标量曾因 len==n_elements 被误判为
    单元数据 → 抛 "unreliable" ValueError. 显式 location 必须短路推断.
    """
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    elems = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                     dtype=int)
    mesh = Mesh(nodes=nodes, elements=elems, elem_type="CPS3")
    values = np.array([1.0, 2.0, 3.0, 4.0])   # 节点标量, 4 个
    plot_contour(mesh, values, shading="gouraud", location="node")
    plt.close("all")


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


def test_plot_three_agg_closes_figure():
    """Agg 后端非 save 路径必须关闭 figure (曾每调用静默累积一张图)."""
    mesh = _quad()
    result = {"u": np.zeros(8), "stress": np.zeros((1, 3)),
              "vm_stress": np.zeros(1), "stress_qp": None}
    before = len(plt.get_fignums())
    plot_three(mesh, result, tag="ux", scale=1.0)
    assert len(plt.get_fignums()) == before


def test_plot_contour_colorbar_labels_source():
    """统一色条必须标注数据来源 (element/node) 与峰值.

    CST 磨平教训: 磨平后峰值降低 ~17% — 色条不注明单元/节点值会误导.
    flat → [element, max=...]; gouraud 节点数据 → [node, max=...].
    """
    mesh = _quad()
    fig, ax = plt.subplots()
    plot_contour(mesh, np.array([1.0, 2.0]), "flat-title", ax=ax,
                 shading="flat", location="element")
    plot_contour(mesh, np.array([1.0, 2.0, 3.0, 4.0]), "g-t", ax=ax,
                 shading="gouraud", location="node")
    labels = [a.get_ylabel() for a in fig.axes if a.get_ylabel()]
    assert any("[element, max=" in l and "flat-title" in l for l in labels), \
        f"flat 色条缺来源标注: {labels}"
    assert any("[node, max=" in l and "g-t" in l for l in labels), \
        f"gouraud 色条缺来源标注: {labels}"
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 色标常场保护
# ═══════════════════════════════════════════════════════════════

def test_p98_vmax_zero_field_fallback():
    """零场 P98 → (1.0, 'zero-field fallback'); 阈值是 1e-30 相对下限 —
    微尺度常场同样回退 (绝对阈值会让 1e-40 级场误判为"有场")."""
    assert _p98_vmax(np.zeros(20)) == (1.0, "zero-field fallback")
    assert _p98_vmax(np.full(20, 1e-40))[1] == "zero-field fallback"
    p98, label = _p98_vmax(np.arange(20.0))
    assert p98 == pytest.approx(18.62)
    assert label == "P98"


def test_pad_near_constant_range_pads_only_near_constant():
    """近常场 → ±scl·1e-6 padding (基于场自身尺度, 微尺度场不塌缩);
    正常跨度原样返回."""
    lo, hi = _pad_near_constant_range(1.0, 1.0)
    assert lo == pytest.approx(1.0 - 1e-6)
    assert hi == pytest.approx(1.0 + 1e-6)
    lo2, hi2 = _pad_near_constant_range(1e-10, 1e-10 * (1 + 1e-15))
    assert lo2 == pytest.approx(1e-10 - 1e-16, rel=1e-6)
    assert hi2 == pytest.approx(1e-10 + 1e-16, rel=1e-6)
    assert _pad_near_constant_range(0.0, 5.0) == (0.0, 5.0)


def test_plot_contour_constant_field_isoband_no_crash(ax):
    """常场 isoband 12 带 — e_min≈e_max 先 padding 再 linspace (曾
    全同 level 让填充等值线带坍缩/报错)."""
    mesh = _two_tri()
    plot_contour(mesh, np.full(2, 3.0), "constant", ax=ax,
                 shading="isoband", location="element", n=12)
    # 常场 3.0 → padding scl·1e-6=3e-6 → 12 带带宽 6e-6/12=5e-7 —
    # 标题带宽注记证明 padding 生效 (曾全同 level 带坍缩)
    assert "constant" in ax.get_title()
    assert "bands: 5e-07" in ax.get_title()


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


def test_interactive_plot_closes_figure_before_switch(monkeypatch):
    """切换分量前必须关闭旧图 (曾每按一键累积一张 2×2 图)."""
    calls = []
    monkeypatch.setattr(viz, "plot_three",
                        lambda *a, **k: calls.append("plot"))
    monkeypatch.setattr(viz.plt, "close",
                        lambda *a: calls.append("close"))
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    interactive_plot(_quad(), {"u": np.zeros(8)})
    assert calls[0] == "close" and calls[1] == "plot", \
        f"绘图前未先关旧图: {calls}"


def test_interactive_plot_ctrl_c_graceful(monkeypatch, capsys):
    """Ctrl-C (KeyboardInterrupt) → 优雅退出 (曾裸 traceback)."""
    def _input(prompt):
        raise KeyboardInterrupt
    monkeypatch.setattr("builtins.input", _input)
    interactive_plot(_quad(), {"u": np.zeros(8)})
    assert "[INFO] 已退出交互绘图 (Ctrl-C)" in capsys.readouterr().out


def test_parse_coord_fullwidth_and_ascii():
    """坐标解析: NFKC 归一全角括号/逗号/数字, 非坐标/NaN 拒绝."""
    assert _parse_coord("（0.10,0.15）") == (0.10, 0.15)
    assert _parse_coord("（０.10，０.15）") == (0.10, 0.15)
    assert _parse_coord("0.72, 0.30") == (0.72, 0.30)
    assert _parse_coord("1e-3 2") == (1e-3, 2.0)
    assert _parse_coord("zz") is None
    assert _parse_coord("5") is None
    assert _parse_coord("nan 1") is None


# ═══════════════════════════════════════════════════════════════
# 状态栏实时读值 (_readout_line / _READOUT_MODES)
# ═══════════════════════════════════════════════════════════════

def _probe_result():
    """双三角网格 + 非常量应力 + 零位移 — 读值测试共用."""
    mesh = _two_tri()
    result = {"stress": np.array([[1e6, -2e6, 3e6], [2e6, 1e6, -1e6]]),
              "stress_qp": None, "u": np.zeros(8)}
    return mesh, result


def test_readout_line_stress_matches_probe_rows():
    """读值文本: 各应力分量/口径与 stress_probe 行逐分量一致.

    taumax = (s1-s2)/2 — 与 plot_three 的 principal_stresses 口径一致.
    """
    mesh, result = _probe_result()
    e_row, r_row = stress_probe(mesh, result, 0.25, 0.25)
    for mode, row in (("element", e_row), ("recovered", r_row)):
        for tag, idx in (("sx", 0), ("sy", 1), ("txy", 2),
                         ("s1", 3), ("s2", 4), ("vm", 5)):
            line = _readout_line(mesh, result, 0.25, 0.25, tag, mode)
            assert f"{row[idx]:.4e}" in line, f"{tag}/{mode}: {line}"
        line = _readout_line(mesh, result, 0.25, 0.25, "taumax", mode)
        assert f"{0.5 * (row[3] - row[4]):.4e}" in line


def test_readout_line_mode_suffix_labels():
    """读值文本后缀注明口径 — 数据来源必须可见 (repo 色条标注同风格)."""
    mesh, result = _probe_result()
    line_e = _readout_line(mesh, result, 0.25, 0.25, "sx", "element")
    line_r = _readout_line(mesh, result, 0.25, 0.25, "sx", "recovered")
    assert "单元代表应力" in line_e and "SPR 恢复场插值" in line_r


def test_readout_line_disp_shape_interpolation():
    """位移读值: 线性场 (节点 ux=x) 形函数插值精确恢复 + umag."""
    mesh = _two_tri()
    u = np.zeros(8)
    u[0::2] = [0.0, 1.0, 0.0, 1.0]   # ux = x
    result = {"u": u}
    line = _readout_line(mesh, result, 0.25, 0.25, "ux", "element")
    assert "u_x = 2.5000e-01" in line and "形函数插值" in line
    line = _readout_line(mesh, result, 0.25, 0.25, "umag", "element")
    assert "|u| = 2.5000e-01" in line


def test_readout_line_no_value_and_outside():
    """mesh/loads/None → 空串; 网格外 → '（模型外）' (不抛异常)."""
    mesh, result = _probe_result()
    assert _readout_line(mesh, result, 0.25, 0.25, "mesh", "element") == ""
    assert _readout_line(mesh, result, 0.25, 0.25, "loads", "element") == ""
    assert _readout_line(mesh, result, 0.25, 0.25, None, "element") == ""
    assert _readout_line(mesh, result, 2.0, 2.0, "sx", "element") == "（模型外）"
    assert _readout_line(mesh, result, 2.0, 2.0, "ux", "element") == "（模型外）"


def test_readout_registry_contract():
    """口径注册表: 默认 element (≈质心), 每键取数函数返回 (6,) 应力行."""
    assert _DEFAULT_READOUT == "element"
    assert set(_READOUT_MODES) == {"element", "recovered"}
    mesh, result = _probe_result()
    for key, (label, fn) in _READOUT_MODES.items():
        row = fn(mesh, result, 0.25, 0.25)
        assert row.shape == (6,), f"{key}: {label}"


def test_interactive_plot_direct_coord_probe(monkeypatch, capsys):
    """主菜单直接输入坐标 (不按 p) → 探针打印两口径行."""
    monkeypatch.setattr(viz, "plot_three", lambda *a, **k: None)
    answers = iter(["0.25, 0.25", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    mesh = _two_tri()
    result = {"stress": np.ones((2, 3)), "stress_qp": None}
    interactive_plot(mesh, result)
    out = capsys.readouterr().out
    assert "[probe] (0.2500, 0.2500)" in out
    assert "element  :" in out and "recovered:" in out


def test_interactive_plot_click_toggle(monkeypatch, capsys):
    """c 键 → 点击探针开关状态打印并翻转."""
    monkeypatch.setattr(viz, "plot_three", lambda *a, **k: None)
    answers = iter(["c", "c", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    interactive_plot(_quad(), {"u": np.zeros(8)})
    out = capsys.readouterr().out
    assert "点击探针已开启" in out and "点击探针已关闭" in out


def test_interactive_plot_readout_toggle(monkeypatch, capsys):
    """v 键 → 状态栏读值口径循环切换 (element ↔ recovered) 并打印."""
    monkeypatch.setattr(viz, "plot_three", lambda *a, **k: None)
    answers = iter(["v", "v", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    interactive_plot(_quad(), {"u": np.zeros(8)})
    out = capsys.readouterr().out
    assert "状态栏读值口径" in out
    assert "SPR 恢复场插值" in out and "单元代表应力" in out


def test_interactive_plot_plot_branch_no_blank_figure(monkeypatch):
    """plot 分支无图时不 gcf() 凭空建图 (Agg/测试下 plot_three 已 close).

    曾无条件 plt.gcf().mpl_connect — plot_three 无图时静默多建一张空白图.
    """
    monkeypatch.setattr(viz, "plot_three", lambda *a, **k: None)
    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    before = len(plt.get_fignums())
    interactive_plot(_quad(), {"u": np.zeros(8)})
    assert len(plt.get_fignums()) == before


def test_interactive_plot_click_readout_end_to_end(monkeypatch):
    """左键点击 → 右下角文本 = 当前分量 (vm) 单元代表值 (端到端).

    plot_three 桩建真实 Agg figure; plt.close 桩为 no-op 让图跨 q 存活,
    随后合成 button_press MouseEvent 走真实 callbacks.process 管线。
    """
    def _plot(*a, **k):
        fig = plt.figure()
        fig.add_subplot(111)
        return fig
    monkeypatch.setattr(viz, "plot_three", _plot)
    monkeypatch.setattr(viz.plt, "close", lambda *a, **k: None)
    answers = iter(["8", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    mesh = _two_tri()
    result = {"stress": np.ones((2, 3)), "stress_qp": None,
              "u": np.zeros(8)}
    interactive_plot(mesh, result)

    from matplotlib.backend_bases import MouseEvent
    fig = plt.gcf()
    ev = MouseEvent("button_press_event", fig.canvas, 10, 10, button=1)
    ev.inaxes = fig.axes[0]
    ev.xdata, ev.ydata = 0.25, 0.25
    fig.canvas.callbacks.process("button_press_event", ev)

    text = fig.texts[-1].get_text()
    # 常应力 [1,1,1] → vm = sqrt(1+1-1+3·1) = 2
    assert "σ_vm = 2.0000e+00" in text
    assert "单元代表应力" in text

    # 点击同时圈出命中单元外轮廓: (0.25, 0.25) 落在三角形 [0,1,2] —
    # 高亮闭圈为 (0,0)→(1,0)→(0,1)→(0,0) (label 供定位)
    hls = [c for c in fig.axes[0].collections if c.get_label() == "_readout_hl"]
    assert len(hls) == 1
    segs = hls[0].get_segments()
    assert len(segs) == 1
    assert np.allclose(segs[0], [[0, 0], [1, 0], [0, 1], [0, 0]])

    # blit 回归: 首帧全量重画后 (draw_event 捕获背景), 后续读值更新只
    # restore+blit 文本框/高亮, 不得再 draw_idle 全量重绘整张图 — 4 面板
    # Gouraud 图全量重绘曾使图窗整体卡顿
    fig.canvas.draw()
    calls = {"n": 0}
    monkeypatch.setattr(fig.canvas, "draw_idle",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    ev2 = MouseEvent("button_press_event", fig.canvas, 10, 10, button=1)
    ev2.inaxes = fig.axes[0]
    ev2.xdata, ev2.ydata = 2.0, 2.0  # 单位方外 → 域外文案
    fig.canvas.callbacks.process("button_press_event", ev2)
    assert "（模型外）" in fig.texts[-1].get_text()
    assert len(hls[0].get_segments()) == 0  # 点外 → 高亮清除
    assert calls["n"] == 0

    # 右键不读值 (缩放/上下文菜单点击不得覆盖读值/高亮)
    ev3 = MouseEvent("button_press_event", fig.canvas, 10, 10, button=3)
    ev3.inaxes = fig.axes[0]
    ev3.xdata, ev3.ydata = 0.25, 0.25
    fig.canvas.callbacks.process("button_press_event", ev3)
    assert "（模型外）" in fig.texts[-1].get_text()  # 文本未被右键覆盖
    assert len(hls[0].get_segments()) == 0          # 高亮亦不被右键覆盖
    plt.close(fig)


def test_interactive_plot_pump_path_no_hang(monkeypatch, capsys):
    """有图时后台线程读 + 主线程泵事件 — 输入照常返回, 不挂死.

    pause 桩改为绊线 — 泵循环再碰 plt.pause 即失败: pause 内部每轮
    show(block=False) → FigureManagerTk.show → canvas.draw_idle, 图窗
    每 50ms 全量重绘一次 4 面板 Gouraud 图 (blit 局部刷新救不了 —
    重绘由输入泵驱动, 与读值无关), 窗口永远卡顿 (用户实测).
    """
    def _no_pause(*a, **k):
        raise AssertionError("交互泵不得再用 plt.pause — 每 50ms 全量重绘")
    monkeypatch.setattr(viz, "plot_three", lambda *a, **k: None)
    monkeypatch.setattr(viz.plt, "get_fignums", lambda: [1])
    monkeypatch.setattr(viz.plt, "pause", _no_pause)

    def _slow_input(*a):
        # 首轮 lines.get(timeout=0.1) 必然超时 → 泵分支至少走一次,
        # pause 绊线才真正被泵路径经过
        time.sleep(0.15)
        return "q"
    monkeypatch.setattr("builtins.input", _slow_input)
    interactive_plot(_quad(), {"u": np.zeros(8)})
    out = capsys.readouterr().out
    assert "网格 + 载荷 + 边界条件" in out  # 菜单打印后正常退出


def test_plot_three_spr_seeds_cache_only_for_spr():
    """Gouraud 用 SPR 恢复时把节点场播种进 result['_spr_cache'] — 与
    stress_at_point 的 recovered 探针同源同算, 首次点击读值不再付 SPR
    预热 (粗网格 0.09s/细网格秒级); 其他恢复方法不播种 — _spr_cache
    语义恒为 SPR 场, L2/weighted 播种会污染 recovered 探针口径."""
    mesh = _two_tri()
    res = {"u": np.zeros(8), "stress": np.ones((2, 3)),
           "vm_stress": np.ones(2), "stress_qp": None}
    plot_three(mesh, res, tag="vm", recovery="SPR")
    assert "_spr_cache" in res
    res2 = {"u": np.zeros(8), "stress": np.ones((2, 3)),
            "vm_stress": np.ones(2), "stress_qp": None}
    plot_three(mesh, res2, tag="vm", recovery="L2")
    assert "_spr_cache" not in res2


def test_interactive_plot_p_key_probe_flow(monkeypatch, capsys):
    """p 键 → 坐标输入 → 终端打印两口径探针行 (element + recovered);
    域外坐标 → "not in mesh"; 格式非法 → "? 坐标格式: 如 0.72, 0.30"."""
    monkeypatch.setattr(viz, "plot_three", lambda *a, **k: None)
    answers = iter(["p", "0.25, 0.25", "p", "5, 5", "p", "zz", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    mesh = _two_tri()
    result = {"stress": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
    interactive_plot(mesh, result)
    out = capsys.readouterr().out
    assert "[probe] (0.2500, 0.2500)" in out
    assert "element  :" in out and "recovered:" in out
    assert "not in mesh" in out
    assert "? 坐标格式: 如 0.72, 0.30" in out


def test_interactive_plot_click_probe_with_toolbar_guard(monkeypatch,
                                                         capsys):
    """c 键开点击探针 → 真实 MouseEvent 点击走 stress_probe 打印;
    工具栏缩放/平移模式非空时点击不探针 (拖拽起止点击不得误读)."""
    def _plot(*a, **k):
        fig = plt.figure()
        fig.add_subplot(111)
        return fig
    monkeypatch.setattr(viz, "plot_three", _plot)
    monkeypatch.setattr(viz.plt, "close", lambda *a, **k: None)
    answers = iter(["c", "8", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    mesh = _two_tri()
    result = {"stress": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
              "stress_qp": None, "u": np.zeros(8)}
    interactive_plot(mesh, result)

    from matplotlib.backend_bases import MouseEvent
    fig = plt.gcf()
    ev = MouseEvent("button_press_event", fig.canvas, 10, 10, button=1)
    ev.inaxes = fig.axes[0]
    ev.xdata, ev.ydata = 0.25, 0.25
    fig.canvas.callbacks.process("button_press_event", ev)
    out = capsys.readouterr().out
    assert "[probe] (0.2500, 0.2500)" in out
    assert "element  :" in out and "recovered:" in out

    # 工具栏模式非空 (zoom 拖拽中) → 点击只走工具栏语义, 不探针
    class _Bar:
        mode = "zoom rect"
    fig.canvas.toolbar = _Bar()
    ev2 = MouseEvent("button_press_event", fig.canvas, 10, 10, button=1)
    ev2.inaxes = fig.axes[0]
    ev2.xdata, ev2.ydata = 0.25, 0.25
    fig.canvas.callbacks.process("button_press_event", ev2)
    assert "[probe]" not in capsys.readouterr().out
    fig.canvas.toolbar = None
    plt.close(fig)


def test_interactive_plot_deformed_panel_click_inverse_mapping(
        monkeypatch):
    """变形面板点击逆映射: 在变形位置点击 → 读未变形网格对应点.

    ux=0.1 常场 + scale=100 → 面板几何右移 10 (x∈[10,11]); 点击
    (10.25, 0.25) 即变形后三角形 [0,1,2] 内点 → 逆映射 (0.25, 0.25)
    读 u_x=0.1, 高亮轮廓为变形外轮廓 [[10,0],[11,0],[10,1],[10,0]].
    曾按点击坐标直接查未变形网格: (10.25,0.25) 域外 → 恒"模型外".
    """
    def _plot(*a, **k):
        fig = plt.figure()
        for _ in range(3):
            fig.add_subplot(1, 3, len(fig.axes) + 1)
        return fig
    monkeypatch.setattr(viz, "plot_three", _plot)
    monkeypatch.setattr(viz.plt, "close", lambda *a, **k: None)
    answers = iter(["2", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    mesh = _two_tri()
    u = np.zeros(8)
    u[0::2] = 0.1                       # ux 常场, uy=0
    result = {"u": u, "stress": np.ones((2, 3)), "stress_qp": None}
    interactive_plot(mesh, result)

    from matplotlib.backend_bases import MouseEvent
    fig = plt.gcf()
    ev = MouseEvent("button_press_event", fig.canvas, 10, 10, button=1)
    ev.inaxes = fig.axes[1]             # 中面板 = 变形形状
    ev.xdata, ev.ydata = 10.25, 0.25
    fig.canvas.callbacks.process("button_press_event", ev)

    text = fig.texts[-1].get_text()
    assert "u_x = 1.0000e-01" in text
    assert "（模型外）" not in text
    # 读值行含 CJK 标签, 必须 sans-serif (monospace 无 CJK 字体且
    # matplotlib 无跨族回退 → 方框 + missing glyph 警告)
    assert "sans-serif" in fig.texts[-1].get_family()

    hls = [c for c in fig.axes[1].collections
           if c.get_label() == "_readout_hl"]
    assert len(hls) == 1
    segs = hls[0].get_segments()
    assert len(segs) == 1
    assert np.allclose(segs[0], [[10, 0], [11, 0], [10, 1], [10, 0]])
    plt.close(fig)
