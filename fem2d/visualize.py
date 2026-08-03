"""可视化 — 交互式选择云图"""
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize

from .element import evaluate_vector_field
from .error_est import compute_traction_jumps  # 唯一公式, 供 visualize 和 error_est 共用
from .material import von_mises
from .spr import spr_recovery
from .stress import nodal_L2_projection, nodal_simple, nodal_weighted, principal_stresses

# ── 应力恢复方法注册表 ──
_RECOVERY_METHODS = {
    "SPR":      ("SPR (超收敛 patch 拟合)", spr_recovery),
    "weighted": ("面积加权平均 (≈Abaqus banded)", nodal_weighted),
    "L2":       ("L2 投影 (全局光滑)", nodal_L2_projection),
    "simple":   ("算术平均", nodal_simple),
}
_DEFAULT_RECOVERY = "SPR"

# 中文字体 — 修复豆包反馈的"方框"问题
import matplotlib.font_manager as fm

_CJK_FONTS = [f.name for f in fm.fontManager.ttflist
              if any(k in f.name for k in ['YaHei', 'SimHei', 'SimSun', 'CJK', 'Heiti', 'Songti', 'Noto Sans CJK'])]
if _CJK_FONTS:
    plt.rcParams['font.sans-serif'] = _CJK_FONTS + ['DejaVu Sans']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CMAP = LinearSegmentedColormap.from_list('fem',
    ['#000080','#0040c0','#0080ff','#00c0c0','#00e080','#80f000',
     '#ffff00','#ffc000','#ff6000','#ff0000','#800000'], N=256)


def _display_triangulation(nodes, elements):
    """Triangulate only for scalar rendering and retain parent element ids."""
    elements = np.asarray(elements, dtype=int)
    if elements.shape[1] == 3:
        display_elements = elements
        parent = np.arange(len(elements), dtype=int)
    elif elements.shape[1] == 4:
        display_elements = np.stack(
            [elements[:, [0, 1, 2]], elements[:, [0, 2, 3]]],
            axis=1,
        ).reshape(-1, 3)
        parent = np.repeat(np.arange(len(elements), dtype=int), 2)
    else:
        raise ValueError(
            f"Visualization supports 3-node and 4-node elements, got "
            f"{elements.shape[1]} nodes per element")
    return mtri.Triangulation(
        nodes[:, 0], nodes[:, 1], display_elements), parent


def _draw_element_edges(ax, nodes, elements, **kwargs):
    """Draw true polygon edges without display-only Q4 diagonals.

    网格过密时自动降低线宽和透明度，避免黑糊一片。
    """
    n_elem = len(elements)
    defaults = {"facecolors": "none", "edgecolors": "k",
                "linewidths": 0.3, "alpha": 0.5}
    if n_elem > 5000:
        defaults["linewidths"] = 0.08
        defaults["alpha"] = 0.15
    elif n_elem > 1000:
        defaults["linewidths"] = 0.15
        defaults["alpha"] = 0.30
    defaults.update(kwargs)
    collection = PolyCollection(nodes[elements], **defaults)
    ax.add_collection(collection)
    ax.autoscale_view()
    return collection


def _to_node(mesh, ev, integration_values=None, method="SPR"):
    """应力恢复 — 可切换方法

    method: "SPR" | "weighted" | "L2" | "simple"
    """
    data = integration_values if integration_values is not None else ev
    if method == "SPR":
        result = spr_recovery(mesh, data)
    elif method == "L2":
        # L2 投影直接消费积分点数据 (按内核积分规则加权)。
        # 先算术平均会丢掉积分点分布信息 — 云图与正确 L2 投影相差 ~30%。
        result = nodal_L2_projection(mesh, data)
    else:
        # weighted / simple: 接受 (ne, ncomp) 或 (ne, nqp, ncomp);
        # 单元代表值: weighted 取 dA 加权平均 (与 solver 的 stress 一致),
        # simple 取算术平均 (曾对 simple 也做 dA 加权, 两种方法在歪扭
        # 网格上无差别)。
        if data.ndim == 3:
            if method == "simple":
                data = data.mean(axis=1)
            else:
                weights = mesh.element_kernel.recovery_weights(mesh)
                if weights is not None:
                    weights = np.asarray(weights, dtype=float)
                    if weights.shape == (data.shape[0], data.shape[1]):
                        denom = weights.sum(axis=1)
                        denom[denom == 0.0] = 1.0
                        data = np.sum(
                            data * weights[:, :, None], axis=1) / denom[:, None]
                    else:
                        data = data.mean(axis=1)
                else:
                    data = data.mean(axis=1)
        result = _RECOVERY_METHODS[method][1](mesh, data)
    return result[:, 0] if result.shape[1] == 1 else result


# ═══════════════════════════════════════════════════════════════
# Bathe §4.3.6 Eq (4.107): 内部边牵引跳跃
# compute_traction_jumps 从 error_est 导入 — 唯一公式, 避免重复实现
# ═══════════════════════════════════════════════════════════════

def plot_traction_jumps(mesh, elem_stress, ax=None, cmap=None,
                        vmax=None, linewidth=0.3, title=None,
                        sigma_ref=None):
    """单元间牵引跳跃图 — 基于 Bathe §4.3.6 Eq (4.107) 衍生

    每条内部边按 ‖(σ⁺ − σ⁻)n‖ / denom 着色, 其中:
      sigma_ref=None → denom = max((‖σ⁺‖_F+‖σ⁻‖_F)/2, 0.05·σ_95) (局部+全局混合)
      sigma_ref=1e6   → denom = 1e6 (固定名义应力, 适合跨网格收敛对比)

    sigma_ref: 固定参考应力 [Pa]; 跨网格对比时传入同一值, 色标才有可比性.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 7))
    if cmap is None:
        cmap = plt.cm.YlOrRd

    jumps = compute_traction_jumps(mesh, elem_stress, sigma_ref=sigma_ref)

    segments, values = [], []
    nodes = mesh.nodes
    for j in jumps:
        xa, ya = float(nodes[j['node_a'], 0]), float(nodes[j['node_a'], 1])
        xb, yb = float(nodes[j['node_b'], 0]), float(nodes[j['node_b'], 1])
        segments.append([(xa, ya), (xb, yb)])
        values.append(float(j['jump_rel']))

    if not values:
        ax.set_title("(no internal edges)")
        return None

    values_arr = np.asarray(values, dtype=np.float64).ravel()
    actual_max = float(values_arr.max())

    auto_vmax = vmax is None
    if auto_vmax:
        p98 = float(np.percentile(values_arr, 98))
        if p98 < 1e-30:
            vmax = 1.0
            vmax_label = 'zero-field fallback'
        else:
            vmax = p98
            vmax_label = 'P98'
    else:
        vmax_label = 'user'
    print(f"  traction_jump: actual max={actual_max:.4f}, plot vmax({vmax_label})={vmax:.4f}")

    norm = Normalize(vmin=0.0, vmax=vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm,
                        linewidths=linewidth)
    lc.set_array(values_arr)
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(values_arr)
    if sigma_ref is not None:
        denom_label = rf'{sigma_ref:.3g} Pa (fixed)'
    else:
        denom_label = r'\max(\bar{\sigma}_{\rm local},\,0.05\sigma_{95})'
    plt.colorbar(sm, ax=ax, shrink=0.8,
                 label=rf'$\|(\sigma^+-\sigma^-)n\| \;/\; {denom_label}$  '
                       rf'(interelement traction jump, vmax={vmax_label})')
    if title:
        ax.set_title(title)

    return lc


def _plot_loads(mesh, ax):
    """在网格上叠加载荷可视化 — 面力(红) / 体力(蓝) / 集中力(绿)

    箭头长度 = 力的大小, 自动缩放到模型尺寸的 10%。
    """
    nodes = mesh.nodes
    # 自动缩放: 最大箭头 = 模型跨度的 8%
    span = (nodes[:,0].max()-nodes[:,0].min() + nodes[:,1].max()-nodes[:,1].min())/2
    # 仅零跨度退化模型才回退 — 曾 1e-15 下限使微尺度模型 (span 1e-16)
    # 的箭头按 1.0 缩放, 画到模型外
    if span == 0.0: span = 1.0

    # ── 1. 面力 (红色箭头, 沿边界边中点) ──
    if mesh.surface_tractions:
        # 收集所有面力的最大幅值用于统一缩放
        all_tx, all_ty = [], []
        for st in mesh.surface_tractions:
            a, b = st["nodes"]; t = st["traction"]
            xa, ya = nodes[a]; xb, yb = nodes[b]
            xm = 0.5*(xa + xb); ym = 0.5*(ya + yb)
            if st.get("is_pressure"):
                p_val = t[0]
                p = p_val(xm, ym) if callable(p_val) else p_val
                # 法向由当前几何实时计算 (不缓存 — 几何变更后自动跟随)
                nx, ny = mesh.boundary_outward_normal(a, b)
                tx, ty = -p*nx, -p*ny
            else:
                tx, ty = evaluate_vector_field(t, xm, ym)
            all_tx.append(abs(tx)); all_ty.append(abs(ty))
        # 1.0 下限曾使微尺度载荷 (t_max=1e-20) 的箭头缩放失真, 全部
        # 变成 ~1e-17 不可见
        t_max = max(max(all_tx), max(all_ty), np.finfo(float).tiny)
        arrow_scale = span * 0.08 / t_max

        # 下采样: 最多画 60 个箭头, 均匀抽取
        n_st = len(mesh.surface_tractions)
        st_step = max(1, n_st // 60)
        for k, st in enumerate(mesh.surface_tractions):
            if k % st_step != 0:
                continue
            a, b = st["nodes"]; t = st["traction"]
            xm, ym = (nodes[a,0]+nodes[b,0])/2, (nodes[a,1]+nodes[b,1])/2
            if st.get("is_pressure"):
                p_val = t[0]
                p = p_val(xm, ym) if callable(p_val) else p_val
                nx, ny = mesh.boundary_outward_normal(a, b)
                tx, ty = -p*nx, -p*ny
            else:
                tx, ty = evaluate_vector_field(t, xm, ym)
            dx, dy = tx * arrow_scale, ty * arrow_scale
            # 曾用 1e-15 绝对阈值: 微尺度模型的归一化箭头 (长度 ~8%·span)
            # 全被跳过, 载荷不可见
            if abs(dx) + abs(dy) > 0.0:
                ax.arrow(xm, ym, dx, dy, head_width=span*0.012,
                        head_length=span*0.015, fc='red', ec='red',
                        alpha=0.85, lw=1.5, zorder=5,
                        length_includes_head=True)

    # ── 2. 体力 (蓝色箭头, 每单元质心) ──
    if mesh.body_force is not None:
        bf = mesh.body_force
        # 抽样: 最多约 40 个箭头
        n_elem = mesh.n_elements
        step = max(1, n_elem // 40)
        bx_all, by_all = [], []
        for eid in range(0, n_elem, step):
            xc, yc = mesh.centroids[eid]
            bx, by = evaluate_vector_field(bf, xc, yc)
            bx_all.append(abs(bx)); by_all.append(abs(by))
        # 1.0 下限曾使微尺度/非 SI 单位体力 (f_max=7.86e-4) 箭头缩短
        # 1/f_max 倍静默不可见 — 与面力分支 223 行统一
        b_max = max(max(bx_all), max(by_all), np.finfo(float).tiny)
        arrow_scale_b = span * 0.05 / b_max

        for eid in range(0, n_elem, step):
            xc, yc = mesh.centroids[eid]
            bx, by = evaluate_vector_field(bf, xc, yc)
            dx, dy = bx * arrow_scale_b, by * arrow_scale_b
            # 曾用 1e-15 绝对阈值: 微尺度模型的归一化箭头 (长度 ~8%·span)
            # 全被跳过, 载荷不可见
            if abs(dx) + abs(dy) > 0.0:
                ax.arrow(xc, yc, dx, dy, head_width=span*0.008,
                        head_length=span*0.01, fc='blue', ec='blue',
                        alpha=0.3, lw=0.8, zorder=3,
                        length_includes_head=True)

    # ── 3. 集中力 (绿色箭头, 节点位置) ──
    if mesh.concentrated_forces:
        cf_all = [abs(cf["force"][0])+abs(cf["force"][1]) for cf in mesh.concentrated_forces]
        cf_max = max(cf_all) if cf_all else 1.0
        arrow_scale_c = span * 0.10 / cf_max

        for cf in mesh.concentrated_forces:
            nid = cf["node"]; fx, fy = cf["force"]
            x, y = nodes[nid]
            dx, dy = fx * arrow_scale_c, fy * arrow_scale_c
            # 曾用 1e-15 绝对阈值: 微尺度模型的归一化箭头 (长度 ~8%·span)
            # 全被跳过, 载荷不可见
            if abs(dx) + abs(dy) > 0.0:
                ax.arrow(x, y, dx, dy, head_width=span*0.015,
                        head_length=span*0.02, fc='green', ec='darkgreen',
                        alpha=0.9, lw=2.0, zorder=6,
                        length_includes_head=True)

    # ── 图例 ──
    from matplotlib.lines import Line2D
    legend_items = []
    if mesh.surface_tractions:
        legend_items.append(Line2D([0],[0], color='red', lw=2, label=f'面力 ({len(mesh.surface_tractions)} 边)'))
    if mesh.body_force is not None:
        bf_label = '体力'
        bf = mesh.body_force
        if isinstance(bf, (tuple, list)) and len(bf) == 2:
            bfx = bf[0] if isinstance(bf[0], (int, float)) else 'f(x,y)'
            bfy = bf[1] if isinstance(bf[1], (int, float)) else 'f(x,y)'
            bf_label += f' ({bfx},{bfy}) N/m³'
        legend_items.append(Line2D([0],[0], color='blue', lw=2, label=bf_label))
    if mesh.concentrated_forces:
        legend_items.append(Line2D([0],[0], color='green', lw=2, label=f'集中力 ({len(mesh.concentrated_forces)} 点)'))
    if legend_items:
        ax.legend(handles=legend_items, loc='lower right', fontsize=8,
                 framealpha=0.85)


def plot_mesh(mesh, ax=None, title="Mesh + BCs", figsize=(9,7), show_loads=True):
    mesh.build_connectivity()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    _draw_element_edges(ax, mesh.nodes, mesh.elements)
    # 固定节点 — 下采样: 最多画 40 个标记
    fixed_nodes = np.unique(mesh.fixed_dofs // 2)
    n_fixed = len(fixed_nodes)
    if n_fixed > 40:
        step = max(1, n_fixed // 40)
        fixed_nodes = fixed_nodes[::step]
    ax.plot(mesh.nodes[fixed_nodes, 0], mesh.nodes[fixed_nodes, 1],
            'bs', ms=5, mfc='none', mew=1.5, label=f'fixed (~{len(fixed_nodes)}/{n_fixed} nodes)')
    # 载荷
    if show_loads:
        _plot_loads(mesh, ax)
    ax.set_aspect('equal'); ax.set_title(title)

def _resolve_values_location(values, n_elements, location):
    """确定数据位置 (显式参数 > 自动推断)."""
    if location == 'auto':
        return len(values) == n_elements
    return location == 'element'


def _style_colorbar(cbar, e_min, e_max, title=None, n_ticks=8):
    """色条刻度格式化 — 按量级选择 e6/e3/e-6 显示 (isoband 与统一色条共用).

    title=None 时不改动 label (isoband 已自行设置含 band 宽度的 label).
    """
    if title is not None:
        cbar.set_label(title)
    tick_vals = np.linspace(e_min, e_max, n_ticks)
    rng = e_max - e_min if e_max > e_min else 1.0
    if rng > 1e6:
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f'{t:.2f}e6' for t in np.round(tick_vals / 1e6, 2)])
    elif rng > 1e3:
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f'{t:.2f}e3' for t in np.round(tick_vals / 1e3, 2)])
    elif rng < 1e-3:
        # 小值域: 统一科学计数显示 (曾 "0.01e-6" 可读性差)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f'{t:.3g}' for t in tick_vals])
    else:
        tick_vals = np.round(tick_vals, 4)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f'{t:.4g}' for t in tick_vals])


def _plot_flat_contour(ax, tri, display_parent, values, e_min, e_max):
    """flat: 单元代表应力 (边界跳跃可见)."""
    return ax.tripcolor(
        tri, facecolors=np.asarray(values)[display_parent], cmap=CMAP,
        shading='flat', vmin=e_min, vmax=e_max)


def _plot_scalar_jump_contour(ax, tri, display_parent, values, mesh, nd,
                              e_min, e_max):
    """scalar_jump: 内部边按 |Δs|/σ_range 着色 (标量跳跃, 非牵引跳跃)."""
    sigma_range = e_max - e_min
    # 常场回退基于场尺度 — 曾 1e-30 绝对下限使微尺度场 (1e-40) 的
    # 跳跃归一化失真
    if sigma_range <= 1e-14 * max(abs(e_max), abs(e_min),
                                  np.finfo(float).tiny):
        sigma_range = max(abs(e_max), abs(e_min), np.finfo(float).tiny)
    # 单元底色 — 半透明 flat 应力场
    tpc = ax.tripcolor(
        tri, facecolors=np.asarray(values)[display_parent], cmap=CMAP,
        shading='flat', vmin=e_min, vmax=e_max, alpha=0.55)
    # 内部边跳跃
    jumps = _compute_edge_jumps_scalar(mesh, values, sigma_range)
    if jumps:
        jvals = np.asarray(
            [float(j['jump_rel']) for j in jumps], dtype=np.float64).ravel()
        actual_max = float(jvals.max())
        p98 = float(np.percentile(jvals, 98))
        if p98 < 1e-30:
            jvmax = 1.0
            jvmax_label = 'zero-field fallback'
        else:
            jvmax = p98
            jvmax_label = 'P98'
        print(f"  scalar_jump: actual max={actual_max:.4f}, "
              f"plot vmax({jvmax_label})={jvmax:.4f}")
        jnorm = Normalize(vmin=0.0, vmax=jvmax)
        jsegs = [[(float(s[0][0]), float(s[0][1])),
                  (float(s[1][0]), float(s[1][1]))]
                 for s in [j['segment'] for j in jumps]]
        jlc = LineCollection(jsegs, cmap=plt.cm.YlOrRd, norm=jnorm,
                             linewidths=0.3)
        jlc.set_array(jvals)
        ax.add_collection(jlc)
        jsm = ScalarMappable(cmap=plt.cm.YlOrRd, norm=jnorm)
        jsm.set_array(jvals)
        plt.colorbar(jsm, ax=ax, shrink=0.8,
                     label=r'$|\Delta s| / (\sigma_{\max} - \sigma_{\min})$  '
                           f'(scalar edge jump, vmax={jvmax_label})')
    # 叠加细网格线
    _draw_element_edges(ax, nd, mesh.elements, linewidths=0.3, alpha=0.25)
    return tpc


def _validate_isoband_levels(levels):
    """isoband 固定区间校验: 1D / ≥2 值 / 有限 / 严格递增 / 等距."""
    band_levels = np.asarray(levels, dtype=float)
    if band_levels.ndim != 1 or len(band_levels) < 2:
        raise ValueError(
            f'isoband levels must be a 1D array with >= 2 values, '
            f'got ndim={band_levels.ndim}, len={len(band_levels)}')
    if not np.all(np.isfinite(band_levels)):
        raise ValueError('isoband levels contain NaN or Inf')
    delta = np.diff(band_levels)
    if np.any(delta <= 0.0):
        raise ValueError('isoband levels must be strictly increasing')
    if not np.allclose(delta, delta[0], rtol=1e-10, atol=1e-14):
        raise ValueError(
            'Bathe isobands require uniformly spaced levels '
            f'(first step={delta[0]:.3g}, min={delta.min():.3g}, '
            f'max={delta.max():.3g})')
    return band_levels


def _plot_isoband_contour(ax, tri, display_parent, values, title,
                          n, levels, e_min, e_max):
    """Bathe §4.3.6 等应力带法 (Sussman & Bathe) — 固定应力区间离散着色.

    自行处理色条后返回 None (调用方跳过统一色条逻辑).
    """
    if levels is not None:
        band_levels = _validate_isoband_levels(levels)
        n_bands = len(band_levels) - 1
    else:
        # 常应力保护: e_min≈e_max 时加 padding — 阈值和 padding 都须基于
        # 场本身尺度, 曾用 max(..., 1.0) 把微尺度场 (1e-12) 的色标抬到
        # [~1e-12, 1.0] 云图塌缩为单色
        scl = max(abs(e_min), abs(e_max), np.finfo(float).tiny)
        if abs(e_max - e_min) <= 1e-14 * scl:
            pad = scl * 1e-6
            b_min, b_max = e_min - pad, e_max + pad
        else:
            b_min, b_max = e_min, e_max
        n_bands = max(2, int(n))
        band_levels = np.linspace(b_min, b_max, n_bands + 1)
    band_width = (band_levels[-1] - band_levels[0]) / n_bands

    # 超界警告: 值超出 levels 范围会被 clip 着色, 需提醒用户
    below = int(np.count_nonzero(values < band_levels[0]))
    above = int(np.count_nonzero(values > band_levels[-1]))
    if below or above:
        print(f'  isoband warning: {below} elems below min={band_levels[0]:.3g}, '
              f'{above} elems above max={band_levels[-1]:.3g}')

    # 离散 colormap + BoundaryNorm
    band_rgba = CMAP(np.linspace(0.05, 0.95, n_bands))
    band_cmap = ListedColormap(band_rgba, name='isoband')
    band_norm = BoundaryNorm(band_levels, ncolors=n_bands, clip=True)
    tpc = ax.tripcolor(
        tri, facecolors=np.asarray(values)[display_parent],
        cmap=band_cmap, norm=band_norm, shading='flat')

    # 不画网格线 — Sussman–Bathe: 全部信息量在"带边界在单元边处断裂",
    # 满屏网格线会使带断裂与网格线无法区分 (Bathe §4.3.6 Fig 4.15)

    # 色条: BoundaryNorm 直接显示离散应力带
    cbar = plt.colorbar(tpc, ax=ax, shrink=0.8,
                        boundaries=band_levels, ticks=band_levels)
    range_label = f'  [bands: {band_width:.3g} wide, {n_bands} bands]'
    cbar.set_label(f'{title}{range_label}' if title else f'Isobands{range_label}')
    _style_colorbar(cbar, band_levels[0], band_levels[-1],
                    n_ticks=len(band_levels))

    ax.set_aspect('equal')
    if title:
        ax.set_title(f'{title}  [bands: {band_width:.3g}]')
    return None


def _plot_gouraud_contour(ax, tri, values, mesh, n, recovery):
    """Gouraud: SPR 磨平到节点 → 光滑云图 + 等值线 (推荐用于报告)."""
    if len(values) == mesh.n_elements:
        # 标量场保护: 对 vm/s1/s2/taumax 等非线性不变量, SPR 直接在标量上
        # 做最小二乘可能产生非物理解 (如负 Mises)。plot_three() 已正确处理
        # (先恢复应力分量再从分量算不变量)。此处拒绝标量+gouraud 组合。
        if values.ndim == 1:
            raise ValueError(
                "plot_contour with shading='gouraud' and element-located scalar "
                "(e.g. von Mises, principal stress) is unreliable — SPR on the "
                "scalar directly may produce negative Mises or other artifacts. "
                "Use plot_three() instead, which recovers stress components first "
                "and then computes the invariant from the recovered components. "
                "Or use shading='flat' / 'isoband' which operate on raw element values.")
        values = _to_node(mesh, values, method=recovery)

    # 统一使用恢复后的节点值范围 (颜色 + 等值线一致)
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmax - vmin <= 1e-14 * max(abs(vmin), abs(vmax),
                                  np.finfo(float).tiny):
        # 近常场 padding 基于场自身尺度 — 曾用绝对 1e-15 阈值 + 1.0 pad,
        # 微尺度场 (跨度 1e-16 绝对单位) 色标变 [1e-12, 1.0] 单色塌缩
        #
        pad = max(abs(vmin), abs(vmax), np.finfo(float).tiny) * 1e-6
        vmin -= pad
        vmax += pad
    tpc = ax.tripcolor(tri, values, cmap=CMAP, shading='gouraud',
                       vmin=vmin, vmax=vmax)
    ax.tricontour(tri, values,
                  levels=np.linspace(vmin, vmax, min(12, n)),
                  colors='k', linewidths=0.6, alpha=0.55)
    return tpc, vmin, vmax


def plot_contour(mesh, values, title="", n=30, ax=None, figsize=(9,7),
                 deformed=False, u=None, scale=1.0, shading='gouraud',
                 location='auto', levels=None, recovery="SPR"):
    """应力云图 — Bathe §4.3.6

    shading 模式:
      'flat':       单元代表应力 (边界跳跃可见)
      'gouraud':    SPR 磨平到节点 → 光滑云图 + 等值线 (推荐用于报告)
      'isoband':    Bathe §4.3.6 等应力带法 — 应力带在单元边界的不连续可定性指示局部离散误差
      'scalar_jump': 边标量跳跃图 — 内部边按 |Δs|/σ_range 着色

    location: 'auto' (按数组长度推断), 'element', 'node'
    levels: 用于 isoband 的固定应力区间边界, 如 np.arange(0, 40e6, 2.5e6)
            未指定时自动按 (e_min, e_max, n_bands+1) 生成
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    nd = (mesh.nodes + scale*u.reshape(-1, 2)
          if (deformed and u is not None) else mesh.nodes)
    tri, display_parent = _display_triangulation(nd, mesh.elements)

    # ── 数据位置 + shading 兼容性 ──
    is_element = _resolve_values_location(values, mesh.n_elements, location)
    if shading in ("flat", "scalar_jump", "isoband") and not is_element:
        # 这些 shading 只接受单元定位数据; 静默落入 gouraud 分支会画出与
        # 用户请求不符的云图且无提示
        raise ValueError(
            f"shading={shading!r} requires element-located values "
            f"(got {len(values)} values for {mesh.n_elements} elements). "
            "Use location='element', or shading='gouraud' with node data.")
    e_min = e_max = None
    if is_element:
        e_min, e_max = float(np.min(values)), float(np.max(values))

    # ── 各 shading 分支 (flat 显式 facecolors 避免 n_nodes==n_elem 误判) ──
    if shading == 'flat' and is_element:
        tpc = _plot_flat_contour(ax, tri, display_parent, values, e_min, e_max)
    elif shading == 'scalar_jump' and is_element:
        tpc = _plot_scalar_jump_contour(
            ax, tri, display_parent, values, mesh, nd, e_min, e_max)
    elif shading == 'isoband' and is_element:
        tpc = _plot_isoband_contour(
            ax, tri, display_parent, values, title, n, levels, e_min, e_max)
        return  # isoband 已自行处理色条
    else:
        tpc, e_min, e_max = _plot_gouraud_contour(
            ax, tri, values, mesh, n, recovery)

    # ── 统一色条 ──
    if shading != 'scalar_jump':  # scalar_jump 有自己的边色条
        cbar = plt.colorbar(tpc, ax=ax, shrink=0.8, label=title)
        _style_colorbar(cbar, e_min, e_max, title)

    ax.set_aspect('equal')
    if title:
        ax.set_title(title)

def _compute_edge_jumps_scalar(mesh, elem_values, sigma_range):
    """计算内部边应力跳跃 — Bathe §4.3.6 标量版

    对每条内部边: jump = |s⁺ − s⁻| / σ_range

    使用全局应力范围 σ_range = σ_max − σ_min 做归一化,
    避免旧公式 |s1−s2|/max(|s1|,|s2|) 在应力过零处的 200% 虚高问题。

    返回 list of dict: {segment: [(xa,ya), (xb,yb)], jump_rel: float}
    """
    mesh.build_connectivity()
    jumps = []
    for (a, b), eids in mesh.edge_to_elems.items():
        if len(eids) == 2:
            diff = abs(elem_values[eids[0]] - elem_values[eids[1]])
            jumps.append({
                'segment': [mesh.nodes[a], mesh.nodes[b]],
                'jump_rel': diff / sigma_range,
            })
    return jumps


# ---- 所有可画的图 ----
PLOTS = {
    "1":  ("mesh",          "网格 + 载荷 + 边界条件"),
    "2":  ("ux",            "位移 u_x"),
    "3":  ("uy",            "位移 u_y"),
    "4":  ("umag",          "位移 |u|"),
    "5":  ("sx",            "应力 sigma_x"),
    "6":  ("sy",            "应力 sigma_y"),
    "7":  ("txy",           "剪应力 tau_xy"),
    "8":  ("vm",            "von Mises 应力"),
    "9":  ("s1",            "面内最大主应力 sigma_1 (in-plane)"),
    "10": ("s2",            "面内最小主应力 sigma_2 (in-plane)"),
    "11": ("taumax",        "面内最大剪应力 tau_max (in-plane)"),
    "12": ("loads",         "载荷明细 (面力/体力/集中力)"),
}

def plot_three(mesh, result, tag='vm', scale=100, save=None,
               isoband_levels=None, isoband_tag=None, sigma_ref=None,
               recovery="SPR"):
    """Bathe §4.3.6 三连图: Gouraud / Isoband / Traction Jump 并排对比

    recovery: "SPR" | "weighted" | "L2" | "simple" — 左上 Gouraud 图的应力恢复方法

    isoband_levels: 固定应力区间, 如 np.arange(0, 40e6, 2.5e6)
    isoband_tag: 固定带宽仅当 tag==isoband_tag 时生效 (避免 Mises 范围误用于 S11)
                 为 None 时所有分量均使用固定带宽
    """
    u = result["u"]; u2 = u.reshape(-1,2); u_mag = np.sqrt(u2[:,0]**2+u2[:,1]**2)
    s = result["stress"]  # (n_elem, 3) — [σ_x, σ_y, τ_xy]

    is_stress = tag not in ("ux", "uy", "umag", "mesh", "loads")
    if is_stress:
        # ── 应力恢复 → 从恢复分量计算 Mises/主应力 ──
        # 仅应力云图需要 — 只画网格/载荷/位移时不预跑 (大网格无效开销)。
        s_node = _to_node(mesh, s, result.get("stress_qp"), method=recovery)  # (n_nodes, 3)
        sx_n, sy_n, txy_n = s_node[:, 0], s_node[:, 1], s_node[:, 2]
        vm_n = von_mises(s_node, mesh.plane_type, mesh.nu)

        s1_n, s2_n, tau_n, _ = principal_stresses(s_node)

        vm_e = result["vm_stress"]
        s1_e, s2_e, tau_e, _ = principal_stresses(s)

        g_vals = {"sx": sx_n, "sy": sy_n, "txy": txy_n, "vm": vm_n,
                  "s1": s1_n, "s2": s2_n, "taumax": tau_n}
        f_vals = {"sx": s[:,0], "sy": s[:,1], "txy": s[:,2], "vm": vm_e,
                  "s1": s1_e, "s2": s2_e, "taumax": tau_e}

    titles = {"sx": r"$\sigma_x$ (Pa)", "sy": r"$\sigma_y$ (Pa)",
              "txy": r"$\tau_{xy}$ (Pa)", "vm": "von Mises (Pa)",
              "s1": r"$\sigma_1$ (in-plane) (Pa)",
              "s2": r"$\sigma_2$ (in-plane) (Pa)",
              "taumax": r"$\tau_{max}$ (in-plane) (Pa)",
              "ux": "$u_x$ (m)", "uy": "$u_y$ (m)", "umag": "$|u|$ (m)"}

    if is_stress:
        fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    if tag in ("ux", "uy", "umag"):
        vals = u2[:,0] if tag == 'ux' else (u2[:,1] if tag == 'uy' else u_mag)
        # 左: 未变形形状
        plot_contour(mesh, vals, titles[tag], ax=axes[0],
                    deformed=False, shading='gouraud', location='node')
        axes[0].set_title('Undeformed shape', fontsize=12)
        # 中: 变形形状
        plot_contour(mesh, vals, titles[tag], ax=axes[1],
                    deformed=True, u=u, scale=scale, shading='gouraud',
                    location='node')
        axes[1].set_title('Deformed shape', fontsize=12)
        # 右: 变形形状 + 网格
        plot_contour(mesh, vals, titles[tag], ax=axes[2],
                    deformed=True, u=u, scale=scale, shading='gouraud',
                    location='node')
        def_nodes = mesh.nodes + scale * u.reshape(-1, 2)
        _draw_element_edges(
            axes[2], def_nodes, mesh.elements,
            linewidths=0.25, alpha=0.5)
        axes[2].set_title('Deformed shape + mesh', fontsize=12)
    elif tag == "mesh":
        plot_mesh(mesh, ax=axes[0], show_loads=False)
        axes[0].set_title('Mesh only', fontsize=12)
        plot_mesh(mesh, ax=axes[1], show_loads=True)
        axes[1].set_title('Mesh + BCs + Loads', fontsize=12)
        # 曾 axes[2] 也是 show_loads=True 的重复图 — 无载荷 vs 含载荷
        # 两张对比已覆盖全部信息, 第三格置空
        axes[2].axis('off')
    elif tag == "loads":
        plot_mesh(mesh, ax=axes[0], show_loads=True)
        axes[0].set_title('All loads', fontsize=12)
        # 曾三张图内容完全相同, 仅标题不同 — 只留一张
        axes[1].axis('off')
        axes[2].axis('off')
    else:
        # 2×2 四图: Gouraud / Isoband / Traction Jump / η_K
        plot_contour(mesh, g_vals[tag], titles[tag], ax=axes[0,0],
                    shading='gouraud', location='node')
        axes[0,0].set_title(f'Gouraud ({recovery}磨平)', fontsize=10)

        levels_for_tag = isoband_levels if (isoband_tag is None or tag == isoband_tag) else None
        plot_contour(mesh, f_vals[tag], titles[tag], ax=axes[0,1],
                    shading='isoband', location='element', n=12,
                    levels=levels_for_tag)
        axes[0,1].set_title(r'Isoband (Bathe §4.3.6 Fig 4.15)', fontsize=10)

        plot_traction_jumps(mesh, s, ax=axes[1,0],
                          title='Interelement Traction Jump\n(full stress tensor)',
                          sigma_ref=sigma_ref)
        axes[1,0].set_title('Interelement Traction Jump\n(full stress tensor)', fontsize=10)

        from .error_est import element_refinement_indicator
        eta_K = element_refinement_indicator(mesh, result)
        plot_contour(mesh, eta_K, r'$\eta_K$ (residual-based estimator) [Pa·m]',
                    ax=axes[1,1], shading='flat', location='element')
        axes[1,1].set_title(r'$\eta_K$ — residual error indicator (Verfürth 1996)', fontsize=10)

    fig.suptitle(f'Bathe §4.3.6 — {titles.get(tag, tag)}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f'  → {save}')
        # 批量保存后主动关闭 — 曾长期不 close, 参数扫描逐次累积内存
        #
        plt.close(fig)
        return
    if plt.get_backend() != 'Agg':
        plt.show(block=False)


def interactive_plot(mesh, result, scale=100, isoband_levels=None, isoband_tag=None, sigma_ref=None):
    """轻量交互 — 终端输入分量名, 图里选着色模式

    按键:
      1-12  选择分量
      m     切换应力恢复方法 (SPR → weighted → L2 → simple → SPR ...)
      q     退出
    """
    recovery = _DEFAULT_RECOVERY
    recovery_keys = list(_RECOVERY_METHODS.keys())

    while True:
        print(f"\n  [恢复方法: {recovery}]  "
              f"(按 m 切换: {' | '.join(recovery_keys)})")
        for k, (_, label) in PLOTS.items():
            print(f"  {k:>2}. {label}")
        print("   q. 退出")

        try:
            tag = input("  > ").strip().lower()
        except KeyboardInterrupt:
            # Ctrl-C 曾裸 traceback (退出码非零), 与 EOF 分支不对称 —
            # 求解已成功, 优雅退出
            print("\n  [INFO] 已退出交互绘图 (Ctrl-C)")
            plt.close('all')
            return
        if tag in ('q', 'quit', 'exit'):
            plt.close('all'); break
        if tag == 'm':
            idx = recovery_keys.index(recovery)
            recovery = recovery_keys[(idx + 1) % len(recovery_keys)]
            plt.close('all')
            continue
        if tag not in PLOTS:
            if tag: print(f"    ? {tag}")
            continue
        tag = PLOTS[tag][0]
        plot_three(mesh, result, tag=tag, scale=scale,
                  isoband_levels=isoband_levels, isoband_tag=isoband_tag,
                  sigma_ref=sigma_ref, recovery=recovery)


