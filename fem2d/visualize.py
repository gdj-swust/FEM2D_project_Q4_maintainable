"""可视化 — 交互式选择云图"""
import math
import queue
import threading
import unicodedata

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize

from .cli import ask
from .element import evaluate_vector_field
from .error_est import (  # 唯一公式, 供 visualize 和 error_est 共用
    compute_traction_jumps,
    element_refinement_indicator,
)
from .errors import CliError
from .material import von_mises
from .mesh import Mesh
from .spr import spr_recovery
from .stress import (
    nodal_L2_projection,
    nodal_simple,
    nodal_weighted,
    point_in_element,
    principal_stresses,
    stress_probe,
)

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


# SPR 恢复备忘 — 一次切图同 (mesh, 应力数组) 组合要重算 3 遍
# (plot_three 直接算 1 次 + 两个 gouraud 面板各 1 次), 粗网格 0.09s/次,
# 细网格秒级; 值持有强引用 (mesh/data) 防 GC 后 id 复用错配; 满则全清 —
# 会话内网格数量有限。键按 id() 匹配对象身份, 另存内容指纹 (抽样字节
# 哈希) 防陈旧: 用户对 result['stress'] 就地单位换算 (*=1e-6 常见流程)
# 后再画, 身份不变内容变 — 只靠 id() 会返回过期恢复场, 而 isoband/
# eta_K 面板无缓存用新值, 四面板互相矛盾且无告警 (强化扫描实测复现)
_TO_NODE_CACHE = {}
_TO_NODE_CACHE_MAX = 8


def _array_fingerprint(arr):
    """数组内容指纹 — 备忘键的防陈旧校验 (就地修改会换指纹).

    首末各 512 元素的字节抽样 + shape/dtype: 命中时先比对, 不符即
    重算覆盖 — 抽样开销微秒级, 相对 SPR 恢复 (粗网格 0.09s) 可忽略。
    hash() 的进程级盐值随机化无妨 — 缓存本就是会话内数据结构。
    """
    arr = np.asarray(arr)
    flat = arr.ravel()
    if flat.size <= 1024:
        sample = flat.tobytes()
    else:
        sample = flat[:512].tobytes() + flat[-512:].tobytes()
    return (arr.shape, arr.dtype.str, hash(sample))


def _to_node(mesh, ev, integration_values=None, method="SPR"):
    """应力恢复 — 可切换方法

    method: "SPR" | "weighted" | "L2" | "simple"
    """
    data = integration_values if integration_values is not None else ev
    if method == "SPR":
        key = (id(mesh), id(data))
        fp = _array_fingerprint(data)
        entry = _TO_NODE_CACHE.get(key)
        if entry is None or entry[3] != fp:
            result = spr_recovery(mesh, data)
            if len(_TO_NODE_CACHE) >= _TO_NODE_CACHE_MAX:
                _TO_NODE_CACHE.clear()
            _TO_NODE_CACHE[key] = (mesh, data, result, fp)  # 强引用防 id 复用
        else:
            result = entry[2]
    elif method == "L2":
        # L2 投影直接消费积分点数据 (按内核积分规则加权)。
        # 先算术平均会丢掉积分点分布信息 — 云图与正确 L2 投影相差 ~30%。
        result = nodal_L2_projection(mesh, data)
    else:
        # weighted / simple: 接受 (ne, ncomp) 或 (ne, nqp, ncomp);
        # 单元代表值: weighted 取 dA 加权平均 (与 solver 的 stress 一致),
        # simple 取算术平均 (对 simple 做 dA 加权时, 两种方法在歪扭
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
                        # 形状不符 = 内核恢复点与积分点不一致 — 回退算术
                        # 平均是可用解, 但必须响亮 (静默替换恢复方法会让
                        # 学生误以为 weighted 生效)
                        print(
                            f"  [WARN] {method} 恢复权重形状 "
                            f"{weights.shape} 与积分点数据 "
                            f"{data.shape[:2]} 不符 — 回退算术平均")
                        data = data.mean(axis=1)
                else:
                    data = data.mean(axis=1)
        result = _RECOVERY_METHODS[method][1](mesh, data)
    return result[:, 0] if result.shape[1] == 1 else result


# ═══════════════════════════════════════════════════════════════
# Bathe §4.3.6 Eq (4.107): 内部边牵引跳跃
# compute_traction_jumps 从 error_est 导入 — 唯一公式, 避免重复实现
# ═══════════════════════════════════════════════════════════════

# 牵引跳跃段备忘 — 切分量时四面板的 (mesh, 应力) 组合不变, 跳跃计算 +
# 4 万段 python 循环构建 (~0.1s 粗网格) 每次切图重算纯属浪费; 强引用
# 防 GC 后 id 复用错配, 满则全清 (同 _TO_NODE_CACHE 口径); 命中先比对
# 内容指纹 — 就地改应力数组会换指纹重算
_TRACTION_JUMP_CACHE = {}
_TRACTION_JUMP_CACHE_MAX = 8


def _p98_vmax(values):
    """P98 色标上限 + 标签 — 常场回退基于场尺度 (1e-30 绝对下限会让
    微尺度场色标失真)。plot_traction_jumps 与 scalar_jump 两处共用,
    改分位数只改一处。
    """
    p98 = float(np.percentile(values, 98))
    if p98 < 1e-30:
        return 1.0, 'zero-field fallback'
    return p98, 'P98'


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

    key = (id(mesh), id(elem_stress), sigma_ref)
    fp = _array_fingerprint(elem_stress)
    entry = _TRACTION_JUMP_CACHE.get(key)
    if entry is None or entry[3] != fp:
        jumps = compute_traction_jumps(mesh, elem_stress, sigma_ref=sigma_ref)
        segments, values = [], []
        nodes = mesh.nodes
        for j in jumps:
            xa, ya = float(nodes[j['node_a'], 0]), float(nodes[j['node_a'], 1])
            xb, yb = float(nodes[j['node_b'], 0]), float(nodes[j['node_b'], 1])
            segments.append([(xa, ya), (xb, yb)])
            values.append(float(j['jump_rel']))
        entry = (mesh, elem_stress,
                 (segments, np.asarray(values, dtype=np.float64).ravel()), fp)
        if len(_TRACTION_JUMP_CACHE) >= _TRACTION_JUMP_CACHE_MAX:
            _TRACTION_JUMP_CACHE.clear()
        _TRACTION_JUMP_CACHE[key] = entry
    segments, values_arr = entry[2]

    if not values_arr.size:
        ax.set_title("(no internal edges)")
        return None

    actual_max = float(values_arr.max())

    auto_vmax = vmax is None
    if auto_vmax:
        vmax, vmax_label = _p98_vmax(values_arr)
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
    # 仅零跨度退化模型才回退 — 1e-15 下限会使微尺度模型 (span 1e-16)
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
        # 1.0 下限会使微尺度载荷 (t_max=1e-20) 的箭头缩放失真, 全部
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
            # 1e-15 绝对阈值会让微尺度模型的归一化箭头 (长度 ~8%·span)
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
        # 1.0 下限会使微尺度/非 SI 单位体力 (f_max=7.86e-4) 箭头缩短
        # 1/f_max 倍静默不可见 — 与面力分支 223 行统一
        b_max = max(max(bx_all), max(by_all), np.finfo(float).tiny)
        arrow_scale_b = span * 0.05 / b_max

        for eid in range(0, n_elem, step):
            xc, yc = mesh.centroids[eid]
            bx, by = evaluate_vector_field(bf, xc, yc)
            dx, dy = bx * arrow_scale_b, by * arrow_scale_b
            # 1e-15 绝对阈值会让微尺度模型的归一化箭头 (长度 ~8%·span)
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
            # 1e-15 绝对阈值会让微尺度模型的归一化箭头 (长度 ~8%·span)
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
    """色条刻度格式化 — 大/小量级直接原始值科学计数 (isoband 与统一色条共用).

    title=None 时不改动 label (isoband 已自行设置含 band 宽度的 label).
    """
    if title is not None:
        cbar.set_label(title)
    tick_vals = np.linspace(e_min, e_max, n_ticks)
    rng = e_max - e_min if e_max > e_min else 1.0
    if rng > 1e3 or rng < 1e-3:
        # 原始值科学计数 ("1.2e+07") — 曾压缩成 "12.00e6" 式缩写, 刻度
        # 只剩 12/11/9 一类数字, 用户反馈要求直接显示原始值
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
    # 常场回退基于场尺度 — 1e-30 绝对下限会使微尺度场 (1e-40) 的
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
        jvmax, jvmax_label = _p98_vmax(jvals)
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


def _pad_near_constant_range(vmin, vmax):
    """近常场 padding — 基于场自身尺度, 绝对 1e-15 阈值 + 1.0 pad 会让
    微尺度场 (跨度 1e-16 绝对单位) 色标变 [~1e-12, 1.0] 单色塌缩。
    gouraud 与 isoband 两处共用同一公式 (scalar_jump 的 sigma_range
    回退是归一化分母, 用途不同不合并)。
    """
    scl = max(abs(vmin), abs(vmax), np.finfo(float).tiny)
    if abs(vmax - vmin) <= 1e-14 * scl:
        pad = scl * 1e-6
        return vmin - pad, vmax + pad
    return vmin, vmax


def _plot_isoband_contour(ax, tri, display_parent, values, title,
                          n, levels, e_min, e_max):
    """Bathe §4.3.6 等应力带法 (Sussman & Bathe) — 固定应力区间离散着色.

    自行处理色条后返回 None (调用方跳过统一色条逻辑).
    """
    if levels is not None:
        band_levels = _validate_isoband_levels(levels)
        n_bands = len(band_levels) - 1
    else:
        # 常应力保护: e_min≈e_max 时加 padding (与 gouraud 同公式 —
        # _pad_near_constant_range, 阈值/padding 都基于场本身尺度)
        b_min, b_max = _pad_near_constant_range(e_min, e_max)
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


def _plot_gouraud_contour(ax, tri, values, mesh, n, recovery, is_element=None):
    """Gouraud: SPR 磨平到节点 → 光滑云图 + 等值线 (推荐用于报告).

    is_element: 数据定位 (plot_contour 已解析显式 location 参数);
    None → 按长度推断 (兼容直接调用方)。
    """
    if is_element is None:
        is_element = len(values) == mesh.n_elements
    if is_element:
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
    # 近常场 padding 与 isoband 同公式 (_pad_near_constant_range)
    vmin, vmax = _pad_near_constant_range(vmin, vmax)
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
            ax, tri, values, mesh, n, recovery, is_element=is_element)

    # ── 统一色条 ──
    if shading != 'scalar_jump':  # scalar_jump 有自己的边色条
        # 标注数据来源与峰值: CST 磨平后峰值降低 ~17% 的教训 — 色条必须
        # 让学生一眼看出显示的是单元值 (flat, 原始峰值) 还是节点值
        # (gouraud, 磨平峰值); 与 isoband 的 [bands: ...] 风格对齐
        source = "element" if is_element else "node"
        label = (f"{title}  [{source}, max={e_max:.3g}]"
                 if title else f"{source}, max={e_max:.3g}")
        cbar = plt.colorbar(tpc, ax=ax, shrink=0.8, label=label)
        _style_colorbar(cbar, e_min, e_max, None)

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

# plot_three 合法 tag 集 — 直接派生自 PLOTS (单一事实源), 12 个:
# mesh/ux/uy/umag/sx/sy/txy/vm/s1/s2/taumax/loads
_VALID_TAGS = {v[0] for v in PLOTS.values()}

def plot_three(mesh, result, tag='vm', scale=100, save=None,
               isoband_levels=None, isoband_tag=None, sigma_ref=None,
               recovery="SPR"):
    """Bathe §4.3.6 三连图: Gouraud / Isoband / Traction Jump 并排对比

    recovery: "SPR" | "weighted" | "L2" | "simple" — 左上 Gouraud 图的应力恢复方法

    isoband_levels: 固定应力区间, 如 np.arange(0, 40e6, 2.5e6)
    isoband_tag: 固定带宽仅当 tag==isoband_tag 时生效 (避免 Mises 范围误用于 S11)
                 为 None 时所有分量均使用固定带宽
    """
    # tag 白名单: 非法分量名曾从 g_vals[tag]/f_vals[tag] 冒裸 KeyError
    if tag not in _VALID_TAGS:
        raise ValueError(
            "tag 必须为合法分量名之一: "
            f"{sorted(_VALID_TAGS)} — 得到 {tag!r}")
    u = result["u"]; u2 = u.reshape(-1,2)
    u_mag = np.hypot(u2[:,0], u2[:,1])  # 同 reporting: 防 |u|~1e308 平方溢出
    s = result["stress"]  # (n_elem, 3) — [σ_x, σ_y, τ_xy]

    is_stress = tag not in ("ux", "uy", "umag", "mesh", "loads")
    if is_stress:
        # ── 应力恢复 → 从恢复分量计算 Mises/主应力 ──
        # 仅应力云图需要 — 只画网格/载荷/位移时不预跑 (大网格无效开销)。
        s_node = _to_node(mesh, s, result.get("stress_qp"), method=recovery)  # (n_nodes, 3)
        if recovery == "SPR":
            # 与 stress_at_point 的 recovered 探针同源同算 — 共享缓存,
            # 首次点击读值不再付 SPR 预热 (粗网格 0.09s/细网格秒级)
            result["_spr_cache"] = s_node
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
        # axes[2] 也是 show_loads=True 的重复图 — 无载荷 vs 含载荷
        # 两张对比已覆盖全部信息, 第三格置空
        axes[2].axis('off')
    elif tag == "loads":
        plot_mesh(mesh, ax=axes[0], show_loads=True)
        axes[0].set_title('All loads', fontsize=12)
        # 三张图内容完全相同, 仅标题不同 — 只留一张
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

        eta_K = element_refinement_indicator(mesh, result)
        plot_contour(mesh, eta_K, r'$\eta_K$ (residual-based estimator) [Pa·m]',
                    ax=axes[1,1], shading='flat', location='element')
        axes[1,1].set_title(r'$\eta_K$ — residual error indicator (Verfürth 1996)', fontsize=10)

    fig.suptitle(f'Bathe §4.3.6 — {titles.get(tag, tag)}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches='tight')
        print(f'  → {save}')
        # 批量保存后主动关闭 — 长期不 close, 参数扫描逐次累积内存
        #
        plt.close(fig)
        return
    if plt.get_backend() != 'Agg':
        plt.show(block=False)
    else:
        # Agg (CI/测试/无显示器): show 无效 — 不 close 则每次调用静默
        # 累积一张图, 交互循环按 1-12 一键一张
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# 状态栏读值 — interactive_plot 右下角显示鼠标左键点击处当前分量点值
# (点击触发而非 motion — motion 持续定位对 Python 压力大, 用户钦定)
# 点击同时圈出命中单元的外轮廓 — 选中哪个单元一目了然 (用户钦定)
# ═══════════════════════════════════════════════════════════════

def _readout_row_element(mesh, result, x, y):
    """单元代表应力 (线性单元 ≈ 质心) — 跨单元跳变, 对应 flat/isoband 云图.

    stress_probe 顺带算 recovered 行: 首次触发预热 SPR 缓存 (一次性
    开销, 与 plot_three 的 Gouraud 恢复同量级), 之后每次点击只是
    两次桶定位 + 局部插值 — ElementLocator 桶索引保证 O(1) 候选。
    """
    return stress_probe(mesh, result, x, y)[0]


def _readout_row_recovered(mesh, result, x, y):
    """SPR 恢复场插值 — 逐点连续, 对应 Gouraud 云图.

    口径恒为 SPR: m 键把恢复方法切到 weighted/L2/simple 后, Gouraud
    面板换场, 本行仍读 SPR 场 (与云图不再同源) — 逐点对比时需把恢复
    方法切回 SPR。恒 SPR 是刻意设计: 读值口径须可复现, SPR 是默认
    推荐恢复方法。
    """
    return stress_probe(mesh, result, x, y)[1]


# 读值口径插件注册表 — 新口径在此注册一行 (键 → 显示名 + 6 值应力行
# 取数函数); 应力行装配单一来源 fem2d.stress.stress_probe, 派生量
# (主应力/von Mises) 只在那处计算。interactive_plot 的 v 键在此注册
# 表上循环切换, 默认 element (用户要求: 显示质心好)。
_READOUT_MODES = {
    "element":   ("单元代表应力 (≈质心)", _readout_row_element),
    "recovered": ("SPR 恢复场插值",       _readout_row_recovered),
}
_DEFAULT_READOUT = "element"

# 云图分量 → 6 值应力行下标 (taumax 由行内 s1/s2 组合 — 与 plot_three
# 的 principal_stresses 口径一致, τ_max = (σ1−σ2)/2)
_READOUT_ROW_INDEX = {"sx": 0, "sy": 1, "txy": 2, "s1": 3, "s2": 4, "vm": 5}

# 位移分量 → u2 列 (umag = hypot(ux, uy))
_READOUT_DISP_INDEX = {"ux": 0, "uy": 1}

# 状态栏短名 — PLOTS 的 label 过长 (状态栏一行放不下), 短名与 PLOTS
# 分量一一对应; 改 PLOTS 分量名时需同步此处
_READOUT_NAMES = {
    "sx": "σ_x", "sy": "σ_y", "txy": "τ_xy", "vm": "σ_vm",
    "s1": "σ_1", "s2": "σ_2", "taumax": "τ_max",
    "ux": "u_x", "uy": "u_y", "umag": "|u|",
}


def _disp_at_point(mesh, result, x, y):
    """光标处位移 (ux, uy) — 形函数插值, 与位移云图同源.

    evaluate_vector_field 只服务载荷场校验, 位移点值须走形函数模式
    (桶定位 + shape @ u[conn]); 返回 None 表示点不在网格。
    """
    eid = point_in_element(mesh, x, y)
    if eid < 0:
        return None
    conn = mesh.elements[eid]
    shape = mesh.element_kernel.shape_values_at(mesh.nodes[conn], x, y)
    if shape is None:
        return None
    u2 = result["u"].reshape(-1, 2)
    return tuple(shape @ u2[conn])


def _readout_line(mesh, result, x, y, tag, readout):
    """状态栏读值文本 — 纯函数, 点击回调只做薄壳 (可直接单测).

    tag: 当前云图分量 (PLOTS 值列); readout: _READOUT_MODES 键。
    应力分量走 _READOUT_MODES 插件口径; 位移分量走形函数插值 (位移场
    本是节点量, 插值即场值, 无口径之分); mesh/loads 无点值 → ""
    (回调据此清空); 点不在网格 → "（模型外）" (stress_at_point /
    point_in_element 的 ValueError 收敛 — 光标在空白区/孔内是常态,
    不是错误)。
    """
    if tag in (None, "mesh", "loads"):
        return ""
    try:
        if tag in _READOUT_ROW_INDEX or tag == "taumax":
            # taumax 无行下标 (s1/s2 的组合量), 借同一分支
            row = _READOUT_MODES[readout][1](mesh, result, x, y)
            if tag == "taumax":
                value = 0.5 * (row[3] - row[4])
            else:
                value = row[_READOUT_ROW_INDEX[tag]]
            return (f"{_READOUT_NAMES[tag]} = {value:.4e}  "
                    f"[{_READOUT_MODES[readout][0]}]")
        if tag in ("ux", "uy", "umag"):
            uv = _disp_at_point(mesh, result, x, y)
            if uv is None:
                return "（模型外）"
            if tag == "umag":
                value = math.hypot(uv[0], uv[1])
            else:
                value = uv[_READOUT_DISP_INDEX[tag]]
            return f"{_READOUT_NAMES[tag]} = {value:.4e}  [形函数插值]"
    except ValueError:
        return "（模型外）"
    return ""


def _parse_coord(raw):
    """坐标输入解析 — 返回 (x, y) 或 None (非坐标输入).

    NFKC 归一覆盖中文输入法的全角括号/逗号/数字 ("（０.10，０.15）"
    亦可), 归一后的包裹括号剥离; NaN/Inf 视为非法 (当坐标传下去只会
    得到 "not in mesh", 格式提示更友好).
    """
    parts = (unicodedata.normalize("NFKC", raw)
             .replace(",", " ")
             .strip("()[]{}")
             .split())
    if len(parts) < 2:
        return None
    try:
        x, y = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def interactive_plot(mesh, result, scale=100, isoband_levels=None, isoband_tag=None, sigma_ref=None):
    """轻量交互 — 终端输入分量名, 图里选着色模式

    按键:
      1-12  选择分量
      m     切换应力恢复方法 (SPR → weighted → L2 → simple → SPR ...)
      p     探针: 输入坐标读该点应力
      c     点击探针开关 (开: 点击图上点读应力)
      v     切换状态栏读值口径 (element 单元代表应力≈质心 ↔ recovered SPR)
      q     退出
    鼠标:
      点击探针开启后点击图上任意点 → 终端打印该点应力 (element 单元
      代表值 + recovered SPR 恢复点值), 对应 Abaqus Probe Values 对点
      比较; 左键点击图上任意点 → 右下角显示当前分量在该点的值 (口径
      见 v, 默认 element ≈ 质心, 停留到下次点击), 同时每个面板用红色
      闭圈圈出点击所在单元的外轮廓 (点外/孔内点击清除高亮); 菜单输入
      期间 GUI 事件持续泵送, 状态栏光标坐标同步刷新 (input() 阻塞主
      线程会让图冻死 — 有图时改后台线程读输入)
    """
    recovery = _DEFAULT_RECOVERY
    recovery_keys = list(_RECOVERY_METHODS.keys())
    readout = _DEFAULT_READOUT
    readout_keys = list(_READOUT_MODES.keys())
    click_probe = False
    # 当前云图分量 (PLOTS 值列) — 状态栏读值按它选分量; 首张图之前
    # 为 None, _readout_line 对 None 返回空串
    current_tag = None

    # 探针打印 — 点击回调与键盘 p 键共用 (element 单元代表值 +
    # recovered SPR 恢复点值), 对应 Abaqus Probe Values 对点比较
    def _probe_at(x, y):
        try:
            e_row, r_row = stress_probe(mesh, result, x, y)
        except ValueError as err:
            print(f"\n  [probe] ({x:.4f}, {y:.4f}): {err} — "
                  "变形图坐标为原坐标+scale·u, 小变形下可直接按点击坐标读")
            return
        print(f"\n  [probe] ({x:.4f}, {y:.4f})")
        sx, sy, txy, s1, s2, vm = e_row
        print(f"    element  : sx={sx:.4e} sy={sy:.4e} txy={txy:.4e}"
              f"  s1={s1:.4e} s2={s2:.4e} vm={vm:.4e}")
        sx, sy, txy, s1, s2, vm = r_row
        print(f"    recovered: sx={sx:.4e} sy={sy:.4e} txy={txy:.4e}"
              f"  s1={s1:.4e} s2={s2:.4e} vm={vm:.4e}")

    # 点击探针 — 每张新图挂一次 (切图 plt.close('all') 重建, 旧连接随
    # 旧 canvas 销毁不累积); c 键关闭时点击只走工具栏语义
    def _probe_on_click(event):
        if not click_probe or event.inaxes is None or event.xdata is None:
            return
        bar = getattr(plt.gcf().canvas, "toolbar", None)
        if bar is not None and bar.mode:
            return  # 缩放/平移拖拽的起止点击不探针 (工具栏模式非空)
        _probe_at(float(event.xdata), float(event.ydata))

    # 右下角读值 — 每张新图挂一次 (与点击探针同生命周期)。读值口径
    # 由 v 键改 readout 闭包变量, 下一次点击立即生效 (无需重画)
    def _attach_readout(fig):
        """figure 右下角文本 + 单元外轮廓高亮 + 左键点击回调 (blit 局部刷新).

        点击触发而非 motion — motion 下 Python 要持续定位插值, 压力
        大; 点击命中某单元后每个面板用红色闭圈圈出其外轮廓 (点外/孔内
        点击清除), 文本+高亮更新只恢复画布背景 + 重画 + blit, 不触发
        整张 4 面板图全量重绘。
        """
        # family 必须 sans-serif 而非 monospace: 读值行含 CJK 标签
        # ("单元代表应力 (≈质心)" 等), font.monospace 列表无任何 CJK
        # 字体且 matplotlib 无跨族回退 — 字形缺失渲染成方框 + 每帧
        # missing glyph 警告 (已实测复现); 模块头部的中文字体修复只改
        # font.sans-serif, monospace 绕过了它
        text = fig.text(0.99, 0.01, "", ha="right", va="bottom",
                        fontsize=9, family="sans-serif",
                        transform=fig.transFigure,
                        bbox=dict(facecolor="white", edgecolor="gray",
                                  alpha=0.8, boxstyle="round,pad=0.3"),
                        animated=True)  # 全量重绘跳过文本 → blit 背景
                                        # 不含旧文本, 无残影

        # ── 单元外轮廓高亮 — 每面板一条 LineCollection ──
        # animated=True 使全量重绘跳过它 (与文本同理) — 旧高亮不会烘焙
        # 进背景; 空段 = 隐藏; label 供测试定位 (面板间同一元素轮廓)
        hl = []
        u2 = result["u"].reshape(-1, 2)
        # 逆映射定位网格 — 点击坐标是面板上的变形位置 x + scale·u(x),
        # scale·u 超过网格尺度时直接查未变形网格必判域外 (AABB 拒绝);
        # 与面板绘制同口径 (nodes + scale·u2), 每张图只建一次
        def_mesh = Mesh(nodes=mesh.nodes + scale * u2,
                        elements=mesh.elements, elem_type=mesh.elem_type)

        def _deformed(ax):
            # plot_three 布局: ux/uy/umag 三连图为 [未变形, 变形, 变形+网格],
            # 应力 2×2 四图全未变形 — 变形面板的轮廓须用 x+scale·u, 否则
            # 偏移 scale·u 与云图错位 (与点击探针的小变形近似同口径)
            return (current_tag in ("ux", "uy", "umag")
                    and ax is not fig.axes[0])

        for ax in fig.axes:
            lc = LineCollection([], colors="#e53935", linewidths=2.0,
                                zorder=100, animated=True,
                                label="_readout_hl")
            ax.add_collection(lc)
            hl.append((ax, lc))

        def _elem_outline(eid, ax):
            """单元 eid 的外轮廓闭圈折线 (面板自身坐标系)."""
            conn = mesh.elements[eid]
            pts = mesh.nodes[conn]
            if _deformed(ax):
                pts = pts + scale * u2[conn]
            return np.vstack([pts, pts[0]])

        def _undef_xy(x, y):
            """变形面板点击坐标 → 未变形坐标 (不动点逆映射).

            面板绘制 x_d = x_u + scale·u(x_u) — 解 x_u = x_d − scale·u(x_u):
            ① 在变形网格 def_mesh 上定位 x_d (点击坐标是变形位置, 直接
               查未变形网格在 scale·u > 网格尺度时恒域外); 等参形函数
               同参 — 该参数点上 u 的节点场插值恰为 u(x_u), 一轮即达
               精确不动点 (任意插值场; 旧实现固定 2 轮增量式
               x -= scale·u 对常位移场第二轮继续减 scale·u 越减越远,
               且首轮就查未变形网格 → 变形面板点击恒报"模型外");
            ② 第二轮纯防御 (自穿透/退化形变), 常态下 x_u 落在变形网格
               AABB 之外直接 break; 中途失败返回当前近似, 由下游
               "（模型外）" 收敛。
            """
            x_d, y_d = x, y
            for _ in range(2):
                eid = point_in_element(def_mesh, x, y)
                if eid < 0:
                    break
                conn = mesh.elements[eid]
                shape = mesh.element_kernel.shape_values_at(
                    def_mesh.nodes[conn], x, y)
                if shape is None:
                    break
                u_at = shape @ u2[conn]
                x = x_d - scale * float(u_at[0])
                y = y_d - scale * float(u_at[1])
            return x, y

        last = {"txt": "", "eid": -1}
        bg = [None]  # 全量重绘后的画布背景 (draw_event 捕获)

        def _refresh():
            # 背景未捕获 (首帧) → 全量重画一次, 由 draw_event 抓背景;
            # 之后只 restore + 重画高亮/文本框 + blit, 不触碰 4 面板图
            if bg[0] is None:
                fig.canvas.draw_idle()
                return
            fig.canvas.restore_region(bg[0])
            for _, lc in hl:
                lc.draw(fig.canvas.get_renderer())
            text.draw(fig.canvas.get_renderer())
            fig.canvas.blit(fig.bbox)

        def _on_draw(_event):
            # 每次全量重绘 (缩放/平移/切图) 后背景失效 — 重新捕获并补画
            # 文本+高亮 (animated=True 使全量重绘跳过它们, 不补画会一直
            # 消失到下次点击)
            bg[0] = fig.canvas.copy_from_bbox(fig.bbox)
            if last["txt"] or last["eid"] >= 0:
                for _, lc in hl:
                    lc.draw(fig.canvas.get_renderer())
                text.draw(fig.canvas.get_renderer())
                fig.canvas.blit(fig.bbox)

        def _on_click(event):
            # 只读左键 (右键缩放/滚轮不触发); 工具栏缩放/平移模式点击
            # 不读值 (与点击探针同守卫)
            if event.button != 1 or event.inaxes is None or event.xdata is None:
                return
            bar = getattr(fig.canvas, "toolbar", None)
            if bar is not None and bar.mode:
                return
            x, y = float(event.xdata), float(event.ydata)
            if _deformed(event.inaxes):
                try:
                    x, y = _undef_xy(x, y)
                except ValueError:
                    pass  # 非有限坐标 → 保留原始, 下游按域外收敛
            line = _readout_line(mesh, result, x, y, current_tag, readout)
            try:
                eid = point_in_element(mesh, x, y)
            except ValueError:
                eid = -1  # 非有限坐标当域外 (契约异常, 回调不炸)
            if line != last["txt"] or eid != last["eid"]:
                for ax, lc in hl:
                    lc.set_segments([_elem_outline(eid, ax)] if eid >= 0 else [])
                text.set_text(line)
                last["txt"], last["eid"] = line, eid
                _refresh()

        fig.canvas.mpl_connect("draw_event", _on_draw)
        fig.canvas.mpl_connect("button_press_event", _on_click)

    def _ask_catch(prompt):
        """ask 一行 — Ctrl-C 转 CliError 时收敛为 None (主线程统一收尾)."""
        try:
            return ask(prompt).strip()
        except CliError:
            return None

    def _ask_or_quit(prompt):
        """读一行输入 — Ctrl-C 收敛为 None, 由调用方 _quit_interactive() 收尾.

        有图时后台线程读 + 主线程泵 GUI 事件 (flush_events — 只处理已
        排队事件, 不触发重绘): 点击回调/状态栏光标坐标只在事件循环运转
        时更新, 终端 input() 阻塞主线程会让图冻死 (点击"鼠标转圈后无
        反应"即此因); 无图 (测试/无显示) 退化为直接阻塞读, 交互测试的
        builtins.input 打桩路径不变.
        """
        if not plt.get_fignums():
            return _ask_catch(prompt)
        lines = queue.Queue()
        threading.Thread(target=lambda: lines.put(_ask_catch(prompt)),
                         daemon=True).start()
        try:
            while True:
                try:
                    return lines.get(timeout=0.1)
                except queue.Empty:
                    if not plt.get_fignums():
                        # 用户已关完所有图 — gcf() 会凭空建一张空白图,
                        # 退化为阻塞读
                        return lines.get()
                    # 纯事件泵送: 只处理已排队的 GUI 事件, 不触发任何
                    # 重绘 — 曾用 plt.pause(0.05), 其内部每轮调
                    # show(block=False) → FigureManagerTk.show →
                    # canvas.draw_idle(), 图窗每 50ms 全量重绘一次
                    # 4 面板 Gouraud 图 (blit 局部刷新也救不了 — 重绘
                    # 由输入泵驱动, 与读值无关), 窗口永远卡顿
                    try:
                        plt.gcf().canvas.flush_events()
                    except Exception:
                        pass  # 图刚被关的竞态 — 下一轮 get_fignums 收敛
        except KeyboardInterrupt:
            return None

    def _quit_interactive():
        print("\n  [INFO] 已退出交互绘图 (Ctrl-C)")
        plt.close('all')

    while True:
        print(f"\n  [恢复方法: {recovery}]  "
              f"(按 m 切换: {' | '.join(recovery_keys)})")
        for k, (_, label) in PLOTS.items():
            print(f"  {k:>2}. {label}")
        print("   p. 探针: 输入坐标读该点应力 (如 0.72, 0.30)")
        print(f"   c. 点击探针: [{'开' if click_probe else '关'}] 点击图上点读应力")
        print(f"   v. 状态栏读值口径 (当前: {_READOUT_MODES[readout][0]})")
        print("   q. 退出   |   主菜单直接输入坐标也可探针")

        tag = _ask_or_quit("  > ")
        if tag is None:
            _quit_interactive()
            return
        tag = tag.lower()
        if not tag:
            # EOF (stdin 关闭 → ask 返回 "") 视为退出键 — 裸 input() 曾
            # 抛 EOFError 泄漏 traceback
            break
        if tag in ('q', 'quit', 'exit'):
            plt.close('all'); break
        if tag == 'm':
            idx = recovery_keys.index(recovery)
            recovery = recovery_keys[(idx + 1) % len(recovery_keys)]
            plt.close('all')
            continue
        if tag == 'c':
            click_probe = not click_probe
            print(f"    [probe] 点击探针已{'开启' if click_probe else '关闭'}"
                  f" — {'点击图上点读该点应力' if click_probe else '点击不再读应力'}")
            continue
        if tag == 'v':
            readout = readout_keys[
                (readout_keys.index(readout) + 1) % len(readout_keys)]
            print(f"    [readout] 状态栏读值口径 → {_READOUT_MODES[readout][0]}")
            continue
        if tag == 'p':
            raw = _ask_or_quit("  坐标 x, y > ")
            if raw is None:
                _quit_interactive()
                return
            coord = _parse_coord(raw)
            if coord is None:
                print("    ? 坐标格式: 如 0.72, 0.30")
                continue
            _probe_at(*coord)
            continue
        if tag not in PLOTS:
            # 主菜单直接输入坐标也当探针 (用户常不按 p 直接敲 "(x,y)";
            # 全角括号/逗号由 _parse_coord 归一) — 与菜单键无歧义:
            # 键都是 1-12/m/c/p/q, 坐标必然含至少两个数
            coord = _parse_coord(tag)
            if coord is not None:
                _probe_at(*coord)
                continue
            if tag: print(f"    ? {tag}")
            continue
        tag = PLOTS[tag][0]
        # 切换分量前关闭上一张图 — 每按一键累积一张 2×2 图 (内存泄漏)
        plt.close('all')
        plot_three(mesh, result, tag=tag, scale=scale,
                  isoband_levels=isoband_levels, isoband_tag=isoband_tag,
                  sigma_ref=sigma_ref, recovery=recovery)
        current_tag = tag
        if plt.get_fignums():
            # Agg/测试下 plot_three 自行 close → 无图时 gcf() 会凭空
            # 建一张空白图 (曾每次绘图静默多一张)
            fig = plt.gcf()
            fig.canvas.mpl_connect("button_press_event", _probe_on_click)
            _attach_readout(fig)


