"""外部 Gmsh 网格的收敛率验证 — Bathe §4.3.5.

本脚本保留给已有 ``test_spec.geo`` 工作流。内置、可重复的 CST/Q4
悬臂梁验证请优先运行 ``python -m fem2d.convergence``。
"""
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 导入 FEM2D 模块 ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fem2d import Mesh, solve
from fem2d import estimate_error as estimate

# ── 参数 ──
GEO_TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'test_spec.geo')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'convergence')

LC_VALUES = [0.08, 0.04, 0.02, 0.01]  # 网格逐级加密2×
E, nu, t = 2.10e11, 0.3, 0.01
BY = -78000.0

def run_gmsh(lc):
    """生成指定 lc 的 .geo 并运行 Gmsh"""
    import re
    # 读模板 .geo
    geo = os.path.join(OUT_DIR, f'test_lc{lc:.4f}.geo')
    with open(GEO_TEMPLATE, 'r', encoding='utf-8') as f:
        content = f.read()
    # 改 lc
    content = re.sub(r'^lc\s*=\s*[\d.]+;', f'lc = {lc};', content, flags=re.MULTILINE)
    # 输出由 gmsh_runner 统一管理 (原生 .msh)；移除模板内的显式 Save，
    # 避免绕过临时文件验证与发布步骤。
    msh = os.path.join(OUT_DIR, f'test_lc{lc:.4f}.msh')
    content = re.sub(r'^\s*Save\s+".*?"\s*;\s*$', '', content, flags=re.MULTILINE)
    with open(geo, 'w', encoding='utf-8') as f:
        f.write(content)

    # 运行 Gmsh — 走 generate_geo_with_topology: 临时生成 → 验证 → 原子发布
    # (评审: 与主 CLI 同一条安全发布链, 避免直接发布未验证网格)
    print(f'  [Gmsh] lc={lc} ...', end=' ', flush=True)
    t0 = time.time()
    from fem2d.input_source import generate_geo_with_topology
    generated, _import = generate_geo_with_topology(
        geo, output_path=msh, plane_type="stress")
    if not generated:
        raise RuntimeError(f'Gmsh failed for lc={lc}')
    print(f'{time.time()-t0:.1f}s')
    return msh

def run_fem(msh_path):
    """运行 FEM2D 求解，返回结果字典 (2026-08: .msh 经 Gmsh API 导入)"""
    print(f'  [FEM] {os.path.basename(msh_path)} ...', end=' ', flush=True)
    t0 = time.time()

    from fem2d.gmsh_adapter import import_msh
    g = import_msh(msh_path)
    coords, elems, elem_type = g.nodes, g.elements, g.elem_type
    plane_type = (
        'strain' if str(elem_type).upper().startswith('CPE') else 'stress')
    mesh = Mesh(nodes=coords, elements=elems, E=E, nu=nu,
                thickness=t, plane_type=plane_type, elem_type=elem_type)

    # 左边界固定 (x ≈ x_min)
    x_min = coords[:, 0].min()
    left_nodes = np.where(np.abs(coords[:, 0] - x_min) < 1e-6)[0]
    for n in left_nodes:
        mesh.fix_node(int(n), 'x', 0.0)
        mesh.fix_node(int(n), 'y', 0.0)
    mesh.body_force = (0.0, BY)
    result = solve(mesh, method='elimination')

    # 误差估计
    eta_info = estimate(mesh, result)

    elapsed = time.time() - t0
    # 平均单元尺寸 = sqrt(avg area)，几何由当前 kernel 统一提供。
    mesh.build_connectivity()
    h_mean = np.sqrt(np.mean(mesh.areas))

    stats = {
        'lc': float(os.path.basename(msh_path).split('lc')[1].replace('.msh', '')),
        'h': h_mean,
        'n_nodes': mesh.n_nodes,
        'n_elements': mesh.n_elements,
        'n_dof': mesh.n_dof,
        'n_free': mesh.n_dof - len(mesh.fixed_dofs),
        'max_u': float(np.max(np.abs(result['u']))),
        'max_vm': float(np.max(result['vm_stress'])),
        'eta': float(eta_info.get('eta', 0)),
        'energy_err': float(eta_info.get('total_error', 0)),
        'energy_norm': float(eta_info.get('energy_norm', 1)),
        'residual': float(result.get('residual', 0)),
        'elapsed': elapsed,
    }
    print(f'{elapsed:.1f}s | nodes={stats["n_nodes"]} | '
          f'max|U|={stats["max_u"]:.3e} | max_vm={stats["max_vm"]:.3e} | eta={stats["eta"]:.2f}%')
    return stats

def convergence_rate(h, values, finest_value=None):
    """从 h vs error 的幂律拟合估算收敛阶.

    finest_value 给定: self-reference → 排除最细网格 (误差=0 扭曲斜率).
    finest_value=None:  直接拟合 values ~ C*h^k → 使用全部数据.

    自参考误差有系统性高估: err(h) = |v(h)−v_f|/|v_f| 的局部斜率在
    h=2h_f 处 = 2k 而非 k, 拟合斜率虚高 (实测 k=1 报 1.40, k=2 报 2.20),
    WARN 分支永不触发 (审计 2026-08-03)。改用 Richardson 真参考消除偏差:
      1. 最后 3 层相邻差比估计阶 p = log2(|Δ_{i-1}|/|Δ_i|) (均匀加密 2×)
      2. v_ref = v_f + (v_f − v_2f)/(2^p − 1)   (Richardson 外推)
      3. 拟合 err(v_i, v_ref) ~ C·h^k
    """
    if len(h) < 2:
        return 0.0
    if finest_value is not None:
        if len(h) < 3:
            return 0.0
        values = np.asarray(values, dtype=float)
        diffs = np.abs(np.diff(values))
        ratio = (diffs[-2] / diffs[-1]) if diffs[-1] > 0.0 else np.inf
        if not np.isfinite(ratio) or ratio <= 1.0:
            # 相邻差为 0 / 不衰减 → 无收敛信息, 保守报 0 让 WARN 触发
            return 0.0
        p = float(np.log2(ratio))
        p = min(max(p, 0.01), 4.0)   # 病态数据防护
        v_ref = values[-1] + (values[-1] - values[-2]) / (2.0 ** p - 1.0)
        h_fit, v_fit = h[:-1], values[:-1]
        err = np.abs(v_fit - v_ref) / (
            np.abs(v_ref) + np.finfo(float).tiny)
        log_v = np.log(np.maximum(err, np.finfo(float).tiny))
    else:
        h_fit, v_fit = h, values
        log_v = np.log(np.maximum(
            np.abs(v_fit), np.finfo(float).tiny))
    log_h = np.log(h_fit)
    A = np.vstack([log_h, np.ones(len(log_h))]).T
    k, _ = np.linalg.lstsq(A, log_v, rcond=None)[0]
    return k

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    os.makedirs(OUT_DIR, exist_ok=True)
    # ══════════════════════════════════════════════════════════════
    print('='*60)
    print('  Convergence Study — Bathe §4.3.5')
    print('  CST理论: 应力 O(h) | 位移 O(h²)')
    print(f'  lc = {LC_VALUES}')
    print('='*60)

    # 运行所有网格
    results = []
    for lc in LC_VALUES:
        try:
            inp = run_gmsh(lc)
            stats = run_fem(inp)
            results.append(stats)
        except Exception as e:
            print(f'  [FAIL] lc={lc}: {e}')

    if len(results) < 2:
        print('\n  [ERROR] 需要至少2组数据才能算收敛率')
        sys.exit(1)

    # ── 收敛率计算 ──
    # 以最细网格为"真值"计算相对误差: error(h) = |value(h) - value(finest)| / |value(finest)|
    hs = np.array([r['h'] for r in results])
    vms = np.array([r['max_vm'] for r in results])
    us = np.array([r['max_u'] for r in results])
    etas = np.array([r['eta'] for r in results])

    # 最细网格作参考
    vm_finest = vms[-1]
    u_finest = us[-1]

    k_vm = convergence_rate(hs, vms, vm_finest)
    k_u = convergence_rate(hs, us, u_finest)
    k_eta = convergence_rate(hs, etas)  # eta 本身已趋于零

    print(f'\n{"="*60}')
    print('  Convergence Rates (relative error vs finest mesh)')
    print(f'{"="*60}')
    print(f'  Stress (vm):     k = {k_vm:+.3f}  (theory: +1.0)')
    print(f'  Displacement:    k = {k_u:+.3f}  (theory: +2.0)')
    print(f'  Energy error:    k = {k_eta:+.3f}  (theory: +1.0)')

    if abs(k_vm - 1.0) < 0.4:
        print('  [PASS] Stress convergence near O(h)')
    elif k_vm > 0.5:
        print('  [OK] Stress convergence roughly O(h)')
    else:
        print('  [WARN] Stress convergence slower than O(h) — stress singularity at hole?')

    if abs(k_u - 2.0) < 0.8:
        print('  [PASS] Displacement convergence near O(h^2)')
    elif k_u > 1.0:
        print('  [OK] Displacement convergence > O(h)')
    else:
        print('  [WARN] Displacement convergence slower than expected')

    # 输出相对误差
    print(f'\n{"="*60}')
    print(f'  Relative Error vs Finest Mesh (h={hs[-1]:.5f})')
    print(f'{"="*60}')
    for r in results[:-1]:
        evm = abs(r['max_vm'] - vm_finest) / vm_finest * 100
        eu = abs(r['max_u'] - u_finest) / u_finest * 100
        print(f'  lc={r["lc"]:.4f} h={r["h"]:.5f} | err_vm={evm:6.2f}% | err_U={eu:6.2f}% | eta={r["eta"]:.2f}%')
    print(f'  lc={results[-1]["lc"]:.4f} h={results[-1]["h"]:.5f} | (finest mesh — reference)')

    # ── 绘图 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('CST Convergence Study — Bathe §4.3.5', fontsize=14, fontweight='bold')

    # 1. von Mises — 相对误差
    ax = axes[0]
    evm = np.abs(vms - vm_finest) / np.abs(vm_finest) * 100
    ax.loglog(hs, np.maximum(evm, 1e-10), 'ro-', ms=8, lw=2, label=f'Relative error (k={k_vm:+.2f})')
    href = np.array([hs[0], hs[-1]])
    ax.loglog(href, [evm[0], evm[0]*(href[1]/href[0])**1.0],
              'k--', lw=1, alpha=0.5, label='O(h) theory')
    ax.set_xlabel('Element size $h = \\sqrt{\\bar{A}}$'); ax.set_ylabel('Relative error in $\\sigma_{vm}^{max}$ [%]')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Stress Convergence')
    ax.invert_xaxis()

    # 2. Displacement — 相对误差
    ax = axes[1]
    eu = np.abs(us - u_finest) / np.abs(u_finest) * 100
    ax.loglog(hs, np.maximum(eu, 1e-10), 'bo-', ms=8, lw=2, label=f'Relative error (k={k_u:+.2f})')
    ax.loglog(href, [eu[0], eu[0]*(href[1]/href[0])**2.0],
              'k--', lw=1, alpha=0.5, label='O(h^2) theory')
    ax.set_xlabel('Element size $h = \\sqrt{\\bar{A}}$'); ax.set_ylabel('Relative error in $|U|_{max}$ [%]')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Displacement Convergence')
    ax.invert_xaxis()

    # 3. Energy error
    ax = axes[2]
    ax.loglog(hs, np.maximum(etas, 1e-10), 'go-', ms=8, lw=2, label=f'eta (k={k_eta:+.2f})')
    ax.loglog(href, [etas[0], etas[0]*(href[1]/href[0])**1.0],
              'k--', lw=1, alpha=0.5, label='O(h) theory')
    ax.set_xlabel('Element size $h = \\sqrt{\\bar{A}}$'); ax.set_ylabel('Z2 Energy error $\\eta$ [%]')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_title('Z2 Error Estimate')
    ax.invert_xaxis()

    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, 'convergence.png')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'\n  图已保存: {png_path}')
    plt.close()

    # ── 数据表格 ──
    print(f'\n{"="*80}')
    print(f'  {"lc":>8} | {"h":>9} | {"nodes":>8} | {"max_vm":>12} | {"max|U|":>12} | {"eta":>6} | {"残差":>10}')
    print(f'{"-"*80}')
    for r in results:
        print(f'  {r["lc"]:8.4f} | {r["h"]:9.5f} | {r["n_nodes"]:>8} | {r["max_vm"]:12.3e} | '
              f'{r["max_u"]:12.3e} | {r["eta"]:5.2f}% | {r["residual"]:10.2e}')
    print(f'{"="*80}')
    print(f'  收敛阶: stress={k_vm:+.2f} | displ={k_u:+.2f} | eta={k_eta:+.2f}')
    print('  理论值:  +1.00         | +2.00         | +1.00')
