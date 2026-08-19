"""SPR-BC-2026-001 验收测量脚本 — 边界节点恢复精度/收敛阶/效应指数.

只调用公共 API (Mesh / solve / spr_recovery / estimate 的同源输入),
与实现无关 — 改前 (baseline) 与改后 (check) 用同一脚本测出对比
数据。全流程确定性 (结构化网格 + 直接求解, 无随机数)。

用法:
  python scripts/spr_boundary_acceptance.py --mode baseline --out <dir>
  python scripts/spr_boundary_acceptance.py --mode check    --out <dir>

测量项 (验收标准 B/C/D/E):
  B: TG 悬臂梁 (CST/Q4) + Kirsch (CST) 非边界节点恢复值快照 (.npz);
     check 模式逐位比对 (改动后非边界节点必须逐位一致)
  C1: TG 顶边中点 (L/2, H/2) σy 精确解为 0 → |σy_rec| / σx_max
  C2: TG 右端面节点 τxy 对抛物线剪流 — 最大/平均归一化误差
  C3: Kirsch 孔边 θ=0/30/60/90° 恢复 σθθ 对解析解 (σ∞ 归一化)
  D: 边界采样点恢复应力误差收敛阶 (CST/Q4, 5 层, ln|e|~ln h 斜率)
  E: 含边界单元区域效应指数 θ = ||e_est|| / ||e_exact||

TG 闭式解 (Timoshenko-Goodier, 平面应力, P>0 向下):
    σx = +P(L-x)y/I,  σy = 0,  τxy = -P/(2I)·(H²/4 - y²),  I = tH³/12
右端面抛物线剪流为精确施加 → 该解为远离固支端 (St-Venant 衰减) 的
精确参考场, 采样点/区域全部取 x ≥ H 远离固支端边界层。
"""
import argparse
import json
import os

import numpy as np

from fem2d import Mesh, solve
from fem2d.convergence import _gen_cantilever_mesh, _parabolic_shear_traction
from fem2d.material import D_matrix
from fem2d.spr import spr_recovery

# ── TG 悬臂梁模型 (与 tests/test_analytic_e3_z2_rate.py 同款) ──
L, H, T = 5.0, 1.0, 0.1
E_MOD, NU = 210e9, 0.3
P_SHEAR = 10000.0
I_BEAM = T * H**3 / 12.0
SXX_MAX = P_SHEAR * L * (H / 2.0) / I_BEAM   # 固支端梁理论最大弯曲应力
TAU_MAX = P_SHEAR / (2.0 * I_BEAM) * (H**2 / 4.0)  # 剪流最大值 (y=0)

# ── Kirsch 模型 (与 tests/test_analytic_e2_kirsch.py 同款) ──
A_HOLE, W_PLATE = 1.0, 10.0
SIGMA_INF = 1e6
KIRSCH_THETAS = (0.0, 30.0, 60.0, 90.0)


def tg_exact_stress(x, y):
    """TG 闭式解 (P>0 向下, 平面应力).

    符号以 FE 实测为准: 下弯梁顶边受拉, σx = +P(L-x)y/I > 0 (y>0);
    τxy = -P/(2I)(H²/4-y²) < 0 与 _parabolic_shear_traction 同约定
    (convergence.py 注释中 "-P(L-L/2)(H/2)/I = PLH/(4I)" 的代数
    符号笔误以本处为准)。
    """
    return np.array([
        P_SHEAR * (L - x) * y / I_BEAM,
        0.0,
        -P_SHEAR / (2.0 * I_BEAM) * (H**2 / 4.0 - y**2),
    ])


def _tg_solve(elem_type, nx):
    """TG 抛物线剪流悬臂梁: 构建/求解/恢复 (estimate 同源输入)."""
    ny = nx // 2
    nodes, elements = _gen_cantilever_mesh(L, H, nx, ny,
                                           elem_type=elem_type)
    m = Mesh(nodes=nodes, elements=elements, E=E_MOD, nu=NU, thickness=T,
             plane_type="stress", elem_type=elem_type)
    for n in m.nodes_on_edge("x", "min", tol=1e-6):
        m.fix_node(int(n), "both", 0.0)
    right = sorted(m.nodes_on_edge("x", "max", tol=1e-6),
                   key=lambda n: m.nodes[int(n), 1])

    def shear(x, y):
        del x
        return _parabolic_shear_traction(y, H, T, P_SHEAR)

    for a_, b_ in zip(right, right[1:]):
        m.add_traction(int(a_), int(b_), 0.0, shear)
    result = solve(m, method="elimination", verbose=False)
    qp = result["stress_qp"]
    recovered = spr_recovery(m, qp if qp is not None else result["stress"])
    return m, result, recovered


def _node_at(mesh, x, y, tol=1e-9):
    d = np.linalg.norm(mesh.nodes - np.array([x, y]), axis=1)
    nid = int(np.argmin(d))
    if d[nid] > tol:
        raise RuntimeError(f"节点 ({x},{y}) 不存在, 最近距离 {d[nid]:.3e}")
    return nid


def _boundary_mask(mesh):
    mask = np.zeros(mesh.n_nodes, dtype=bool)
    for lo, hi in mesh.boundary_edges:
        mask[int(lo)] = mask[int(hi)] = True
    return mask


def _measure_c1(elem_type, nx=64):
    """C1: 顶边中点 (L/2,H/2) σy=0 → |σy_rec|/σx_max."""
    mesh, _result, rec = _tg_solve(elem_type, nx)
    nid = _node_at(mesh, L / 2.0, H / 2.0)
    return float(abs(rec[nid, 1])) / SXX_MAX


def _measure_c2(elem_type, nx=64):
    """C2: 右端面节点恢复 τxy 对抛物线剪流 — 最大/平均归一化误差."""
    mesh, _result, rec = _tg_solve(elem_type, nx)
    face = [int(n) for n in mesh.nodes_on_edge("x", "max", tol=1e-9)]
    errs = [abs(rec[n, 2] - tg_exact_stress(*mesh.nodes[n])[2]) / TAU_MAX
            for n in face]
    return {"max": float(np.max(errs)), "rms": float(np.sqrt(np.mean(
        np.array(errs) ** 2)))}


def _kirsch_mesh(n_ang, n_rad):
    """1/4 板径向网格 (复制自 tests/test_analytic_e2_kirsch.py —
    网格生成属测量基础设施, 非生产逻辑)."""
    nodes = []
    for i in range(n_ang + 1):
        th = 0.5 * np.pi * i / n_ang
        c, s = np.cos(th), np.sin(th)
        rmax = W_PLATE / max(abs(c), abs(s))
        for j in range(n_rad + 1):
            r = A_HOLE + (rmax - A_HOLE) * j / n_rad
            nodes.append([r * c, r * s])
    nodes = np.array(nodes)
    elems = []
    for i in range(n_ang):
        for j in range(n_rad):
            n0 = i * (n_rad + 1) + j
            n1 = n0 + 1
            n2 = (i + 1) * (n_rad + 1) + j + 1
            n3 = n2 - 1
            elems.append([n0, n1, n2])
            elems.append([n0, n2, n3])
    elems = np.array(elems, dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=1.0,
             plane_type="stress", elem_type="CST")
    for j in range(n_rad + 1):
        m.fix_node(j, "y")                        # θ=0 轴: uy=0
        m.fix_node(n_ang * (n_rad + 1) + j, "x")  # θ=π/2 轴: ux=0
    return m, nodes


def _measure_c3(n_ang=96, n_rad=48):
    """C3: Kirsch 孔边 θ=0/30/60/90° 恢复 σθθ 对解析解 (σ∞ 归一化)."""
    m, nodes = _kirsch_mesh(n_ang, n_rad)
    right = sorted(np.flatnonzero(np.abs(nodes[:, 0] - W_PLATE) < 1e-9),
                   key=lambda n: nodes[n, 1])
    for a_, b_ in zip(right, right[1:]):
        m.add_traction(int(a_), int(b_), SIGMA_INF, 0.0)
    result = solve(m, method="elimination", verbose=False)
    qp = result["stress_qp"]
    rec = spr_recovery(m, qp if qp is not None else result["stress"])
    out = {"theta_deg": [], "rec": [], "exact": [], "abs_err": []}
    for th_deg in KIRSCH_THETAS:
        th = np.radians(th_deg)
        i = int(round(th / (0.5 * np.pi) * n_ang))
        nid = i * (n_rad + 1)     # j=0 → r=a 孔边节点
        sx, sy, txy = rec[nid]
        s_theta = (sx * np.sin(th)**2 + sy * np.cos(th)**2
                   - 2.0 * txy * np.sin(th) * np.cos(th))
        exact = SIGMA_INF * (1.0 - 2.0 * np.cos(2.0 * th))
        out["theta_deg"].append(th_deg)
        out["rec"].append(float(s_theta) / SIGMA_INF)
        out["exact"].append(float(exact) / SIGMA_INF)
        out["abs_err"].append(float(abs(s_theta - exact)) / SIGMA_INF)
    return out


def _measure_d():
    """D: 边界采样点恢复应力误差收敛阶 (CST/Q4, 5 层).

    采样点 (全部为真实节点, 避开固支角点, x ≥ H 远离固支端边界层):
      syy_top     顶边中点 (L/2,H/2)  σy=0            (norm σx_max)
      txy_right_q 右端面 1/4 高 (L,H/4) τ=0.75τmax    (norm τmax; 单侧
                   薄 patch 对 y 非对称 → 改前基线应为 ~O(h))
      txy_top     顶边中点 (L/2,H/2)  τ=0             (norm τmax)
      sxx_top     顶边中点 (L/2,H/2)  σx=σx_max/2     (norm σx_max, 参考)
    右端面中点 (L,0) 因 patch 对 y 对称+偶场, 改前即超收敛, 不具判别力,
    故改用 y=H/4 非对称点。斜率 = ln|e| ~ ln h 最小二乘, 跳过最粗
    非渐近层 (txy_right_q 在 ny=2 层无节点, 自动跳过该层)。
    """
    levels = (4, 8, 16, 32, 64)
    comp = {"syy_top": 1, "txy_top": 2, "txy_right_q": 2, "sxx_top": 0}
    norm = {"syy_top": SXX_MAX, "txy_top": TAU_MAX,
            "txy_right_q": TAU_MAX, "sxx_top": SXX_MAX}

    def coord_for(key):
        return (L, H / 4.0) if key == "txy_right_q" else (L / 2.0, H / 2.0)

    def exact_for(key):
        if key == "syy_top" or key == "txy_top":
            return 0.0
        if key == "txy_right_q":
            return tg_exact_stress(L, H / 4.0)[2]
        return tg_exact_stress(L / 2.0, H / 2.0)[0]

    rates = {}
    for elem_type in ("CPS3", "CPS4"):
        seq = {k: [] for k in comp}
        hs = {k: [] for k in comp}
        for nx in levels:
            mesh, _result, rec = _tg_solve(elem_type, nx)
            for key in comp:
                try:
                    nid = _node_at(mesh, *coord_for(key))
                except RuntimeError:
                    continue  # 该层网格无此节点 (ny=2 无 y=H/4 节点)
                err = abs(rec[nid, comp[key]] - exact_for(key))
                seq[key].append(err / norm[key])
                hs[key].append(L / nx)
        slopes = {}
        for key, errs in seq.items():
            e = np.array(errs)
            h_arr = np.array(hs[key])
            # 跳过最粗非渐近层, 拟合 ln|e| ~ ln h
            slopes[key] = float(np.polyfit(
                np.log(h_arr[1:]),
                np.log(np.maximum(e[1:], np.finfo(float).tiny)), 1)[0])
        rates[elem_type] = {"h": [L / nx for nx in levels], **seq,
                            **{f"slope_{k}": v for k, v in slopes.items()}}
    return rates


def _region_effectivity(mesh, result, recovered):
    """E: 含边界单元区域效应指数 θ = ||e_est||/||e_exact||.

    区域 = 顶边或右端面相邻单元, 且质心 x ≥ H (排除固支端边界层 —
    TG 闭式解在固支端不精确)。误差在积分点离散:
        e_est   = σ*(q) − σh(q),   e_exact = σexact(q) − σh(q)
    能量内积用 D⁻¹, 积分权与 estimate 的恢复积分规则同源。
    """
    kernel = mesh.element_kernel
    shape = kernel.recovery_shape_matrix(mesh)
    weights = kernel.recovery_weights(mesh)
    if shape is None or weights is None:
        raise RuntimeError("效应指数测量需要批量恢复规则 (CST/Q4)")
    shape = np.asarray(shape, dtype=float)
    weights = np.asarray(weights, dtype=float)
    stress_qp = result["stress_qp"]
    if stress_qp is None:
        stress_qp = np.broadcast_to(
            result["stress"][:, None, :],
            (mesh.n_elements, shape.shape[0], 3))

    bmask = _boundary_mask(mesh)
    on_top = np.any(np.abs(mesh.nodes[mesh.elements, 1] - H / 2.0) < 1e-9,
                    axis=1)
    on_right = np.any(np.abs(mesh.nodes[mesh.elements, 0] - L) < 1e-9,
                      axis=1)
    region = (on_top | on_right) & (mesh.centroids[:, 0] >= H) \
        & np.any(~bmask[mesh.elements], axis=1) & np.any(bmask[mesh.elements],
                                                         axis=1)
    if not np.any(region):
        raise RuntimeError("效应指数区域为空 — 检查网格/区域定义")

    d_inv = np.linalg.inv(D_matrix(mesh.E, mesh.nu, mesh.plane_type))
    e_est_sq = 0.0
    e_exact_sq = 0.0
    for eid in np.flatnonzero(region):
        conn = mesh.elements[eid]
        s_star_pt = shape @ recovered[conn]
        s_h = stress_qp[eid]
        s_exact = np.array([tg_exact_stress(*p)
                            for p in shape @ mesh.nodes[conn]])
        w = weights[eid]
        d_est = s_star_pt - s_h
        d_exact = s_exact - s_h
        e_est_sq += float(np.sum(
            w * np.einsum("qi,ij,qj->q", d_est, d_inv, d_est)))
        e_exact_sq += float(np.sum(
            w * np.einsum("qi,ij,qj->q", d_exact, d_inv, d_exact)))
    if e_exact_sq <= 0.0:
        return None
    return float(np.sqrt(e_est_sq / e_exact_sq))


def _measure_e():
    """E: TG 悬臂梁含边界单元区域效应指数 (CST/Q4 × 3 层)."""
    out = {}
    for elem_type in ("CPS3", "CPS4"):
        thetas = []
        for nx in (16, 32, 64):
            mesh, result, rec = _tg_solve(elem_type, nx)
            thetas.append(_region_effectivity(mesh, result, rec))
        out[elem_type] = {"nx": [16, 32, 64], "theta": thetas}
    return out


def _snapshot_interior():
    """B: 非边界节点恢复值快照 (TG CST/Q4 + Kirsch CST)."""
    snap = {}
    for elem_type in ("CPS3", "CPS4"):
        mesh, _result, rec = _tg_solve(elem_type, 32)
        snap[f"tg_{elem_type}"] = rec[~_boundary_mask(mesh)]
    m, nodes = _kirsch_mesh(96, 48)
    right = sorted(np.flatnonzero(np.abs(nodes[:, 0] - W_PLATE) < 1e-9),
                   key=lambda n: nodes[n, 1])
    for a_, b_ in zip(right, right[1:]):
        m.add_traction(int(a_), int(b_), SIGMA_INF, 0.0)
    result = solve(m, method="elimination", verbose=False)
    qp = result["stress_qp"]
    rec = spr_recovery(m, qp if qp is not None else result["stress"])
    snap["kirsch_CPS3"] = rec[~_boundary_mask(m)]
    return snap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("baseline", "check"),
                    default="baseline")
    ap.add_argument("--out", default=".", help="输出目录 (JSON + npz)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    snap_path = os.path.join(args.out, "spr_boundary_interior.npz")
    interior_identical = None
    if args.mode == "check":
        old = dict(np.load(snap_path))
        new = _snapshot_interior()
        interior_identical = {
            k: bool(np.array_equal(old[k], new[k])) for k in sorted(old)
        }
        if not all(interior_identical.values()):
            print("!! B 内部不变性失败:", interior_identical)
    else:
        np.savez(snap_path, **_snapshot_interior())

    report = {
        "mode": args.mode,
        "B_interior_identical": interior_identical,
        "C1_syy_top_ratio": {
            "CPS3": _measure_c1("CPS3"),
            "CPS4": _measure_c1("CPS4"),
        },
        "C2_txy_right_face": {
            "CPS3": _measure_c2("CPS3"),
            "CPS4": _measure_c2("CPS4"),
        },
        "C3_kirsch_hole_edge": _measure_c3(),
        "D_boundary_convergence_rates": _measure_d(),
        "E_region_effectivity": _measure_e(),
    }
    out_path = os.path.join(args.out, f"spr_boundary_{args.mode}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[written] {out_path}")


if __name__ == "__main__":
    main()
