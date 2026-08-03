"""Topology-aware element quality report for CST and Q4 meshes."""
import numpy as np


def _compute(mesh):
    """Compute mesh-quality indicators without printing."""
    mesh.build_connectivity()
    n = mesh.n_elements
    area = np.asarray(mesh.signed_areas, dtype=float).copy()
    polygon = mesh.nodes[mesh.elements]
    following = np.roll(polygon, -1, axis=1)
    previous = np.roll(polygon, 1, axis=1)
    edge_vectors = following - polygon
    lengths = np.linalg.norm(edge_vectors, axis=2)
    ratio = lengths.max(axis=1) / (
        lengths.min(axis=1) + np.finfo(float).tiny)

    incoming = polygon - previous
    outgoing = following - polygon
    cross = (
        incoming[:, :, 0] * outgoing[:, :, 1]
        - incoming[:, :, 1] * outgoing[:, :, 0])
    dot = np.einsum("eia,eia->ei", incoming, outgoing)
    turn = np.degrees(np.arctan2(cross, dot))
    # 外转角带符号: CCW 单元内角 = 180° − turn, CW (负面积) 单元内角 =
    # 180° + turn。只按 CCW 公式会把手性反转单元的内角算成补角
    # (矩形 CW 单元报 270° 而非 90°)。
    ccw = (np.asarray(mesh.signed_areas, dtype=float) > 0.0)[:, None]
    interior = np.where(ccw, 180.0 - turn, 180.0 + turn)
    interior = np.where(interior <= 0.0, interior + 360.0, interior)
    min_ang = interior.min(axis=1)
    max_ang = interior.max(axis=1)

    # 评级
    # Q4 must be checked at every kernel-supplied Jacobian sample: polygon
    # area alone cannot detect a folded/bow-tie isoparametric mapping.
    # 分类逻辑统一走 kernel.jacobian_report — 与求解器拦截共享同一容差
    # (base.ElementKernel.jacobian_report), 避免两处实现分叉。
    # det_j 原始数组仅用于 jacobian_min 统计。
    det_j = np.asarray(
        mesh.element_kernel.jacobian_determinants(mesh), dtype=float)
    if det_j.ndim == 1:
        det_j = det_j[:, None]
    report = mesh.element_kernel.jacobian_report(mesh)
    inverted = report.inverted
    degenerate = report.degenerate
    jacobian_neg = int(report.bad.size)

    a_ok = n - jacobian_neg
    r_ok = int(np.sum(ratio < 3))
    r_warn = int(np.sum((ratio >= 3) & (ratio < 5)))
    r_bad = int(np.sum(ratio >= 5))
    ang_bad_mask = (min_ang <= 15) | (max_ang >= 150)
    ang_warn_mask = (
        ((min_ang > 15) & (min_ang <= 30))
        | ((max_ang >= 120) & (max_ang < 150)))
    ang_bad = int(np.sum(ang_bad_mask))
    ang_warn = int(np.sum(ang_warn_mask & ~ang_bad_mask))
    ang_ok = n - ang_bad - ang_warn

    score = (a_ok/n*30 + r_ok/n*30 + ang_ok/n*40)
    grade = "A" if score > 95 else "B" if score > 85 else "C" if score > 70 else "D"
    if inverted > 0 or degenerate > 0:
        grade = "F"  # 反向/退化单元一票否决

    return {
        "n": n, "grade": grade, "score": score,
        "area_min": np.abs(area).min(), "area_max": np.abs(area).max(),
        "area_mean": float(np.abs(area).mean()),
        # 1e-30 绝对地板曾使面积 ~1e-32 的微尺度网格 CV 静默失真
        # (真实 27.2% 报成 0.4%)
        "area_cv": float(np.abs(area).std()/(np.abs(area).mean()+np.finfo(float).tiny)),
        "ratio_max": ratio.max(), "ratio_ok": r_ok,
        "ratio_warn": r_warn, "ratio_bad": r_bad,
        "angle_min": min_ang.min(), "angle_max": max_ang.max(),
        "angle_ok": ang_ok, "angle_warn": ang_warn, "angle_bad": ang_bad,
        "jacobian_neg": jacobian_neg,
        "inverted": inverted,
        "degenerate": degenerate,
        "jacobian_min": float(det_j.min()),
    }

def evaluate(mesh):
    """返回网格质量字典 (不打印, 供中文总结调用)"""
    return _compute(mesh)

def report(mesh):
    """打印网格质量报告 (英文, 保持向后兼容)"""
    q = _compute(mesh)
    n = q['n']

    print(f"\n{'='*55}")
    print(f"  Mesh Quality  |  {n} elements  |  Grade: {q['grade']} ({q['score']:.0f}/100)")
    print(f"{'='*55}")
    print(f"  Area     min:{q['area_min']:.3e}  max:{q['area_max']:.3e}  "
          f"mean:{q['area_mean']:.3e}")
    print(f"  Ratio    <3:{q['ratio_ok']:5d}  3-5:{q['ratio_warn']:4d}  >5:{q['ratio_bad']:4d}"
          f"  (max: {q['ratio_max']:.1f})")
    print(f"  Angle    ok:{q['angle_ok']:5d}  warn:{q['angle_warn']:4d}  bad:{q['angle_bad']:4d}"
          f"  (min: {q['angle_min']:.0f} deg)")
    print(f"  Jacobian: inverted={q['inverted']}  degenerate={q['degenerate']}  (must be 0)")
    if q['jacobian_neg'] == 0 and q['ratio_bad'] == 0 and q['angle_bad'] == 0:
        print("  All elements OK")
    print(f"{'='*55}")
    return q
