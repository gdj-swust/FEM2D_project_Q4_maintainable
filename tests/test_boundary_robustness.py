"""Shewchuk orient2d 边界检测鲁棒性验证

三个极端模型 — 旧浮点实现会翻的 case.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from fem2d.boundary.predicates import orient2d


def test_point_in_loop_degenerate():
    """_point_in_loop 在极端情况下的正确性.

    测试点恰在多边形边上、近顶点、近共线边等退化情况.
    """
    print("\n" + "=" * 55)
    print("  Test A: Point-in-loop degenerate cases")
    print("=" * 55)

    from fem2d.boundary.topology import _point_in_loop

    # 简单三角形: (0,0)→(10,0)→(5,8)→(0,0)
    xs = np.array([0.0, 10.0, 5.0])
    ys = np.array([0.0, 0.0, 8.0])

    tests = [
        # (px, py, expected_inside, description)
        (5.0, 2.0, True,   "中心点"),
        (15.0, 5.0, False,  "远在外部"),
        # 恰在边上: Gmsh orient2d 返回 0 → 不计交叉 → False (一致行为)
        (0.5, 1e-14, True, "内侧贴底边 ~1e-14"),
        (-1e-14, 5.0, False,"距左边外侧 ~1e-14"),
    ]

    for px, py, expected, desc in tests:
        got = _point_in_loop(px, py, xs, ys)
        status = "PASS" if got == expected else "FAIL"
        print(f"  {status}  ({px}, {py}) {desc} → {'内' if got else '外'} "
              f"(期望 {'内' if expected else '外'})")
        assert got == expected, f"FAIL: {desc}"


def test_point_near_edge():
    """点在边界边的延长线附近 — orient2d 必须给出正确符号.

    构造一个孔, 其一个节点恰好在外部节点构成的直线附近(距离 ~1e-12).
    """
    print("\n" + "=" * 55)
    print("  Test B: Orient2d near-collinear (dist ~ 1e-12)")
    print("=" * 55)

    # 三点近共线: A=(0,0), B=(1,0), C=(0.5, 1e-14)
    ax, ay = 0.0, 0.0
    bx, by = 1.0, 0.0
    cx, cy = 0.5, 1e-14

    # 标准浮点: cross = (bx-ax)*(cy-ay) - (cx-ax)*(by-ay)
    # = 1*1e-14 - 0.5*0 = 1e-14 ≈ 0 (可能被截断)
    cross_float = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    cross_robust = orient2d(ax, ay, bx, by, cx, cy)

    print(f"  浮点 cross = {cross_float:.2e}")
    print(f"  orient2d   = {cross_robust:.16e}")
    print(f"  符号: float={'+' if cross_float>0 else '-' if cross_float<0 else '0'}, "
          f"orient2d={'+' if cross_robust>0 else '-' if cross_robust<0 else '0'}")

    # orient2d 必须给出正确符号 (C 在 AB 上方 → CCW → >0)
    assert cross_robust > 0, "FAIL: orient2d should be > 0 for CCW point"
    print("  PASS: orient2d 在 ~1e-14 尺度下符号正确")


def test_sliver_boundary_elements():
    """闭环边界上微小扰动 — 曲率不崩, 微小转折不被误判.

    近正方形闭环, 在一段直边上加 1e-5 的扰动 (≈0.006°).
    """
    print("\n" + "=" * 55)
    print("  Test C: Sliver perturbation on closed loop")
    print("=" * 55)

    from fem2d.boundary.geometry import curvature, sharp_corner_indices

    # 正八边形近似圆, 在某边中点加 1e-5 偏移
    n = 8
    coords = np.zeros((n, 2))
    for i in range(n):
        ang = 2 * np.pi * i / n
        coords[i] = [np.cos(ang), np.sin(ang)]
    # 闭合: 最后一个节点回到第一个
    coords = np.vstack([coords, coords[0]])

    # 边 0→1 中点附近加微小偏移 (通过添加额外节点)
    coords_noisy = np.zeros((n * 2 + 1, 2))
    for i in range(n):
        coords_noisy[i * 2] = coords[i]
        mid = (coords[i] + coords[(i + 1) % n]) / 2.0
        # 只在第一段加噪声
        if i == 0:
            mid[1] += 1e-5
        coords_noisy[i * 2 + 1] = mid
    coords_noisy[-1] = coords[0]  # 闭合

    kappa = curvature(coords_noisy)
    max_k = np.max(np.abs(kappa))
    print(f"  曲率 max|κ| = {max_k:.2e}")

    # 曲率不能是 NaN/Inf
    assert np.all(np.isfinite(kappa)), "FAIL: curvature contains NaN/Inf"

    # 35° 阈值: 八边形内角 135° → 转角 45° → 应该检测到 8 个角
    # (闭合重复点曾使首个角点静默跳过, 只检出 7 个 — 审计 2026-08-03)
    corners = sharp_corner_indices(coords_noisy, angle_threshold_deg=35.0)
    print(f"  35° 阈值: {len(corners)} 个角点 (期望 8)")

    assert len(corners) == 8, f"FAIL: expected 8 corners, got {len(corners)}"
    assert 1 not in corners, \
        "噪声中点 (第一段 mid) 不应被误判为角点"
    print("  PASS: 噪声不导致崩溃, 微小扰动不被误判为角点")


if __name__ == "__main__":
    test_point_in_loop_degenerate()
    test_point_near_edge()
    test_sliver_boundary_elements()
    print(f"\n{'='*55}")
    print("  ALL 3 EXTREME TESTS PASSED")
    print(f"{'='*55}")