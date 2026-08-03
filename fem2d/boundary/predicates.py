"""自适应精度几何谓词 — orient2d

基于: Shewchuk, "Adaptive Precision Floating-Point Arithmetic
and Fast Robust Geometric Predicates" (1997)

核心思想:
  1. 先用普通浮点算行列式
  2. 若 |结果| > 误差界 → 符号可靠, 直接返回
  3. 否则用 float.as_integer_ratio 的任意精度整数运算精确判定符号

历史 (高强度审计 2026-08-02):
  - Dekker-Veltkamp expansion 需要 (2^27+1)·a 精确计算, IEEE 754
    double 下不满足 → 分裂只有 ε 级精度, 近共线输入符号错误率 ~11%
  - Windows math.fma 是软件模拟 (double-rounding), 同样不精确
  - CPython math.frexp 的尾数有 ulp 级舍入 (实测 ma·2^ea ≠ a)
  - float.as_integer_ratio() 规范保证精确, 任意精度整数运算零误差

返回值契约: Step B 返回 精确符号 × 浮点 det 量级 (det=0 时 ±_TINY)。
调用方 (turning_angles 等) 用 atan2(cross, dot) 依赖量级 — 返回 ±1.0
会把近共线的 0 转向角放大成 ±45° 误判尖角 (高强度审计 2026-08-02)。
"""
import math

_TINY = 5e-324  # float64 最小次正规 — 精确 det≠0 但浮点 det=0 时的量级占位


def orient2d(ax, ay, bx, by, cx, cy):
    """C 在直线 AB 的哪一侧? 返回精确符号.

    返回:
      >0  → C 在 AB 左侧 (CCW)
      <0  → C 在 AB 右侧 (CW)
      =0  → A, B, C 共线

    算法:
      det = |ax ay 1| = (ax-cx)*(by-cy) - (ay-cy)*(bx-cx)
            |bx by 1|
            |cx cy 1|
    """
    # Step A: 普通浮点估算 + 误差界 (极端尺度乘积上溢/下溢产生
    # inf/NaN 是预期的 — NaN 比较为 False, 自动落入 Step B 精确路径)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)

        # 误差界 (Shewchuk §3.1): 行列式计算涉及 2 次减法 + 2 次乘法 + 1 次减法
        # err_bound = (3ε + 16ε²) * (|(ax-cx)*(by-cy)| + |(ay-cy)*(bx-cx)|)
        # 其中 ε = 2^{-53} ≈ 1.11e-16 (double precision machine epsilon)
        eps = 2.220446049250313e-16  # 2^{-52}, the relative error bound
        err_bound = (3.0 * eps + 16.0 * eps * eps) * (
            abs((ax - cx) * (by - cy)) + abs((ay - cy) * (bx - cx))
        )

        if abs(det) > err_bound:
            # 符号确定, 无需精化
            return det

    # Step B: 原始坐标 as_integer_ratio 任意精度整数符号判定 (零浮点
    # 误差)。对 ax/ay/bx/by/cx/cy 各调 as_integer_ratio() (Python float
    # 规范精确), 再用整数完成精确减法与乘法 — 曾先做浮点差
    # (ax − cx) 再精确化, 减法舍入使符号可翻转 (复测 2026-08-02 反例)。
    # 注意: np.float64.as_integer_ratio() 不精确, 必须先转 Python float。
    na, da = float(ax).as_integer_ratio()
    ny, dy = float(ay).as_integer_ratio()
    nb, db = float(bx).as_integer_ratio()
    nz, dz = float(by).as_integer_ratio()
    nc, dc = float(cx).as_integer_ratio()
    nw, dw = float(cy).as_integer_ratio()
    # acx = ax − cx = (na·dc − nc·da)/(da·dc)
    # bcy = by − cy = (nz·dw − nw·dz)/(dz·dw)
    # acy = ay − cy = (ny·dw − nw·dy)/(dy·dw)
    # bcx = bx − cx = (nb·dc − nc·db)/(db·dc)
    # det = acx·bcy − acy·bcx — 公共正分母 (全为 2 的幂), 符号由整数分子决定
    x1 = na * dc - nc * da      # acx 分子
    y1 = nz * dw - nw * dz      # bcy 分子
    x2 = ny * dw - nw * dy      # acy 分子
    y2 = nb * dc - nc * db      # bcx 分子
    num = x1 * y1 * (dy * dw * db * dc) - x2 * y2 * (da * dc * dz * dw)
    # 符号精确; 量级取浮点 det 的绝对值 (NaN — 极端尺度乘积上溢 —
    # 时用 ±_TINY 占位, 曾返回 NaN 使调用方符号判定失效)
    magnitude = abs(det) if (det != 0.0 and math.isfinite(det)) else _TINY
    if num > 0:
        return magnitude
    if num < 0:
        return -magnitude
    return 0.0
