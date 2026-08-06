"""P-γ 性能对比基准 — candidates() 3×3 回退向量化 (查询路径).

用法: python tests/bench_p_gamma_candidates.py
无 gmsh 依赖 (手写网格数组); 不做墙钟断言 — 判别性由
tests/test_p_gamma_query_path.py 的逐元素等价测试承担 (两实现
输出必须逐元素一致, 本基准只量化吞吐差)。输出对比表:

    网格 × 批次 (各 100k 查询, 3 次取最优)
      旧实现 (3×3 两层 Python 循环 + list + concatenate + unique)
      新实现 (批量 gather + unique)
      每查询 µs 与加速比

网格选择依据 (实测): 均匀网格因 origin 偏移使单元跨 2×2 桶,
桶占用率 4 → 无空桶 (3×3 回退只经域外点触发); 带孔网格存在
真实空桶 (实测 200×200 挖 r=0.3 圆孔 → 14% 空桶), 是回退路径
的代表性场景 (孔洞/空洞区逐点采样).

直接 `python tests/bench_p_gamma_candidates.py` 时 sys.path[0]=tests/ —
与 scripts/ 下脚本同款引导 (editable install 指向其他 worktree 时
会静默测到旧实现, 数据失真)。
"""
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fem2d.topology_core import ElementLocator


def legacy_candidates(loc, x, y):
    """重构前 candidates() 的 3×3 回退实现 (计时参照, 语义同测试参考)."""
    try:
        finite = bool(np.isfinite(x) and np.isfinite(y))
    except TypeError:
        finite = False
    if not finite:
        return np.empty(0, dtype=np.int64)
    nx, ny = loc.shape
    gx = int((x - loc.origin[0]) * loc.inv_cell[0])
    gy = int((y - loc.origin[1]) * loc.inv_cell[1])
    if 0 <= gx < nx and 0 <= gy < ny:
        cell = gx * ny + gy
        found = loc.flat[loc.ptr[cell]:loc.ptr[cell + 1]]
        if found.size:
            return found
    blocks = []
    for ix in range(max(gx - 1, 0), min(gx + 2, nx)):
        for iy in range(max(gy - 1, 0), min(gy + 2, ny)):
            cell = ix * ny + iy
            blocks.append(loc.flat[loc.ptr[cell]:loc.ptr[cell + 1]])
    if not blocks:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(blocks))


def grid_mesh(nx, ny, scale=1.0, hole_r=0.0):
    """nx×ny quad 网格 (可选挖圆孔) — 向量化构造, 无 gmsh."""
    xs = np.linspace(0.0, 2.0, nx + 1) * scale
    ys = np.linspace(0.0, 1.0, ny + 1) * scale
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def idx(i, j):
        return i * (ny + 1) + j

    quads = np.array(
        [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
         for i in range(nx) for j in range(ny)], dtype=np.int64)
    if hole_r > 0.0:
        centers = nodes[quads].mean(axis=1)
        keep = ~((centers[:, 0] - 1.0) ** 2 + (centers[:, 1] - 0.5) ** 2
                 < hole_r ** 2)
        quads = quads[keep]
    return nodes, quads


def empty_bucket_points(loc, count, seed=3):
    """count 个空桶中心 ±0.25 cell 抖动 — 全部走 3×3 回退路径."""
    nx, ny = loc.shape
    centers = []
    for gx in range(nx):
        for gy in range(ny):
            cell = gx * ny + gy
            if loc.ptr[cell] == loc.ptr[cell + 1]:
                centers.append(
                    (loc.origin[0] + (gx + 0.5) / loc.inv_cell[0],
                     loc.origin[1] + (gy + 0.5) / loc.inv_cell[1]))
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(count):
        cx, cy = centers[rng.integers(len(centers))]
        jx = rng.uniform(-0.25, 0.25) / loc.inv_cell[0]
        jy = rng.uniform(-0.25, 0.25) / loc.inv_cell[1]
        pts.append((cx + jx, cy + jy))
    return pts, len(centers)


def mixed_points(scale, count, seed=11):
    """count 个随机点 (扩展域) — 快路径与回退路径混合."""
    rng = np.random.default_rng(seed)
    return [tuple(p) for p in rng.uniform(
        [-0.3, -0.3], [2.3 * scale, 1.3 * scale], size=(count, 2))]


def near_outside_points(loc, count, seed=5):
    """域外 ≤1 桶的随机点 — 全走回退且窗口部分有效 (能聚到边界桶数据).

    注意 int() 向零截断: 负向域外必须用 -1 - f (f>0), -1 + f 会截断到 0
    变成域内点 — 那批点会走快路径, 静默稀释回退测样.
    """
    nx, ny = loc.shape
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(count):
        side = rng.integers(4)
        f = rng.uniform(0.1, 1.0)
        if side == 0:      # x 负向
            gx, gy = -1 - f, rng.uniform(-0.5, ny - 0.5)
        elif side == 1:    # x 正向
            gx, gy = nx + f, rng.uniform(-0.5, ny - 0.5)
        elif side == 2:    # y 负向
            gx, gy = rng.uniform(-0.5, nx - 0.5), -1 - f
        else:              # y 正向
            gx, gy = rng.uniform(-0.5, nx - 0.5), ny + f
        pts.append((loc.origin[0] + gx / loc.inv_cell[0],
                    loc.origin[1] + gy / loc.inv_cell[1]))
    return pts


def best_of(fn, pts, reps=3):
    """fn 对 pts 逐点查询, 返回 (最优耗时 s, 每查询 µs)."""
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        for x, y in pts:
            fn(x, y)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, best / len(pts) * 1e6


def bench(mesh_name, nodes, elements, batches, n_query=100_000):
    loc = ElementLocator(nodes, elements)
    n_bucket = int(loc.shape[0]) * int(loc.shape[1])
    n_empty = sum(1 for c in range(n_bucket)
                  if loc.ptr[c] == loc.ptr[c + 1])
    print(f"\nmesh: {mesh_name} ({elements.shape[0]} elements, "
          f"{n_bucket} buckets, {n_empty} empty)")
    new_fn = lambda x, y: loc.candidates(x, y)
    old_fn = lambda x, y: legacy_candidates(loc, x, y)

    for label, pts in batches:
        if pts is None:
            pts, _ = empty_bucket_points(loc, n_query)
        # 预热 (两实现同待遇)
        for x, y in pts[:2000]:
            old_fn(x, y)
            new_fn(x, y)
        old_s, old_us = best_of(old_fn, pts)
        new_s, new_us = best_of(new_fn, pts)
        print(f"  {label}")
        print(f"    legacy   {old_s * 1000:9.2f} ms   "
              f"{old_us:7.2f} us/query")
        print(f"    new      {new_s * 1000:9.2f} ms   "
              f"{new_us:7.2f} us/query")
        print(f"    speedup  {old_s / new_s:6.2f}x")


def main():
    NQ = 100_000

    nodes, elements = grid_mesh(200, 200)
    bench("uniform 200x200 quad (dense, 0% empty buckets)", nodes, elements,
          [("mixed random", mixed_points(1.0, NQ)),
           ("near-outside (all fallback, partial window)",
            near_outside_points(
                ElementLocator(nodes, elements), NQ))])

    nodes, elements = grid_mesh(200, 200, hole_r=0.3)
    bench("200x200 quad with r=0.3 hole (14% empty buckets)", nodes, elements,
          [("mixed random (incl. hole region)", mixed_points(1.0, NQ)),
           ("empty-bucket centers (all fallback)", None)])

    s = 1e-16
    nodes, elements = grid_mesh(24, 16, scale=s)
    bench(f"micro 24x16 quad (scale {s:.0e})", nodes, elements,
          [("mixed random", mixed_points(s, NQ))])


if __name__ == "__main__":
    main()
