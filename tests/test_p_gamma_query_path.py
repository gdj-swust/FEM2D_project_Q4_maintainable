"""P-γ 判别性测试 — ElementLocator.candidates 查询路径向量化等价性.

红侧自证: 参考实现 = 重构前的 3×3 逐桶 Python 循环 + concatenate +
np.unique (topology_core.py 旧 L374-380 的独立复写). 批量查询点对
参考实现逐元素断言: 重构前新测试对旧实现绿, 重构后对新实现仍绿.
返回数组的值集与顺序 (np.unique 排序语义) 必须与参考实现逐元素一致 —
调用方 (element/base.py find_containing_element) 按候选顺序取首个
包含点, 顺序改变即行为改变.

覆盖 (每类网格都要求快路径与 3×3 回退路径均被实际触发):
  规则 quad/tri 网格 / 共线退化 / 微尺度 1e-16 / jittered /
  边界点 / 角点 / 桶边界线上 / 空桶中心 / 域外 1 桶 / 域外多桶 /
  大而有限坐标 (1e300) / 非有限坐标 (NaN/Inf).
"""
import numpy as np
import pytest

from fem2d.topology_core import ElementLocator


def legacy_candidates(loc, x, y):
    """重构前 candidates() 的 3×3 回退实现 (参考实现, 逐元素语义).

    只读 locator 的公开构建产物 (shape/origin/inv_cell/ptr/flat),
    不调用 candidates 本身, 避免与待测实现共享逻辑.
    """
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


def _grid(nx, ny, quad=True, lx=2.0, ly=1.0, scale=1.0):
    xs = np.linspace(0.0, lx, nx + 1) * scale
    ys = np.linspace(0.0, ly, ny + 1) * scale
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def idx(i, j):
        return i * (ny + 1) + j

    quads = np.array(
        [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
         for i in range(nx) for j in range(ny)], dtype=np.int64)
    if quad:
        return nodes, quads
    return nodes, np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])


def _hits_fallback(loc, x, y):
    """True 当该点走 3×3 回退路径 (域外或域内空桶).

    非有限坐标走早期空返回, 不属快/回退路径 — 返回 False 且不崩溃.
    """
    try:
        if not (np.isfinite(x) and np.isfinite(y)):
            return False
    except TypeError:
        return False
    nx, ny = loc.shape
    gx = int((x - loc.origin[0]) * loc.inv_cell[0])
    gy = int((y - loc.origin[1]) * loc.inv_cell[1])
    if not (0 <= gx < nx and 0 <= gy < ny):
        return True
    cell = gx * ny + gy
    return loc.ptr[cell] == loc.ptr[cell + 1]


def _empty_bucket_points(loc, count):
    """取最多 count 个空桶中心 (±0.25 cell 抖动, 保证仍落回该桶) 的查询点.

    空桶中心的点必然触发 3×3 回退 — 判别性覆盖的确定性来源.
    网格无空桶 (如单单元 1×1 网格) 时返回空列表.
    """
    nx, ny = loc.shape
    centers = []
    for gx in range(nx):
        for gy in range(ny):
            cell = gx * ny + gy
            if loc.ptr[cell] == loc.ptr[cell + 1]:
                centers.append(
                    (loc.origin[0] + (gx + 0.5) / loc.inv_cell[0],
                     loc.origin[1] + (gy + 0.5) / loc.inv_cell[1]))
    if not centers:
        return []
    rng = np.random.default_rng(3)
    out = []
    for _ in range(count):
        cx, cy = centers[rng.integers(len(centers))]
        jx = rng.uniform(-0.25, 0.25) / loc.inv_cell[0]
        jy = rng.uniform(-0.25, 0.25) / loc.inv_cell[1]
        out.append((cx + jx, cy + jy))
    return out


def _query_batch(loc, nodes, scale=1.0, huge=True):
    """混合查询点: 随机域内/域外 + 角点 + 边界线 + 空桶中心 + 非有限.

    huge=False 用于 inv_cell >= 1 的微尺度网格 — 大而有限的坐标
    (1e300) 在 (x-origin)*inv_cell 中浮点溢出为 inf, int(inf) 抛
    OverflowError 是重构前后一致的既有行为 (candidates 注释已声明,
    point_in_element 前置拦截), 不在等价性覆盖范围内.
    """
    rng = np.random.default_rng(11)
    pts = [tuple(p) for p in rng.uniform(
        [-0.3, -0.3], [2.3 * scale, 1.3 * scale], size=(250, 2))]
    pts += [(0.0, 0.0), (2.0 * scale, 0.0), (0.0, scale),
            (2.0 * scale, scale)]
    # 桶边界线上 (桶坐标整数) — 覆盖与桶边界对齐的几何退化点
    pts += [(loc.origin[0] + i / loc.inv_cell[0], 0.4 * scale)
            for i in range(0, loc.shape[0] + 1, max(loc.shape[0] // 8, 1))]
    pts += [(0.4 * scale, loc.origin[1] + j / loc.inv_cell[1])
            for j in range(0, loc.shape[1] + 1, max(loc.shape[1] // 8, 1))]
    # 域外 1 桶 (窗口部分有效) 与域外多桶 (窗口为空)
    pts += [(-loc.inv_cell[0], 0.4 * scale),
            (2.0 * scale + loc.inv_cell[0], 0.4 * scale),
            (0.4 * scale, -loc.inv_cell[1]),
            (0.4 * scale, scale + loc.inv_cell[1]),
            (-2.0 * loc.inv_cell[0], -2.0 * loc.inv_cell[1]),
            (2.0 * scale + 3.0 * loc.inv_cell[0], 0.5 * scale)]
    if huge:
        # 大而有限坐标 (旧实现 Python int range 免疫, 向量化不得溢出)
        pts += [(1e300, 0.4 * scale), (-1e300, 0.4 * scale),
                (0.4 * scale, 1e300), (0.4 * scale, -1e300)]
    # 非有限坐标 → 空 int64
    pts += [(np.nan, 0.4 * scale), (0.4 * scale, np.inf),
            (np.inf, np.inf), (-np.inf, 0.4 * scale)]
    pts += _empty_bucket_points(loc, 60)
    return pts


def _assert_batch_equal(loc, pts, require_both_paths=True):
    n_fast = n_fallback = 0
    for x, y in pts:
        new = loc.candidates(x, y)
        ref = legacy_candidates(loc, x, y)
        assert new.dtype == np.int64, (x, y, new.dtype)
        assert np.array_equal(new, ref), (
            f"candidates({x:.6g}, {y:.6g}) != 参考实现\n"
            f"  new = {new}\n  ref = {ref}")
        if _hits_fallback(loc, x, y):
            n_fallback += 1
        else:
            n_fast += 1
    if require_both_paths:
        assert n_fast > 0, "快路径 (域内非空桶) 未被任何查询点触发"
        assert n_fallback > 0, "3×3 回退路径未被任何查询点触发"
    return n_fast, n_fallback


@pytest.mark.parametrize("quad", [True, False])
def test_candidates_matches_legacy_regular_grids(quad):
    """规则 quad/tri 网格: 多种长宽比, 批量查询点逐元素一致."""
    for shape in ((1, 1), (7, 3), (40, 17), (3, 1), (1, 5), (24, 12)):
        nodes, elements = _grid(*shape, quad=quad)
        loc = ElementLocator(nodes, elements)
        pts = _query_batch(loc, nodes)
        fast, fallback = _assert_batch_equal(loc, pts)
        assert fallback >= 20, (shape, fast, fallback)


def test_candidates_matches_legacy_jittered_mesh():
    """jittered 网格 (2×2 不变式测试同款): 乱序桶占用仍逐元素一致."""
    rng = np.random.default_rng(17)
    nodes, elements = _grid(8, 6)
    interior = ((nodes[:, 0] > 1e-9) & (nodes[:, 0] < 2.0 - 1e-9)
                & (nodes[:, 1] > 1e-9) & (nodes[:, 1] < 1.0 - 1e-9))
    jittered = nodes.copy()
    jittered[interior] += rng.uniform(-0.05, 0.05, size=(int(interior.sum()), 2))
    loc = ElementLocator(jittered, elements)
    _assert_batch_equal(loc, _query_batch(loc, jittered))


def test_candidates_matches_legacy_degenerate_collinear():
    """共线退化网格 (y 零跨度): 轴 ULP 回落路径 + 沿直线查询."""
    s = 1e-16
    nodes = np.array([[0.0, 0.0], [s, 0.0], [2 * s, 0.0]], dtype=float)
    loc = ElementLocator(nodes, np.array([[0, 1, 2]]))
    pts = [(0.0, 0.0), (s, 0.0), (1.5 * s, 0.0), (3 * s, 0.0),
           (-s, 0.0), (s, s), (s, -s), (np.nan, 0.0)]
    _assert_batch_equal(loc, pts, require_both_paths=False)
    # 共线退化: 域内非空桶只有 (0,0) 附近 — 至少回退路径必须触发
    assert any(_hits_fallback(loc, x, y) for x, y in pts)


def test_candidates_matches_legacy_micro_scale():
    """微尺度 1e-16 网格: 坐标 ULP 尺度下限路径 + 容差边界点."""
    s = 1e-16
    nodes, elements = _grid(5, 4, scale=s)
    loc = ElementLocator(nodes, elements)
    rng = np.random.default_rng(29)
    pts = [tuple(p) for p in rng.uniform(
        [-0.5 * s, -0.5 * s], [2.5 * s, 1.5 * s], size=(200, 2))]
    pts += [(0.0, 0.0), (2 * s, s), (s, 0.5 * s),
            (s * (1 + 1e-9), s / 2), (2.5 * s, 0.5 * s), (-s, 0.5 * s),
            (s, 1.5 * s), (np.nan, 0.0)]
    pts += _empty_bucket_points(loc, 40)
    fast, fallback = _assert_batch_equal(loc, pts)
    assert fallback >= 20, (fast, fallback)


def test_candidates_result_is_sorted_unique():
    """返回数组的文档语义: 排序后的唯一值 (np.unique 语义保持)."""
    for quad in (True, False):
        nodes, elements = _grid(9, 5, quad=quad)
        loc = ElementLocator(nodes, elements)
        for x, y in _query_batch(loc, nodes):
            r = loc.candidates(x, y)
            assert np.array_equal(r, np.unique(r)), (x, y, r)


def test_candidates_nonfinite_and_out_of_range_empty():
    """非有限坐标与域外远点 → 空 int64 候选集 (不抛异常)."""
    nodes, elements = _grid(9, 5)
    loc = ElementLocator(nodes, elements)
    for x, y in [(np.nan, 0.5), (0.5, np.inf), (-np.inf, 0.5),
                 (np.inf, np.inf), (1e300, 0.5), (-1e300, 0.5),
                 (0.5, 1e300), (0.5, -1e300), (100.0, 100.0)]:
        r = loc.candidates(x, y)
        assert r.dtype == np.int64 and r.size == 0, (x, y, r)
        assert np.array_equal(r, legacy_candidates(loc, x, y))
