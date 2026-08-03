"""Q4 逆映射消差判别性测试 (外部审查 P2-5).

旧实现直接 `N @ coords - target` 在绝对坐标下做 Newton 迭代 — 大坐标
原点 + 小局部尺寸时浮点求和含 ~ulp(原点) 消差误差, 迭代无法收敛:
  * 单位方形平移到 1e12 后, 1000 个合法内部点 ~10% 定位失败
  * 坐标 ~1e6、边长 1e-6 时 ~2% 失败

修复: 迭代前先减去单元局部原点 (首个节点坐标),
在单元局部坐标系中迭代。以下测试放回旧实现必须失败。
"""
import numpy as np
import pytest

from fem2d import Mesh
from fem2d.element.q4 import Q4Element, shape_values
from fem2d.stress import point_in_element

_N_POINTS = 1000
_RNG_SEED = 20260803


def _unit_square(offset, size=1.0):
    """单位方形 (或边长 size) 平移到 offset 的 Q4 节点坐标."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float) * size
    return square + offset


def _interior_points(coords, n=_N_POINTS):
    """合法内部点: 随机自然坐标经单元映射的浮点像 (含浮点误差).

    x, y = N @ coords 本身在绝对坐标下求和, 旧实现的定位失败正是
    来自这个含消差噪声的查询点与绝对坐标迭代的残差无法对账。
    """
    rng = np.random.default_rng(_RNG_SEED)
    xi = rng.uniform(-0.9, 0.9, n)
    eta = rng.uniform(-0.9, 0.9, n)
    points = np.empty((n, 2))
    for i in range(n):
        N = shape_values(xi[i], eta[i])
        points[i] = N @ coords
    return points


def _count_failures(coords, points):
    fails = 0
    for x, y in points:
        if Q4Element().shape_values_at(coords, float(x), float(y)) is None:
            fails += 1
    return fails


def _assert_all_located(coords, points):
    """全部定位成功, 且形函数回映目标点 (相对单元尺度)."""
    origin = coords[0]
    local = coords - origin
    scale = max(float(np.ptp(local[:, 0])), float(np.ptp(local[:, 1])),
                np.finfo(float).tiny)
    for x, y in points:
        N = Q4Element().shape_values_at(coords, float(x), float(y))
        assert N is not None, f"内部点 ({x:.17g},{y:.17g}) 定位失败"
        assert N.shape == (4,)
        assert np.isclose(np.sum(N), 1.0, rtol=0.0, atol=1e-12)
        # 回映检查在局部坐标进行 — 绝对坐标求和会重新引入消差噪声
        local_target = np.array([x, y]) - origin
        back = N @ local - local_target
        assert np.linalg.norm(back, ord=np.inf) <= 1e-9 * scale, (
            f"形函数回映误差 {np.linalg.norm(back, ord=np.inf):.3e}"
            f" 超过 1e-9×scale={1e-9 * scale:.3e}")


# ═══════════════════════════════════════════════════════════════
# 判别性: 两个审查场景 (旧实现必须失败)
# ═══════════════════════════════════════════════════════════════

def test_inverse_mapping_unit_square_at_1e12_all_located():
    """单位方形平移到 1e12: 1000 个合法内部点必须全部定位成功.

    旧实现 (绝对坐标迭代) 实测 ~10% 失败。
    """
    coords = _unit_square(1e12)
    points = _interior_points(coords)
    _assert_all_located(coords, points)


def test_inverse_mapping_micro_element_at_large_offset_all_located():
    """坐标 ~1e6、边长 1e-6: 1000 个合法内部点必须全部定位成功.

    旧实现实测 ~2% 失败。
    """
    coords = _unit_square(1e6, size=1e-6)
    points = _interior_points(coords)
    _assert_all_located(coords, points)


# ═══════════════════════════════════════════════════════════════
# 受影响链路: point_in_element → find_containing_element (Q4/Q4I/Q4R)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("elem_type", ["CPS4", "CPS4I", "CPS4R"])
def test_point_in_element_large_offset_all_families(elem_type):
    """Q4 族三单元共享同一逆映射: 1e12 平移网格内部点全部定位."""
    coords = _unit_square(1e12)
    mesh = Mesh(
        nodes=coords, elements=np.array([[0, 1, 2, 3]]),
        E=2.1e11, nu=0.3, thickness=1.0, plane_type="stress",
        elem_type=elem_type,
    )
    rng = np.random.default_rng(_RNG_SEED + 1)
    xi = rng.uniform(-0.9, 0.9, 200)
    eta = rng.uniform(-0.9, 0.9, 200)
    for i in range(200):
        x, y = shape_values(xi[i], eta[i]) @ coords
        assert point_in_element(mesh, float(x), float(y)) == 0
    # 域外点必须拒绝 (旧容差量纲混用曾误判, 复测 2026-08-02)
    assert point_in_element(mesh, 1e12 + 2.0, 1e12 + 0.5) == -1
    assert point_in_element(mesh, 1e12 + 0.5, 1e12 - 1.0) == -1


def test_point_in_element_normal_scale_unchanged():
    """正常尺度行为不变 (回归)."""
    coords = _unit_square(0.0)
    mesh = Mesh(
        nodes=coords, elements=np.array([[0, 1, 2, 3]]),
        E=2.1e11, nu=0.3, elem_type="CPS4",
    )
    assert point_in_element(mesh, 0.25, 0.75) == 0
    assert point_in_element(mesh, 1.5, 0.5) == -1
    assert point_in_element(mesh, -0.5, -0.5) == -1


# ═══════════════════════════════════════════════════════════════
# 高强度自查补充: 微尺度 / 畸形输入 / 形状
# ═══════════════════════════════════════════════════════════════

def test_inverse_mapping_micro_scale_1e150():
    """微尺度几何 (1e-150) 定位正常 — 无绝对阈值."""
    coords = _unit_square(0.0, size=1e-150)
    points = _interior_points(coords)
    _assert_all_located(coords, points)
    assert _count_failures(coords, points) == 0


def test_inverse_mapping_distorted_element_large_offset():
    """扭歪四边形在 1e12 平移下 Newton 仍收敛."""
    coords = np.array([[0, 0], [1.3, -0.2], [1.1, 1.0], [-0.1, 0.9]],
                      dtype=float) + 1e12
    points = _interior_points(coords)
    _assert_all_located(coords, points)
    # 域外点拒绝
    assert Q4Element().shape_values_at(
        coords, float(1e12 + 2.0), float(1e12 + 0.5)) is None
    assert Q4Element().shape_values_at(
        coords, float(1e12 + 0.5), float(1e12 - 1.0)) is None


def test_inverse_mapping_nan_inf_query_rejected():
    """NaN/Inf 查询必须返回 None — 曾静默返回 NaN 形函数被当作单元内."""
    coords = _unit_square(0.0)
    kernel = Q4Element()
    assert kernel.shape_values_at(coords, float("nan"), 0.5) is None
    assert kernel.shape_values_at(coords, 0.5, float("inf")) is None
    assert kernel.shape_values_at(coords, float("inf"), float("-inf")) is None


def test_inverse_mapping_degenerate_element_returns_none():
    """退化 (共线/零面积) 单元: 不在映射像上的查询返回 None, 不崩溃.

    注: 查询点恰好落在退化单元的映射像上时 residual=0 直接跳出,
    Jacobian 检查不会触发 — 与旧实现行为一致, 有意保留。
    """
    kernel = Q4Element()
    collapsed = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    assert kernel.shape_values_at(collapsed, 1.5, 1.0) is None
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 0.0]])
    assert kernel.shape_values_at(line, 0.5, 1.0) is None
