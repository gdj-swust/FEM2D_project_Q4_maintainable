"""P-θ 判别性测试 — 整体 callable 体力每积分点单次求值.

缺陷 (外部审查 2026-08-06): bc_apply._apply_body_force 对整体
callable body 返回分量 lambda 对 — 每个 lambda 独立调用
``body(x, y)``。下游消费方对同一积分点分别调用两个分量 → body
被执行两次; 状态型 body 的两分量是两次不同调用的交叉产物。

判别性 (红侧 ×2 → 修复后 ×1):
- 计数器型: 逐积分点消费 (bfx, bfy) → 调用次数 == 积分点总数 ×1
  (修复前 == ×2); 真实 assemble 积分路径对照每积分点 1 次。
- 状态型: 两分量必须来自同一次调用 (与任何单次完整产物相等,
  而非交叉)。
- 纯函数: 装配结果与参考路径逐位一致 (数值冻结, 零容差断言)。
- 精确缓存: 修复引入的缓存必须逐点精确命中 — 近邻浮点坐标
  (差 1 ulp) 是不同积分点, 各自求值 (禁止近似匹配)。
"""
import numpy as np

from fem2d import Mesh
from fem2d.bc_apply import _apply_body_force
from fem2d.config import AnalysisConfig
from fem2d.loads_core import assemble

# 2×2 单位方板 CPS4 (CCW 节点序) — 4 单元 × 4 Gauss = 16 积分点
_NODES = np.array([
    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    [0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5],
])
_ELEMS = np.array([[0, 1, 5, 4], [1, 2, 6, 5],
                   [2, 3, 7, 6], [3, 0, 4, 7]], dtype=int)


def _quad_mesh():
    """2×2 单位方板 CPS4 网格 (判别性测试专用, 无 gmsh 依赖)."""
    return Mesh(nodes=_NODES.copy(), elements=_ELEMS.copy(),
                E=210e9, nu=0.3, thickness=1.0,
                plane_type="stress", elem_type="CPS4")


def _gauss_points(mesh):
    """元素 kernel 实际使用的积分点 — 与积分路径同款浮点坐标.

    断言"调用次数 == 积分点总数"必须用积分路径真实坐标: 缓存是
    逐点精确命中, 手写近似坐标会使计数与积分路径不一致。
    """
    mesh.build_connectivity()
    N = mesh.q4_N_gauss  # (4 gauss, 4 形函数)
    return [
        tuple(float(v) for v in N[q] @ mesh.nodes[mesh.elements[eid]])
        for eid in range(mesh.n_elements)
        for q in range(N.shape[0])
    ]


def test_counter_body_single_eval_per_gauss_point():
    """计数器型整体 callable 体力: 每积分点恰好求值 1 次 (红侧 ×2 → ×1).

    双重求值落点 = _apply_body_force 返回的 (bfx, bfy) 分量 lambda
    (各自独立调用 body)。按下游消费模式对每个积分点分别调用两个
    分量: 修复前 body 被调用 2×积分点总数 次, 修复后 == 积分点总数。
    再以真实 assemble 积分路径对照: 每积分点 1 次 (数值路径不受影响)。
    """
    counter = [0]

    def body(x, y):
        counter[0] += 1
        return (1.0, -78000.0)

    m = _quad_mesh()
    cfg = AnalysisConfig(body=body)
    bfx, bfy = _apply_body_force(cfg, m, batch_mode=True)
    assert callable(bfx) and callable(bfy)
    points = _gauss_points(m)
    assert len(points) == 16

    # 下游消费模式: 每个积分点分别调用两个分量 lambda
    for xg, yg in points:
        bfx(xg, yg)
        bfy(xg, yg)
    assert counter[0] == len(points), (
        f"整体 callable 体力被双重求值: 调用 {counter[0]} 次, "
        f"期望 {len(points)} 次 (积分点总数 ×1); 修复前为 ×2")

    # 真实积分路径 (loads_core.assemble): 每积分点 1 次, 数值有限
    before = counter[0]
    F = assemble(m, m.n_dof)
    assert counter[0] - before == len(points), (
        f"assemble 积分路径调用 {counter[0] - before} 次, "
        f"期望 {len(points)} 次 (积分点总数 ×1)")
    assert np.isfinite(F).all()


def test_stateful_body_components_from_same_call():
    """状态型整体 callable 体力: 两分量必须来自同一次调用 (红侧交叉).

    body 第 1 次调用返回 (1,0), 第 2 次返回 (0,1) (交替)。修复前
    bfx(x,y) 取第 k 次调用的分量 0, bfy(x,y) 取第 k+1 次调用的分量 1
    — 同一积分点上是两次调用的交叉产物 (1,1), 与任何一次完整产物
    (1,0)/(0,1) 都不相等。修复后两分量共享同一次调用。
    """
    seq = [(1.0, 0.0), (0.0, 1.0)]
    state = [0]

    def body(x, y):
        value = seq[state[0] % 2]
        state[0] += 1
        return value

    m = _quad_mesh()
    cfg = AnalysisConfig(body=body)
    bfx, bfy = _apply_body_force(cfg, m, batch_mode=True)

    x0, y0 = 0.3, 0.4
    bx, by = bfx(x0, y0), bfy(x0, y0)
    assert (bx, by) in seq, (
        f"两分量不是同一次调用的产物: ({bx}, {by}) 为交叉值, "
        f"完整产物只能是 {seq} 之一")
    assert (bx, by) == (1.0, 0.0)
    # 第二个积分点 → 第 2 次调用的完整产物
    x1, y1 = 0.7, 0.2
    assert (bfx(x1, y1), bfy(x1, y1)) == (0.0, 1.0)
    # 同一积分点重复访问 → 缓存命中, 产物不变
    assert (bfx(x0, y0), bfy(x0, y0)) == (1.0, 0.0)


def test_pure_function_body_bitwise_unchanged():
    """纯函数整体 callable 体力: 装配结果逐位一致 (数值冻结).

    修复只消除重复求值, 不得改变载荷数值路径: 经 apply_bcs 装配的
    等效节点力与直接设置 body_force 的参考路径必须逐位相等
    (零容差 — 同一求值顺序与坐标的确定性结果)。
    """
    body = lambda x, y: (x * 1e5, -78000.0 + y * 1e3)  # noqa: E731

    m = _quad_mesh()
    cfg = AnalysisConfig(body=body)
    bfx, bfy = _apply_body_force(cfg, m, batch_mode=True)
    assert callable(bfx) and callable(bfy)
    # 返回对的分量值正确 (显示契约不变)
    assert bfx(0.25, 0.25) == 0.25e5
    assert bfy(0.25, 0.25) == -78000.0 + 0.25e3
    F_apply = assemble(m, m.n_dof)

    m_ref = _quad_mesh()
    m_ref.body_force = body  # 参考路径: 直接透传, 不经 apply
    F_ref = assemble(m_ref, m_ref.n_dof)
    assert np.array_equal(F_apply, F_ref), (
        "修复后装配结果与参考路径不一致 (数值路径被改动)")


def test_body_cache_exact_coordinates_no_approx():
    """精确缓存: 近邻浮点坐标是不同积分点, 各自求值 (禁止近似匹配).

    修复引入的缓存必须逐点精确命中: (x, y) 与 (x+1ulp, y) 是不同的
    键, 各求值 1 次; 同一点第二次访问 → 命中缓存, 不再求值。
    """
    counter = [0]

    def body(x, y):
        counter[0] += 1
        return (1.0, 0.0)

    m = _quad_mesh()
    cfg = AnalysisConfig(body=body)
    bfx, bfy = _apply_body_force(cfg, m, batch_mode=True)

    x0, y0 = 1.0, 2.0
    dx = np.nextafter(x0, x0 + 1.0) - x0  # 1 ulp, 精确可区分
    assert dx > 0.0

    bfx(x0, y0)
    bfy(x0, y0)          # 点 A 两个分量: 共享 1 次求值
    assert counter[0] == 1, f"同一点两分量求值 {counter[0]} 次, 期望 1"
    bfx(x0 + dx, y0)     # 近邻点 B (不同键): 各自求值
    bfy(x0 + dx, y0)
    assert counter[0] == 2, f"近邻积分点被近似合并: {counter[0]} 次, 期望 2"
    bfx(x0, y0)          # 点 A 重复访问: 缓存命中
    bfy(x0, y0)
    assert counter[0] == 2, f"缓存未命中: {counter[0]} 次, 期望 2"
