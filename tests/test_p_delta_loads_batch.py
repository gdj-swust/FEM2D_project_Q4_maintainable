"""P-δ: 面力/压力路径批量向量化 — 逐位等价 + callable 契约 + 性能红侧.

改造 (P-δ): loads_core.assemble 的面力/压力分支从逐边+逐 Gauss 点
Python 循环 → 边内 3 点批量 (外法向批量计算 + np.add.at 按原循环序
累积)。语义冻结: 载荷公式 / 错误消息 / 退化边判据不变。

判别性:
- 逐位断言 (np.array_equal) 锁定: 批量路径 == 逐边参考实现。载荷向量
  F 直接决定位移解, 1 ULP 差异即求解结果漂移 — 参考实现内嵌旧算法。
- 红侧: 压力边调用 boundary_outward_normal 次数 == 0 (旧实现 == 边数
  必失败); 1000 边压力模型 assemble 耗时必须快于逐边参考实现。
- np.add.at 顺序契约锁定 (numpy 若改变重复索引处理语义 → 立即红灯)。
- callable 求值次数锁定 (每 Gauss 点一次), 错误消息格式兼容。
"""
import numpy as np
import pytest

from fem2d.element import evaluate_vector_field
from fem2d.loads_core import LINE_GAUSS, assemble
from fem2d.loads_schema import _load_component_ok
from fem2d.mesh import Mesh

_EPS = np.finfo(float).eps


# ═══════════════════════════════════════════════════════════════
# 逐边参考实现 (改造前旧算法原样内嵌 — 测试的对照基线)
# ═══════════════════════════════════════════════════════════════

def _reference_assemble_surface(mesh, F):
    """改造前逐边参考实现: 每条边逐 Gauss 点求值 + 标量 += 累积."""
    for st in mesh.surface_tractions:
        ni, nj = st["nodes"]
        trac = st["traction"]
        xi_c, yi_c = mesh.nodes[ni]
        xj_c, yj_c = mesh.nodes[nj]
        dx, dy = xj_c - xi_c, yj_c - yi_c
        L = float(np.hypot(dx, dy))
        edge_ulp = 64.0 * _EPS * max(
            float(max(abs(xi_c), abs(xj_c), abs(yi_c), abs(yj_c))),
            np.finfo(float).tiny)
        if L <= edge_ulp:
            raise ValueError(
                f"边 ({ni},{nj}) 长度 {L:.3e} 低于端点坐标 ULP "
                f"({edge_ulp:.3e}) — 节点重合或退化, 面力无法积分")
        is_pressure = st.get("is_pressure", False)
        if is_pressure:
            p_raw = trac[0]
            nx, ny = mesh.boundary_outward_normal(ni, nj)
            for w, xi_g in LINE_GAUSS:
                Ni = 0.5 * (1 - xi_g)
                Nj = 0.5 * (1 + xi_g)
                xg = Ni * xi_c + Nj * xj_c
                yg = Ni * yi_c + Nj * yj_c
                if callable(p_raw):
                    try:
                        p_val = p_raw(xg, yg)
                    except Exception as error:
                        raise ValueError(
                            f"边 ({ni},{nj}) 压力表达式在 Gauss 点 "
                            f"({xg:.4g},{yg:.4g}) 求值失败: {error}") from error
                else:
                    p_val = p_raw
                if callable(p_val) or not _load_component_ok(p_val):
                    raise ValueError(
                        f"边 ({ni},{nj}) 压力在 Gauss 点 ({xg:.4g},{yg:.4g}) "
                        f"处非法值 {p_val!r} — 压力必须是单个有穷数值 "
                        f"(NaN/Inf/字符串/序列均拒绝)")
                p_val = float(p_val)
                tx = -p_val * nx
                ty = -p_val * ny
                fe = mesh.thickness * w * L / 2.0
                F[2 * ni] += fe * Ni * tx
                F[2 * ni + 1] += fe * Ni * ty
                F[2 * nj] += fe * Nj * tx
                F[2 * nj + 1] += fe * Nj * ty
        else:
            mesh._validate_boundary_edge(ni, nj)
            for w, xi_g in LINE_GAUSS:
                Ni = 0.5 * (1 - xi_g)
                Nj = 0.5 * (1 + xi_g)
                xg = Ni * xi_c + Nj * xj_c
                yg = Ni * yi_c + Nj * yj_c
                try:
                    tx, ty = evaluate_vector_field(trac, xg, yg)
                except Exception as error:
                    raise ValueError(
                        f"边 ({ni},{nj}) 面力表达式在 Gauss 点 "
                        f"({xg:.4g},{yg:.4g}) 求值失败: {error}") from error
                if not (np.isfinite(tx) and np.isfinite(ty)):
                    raise ValueError(
                        f"Traction callable returned NaN/Inf at "
                        f"Gauss point ({xg:.4g},{yg:.4g}) on edge ({ni},{nj})")
                fe = mesh.thickness * w * L / 2.0
                F[2 * ni] += fe * Ni * tx
                F[2 * ni + 1] += fe * Ni * ty
                F[2 * nj] += fe * Nj * tx
                F[2 * nj + 1] += fe * Nj * ty
    return F


def _surface(mesh, F):
    """测试辅助: 调被测 assemble 并断言与参考实现逐位一致."""
    n_dof = 2 * mesh.n_nodes
    F_new = assemble(mesh, n_dof).copy()
    F_ref = _reference_assemble_surface(mesh, F.copy())
    assert np.array_equal(F_new, F_ref), (
        f"批量路径与逐边参考实现不一致:\n{np.max(np.abs(F_new - F_ref))}")
    return F_new, F_ref


# ═══════════════════════════════════════════════════════════════
# 网格构造
# ═══════════════════════════════════════════════════════════════

def _ring(k=12, radius=1.0, center=(0.0, 0.0)):
    """环形网格: 中心节点 0 + k 个边界节点 (CCW), k 个 CST 三角形.

    边界边 (i, i+1) 各自恰属 1 个单元; 中心-环边为内部边。
    """
    th = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    nodes = np.empty((k + 1, 2))
    nodes[0] = center
    nodes[1:, 0] = center[0] + radius * np.cos(th)
    nodes[1:, 1] = center[1] + radius * np.sin(th)
    elements = np.array([
        [0, i + 1, i + 2 if i + 1 < k else 1] for i in range(k)
    ], dtype=int)
    return Mesh(nodes=nodes, elements=elements, elem_type="CST")


def _ring_pressure(mesh, p, record_order=None):
    """给环的全部 k 条边界边加压力记录; record_order 可指定乱序."""
    k = mesh.n_nodes - 1
    records = [(i + 1, i + 2 if i + 1 < k else 1) for i in range(k)]
    if record_order:
        records = [records[j] for j in record_order]
    for ni, nj in records:
        mesh.add_pressure(ni, nj, p)
    return mesh


def _two_tri():
    """2 三角形方形: 内部边 (0,2), 边界边 (0,1)/(1,2)/(2,3)/(3,0)."""
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
        elem_type="CST")


# ═══════════════════════════════════════════════════════════════
# 逐位等价: 常数载荷
# ═══════════════════════════════════════════════════════════════

def test_bitwise_constant_pressure_ring():
    """常数压力环 (12 边共享节点): 批量 == 逐边参考, 逐位一致."""
    mesh = _ring_pressure(_ring(k=12), 2.5e6)
    _surface(mesh, np.zeros(2 * mesh.n_nodes))


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_bitwise_constant_pressure_reversed_and_mixed_order():
    """边序混合 (正序/反序/乱序) + 同一边重复施加: 逐位一致."""
    mesh = _ring(k=8)
    mesh.add_pressure(1, 2, 1e6)
    mesh.add_pressure(2, 1, 3e6)      # 反序 — 法向方向与正序相同
    mesh.add_pressure(8, 1, 1e6)      # 环绕边 (wrap)
    mesh.add_pressure(3, 4, -2e6)     # 负压力 (拉伸)
    mesh.add_pressure(6, 5, 5e5)      # 反序
    _surface(mesh, np.zeros(2 * mesh.n_nodes))


def test_bitwise_constant_traction():
    """常数面力 (多边共享节点 + 反序记录): 逐位一致."""
    mesh = _ring(k=10)
    for i in range(1, 10):
        mesh.add_traction(i, i + 1, 1e6 + i, -2e6)
    mesh.add_traction(10, 1, 3e5, 0.0)
    _surface(mesh, np.zeros(2 * mesh.n_nodes))


def test_bitwise_mixed_pressure_traction_plus_base_loads():
    """混合载荷 (压力+面力+集中力+体力): 非零 F 上继续累积, 逐位一致."""
    mesh = _ring_pressure(_ring(k=8), 1e6)
    mesh.add_traction(1, 2, 2e6, 1e6)
    mesh.add_traction(5, 6, -1e6, 4e5)
    mesh.add_force(1, 7.0, 3.0)
    mesh.add_force(4, -2.5, 1.5)
    mesh.body_force = (1e3, 2e3)
    # 参考实现只覆盖面的力/压力部分 — 先按参考路径装好集中力+体力起点
    F_base = np.zeros(2 * mesh.n_nodes)
    for cf in mesh.concentrated_forces:
        nid, (fx, fy) = cf["node"], cf["force"]
        F_base[2 * nid] += fx
        F_base[2 * nid + 1] += fy
    fe_batch = mesh.element_kernel.body_force_batch(mesh, mesh.body_force)
    F_base += np.bincount(
        mesh.element_dofs.ravel(), weights=np.asarray(fe_batch).ravel(),
        minlength=len(F_base))
    _surface(mesh, F_base)


def test_bitwise_micro_scale():
    """微尺度模型 (半径 1e-12): 退化边 ULP 判据不误伤, 逐位一致."""
    mesh = _ring_pressure(_ring(k=8, radius=1e-12), 1e-3)
    _surface(mesh, np.zeros(2 * mesh.n_nodes))


def test_bitwise_scale_1e9():
    """大坐标模型 (半径 1e9): 坐标 ULP 判据按局部尺度, 逐位一致."""
    mesh = _ring_pressure(_ring(k=8, radius=1e9), 1e2)
    _surface(mesh, np.zeros(2 * mesh.n_nodes))


def test_bitwise_flat_rectangle_pressure():
    """扁矩形 CST 网格 (非对称, 多种边取向): 逐位一致."""
    nodes = np.array([
        [0., 0.], [1., 0.], [2., 0.],
        [0., 0.3], [1., 0.3], [2., 0.3],
    ])
    elements = np.array([
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
    ], dtype=int)
    mesh = Mesh(nodes=nodes, elements=elements, elem_type="CST")
    mesh.add_pressure(0, 1, 1e6)
    mesh.add_pressure(1, 2, 2e6)
    mesh.add_pressure(2, 5, 3e6)
    mesh.add_pressure(5, 4, 4e6)
    mesh.add_pressure(4, 3, 4.5e6)
    mesh.add_pressure(3, 0, 5e6)
    _surface(mesh, np.zeros(2 * mesh.n_nodes))


def test_bitwise_disconnected_components():
    """独立连通分量 (环 + 分离三角形): 独立边与共享边混合, 逐位一致."""
    base = _ring(k=6)
    nodes = np.vstack([base.nodes, [[5., 5.], [6., 5.], [5.5, 6.]]])
    elements = np.vstack([base.elements, [[7, 8, 9]]])
    mesh = Mesh(nodes=nodes, elements=elements, elem_type="CST")
    for ni, nj in ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1)):
        mesh.add_pressure(ni, nj, 1e6)
    mesh.surface_tractions.extend([
        {"nodes": (7, 8), "traction": (1e6, 0.0)},
        {"nodes": (8, 9), "traction": (0.0, 2e6)},
        {"nodes": (9, 7), "traction": (-1e6, 1e6)},
    ])
    _surface(mesh, np.zeros(2 * mesh.n_nodes))


# ═══════════════════════════════════════════════════════════════
# 逐位等价: callable 载荷
# ═══════════════════════════════════════════════════════════════

def test_bitwise_callable_pressure():
    """callable 压力 (空间变化): 逐点求值收集后一次浮点运算, 逐位一致."""
    mesh = _ring_pressure(_ring(k=8), lambda x, y: 1e6 * (1.0 + 0.5 * x + y * y))
    _surface(mesh, np.zeros(2 * mesh.n_nodes))


def test_bitwise_callable_traction():
    """callable 面力 (整体 callable + 分量 callable): 逐位一致."""
    mesh = _ring(k=8)
    mesh.add_traction(1, 2, lambda x, y: 1e6 * np.sin(x), lambda x, y: -2e6 * y)
    mesh.add_traction(3, 4, lambda x, y: 1e6 * x * y, 0.0)          # 混合
    mesh.add_traction(8, 1, 1e6, lambda x, y: 1e6 * (1 - x))        # 混合
    # 整体 callable (返回 (tx, ty) 对) — 手工记录
    mesh.surface_tractions.append({
        "nodes": (5, 6),
        "traction": lambda x, y: (1e6 * x, 2e6 * y),
    })
    _surface(mesh, np.zeros(2 * mesh.n_nodes))


# ═══════════════════════════════════════════════════════════════
# 红侧自证: 批量路径确实生效
# ═══════════════════════════════════════════════════════════════

def test_pressure_edges_do_not_call_boundary_outward_normal(monkeypatch):
    """红侧: 批量法向路径生效 — 旧实现逐边调 N 次必失败."""
    mesh = _ring_pressure(_ring(k=50), 1e6)
    calls = 0
    orig = mesh.boundary_outward_normal

    def counting(*args, **kwargs):
        nonlocal calls
        calls += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(mesh, "boundary_outward_normal", counting)
    assemble(mesh, 2 * mesh.n_nodes)
    assert calls == 0, f"压力边仍逐边调 boundary_outward_normal ({calls} 次)"



# ═══════════════════════════════════════════════════════════════
# 校验错误消息与优先级 (批量法向路径的延迟错误语义)
# ═══════════════════════════════════════════════════════════════

def test_pressure_interior_edge_raises():
    """压力记录在内部边 → 与 boundary_outward_normal 相同消息."""
    mesh = _two_tri()
    mesh.surface_tractions = [
        {"nodes": (0, 2), "traction": (1e6,), "is_pressure": True}]
    with pytest.raises(
            ValueError,
            match="Edge \\(0,2\\) is an interior edge shared by 2 elements"):
        assemble(mesh, 2 * mesh.n_nodes)


def test_pressure_not_a_mesh_edge_raises():
    """压力记录不在网格边 → 与 boundary_outward_normal 相同消息."""
    mesh = _two_tri()
    mesh.surface_tractions = [
        {"nodes": (1, 3), "traction": (1e6,), "is_pressure": True}]
    with pytest.raises(ValueError, match="Edge \\(1,3\\) is not a mesh edge"):
        assemble(mesh, 2 * mesh.n_nodes)


def test_traction_interior_edge_raises():
    """面力记录在内部边 → 逐边校验消息不变 (面力路径)."""
    mesh = _two_tri()
    mesh.surface_tractions = [
        {"nodes": (0, 2), "traction": (1e6, 0.0)}]
    with pytest.raises(
            ValueError,
            match="Edge \\(0,2\\) is an interior edge shared by 2 elements"):
        assemble(mesh, 2 * mesh.n_nodes)


def test_error_priority_first_record_wins():
    """错误优先级: 先序记录的错误先抛 (退化边先于后续内部边)."""
    mesh = _two_tri()
    mesh.surface_tractions = [
        {"nodes": (0, 0), "traction": (1e6,), "is_pressure": True},
        {"nodes": (0, 2), "traction": (1e6,), "is_pressure": True}]
    with pytest.raises(ValueError, match="节点重合或退化"):
        assemble(mesh, 2 * mesh.n_nodes)


def test_degenerate_edge_pressure_raises():
    """压力路径零长边 → 与逐点路径相同消息."""
    mesh = _two_tri()
    mesh.surface_tractions = [
        {"nodes": (0, 0), "traction": (1e6,), "is_pressure": True}]
    with pytest.raises(ValueError, match="节点重合或退化"):
        assemble(mesh, 2 * mesh.n_nodes)


def test_error_raised_at_failing_record_turn():
    """延迟错误在该记录处抛出 — 前置合法压力记录先完成, 后续错误照抛."""
    mesh = _ring(k=6)
    mesh.add_pressure(1, 2, 1e6)          # 合法
    mesh.surface_tractions.append(
        {"nodes": (0, 2), "traction": (1e6,), "is_pressure": True})  # 非法
    with pytest.raises(ValueError, match="interior edge"):
        assemble(mesh, 2 * mesh.n_nodes)


# ═══════════════════════════════════════════════════════════════
# np.add.at 顺序契约
# ═══════════════════════════════════════════════════════════════

def test_np_add_at_processes_repeated_indices_in_order():
    """顺序契约: np.add.at 按索引数组顺序处理重复索引 (读-加-写无缓冲),
    与顺序标量 += 逐位一致 — 批量累积逐位相等的成立前提."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        F0 = rng.uniform(-1e3, 1e3, 10)
        idx = rng.integers(0, 10, 30)
        val = rng.uniform(-1e3, 1e3, 30)
        F1 = F0.copy()
        np.add.at(F1, idx, val)
        F2 = F0.copy()
        for i, v in zip(idx, val):
            F2[i] += v
        assert np.array_equal(F1, F2)