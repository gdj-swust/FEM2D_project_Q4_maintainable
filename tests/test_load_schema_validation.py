"""载荷 schema 统一校验 (P2-4 外部审查).

覆盖: 四种载荷 (体力/面力/压力/集中力) 的容器形状与分量校验 —
多余分量曾静默忽略, 单分量/标量曾裸 IndexError/TypeError 冒出。
判别性: 放回旧实现 (无形状校验) 时本文件各用例必须失败。

合法形状 (集中定义于 mesh._check_load_pair / _check_load_scalar):
  body_force:          None | callable | 恰好 2 个分量 (数值或 callable)
  surface_tractions:   普通面力 (tx, ty) 恰好 2 个分量 | 压力 (p,) 恰好 1 个标量
  concentrated_forces: (fx, fy) 恰好 2 个数值分量
"""
import numpy as np
import pytest

from fem2d import Mesh, solve

# 3 节点单 CST 单元: 所有边均为边界边, 可加面力/压力
_NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_ELEMS = np.array([[0, 1, 2]])


def _mesh(**kw):
    return Mesh(nodes=_NODES, elements=_ELEMS, E=210e9, nu=0.3,
                thickness=1.0, plane_type="stress", **kw)


# ═══════════════════════════════════════════════════════════════
# 体力 — 恰好 2 个分量
# ═══════════════════════════════════════════════════════════════

def test_body_force_extra_component_rejected():
    """3 分量体力第三分量曾静默忽略 — 必须响亮报错并带原始值."""
    m = _mesh()
    m.body_force = (1.0, 2.0, 999.0)
    with pytest.raises(ValueError, match=r"body_force.*exactly 2.*999"):
        m.validate_state()


def test_body_force_scalar_rejected_with_context():
    """标量体力曾裸 TypeError ('float' not subscriptable) — 须带上下文 ValueError."""
    m = _mesh()
    m.body_force = 5.0
    with pytest.raises(ValueError, match="body_force.*2-component"):
        m.validate_state()


def test_body_force_single_component_rejected():
    """单分量体力曾通过校验、装配时裸 IndexError — 校验入口必须拦截."""
    m = _mesh()
    m.body_force = (1.0,)
    with pytest.raises(ValueError, match=r"body_force.*exactly 2"):
        m.validate_state()


def test_body_force_nan_component_rejected():
    """非有限分量须带字段名与下标拒绝 (曾延迟到装配/静默)."""
    m = _mesh()
    m.body_force = (np.nan, 0.0)
    with pytest.raises(ValueError, match=r"body_force\[0\].*finite"):
        m.validate_state()


# ═══════════════════════════════════════════════════════════════
# 普通面力 — 恰好 2 个分量 (tx, ty)
# ═══════════════════════════════════════════════════════════════

def test_traction_extra_component_rejected():
    """3 分量面力第三分量曾静默忽略 — 必须响亮报错."""
    m = _mesh()
    m.surface_tractions = [{"nodes": (0, 1), "traction": (1e6, 2e6, 999.0)}]
    with pytest.raises(ValueError, match=r"surface_tractions\[0\].*exactly 2.*999"):
        m.validate_state()


def test_traction_single_component_rejected():
    """单分量面力 (缺 ty) 曾裸 ValueError — 须带字段上下文."""
    m = _mesh()
    m.surface_tractions = [{"nodes": (0, 1), "traction": (1e6,)}]
    with pytest.raises(ValueError, match=r"surface_tractions\[0\].*exactly 2"):
        m.validate_state()


def test_traction_whole_callable_rejected():
    """整体 callable 面力不属于 schema (面力 = 恰好 2 个分量) — 明确拒绝."""
    m = _mesh()
    m.surface_tractions = [{"nodes": (0, 1), "traction": lambda x, y: (x, y)}]
    with pytest.raises(ValueError, match=r"surface_tractions\[0\].*2-component"):
        m.validate_state()


def test_traction_missing_key_rejected():
    """记录缺 traction 键曾裸 KeyError — 须带上下文 ValueError."""
    m = _mesh()
    m.surface_tractions = [{"nodes": (0, 1)}]
    with pytest.raises(ValueError, match=r"surface_tractions\[0\].*missing key"):
        m.validate_state()


def test_traction_nodes_wrong_shape_rejected():
    """nodes 非节点对曾裸 unpack ValueError — 须带字段上下文."""
    m = _mesh()
    m.surface_tractions = [{"nodes": (0, 1, 2), "traction": (1e6, 0.0)}]
    with pytest.raises(ValueError, match=r"surface_tractions\[0\].*node pair"):
        m.validate_state()


def test_traction_extra_component_rejected_at_solve():
    """端到端: 多余分量必须在 solve 时拦截 (曾静默求解出错误载荷)."""
    m = _mesh()
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.surface_tractions = [{"nodes": (1, 2), "traction": (1e6, 0.0, 5.0)}]
    with pytest.raises(ValueError, match=r"exactly 2"):
        solve(m, method="elimination", verbose=False)


# ═══════════════════════════════════════════════════════════════
# 压力 — 恰好 1 个标量
# ═══════════════════════════════════════════════════════════════

def test_pressure_extra_component_rejected():
    """压力 2 分量第二分量曾静默忽略 (消费方只取 trac[0]) — 必须报错."""
    m = _mesh()
    m.surface_tractions = [{"nodes": (0, 1), "traction": (5e5, 999.0),
                            "is_pressure": True}]
    with pytest.raises(ValueError, match=r"exactly 1.*999"):
        m.validate_state()


def test_pressure_3_element_rejected():
    m = _mesh()
    m.surface_tractions = [{"nodes": (0, 1), "traction": (5e5, 0.0, 1.0),
                            "is_pressure": True}]
    with pytest.raises(ValueError, match=r"exactly 1"):
        m.validate_state()


# ═══════════════════════════════════════════════════════════════
# 集中力 — 恰好 2 个数值分量
# ═══════════════════════════════════════════════════════════════

def test_concentrated_force_extra_component_rejected():
    """3 分量集中力曾裸 unpack ValueError (无字段名) — 须带上下文."""
    m = _mesh()
    m.concentrated_forces = [{"node": 1, "force": (1e3, 2e3, 999.0)}]
    with pytest.raises(ValueError,
                       match=r"concentrated_forces\[0\].*exactly 2.*999"):
        m.validate_state()


def test_concentrated_force_missing_key_rejected():
    """记录缺 force 键曾裸 KeyError — 须带上下文 ValueError."""
    m = _mesh()
    m.concentrated_forces = [{"node": 1}]
    with pytest.raises(ValueError,
                       match=r"concentrated_forces\[0\].*missing key"):
        m.validate_state()


def test_concentrated_force_callable_component_rejected():
    """集中力不支持 callable 分量 (add_force 仅数值) — 明确拒绝."""
    m = _mesh()
    m.concentrated_forces = [{"node": 1, "force": (1e3, lambda x, y: 1.0)}]
    with pytest.raises(ValueError, match=r"concentrated_forces\[0\].*callable"):
        m.validate_state()


def test_zero_dim_ndarray_body_force_rejected():
    """0-d ndarray 体力的 len() 曾裸 TypeError — 须带上下文 ValueError."""
    m = _mesh()
    m.body_force = np.array(5.0)
    with pytest.raises(ValueError, match=r"body_force.*exactly 2"):
        m.validate_state()


def test_zero_dim_ndarray_pressure_rejected():
    m = _mesh()
    m.surface_tractions = [{"nodes": (0, 1), "traction": np.array(5e5),
                            "is_pressure": True}]
    with pytest.raises(ValueError, match=r"exactly 1"):
        m.validate_state()


def test_zero_dim_ndarray_nodes_rejected():
    m = _mesh()
    m.surface_tractions = [{"nodes": np.array(0), "traction": (1e6, 0.0)}]
    with pytest.raises(ValueError, match=r"node pair"):
        m.validate_state()


# ═══════════════════════════════════════════════════════════════
# 尺度不变性 — 校验不得引入绝对阈值
# ═══════════════════════════════════════════════════════════════

def test_load_validation_scale_invariant():
    """微尺度 (1e-150 几何 / 1e-310 载荷) 与大坐标 (1e12) 下合法载荷均通过 —
    校验只做 float 转换 + isfinite, 无绝对阈值."""
    for scale, node_scale in ((1e-150, 1e-150), (1e12, 1e12)):
        m = Mesh(nodes=_NODES * node_scale, elements=_ELEMS, E=210e9, nu=0.3,
                 thickness=1.0, plane_type="stress")
        m.body_force = (0.0, -78000.0 * scale)
        m.surface_tractions = [
            {"nodes": (0, 1), "traction": (1e-310, 0.0)},
            {"nodes": (1, 2), "traction": (-5e-310,), "is_pressure": True},
        ]
        m.concentrated_forces = [{"node": 1, "force": (1e-310, 2e-310)}]
        m.validate_state()   # 不抛


def test_concentrated_force_nan_rejected_with_field():
    """非有限分量须带字段名与下标 (曾 'concentrated force contains NaN/Inf')."""
    m = _mesh()
    m.concentrated_forces = [{"node": 1, "force": (1e3, np.inf)}]
    with pytest.raises(ValueError,
                       match=r"concentrated_forces\[0\]\['force'\]\[1\].*finite"):
        m.validate_state()


# ═══════════════════════════════════════════════════════════════
# 合法形状回归 — 校验不得破坏生产路径
# ═══════════════════════════════════════════════════════════════

def test_legal_load_shapes_pass_and_normalize():
    """全部合法形状通过: callable 体力 / 混合分量元组 / 表达式面力 /
    标量与 1 元组压力 / 字符串数值分量 / ndarray 分量."""
    m = _mesh()
    m.body_force = (0.0, lambda x, y: -78000.0 * (1.0 + x / 2.0))  # 混合元组
    m.surface_tractions = [
        {"nodes": (0, 1), "traction": (1e6, lambda x, y: 0.0)},    # profile 面力
        {"nodes": (1, 2), "traction": (5e5,), "is_pressure": True},  # 压力 1 元组
        {"nodes": (0, 2), "traction": 2e6, "is_pressure": True},   # 压力裸标量
        {"nodes": (0, 1), "traction": ("1e6", "0")},               # 字符串数值
        {"nodes": (1, 2), "traction": np.array([0.0, 3e5])},       # ndarray 分量
    ]
    m.concentrated_forces = [{"node": 1, "force": (100.0, 200.0)}]
    m.validate_state()   # 不抛
    # 裸标量压力规范化写回 1 元组 (消费方统一 trac[0])
    assert m.surface_tractions[2]["traction"] == (2e6,)


def test_whole_callable_body_force_still_allowed():
    """整体 callable 体力保持合法 — 返回契约在真实 Gauss 点检查,
    不入 validate_state 预调用 (带孔/凹域形心在材料域外)."""
    m = _mesh()
    m.body_force = lambda x, y: (1000.0 * x, -500.0)
    m.validate_state()   # 不抛 (Gauss 点契约由 loads_core 检查)


def test_legal_shapes_survive_full_solve():
    """合法载荷形状端到端: 混合 callable 体力 + 压力 + 面力 + 集中力."""
    m = _mesh()
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.body_force = (0.0, -78000.0)
    m.surface_tractions = [{"nodes": (1, 2), "traction": (1e6, 0.0)}]
    m.surface_tractions.append(
        {"nodes": (0, 2), "traction": (5e5,), "is_pressure": True})
    m.concentrated_forces = [{"node": 2, "force": (1e3, 0.0)}]
    r = solve(m, method="elimination", verbose=False)
    assert np.all(np.isfinite(r["u"]))
