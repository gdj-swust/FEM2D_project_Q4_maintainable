"""loads_core.py 防御分支补测 — 包 2 覆盖率任务.

未覆盖行集中: assemble 的 kernel 形状/NaN 守卫、退化边、压力/面力
表达式求值失败上下文、parse_traction 分布类型白名单、_compile_expr
AST 白名单各拒绝分支、make_edge_profile_func 早期返回。

判别性: 每条断言具体异常类型/消息内容/数值结果。
"""
import numpy as np
import pytest

from fem2d.loads_core import (
    _compile_expr,
    _profile_factor,
    assemble,
    make_edge_profile_func,
    parse_traction,
    parse_vec2,
)
from fem2d.mesh import Mesh


def _quad(body_force=None, surface_tractions=None):
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        elem_type="CPS4",
        body_force=body_force,
        surface_tractions=surface_tractions)


# ═══════════════════════════════════════════════════════════════
# assemble: kernel 形状/有限性守卫
# ═══════════════════════════════════════════════════════════════

def test_assemble_body_batch_shape_mismatch_raises(monkeypatch):
    """批量体力 kernel 返回错误形状 → RuntimeError 带期望形状."""
    mesh = _quad(body_force=(0.0, 0.0))
    monkeypatch.setattr(
        mesh.element_kernel, "body_force_batch",
        lambda m, bf: np.zeros((3, 4)))
    with pytest.raises(RuntimeError, match="body-force shape"):
        assemble(mesh, 8)


def test_assemble_body_batch_nan_raises(monkeypatch):
    """批量体力含 NaN/Inf → ValueError (静默 NaN 载荷会进求解)."""
    mesh = _quad(body_force=(0.0, 0.0))
    monkeypatch.setattr(
        mesh.element_kernel, "body_force_batch",
        lambda m, bf: np.full((1, 8), np.nan))
    with pytest.raises(ValueError, match="NaN/Inf"):
        assemble(mesh, 8)


def test_assemble_body_scalar_shape_mismatch_raises(monkeypatch):
    """标量体力路径 (kernel 无批量实现) 形状错误 → RuntimeError."""
    mesh = _quad(body_force=(0.0, 0.0))
    monkeypatch.setattr(
        mesh.element_kernel, "body_force_batch", lambda m, bf: None)
    monkeypatch.setattr(
        mesh.element_kernel, "body_force_vector",
        lambda m, eid, bf: np.zeros(5))
    with pytest.raises(RuntimeError, match="body-force shape"):
        assemble(mesh, 8)


def test_assemble_degenerate_edge_raises():
    """零长边 (节点重合) → ValueError, 面力无法积分."""
    mesh = Mesh(
        nodes=np.array([[0., 0.], [0., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        elem_type="CPS4",
        surface_tractions=[{"nodes": (0, 1), "traction": (1e6, 0.0)}])
    with pytest.raises(ValueError, match="节点重合或退化"):
        assemble(mesh, 8)


def test_assemble_pressure_callable_nan_raises():
    """压力表达式返回 NaN → ValueError 带 Gauss 点坐标."""
    def _nan_pressure(x, y):
        return np.nan
    mesh = _quad(surface_tractions=[
        {"nodes": (0, 1), "traction": (_nan_pressure,),
         "is_pressure": True}])
    with pytest.raises(ValueError, match="NaN/Inf"):
        assemble(mesh, 8)


def test_assemble_traction_callable_raise_wrapped():
    """面力表达式除零等异常 → ValueError 带边/点上下文 (曾裸抛)."""
    def _boom(x, y):
        raise ZeroDivisionError("1/x")
    mesh = _quad(surface_tractions=[
        {"nodes": (0, 1), "traction": (_boom, 0.0)}])
    with pytest.raises(ValueError, match="面力表达式在 Gauss 点"):
        assemble(mesh, 8)


def test_assemble_traction_callable_nan_wrapped():
    """面力表达式返回 NaN → evaluate_vector_field 抛错 → 包装带边上下文.

    注: evaluate_vector_field 自检 NaN/Inf 先抛, assemble 内层的
    isfinite 复检 (151-154) 不可达 — 双保险死防御, 此处锁定实际路径.
    """
    def _nan(x, y):
        return np.nan, 0.0
    mesh = _quad(surface_tractions=[
        {"nodes": (0, 1), "traction": (_nan, 0.0)}])
    with pytest.raises(ValueError, match="面力表达式在 Gauss 点"):
        assemble(mesh, 8)


def test_assemble_traction_integral_value():
    """常数面力 tx 均匀 → 等效节点力 = tx*t*L/2 每端 (守恒)."""
    mesh = _quad(surface_tractions=[
        {"nodes": (0, 1), "traction": (1e6, 0.0)}])
    F = assemble(mesh, 8)
    # 边 (0,1) 长 1, t=1 → 总力 1e6 均分两端 x 方向
    assert F[0] == pytest.approx(5e5, rel=1e-12)
    assert F[2] == pytest.approx(5e5, rel=1e-12)
    assert F[1] == 0.0 and F[3] == 0.0


def test_assemble_concentrated_force_and_body():
    """集中力 + 常数体力 → 等效节点力叠加."""
    mesh = Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        elem_type="CPS4",
        body_force=(10.0, 0.0),
        concentrated_forces=[{"node": 2, "force": (7.0, 3.0)}])
    F = assemble(mesh, 8)
    assert F[4] == pytest.approx(7.0 + 10.0 / 4, rel=1e-12)  # 体力均分四节点
    assert F[5] == pytest.approx(3.0, rel=1e-12)


# ═══════════════════════════════════════════════════════════════
# parse_traction 分布类型白名单
# ═══════════════════════════════════════════════════════════════

def test_parse_traction_no_colon_returns_none():
    """无冒号 → (None, 0, 0, None) — 非面力规格."""
    assert parse_traction("right") == (None, 0, 0, None)


def test_parse_traction_constant_and_profiles():
    assert parse_traction("right:1e6,0") == ("right", 1e6, 0.0, None)
    assert parse_traction("right:1e6,0:p") == ("right", 1e6, 0.0, "p")
    assert parse_traction("right:1e6,0:l") == ("right", 1e6, 0.0, "l")
    edge, tx, ty, profile = parse_traction("right:2e5:n")
    assert (edge, profile) == ("right", "n") and tx == 2e5 and ty == 0.0


def test_parse_traction_invalid_profile_rejected():
    """'x' 分布类型 → ValueError (曾静默按常数处理, 载荷失真)."""
    with pytest.raises(ValueError, match="分布类型 'x' 无效"):
        parse_traction("right:1e6,0:x")


def test_parse_traction_invalid_pressure_value_rejected():
    with pytest.raises(ValueError, match="法向压力值无效"):
        parse_traction("right:abc:n")


def test_parse_traction_too_many_parts_rejected():
    with pytest.raises(ValueError, match="面力格式无效"):
        parse_traction("right:1e6,0:p:extra")


# ═══════════════════════════════════════════════════════════════
# _profile_factor / make_edge_profile_func
# ═══════════════════════════════════════════════════════════════

def test_profile_factor_values():
    """p 抛物线 (端点 0 中心 1), l 线性, 其余恒 1, 坐标夹到 [0,1]."""
    assert _profile_factor("p", 0.5) == pytest.approx(1.0)
    assert _profile_factor("p", 0.0) == pytest.approx(0.0)
    assert _profile_factor("l", 0.0) == pytest.approx(0.0)
    assert _profile_factor("l", 0.5) == pytest.approx(0.5)
    assert _profile_factor("l", 2.0) == pytest.approx(1.0)   # 夹紧
    assert _profile_factor(None, 0.5) == 1.0
    assert _profile_factor("n", 0.5) == 1.0


def test_make_edge_profile_none_returns_constant():
    """无分布 → 原样返回 (不包装)."""
    fx, fy = make_edge_profile_func(1e6, 2e6, None, (0, 0), (1, 0), 0.0, 1.0)
    assert fx == 1e6 and fy == 2e6


def test_make_edge_profile_degenerate_edge_returns_constant():
    """零长边 → 原样返回 (分布无法定义)."""
    fx, fy = make_edge_profile_func(
        1e6, 2e6, "l", (1, 1), (1, 1), 0.0, 1.0)
    assert fx == 1e6 and fy == 2e6


def test_make_edge_profile_linear_values():
    """线性分布: 起点 0, 终点 1, 中点 0.5 (弧长坐标归一化)."""
    fx, fy = make_edge_profile_func(
        100.0, 200.0, "l", (0, 0), (1, 0), 0.0, 1.0)
    assert fx(0.0, 0.0) == pytest.approx(0.0)
    assert fx(0.5, 0.0) == pytest.approx(50.0)
    assert fx(1.0, 0.0) == pytest.approx(100.0)
    assert fy(1.0, 0.0) == pytest.approx(200.0)


def test_make_edge_profile_callable_component():
    """表达式面力 × 弧长分布: f = tx(x,y)·s(arc) (曾对 callable 抛 TypeError)."""
    fx, fy = make_edge_profile_func(
        lambda x, y: 10.0 * x, 0.0, "l", (0, 0), (1, 0), 0.0, 1.0)
    assert fx(1.0, 0.0) == pytest.approx(10.0)
    assert fx(0.5, 0.0) == pytest.approx(2.5)


# ═══════════════════════════════════════════════════════════════
# _compile_expr: AST 白名单
# ═══════════════════════════════════════════════════════════════

def test_compile_expr_valid_sin():
    f = _compile_expr("sin(pi*x/2)")
    assert f(1.0, 0.0) == pytest.approx(1.0, rel=1e-12)
    assert f(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_compile_expr_syntax_error_message():
    with pytest.raises(ValueError, match="表达式语法错误"):
        _compile_expr("1e6x,0")


def test_compile_expr_disallowed_binop():
    with pytest.raises(ValueError, match="不允许该运算符"):
        _compile_expr("x % 2")


def test_compile_expr_disallowed_unaryop():
    with pytest.raises(ValueError, match="不允许该一元运算符"):
        _compile_expr("~x")


def test_compile_expr_disallowed_name():
    with pytest.raises(ValueError, match="不允许使用 'z'"):
        _compile_expr("z + 1")


def test_compile_expr_disallowed_constant():
    with pytest.raises(ValueError, match="仅允许数字常量"):
        _compile_expr("'abc'")


def test_compile_expr_disallowed_call():
    with pytest.raises(ValueError, match="不允许该函数调用"):
        _compile_expr("foo(1)")


def test_compile_expr_disallowed_keywords():
    with pytest.raises(ValueError, match="不允许关键字参数"):
        _compile_expr("sin(x=1)")


def test_compile_expr_unsupported_node():
    with pytest.raises(ValueError, match="不支持的操作"):
        _compile_expr("x[0]")


# ═══════════════════════════════════════════════════════════════
# parse_vec2
# ═══════════════════════════════════════════════════════════════

def test_parse_vec2_constants():
    assert parse_vec2("1e6,0") == (1e6, 0.0)
    assert parse_vec2("1e6,") == (1e6, 0.0)     # 空分量 → 0.0


def test_parse_vec2_wrong_count_rejected():
    with pytest.raises(ValueError, match="恰好两个分量"):
        parse_vec2("1e6")


def test_parse_vec2_expression_component():
    fx, fy = parse_vec2("0,-1000*(1-y/2)")
    assert fx == 0.0                            # 纯数字 → float
    assert callable(fy)                         # 含 y → callable
    assert fy(0.0, 2.0) == pytest.approx(0.0)
    assert fy(0.0, 0.0) == pytest.approx(-1000.0)
