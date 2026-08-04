"""契约清账阶段 2 — 标量有限性共享 helper (fem2d/checks.py) 判别性测试.

每一条都是判别性测试: 放回旧实现 (np.isfinite 裸调用) 必须失败 —
旧实现要么冒裸 TypeError (非数值类型), 要么静默接受 (NaN 数组输入)。
"""
import numpy as np
import pytest

from fem2d.checks import (
    require_finite_positive,
    require_finite_scalar,
    require_nu_valid,
)
from fem2d.material import D_matrix
from fem2d.mesh import Mesh
from fem2d.spr import spr_recovery
from fem2d.stress import (
    compute_stresses,
    nodal_L2_projection,
    nodal_average,
    nodal_weighted,
)


def _mesh():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 3], [0, 3, 2]])
    return Mesh(nodes, elements, E=1e6, nu=0.3, thickness=1.0)


# ── helper 自身 ──

def test_require_finite_scalar_rejects_non_numeric():
    with pytest.raises(TypeError, match="must be a finite real number"):
        require_finite_scalar("abc", "value")
    with pytest.raises(TypeError, match="value=None"):
        require_finite_scalar(None, "value")
    with pytest.raises(TypeError, match="1\\+2j"):
        require_finite_scalar(1 + 2j, "value")
    with pytest.raises(TypeError, match="\\[1, 2\\]"):
        require_finite_scalar([1, 2], "value")


def test_require_finite_scalar_rejects_nan_inf():
    with pytest.raises(ValueError, match="NaN/Inf"):
        require_finite_scalar(float("nan"), "value")
    with pytest.raises(ValueError, match="NaN/Inf"):
        require_finite_scalar(float("inf"), "value")
    assert require_finite_scalar(np.float64(2.5), "v") == 2.5
    assert require_finite_scalar(1e-310, "v") == 1e-310  # 微尺度值不拒


def test_require_finite_positive_rejects_zero_and_negative():
    with pytest.raises(ValueError, match="must be > 0"):
        require_finite_positive(0.0, "thickness")
    with pytest.raises(ValueError, match="must be > 0"):
        require_finite_positive(-1e6, "E")
    assert require_finite_positive(1e-150, "t") == 1e-150


def test_require_nu_valid_rejects_bounds():
    with pytest.raises(ValueError, match="\\(-1, 0.5\\)"):
        require_nu_valid(0.5, "nu")
    with pytest.raises(ValueError, match="\\(-1, 0.5\\)"):
        require_nu_valid(-1.0, "nu")
    with pytest.raises(ValueError, match="\\(-1, 0.5\\)"):
        require_nu_valid(0.7, "nu")
    assert require_nu_valid(-0.3) == -0.3  # 负泊松比合法


# ── Mesh 标量入口 (旧实现: np.isfinite 裸 TypeError / 无上下文) ──

def test_fix_node_value_non_numeric_has_context():
    m = _mesh()
    with pytest.raises(TypeError, match="fix_node: prescribed displacement"):
        m.fix_node(0, "both", "abc")
    with pytest.raises(TypeError, match="fix_node: prescribed displacement"):
        m.fix_node(0, "both", None)


def test_fix_node_value_nan_inf_rejected():
    m = _mesh()
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="NaN/Inf"):
            m.fix_node(0, "both", bad)


def test_add_force_scalar_checks():
    m = _mesh()
    with pytest.raises(TypeError, match="add_force: fx"):
        m.add_force(0, "abc", 1.0)
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.add_force(0, float("nan"), 1.0)
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.add_force(0, 1.0, float("inf"))


def test_add_traction_scalar_checks():
    m = _mesh()
    with pytest.raises(TypeError, match="add_traction: ty"):
        m.add_traction(0, 1, 1e6, "x")  # callable 短路径不受影响
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.add_traction(0, 1, float("nan"), 1e6)


def test_add_pressure_scalar_checks():
    m = _mesh()
    with pytest.raises(TypeError, match="add_pressure: p"):
        m.add_pressure(0, 1, "x")
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.add_pressure(0, 1, float("nan"))
    # callable 压力不预调用 (形心在材料域外的合法表达式)
    m.add_pressure(0, 1, lambda x, y: 1e6 * x)
    assert len(m.surface_tractions) == 1


def test_mesh_constructor_material_scalars():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]])
    with pytest.raises(TypeError, match="E='abc'"):
        Mesh(nodes, elems, E="abc")
    with pytest.raises(ValueError, match="E=nan"):
        Mesh(nodes, elems, E=float("nan"))
    with pytest.raises(ValueError, match="thickness=0"):
        Mesh(nodes, elems, thickness=0.0)
    # nu 范围检查在 validate_state (求解前) — 构造期仅有限性
    m = Mesh(nodes, elems, nu=0.6)
    with pytest.raises(ValueError, match="\\(-1, 0.5\\)"):
        m.validate_state()


# ── prescribed_vals 入口 (契约 A0; 旧实现: 非 dict 裸 AttributeError / 值字符串裸 TypeError) ──

def test_prescribed_vals_non_dict_rejected():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]])
    # 构造期 + validate_state (构造后可重写) 都拦截
    with pytest.raises(ValueError, match="prescribed_vals must be a dict"):
        Mesh(nodes, elems, fixed_dofs=[0], prescribed_vals=[1.0])
    m = Mesh(nodes, elems, fixed_dofs=[0], prescribed_vals={0: 1.0})
    m.prescribed_vals = [1.0]
    with pytest.raises(ValueError, match="prescribed_vals must be a dict"):
        m.validate_state()


def test_prescribed_vals_non_int_key_rejected():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="prescribed_vals keys"):
        Mesh(nodes, elems, fixed_dofs=[0], prescribed_vals={"x": 1.0})


def test_prescribed_vals_value_checks_have_context():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]])
    # 判别性: 旧实现 np.isfinite 对 str 裸 TypeError (无上下文), NaN 消息带空格
    m = Mesh(nodes, elems, fixed_dofs=[0], prescribed_vals={0: "abc"})
    with pytest.raises(TypeError, match=r"prescribed_vals\[0\]='abc'"):
        m.validate_state()
    m2 = Mesh(nodes, elems, fixed_dofs=[0], prescribed_vals={0: float("nan")})
    with pytest.raises(ValueError, match=r"prescribed_vals\[0\]=nan"):
        m2.validate_state()


# ── 材料 API ──

def test_D_matrix_non_numeric_has_context():
    with pytest.raises(TypeError, match="E='abc'"):
        D_matrix("abc", 0.3)
    with pytest.raises(ValueError, match="NaN/Inf"):
        D_matrix(float("inf"), 0.3)
    with pytest.raises(ValueError, match="\\(-1, 0.5\\)"):
        D_matrix(2.1e11, 0.7)


# ── 恢复/后处理数组有限性 (旧实现: 静默 NaN 输出) ──

def test_compute_stresses_nan_u_rejected():
    m = _mesh()
    with pytest.raises(ValueError, match="u contains NaN/Inf"):
        compute_stresses(m, np.array([float("nan")] * m.n_dof))


def test_nodal_recovery_nan_rejected():
    m = _mesh()
    stress = np.ones((m.n_elements, 3))
    with pytest.raises(ValueError, match="NaN/Inf"):
        nodal_average(m, stress, weights=np.array([1.0, float("nan")]))
    bad = stress.copy()
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        nodal_average(m, bad)
    with pytest.raises(ValueError, match="NaN/Inf"):
        nodal_weighted(m, bad)
    with pytest.raises(ValueError, match="NaN/Inf"):
        nodal_L2_projection(m, bad)
    with pytest.raises(ValueError, match="NaN/Inf"):
        spr_recovery(m, bad)
