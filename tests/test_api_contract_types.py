"""契约清账阶段 2 — 类型与契约校验 (K3/K4/K5/K6/K7) 判别性测试.

每一条放回旧实现必须失败: 裸 IndexError (principal 形状)、裸 KeyError
(estimate result 缺键)、裸 AttributeError (solve/parse_vec2)、gmsh
底层异常透传 (import_msh 缺文件)。
"""
import numpy as np
import pytest

from fem2d.error_est import estimate
from fem2d.gmsh_adapter import import_msh
from fem2d.loads_core import parse_traction, parse_vec2
from fem2d.mesh import Mesh
from fem2d.solver import solve
from fem2d.stress import principal_stresses, stress_at_point


def _mesh():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 3], [0, 3, 2]])
    return Mesh(nodes, elements, E=1e6, nu=0.3, thickness=1.0)


# ── K3: principal_stresses 形状与有限性 ──

def test_principal_stresses_wrong_shape_has_context():
    # (3,) 单向量是合法输入 (2026-08-05 契约扩展); 标量/末维≠3 仍拒绝
    with pytest.raises(ValueError, match="\\(n, 3\\)"):
        principal_stresses(np.ones((2, 2)))
    with pytest.raises(ValueError, match="\\(n, 3\\)"):
        principal_stresses(np.ones(4))
    with pytest.raises(ValueError, match="\\(n, 3\\)"):
        principal_stresses(5.0)


def test_principal_stresses_nan_rejected():
    with pytest.raises(ValueError, match="NaN/Inf"):
        principal_stresses(np.array([[np.nan, 1.0, 0.0]]))


def test_principal_stresses_valid_shape_ok():
    s1, s2, radius, _ = principal_stresses(np.array([[100.0, 50.0, 10.0]]))
    assert np.isfinite(s1) and np.isfinite(s2) and np.isfinite(radius)


def test_von_mises_wrong_shape_has_context():
    # fuzz 发现: 标量/1-D/末维≠3 曾冒裸 IndexError
    # (3,) 单向量是合法输入 (2026-08-05 契约扩展); 标量/末维≠3 仍拒绝
    from fem2d import von_mises
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        von_mises(5.0)
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        von_mises(np.ones(4))
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        von_mises(np.ones((2, 2)))


def test_von_mises_nan_rejected():
    from fem2d import von_mises
    with pytest.raises(ValueError, match="NaN/Inf"):
        von_mises(np.array([[np.nan, 1.0, 0.0]]))


def test_refinement_indicator_missing_key_has_context():
    # 复查轮审计发现: element_refinement_indicator 缺 'stress' 键曾冒裸 KeyError
    from fem2d.error_est import element_refinement_indicator
    m = _mesh()
    with pytest.raises(ValueError, match="'stress'"):
        element_refinement_indicator(m, {"u": np.zeros(8)})
    with pytest.raises(ValueError, match="result"):
        element_refinement_indicator(m, None)


def test_assemble_loads_wrong_ndof_has_context():
    # 复查轮审计: n_dof 不匹配曾裸 IndexError (集中力越界写)
    from fem2d import assemble_loads
    m = _mesh()
    m.add_force(2, 1.0, 0.0)
    with pytest.raises(ValueError, match="n_dof=3"):
        assemble_loads(m, 3)
    with pytest.raises(ValueError, match="n_dof=16"):
        assemble_loads(m, 16)
    # 正确 n_dof 不受影响
    f = assemble_loads(m, 8)
    assert f[2 * 2] == 1.0


# ── K4: estimate_error result 契约 ──

def test_estimate_missing_key_has_context():
    m = _mesh()
    with pytest.raises(ValueError, match="'stress'"):
        estimate(m, {"u": np.zeros(8)})


def test_estimate_non_dict_result_rejected():
    m = _mesh()
    with pytest.raises(ValueError, match="result"):
        estimate(m, None)


def test_estimate_wrong_stress_shape_has_context():
    # 形状错曾一路带到 SPR 才报 — 现有错误已带上下文 (经 spr/恢复入口)
    m = _mesh()
    with pytest.raises(ValueError, match="elem_stress first dimension"):
        estimate(m, {"stress": np.zeros((5, 3))})


def test_stress_at_point_missing_result_key():
    m = _mesh()
    with pytest.raises(ValueError, match="'stress'"):
        stress_at_point(m, {"u": np.zeros(8)}, 0.5, 0.5)


# ── K5: solve 类型契约 ──

def test_solve_non_mesh_rejected():
    with pytest.raises(TypeError, match="mesh 必须是 fem2d.Mesh"):
        solve({})
    with pytest.raises(TypeError, match="mesh 必须是 fem2d.Mesh"):
        solve(None)


def test_solve_valid_mesh_still_works():
    m = _mesh()
    for i in range(4):
        m.fix_node(i, "both", 0.0)
    m.add_force(2, 1.0, 0.0)
    result = solve(m, verbose=False)
    assert np.all(np.isfinite(result["u"]))


# ── K6: 字符串解析器类型契约 ──

def test_parse_vec2_non_string_rejected():
    with pytest.raises(ValueError, match="parse_vec2"):
        parse_vec2(None)
    with pytest.raises(ValueError, match="parse_vec2"):
        parse_vec2(5)


def test_parse_traction_non_string_rejected():
    with pytest.raises(ValueError, match="parse_traction"):
        parse_traction(5)
    with pytest.raises(ValueError, match="parse_traction"):
        parse_traction(None)


# ── K7: import_msh 缺文件前置检查 (无需 gmsh) ──

def test_import_msh_missing_file_filenotfound():
    # 文件不存在曾透传 gmsh 底层异常 — 前置 FileNotFoundError 无 gmsh 依赖
    with pytest.raises(FileNotFoundError, match="Mesh file not found"):
        import_msh("C:/definitely_not_existing_dir_xyz/no.msh")


def test_region_registry_by_name_invalid_dimension_valueerror():
    """by_name 非法维度 → 带参数名 ValueError (pkg11 A14).

    判别性: dimension=5 曾裸 KeyError (collections[int(dimension)]
    越界); 0/1/2/None 正常工作, 越界值必须响亮失败。
    """
    from fem2d.regions import RegionRegistry

    registry = RegionRegistry(points=[])
    with pytest.raises(ValueError, match=r"dimension=5"):
        registry.by_name("x", dimension=5)
    with pytest.raises(ValueError, match="dimension"):
        registry.by_name("x", dimension=-1)
    # 合法维度与 None 不受影响
    assert registry.by_name("x") == []
    assert registry.by_name("x", dimension=0) == []
    assert registry.by_name("x", dimension=2) == []
