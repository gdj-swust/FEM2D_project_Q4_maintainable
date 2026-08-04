"""internal_energy 舍入负值钳 0 测试 (审查修复包第 4 项).

刚体位移 (纯 Dirichlet 均匀给定位移) 时 uᵀKu 精确为 0, 浮点舍入给出
-4.66e-13 级负值 (GOLDEN 曾记录 dirichlet_only = -4.656612873077393e-13)。
旧实现原样返回负内能; 判别性: 放回旧实现 (无钳制) 必须失败。
"""
import numpy as np
import pytest

from fem2d import Mesh
from fem2d.solver import solve


def test_rigid_displacement_internal_energy_clamped_to_zero():
    """纯 Dirichlet 均匀位移 (刚体平移) → internal_energy == 0.0."""
    mesh = Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        E=2.1e11, nu=0.3, thickness=0.01, plane_type="stress",
        elem_type="CPS4")
    for node in range(4):
        mesh.fix_node(node, "both", 1e-3)
    result = solve(mesh, verbose=False)
    assert result["internal_energy"] == 0.0


def test_loaded_model_internal_energy_bit_identical():
    """正常受载模型内能逐位不变 — 钳制只作用于舍入负值, 不影响正值."""
    mesh = Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        E=2.1e11, nu=0.3, thickness=0.01, plane_type="stress",
        elem_type="CPS4")
    for node in (0, 1):
        mesh.fix_node(node, "both", 0.0)
    for node in (2, 3):
        mesh.add_force(node, 0.0, -500.0)
    result = solve(mesh, verbose=False)
    # 与 test_solve_refactor_lock GOLDEN["q4"] 逐位一致
    assert result["internal_energy"] == 0.0002310049019607843
