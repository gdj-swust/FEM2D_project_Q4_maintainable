"""契约清账阶段 2 — DOF 校验共享 helper (checks.require_dof_index_array) 判别性测试.

每一条放回旧实现必须失败: mesh/bc 曾各自实现 (bool→不同异常类型、
标量静默接受、str 数组冒裸 TypeError)。
"""
import numpy as np
import pytest

from fem2d.checks import require_dof_index_array
from fem2d.bc import apply_elimination, apply_penalty
from fem2d.mesh import Mesh


# ── helper 自身 ──

def test_dof_helper_rejects_boolean_mask():
    with pytest.raises(TypeError, match="boolean mask"):
        require_dof_index_array(np.array([True, False, True]), "fixed_dofs")


def test_dof_helper_rejects_scalar():
    # 标量曾静默接受 (mesh 0-d 数组广播) — 应为 1-D 数组
    with pytest.raises(ValueError, match="must be 1-D"):
        require_dof_index_array(5, "fixed_dofs")


def test_dof_helper_rejects_nonnumeric_dtype():
    with pytest.raises(ValueError, match="numeric DOF indices"):
        require_dof_index_array(np.array(["a", "b"]), "fixed_dofs")


def test_dof_helper_rejects_nan_noninteger_range():
    with pytest.raises(ValueError, match="NaN/Inf"):
        require_dof_index_array(np.array([0.0, np.nan]), "fixed_dofs")
    with pytest.raises(ValueError, match="must be integers"):
        require_dof_index_array(np.array([0.5, 1.5]), "fixed_dofs")
    with pytest.raises(ValueError, match="out of range"):
        require_dof_index_array(np.array([0, 8]), "fixed_dofs", n_dof=8)
    with pytest.raises(ValueError, match="out of range"):
        require_dof_index_array(np.array([-1]), "fixed_dofs", n_dof=8)


def test_dof_helper_accepts_integer_like_floats():
    out = require_dof_index_array([2.0, 1.0], "fixed_dofs", n_dof=8)
    assert out.dtype == np.int64
    assert out.tolist() == [2, 1]


# ── Mesh 构造/校验路径 ──

def test_mesh_constructor_scalar_fixed_dofs_rejected():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]])
    with pytest.raises(ValueError, match="must be 1-D"):
        Mesh(nodes, elems, fixed_dofs=5)


def test_mesh_constructor_bool_mask_still_typeerror():
    # 既有行为锁定: mesh 布尔掩码 → TypeError (唯一保留的 TypeError 路径)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]])
    with pytest.raises(TypeError, match="boolean mask"):
        Mesh(nodes, elems, fixed_dofs=np.array([True, False, False, False, False, False]))


def test_mesh_validate_state_nan_fixed_dofs():
    m = Mesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
             np.array([[0, 1, 2]]))
    m.fixed_dofs = np.array([0, 1, np.nan], dtype=float)
    with pytest.raises(ValueError, match="NaN/Inf"):
        m.validate_state()


def test_mesh_validate_state_duplicate_fixed_dofs():
    m = Mesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
             np.array([[0, 1, 2]]))
    m.fixed_dofs = np.array([0, 0], dtype=int)
    with pytest.raises(ValueError, match="重复"):
        m.validate_state()


def test_mesh_constructor_float_dofs_normalized():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2]])
    m = Mesh(nodes, elems, fixed_dofs=[0.0, 1.0])
    assert m.fixed_dofs.dtype == np.int64
    assert m.fixed_dofs.tolist() == [0, 1]


# ── bc 路径 (apply_penalty / apply_elimination) ──

def _system():
    K = np.diag([1.0, 2.0, 3.0, 4.0])
    F = np.zeros(4)
    return K, F


def test_apply_penalty_boolean_mask_valueerror():
    # bc 布尔掩码保持既有 ValueError 锁定; helper 经 bool_error 参数分流
    K, F = _system()
    with pytest.raises(ValueError, match="boolean mask"):
        apply_penalty(K, F, np.array([True, False, True, False]))


def test_apply_penalty_scalar_fixed_rejected():
    K, F = _system()
    with pytest.raises(ValueError, match="must be 1-D"):
        apply_penalty(K, F, 2)


def test_apply_penalty_string_dofs_context():
    K, F = _system()
    with pytest.raises(ValueError, match="numeric DOF indices"):
        apply_penalty(K, F, np.array(["dof0", "dof1"]))


def test_apply_penalty_duplicate_same_value_idempotent():
    # 幂等去重保留 (既有行为): 同 DOF 同值 → 去重不报错
    K, F = _system()
    K_mod, F_mod, penalty = apply_penalty(K, F, [2, 2], [0.0, 0.0])
    assert penalty == 1e8 * 4.0
    assert K_mod.shape == (4, 4)


def test_apply_penalty_duplicate_diff_value_rejected():
    K, F = _system()
    with pytest.raises(ValueError, match="重复约束"):
        apply_penalty(K, F, [2, 2], [0.0, 1.0])


def test_apply_elimination_free_dofs_boolean_valueerror():
    K, F = _system()
    with pytest.raises(ValueError, match="boolean mask"):
        apply_elimination(
            K, F, np.array([True, False, True, False]), [0], [0.0])


def test_apply_elimination_partition_gaps_rejected():
    # 遗漏 DOF 曾静默设 0 — 必须报错 (既有行为, 经共享 helper 路径)
    K, F = _system()
    with pytest.raises(ValueError, match="遗漏"):
        apply_elimination(K, F, [0, 2], [1], [0.0])


def test_apply_elimination_free_fixed_overlap_rejected():
    K, F = _system()
    with pytest.raises(ValueError, match="重叠"):
        apply_elimination(K, F, [0, 1], [1, 2, 3], [0.0, 0.0, 0.0])
