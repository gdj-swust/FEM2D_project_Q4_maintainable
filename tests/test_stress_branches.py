"""stress.py 分支补测 — 包 2 覆盖率任务.

未覆盖行集中: 恢复路径的输入守卫 (NaN/孤立节点/权重形状)、
stress_at_point 的模式契约/域外拒绝/SPR 缓存与退化边容差。

判别性: 断言具体异常消息/数值结果/返回结构。
"""
import numpy as np
import pytest

from fem2d.mesh import Mesh
from fem2d.stress import (
    compute_stresses,
    nodal_average,
    nodal_L2_projection,
    point_in_element,
    stress_at_point,
    stress_probe,
)


def _tri():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.]]),
        elements=np.array([[0, 1, 2]], dtype=int),
        elem_type="CPS3")


def _two_tri():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        elements=np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        elem_type="CPS3")


# ═══════════════════════════════════════════════════════════════
# compute_stresses / nodal_average 守卫
# ═══════════════════════════════════════════════════════════════

def test_compute_stresses_nan_displacement_rejected():
    """位移含 NaN → ValueError (曾 NaN 应力静默进入云图)."""
    mesh = _tri()
    with pytest.raises(ValueError, match="NaN/Inf"):
        compute_stresses(mesh, np.array([1.0, np.nan, 0.0, 0.0, 0.0, 0.0]))


def test_nodal_average_nan_stress_rejected():
    with pytest.raises(ValueError, match="NaN/Inf"):
        nodal_average(_tri(), np.array([[1.0, 0.0, np.nan]]))


def test_nodal_average_orphan_node_rejected():
    """孤立节点 → ValueError (曾静默填 0, 与 L2 路径行为分叉)."""
    mesh = _tri()
    mesh._elements = np.array([[0, 1, 2]], dtype=int)
    mesh._nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [5., 5.]])
    with pytest.raises(ValueError, match="孤立节点"):
        nodal_average(mesh, np.ones((1, 3)))


def test_nodal_average_weights_shape_rejected():
    with pytest.raises(ValueError, match="weights must have shape"):
        nodal_average(_tri(), np.ones((1, 3)), weights=np.ones(5))


def test_nodal_average_weights_nan_rejected():
    with pytest.raises(ValueError, match="weights contain NaN/Inf"):
        nodal_average(_tri(), np.ones((1, 3)),
                      weights=np.array([np.nan]))


def test_nodal_average_values():
    """面积加权平均: 常数应力场精确恢复 (判别性)."""
    recovered = nodal_average(_two_tri(), np.ones((2, 3)),
                              weights="area")
    assert recovered.shape == (4, 3)
    assert np.allclose(recovered, 1.0)


# ═══════════════════════════════════════════════════════════════
# nodal_L2_projection 守卫
# ═══════════════════════════════════════════════════════════════

def test_l2_projection_ndim_rejected():
    with pytest.raises(ValueError, match="must have shape"):
        nodal_L2_projection(_tri(), np.zeros((1, 2, 2, 3)))


def test_l2_projection_shape_matrix_mismatch():
    """形状矩阵节点数 ≠ 单元节点数 → ValueError."""
    mesh = _tri()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        mesh.element_kernel, "recovery_shape_matrix",
        lambda m: np.ones((1, 2)))        # 2 ≠ 3 节点
    monkeypatch.setattr(
        mesh.element_kernel, "recovery_weights",
        lambda m: np.ones((1, 1)))
    # 计数对账: 形状/权重与 quadrature 点数一致才进入形状检查
    monkeypatch.setattr(
        mesh.element_kernel, "recovery_quadrature",
        lambda m, eid: (np.ones((1, 3)), np.ones(1)))
    with pytest.raises(ValueError, match="shape matrix has shape"):
        nodal_L2_projection(mesh, np.ones((1, 3)))
    monkeypatch.undo()


def test_l2_projection_weights_shape_mismatch():
    """恢复权重形状错误 → ValueError (曾静默广播错位)."""
    mesh = _tri()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        mesh.element_kernel, "recovery_shape_matrix",
        lambda m: np.ones((1, 3)))
    monkeypatch.setattr(
        mesh.element_kernel, "recovery_weights",
        lambda m: np.ones((5, 1)))        # (ne=1, nqp=1) 期望
    monkeypatch.setattr(
        mesh.element_kernel, "recovery_quadrature",
        lambda m, eid: (np.ones((1, 3)), np.ones(1)))
    with pytest.raises(ValueError, match="recovery weights have shape"):
        nodal_L2_projection(mesh, np.ones((1, 3)))
    monkeypatch.undo()


# ═══════════════════════════════════════════════════════════════
# stress_at_point
# ═══════════════════════════════════════════════════════════════

def test_stress_at_point_unknown_mode_rejected():
    mesh = _tri()
    result = {"stress": np.ones((1, 3))}
    with pytest.raises(ValueError, match="Unknown stress query mode"):
        stress_at_point(mesh, result, 0.2, 0.2, mode="bogus")


def test_stress_at_point_result_contract():
    mesh = _tri()
    with pytest.raises(ValueError, match="必须是 solve"):
        stress_at_point(mesh, 0.2, 0.2, {"u": np.zeros(6)})


def test_stress_at_point_outside_mesh():
    mesh = _tri()
    result = {"stress": np.ones((1, 3))}
    with pytest.raises(ValueError, match="not in mesh"):
        stress_at_point(mesh, result, 5.0, 5.0)


def test_stress_at_point_element_mode():
    mesh = _tri()
    result = {"stress": np.array([[2.0, 3.0, 4.0]])}
    assert stress_at_point(mesh, result, 0.2, 0.2, mode="element")[0] == 2.0


def test_stress_at_point_recovered_shape_none_fallback(monkeypatch):
    """内核无形状函数 → recovered 回退到单元代表应力.

    注: find_containing_element 内部也走 shape_values_at, 必须同时
    固定单元定位 (否则补丁让所有点都判'不在网格内').
    """
    mesh = _tri()
    result = {"stress": np.array([[2.0, 3.0, 4.0]])}
    monkeypatch.setattr(mesh.element_kernel, "find_containing_element",
                        lambda m, x, y: 0)
    monkeypatch.setattr(mesh.element_kernel, "shape_values_at",
                        lambda *a, **k: None)
    out = stress_at_point(mesh, result, 0.25, 0.25, mode="recovered")
    assert out[0] == 2.0


def test_stress_at_point_sides_on_internal_edge():
    """内部边中点 → 双侧应力元组 (mode='sides')."""
    mesh = _two_tri()
    result = {"stress": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])}
    first, second = stress_at_point(mesh, result, 0.5, 0.5, mode="sides")
    assert {first[0], second[0]} == {1.0, 2.0}


def test_stress_at_point_average_on_internal_edge():
    """内部边中点 → 两侧算术平均 (mode='average')."""
    mesh = _two_tri()
    result = {"stress": np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])}
    out = stress_at_point(mesh, result, 0.5, 0.5, mode="average")
    assert out[0] == pytest.approx(1.5)


def test_stress_at_point_recovered_cache_used(monkeypatch):
    """recovered 模式: SPR 缓存建一次, 插值返回 (判别性数值)."""
    mesh = _two_tri()
    stress = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    result = {"stress": stress}
    out = stress_at_point(mesh, result, 0.25, 0.25, mode="recovered")
    assert "_spr_cache" in result          # 缓存已建立
    assert np.allclose(out, [1.0, 0.0, 0.0])   # 常场 SPR 精确恢复


def test_stress_at_point_recovered_stress_qp_none():
    """CST 输出契约 stress_qp=None → recovered 回退单元应力 (曾 TypeError)."""
    mesh = _two_tri()
    stress = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = {"stress": stress, "stress_qp": None}
    out = stress_at_point(mesh, result, 0.25, 0.25, mode="recovered")
    assert "_spr_cache" in result
    assert out.shape == (3,)


def test_stress_probe_rows_numeric():
    """stress_probe 行装配: 常场下两口径同值, 解析值 [1,0,0,1,0,1]."""
    mesh = _two_tri()
    result = {"stress": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
    e_row, r_row = stress_probe(mesh, result, 0.25, 0.25)
    expect = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]  # s1=1, s2=0, vm=1
    assert np.allclose(e_row, expect)
    assert np.allclose(r_row, expect)


def test_stress_probe_single_lookup_and_eid_passthrough(monkeypatch):
    """stress_probe 两口径只做一次 point_in_element 定位 (曾各查一遍,
    交互探针每次点击付两次 kernel 查询); _eid 透传输出与旧路径一致."""
    mesh = _two_tri()
    result = {"stress": np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])}
    calls = {"n": 0}
    real = point_in_element

    def counting(m, x, y):
        calls["n"] += 1
        return real(m, x, y)

    monkeypatch.setattr("fem2d.stress.point_in_element", counting)
    e_row, r_row = stress_probe(mesh, result, 0.25, 0.25)
    assert calls["n"] == 1, f"point_in_element 应调用 1 次, 实际 {calls['n']}"
    expect = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
    assert np.allclose(e_row, expect)
    assert np.allclose(r_row, expect)


def test_stress_at_point_eid_outside_mesh_raises():
    """_eid=-1 (定位域外) 与未传 _eid 同契约: "not in mesh" ValueError."""
    mesh = _tri()
    result = {"stress": np.ones((1, 3))}
    with pytest.raises(ValueError, match="not in mesh"):
        stress_at_point(mesh, result, 5.0, 5.0, _eid=-1)
