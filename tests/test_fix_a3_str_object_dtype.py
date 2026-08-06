"""A3 (P3, R-α): str/object 单元索引与应力数组 → 带参数名 TypeError.

审查现象 (2026-08-05): np.isfinite 对 str/object 数组抛裸 TypeError
("ufunc 'isfinite' not supported"), 材料参数侧已修 (require_finite_scalar),
此处 (mesh elements / von_mises / principal_stresses / nodal_average /
spr_recovery) 仍原样。

修复: isfinite/转换前先做 dtype 检查 — 非数值 → TypeError 带参数名;
NaN/Inf → ValueError (既有行为不变)。complex 数组在同族入口 → ValueError。

判别性: 回滚本提交 → 裸 TypeError / numpy 裸转换错误回归。
"""
import numpy as np
import pytest

from fem2d.material import von_mises
from fem2d.mesh import Mesh
from fem2d.spr import spr_recovery
from fem2d.stress import nodal_average, principal_stresses

NODES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])


def _mesh():
    return Mesh(NODES, ELEMS)


# ── mesh: str/object 单元索引 ──

def test_constructor_str_elements_typeerror():
    # 回滚 → 裸 TypeError "ufunc 'isfinite' not supported"
    with pytest.raises(TypeError, match="elements"):
        Mesh(NODES, [["a", "b", "c"]])


def test_constructor_object_elements_typeerror():
    with pytest.raises(TypeError, match="elements"):
        Mesh(NODES, np.array([[None, "x", 0.5]], dtype=object))


def test_replace_elements_str_typeerror():
    m = _mesh()
    with pytest.raises(TypeError, match="replace_elements"):
        m.replace_elements(np.array([["a", "b", "c"]]))


# ── 应力侧: str/object 必须 TypeError, complex 必须 ValueError ──

def test_von_mises_str_vector_typeerror():
    # 回滚 → 裸 TypeError "ufunc 'isfinite' not supported"
    with pytest.raises(TypeError, match="von_mises"):
        von_mises(np.array(["a", "b", "c"]))


def test_principal_str_batch_typeerror():
    with pytest.raises(TypeError, match="principal_stresses"):
        principal_stresses(np.array([["a", "b", "c"], ["d", "e", "f"]]))


def test_nodal_average_str_elem_stress_typeerror():
    with pytest.raises(TypeError, match="nodal_average"):
        nodal_average(_mesh(), np.array([["a", "b", "c"], ["d", "e", "f"]]))


def test_nodal_average_complex_elem_stress_valueerror():
    # 回滚 → 裸 TypeError (np.bincount 不支持 complex 权重)
    with pytest.raises(ValueError, match="nodal_average"):
        nodal_average(_mesh(), np.array([[1 + 1j, 1.0, 0.0], [1.0, 2.0, 0.5]]))


def test_nodal_average_complex_weights_valueerror():
    with pytest.raises(ValueError, match="weights"):
        nodal_average(_mesh(), np.ones((2, 3)), weights=np.array([1 + 1j, 2.0]))


def test_nodal_average_str_weights_typeerror():
    with pytest.raises(TypeError, match="weights"):
        nodal_average(_mesh(), np.ones((2, 3)), weights=np.array(["a", "b"]))


def test_spr_str_typeerror():
    # 回滚 → numpy 裸 ValueError "could not convert string to float"
    with pytest.raises(TypeError, match="spr_recovery"):
        spr_recovery(_mesh(), np.array([["a", "b", "c"], ["d", "e", "f"]]))


def test_spr_complex_valueerror():
    # 回滚 → 裸 TypeError "can't convert complex to float"
    with pytest.raises(ValueError, match="spr_recovery"):
        spr_recovery(_mesh(), np.array([[1 + 1j, 1.0, 0.0], [1.0, 2.0, 0.5]]))


# ── 绿侧: 合法输入路径不受影响 ──

def test_float_nan_still_valueerror():
    with pytest.raises(ValueError, match="NaN/Inf"):
        von_mises(np.array([np.nan, 1.0, 0.0]))


def test_nodal_average_valid_weights_unchanged():
    m = _mesh()
    out = nodal_average(m, np.ones((2, 3)), weights="area")
    assert out.shape == (4, 3)
