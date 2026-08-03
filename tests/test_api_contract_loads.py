"""契约清账阶段 2 — 载荷形状校验收敛 (bc_apply 改调共享 schema) 判别性测试.

旧实现: bc_apply 元组解包绕开 _check_load_pair — 1/3 分量报无上下文
ValueError ('too many values to unpack'), ndarray 在真值判据处抛裸
ValueError (ambiguous truth value)。
"""
import numpy as np
import pytest

from fem2d.bc_apply import _apply_body_force
from fem2d.config import AnalysisConfig
from fem2d.mesh import Mesh


def _mesh():
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    elements = np.array([[0, 1, 3], [0, 3, 2]])
    return Mesh(nodes, elements)


def _config(body):
    cfg = AnalysisConfig()
    cfg.body = body
    return cfg


def test_body_tuple_wrong_length_has_context():
    m = _mesh()
    cfg = _config((1e4, 2e4, 3e4))
    with pytest.raises(ValueError, match="body"):
        _apply_body_force(cfg, m, batch_mode=True)


def test_body_tuple_nan_component_rejected():
    m = _mesh()
    cfg = _config((float("nan"), 1e4))
    with pytest.raises(ValueError, match="body"):
        _apply_body_force(cfg, m, batch_mode=True)


def test_body_ndarray_accepted_without_truth_value_error():
    # 旧实现: ndarray 在 'if body_input' 抛 ambiguous truth value
    m = _mesh()
    cfg = _config(np.array([1e4, 2e4]))
    bfx, bfy = _apply_body_force(cfg, m, batch_mode=True)
    assert bfx == 1e4 and bfy == 2e4
    assert m.body_force == (1e4, 2e4)


def test_body_ndarray_wrong_length_rejected():
    m = _mesh()
    cfg = _config(np.array([1e4, 2e4, 3e4]))
    with pytest.raises(ValueError, match="body"):
        _apply_body_force(cfg, m, batch_mode=True)


def test_body_zero_and_empty_string_skip():
    m = _mesh()
    bfx, bfy = _apply_body_force(_config("0"), m, batch_mode=True)
    assert (bfx, bfy) == (0.0, 0.0)
    bfx, bfy = _apply_body_force(_config(""), m, batch_mode=True)
    assert (bfx, bfy) == (0.0, 0.0)
    assert m.body_force is None


# ── mesh.validate_state 载荷 schema (构造直传) 仍由共享 helper 拦截 ──

def test_validate_state_body_force_wrong_shape():
    m = _mesh()
    m.body_force = (1e4, 2e4, 3e4)
    with pytest.raises(ValueError, match="body_force"):
        m.validate_state()


def test_validate_state_concentrated_force_wrong_shape():
    m = _mesh()
    m.concentrated_forces.append({"node": 0, "force": (1.0, 2.0, 3.0)})
    with pytest.raises(ValueError, match="concentrated_forces"):
        m.validate_state()


def test_validate_state_pressure_wrong_shape():
    m = _mesh()
    m.surface_tractions.append({
        "nodes": (0, 1), "traction": (1e6, 2e6), "is_pressure": True})
    with pytest.raises(ValueError, match="pressure"):
        m.validate_state()
