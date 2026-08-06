"""A4 (P3, R-α): apply_elimination/apply_penalty 的 K 形状校验前置.

审查现象 (2026-08-05): _validate_system_inputs 直接访问 K.shape —
K=np.array(2.0) → 裸 IndexError (0 维 shape[0] 越界); K=2.0 → 裸
AttributeError ('float' object has no attribute 'shape')。

修复: 与 estimate_condition 同款 asarray + ndim/shape 前置检查,
ValueError 带上下文。有限输入 (方阵稀疏/稠密) 行为不变 (冻结区 —
apply_penalty 有限输入逐位不变)。

判别性: 回滚本提交 → 裸 IndexError / AttributeError 回归。
"""
import numpy as np
import pytest

from fem2d.bc import apply_elimination, apply_penalty


def test_elimination_scalar_array_k_valueerror():
    # 回滚 → 裸 IndexError "tuple index out of range"
    with pytest.raises(ValueError, match="square matrix"):
        apply_elimination(np.array(2.0), np.zeros(4), [0, 1], [2, 3],
                          [0.0, 0.0])


def test_elimination_float_k_valueerror():
    # 回滚 → 裸 AttributeError "'float' object has no attribute 'shape'"
    with pytest.raises(ValueError, match="square matrix"):
        apply_elimination(2.0, np.zeros(4), [0, 1], [2, 3], [0.0, 0.0])


def test_penalty_float_k_valueerror():
    with pytest.raises(ValueError, match="square matrix"):
        apply_penalty(2.0, np.zeros(4), np.array([0]))


def test_penalty_0d_k_valueerror():
    with pytest.raises(ValueError, match="square matrix"):
        apply_penalty(np.array(2.0), np.zeros(4), np.array([0]))


def test_penalty_non_square_valueerror():
    with pytest.raises(ValueError, match="square matrix"):
        apply_penalty(np.eye(3, 4), np.zeros(4), np.array([0]))


def test_penalty_dense_square_still_works():
    # 稠密方阵 K 通过形状校验 — 有限输入逐位不变 (冻结区)
    K, F, penalty = apply_penalty(np.eye(2), np.zeros(2), np.array([0]))
    assert penalty > 0.0
    assert K.shape == (2, 2)


def test_penalty_sparse_square_still_works():
    from scipy.sparse import eye
    K, F, penalty = apply_penalty(eye(2, format="csr"), np.zeros(2),
                                  np.array([0]))
    assert penalty > 0.0
    assert K.shape == (2, 2)
