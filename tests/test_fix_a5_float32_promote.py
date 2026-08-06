"""A5 (P3, R-α): principal_stresses 批量路径 float32/16 精度提升.

审查现象 (2026-08-05): float32 批量输入返回 float32 (相对误差 5.9e-8),
单向量路径恒 float64 — 同数据两形态精度不一致 (von_mises 因 value-based
casting 已提升, 无此问题)。

修复: 批量入口对 dtype.itemsize < 8 的浮点输入提升 float64 (float32 ⊂
float64 精确可表示, 不引入舍入)。float64 输入不动 — 冻结区逐位不变。

判别性: 回滚本提交 → 批量 float32 输出 dtype 回归 float32。
"""
import numpy as np

from fem2d.stress import principal_stresses

# 修复前 (基线录制) 的 float64 输出 — 冻结区逐位不变金标准
GOLD_BATCH = (
    np.array([4.541381265149109, 10.520797289396148, -0.7928932188134524]),
    np.array([-1.5413812651491097, -1.5207972893961479, -2.2071067811865475]),
    np.array([3.0413812651491097, 6.020797289396148, 0.7071067811865476]),
    np.array([0.8679725021047617, 0.8269687793416689, 0.39269908169872414]),
)
GOLD_SINGLE = (4.541381265149109, -1.5413812651491097,
               3.0413812651491097, 0.8679725021047617)


def test_batch_float32_promoted_to_float64():
    # 回滚 → 输出 dtype 回归 float32
    s = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, -2.0, 0.5]],
                 dtype=np.float32)
    out = principal_stresses(s)
    assert all(o.dtype == np.float64 for o in out)


def test_batch_float16_promoted_to_float64():
    s = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float16)
    out = principal_stresses(s)
    assert all(o.dtype == np.float64 for o in out)


def test_float64_batch_bit_exact_gold():
    # 冻结区: float64 有限输入输出逐位不变
    s = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, -2.0, 0.5]])
    out = principal_stresses(s)
    for got, gold in zip(out, GOLD_BATCH):
        np.testing.assert_array_equal(got, gold)


def test_float64_single_bit_exact_gold():
    assert principal_stresses(np.array([1.0, 2.0, 3.0])) == GOLD_SINGLE


def test_int_batch_still_float64():
    out = principal_stresses(np.array([[1, 2, 3], [4, 5, 6]]))
    assert all(o.dtype == np.float64 for o in out)
