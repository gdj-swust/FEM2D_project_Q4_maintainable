"""奇异守卫警告转发测试 (审查修复包第 2 项).

旧实现 catch_warnings(record=True) 只检查 MatrixRankWarning, 其余警告
被静默丢弃 — 调用方收不到任何非秩警告 (审查者用普通 UserWarning 探针
复现 0 条送达)。判别性: 放回旧实现 (只转发秩警告) 必须失败。
"""
import warnings

import pytest
from scipy.sparse.linalg import MatrixRankWarning

from fem2d.solver import _solve_with_singular_guard


def test_non_rank_warning_forwarded_to_caller():
    """注入普通 UserWarning → 外层必须收到 (旧实现静默丢弃)."""
    def probe(*args, **kwargs):
        warnings.warn("probe warning from solver", UserWarning)
        return "ok"

    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        result = _solve_with_singular_guard(probe)
    assert result == "ok"
    categories = [w.category for w in outer]
    assert UserWarning in categories, \
        f"UserWarning 被守卫吞掉 (仅 {categories})"
    assert str(outer[0].message) == "probe warning from solver"


def test_rank_warning_still_raises_runtime_error():
    """MatrixRankWarning 仍转 RuntimeError — 现有奇异行为零变化."""
    def rank(*args, **kwargs):
        warnings.warn("Matrix is singular", MatrixRankWarning)
        return "unused"

    with pytest.raises(RuntimeError, match="singular"):
        _solve_with_singular_guard(rank)


def test_rank_and_other_warnings_both_surface():
    """秩警告转异常且同批其他警告也转发 — 守卫不选择性丢弃."""
    def mixed(*args, **kwargs):
        warnings.warn("plain user warning", UserWarning)
        warnings.warn("Matrix is singular", MatrixRankWarning)
        return None

    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="singular"):
            _solve_with_singular_guard(mixed)
    assert UserWarning in [w.category for w in outer]
