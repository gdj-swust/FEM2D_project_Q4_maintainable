"""R-δ 轮判别性测试 — D9 死参数清理.

判别性 (回滚必须红): 死参数恢复进签名 → inspect 断言红。
runner.py 冗余 global 由 lint 门 (pyflakes 关键文件 0 命中) 锁定,
本文件对 _ensure_patch_test 源码做直接断言兜底。
"""
import inspect

from fem2d.bc import _pure_dirichlet_solution
from fem2d.convergence import _sample_level
from fem2d.runner import _ensure_patch_test


def test_d9_pure_dirichlet_no_free_dofs():
    """_pure_dirichlet_solution 不再接收未使用的 free_dofs."""
    params = inspect.signature(_pure_dirichlet_solution).parameters
    assert "free_dofs" not in params, f"死参数 free_dofs 未删: {params}"


def test_d9_sample_level_no_unused_params():
    """_sample_level 不再接收未使用的 I / P_mag."""
    params = inspect.signature(_sample_level).parameters
    assert "I" not in params, f"死参数 I 未删: {params}"
    assert "P_mag" not in params, f"死参数 P_mag 未删: {params}"


def test_d9_runner_no_redundant_global():
    """_ensure_patch_test 不声明冗余 global (函数内从不重绑定该名字)."""
    src = inspect.getsource(_ensure_patch_test)
    assert "global _patch_checked" not in src, "冗余 global 未删"
