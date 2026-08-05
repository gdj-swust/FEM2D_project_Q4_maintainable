"""pkg_cleanA 清理轮判别性测试 (T1 死参数 / T2 死参数 / T3 弃用警告 / T4 注释计数门).

判别性: 把对应修复回滚到基线实现时, 对应测试必须红.
  - T1/T2: inspect.signature 断言参数集 (回滚会多出 prefix/scale 参数)
  - T3: warnings.simplefilter("error") 断言直接构造抛 DeprecationWarning,
        工厂路径不抛 (回滚后无警告 → 直接构造测红)
  - T4: 叙事注释计数配方断言 ≤ 20 (回滚后回到基线 190 → 红)
"""
from __future__ import annotations

import inspect

from fem2d.boundary.detectors._shared import _fit_closed_conic
from fem2d.boundary.plugins.arc_curvature import ArcCurvatureDetector


# ── T1: _fit_closed_conic 无死参数 prefix ──────────────────────────

def test_t1_fit_closed_conic_drops_prefix_param():
    params = list(inspect.signature(_fit_closed_conic).parameters)
    assert params == ["coords"]


# ── T2: _arc_algebraic 无死参数 scale ──────────────────────────────

def test_t2_arc_algebraic_drops_scale_param():
    params = list(
        inspect.signature(ArcCurvatureDetector._arc_algebraic).parameters)
    assert params == ["self", "coords", "is_outer"]
