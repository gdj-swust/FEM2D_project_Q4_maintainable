"""pkg_cleanA 清理轮判别性测试 (T1 死参数 / T2 死参数 / T3 弃用警告 / T4 注释计数门).

判别性: 把对应修复回滚到基线实现时, 对应测试必须红.
  - T1/T2: inspect.signature 断言参数集 (回滚会多出 prefix/scale 参数)
  - T3: warnings.simplefilter("error") 断言直接构造抛 DeprecationWarning,
        工厂路径不抛 (回滚后无警告 → 直接构造测红)
  - T4: 叙事注释计数配方断言 ≤ 20 (回滚后回到基线 190 → 红)
"""
from __future__ import annotations

import inspect
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from fem2d.boundary.detectors._shared import _fit_closed_conic
from fem2d.boundary.model import BoundaryDiagnostics
from fem2d.boundary.physical_mapping import (
    PhysicalEdgeMapper,
    map_physical_edges,
)
from fem2d.boundary.plugins.arc_curvature import ArcCurvatureDetector
from fem2d.mesh import Mesh

_NARRATIVE_COMMENT_RE = re.compile(
    r"^\s*#.*(曾|旧实现|旧版本|当时|最初|后来|2026-0|遗留)")
_NARRATIVE_LIMIT = 20


def _tiny_mesh() -> Mesh:
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    elems = np.array([[0, 1, 2], [0, 2, 3]])
    return Mesh(nodes, elems)


# ── T1: _fit_closed_conic 无死参数 prefix ──────────────────────────

def test_t1_fit_closed_conic_drops_prefix_param():
    params = list(inspect.signature(_fit_closed_conic).parameters)
    assert params == ["coords"]


# ── T2: _arc_algebraic 无死参数 scale ──────────────────────────────

def test_t2_arc_algebraic_drops_scale_param():
    params = list(
        inspect.signature(ArcCurvatureDetector._arc_algebraic).parameters)
    assert params == ["self", "coords", "is_outer"]


# ── T3: PhysicalEdgeMapper 弃用 — 直接构造警告, 工厂静默 ──────────

def test_t3_direct_construction_warns_deprecation():
    mesh = _tiny_mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning):
            PhysicalEdgeMapper(mesh, {}, {}, None, BoundaryDiagnostics())


def test_t3_factory_path_is_silent():
    mesh = _tiny_mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        mapped = map_physical_edges(
            mesh, {}, {}, None, BoundaryDiagnostics())
    assert mapped is not None
    assert mapped.boundary_edges


# ── T4: 历史叙事注释计数门 (fem2d/ 全部 .py ≤ 20) ──────────────────

def test_t4_narrative_comment_count_within_limit():
    repo_root = Path(__file__).resolve().parents[1]
    violations = []
    hits = 0
    for path in sorted((repo_root / "fem2d").rglob("*.py")):
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if _NARRATIVE_COMMENT_RE.match(line):
                hits += 1
                violations.append(
                    f"{path.relative_to(repo_root)}:{lineno}: {line}")
    assert hits <= _NARRATIVE_LIMIT, (
        f"叙事注释 {hits} > 上限 {_NARRATIVE_LIMIT}:\n"
        + "\n".join(violations))
