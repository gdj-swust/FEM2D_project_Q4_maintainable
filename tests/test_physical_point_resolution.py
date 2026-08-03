"""Physical Point 解析失败原因区分 — physical_point_from_geo 的失败原因
必须可区分 (not_found / ambiguous / outside_domain / no_geo_source),
不能统一报"未找到" (曾静默歧义/域外, 集中力施加到错误位置).

判别性: 旧实现 (统一 None) 下这些测试必须失败.
"""
import tempfile
from pathlib import Path

import pytest

try:
    import gmsh as _gmsh
except (ImportError, OSError):
    _gmsh = None

pytestmark = pytest.mark.skipif(
    _gmsh is None, reason="Gmsh Python API not available")

import numpy as np


def _square_mesh():
    from fem2d.mesh import Mesh
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2, 3]], dtype=int)
    return Mesh(nodes=nodes, elements=elems, elem_type="CPS4")


def _geo_with_point(tmp_path, point_coords, name="load", extra_points=()):
    """写含 Physical Point 的 .geo (无需网格)."""
    p = Path(tmp_path) / "pt.geo"
    lines = ['SetFactory("OpenCASCADE");']
    for i, (x, y) in enumerate([point_coords, *extra_points], start=1):
        lines.append(f"Point({i}) = {{{x}, {y}, 0, 0.5}};")
    ids = [str(i) for i in range(1, 1 + len(extra_points) + 1)]
    lines.append(f'Physical Point("{name}", 1) = {{{",".join(ids)}}};')
    p.write_text("\n".join(lines), encoding="utf-8")
    return str(p)


def test_point_found_returns_node():
    from fem2d.input_source import physical_point_from_geo
    with tempfile.TemporaryDirectory() as d:
        geo = _geo_with_point(d, (0.5, 0.5))
        nid, label, dist, reason = physical_point_from_geo(
            geo, "load", _square_mesh())
        assert reason is None, f"合法点被拒: {reason}"
        assert nid in (0, 1, 2, 3), f"应命中某个最近节点, got {nid}"
        assert dist == pytest.approx(np.sqrt(0.5), rel=1e-12), \
            f"中心点到角点距离应约 0.707, got {dist}"
        assert "Physical Point 'load'" in label


def test_not_found_reason():
    from fem2d.input_source import physical_point_from_geo
    with tempfile.TemporaryDirectory() as d:
        geo = _geo_with_point(d, (0.5, 0.5))
        nid, label, dist, reason = physical_point_from_geo(
            geo, "ghost", _square_mesh())
        assert nid is None and reason == "not_found", f"got {reason}"


def test_ambiguous_reason():
    from fem2d.input_source import physical_point_from_geo
    with tempfile.TemporaryDirectory() as d:
        geo = _geo_with_point(
            d, (0.5, 0.5), "dup", extra_points=[(0.2, 0.8)])
        nid, label, dist, reason = physical_point_from_geo(
            geo, "dup", _square_mesh())
        assert nid is None and reason == "ambiguous", f"got {reason}"


def test_outside_domain_reason():
    from fem2d.input_source import physical_point_from_geo
    with tempfile.TemporaryDirectory() as d:
        geo = _geo_with_point(d, (10.0, 10.0))
        nid, label, dist, reason = physical_point_from_geo(
            geo, "load", _square_mesh())
        assert nid is None and reason == "outside_domain", f"got {reason}"


def test_no_geo_source_reason():
    from fem2d.input_source import physical_point_from_geo
    nid, label, dist, reason = physical_point_from_geo(
        None, "load", _square_mesh())
    assert nid is None and reason == "no_geo_source", f"got {reason}"
