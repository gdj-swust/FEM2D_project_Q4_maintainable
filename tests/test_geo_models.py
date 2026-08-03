"""Static and meshing guards for bundled Gmsh model streams."""

import re
from pathlib import Path

import pytest

from fem2d.gmsh_adapter import GmshUnavailableError
from fem2d.preprocess import read_geo_groups

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_all_geo_models_mesh_through_production_pipeline(tmp_path):
    """Every bundled ``.geo`` must mesh via the exact ``run.py`` pipeline.

    Invalid model geometry (overlapping holes, holes outside the domain,
    wrong curve sweeps) used to surface as confusing Gmsh errors or be
    silently dropped.  ``run_gmsh`` is the same stripped-copy + topology
    validation path ``run.py`` uses; output goes to ``tmp_path`` so the
    models directory stays clean.
    """
    from scripts.gmsh_runner import find_gmsh, run_gmsh

    if find_gmsh() is None:
        pytest.skip("bundled gmsh executable not available")
    files = sorted(MODELS.glob("*.geo"))
    assert files
    for path in files:
        published = run_gmsh(
            str(path),
            output_path=str(tmp_path / f"{path.stem}.msh"),
        )
        assert published is not None, path.name


def test_all_high_pressure_geo_models_export_complete_meshes():
    files = sorted(MODELS.glob("hp_*.geo"))
    assert files
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert re.search(r"\bMesh\.SaveAll\s*=\s*1\s*;", source), path.name


def test_curve_complex_has_semantic_regions_and_no_stale_invalid_inp():
    path = MODELS / "hp_curve_complex.geo"
    groups = read_geo_groups(path)

    assert set(groups) == {
        "outer_racetrack",
        "round_hole_1",
        "round_hole_2",
        "triangle_hole",
        "square_hole",
    }
    assert 'Physical Surface("domain", 201)' in path.read_text(
        encoding="utf-8")
    assert not (MODELS / "hp_curve_complex.msh").exists()


def test_curve_complex_curve_tags_do_not_overlap():
    source = (MODELS / "hp_curve_complex.geo").read_text(encoding="utf-8")
    explicit = [
        (kind, int(tag))
        for kind, tag in re.findall(
            r"\b(Line|Circle|Ellipse)\s*\(\s*(\d+)\s*\)", source)
    ]
    tags = [tag for _, tag in explicit]
    assert len(tags) == len(set(tags))


# ═══════════════════════════════════════════════════════════════
# 第六轮: Kirsch 应力集中经典验证 (教学基准)
# x 拉伸带孔板: σ_θ = σ(1 − 2cos2θ) → 集中 +3σ 在 θ=±90° (上下缘),
# 右缘 σ_θ = −σ 压缩. 曾因验证脚本公式方向记错被误判为 bug
# (2026-08-03 自查确认代码正确).
# ═══════════════════════════════════════════════════════════════

def test_kirsch_stress_concentration(tmp_path):
    """带孔板 x 拉伸: 最大主应力 ≈ 3σ 在孔上下缘, 右缘环向 ≈ −σ."""
    import os

    import numpy as np
    from fem2d import Mesh, solve
    from fem2d.input_source import generate_geo_with_topology
    from fem2d.stress import principal_stresses

    geo_text = """lc = 0.3;
Point(1) = {-6, -6, 0, lc}; Point(2) = {6, -6, 0, lc};
Point(3) = {6, 6, 0, lc}; Point(4) = {-6, 6, 0, lc};
Line(1) = {1,2}; Line(2) = {2,3}; Line(3) = {3,4}; Line(4) = {4,1};
Curve Loop(1) = {1,2,3,4};
n = 24;
Point(100) = {0, 0, 0, lc};
For i In {0:n-1}
  ang = 2*Pi*i/n;
  Point(200+i) = {Cos(ang), Sin(ang), 0, lc};
EndFor
For i In {0:n-1}
  Circle(300+i) = {200+i, 100, 200+((i+1)%n)};
EndFor
Curve Loop(2) = {-300:-323};
Plane Surface(1) = {1, 2};
Mesh 2;"""
    geo = str(tmp_path / "kirsch.geo")
    with open(geo, "w", encoding="utf-8") as f:
        f.write(geo_text)
    try:
        msh, g = generate_geo_with_topology(
            geo, output_path=str(tmp_path / "kirsch.msh"),
            plane_type="stress")
    except (ImportError, OSError, GmshUnavailableError) as error:
        # 仅跳过 Gmsh 依赖不可用 (缺 libGLU/API) — 真实回归 (网格生成
        # 逻辑错误) 必须报错而非被 skip 掩盖; _load_gmsh_module 把
        # (ImportError, OSError) 包装成 GmshUnavailableError, 必须同捕
        pytest.skip(f"Gmsh 依赖不可用: {error}")
    if msh is None or g is None:
        from scripts.gmsh_runner import find_gmsh
        if find_gmsh() is None:
            pytest.skip("Gmsh 网格生成返回 None — 环境缺 gmsh 可执行文件")
        raise RuntimeError(
            "Gmsh 可用但网格生成返回 None — 真实生成回归: "
            f"msh={msh!r}, geometry={g!r}")
    try:
        m = Mesh(nodes=g.nodes, elements=g.elements, E=2.1e11, nu=0.3,
                 thickness=0.01, plane_type="stress", elem_type=g.elem_type)
        xmin, xmax = g.nodes[:, 0].min(), g.nodes[:, 0].max()
        for i, n in enumerate(g.nodes):
            if abs(n[0] - xmin) < 1e-6:
                m.fix_node(i, "both", 0.0)
        from fem2d.boundary import (
            BoundaryDiagnostics, build_boundary_segments)
        segs = build_boundary_segments(
            m, edge_labels=None, diagnostics=BoundaryDiagnostics())
        for s in segs:
            ns = list(map(int, s["nodes"]))
            if all(abs(g.nodes[n][0] - xmax) < 1e-6 for n in ns):
                for a, b in zip(ns, ns[1:]):
                    m.add_traction(a, b, 1e6, 0.0)
        r = solve(m, verbose=False)
        s = r["stress"]
        s1, _, _, _ = principal_stresses(s)
        cent = m.centroids
        idx = int(np.argmax(s1))
        # 集中 3σ 在孔上下缘 (θ=±90°)
        assert 2.5 < s1[idx] / 1e6 < 3.5, \
            f"K_t = {s1[idx]/1e6:.3f} 偏离 Kirsch 3.0"
        assert abs(cent[idx, 1]) > 0.8, \
            f"集中位置应在孔上下缘, 得到 ({cent[idx,0]:.2f}, {cent[idx,1]:.2f})"
        # 右缘环向 ≈ −σ (压缩)
        near = np.linalg.norm(cent, axis=1)
        ring = np.where((near > 0.9) & (near < 1.3))[0]
        ang = np.degrees(np.arctan2(cent[ring, 1], cent[ring, 0]))
        right = ring[(ang >= -22) & (ang <= 22)]
        if len(right):
            assert s[right, 1].min() / 1e6 < -0.5, \
                f"右缘 σy 应压缩 ≈ −σ, 得到 {s[right,1].min()/1e6:.3f}"
    finally:
        if msh and os.path.isfile(msh):
            os.unlink(msh)
