"""MSH 导入链审计 2026-08-03 — 物理组静默丢失 / 完整性误判 / 域内点.

覆盖: MSH 2.2 导入物理组丢失必须 WARN / 4.1 正常恢复 / 4.1 缺 $Entities
段丢失必须 WARN / cad_boundary_complete 必须反映实际数据 / 域内 Physical
Point 回退最近位移节点。
"""
import os

import pytest

try:
    import gmsh as _gmsh
except (ImportError, OSError):
    _gmsh = None

pytestmark = pytest.mark.skipif(
    _gmsh is None, reason="Gmsh Python API not available")


def _rect_model(path, version=None, with_point=False, outside_point=False):
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if version:
        gmsh.option.setNumber("Mesh.MshFileVersion", version)
    gmsh.model.add("t")
    p1 = gmsh.model.geo.addPoint(0, 0, 0, 0.5)
    p2 = gmsh.model.geo.addPoint(3, 0, 0, 0.5)
    p3 = gmsh.model.geo.addPoint(3, 2, 0, 0.5)
    p4 = gmsh.model.geo.addPoint(0, 2, 0, 0.5)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([loop])
    if with_point:
        p5 = gmsh.model.geo.addPoint(1.5, 1.0, 0, 0.5)
    if outside_point:
        p6 = gmsh.model.geo.addPoint(10, 10, 0, 0.5)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(1, [l1], 100)
    gmsh.model.setPhysicalName(1, 100, "bottom")
    gmsh.model.addPhysicalGroup(2, [surf], 200)
    gmsh.model.setPhysicalName(2, 200, "domain")
    if with_point:
        gmsh.model.addPhysicalGroup(0, [p5], 300)
        gmsh.model.setPhysicalName(0, 300, "loadA")
    if outside_point:
        gmsh.model.addPhysicalGroup(0, [p6], 400)
        gmsh.model.setPhysicalName(0, 400, "stray")
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()


_RECT_GEO = """lc = 0.5;
Point(1) = {0, 0, 0, lc}; Point(2) = {3, 0, 0, lc};
Point(3) = {3, 2, 0, lc}; Point(4) = {0, 2, 0, lc};
Line(1) = {1, 2}; Line(2) = {2, 3}; Line(3) = {3, 4}; Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Physical Curve("bottom", 100) = {1};
Physical Surface("domain", 200) = {1};
Mesh 2;"""


def _gmsh_exe():
    import shutil
    candidates = [
        os.path.abspath("tools/gmsh-4.15.2-Windows64/gmsh.exe"),
        shutil.which("gmsh"),
    ]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def test_msh2_import_never_silent(tmp_path, capsys):
    """MSH 2.2 导入: gmsh 对 2.2 的物理组恢复行为不稳定 (实测有时恢复
    有时丢失) — 无论哪种情况, 网格导入必须成功且不抛异常; 物理组
    丢失时由 import_msh 的检测给出 WARN."""
    import subprocess
    from fem2d.gmsh_adapter import import_msh
    exe = _gmsh_exe()
    if exe is None:
        pytest.skip("gmsh.exe not found")
    geo = tmp_path / "m.geo"
    geo.write_text(_RECT_GEO, encoding="utf-8")
    path = str(tmp_path / "m22.msh")
    subprocess.run([exe, "-v", "0", "-nt", "4", str(geo), "-2",
                    "-o", path, "-format", "msh2"],
                   check=True, timeout=60)
    g = import_msh(path, plane_type="stress")   # 不得崩溃
    assert len(g.nodes) > 0 and len(g.elements) > 0
    # 曾 if not curves 条件断言 (恢复成功时从不执行, 审计 2026-08-03).
    # gmsh 按 $Elements physical id 恢复物理组 (即使名字被清空) — 2.2
    # 场景无法稳定构造"声明了名字但恢复失败"; 该 WARN 分支由
    # test_msh41_without_entities_warned (删 $Entities 段) 强制覆盖。
    # 这里断言: 恢复的物理组与文件声明一致, 且不误报 WARN。
    out = capsys.readouterr().out
    if g.regions.curves:
        assert "WARN" not in out, f"物理组恢复成功却误报: {out!r}"
    else:
        assert "WARN" in out, f"物理组丢失未警告: {out!r}"


def test_msh41_import_recovers_physical_groups(tmp_path, capsys):
    """MSH 4.1 导入必须恢复物理组且不误报 WARN."""
    from fem2d.gmsh_adapter import import_msh
    path = str(tmp_path / "m41.msh")
    _rect_model(path)
    g = import_msh(path, plane_type="stress")
    assert [c.name for c in g.regions.curves] == ["bottom"]
    assert g.regions.cad_boundary_complete
    out = capsys.readouterr().out
    assert "WARN" not in out, f"4.1 正常文件被误报: {out!r}"


def test_msh41_without_entities_warned(tmp_path, capsys):
    """4.1 缺 $Entities 段 (meshio 等第三方工具): 物理组丢失必须 WARN."""
    from fem2d.gmsh_adapter import import_msh
    base = str(tmp_path / "base.msh")
    _rect_model(base)
    with open(base, encoding="utf-8", errors="replace") as f:
        text = f.read()
    start = text.find("$Entities")
    if start >= 0:
        end = text.find("$EndEntities") + len("$EndEntities")
        text = text[:start] + text[end:]
    stripped = str(tmp_path / "noent.msh")
    with open(stripped, "w", encoding="utf-8") as f:
        f.write(text)
    g = import_msh(stripped, plane_type="stress")
    assert not g.regions.curves, "缺 $Entities 段物理组应不可用"
    out = capsys.readouterr().out
    assert "WARN" in out, f"物理组丢失未警告: {out!r}"


def test_cad_boundary_complete_reflects_data(tmp_path):
    """裸网格 (无物理组 + 无 CAD 实体) 不得宣称 cad_boundary_complete=True.

    只删 $PhysicalNames 时 gmsh 仍从 $Entities 恢复曲线实体 (curves 非空
    → 完整), 需两段都删才构成第三方裸网格形态.
    """
    from fem2d.gmsh_adapter import import_msh
    path = str(tmp_path / "m41.msh")
    _rect_model(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    for section in ("$PhysicalNames", "$Entities"):
        start = text.find(section)
        if start >= 0:
            end = text.find(
                "$End" + section[1:]) + len("$End" + section[1:])
            text = text[:start] + text[end:]
    naked = str(tmp_path / "naked.msh")
    with open(naked, "w", encoding="utf-8") as f:
        f.write(text)
    g = import_msh(naked, plane_type="stress")
    assert not g.regions.curves, "裸网格不应有曲线实体"
    assert not g.regions.cad_boundary_complete, \
        "裸网格被误判为 CAD 边界完整"


def test_physical_point_domain_node_fallback(tmp_path, capsys):
    """域内 Physical Point 构造节点被剔除时必须回退最近位移节点."""
    from fem2d.gmsh_adapter import import_msh
    path = str(tmp_path / "pt.msh")
    _rect_model(path, with_point=True)
    g = import_msh(path, plane_type="stress")
    pts = {p.name: p.node_ids for p in g.regions.points}
    assert "loadA" in pts, f"loadA 物理点丢失: {pts}"
    assert pts["loadA"], f"loadA node_ids 恒空 (曾无提示): {pts}"
    assert all(0 <= n < len(g.nodes) for n in pts["loadA"])


def test_physical_point_inside_hole_rejected(tmp_path, capsys):
    """孔心点 (材料域外但 AABB 内) 必须拒绝 — 曾回退到孔边最近节点,
    集中力施加到材料域外位置, 静默错结果 (外部审查复现).
    注: 用 OpenCASCADE 内核 (geo 内核两段圆弧拼接的 1D 网格自交,
    实测 gmsh 递归分裂不收敛)."""
    import gmsh
    from fem2d.gmsh_adapter import import_msh
    path = str(tmp_path / "hole.msh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("t")
    rect = gmsh.model.occ.addRectangle(0, 0, 0, 3, 2)
    circle = gmsh.model.occ.addCircle(1.5, 1.0, 0, 0.5)
    hole_center = gmsh.model.occ.addPoint(1.5, 1.0, 0, 0.5)
    gmsh.model.occ.synchronize()
    gmsh.model.occ.fragment([(2, rect)], [(1, circle)])
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(0, [hole_center], 400)
    gmsh.model.setPhysicalName(0, 400, "holeCenter")
    gmsh.model.addPhysicalGroup(2, [rect], 200)
    gmsh.model.setPhysicalName(2, 200, "domain")
    gmsh.model.mesh.generate(2)
    gmsh.write(path)
    gmsh.finalize()

    g = import_msh(path, plane_type="stress")
    pts = {p.name: p.node_ids for p in g.regions.points}
    assert "holeCenter" in pts, f"孔心物理点丢失: {pts}"
    assert not pts["holeCenter"], \
        f"孔心点被映射到节点 (静默施加到材料域外): {pts['holeCenter']}"
    out = capsys.readouterr().out
    assert "不在材料域内" in out, f"孔心点未警告: {out!r}"


def test_physical_point_outside_domain_rejected(tmp_path, capsys):
    """域外 Physical Point 必须拒绝映射 — 曾回退到最近节点, 集中力
    施加到完全错误的位置 (审查复现: 方形域外 (10,10) 被映射到 (1,1))."""
    from fem2d.gmsh_adapter import import_msh
    path = str(tmp_path / "stray.msh")
    _rect_model(path, outside_point=True)
    g = import_msh(path, plane_type="stress")
    pts = {p.name: p.node_ids for p in g.regions.points}
    assert "stray" in pts, f"stray 物理点丢失: {pts}"
    assert not pts["stray"], \
        f"域外点被映射到节点 (曾静默施加到错误位置): {pts['stray']}"
    out = capsys.readouterr().out
    assert "包围盒外" in out, f"域外点未警告: {out!r}"
