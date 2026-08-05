"""覆盖轮 C1 — bc_apply 交互/错误分支缺口 (第 2 轮, 直接调阶段函数).

交互输入用 monkeypatch bc_apply.ask; Physical Point 回退用
monkeypatch physical_point_from_geo 或伪 registry.
"""
import contextlib
import io

import numpy as np
import pytest

from fem2d import Mesh
from fem2d.bc_apply import (
    _apply_concentrated_forces,
    _apply_fix_bcs,
    _apply_traction_profile,
    _apply_tractions,
    _interactive_edge_index,
    _print_segment_menu,
    _resolve_boundary_selection,
)
from fem2d.config import AnalysisConfig
from fem2d.errors import CliError, GeoScriptRejected


def _mesh_segs():
    nodes = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    mesh = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 2, 3]]),
                E=2.1e11, nu=0.3, thickness=0.01, elem_type="CPS3")

    def seg(label, tp, ns, **info):
        return {"label": label, "type": tp, "nodes": list(ns),
                "info": info, "coords": nodes[ns]}

    segs = [
        seg("左端", "line", [0, 1]),
        seg("右端", "line", [3, 2]),
        seg("顶弧", "arc", [1, 2], radius=0.5),
        seg("曲线段", "curve", [0, 3], curvature_mean=0.1),
    ]
    return mesh, segs


def _run(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ── _resolve_boundary_selection 错误分支 ────────────────────────────────────

def test_resolve_selection_fatal_raises_cli():
    """@组名 混用编号 → ValueError 被 fatal 模式转 CliError."""
    mesh, segs = _mesh_segs()
    with pytest.raises(CliError):
        _resolve_boundary_selection("@1,2", segs, fatal=True)


def test_resolve_selection_nonfatal_warns():
    """非 fatal 模式 (交互) → WARN + 空列表, 不终止."""
    mesh, segs = _mesh_segs()
    with contextlib.redirect_stdout(io.StringIO()) as out:
        assert _resolve_boundary_selection("@1,2", segs, fatal=False) == []
    assert "WARN" in out.getvalue()


# ── 交互式边菜单/输入 (monkeypatch ask) ─────────────────────────────────────

def test_print_segment_menu_arc_ellipse_curve(monkeypatch, capsys):
    mesh, segs = _mesh_segs()
    segs[3]["type"] = "ellipse"
    segs[3]["info"] = {"semi_major": 1.0, "semi_minor": 0.5}
    segs.append({"label": "未知", "type": "spline", "nodes": [0, 1],
                 "info": {}, "coords": mesh.nodes[:2]})
    _print_segment_menu(segs)
    out = capsys.readouterr().out
    assert "圆弧" in out and "椭圆" in out and "曲线" in out


def test_interactive_edge_index_invalid_choices(monkeypatch):
    """先无效编号 (重问), 再空输入退出."""
    answers = iter(["999", ""])
    monkeypatch.setattr("fem2d.bc_apply.ask", lambda msg: next(answers))
    mesh, segs = _mesh_segs()
    collected = list(_interactive_edge_index(segs))
    assert collected == []


def test_interactive_edge_index_partial_invalid(monkeypatch, capsys):
    """'1,999' 部分无效 → 整组丢弃并提示, 不施加任何边."""
    answers = iter(["1,999", ""])
    monkeypatch.setattr("fem2d.bc_apply.ask", lambda msg: next(answers))
    mesh, segs = _mesh_segs()
    from fem2d.bc_apply import _interactive_edge_index
    collected = list(_interactive_edge_index(segs))
    assert collected == []
    assert "无效编号" in capsys.readouterr().out


def test_apply_fix_interactive_ux_only(monkeypatch):
    """交互固定: 只给 Ux → 仅约束 x 方向 (uy 留空)."""
    answers = iter(["1", "0.5", "", ""])
    monkeypatch.setattr("fem2d.bc_apply.ask", lambda msg: next(answers))
    mesh, segs = _mesh_segs()
    config = AnalysisConfig()
    _apply_fix_bcs(config, mesh, segs, batch_mode=False, region_registry=None)
    assert len(mesh.fixed_dofs) == 2
    assert all(v == 0.5 for v in mesh.prescribed_vals.values())


def test_apply_tractions_interactive_empty_tx_breaks(monkeypatch):
    """交互面力: tx 空输入 → 终止收集."""
    answers = iter(["1", ""])
    monkeypatch.setattr("fem2d.bc_apply.ask", lambda msg: next(answers))
    mesh, segs = _mesh_segs()
    config = AnalysisConfig()
    _apply_tractions(config, mesh, segs, batch_mode=False, region_registry=None)
    assert mesh.surface_tractions == []


def test_apply_tractions_interactive_ty_breaks(monkeypatch):
    """交互面力: tx 有值 ty 空 → 不收集."""
    answers = iter(["1", "1e6", ""])
    monkeypatch.setattr("fem2d.bc_apply.ask", lambda msg: next(answers))
    mesh, segs = _mesh_segs()
    config = AnalysisConfig()
    _apply_tractions(config, mesh, segs, batch_mode=False, region_registry=None)
    assert mesh.surface_tractions == []


# ── 面力规格错误 ────────────────────────────────────────────────────────────

def test_traction_missing_edge_prefix_batch_fatal():
    """批处理 + 缺边前缀 ('1e6,0') → FATAL CliError."""
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(traction="1e6,0")
    with pytest.raises(CliError, match="缺少边前缀"):
        _apply_tractions(config, mesh, segs, batch_mode=True,
                         region_registry=None)


def test_traction_parse_error_fatal():
    """面力规格解析失败 (值非数值) → FATAL (用户错误, 退出码 1)."""
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(traction="左端:bad,0")
    with pytest.raises(CliError):
        _apply_tractions(config, mesh, segs, batch_mode=True,
                         region_registry=None)


# ── 剖面面力链方向/闭合校验 ─────────────────────────────────────────────────

def test_traction_profile_chain_reversed():
    """闭合链方向与段记录相反 → 先反转再按弧长施加."""
    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    mesh = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 3, 2]]),
                E=2.1e11, nu=0.3, thickness=0.01)
    seg = {"label": "环", "type": "line", "nodes": [0, 3, 2, 1, 0],
           "info": {}, "coords": nodes}
    _apply_traction_profile(mesh, [seg], [0], "环", 1e6, 0.0, "p")
    assert len(mesh.surface_tractions) == 4  # 闭合链四条边逐一施加


def test_traction_linear_profile_closed_loop_fatal():
    """:l 线性面力施加到闭合边界 → FATAL (闭环起点值不唯一)."""
    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    mesh = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 3, 2]]),
                E=2.1e11, nu=0.3, thickness=0.01)
    seg = {"label": "环", "type": "line", "nodes": [0, 1, 2, 3, 0],
           "info": {}, "coords": nodes}
    with pytest.raises(CliError, match="闭合"):
        _apply_traction_profile(mesh, [seg], [0], "环", 1e6, 0.0, "l")


# ── 集中力错误分支 ──────────────────────────────────────────────────────────

def test_force_unparseable_components():
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(force="0,abc,0")
    with pytest.raises(CliError, match="无法解析"):
        _apply_concentrated_forces(config, mesh, None, {}, None)


class _FakeRegistry:
    """dimension=0 名称 → 多节点 PointRegion 的伪 registry."""

    def __init__(self, node_ids):
        self._node_ids = node_ids

    def by_name(self, name, dimension=None):
        return [type("R", (), {"node_ids": list(self._node_ids)})()]


def test_force_physical_point_multiple_nodes_fatal():
    """Physical Point 映射到多个节点 → 集中力目标必须唯一."""
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(force="p1,1,2")
    with pytest.raises(CliError, match="唯一"):
        _apply_concentrated_forces(config, mesh, _FakeRegistry([0, 1]),
                                   {}, "unused.geo")


def test_force_geo_rejected(monkeypatch):
    """源 .geo 含危险指令 → GeoScriptRejected 转 FATAL."""
    def rejected(*args, **kwargs):
        raise GeoScriptRejected("SystemCall blocked")
    monkeypatch.setattr("fem2d.bc_apply.physical_point_from_geo", rejected)
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(force="p1,1,2")
    with pytest.raises(CliError, match="SystemCall"):
        _apply_concentrated_forces(config, mesh, None, {}, "bad.geo")


def test_force_point_nearest_node_warn(monkeypatch, capsys):
    """Physical Point 未落在节点上 → WARN 最近节点."""
    monkeypatch.setattr(
        "fem2d.bc_apply.physical_point_from_geo",
        lambda *a, **k: (2, "Physical Point 'p1'", 0.4, None))
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(force="p1,1,2")
    _apply_concentrated_forces(config, mesh, None, {}, "g.geo")
    assert mesh.concentrated_forces[0]["node"] == 2
    assert "WARN" in capsys.readouterr().out


def test_force_point_not_found_reason_hint(monkeypatch):
    """点既不是节点号也不可解析 → reason_hint 区分歧义原因."""
    monkeypatch.setattr(
        "fem2d.bc_apply.physical_point_from_geo",
        lambda *a, **k: (None, None, None, "ambiguous"))
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(force="p1,1,2")
    with pytest.raises(CliError, match="多个 Physical Point"):
        _apply_concentrated_forces(config, mesh, None, {}, "g.geo")
