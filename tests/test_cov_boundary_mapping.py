"""覆盖轮 C1 — boundary/registry_mapping + plugins/arc_curvature + conic_merge 缺口行.

registry_mapping: 直接用构造的 mesh + 假 diagnostics 调私有校验方法;
arc_curvature: 退化/共线/圆链保护分支;
conic_merge: 合并门控 (键拒绝/边歧义/切线不连续/方向恢复).
"""
import numpy as np

from fem2d.boundary.conic_merge import ConicSegmentMerger
from fem2d.boundary.plugins.arc_curvature import ArcCurvatureDetector
from fem2d.boundary.registry_mapping import RegionBoundaryMapper
from fem2d.mesh import Mesh
from fem2d.regions import CadCurveRegion, RegionRegistry


def _mesh():
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
             elements=np.array([[0, 1, 2], [0, 3, 2]]),
             E=1e6, nu=0.3, thickness=1.0)
    m.build_connectivity()
    return m


class _Diag:
    """假 diagnostics — 记录 (kind, level) 调用."""

    def __init__(self):
        self.entries = []

    def add(self, kind, level, msg, **kw):
        self.entries.append((kind, level))


def _mapper(diag=None):
    registry = RegionRegistry(cad_boundary_complete=True)
    return RegionBoundaryMapper(_mesh(), registry, diag or _Diag())


def _cad_curve(tag=1, etype="Line", edges=((0, 1), (1, 2)),
               occurrences=((1, 1), (2, -1))):
    return CadCurveRegion(
        entity_tag=tag, entity_type=etype, node_ids=(0, 1, 2),
        edge_pairs=edges, surface_occurrences=occurrences)


# ── registry_mapping: 完整 CAD 契约校验分支 ─────────────────────────────────

def test_cad_curve_empty_and_unowned():
    """空边集 → cad_curve_empty; 无曲面归属 → cad_curve_unowned."""
    diag = _Diag()
    m = _mapper(diag)
    m._validate_complete_cad_curve(
        _cad_curve(edges=(), occurrences=()), set(), (), set(),
        set(), set(), set())
    kinds = {k for k, _ in diag.entries}
    assert "cad_curve_empty" in kinds and "cad_curve_unowned" in kinds


def test_cad_curve_nonmanifold():
    """3+ 曲面共享一条边 → 非流形错误."""
    diag = _Diag()
    m = _mapper(diag)
    m._validate_complete_cad_curve(
        _cad_curve(occurrences=((1, 1), (2, 1), (3, 1))),
        {(0, 1)}, ((1, 1), (2, 1), (3, 1)), {1, 2, 3},
        set(), {(0, 1)}, set())
    assert ("cad_curve_nonmanifold", "error") in diag.entries


def test_cad_curve_missing_edges():
    """线网格边不在位移网格 → cad_curve_missing_edges."""
    diag = _Diag()
    m = _mapper(diag)
    m._validate_complete_cad_curve(
        _cad_curve(), {(9, 10)}, ((1, 1),), {1},
        {(9, 10)}, set(), set())
    assert ("cad_curve_missing_edges", "error") in diag.entries


def test_cad_interface_exposed_and_orientation():
    """双曲面共享 + 同向出现 → 接口暴露错误 + 定向警告."""
    diag = _Diag()
    m = _mapper(diag)
    m._validate_complete_cad_curve(
        _cad_curve(edges=((0, 1),), occurrences=((1, 5), (2, 5))),
        {(0, 1)}, ((1, 5), (2, 5)), {1, 2},
        set(), {(0, 1)}, set())
    kinds = {k for k, _ in diag.entries}
    assert "cad_interface_exposed" in kinds
    assert "cad_interface_orientation" in kinds


def test_cad_partition_edge_overlap():
    """一条边属于多个 CAD 实体 → cad_entity_overlap."""
    from fem2d.boundary.registry_mapping import canonical_edge
    diag = _Diag()
    m = _mapper(diag)
    m.edge_entity_tags[canonical_edge(0, 1)].update({1, 2})
    m._validate_cad_partition()
    assert ("cad_entity_overlap", "error") in diag.entries


def test_physical_region_empty_and_missing():
    """物理曲线无活动边 → empty; 引用缺失边 → missing_edges."""
    from fem2d.regions import CurveRegion
    diag = _Diag()
    m = _mapper(diag)
    region = CurveRegion(
        name="bottom", physical_tag=1, entity_tags=(1,),
        entity_types=("Line",), node_ids=(0, 1), edge_pairs=((0, 1),))
    m._validate_physical_region(
        region, set(), set(), set(), {})
    assert ("physical_curve_empty", "error") in diag.entries

    diag2 = _Diag()
    m2 = _mapper(diag2)
    m2._validate_physical_region(
        region, {(9, 10)}, set(), set(), {})
    assert ("physical_curve_missing_edges", "error") in diag2.entries


def test_record_physical_region_compat_path():
    """无逐边 provenance → 组级 entity_tags 兼容挂载 (line 246)."""
    from fem2d.boundary.registry_mapping import canonical_edge
    diag = _Diag()
    m = _mapper(diag)
    # 用 CurveRegion 形状的假区域: 需要 physical_tag/name/entity_tags/
    # entity_types/edge_pairs 字段
    from fem2d.regions import CurveRegion
    curve = CurveRegion(
        name="ring", physical_tag=2, entity_tags=(7, 8),
        entity_types=("Circle",), node_ids=(0, 1, 2, 3),
        edge_pairs=((0, 1), (1, 2), (2, 3), (3, 0)))
    m._record_physical_region(curve)
    # 外框边 0-1 在 boundary_edges 中且无逐边 provenance → 组级挂载
    edge = canonical_edge(0, 1)
    assert m.edge_entity_tags[edge] == {7, 8}
    assert m.edge_entity_types[edge] == {"Circle"}


# ── arc_curvature: 保护分支 ─────────────────────────────────────────────────

DET = ArcCurvatureDetector()


def test_arc_algebraic_too_few_interior_points():
    """内部采样点 <2 (3 点链) → 无解 (line 102)."""
    coords = np.array([[0., 0.], [1., 0.], [2., 0.]])
    assert DET._arc_algebraic(coords, False) is None


def test_arc_algebraic_nearly_straight():
    """近平直链 → 曲率地板返回 None (line 111)."""
    coords = np.array([[0., 0.], [1., 0.], [2., 1e-12], [3., 0.]])
    assert DET._arc_algebraic(coords, False) is None


def test_algebraic_center_repeated_points():
    """全部点重合 → 弦长 ≤0 → None (line 179)."""
    coords = np.array([[1., 1.], [1., 1.], [1., 1.], [1., 1.]])
    assert DET._algebraic_center(coords) is None


def test_algebraic_center_collinear():
    """共线点 → 垂平分线奇异 (LinAlgError) → None (line 203)."""
    coords = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    assert DET._algebraic_center(coords) is None


def test_short_ellipse_guard_circle_ratio():
    """圆链 (a≈b) → 让位内置圆探测器 → None (line 226)."""
    t = np.linspace(0.0, 1.5 * np.pi, 8)
    coords = np.column_stack([np.cos(t), np.sin(t)])
    assert DET._short_ellipse_guard(coords, 1.0, False) is None


# ── conic_merge: 合并门控 ───────────────────────────────────────────────────

def _seg(nodes, info=None):
    nodes = list(map(int, nodes))
    return {"type": "arc", "nodes": nodes,
            "coords": np.array([[0., 0.]] * len(nodes)),
            "label": "seg", "info": info or {}, "closed": False}


def test_merge_single_segment_returns_as_is():
    """少于 2 段 → 原样返回 (line 41)."""
    m = ConicSegmentMerger(_mesh(), [_seg([0, 1])])
    assert m.merge() == m.segments


def test_merge_key_unsupported_cad_types():
    """类型不唯一/不支持 → 不可合并键 (line 107)."""
    seg = _seg([0, 1], {"cad_entity_types": ("Line", "Circle")})
    assert ConicSegmentMerger._merge_key(seg) is None


def test_merge_key_negative_loop_id():
    """loop_id 缺失 (默认 -1) → 不可合并键 (line 111)."""
    seg = _seg([0, 1], {"cad_entity_types": ("Circle",)})
    assert ConicSegmentMerger._merge_key(seg) is None


def test_unique_edge_owners_ambiguity():
    """两条段共享边 → 归属歧义 → None (line 132)."""
    segs = [_seg([0, 1]), _seg([0, 1])]
    m = ConicSegmentMerger(_mesh(), segs)
    assert m._unique_edge_owners([0, 1]) is None


def test_tangent_continuous_join_turn():
    """CAD 实体接点处切线折角门: 直折 (90°) → False; 直通 → True."""
    mesh = _mesh()
    m = ConicSegmentMerger(mesh, [])
    # 0→1→2: 向量 (1,0) 与 (0,1) 夹角 90° (直折) → 不连续
    assert m._tangent_continuous({(0, 1), (1, 2)},
                                  {(0, 1): 0, (1, 2): 1}) is False
    # 0→1→3: 向量 (1,0) 与 (0,1)... 节点 3=(1,1): (1,0)→(0,1) 仍是 90°
    # 用单位方板对角线路径 0→1→3→2: 0→1 水平, 1→3 竖直 — 折角
    assert m._tangent_continuous({(1, 3), (3, 2)},
                                  {(1, 3): 0, (3, 2): 1}) is False
    # 共线延伸 (同一段内) → 不检查 (owner 相同)
    assert m._tangent_continuous({(0, 1)}, {(0, 1): 0}) is True


def test_restore_source_direction_reverses_cw_outer():
    """闭合外环 CW (负面积) → 节点反转 (line 258)."""
    mesh = _mesh()
    m = ConicSegmentMerger(mesh, [])
    nodes = [0, 2, 3, 1, 0]              # 外环 CW → 负有向面积
    source = [_seg(nodes, {"is_outer": True})]
    m._restore_source_direction(nodes, source, closed=True)
    # 反转后为 CCW: 0 → 1 → 3 → 2 → 0 (正面积)
    assert nodes[1] == 1
