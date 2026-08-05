"""覆盖轮 C1 — Mesh 构造/状态校验分支 + regions 区域注册表校验."""
import numpy as np
import pytest

from fem2d import Mesh
from fem2d.regions import (
    CadCurveRegion,
    CurveRegion,
    PointRegion,
    RegionRegistry,
    SurfaceRegion,
    ordered_edge_chains,
)


def _mesh():
    return Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                elements=np.array([[0, 1, 2], [0, 2, 3]]),
                E=1e6, nu=0.3, thickness=1.0)


# ── mesh.py 构造校验 ────────────────────────────────────────────────────────

def test_mesh_empty_elements_rejected():
    with pytest.raises(ValueError, match="at least one element"):
        Mesh(nodes=np.array([[0., 0.]]), elements=np.zeros((0, 3), dtype=int))


def test_mesh_nan_elements_rejected():
    with pytest.raises(ValueError, match="NaN or Inf"):
        Mesh(nodes=np.zeros((4, 2)), elements=np.array([[0., 1., np.nan]]))


def test_mesh_float_integer_elements_rounded():
    """浮点但恰为整数的连接 → rint 规范化取整 (不拒绝)."""
    m = Mesh(nodes=np.zeros((4, 2)),
             elements=np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 3.0]]))
    assert m.elements.dtype == np.int64


def test_mesh_out_of_bounds_rejected():
    with pytest.raises(ValueError, match="out of bounds"):
        Mesh(nodes=np.zeros((4, 2)), elements=np.array([[0, 1, 9]]))


def test_mesh_duplicate_elements_rejected():
    with pytest.raises(ValueError, match="重复单元"):
        Mesh(nodes=np.zeros((4, 2)),
             elements=np.array([[0, 1, 2], [0, 1, 2]]))


def test_replace_elements_wrong_shape():
    m = _mesh()
    with pytest.raises(ValueError, match="n_elem"):
        m.replace_elements(np.zeros((2, 4)))


def test_validate_state_empty_nodes_after_rewrite():
    """构造后私有缓存被外部改写为空 → 求解前复检拒绝."""
    m = _mesh()
    m._nodes = np.zeros((0, 2))
    with pytest.raises(ValueError, match="at least one node"):
        m.validate_state()


def test_validate_state_nan_nodes():
    m = _mesh()
    m._nodes = np.array([[0., 0.], [np.nan, 0.], [0., 1.], [1., 1.]])
    with pytest.raises(ValueError, match="nodes contain NaN"):
        m.validate_state()


def test_validate_state_nan_elements():
    m = _mesh()
    m._elements = np.array([[0, 1, 2], [0, 2, np.nan]], dtype=float)
    with pytest.raises(ValueError, match="elements contain NaN"):
        m.validate_state()


def test_validate_bc_prescribed_key_not_fixed():
    m = _mesh()
    m.fixed_dofs = np.array([0], dtype=int)
    m.prescribed_vals = {99: 0.0}
    with pytest.raises(ValueError, match="not in fixed_dofs"):
        m.validate_state()


def test_validate_loads_concentrated_not_dict():
    m = _mesh()
    m.concentrated_forces.append(123)
    with pytest.raises(ValueError, match="must be a dict"):
        m.validate_state()


def test_validate_loads_traction_not_dict():
    m = _mesh()
    m.surface_tractions.append(123)
    with pytest.raises(ValueError, match="must be a dict"):
        m.validate_state()


def test_validate_loads_nodes_pair_invalid_type():
    m = _mesh()
    m.surface_tractions.append({"nodes": 5, "traction": (1.0, 0.0)})
    with pytest.raises(ValueError, match="node pair"):
        m.validate_state()


def test_build_connectivity_missing_areas(monkeypatch):
    m = _mesh()
    monkeypatch.setattr(m.element_kernel, "build_geometry",
                        lambda nodes, elements: {"centroids": np.zeros(2)})
    with pytest.raises(RuntimeError, match="'areas'"):
        m.build_connectivity()


def test_build_connectivity_missing_centroids(monkeypatch):
    m = _mesh()
    monkeypatch.setattr(m.element_kernel, "build_geometry",
                        lambda nodes, elements: {"areas": np.ones(2)})
    with pytest.raises(RuntimeError, match="'centroids'"):
        m.build_connectivity()


def test_fix_nodes_func_string_rejected():
    m = _mesh()
    with pytest.raises(ValueError, match="字符串"):
        m.fix_nodes_func("0,1", 0.0)


def test_get_edge_elements_not_a_mesh_edge():
    m = _mesh()
    with pytest.raises(ValueError, match="not a mesh edge"):
        m._get_edge_elements(0, 99)


def test_boundary_outward_normal_zero_length_edge():
    """相邻重合节点 → 零长边界边 → 外法向无法定义."""
    nodes = np.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2]]), E=1e6, nu=0.3)
    with pytest.raises(ValueError, match="Zero-length edge"):
        m.boundary_outward_normal(0, 1)


def test_boundary_outward_normal_local_edge_mismatch():
    """边表与单元连接不一致 (edge_to_elems 被破坏) → 防御兜底 RuntimeError."""
    m = _mesh()
    m.build_connectivity()
    m.edge_to_elems = {(0, 1): [1]}  # 元素 1 的 conn=[0,2,3] 不含节点 1
    with pytest.raises(RuntimeError, match="not found in adjacent"):
        m.boundary_outward_normal(0, 1)


def test_add_traction_node_out_of_range():
    m = _mesh()
    with pytest.raises(ValueError, match="add_traction"):
        m.add_traction(0, 99, 1.0, 0.0)


def test_add_pressure_node_out_of_range():
    m = _mesh()
    with pytest.raises(ValueError, match="add_pressure"):
        m.add_pressure(0, 99, 1.0)


def test_nodes_on_edge_degenerate_span():
    """跨度退化 (1e200 坐标, 跨度≪ULP) → tiny 兜底容差, 不误报整条边."""
    nodes = np.array([[1e200, 0.], [1e200, 0.], [1e200, 1e-200]])
    m = Mesh(nodes=nodes, elements=np.array([[0, 1, 2]]), E=1e6, nu=0.3)
    result = m.nodes_on_edge("x", "min")
    assert isinstance(result, (list, np.ndarray))


def test_check_rigid_body_isolated_component():
    """孤立节点 (无单元连接) 分量 → 记录孤立分量问题."""
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [5., 5.]])
    m = Mesh(nodes=nodes,
             elements=np.array([[0, 1, 2], [0, 2, 3]]), E=1e6, nu=0.3)
    m.fix_node(0, "both")
    issues = m.check_rigid_body_constraints()
    assert any("孤立" in issue["issue"] for issue in issues)


def test_check_rigid_body_rank_below_2():
    """仅约束 1 个 DOF → rank<2 → 缺失列表 (数值上缺失转动)."""
    m = _mesh()
    m.fix_node(3, "x")
    issues = m.check_rigid_body_constraints()
    assert any("转动" in issue["issue"] for issue in issues)


def test_check_rigid_body_rank_2():
    """仅固定 1 节点 both → rank=2 → 缺转动 (约束共线)."""
    m = _mesh()
    m.fix_node(0, "both")
    issues = m.check_rigid_body_constraints()
    assert any("转动" in issue["issue"] for issue in issues)


def test_mesh_info_summary():
    m = _mesh()
    text = m.info()
    assert "Mesh:" in text and "nodes" in text


# ── regions.py ─────────────────────────────────────────────────────────────

def test_ordered_edge_chains_branch_rejected():
    """节点度 >2 (分叉) → 线积分有歧义 → 拒绝."""
    with pytest.raises(ValueError, match="branches"):
        ordered_edge_chains([(0, 1), (1, 2), (2, 3), (1, 4)])


def test_curve_region_components_property():
    region = CurveRegion(
        name="c", physical_tag=1, entity_tags=(1,), entity_types=("Line",),
        node_ids=(0, 1, 2), edge_pairs=((0, 1), (1, 2)), edge_entities=())
    assert region.components == ((0, 1, 2),)


def test_cad_curve_region_components_property():
    region = CadCurveRegion(
        entity_tag=1, entity_type="Line",
        node_ids=(0, 1, 2), edge_pairs=((0, 1), (1, 2)),
        surface_occurrences=((1, 1),))
    assert region.components == ((0, 1, 2),)


def _registry_with_invalid_indices():
    reg = RegionRegistry()
    reg.points.append(PointRegion(
        name="p", physical_tag=1, entity_tags=(), node_ids=(99,)))
    reg.curves.append(CurveRegion(
        name="c", physical_tag=2, entity_tags=(), entity_types=(),
        node_ids=(0, 99), edge_pairs=((0, 99),), edge_entities=()))
    reg.cad_curves.append(CadCurveRegion(
        entity_tag=3, entity_type="Line", node_ids=(99,),
        edge_pairs=((99, 100),), surface_occurrences=()))
    reg.surfaces.append(SurfaceRegion(
        name="s", physical_tag=4, entity_tags=(), entity_types=(),
        element_ids=(99,), oriented_boundary_entities=()))
    return reg


def test_registry_validate_indices_accumulates_errors():
    reg = _registry_with_invalid_indices()
    with pytest.raises(ValueError, match="validation failed"):
        reg.validate_indices(node_count=4, element_count=2)


def test_registry_validate_against_mesh_missing_edges():
    m = _mesh()
    reg = RegionRegistry()
    reg.curves.append(CurveRegion(
        name="c", physical_tag=2, entity_tags=(), entity_types=(),
        node_ids=(0, 99), edge_pairs=((0, 99),), edge_entities=()))
    reg.cad_curves.append(CadCurveRegion(
        entity_tag=3, entity_type="Line", node_ids=(99,),
        edge_pairs=((99, 100),), surface_occurrences=()))
    with pytest.raises(ValueError, match="validation failed"):
        reg.validate_against_mesh(m)


def test_registry_surface_area_empty():
    reg = RegionRegistry()
    assert reg.surface_area("nope", _mesh()) == 0.0
