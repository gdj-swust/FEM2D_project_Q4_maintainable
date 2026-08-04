"""插件 2 判别性测试 — @组名批量选择 (CLI 输入层, 轮 2).

判别性 (放回旧实现必须失败):
  - @组名展开结果 == 手输全部编号的施加结果 (组级语义一致, 逐边核对)
  - 组名不存在 → CliError(exit_code=1) (e2e 退出码 1, 锁定决策)
  - 编号+@ 混用 → ValueError (锁定决策)
  - 旧编号/名称语法完全兼容 — 现有全部测试零变化 (全量 pytest 锁定)
  - 大小写不敏感 (by_name 语义)
"""
import numpy as np
import pytest

from fem2d.boundary import _resolve_edge_indices
from fem2d.config import AnalysisConfig
from fem2d.regions import CurveRegion, RegionRegistry
from tests.conftest import _mesh


def _segments_and_registry():
    """方板 4 段: 段 1/3 属组 组A, 段 2 属组 组B, 段 4 无组."""
    segments = [
        {"type": "line", "label": "组A | 外边 直边 (0,0)→(1,0)",
         "info": {"physical_names": ("组A",)},
         "nodes": [0, 1], "coords": np.array([[0, 0], [1, 0]])},
        {"type": "line", "label": "组B | 外边 直边 (1,0)→(1,1)",
         "info": {"physical_names": ("组B",)},
         "nodes": [1, 2], "coords": np.array([[1, 0], [1, 1]])},
        {"type": "line", "label": "组A | 外边 直边 (1,1)→(0,1)",
         "info": {"physical_names": ("组A",)},
         "nodes": [2, 3], "coords": np.array([[1, 1], [0, 1]])},
        {"type": "line", "label": "外边 直边 (0,1)→(0,0)",
         "info": {"physical_names": ()},
         "nodes": [3, 0], "coords": np.array([[0, 1], [0, 0]])},
    ]
    registry = RegionRegistry(curves=[
        CurveRegion(
            name="组A", physical_tag=1, entity_tags=(1, 3),
            entity_types=("Line",), node_ids=(0, 1, 2, 3),
            edge_pairs=((0, 1), (1, 2), (2, 3), (3, 0))),
        CurveRegion(
            name="组B", physical_tag=2, entity_tags=(2,),
            entity_types=("Line",), node_ids=(1, 2),
            edge_pairs=((1, 2),)),
    ])
    return segments, registry


def test_group_expansion_equals_manual_numbers():
    """@组名展开结果 == 手输全部编号的并集 (判别性核心)."""
    segments, registry = _segments_and_registry()
    expanded = _resolve_edge_indices("@组A", segments, registry)
    assert expanded == [0, 2]
    # 手输编号通道
    assert _resolve_edge_indices("1", segments, registry) == [0]
    assert _resolve_edge_indices("3", segments, registry) == [2]
    # 展开 == 手输全部编号的并集
    assert expanded == sorted(
        _resolve_edge_indices("1", segments, registry)
        + _resolve_edge_indices("3", segments, registry))
    assert _resolve_edge_indices("@组B", segments, registry) == [1]


def test_group_expansion_case_insensitive():
    """by_name 语义: @组名 大小写不敏感."""
    segments, registry = _segments_and_registry()
    assert _resolve_edge_indices("@组a", segments, registry) == [0, 2]
    assert _resolve_edge_indices("@组b", segments, registry) == [1]


def test_group_missing_is_error():
    """锁定决策: 组名不存在 → ValueError (批处理路径转 CliError exit 1)."""
    segments, registry = _segments_and_registry()
    with pytest.raises(ValueError, match="不存在"):
        _resolve_edge_indices("@不存在", segments, registry)
    # 无注册表 (纯 .txt 输入) → 同样响亮报错
    with pytest.raises(ValueError, match="注册表"):
        _resolve_edge_indices("@组A", segments, None)
    # 空组名
    with pytest.raises(ValueError, match="缺组名"):
        _resolve_edge_indices("@", segments, registry)


def test_group_number_mixing_rejected():
    """锁定决策: 不支持编号+@ 混用 (单个选择串内)."""
    segments, registry = _segments_and_registry()
    with pytest.raises(ValueError, match="混用"):
        _resolve_edge_indices("12,@组A", segments, registry)
    with pytest.raises(ValueError, match="混用"):
        _resolve_edge_indices("@组A,12", segments, registry)
    with pytest.raises(ValueError, match="混用"):
        _resolve_edge_indices("左,@组A", segments, registry)


def test_group_apply_equals_manual_apply():
    """施加级判别: @组名 施加 == 手输全部编号施加 (逐边 traction 一致)."""
    from fem2d.bc_apply import _apply_tractions

    def _tractions(traction_spec):
        mesh = _mesh()
        segments, registry = _segments_and_registry()
        config = AnalysisConfig(traction=traction_spec)
        # 批处理模式: config.traction 非空 → 走 CLI 路径, 不交互
        _apply_tractions(config, mesh, segments, batch_mode=True,
                         region_registry=registry)
        return list(mesh.surface_tractions)

    grouped = _tractions("@组A:0,1e6")
    manual = _tractions("1:0,1e6;3:0,1e6")
    assert len(grouped) == 2, f"@展开应施加 2 段共 2 条边: {grouped}"
    assert grouped == manual, (
        f"@组名 施加与手输编号不一致:\n  @组A: {grouped}\n  手输: {manual}")


def test_group_missing_exit_code_1_runner():
    """e2e 判别: 组名不存在 (无 gmsh 注册表) → 退出码 1."""
    from pathlib import Path

    from fem2d.runner import main

    model = str(Path(__file__).resolve().parents[1]
                / "models" / "test_simple.txt")
    assert main([model, "--traction", "@不存在:0,1e6",
                 "--no-plot"]) == 1


@pytest.mark.skipif(
    not __import__("tests.conftest", fromlist=["GMSH_AVAILABLE"]).
    GMSH_AVAILABLE,
    reason="Gmsh Python API unavailable or native dependency missing")
def test_group_expansion_real_gmsh_model():
    """真实 gmsh 模型: demo_complex 椭圆孔组 (20 段 Line 合并整环)
    → @椭圆孔 展开 == 该整环段的编号."""
    from fem2d import Mesh, build_boundary_segments
    from pathlib import Path

    from fem2d.gmsh_adapter import generate_from_geo

    result = generate_from_geo(str(
        Path(__file__).resolve().parents[1] / "models" / "demo_complex.geo"))
    mesh = Mesh(nodes=result.nodes, elements=result.elements,
                E=210e9, nu=0.3, thickness=0.01,
                elem_type=result.elem_type)
    segments = build_boundary_segments(mesh, registry=result.regions)

    expanded = _resolve_edge_indices("@椭圆孔", segments, result.regions)
    assert len(expanded) == 1, f"@椭圆孔 应展开为 1 个合并整环段: {expanded}"
    index = expanded[0]
    assert segments[index]["type"] == "ellipse"
    assert "椭圆" in segments[index]["label"]

    # 手输编号通道一致 (兼容)
    assert _resolve_edge_indices(
        str(index + 1), segments, result.regions) == [index]


@pytest.mark.skipif(
    not __import__("tests.conftest", fromlist=["GMSH_AVAILABLE"]).
    GMSH_AVAILABLE,
    reason="Gmsh Python API unavailable or native dependency missing")
def test_group_missing_exit_code_1_runner_gmsh():
    """e2e 判别: gmsh 注册表存在但组名不存在 → 退出码 1 (锁定决策)."""
    from fem2d.runner import main

    from pathlib import Path
    model = str(Path(__file__).resolve().parents[1]
                / "models" / "demo_complex.geo")
    assert main([model, "--traction", "@不存在:0,1e6",
                 "--no-plot"]) == 1
