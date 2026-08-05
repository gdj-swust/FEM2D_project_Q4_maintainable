"""D1 判别性测试 — @段名 引用 (CLI 段编号易错 → 段标签通道).

判别性 (回滚改动必须红):
  - @段名 命中段标签 == 段编号等效 (无 region_registry 场景旧版报错
    "需要注册表" → 新版正确命中并施加)
  - 段名不存在 → 报错列出可用段名
  - @组名 (物理组) 通道不受影响 — 标签精确匹配不误吞复合标签
"""
import contextlib
import io

import numpy as np
import pytest

from fem2d import Mesh
from fem2d.bc_apply import _resolve_boundary_selection, apply_bcs
from fem2d.config import AnalysisConfig
from fem2d.errors import CliError


def _segs():
    """方形 4 节点 + 3 段边界 (标签: 左端/右端/顶弧)."""
    nodes = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    mesh = Mesh(
        nodes=nodes,
        elements=np.array([[0, 1, 2], [0, 2, 3]]),
        E=2.1e11, nu=0.3, thickness=0.01, elem_type="CPS3")

    def seg(label, tp, ns, **info):
        return {"label": label, "type": tp, "nodes": list(ns),
                "info": info, "coords": nodes[ns]}

    segs = [
        seg("左端", "line", [0, 1]),
        seg("右端", "line", [3, 2]),
        seg("顶弧", "arc", [1, 2], radius=0.5),
    ]
    return mesh, segs


def test_at_label_equals_number_without_registry():
    """判别性核心: 无注册表时 @段名 命中 == 段编号 (旧版此处报错)."""
    mesh, segs = _segs()
    by_label = _resolve_boundary_selection(
        "@右端", segs, fatal=True, region_registry=None)
    by_number = _resolve_boundary_selection(
        "2", segs, fatal=True, region_registry=None)
    assert by_label == by_number == [1]


def test_at_label_case_insensitive():
    mesh, segs = _segs()
    assert _resolve_boundary_selection(
        "@左端", segs, fatal=True) == [0]


def test_at_label_missing_lists_available_names():
    """.spec 判别: 段名不存在 → 报错列出可用段名."""
    mesh, segs = _segs()
    with pytest.raises(CliError) as exc_info:
        _resolve_boundary_selection("@不存在", segs, fatal=True)
    assert "可用段名" in str(exc_info.value)
    assert "左端" in str(exc_info.value)
    assert "顶弧" in str(exc_info.value)


def test_at_label_missing_warns_in_interactive():
    """交互路径 (fatal=False): 同样列出可用段名, 不崩溃."""
    mesh, segs = _segs()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        matched = _resolve_boundary_selection(
            "@不存在", segs, fatal=False)
    assert matched == []
    assert "可用段名" in out.getvalue()


def test_at_group_name_not_swallowed_by_composite_label():
    """.spec 判别: 复合标签 ("组A | 外边 直边 …") 不被子串误命中 —
    @组名 (物理组通道) 仍可到达 (回退边界/naming 原路径)."""
    mesh, segs = _segs()
    # 段 0 改为复合标签: "@组A" 不得命中它 (精确匹配语义)
    segs[0]["label"] = "组A | 外边 直边 (0,0)→(1,0)"
    with pytest.raises(CliError) as exc_info:
        _resolve_boundary_selection(
            "@组A", segs, fatal=True, region_registry=None)
    # 物理组通道错误仍带原文案 (注册表缺失), 标签通道未越权
    assert "注册表" in str(exc_info.value)


def test_at_label_apply_end_to_end():
    """施加级判别: fix/traction 用 @段名 (无注册表) == 编号施加."""
    mesh, segs = _segs()
    config = AnalysisConfig(
        fix="@左端", traction="@右端:1e6,0")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        apply_bcs(config, mesh, segs, None, {}, None)
    assert len(mesh.fixed_dofs) >= 4
    assert len(mesh.surface_tractions) == 1
    assert "面力" in out.getvalue()
