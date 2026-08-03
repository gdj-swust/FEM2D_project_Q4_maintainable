"""bc_apply.apply_bcs 全分支覆盖 — 固定边/面力(常量/压力/profile)/集中力/体力.

原覆盖率 32%: CLI 主路径只经 main 测试覆盖 fix+traction 常量分支.
本文件直接调用 apply_bcs, 覆盖 profile 链、压力、错误分支与体力捷径输入.
"""
import contextlib
import io

import numpy as np
import pytest

from fem2d import Mesh
from fem2d.errors import CliError
from fem2d.bc_apply import apply_bcs
from fem2d.config import AnalysisConfig


def _mesh_segs():
    """方形 4 节点 + 3 段边界 (左/右 直线 + 顶弧)."""
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


def _apply(config, mesh, segs):
    with contextlib.redirect_stdout(io.StringIO()):
        return apply_bcs(config, mesh, segs, None, {}, None)


def test_fix_all_forms_and_traction_constant():
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(
        fix="左端", fix_ux="右端", fix_uy="右端",
        traction="顶弧:1e6,0", body="0,-78000")
    bfx, bfy = _apply(config, mesh, segs)
    assert len(mesh.fixed_dofs) >= 4   # 左端 both (2 节点 × 2) + 右端 x/y
    assert len(mesh.surface_tractions) == 1
    assert bfy == -78000.0
    assert mesh.body_force == (0.0, -78000.0)


def test_traction_normal_pressure_profile_n():
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(fix="左端", traction="右端:2e6:n")
    _apply(config, mesh, segs)
    assert len(mesh.surface_tractions) == 1
    assert mesh.surface_tractions[0].get("is_pressure") is True


def test_traction_arc_length_profiles():
    """:p 抛物线弧长面力 — 走 ordered_edge_chains 链分支."""
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(fix="左端", traction="顶弧:1e6,0:p")
    _apply(config, mesh, segs)
    assert len(mesh.surface_tractions) == 1


def test_force_numeric_node_and_single_number_body():
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(
        fix="左端", force="0,100,200", body="-78000")  # 单数字 = y 分量
    _apply(config, mesh, segs)
    assert mesh.concentrated_forces[0]["node"] == 0
    assert mesh.concentrated_forces[0]["force"] == (100.0, 200.0)
    assert mesh.body_force == (0.0, -78000.0)


def test_force_requires_three_fields():
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(fix="左端", force="0,100")
    with pytest.raises(CliError):
        _apply(config, mesh, segs)


def test_force_numeric_node_with_node_id_map():
    """node_id_map 存在时 — Gmsh 原始节点号映射到内部编号."""
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(fix="左端", force="10,1,2")  # Gmsh 节点 10
    with contextlib.redirect_stdout(io.StringIO()):
        apply_bcs(config, mesh, segs, None, {10: 0, 11: 1, 12: 2, 13: 3},
                  None)
    assert mesh.concentrated_forces[0]["node"] == 0


def test_force_unknown_node_fatal():
    """node_id_map 非空时 — 不在映射中的 Gmsh 节点号必须 FATAL."""
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(fix="左端", force="99,1,2")
    with pytest.raises(CliError), contextlib.redirect_stdout(io.StringIO()):
        apply_bcs(config, mesh, segs, None,
                  {10: 0, 11: 1, 12: 2, 13: 3}, None)


def test_bad_edge_name_fatal_in_batch():
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(fix="不存在的边")
    with pytest.raises(CliError):
        _apply(config, mesh, segs)


def test_bad_traction_edge_fatal():
    mesh, segs = _mesh_segs()
    config = AnalysisConfig(fix="左端", traction="不存在:1e6,0")
    with pytest.raises(CliError):
        _apply(config, mesh, segs)


def test_interactive_bc_path(monkeypatch):
    """交互分支: fix/traction/body 全部经 ask() 提问 (无 CLI 参数时)."""
    from fem2d import bc_apply

    # ask 回答序列: 交互 fix(边1 both=0, 边1 Uy=0.5) → 交互 traction(边1)
    # → 交互 body(单数字补逗号)
    answers = ["1", "", "",          # 边1: Ux 空, Uy 空 → both=0
               "1", "", "0.5",       # 边1: Ux 空, Uy=0.5
               "",                   # fix 结束
               "1", "1e6", "0",      # 交互 traction 边1: tx=1e6, ty=0
               "",                   # traction 结束
               "-78000"]             # body 单数字 = y 分量
    queue = list(answers)

    def fake_ask(prompt):
        return queue.pop(0) if queue else ""

    monkeypatch.setattr(bc_apply, "ask", fake_ask)
    monkeypatch.setattr(bc_apply, "is_batch_mode", lambda config: False)

    mesh, segs = _mesh_segs()
    config = AnalysisConfig()  # 无任何 CLI BC → 交互模式
    bfx, bfy = _apply(config, mesh, segs)
    # fix: 边1 (左端 节点 0,1) — both=0 施加到节点 0/1
    assert len(mesh.fixed_dofs) >= 4
    # traction: 边1 → 左端节点 0-1 一条边
    assert len(mesh.surface_tractions) == 1
    # body: 单数字 → y 分量
    assert bfy == -78000.0


# ═══════════════════════════════════════════════════════════════
# BC 公共 API 输入校验
# ═══════════════════════════════════════════════════════════════

def test_bc_api_rejects_invalid_inputs():
    """apply_penalty/apply_elimination 公开 API 必须拒绝非法输入 —
    曾 NaN/Inf 罚因子进入 K_mod、布尔掩码约束错 DOF、负 DOF 静默
    约束最后一个."""
    import numpy as np
    import pytest
    from scipy.sparse import csr_matrix
    from fem2d.bc import apply_penalty, apply_elimination

    K = csr_matrix(np.eye(6))
    F = np.zeros(6)
    with pytest.raises(ValueError, match="finite"):
        apply_penalty(K, F, [0], penalty=np.nan)
    with pytest.raises(ValueError, match="finite"):
        apply_penalty(K, F, [0], penalty=np.inf)
    with pytest.raises(ValueError, match="boolean"):
        apply_penalty(K, F, [True])
    with pytest.raises(ValueError, match="out of range"):
        apply_elimination(K, F, [1, 2, 3, 4, 5], [-1], [0.0])
    with pytest.raises(ValueError, match="out of range"):
        apply_elimination(K, F, [1, 2, 3, 4, 5], [99], [0.0])
    # 重复 DOF 不同值必须拒绝 (同值幂等去重允许)
    with pytest.raises(ValueError, match="重复"):
        apply_elimination(K, F, [1, 2, 3, 4, 5], [0, 0], [0.0, 1.0])
    u, r = apply_elimination(K, F, [1, 2, 3, 4, 5], [0, 0], [0.0, 0.0])
    assert u.shape == (6,)
