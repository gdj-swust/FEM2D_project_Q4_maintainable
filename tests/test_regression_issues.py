"""审查回归测试 — 6项修复全覆盖"""
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest

from fem2d import Mesh, registered_element_types

# ═══════════════════════════════════════════
# Fix 1: Q4R 已隔离
# ═══════════════════════════════════════════

def test_q4r_is_registered():
    """Q4R 已正式注册为 CPS4R/CPE4R."""
    types = registered_element_types()
    for required in ('Q4R', 'CPS4R', 'CPE4R'):
        assert required in types, f"{required} should be registered"

def test_q4r_file_in_element():
    """Q4R 在 element/ 目录中."""
    assert os.path.exists(
        os.path.join(os.path.dirname(__file__), '..', 'fem2d', 'element', 'q4r.py'))


# ═══════════════════════════════════════════
# Fix 2: allow_internal 已删除
# ═══════════════════════════════════════════

def test_allow_internal_removed_from_signature():
    """add_traction 签名不含 allow_internal."""
    sig = inspect.signature(Mesh.add_traction)
    assert 'allow_internal' not in sig.parameters

def test_internal_edge_traction_rejected():
    """内部边施加面力应直接报错."""
    nodes = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    try:
        m.add_traction(1, 2, 1e6, 0)
        assert False, "interior edge should raise ValueError"
    except ValueError as e:
        assert "boundary edge" in str(e) or "shared by" in str(e)


# ═══════════════════════════════════════════
# Fix 3: lc 科学计数法
# ═══════════════════════════════════════════

def test_lc_scientific_notation():
    """生产 lc 正则可读 1e-3, 0.15, 2.0e-4 — 直接测 input_source._LC_PATTERN
    (曾测测试内联的正则副本, 生产正则改了测试仍绿)."""
    from fem2d.input_source import _LC_PATTERN
    cases = [
        ("lc = 1e-3",    0.001),
        ("lc = 2.0e-4",  0.0002),
        ("lc = 0.15",    0.15),
        ("lc=5E-2",      0.05),
        ("  lc = 0.5",   0.5),   # 前导空白曾漏匹配
    ]
    for text, expected in cases:
        m = re.search(_LC_PATTERN, text, re.MULTILINE)
        assert m is not None, f"regex should match '{text}'"
        assert float(m.group(1)) == expected, f"'{text}' → {float(m.group(1))} ≠ {expected}"


# ═══════════════════════════════════════════
# Fix 4: cKDTree 重复节点
# ═══════════════════════════════════════════

def test_cKDTree_finds_non_adjacent_duplicates():
    """x 接近但 y 差远 → lexsort 漏, cKDTree 不漏."""
    from fem2d.preprocess import validate_mesh
    nodes = np.array([
        [0.0,   0.0],
        [0.001, 0.0],
        [0.002, 100.0],  # 夹在中间但 y 很远
        [0.003, 0.0],
    ], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    r = validate_mesh(nodes, elems, tol=0.01)
    assert len(r["warnings"]) > 0, "cKDTree should detect at least one pair"


# ═══════════════════════════════════════════
# Fix 6: Z 检查 (单元测试)
# ═══════════════════════════════════════════

def test_z_check_error_type_exists():
    """GmshTopologyError 异常类存在."""
    from fem2d.gmsh_adapter import GmshTopologyError
    assert issubclass(GmshTopologyError, ValueError)


def test_z_check_rejects_warped_mesh_behaviorally():
    """Z 检查行为化: 活动节点非平面 z 必须拒绝; 容差相对模型跨度 —
    微米级平坦网格不误杀 (曾测源码字符串, 生产逻辑改了测试仍绿 —
    审计 2026-08-03)."""
    import numpy as np
    from fem2d.gmsh_adapter import GmshTopologyError, _extract_mesh

    class _FakeMeshApi:
        def __init__(self, coords3d):
            self._coords = np.asarray(coords3d, dtype=float)
            self._n = len(self._coords)
        def getNodes(self):
            tags = np.arange(1, self._n + 1, dtype=np.int64)
            return tags, self._coords.reshape(-1).tolist(), []
        def getElements(self, dim, tag=-1):
            # 两个三角形: (1,2,3), (1,3,4)
            blocks = np.array([1, 2, 3, 1, 3, 4], dtype=np.int64)
            return ([2], [np.array([1, 2], dtype=np.int64)], [blocks])
        def getElementProperties(self, element_type):
            return ("Triangle 3", 2, 1, 3, 0, 3)

    class _FakeModel:
        def __init__(self, mesh_api):
            self.mesh = mesh_api
        def getPhysicalGroups(self):
            return []

    class _FakeGmsh:
        def __init__(self, mesh_api):
            self.model = _FakeModel(mesh_api)

    def run(scale, z_warp):
        coords = np.array([[0, 0, 0], [scale, 0, 0],
                           [scale, scale, z_warp], [0, scale, 0]], dtype=float)
        return _extract_mesh(_FakeGmsh(_FakeMeshApi(coords)))

    run(1.0, 0.0)                      # 单位尺度平坦 → 通过
    run(1e-9, 0.0)                     # 微米尺度平坦 → 通过 (不误杀)
    run(1e-9, 1e-19)                   # 微米尺度近平坦 (1e-10×跨度) → 通过
    with pytest.raises(GmshTopologyError, match="non-constant z"):
        run(1.0, 0.5)                  # 翘曲 50% 跨度 → 拒绝


# ═══════════════════════════════════════════
# Fix: 位移驱动近刚体工况不再被力矩平衡检查误杀
# ═══════════════════════════════════════════

def test_rigid_prescribed_displacement_passes_balance_check():
    """整体指定位移 (纯刚体平动, 反力=舍入噪声) 不应触发平衡检查误报.

    修复前: ΣM 与分母 Σ|力矩| 同为舍入量级, 相对判据退化 (rel≈1) →
    RuntimeError 误杀支座沉降/位移控制类合法算例.
    """
    from fem2d import solve
    x = np.linspace(0.0, 1.0, 3)
    y = np.linspace(0.0, 1.0, 3)
    nodes = np.array([[xi, yi] for yi in y for xi in x])
    elems = []
    for i in range(2):
        for j in range(2):
            a = j * 3 + i
            elems.append([a, a + 1, a + 4, a + 3])
    m = Mesh(nodes=nodes, elements=np.array(elems), E=210e9, nu=0.3,
             thickness=0.01, plane_type="stress", elem_type="CPS4")
    for nid in range(9):
        m.fix_node(nid, "y", 1e-4)      # 整体上移 → 纯刚体平动
    for nid in (0, 3, 6):
        m.fix_node(nid, "x", 0.0)       # 左侧列约束 x
    r = solve(m, verbose=False)
    u2 = r["u"].reshape(-1, 2)
    assert np.allclose(u2[:, 1], 1e-4, atol=1e-16)
    assert np.max(np.abs(u2[:, 0])) < 1e-14
