"""恢复的兼容 API 回归锁定 — 防止再次被清理工具删除 (评审建议).

每个测试模拟"外部用户"按旧签名调用:
  通过 = 签名与行为被锁定, 删除或改动会直接红.
  这些 API 曾因死代码清理被误删/改签名, 恢复后必须用测试钉住.
"""
import numpy as np
import pytest

from fem2d import BoundaryDiagnostics, ElementKernel, RegionRegistry
from fem2d.boundary.geometry import segment_by_curvature
from fem2d.config import AnalysisConfig
from fem2d.preprocess import MeshValidationError, validate_mesh
from fem2d.reporting import build_warnings

# ── 1. BoundaryDiagnostics.summary (顶层导出类的公开方法) ──

def test_boundary_diagnostics_summary():
    """summary() 必须存在且返回诊断统计 dict."""
    diag = BoundaryDiagnostics()
    s = diag.summary()
    assert isinstance(s, dict)
    assert "warnings" in s and "errors" in s
    assert s["warnings"] == 0 and s["errors"] == 0


# ── 2. RegionRegistry.summary ──

def test_region_registry_summary():
    """summary() 必须存在且返回区域统计 dict."""
    reg = RegionRegistry()
    s = reg.summary()
    assert isinstance(s, dict)
    assert "point_regions" in s and "surface_regions" in s
    assert "cad_boundary_complete" in s


# ── 3. ElementKernel.matches ──

def test_element_kernel_matches():
    """matches() 必须存在: 大小写不敏感匹配 name 与全部别名."""
    from fem2d.element import get_element_kernel
    q4 = get_element_kernel("Q4")  # Q4 是注册实例, 不是类
    assert q4.matches("CPE4") is True
    assert q4.matches("q4") is True
    assert q4.matches("CPS4R") is False
    # 协议类型本身存在且可调用
    assert callable(ElementKernel.matches)


# ── 4. validate_mesh 旧位置签名 (nodes, elements, elem_type, tol) ──

def test_validate_mesh_old_positional_signature():
    """评审复现场景: validate_mesh(nodes, elements, 'CPS3') 位置传参.

    曾因删除 elem_type 参数变成 TypeError: must be real number, not str.
    """
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    quad = np.array([[0, 1, 2, 3]])

    r = validate_mesh(nodes, tri, "CPS3")
    assert r["ok"] is True
    r2 = validate_mesh(nodes, quad, "Q4")   # 项目自己的名称
    assert r2["ok"] is True
    r3 = validate_mesh(nodes, quad, "CPS4", 1e-9)  # 四参数位置形式
    assert r3["ok"] is True
    # 类型与节点数不匹配必须拒绝
    with pytest.raises(MeshValidationError):
        validate_mesh(nodes, quad, "CPS3")


# ── 5. segment_by_curvature 两种旧形式 (三参 + 两参) ──

def test_segment_by_curvature_old_signatures():
    """三参 (kappa, coords, scale) 与两参 (kappa, scale) 都必须工作.

    两参位置调用曾把 scale 误当 coords 导致 TypeError.
    """
    kappa = np.sin(np.linspace(0.0, 6.28, 40))
    coords = np.zeros((40, 2))
    r3 = segment_by_curvature(kappa, coords, 1.0)
    assert isinstance(r3, list)
    r2 = segment_by_curvature(kappa, 1.0)  # 旧两参形式
    assert isinstance(r2, list)
    assert r2 == r3  # 两种形式必须等价


# ── 6. build_warnings 旧签名 (config, mesh, ...) ──

def test_build_warnings_old_signature():
    """build_warnings(config, mesh, z2, ...) 位置传参必须工作."""
    config = AnalysisConfig(E=3e7, nu=0.3)
    class _FakeMesh:
        pass
    z2 = {"eta": 5.0, "worst_elem": 0, "elem_contrib": [0.5],
          "total_error": 1e-6, "energy_norm": 1.0,
          "stress_jumps": {"avg_jump": 1e-7, "max_jump": 1e-6}}
    w = build_warnings(config, _FakeMesh(), z2, False, 5, 5)
    assert isinstance(w, list)


# ── 7. 顶层惰性导出兼容 (PEP 562) ──

def test_top_level_lazy_export_run_cantilever_convergence():
    """from fem2d import run_cantilever_convergence 必须可用 (惰性导出).

    曾因消除 runpy 警告而移除此顶层 API — 兼容性破坏 (版本号未变,
    公开 API 不得删除)。PEP 562 __getattr__ 惰性导出保持兼容且不
    急切导入 convergence 模块。子进程验证 (pytest 会话内其他测试
    已导入过 fem2d.convergence, sys.modules 断言会误判)。
    """
    import subprocess
    import sys
    code = (
        "import sys; import fem2d; "
        "assert 'fem2d.convergence' not in sys.modules, '急切导入'; "
        "assert 'run_cantilever_convergence' in fem2d.__all__; "
        "from fem2d import run_cantilever_convergence; "
        "assert callable(run_cantilever_convergence); "
        "assert 'fem2d.convergence' in sys.modules, '访问后未导入'"
    )
    r = subprocess.run([sys.executable, "-c", code],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_top_level_unknown_attribute_raises():
    """未导出的名字必须抛 AttributeError (PEP 562 语义)."""
    import fem2d
    try:
        getattr(fem2d, "no_such_symbol_xyz")
    except AttributeError:
        return
    raise AssertionError("未知属性未抛 AttributeError")
