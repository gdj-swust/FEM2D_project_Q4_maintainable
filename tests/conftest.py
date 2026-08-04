"""测试会话级配置 — 中文字体缺字警告收敛 + 公共 fixture (测试基建收敛).

绘图测试经 plot_three/colorbar 渲染带中文标签 (如 "磨平"/"面力") 的
图形。matplotlib 在无 CJK 字体机器上对每个缺失字形发 UserWarning
"Glyph ... missing from font(s)" — 纯渲染噪音: 测试只断言图内容
(collections/lines), 不检查字形; fem2d.visualize 在字体存在时已显式
选用 CJK 字体, 本过滤只兜底字体缺失环境, 不改任何断言语义。

必须用 pytest 的 filterwarnings 机制 (pytest_configure 注册), 而非
模块级 warnings.filterwarnings — pytest 以 catch_warnings(record=True)
捕获 (其 "always" 记录器位于全局过滤器之前), 模块级过滤被绕过,
警告仍会出现在汇总里。

共享脚手架 (曾逐文件复制 ≥6 份, pkg12 收敛于此):
- GMSH_AVAILABLE / gmsh_available: gmsh Python API 可用性守卫。
  try/except 双捕 (ImportError + OSError — 缺 libGLU 等原生依赖时
  gmsh 加载抛 OSError, importorskip 只捕 ImportError); 语义与
  test_msh_import_audit 原有模块级 skipif 块一致。
- mesh_from_geo / mesh_result_from_geo: 临时 .geo → generate_from_geo
  → 3 元组 / 完整结果对象, 复制于 test_boundary_complex/gmsh/stress/
  highpressure 的 4 份同构实现。
- square_mesh / quad_mesh: 2×2 单位方板 CPS4 网格 fixture (节点
  CCW), 对应 test_physical_point_resolution._square_mesh /
  test_solver_branches 等文件的 _quad 语义。
"""
import os
import tempfile

import numpy as np
import pytest

try:
    import gmsh as _gmsh
except (ImportError, OSError):
    _gmsh = None

GMSH_AVAILABLE = _gmsh is not None
GMSH_UNAVAILABLE_REASON = "Gmsh Python API unavailable or native dependency missing"


def pytest_configure(config):
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Glyph \\d+ \\(.*\\) missing from font\\(s\\):UserWarning",
    )


@pytest.fixture()
def gmsh_available():
    """函数级 gmsh 守卫: 无 gmsh 环境跳过当前测试 (模块级用
    GMSH_AVAILABLE 常量 + skipif marker)."""
    if not GMSH_AVAILABLE:
        pytest.skip(GMSH_UNAVAILABLE_REASON)


def mesh_result_from_geo(geo_str):
    """写临时 .geo → generate_from_geo, 返回完整结果对象 (含 .regions)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.geo',
                                     delete=False) as f:
        f.write(geo_str)
        geo = f.name
    try:
        from fem2d.gmsh_adapter import generate_from_geo
        return generate_from_geo(geo)
    finally:
        if os.path.exists(geo):
            try:
                os.unlink(geo)
            except OSError:
                pass


def mesh_from_geo(geo_str):
    """写临时 .geo → generate_from_geo → (nodes, elements, elem_type)."""
    r = mesh_result_from_geo(geo_str)
    return r.nodes, r.elements, r.elem_type


def _mesh():
    """2×2 单位方板 Q4 (CPS4) Mesh — 测试脚手架通用入口."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2, 3]], dtype=int)
    from fem2d.mesh import Mesh
    return Mesh(nodes=nodes, elements=elems, elem_type="CPS4")


@pytest.fixture()
def square_mesh():
    """2×2 单位方板 Q4 (CPS4), 节点 CCW — 与 _mesh() 同构."""
    return _mesh()


@pytest.fixture()
def quad_mesh():
    """2×2 单位方板 Q4 (CPS4), 节点 CCW — _quad 系列 helper 的 fixture 版."""
    return _mesh()
