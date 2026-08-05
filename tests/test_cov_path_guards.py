"""覆盖轮 C2 — open(int) 缺陷修复的判别性测试.

缺陷申报: 各解析函数对非 str 路径直接 open() — Python 的 open(int)
把 int 当已存在 fd 打开 (fd 1 = stdout), 读取/关闭调用进程的 stdout,
静默破坏调用方输出 (fuzz 探测 resolve_spec_overrides(1, cfg) 时发现:
后续所有 print 抛 OSError Bad file descriptor).
修复: 6 处解析入口加 str/os.PathLike 类型校验 → TypeError 带诊断.
判别性: 修复前这些调用抛 OSError 或破坏 stdout, 测试红.
"""
import sys

import pytest

from fem2d.config import AnalysisConfig
from fem2d.gmsh_adapter import _file_declares_physical_names
from fem2d.input_source import (
    _geo_script_entry_hint, _resolve_geo_lc,
)
from fem2d.preprocess import (
    parse_geo_fem_config, parse_spec_config, read_geo_groups,
)


def _stdout_alive():
    """stdout fd 未被 open(int) 关闭."""
    try:
        return sys.stdout.fileno() >= 0
    except (OSError, ValueError):
        return False


def test_read_geo_groups_rejects_non_path():
    with pytest.raises(TypeError, match="str/os.PathLike"):
        read_geo_groups(123)
    assert _stdout_alive()


def test_parse_spec_config_rejects_non_path():
    for bad in (123, 0, None, 1.5):
        with pytest.raises(TypeError, match="str/os.PathLike"):
            parse_spec_config(bad)
    assert _stdout_alive()


def test_parse_geo_fem_config_rejects_non_path():
    with pytest.raises(TypeError, match="str/os.PathLike"):
        parse_geo_fem_config(123)
    assert _stdout_alive()


def test_geo_script_entry_hint_rejects_non_path():
    with pytest.raises(TypeError, match="str/os.PathLike"):
        _geo_script_entry_hint(123)
    assert _stdout_alive()


def test_resolve_geo_lc_rejects_non_path():
    with pytest.raises(TypeError, match="str/os.PathLike"):
        _resolve_geo_lc(123, AnalysisConfig(), ask=None)
    assert _stdout_alive()


def test_file_declares_physical_names_rejects_non_path():
    with pytest.raises(TypeError, match="str/os.PathLike"):
        _file_declares_physical_names(123)
    assert _stdout_alive()


def test_physical_point_from_geo_rejects_non_path():
    """非 str 路径 → TypeError 冒出 (修复前被宽 except 吞成
    gmsh_unavailable 静默元组, 误导排查方向)."""
    from fem2d.input_source import physical_point_from_geo
    from fem2d.mesh import Mesh
    import numpy as np
    mesh = Mesh(np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                np.array([[0, 1, 2], [0, 3, 2]]), E=1e6, nu=0.3)
    for bad in (123, 1.5, [1.0], complex(1, 2), True):
        with pytest.raises(TypeError, match="str/os.PathLike"):
            physical_point_from_geo(bad, "p1", mesh)
    # None → no_geo_source 既有契约 (不误伤)
    assert physical_point_from_geo(None, "p1", mesh)[3] == "no_geo_source"
    assert _stdout_alive()


def test_traction_jumps_rejects_bad_stress_shape():
    """elem_stress 非 (n,3) → ValueError (修复前裸 IndexError 冒
    stress[e1], 把用户引向错误的数组索引方向)."""
    from fem2d.error_est import compute_traction_jumps
    from fem2d.mesh import Mesh
    import numpy as np
    mesh = Mesh(np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                np.array([[0, 1, 2], [0, 3, 2]]), E=1e6, nu=0.3)
    mesh.build_connectivity()
    for bad in (1, 0.5, np.ones(3), np.ones((2, 2)), np.ones((2, 3, 1))):
        with pytest.raises(ValueError, match="n_elem, 3"):
            compute_traction_jumps(mesh, bad)
    # 合法 (2, 3) 数组正常返回 (不误伤)
    out = compute_traction_jumps(mesh, np.ones((2, 3)))
    assert isinstance(out, list)


def test_resolve_boundary_selection_rejects_non_str():
    """非 str 选择 → CliError (修复前 str() 化宽容, bool/float/容器
    静默无匹配返回 [])."""
    from fem2d.bc_apply import _resolve_boundary_selection
    from fem2d.errors import CliError
    for bad in (True, 1.5, complex(1, 2), [1.0], float("nan"), 5):
        with pytest.raises(CliError, match="边界选择"):
            _resolve_boundary_selection(bad, [], fatal=True)
    # None / 空串 → 空选择契约 (不误伤)
    assert _resolve_boundary_selection(None, [], fatal=True) == []
    assert _resolve_boundary_selection("", [], fatal=True) == []
