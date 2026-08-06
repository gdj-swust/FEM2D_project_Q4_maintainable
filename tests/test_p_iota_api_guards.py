# -*- coding: utf-8 -*-
"""P-ι 公共 API 入口校验 — resolve_input_file 类型守卫 + plot_three tag 白名单.

判别性:
  - resolve_input_file(123/None/b"bytes", config) → 带上下文的 TypeError
    (修复前: 裸 TypeError "expected str, bytes or os.PathLike..."; bytes 是
    os.PathLike 虚拟子类, 能通过 splitext, 曾静默落入 "不支持的输入" CliError);
  - plot_three(mesh, result, tag="bad") → ValueError 含合法 tag 列表
    (修复前: 裸 KeyError "bad" 从 g_vals[tag] 冒出);
  - 合法 str / os.PathLike 路径与全部 12 个合法 tag 行为不变.

不依赖 gmsh: 本文件只构造 2 单元三角网格 + 手工 result dict.
"""
import matplotlib
matplotlib.use("Agg")   # 必须先于 pyplot/visualize 导入 (无显示器环境)

import pathlib

import numpy as np
import pytest

from fem2d.config import AnalysisConfig
from fem2d.errors import CliError
from fem2d.input_source import resolve_input_file
from fem2d.mesh import Mesh
from fem2d.visualize import PLOTS, plot_three

# 合法 tag 集直接派生自 PLOTS (单一事实源) — 12 个
VALID_TAGS = sorted({v[0] for v in PLOTS.values()})


def _two_tri():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
        elements=np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        elem_type="CPS3")


def _result():
    return {"u": np.zeros(8), "stress": np.zeros((2, 3)),
            "vm_stress": np.zeros(2), "stress_qp": None}


# ═══════════════════════════════════════════════════════════════
# 任务 1: resolve_input_file 入口类型守卫
# ═══════════════════════════════════════════════════════════════

def test_resolve_input_file_rejects_int():
    """int 路径 → TypeError 带参数名 fp 与类型名 int (修复前裸 TypeError)."""
    with pytest.raises(TypeError, match="fp") as ei:
        resolve_input_file(123, AnalysisConfig())
    assert "int" in str(ei.value)


def test_resolve_input_file_rejects_none():
    """None 路径 → 同款 TypeError (修复前裸 '...not NoneType')."""
    with pytest.raises(TypeError, match="fp"):
        resolve_input_file(None, AnalysisConfig())


def test_resolve_input_file_rejects_bytes():
    """bytes 路径 → TypeError: bytes 非 str/os.PathLike 语义上的路径.

    Python 中 bytes 是 os.PathLike 的虚拟子类, os.path.splitext 能处理 —
    曾静默落入 "不支持的输入" CliError, 而非类型错误. 守卫必须显式排除.
    """
    with pytest.raises(TypeError, match="fp"):
        resolve_input_file(b"bytes", AnalysisConfig())


def test_resolve_input_file_legal_str_passes_guard():
    """合法 str 路径不被守卫误伤 — .inp 分支 CliError (解析层) 而非 TypeError."""
    with pytest.raises(CliError):
        resolve_input_file("whatever.inp", AnalysisConfig())


def test_resolve_input_file_pathlike_passes_guard():
    """os.PathLike (pathlib.Path) 放行到解析层 — CliError 而非 TypeError."""
    with pytest.raises(CliError):
        resolve_input_file(pathlib.Path("whatever.inp"), AnalysisConfig())


# ═══════════════════════════════════════════════════════════════
# 任务 2: plot_three tag 白名单校验
# ═══════════════════════════════════════════════════════════════

def test_plot_three_rejects_unknown_tag():
    """非法 tag → ValueError 且消息列出全部合法 tag (修复前裸 KeyError 'bad')."""
    mesh, result = _two_tri(), _result()
    with pytest.raises(ValueError, match="tag") as ei:
        plot_three(mesh, result, tag="bad")
    msg = str(ei.value)
    for k in VALID_TAGS:
        assert k in msg, f"错误消息未列出合法 tag {k}: {msg}"


def test_plot_three_rejects_non_str_tag():
    """非字符串 tag (None/int) → 同款 ValueError (修复前裸 KeyError)."""
    mesh, result = _two_tri(), _result()
    for bad in (None, 5):
        with pytest.raises(ValueError, match="tag"):
            plot_three(mesh, result, tag=bad)


def test_plot_three_all_valid_tags_work(tmp_path):
    """全部合法 tag 照常出图 — 守卫不误伤任何分量 (保存路径, 无异常)."""
    mesh, result = _two_tri(), _result()
    for tag in VALID_TAGS:
        plot_three(mesh, result, tag=tag, scale=1.0,
                   save=str(tmp_path / f"tag_{tag}.png"))
