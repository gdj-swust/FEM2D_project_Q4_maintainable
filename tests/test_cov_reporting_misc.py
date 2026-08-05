"""覆盖轮 C1 — reporting/verification/patch_test/preprocess/input_source/visualize.

报告层用构造的 result 字典直接喂; 验证与 patch test 走 __main__ 子进程;
输入源用临时文件 + monkeypatch 触发 IO 防御分支.
"""
import io
import contextlib
import subprocess
import sys

import numpy as np
import pytest

from fem2d import Mesh
from fem2d.config import AnalysisConfig
from fem2d.errors import CliError, GeoScriptRejected
from fem2d.reporting import build_warnings, print_result_summary


def _mesh():
    return Mesh(nodes=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                elements=np.array([[0, 1, 2], [0, 3, 2]]),
                E=1e6, nu=0.3, thickness=1.0)


# ── reporting.py ────────────────────────────────────────────────────────────

def _summary_result(cond_info):
    return {
        "u": np.zeros(8), "vm_stress": np.ones(2),
        "stress": np.ones((2, 3)), "residual": 1e-12,
        "condition_info": cond_info,
    }


_Z2 = {"total_error": 0.1, "energy_norm": 1.0, "eta": 5.0,
       "worst_elem": 0, "elem_contrib": np.array([50.0, 50.0]),
       "stress_jumps": {"avg_jump": 0.1, "max_jump": 0.2}}
_Q = {"grade": "A", "area_min": 1e-4, "area_max": 1e-4, "area_mean": 1e-4,
      "area_cv": 0.0, "ratio_max": 1.0, "ratio_ok": 2, "ratio_warn": 0,
      "ratio_bad": 0, "angle_min": 45.0, "angle_max": 90.0,
      "angle_ok": 2, "angle_warn": 0, "angle_bad": 0, "jacobian_neg": 0}


def _print_summary(result, config=None):
    mesh = _mesh()
    mesh.build_connectivity()   # centroids 供 vm 位置打印
    with contextlib.redirect_stdout(io.StringIO()):
        print_result_summary(
            config or AnalysisConfig(), mesh, result, _Z2, _Q, 100.0,
            False, 10, 10)


def test_summary_condition_number_present():
    """cond_info 含 condition_number → 打印精度损失."""
    _print_summary(_summary_result(
        {"condition_number": 1e10, "digits_lost": 8.0}))


def test_summary_condition_singular_status():
    _print_summary(_summary_result({"status": "SINGULAR?", "error": "eig"}))


def test_summary_condition_other_status():
    _print_summary(_summary_result({"status": "SKIP"}))


def test_summary_warnings_block():
    """build_warnings 非空 → [WARN] 列表逐条打印."""
    _print_summary(_summary_result(None))


def test_build_warnings_volumetric_locking():
    """平面应变 + ν>0.40 → 体积自锁警告 (Bathe 4.4.4)."""
    cfg = AnalysisConfig(plane="strain", nu=0.45)
    mesh = _mesh()
    warnings = build_warnings(cfg, mesh, _Z2, False, 10, 10)
    assert any("体积自锁" in w for w in warnings)


def test_build_warnings_bending_stiffness():
    """CST 弯曲层数 <6 → 弯曲刚度警告 (Bathe 5.3.3)."""
    cfg = AnalysisConfig()
    mesh = _mesh()
    warnings = build_warnings(cfg, mesh, _Z2, True, 4, 4)
    assert any("弯曲刚度偏大" in w for w in warnings)


def test_build_warnings_eta_high():
    """Z2 eta > 15% → 应力集中未收敛警告."""
    z2 = dict(_Z2, eta=20.0)
    cfg = AnalysisConfig()
    mesh = _mesh()
    warnings = build_warnings(cfg, mesh, z2, False, 10, 10)
    assert any("eta" in w and "15%" in w for w in warnings)


# ── verification.py / patch_test.py / convergence.py __main__ ───────────────

def test_verification_main_exit_zero():
    """python -m fem2d.verification → 全部 PASS → 退出码 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "fem2d.verification"],
        capture_output=True, text=True, cwd=".", timeout=300)
    assert proc.returncode == 0, proc.stdout[-500:]


def test_patch_test_main_exit_zero():
    """python -m fem2d.patch_test → 三种单元全过 → 退出码 0."""
    proc = subprocess.run(
        [sys.executable, "-m", "fem2d.patch_test"],
        capture_output=True, text=True, cwd=".", timeout=600)
    assert proc.returncode == 0, proc.stdout[-800:]


def test_convergence_insufficient_levels(capsys):
    """少于 3 个细化层 → 提示至少 3 层 (源码英文文案)."""
    import fem2d.convergence as CV
    CV.run_cantilever_convergence(refinements=2, verbose=True, elem_type="CPS3")
    assert "at least 3 refinement levels" in capsys.readouterr().out


def test_convergence_fitted_rates(capsys):
    """≥3 层 → 渐近收敛率段打印 (PASS 分支, CPS3 理论值区间内)."""
    import fem2d.convergence as CV
    CV.run_cantilever_convergence(refinements=3, verbose=True, elem_type="CPS3")
    assert "Asymptotic convergence rates" in capsys.readouterr().out


def test_lazy_run_cantilever_convergence():
    """__getattr__ 惰性导入: F.run_cantilever_convergence."""
    import fem2d as F
    assert callable(F.run_cantilever_convergence)


# ── runner.py 主流程错误路径 ────────────────────────────────────────────────

def test_runner_keyboard_interrupt_returns_130(monkeypatch):
    import fem2d.runner as R
    def ctrl_c(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(R, "_resolve_input", ctrl_c)
    assert R.main(["x.geo"]) == 130


def test_runner_generic_error_returns_2(monkeypatch, capsys):
    import fem2d.runner as R
    def boom(*a, **k):
        raise ValueError("boom")
    monkeypatch.setattr(R, "_resolve_input", boom)
    assert R.main(["x.geo"]) == 2
    assert "boom" in capsys.readouterr().out


def test_runner_debug_reraise(monkeypatch):
    """--debug → 输入解析阶段异常 reraise (config 内 debug 字段驱动)."""
    import fem2d.runner as R
    def boom(*a, **k):
        raise ValueError("boom")
    monkeypatch.setattr(R, "_resolve_input", boom)
    monkeypatch.setattr(
        R, "_parse_cli_config",
        lambda argv: (AnalysisConfig(debug=True), None))
    with pytest.raises(ValueError):
        R.main(["x.geo"])


def test_runner_sys_path_bootstrap(monkeypatch):
    """项目根不在 sys.path 时 main 兜底注入."""
    import os
    import fem2d.runner as R
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    monkeypatch.setattr(sys, "path",
                        [p for p in sys.path if p != root])
    # 走到 parse_args 即注入完成 (路径参数无效但注入发生在错误之前)
    with contextlib.redirect_stdout(io.StringIO()):
        R.main(["definitely_missing.geo"])
    assert root in sys.path


# ── cli.py / config.py ─────────────────────────────────────────────────────

def test_cli_mixed_element_type_rejected():
    from fem2d.cli import _resolve_plane_type
    with pytest.raises(ValueError, match="mixed element"):
        _resolve_plane_type("CPS3,CPS4")


def test_cli_ask_eof_returns_empty(monkeypatch, capsys):
    from fem2d.cli import ask
    def eof(_):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    assert ask("prompt: ") == ""
    assert "标准输入已关闭" in capsys.readouterr().out


def test_config_band_nonfinite_rejected():
    """band 参数含 NaN → __post_init__ 即抛 (须全部有限)."""
    with pytest.raises(ValueError, match="必须有限"):
        AnalysisConfig(band_min=np.nan, band_max=1.0, band_step=0.1)


# ── preprocess.py ──────────────────────────────────────────────────────────

def test_read_geo_groups_range_expansion(monkeypatch, tmp_path):
    """Physical Curve 花括号内 300:319 范围写法 → 展开为连续 ID."""
    import fem2d.preprocess as PP
    geo = tmp_path / "g.geo"
    geo.write_text(
        'Physical Curve("c1", 1) = {1, 2, 300:319};', encoding="utf-8")
    monkeypatch.setattr(PP, "read_geo_curve_groups", lambda *a, **k: {})
    groups = PP.read_geo_groups(str(geo))
    assert groups["c1"] == [1, 2] + list(range(300, 320))


def test_parse_spec_config_blank_lines(tmp_path):
    """空行跳过 + 键值按字符串保留 (类型转换在 config 层)."""
    import fem2d.preprocess as PP
    spec = tmp_path / "s.spec"
    spec.write_text("E = 1e6\n\n\nlc = 0.5\n", encoding="utf-8")
    cfg = PP.parse_spec_config(str(spec))
    assert cfg["E"] == "1e6" and cfg["lc"] == "0.5"


def test_parse_geo_fem_config_missing_file(tmp_path):
    import fem2d.preprocess as PP
    assert PP.parse_geo_fem_config(str(tmp_path / "nope.geo")) == {
        "fix": [], "traction": [], "pressure": [], "body": None}


def test_parse_geo_fem_config_traction_profile(tmp_path):
    import fem2d.preprocess as PP
    geo = tmp_path / "g.geo"
    geo.write_text("// @FEM:traction=right,1e6,0,p\n"
                   "// @FEM:body=0,-78000\n", encoding="utf-8")
    cfg = PP.parse_geo_fem_config(str(geo))
    assert cfg["traction"] == ["right:1e6,0:p"]
    assert cfg["body"] == "0,-78000"


def test_merge_geo_fem_config_auto_apply(tmp_path, capsys):
    import fem2d.preprocess as PP
    geo_fem = {"fix": ["left"], "traction": ["right:1e6,0"],
               "pressure": [], "body": "0,-1"}
    cfg = AnalysisConfig()
    PP.merge_geo_fem_config(geo_fem, cfg, verbose=True)
    assert cfg.traction == "right:1e6,0"
    assert cfg.body == "0,-1"
    assert "auto" in capsys.readouterr().out


def test_merge_geo_fem_config_cli_priority_warns(capsys):
    """CLI 已显式 --body → .geo 体力不覆盖, 仅告警."""
    import fem2d.preprocess as PP
    geo_fem = {"fix": [], "traction": [], "pressure": [], "body": "0,-1"}
    cfg = AnalysisConfig(body="0,-2")
    PP.merge_geo_fem_config(geo_fem, cfg, verbose=True)
    assert cfg.body == "0,-2"
    assert "WARN" in capsys.readouterr().out


def test_validate_mesh_nonfinite_elements():
    """单元索引含 NaN → MeshValidationError (定义于 preprocess 而非 errors)."""
    from fem2d.preprocess import MeshValidationError, validate_mesh
    with pytest.raises(MeshValidationError):
        validate_mesh(np.zeros((4, 2)), np.array([[0., 1., np.nan]]))


# ── input_source.py ────────────────────────────────────────────────────────

def test_ensure_artifact_dir_unwritable(monkeypatch, tmp_path):
    """探测文件写入失败 → CliError (OSError 分支)."""
    import fem2d.input_source as IS
    def raising_open(*a, **k):
        raise PermissionError("denied")
    monkeypatch.setattr("builtins.open", raising_open)
    with pytest.raises(CliError, match="输出目录不可写"):
        IS.ensure_artifact_dir_writable(str(tmp_path / "out"))


def test_generate_geo_import_failure_unlink_error(monkeypatch, tmp_path):
    """import_msh 抛异常 + 临时文件清理失败 (unlink OSError) → 原异常冒出."""
    import fem2d.input_source as IS
    geo = tmp_path / "g.geo"
    geo.write_text("Point(1) = {0,0,0,1};\n", encoding="utf-8")

    class _Dummy:
        def run_gmsh(self, geo_path, quad=False, output_path=None,
                     defer_publish=True):
            tmp = tmp_path / "generated.msh"
            tmp.write_text("tmp", encoding="utf-8")
            return str(tmp)
    monkeypatch.setattr(IS, "_import_scripts", lambda name: _Dummy())
    import fem2d.gmsh_adapter as GA
    def boom(*a, **k):
        raise RuntimeError("topology fail")
    monkeypatch.setattr(GA, "import_msh", boom)
    monkeypatch.setattr(IS.os, "unlink", lambda p: (_ for _ in ()).throw(OSError("locked")))
    with pytest.raises(RuntimeError, match="topology fail"):
        IS.generate_geo_with_topology(str(geo))


def test_physical_point_from_geo_rejected_script(tmp_path):
    """源 .geo 含危险指令 → GeoScriptRejected 响亮冒出.

    无 gmsh 时 physical_point_from_geo 走 "API 不可用" 提前返回路径,
    不会触发脚本清洗 — 该断言只在 API 可用时成立, 无 gmsh 必须 skip.
    """
    try:
        import gmsh  # noqa: F401 — 仅探测 API 可用性
    except ImportError:
        pytest.skip("无 gmsh 环境 — physical_point_from_geo 走 API 不可用路径")
    import fem2d.input_source as IS
    mesh = _mesh()
    geo = tmp_path / "evil.geo"
    geo.write_text('Point(1) = {0, 0, 0, 1.0};\nSystemCall "format c:";\n',
                   encoding="utf-8")
    with pytest.raises(GeoScriptRejected):
        IS.physical_point_from_geo(str(geo), "p1", mesh)


def test_resolve_spec_overrides_empty_output_dir(tmp_path):
    """spec output_dir 为空值 → ValueError."""
    import fem2d.input_source as IS
    spec = tmp_path / "s.spec"
    spec.write_text("output_dir:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output_dir"):
        IS.resolve_spec_overrides(str(spec), AnalysisConfig())


def test_resolve_geo_lc_tempfile_failure(monkeypatch, tmp_path):
    """lc 临时副本创建失败 (只读源目录) → 针对性 CliError."""
    import fem2d.input_source as IS
    geo = tmp_path / "g.geo"
    geo.write_text("lc = 0.5;\nPoint(1) = {0,0,0,lc};\n", encoding="utf-8")
    cfg = AnalysisConfig(lc=0.1)
    def boom(*a, **k):
        raise OSError("read-only dir")
    monkeypatch.setattr(IS.tempfile, "NamedTemporaryFile", boom)
    with pytest.raises(CliError, match="lc 临时副本"):
        IS._resolve_geo_lc(str(geo), cfg, ask=None, temp_dir=None)


# ── visualize.py ───────────────────────────────────────────────────────────

def test_style_colorbar_title(capsys):
    """色条标题非 None → set_label; 量级分支同时触发 set_ticks/ticklabels."""
    import matplotlib
    matplotlib.use("Agg")
    from fem2d.visualize import _style_colorbar
    calls = []
    class _Cbar:
        def set_label(self, t):
            calls.append(("label", t))
        def set_ticks(self, t):
            calls.append(("ticks", len(t)))
        def set_ticklabels(self, t):
            calls.append(("ticklabels", len(t)))
    _style_colorbar(_Cbar(), 0.0, 1.0, title="σ [Pa]")
    assert ("label", "σ [Pa]") in calls
    # 中等量级 → 8 个刻度全部设置 (判据: 刻度格式化分支真实执行)
    assert ("ticks", 8) in calls and ("ticklabels", 8) in calls


def test_plot_three_show_non_agg(monkeypatch):
    """非 Agg 后端 → show(block=False) 分支."""
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    from fem2d.visualize import plot_three
    from fem2d import solve
    mesh = _mesh()
    for n in (0, 1):
        mesh.fix_node(n, "both")
    mesh.add_traction(2, 3, 1e3, 0.0)
    result = solve(mesh, verbose=False)
    monkeypatch.setattr(plt, "get_backend", lambda: "TkAgg")
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    plot_three(mesh, result, scale=1.0)
