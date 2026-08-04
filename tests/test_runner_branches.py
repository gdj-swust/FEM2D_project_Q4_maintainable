"""runner.py 防御分支补测 (覆盖率 65% → ≥80%) — 包 2 覆盖率任务.

未覆盖行集中的路径: 独立自检 (--self-test 无网格)、边界列表、绘图分支
(isoband 固定带宽 / EOFError / 批处理判定)、区域注册表报告、输入选择
交互分支、main 顶层错误退出码。

判别性: 每条测试断言具体退出码/输出文本/异常消息/调用参数, 无恒真断言。
"""
import sys
import types

import numpy as np
import pytest

import fem2d.gmsh_adapter as gmsh_adapter
import fem2d.runner as runner_mod
import fem2d.visualize as visualize_mod
from fem2d.config import AnalysisConfig
from fem2d.errors import CliError
from fem2d.mesh import Mesh
from fem2d.runner import main
from fem2d.boundary import BoundaryDiagnostics


def _fake_msh_import(elem_type="CPS4", elements=None, nodes=None):
    """与 test_exit_code_matrix 同款 fake import 结果 — 无 Gmsh 依赖."""
    nodes = np.array(
        [[0., 0.], [1., 0.], [1., 1.], [0., 1.]]
        if nodes is None else nodes, dtype=float)
    elements = np.array(
        [[0, 1, 2, 3]] if elements is None else elements, dtype=np.int64)

    class _FakeImport:
        pass

    fake = _FakeImport()
    fake.nodes = nodes
    fake.elements = elements
    fake.elem_type = elem_type
    fake.node_tag_to_index = {int(i): int(i) for i in range(len(nodes))}
    fake.element_tag_to_index = {0: 0}
    fake.regions = None
    return fake


@pytest.fixture
def msh_file(tmp_path):
    """内容不重要的 .msh 头文件 — import_msh 被 monkeypatch 接管."""
    path = tmp_path / "fake.msh"
    path.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n",
                    encoding="utf-8")
    return str(path)


@pytest.fixture
def fake_msh(monkeypatch, msh_file):
    """把 .msh 输入接到 fake 导入结果上."""
    def _install(import_result):
        monkeypatch.setattr(
            gmsh_adapter, "import_msh",
            lambda fp, plane_type="stress": import_result)
        return import_result
    return _install


_FAKE_RESOLVED = types.SimpleNamespace(fp="fake.msh")


def _square_mesh():
    return Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=int),
        elem_type="CPS4")


# ═══════════════════════════════════════════════════════════════
# 独立自检 (--self-test 无网格)
# ═══════════════════════════════════════════════════════════════

def test_standalone_self_test_returns_zero(capsys):
    """--self-test (无网格) 全单元 patch test + 材料验证全过 → 0."""
    assert main(["--self-test"]) == 0
    out = capsys.readouterr().out
    assert "CST + Q4 + Q4R + Q4I Patch Tests" in out
    assert "PASS" in out


def test_standalone_self_test_failure_returns_one(monkeypatch):
    """--self-test 任一单元 patch test 失败 → 1."""
    monkeypatch.setattr(
        runner_mod, "run_patch_test",
        lambda *a, **k: {"all_passed": False})
    monkeypatch.setattr(
        runner_mod, "run_plane_verification", lambda *a, **k: (0, 0))
    assert main(["--self-test"]) == 1


def test_standalone_self_test_warns_on_inert_bc_args(monkeypatch, capsys):
    """--self-test 无网格时 --fix/--body 不生效 → WARN 且退出 0.

    曾静默吞掉非法 BC 参数 (载荷从未生效也无提示), 修复后必须响亮警告.
    """
    monkeypatch.setattr(
        runner_mod, "run_patch_test",
        lambda *a, **k: {"all_passed": True})
    monkeypatch.setattr(
        runner_mod, "run_plane_verification", lambda *a, **k: (0, 0))
    assert main(["--self-test", "--fix", "left", "--body", "0,-78000"]) == 0
    out = capsys.readouterr().out
    assert "[WARN] --fix 在独立自检模式下不生效" in out
    assert "[WARN] --body 在独立自检模式下不生效" in out


def test_standalone_self_test_list_boundaries_combo_warns(monkeypatch,
                                                          capsys):
    """--self-test + --list-boundaries: 自检照常执行 + 组合 WARN."""
    monkeypatch.setattr(
        runner_mod, "run_patch_test",
        lambda *a, **k: {"all_passed": True})
    monkeypatch.setattr(
        runner_mod, "run_plane_verification", lambda *a, **k: (0, 0))
    assert main(["--self-test", "--list-boundaries"]) == 0
    out = capsys.readouterr().out
    assert ("[WARN] --self-test 与 --list-boundaries 组合" in out
            and "Patch Tests" in out)


# ═══════════════════════════════════════════════════════════════
# _import_mesh / _build_mesh 防御分支
# ═══════════════════════════════════════════════════════════════

def test_import_mesh_missing_gmsh_import_raises():
    """gmsh_import 为 None (内部状态错误) → RuntimeError 带上下文."""
    resolved = types.SimpleNamespace(gmsh_import=None)
    with pytest.raises(RuntimeError, match="内部状态错误"):
        runner_mod._import_mesh(resolved)


def test_build_mesh_validation_failure_raises(monkeypatch, capsys):
    """网格校验失败 → 打印全部错误 + RuntimeError."""
    monkeypatch.setattr(
        runner_mod, "validate_mesh",
        lambda *a, **k: {"ok": False, "errors": ["e1", "e2"],
                         "warnings": ["w0"]})
    config = AnalysisConfig()
    with pytest.raises(RuntimeError, match="Mesh validation failed"):
        runner_mod._build_mesh(
            config, None,
            np.array([[0., 0.], [1., 0.]]),
            np.array([[0, 1]], dtype=int),
            "CPS4", None)
    out = capsys.readouterr().out
    assert "[WARN] w0" in out and "[ERROR] e1" in out and "[ERROR] e2" in out


def test_build_mesh_validation_warnings_only(monkeypatch, capsys):
    """网格校验仅警告 → 继续构造, plane 自动判型 stress."""
    monkeypatch.setattr(
        runner_mod, "validate_mesh",
        lambda *a, **k: {"ok": True, "errors": [], "warnings": ["w1"]})
    config = AnalysisConfig()
    mesh = runner_mod._build_mesh(
        config, _FAKE_RESOLVED,
        np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        np.array([[0, 1, 2, 3]], dtype=int),
        "CPS4", None)
    assert "[WARN] w1" in capsys.readouterr().out
    assert mesh.elem_type == "CPS4"
    assert config.plane == "stress"


def test_build_mesh_cpe_auto_strain_print(monkeypatch, capsys):
    """CPE 网格无 --plane → 自动 plane=strain 并打印 [auto]."""
    monkeypatch.setattr(
        runner_mod, "validate_mesh",
        lambda *a, **k: {"ok": True, "errors": [], "warnings": []})
    config = AnalysisConfig()
    mesh = runner_mod._build_mesh(
        config, _FAKE_RESOLVED,
        np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        np.array([[0, 1, 2, 3]], dtype=int),
        "CPE4", None)
    assert config.plane == "strain"
    assert "[auto] CPE4 网格 → 默认 plane=strain" in capsys.readouterr().out
    assert mesh.plane_type == "strain"


def test_build_mesh_elem_type_override_preserves_requested_plane(
        monkeypatch, capsys):
    """--elem-type 覆写内核 + 显式 --plane → 覆写后 plane 保持显式值."""
    monkeypatch.setattr(
        runner_mod, "validate_mesh",
        lambda *a, **k: {"ok": True, "errors": [], "warnings": []})
    config = AnalysisConfig(elem_type="Q4", plane="strain")
    mesh = runner_mod._build_mesh(
        config, _FAKE_RESOLVED,
        np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        np.array([[0, 1, 2, 3]], dtype=int),
        "CPS4", None)
    assert config.plane == "strain"      # CLI 显式优先
    assert mesh.elem_type == "Q4"
    assert "[elem] override: CPS4 → Q4" in capsys.readouterr().out


def test_build_mesh_elem_type_incompatible_raises_clierror(monkeypatch):
    """--elem-type 与网格节点数不兼容 → CliError(1) (矩阵测试已覆盖退出码)."""
    monkeypatch.setattr(
        runner_mod, "validate_mesh",
        lambda *a, **k: {"ok": True, "errors": [], "warnings": []})
    config = AnalysisConfig(elem_type="Q4")   # 需要 4 节点, 网格 3 节点
    with pytest.raises(CliError) as exc:
        runner_mod._build_mesh(
            config, None,
            np.array([[0., 0.], [1., 0.], [1., 1.]]),
            np.array([[0, 1, 2]], dtype=int),
            "CPS3", None)
    assert exc.value.exit_code == 1
    assert "单元类型与网格拓扑不兼容" in str(exc.value)


def test_build_mesh_region_registry_reports_areas(monkeypatch, capsys):
    """region 注册表: validate + 面积报告逐面打印."""
    from fem2d.regions import RegionRegistry, SurfaceRegion
    monkeypatch.setattr(
        runner_mod, "validate_mesh",
        lambda *a, **k: {"ok": True, "errors": [], "warnings": []})
    registry = RegionRegistry()
    registry.surfaces.append(SurfaceRegion(
        name="top", physical_tag=1, entity_tags=(1,),
        entity_types=("Surface",), element_ids=(0,)))
    config = AnalysisConfig()
    mesh = runner_mod._build_mesh(
        config, _FAKE_RESOLVED,
        np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        np.array([[0, 1, 2, 3]], dtype=int),
        "CPS4", registry)
    out = capsys.readouterr().out
    assert "[region] Surface 'top': 1 elements, area=" in out
    assert mesh.elem_type == "CPS4"


# ═══════════════════════════════════════════════════════════════
# _build_boundary 语义分支
# ═══════════════════════════════════════════════════════════════

def _real_segs_and_diag():
    mesh = _square_mesh()
    config = AnalysisConfig()
    diag = BoundaryDiagnostics()
    segs = runner_mod.build_boundary_segments(
        mesh, registry=None, edge_labels=None, geo_path=None,
        diagnostics=diag, strict=config.strict_boundary)
    return mesh, config, segs, diag


def _patch_segs(monkeypatch, mesh, config, segs, diag):
    """让 _build_boundary 使用我们构造好的 segs, 并把 diag 状态灌入其
    自建的 diagnostics (检测重跑会丢掉我们注入的 issue/映射)."""
    def _install(*a, registry=None, edge_labels=None, geo_path=None,
                 diagnostics=None, strict=False):
        if diagnostics is not None and diag is not None:
            diagnostics.issues = list(diag.issues)
            diagnostics.declared_physical_names = set(
                diag.declared_physical_names)
            diagnostics.mapped_physical_names = set(
                diag.mapped_physical_names)
        return segs
    monkeypatch.setattr(runner_mod, "build_boundary_segments", _install)
    return segs


def test_build_boundary_issues_printed(monkeypatch, capsys):
    """诊断 issue 按严重度打印 [ERROR]/[WARN] 前缀."""
    mesh, config, segs, diag = _real_segs_and_diag()
    diag.add("test_warn", "warning", "软问题")
    diag.add("test_err", "error", "硬问题")
    _patch_segs(monkeypatch, mesh, config, segs, diag)
    runner_mod._build_boundary(mesh, config, None, None, None)
    out = capsys.readouterr().out
    assert "[WARN] boundary/test_warn: 软问题" in out
    assert "[ERROR] boundary/test_err: 硬问题" in out


def test_build_boundary_semantics_mapped_report(monkeypatch, capsys):
    """Physical Curve 语义恢复 → 映射统计打印 (数量与边覆盖)."""
    mesh, config, segs, diag = _real_segs_and_diag()
    report = {
        "physical_names": ["left", "right"],
        "mapped_physical_names": ["left", "right"],
        "declared_physical_names": ["left", "right"],
        "covered_edges": 4, "total_boundary_edges": 4,
    }
    monkeypatch.setattr(runner_mod, "semantic_coverage",
                        lambda *a, **k: report)
    runner_mod._build_boundary(mesh, config, None, None, None)
    assert ("[boundary semantics] 2/2 Physical Curves mapped, "
            "4/4 boundary edges mapped") in capsys.readouterr().out


def test_build_boundary_no_semantics_warns_with_geo_source(monkeypatch,
                                                           capsys):
    """无 Physical Curve 但存在 .geo 源 → WARN 不致命."""
    mesh, config, segs, diag = _real_segs_and_diag()
    monkeypatch.setattr(
        runner_mod, "semantic_coverage",
        lambda *a, **k: {"physical_names": [],
                         "mapped_physical_names": [],
                         "declared_physical_names": [],
                         "covered_edges": 0, "total_boundary_edges": 4})
    runner_mod._build_boundary(mesh, config, None, None, "some.geo")
    assert "[WARN] 未恢复任何 Physical Curve" in capsys.readouterr().out


def test_build_boundary_require_physical_groups_fatal(monkeypatch):
    """--require-physical-groups 但无可用语义 → CliError(1)."""
    mesh, config, segs, diag = _real_segs_and_diag()
    config.require_physical_groups = True
    monkeypatch.setattr(
        runner_mod, "semantic_coverage",
        lambda *a, **k: {"physical_names": [],
                         "mapped_physical_names": [],
                         "declared_physical_names": [],
                         "covered_edges": 0, "total_boundary_edges": 4})
    with pytest.raises(CliError) as exc:
        runner_mod._build_boundary(mesh, config, None, None, None)
    assert exc.value.exit_code == 1
    assert "--require-physical-groups" in str(exc.value)
    assert "没有可用的 Physical Curve 语义" in str(exc.value)


def test_build_boundary_require_physical_groups_dropped_fatal(monkeypatch):
    """声明但未映射的 Physical Curve → CliError 列出掉落名."""
    mesh, config, segs, diag = _real_segs_and_diag()
    config.require_physical_groups = True
    diag.register_declared(["left", "right"])
    diag.register_mapped(["left"])
    _patch_segs(monkeypatch, mesh, config, segs, diag)
    report = {
        "physical_names": ["left", "right"],
        "mapped_physical_names": ["left"],
        "declared_physical_names": ["left", "right"],
        "covered_edges": 2, "total_boundary_edges": 4,
    }
    monkeypatch.setattr(runner_mod, "semantic_coverage",
                        lambda *a, **k: report)
    with pytest.raises(CliError) as exc:
        runner_mod._build_boundary(mesh, config, None, None, None)
    assert exc.value.exit_code == 1
    assert "未映射到外边界: right" in str(exc.value)


def test_build_boundary_require_physical_groups_errors_fatal(monkeypatch):
    """--require-physical-groups + 边界语义错误 → CliError 列出错误码."""
    mesh, config, segs, diag = _real_segs_and_diag()
    config.require_physical_groups = True
    diag.add("broken_edge", "error", "坏边")
    _patch_segs(monkeypatch, mesh, config, segs, diag)
    report = {
        "physical_names": ["left"],
        "mapped_physical_names": ["left"],
        "declared_physical_names": ["left"],
        "covered_edges": 4, "total_boundary_edges": 4,
    }
    monkeypatch.setattr(runner_mod, "semantic_coverage",
                        lambda *a, **k: report)
    with pytest.raises(CliError) as exc:
        runner_mod._build_boundary(mesh, config, None, None, None)
    assert exc.value.exit_code == 1
    assert "broken_edge" in str(exc.value)


# ═══════════════════════════════════════════════════════════════
# _print_boundaries (--list-boundaries)
# ═══════════════════════════════════════════════════════════════

def test_list_boundaries_end_to_end(fake_msh, msh_file, capsys):
    """--list-boundaries: 列出直线段 + 用法示例, 退出 0."""
    fake_msh(_fake_msh_import())
    assert main([msh_file, "--list-boundaries"]) == 0
    out = capsys.readouterr().out
    assert "--list-boundaries: 仅列出边界" in out
    assert "L=" in out                       # 直线段尺寸
    assert "python run.py fake.msh" in out  # 用法示例引用当前模型


def test_print_boundaries_arc_dimension_and_mesh_fallback(capsys):
    """弧段打印 R= 半径; 直线段打印 L= 长度; 无 mesh 时兜底 '模型文件'."""
    mesh = _square_mesh()
    segs = [
        {"nodes": [0, 1, 2], "type": "arc", "label": "圆角",
         "info": {"radius": 2.5}},
        {"nodes": [2, 3], "type": "line", "label": "底边", "info": None},
    ]
    runner_mod._print_boundaries(AnalysisConfig(), mesh, segs)
    out = capsys.readouterr().out
    assert "R=2.5000" in out and "L=1.0000" in out
    assert "圆角" in out and "底边" in out
    # config.mesh 为空 (交互选文件路径) → basename 兜底, 曾 TypeError
    assert "python run.py 模型文件" in out


# ═══════════════════════════════════════════════════════════════
# _analyze 误差方法分支
# ═══════════════════════════════════════════════════════════════

def _patch_analyze_deps(monkeypatch):
    calls = {}
    monkeypatch.setattr(runner_mod, "report_mesh_quality",
                        lambda mesh: "q")
    monkeypatch.setattr(runner_mod, "solve",
                        lambda mesh, **k: {"u": np.zeros(8)})
    def _est(mesh, result, method):
        calls["method"] = method
        return "z2"
    monkeypatch.setattr(runner_mod, "estimate_error", _est)
    return calls


def test_analyze_auto_weighted_for_large_mesh(monkeypatch, capsys):
    """大网格 (≥50000 单元) auto → weighted + 提示文案."""
    calls = _patch_analyze_deps(monkeypatch)
    mesh = types.SimpleNamespace(n_elements=50000)
    result, z2, q = runner_mod._analyze(mesh, AnalysisConfig())
    assert calls["method"] == "weighted"
    assert "[Error] 大网格自动使用 weighted 恢复" in capsys.readouterr().out
    assert z2 == "z2" and q == "q"


def test_analyze_error_method_spr_and_l2(monkeypatch):
    """显式 --error-method spr/l2 → 大写归一化传入估计器."""
    calls = _patch_analyze_deps(monkeypatch)
    mesh = types.SimpleNamespace(n_elements=4)
    runner_mod._analyze(mesh, AnalysisConfig(error_method="spr"))
    assert calls["method"] == "SPR"
    runner_mod._analyze(mesh, AnalysisConfig(error_method="l2"))
    assert calls["method"] == "L2"


# ═══════════════════════════════════════════════════════════════
# _plot 分支 (isoband / 批处理 / EOFError)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def plot_deps(monkeypatch):
    """绘图三方 (plot_three / interactive_plot / PLOTS) 全部接管."""
    calls = {"plot_three": {}, "interactive": 0}
    def _plot_three(mesh, result, **kw):
        calls["plot_three"] = kw
    def _interactive(*a, **kw):
        calls["interactive"] += 1
    monkeypatch.setattr(visualize_mod, "plot_three", _plot_three)
    monkeypatch.setattr(visualize_mod, "interactive_plot", _interactive)
    monkeypatch.setattr(
        visualize_mod, "PLOTS",
        {"1": ("sx", "σx"), "2": ("vm", "磨平")})
    return calls


def test_plot_fixed_isoband_levels_endpoint_alignment(plot_deps, capsys):
    """固定带宽: (max-min)/step 浮点商向下取整曾漏末带 → 四舍五入 + 尾点归位.

    0.3/0.1 = 2.9999999999999996 — floor 截断曾生成 [0, 0.1, 0.3] (末带宽
    0.2), 判别性: 断言生成的 levels 必须逐值等于 [0, 0.1, 0.2, 0.3]。
    """
    config = AnalysisConfig(
        no_plot=True, band_min=0.0, band_max=0.3, band_step=0.1)
    runner_mod._plot(config, "mesh", "result", 1.0)
    kw = plot_deps["plot_three"]
    levels = kw["isoband_levels"]
    assert levels is not None
    assert list(levels) == [0.0, 0.1, 0.2, 0.3]
    assert kw["isoband_tag"] == "vm"     # 默认 vm
    out = capsys.readouterr().out
    assert "[Isoband] fixed levels: 0 to 0.3, step=0.1 (3 bands)" in out
    assert "[Plot] 云图已生成" in out and "[ 1]" in out and "[ 2]" in out


def test_plot_band_tag_warns_non_vm(plot_deps, capsys):
    """--band-tag ≠ vm: 固定带宽只作用于切换后分量 → WARN 诚实化."""
    config = AnalysisConfig(
        no_plot=True, band_min=0.0, band_max=0.3, band_step=0.1,
        band_tag="sx")
    runner_mod._plot(config, "mesh", "result", 1.0)
    assert plot_deps["plot_three"]["isoband_tag"] == "sx"
    assert ("[WARN] --band-tag sx ≠ vm — 初始/保存图为 vm, 固定带宽不应用"
            in capsys.readouterr().out)


def test_plot_no_plot_is_batch_no_interactive(plot_deps):
    """--no-plot → 批处理判定, interactive_plot 不调用 (曾挂起)."""
    config = AnalysisConfig(no_plot=True)
    runner_mod._plot(config, "mesh", "result", 1.0)
    assert plot_deps["interactive"] == 0
    assert plot_deps["plot_three"]["isoband_levels"] is None


def test_plot_interactive_eof_graceful(plot_deps, monkeypatch, capsys):
    """终端 stdin 关闭 → EOFError 优雅提示, 不冒泡 (求解已成功)."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)  # 非批处理
    def _interactive(*a, **kw):
        raise EOFError
    monkeypatch.setattr(visualize_mod, "interactive_plot", _interactive)
    config = AnalysisConfig(no_plot=False)
    runner_mod._plot(config, "mesh", "result", 1.0)
    assert "[INFO] 非交互环境 (stdin 不可用), 跳过交互绘图" in \
        capsys.readouterr().out


def test_plot_save_forces_batch(plot_deps):
    """--save → 生成文件且不再交互 (曾批处理判定漏 save 挂起)."""
    config = AnalysisConfig(save="out.png", no_plot=False)
    runner_mod._plot(config, "mesh", "result", 1.0)
    assert plot_deps["plot_three"]["save"] == "out.png"
    assert plot_deps["interactive"] == 0


# ═══════════════════════════════════════════════════════════════
# _resolve_input 交互分支
# ═══════════════════════════════════════════════════════════════

def test_resolve_input_no_file_returns_none(monkeypatch, capsys):
    """无输入且非终端 → 提示 '需要指定输入文件', 返回 None."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(runner_mod, "ask", lambda prompt: "")
    assert runner_mod._resolve_input(AnalysisConfig()) is None
    assert "[ERROR] 需要指定输入文件" in capsys.readouterr().out


def test_resolve_input_file_missing_returns_none(monkeypatch, capsys):
    """交互指定不存在的文件 → '文件不存在' + None."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(runner_mod, "ask", lambda prompt: "nope.msh")
    assert runner_mod._resolve_input(AnalysisConfig()) is None
    assert "[ERROR] 文件不存在: nope.msh" in capsys.readouterr().out


def test_resolve_input_wizard_path(monkeypatch, tmp_path):
    """--wizard 显式 → 向导产出路径直接进入解析 (跳过提问)."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    f = tmp_path / "w.msh"
    f.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    sentinel = object()
    monkeypatch.setattr(runner_mod, "run_wizard",
                        lambda config: str(f))
    monkeypatch.setattr(runner_mod, "resolve_input_file",
                        lambda fp, config: sentinel)
    assert runner_mod._resolve_input(AnalysisConfig(wizard=True)) is sentinel


def test_apply_geo_fem_config_no_geo_noop():
    """无 .geo 源路径 → 直接返回 (不解析不合并)."""
    config = AnalysisConfig()
    assert runner_mod._apply_geo_fem_config(config, None) is None
    assert config.fix == ""    # 未被合并逻辑触碰


def test_apply_geo_fem_config_merges_from_geo(tmp_path):
    """.geo 含 @FEM: 注释 → 合并进 config (与 input_source 共用同一逻辑)."""
    geo = tmp_path / "m.geo"
    geo.write_text('# @FEM:fix=left\nlc = 0.5;\n', encoding="utf-8")
    config = AnalysisConfig()
    assert runner_mod._apply_geo_fem_config(config, str(geo)) is None
    assert config.fix == "left"


# ═══════════════════════════════════════════════════════════════
# main 顶层错误退出码
# ═══════════════════════════════════════════════════════════════

def test_main_config_value_error_returns_two(monkeypatch, capsys):
    """配置校验失败 (非法参数组合) → 退出 2 + [ERROR] 摘要."""
    import fem2d.config as config_mod
    def _boom(args):
        raise ValueError("非法参数组合")
    monkeypatch.setattr(config_mod.AnalysisConfig, "from_args",
                        staticmethod(_boom))
    assert main(["nonexistent.msh"]) == 2
    assert "[ERROR] 非法参数组合" in capsys.readouterr().out


def test_main_resolve_unknown_exception_returns_two(monkeypatch, tmp_path,
                                                    capsys):
    """输入解析阶段任意异常 (含 gmsh 缺失) → 友好摘要退出 2."""
    f = tmp_path / "exists.msh"
    f.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    def _boom(fp, config):
        raise RuntimeError("gmsh 内部失败")
    monkeypatch.setattr(runner_mod, "resolve_input_file", _boom)
    assert main([str(f)]) == 2
    assert "[ERROR] gmsh 内部失败" in capsys.readouterr().out


def test_main_resolve_debug_reraises(monkeypatch, tmp_path):
    """解析阶段异常 + --debug → 重新抛出 (完整 traceback)."""
    f = tmp_path / "exists.msh"
    f.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    def _boom(fp, config):
        raise RuntimeError("解析爆炸")
    monkeypatch.setattr(runner_mod, "resolve_input_file", _boom)
    with pytest.raises(RuntimeError, match="解析爆炸"):
        main([str(f), "--debug", "--no-plot"])


def test_main_resolve_clierror_preserves_exit_code(monkeypatch, tmp_path):
    """输入解析 CliError → 保留各自退出码 (原 sys.exit 语义)."""
    f = tmp_path / "exists.msh"
    f.write_text("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n", encoding="utf-8")
    def _boom(fp, config):
        raise CliError("[FATAL] 自定义失败", exit_code=3)
    monkeypatch.setattr(runner_mod, "resolve_input_file", _boom)
    assert main([str(f)]) == 3


def test_main_plot_exception_returns_two(fake_msh, msh_file, monkeypatch,
                                         capsys):
    """绘图阶段异常 → 顶层兜底退出 2 + [ERROR] 摘要."""
    fake_msh(_fake_msh_import())
    def _boom(*a, **k):
        raise RuntimeError("绘图失败")
    monkeypatch.setattr(runner_mod, "_plot", _boom)
    # --save 强制走绘图分支 (--no-plot 会跳过 _plot)
    assert main([msh_file, "--fix", "1", "--save", "out.png"]) == 2
    assert "[ERROR] 绘图失败" in capsys.readouterr().out


def test_main_debug_reraises_unknown_exception(fake_msh, msh_file,
                                               monkeypatch):
    """--debug → 顶层异常重新抛出完整 traceback (不吞)."""
    fake_msh(_fake_msh_import())
    def _boom(*a, **k):
        raise RuntimeError("内部爆炸")
    monkeypatch.setattr(runner_mod, "build_boundary_segments", _boom)
    with pytest.raises(RuntimeError, match="内部爆炸"):
        main([msh_file, "--debug", "--no-plot"])


def test_main_reconfigure_failure_swallowed(monkeypatch):
    """stdout.reconfigure 抛 ValueError (异常流) → 吞掉继续."""
    def _reconfigure(**kw):
        raise ValueError("fake reconfigure failure")
    monkeypatch.setattr(sys.stdout, "reconfigure", _reconfigure)
    assert main(["nonexistent.msh"]) == 1


# ═══════════════════════════════════════════════════════════════
# pkg11 A17 — reconfigure_streams 共享 (run_demo 与 runner.main)
# ═══════════════════════════════════════════════════════════════

def test_reconfigure_streams_shared_helper(monkeypatch):
    """编码安全网统一入口 (pkg11 A17).

    判别性: run_demo.py 与 runner.main 曾 7 行逐字复制 — 两处必须
    调用同一 reconfigure_streams; 支持/不支持 reconfigure 与重配
    失败 (ValueError/OSError) 三种流都不得抛异常。
    """
    from fem2d.errors import reconfigure_streams

    calls = []

    class _NoReconfigure:
        pass

    class _Reconfigurable:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    class _Raising:
        def reconfigure(self, **kwargs):
            raise OSError("cannot set")

    for stream in (_NoReconfigure(), _Reconfigurable(), _Raising()):
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", stream)
        reconfigure_streams()
    # stdout + stderr 各一次 (仅 _Reconfigurable 命中; 其余两轮无调用)
    assert calls == [{"errors": "replace"}, {"errors": "replace"}]
