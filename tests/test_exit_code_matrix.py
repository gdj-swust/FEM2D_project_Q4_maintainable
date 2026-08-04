"""CLI 退出码矩阵锁定测试 (包 5 任务 1 — sys.exit 迁移).

矩阵: 正常 0 / 用户错误 1 / 内部错误 2.
迁移前这些路径以 SystemExit 直接退出, 迁移后 femd2/ 库层抛
CliError、runner.main 捕获返回 int — 进程等价退出码必须不变。

``_process_exit_code`` 把两种形态归一为进程退出码, 使本文件在
迁移前后都能运行并锁定同一行为 (判别性: 任何一处退出码变化即失败).
"""
import contextlib
import io

import numpy as np
import pytest

from fem2d.runner import main

import fem2d.gmsh_adapter as gmsh_adapter
import fem2d.runner as runner_mod
import fem2d.wizard as wizard


def _process_exit_code(argv):
    """runner.main 的进程等价退出码 — 迁移前 SystemExit / 迁移后 int."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            return main(argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1


def _fake_msh_import(elem_type="CPS4", elements=None, nodes=None):
    """test_config 同款 fake import 结果 — 无 Gmsh 依赖."""
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


# ═══════════════════════════════════════════════════════════════
# 矩阵: 正常 0
# ═══════════════════════════════════════════════════════════════

def test_matrix_normal_run_returns_zero(monkeypatch, msh_file, fake_msh):
    """正常求解流程 → 0 (迁移前后一致)."""
    fake_msh(_fake_msh_import())
    assert _process_exit_code(
        [msh_file, "--fix", "1", "--traction", "2:1e6,0", "--no-plot"],
        ) == 0


# ═══════════════════════════════════════════════════════════════
# 矩阵: 用户错误 1
# ═══════════════════════════════════════════════════════════════

def test_matrix_missing_file_returns_one(monkeypatch):
    """文件不存在 → 1 (用户错误)."""
    assert _process_exit_code(
        ["nonexistent.inp", "--no-plot"]) == 1


def test_matrix_plane_conflict_returns_one(monkeypatch, msh_file, fake_msh):
    """CPE 网格 + --plane stress 冲突 → 1 (用户错误).

    曾裸 ValueError 冒泡到 main 顶层 except Exception → 归为内部错误 2;
    --plane 冲突是用户参数问题, 必须与 --elem-type 不兼容同属用户错误。
    """
    fake_msh(_fake_msh_import(elem_type="CPE4"))
    assert _process_exit_code(
        [msh_file, "--plane", "stress", "--no-plot"]) == 1


def test_matrix_elem_type_mismatch_returns_one(monkeypatch, msh_file,
                                               fake_msh):
    """--elem-type 与网格节点数不兼容 → 1 (用户错误).

    曾 runner._build_mesh 直接 sys.exit(1) — 迁移后 CliError(1).
    """
    fake_msh(_fake_msh_import(elem_type="CPS3", elements=[[0, 1, 2]]))
    assert _process_exit_code(
        [msh_file, "--elem-type", "Q4", "--no-plot"]) == 1


def test_matrix_require_physical_groups_returns_one(monkeypatch, msh_file,
                                                    fake_msh):
    """--require-physical-groups 但无 Physical Curve → 1 (用户错误).

    曾 runner._build_boundary 直接 sys.exit(1) — 迁移后 CliError(1).
    """
    fake_msh(_fake_msh_import())
    assert _process_exit_code(
        [msh_file, "--require-physical-groups", "--no-plot"],
        ) == 1


def test_matrix_boundary_value_error_returns_one(monkeypatch, msh_file,
                                                 fake_msh):
    """边界构建抛 ValueError → 1 (用户错误, 网格/边界问题).

    曾 runner._build_boundary 捕获后 sys.exit(1) — 迁移后 CliError(1).
    """
    fake_msh(_fake_msh_import())
    monkeypatch.setattr(
        runner_mod, "build_boundary_segments",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    assert _process_exit_code([msh_file, "--no-plot"]) == 1


def test_matrix_patch_test_failure_returns_one(monkeypatch, msh_file,
                                               fake_msh):
    """--self-test (带网格) patch test 失败 → 1.

    曾 runner._ensure_patch_test 直接 sys.exit(1) — 迁移后 CliError(1).
    """
    fake_msh(_fake_msh_import())
    monkeypatch.setattr(
        runner_mod, "run_patch_test",
        lambda *a, **k: {"all_passed": False})
    monkeypatch.setattr(runner_mod, "run_plane_verification",
                        lambda *a, **k: (0, 0))
    assert _process_exit_code(
        [msh_file, "--self-test", "--no-plot"]) == 1


def test_matrix_plane_verification_failure_returns_one(monkeypatch, msh_file,
                                                       fake_msh):
    """--self-test (带网格) 平面材料验证失败 → 1.

    曾 runner._run_solve_time_self_test 直接 sys.exit(1) — 迁移后
    CliError(1).
    """
    fake_msh(_fake_msh_import())
    monkeypatch.setattr(
        runner_mod, "run_patch_test",
        lambda *a, **k: {"all_passed": True})
    monkeypatch.setattr(runner_mod, "run_plane_verification",
                        lambda *a, **k: (0, 1))
    monkeypatch.setattr(runner_mod, "verify_all_elements",
                        lambda *a, **k: None)
    assert _process_exit_code(
        [msh_file, "--self-test", "--no-plot"]) == 1


def test_matrix_wizard_missing_file_returns_one(monkeypatch):
    """向导 '使用已有文件' 但文件不存在 → 1 (用户错误).

    曾 wizard.run_wizard 直接 SystemExit(1) — 迁移后 CliError(1).
    """
    monkeypatch.setattr(wizard, "_ask_choice", lambda *a, **k: 1)
    monkeypatch.setattr(wizard, "ask", lambda prompt: "nope.geo")
    assert _process_exit_code(["--wizard", "--no-plot"]) == 1


# ═══════════════════════════════════════════════════════════════
# 矩阵: 内部错误 2
# ═══════════════════════════════════════════════════════════════

def test_matrix_inp_input_rejected_returns_one(monkeypatch, tmp_path):
    """.inp 输入口已移除 → 1 (用户错误: 传了不支持的文件类型)."""
    path = tmp_path / "old.inp"
    path.write_text("*NODE\n1,0,0\n", encoding="utf-8")
    assert _process_exit_code([str(path), "--no-plot"]) == 1


def test_matrix_unknown_exception_returns_two(monkeypatch, msh_file,
                                              fake_msh):
    """输入解析阶段任意异常 (内部错误) → 2."""
    fake_msh(_fake_msh_import())
    monkeypatch.setattr(
        runner_mod, "build_boundary_segments",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("internal")))
    assert _process_exit_code([msh_file, "--no-plot"]) == 2
