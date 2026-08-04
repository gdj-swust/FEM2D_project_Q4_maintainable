"""审计工具自身 smoke 测试 — 探针/fuzz/回归/基准需防回归 (评审建议).

参照 tests/test_static_checkers_smoke.py 模式: 运行不崩溃 + 已知 FAIL
场景必须被抓出 (判别性, 注入 bug 放回旧实现必须失败)。探针总数与
docstring 覆盖声明核对 — 覆盖声明非纸面。全部无 gmsh 依赖
(import_msh 对缺 gmsh 环境走文本解析回退, 不 skip)。
"""
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(PROJECT_ROOT, "scripts")


def _run(args, timeout=300):
    return subprocess.run(
        [sys.executable] + args,
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout)


def _load_script(name):
    """从项目根加载 scripts/<name> (脚本内已注入根目录 sys.path)."""
    sys.path.insert(0, PROJECT_ROOT)
    spec = importlib.util.spec_from_file_location(
        name[:-3], os.path.join(SCRIPTS, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_wrapper(code):
    """把 wrapper 代码写到临时 .py (Windows 上 gmsh 句柄可能锁文件,
    rmtree 容错)."""
    d = tempfile.mkdtemp(prefix="fem2d_smoke_")
    path = os.path.join(d, "wrapper.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(code)
    return path, d


# ── audit_contract_probe ──────────────────────────────────────────

def test_probe_runs_clean():
    r = _run(["scripts/audit_contract_probe.py"])
    assert r.returncode == 0, r.stderr[-500:]
    assert "FAILS: 0" in r.stdout, r.stdout[-300:]


def test_probe_count_matches_docstring():
    """探针总数与 docstring 覆盖声明一致 — 覆盖率声明非纸面."""
    path = os.path.join(SCRIPTS, "audit_contract_probe.py")
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    n_calls = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "probe")
    doc = ast.get_docstring(tree)
    assert f"{n_calls} 项探针" in doc, (
        f"docstring 覆盖声明与实际探针数不一致: 实际 {n_calls}, "
        f"声明节选: {doc[-160:]}")


def test_probe_rejects_bare_indexerror_regression():
    """assemble_loads 期望收紧为只接受 ValueError — 裸 IndexError 回归
    必须被探针报出 (判别性: 9.19.0 前的旧探针期望 (IndexError,
    ValueError) 会放行)."""
    mod = _load_script("audit_contract_probe.py")
    import fem2d

    def bare_index(mesh, n_dof):
        # 模拟旧实现: 集中力越界写 → 裸 IndexError
        mesh.build_connectivity()
        F = np.zeros(n_dof)
        for cf in mesh.concentrated_forces:
            F[2 * cf["node"]] += cf["force"][0]
        return F

    real = fem2d.assemble_loads
    try:
        fem2d.assemble_loads = bare_index
        mod.FAILS.clear()
        m, _ = mod._solved()
        mod.probe("regression", lambda: fem2d.assemble_loads(m, 3), ValueError)
        assert mod.FAILS, "裸 IndexError 回归未被探针报出"
        assert "IndexError" in mod.FAILS[0], mod.FAILS[0]
    finally:
        fem2d.assemble_loads = real


# ── combo_fuzz ────────────────────────────────────────────────────

def test_combo_fuzz_runs_clean():
    r = _run(["scripts/combo_fuzz.py"])
    assert r.returncode == 0, r.stderr[-500:]
    assert "problems=0" in r.stdout, r.stdout[-300:]


def test_combo_fuzz_catches_dropped_load():
    """注入"载荷静默丢弃 → 全零解"必须被报出 — 零解判定判别性
    (9.21.1 只查 isfinite, 全零解通过)."""
    mod = _load_script("combo_fuzz.py")
    import fem2d.solver as solver

    real = solver.assemble_loads
    try:
        solver.assemble_loads = lambda mesh, n_dof: np.zeros(n_dof)
        issues = []
        for et in mod.ELEM_TYPES:
            for sc in mod.SCALES:
                for lk in mod.LOAD_KINDS:
                    try:
                        issue = mod.run_combo(et, sc, lk)
                    except Exception as exc:  # noqa: BLE001 — 组合 fuzz 抓一切
                        issues.append(f"CRASH {type(exc).__name__}")
                        continue
                    if issue:
                        issues.append(issue)
        assert issues, "注入零解场景未被 combo_fuzz 报出"
        assert any("静默零解" in i for i in issues), issues[:3]
    finally:
        solver.assemble_loads = real


# ── fuzz_api ──────────────────────────────────────────────────────

def test_fuzz_api_runs_clean():
    r = _run(["scripts/fuzz_api.py", "100"])
    assert r.returncode == 0, r.stderr[-500:]
    assert "problems: 0" in r.stdout, r.stdout[-300:]


def test_fuzz_api_catches_silent_acceptance():
    """值类别非法输入必须断言抛异常 — API 静默接受非法输入必须被报出
    (判别性: 收紧前 silent_ok=True 整体豁免, 静默成功查不出)."""
    wrapper, d = _write_wrapper(f"""\
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
import importlib.util
import fem2d as F
F.get_element_kernel = lambda t: None  # 非法类型静默接受 (契约要求 ValueError)
spec = importlib.util.spec_from_file_location(
    "fuzz", {os.path.join(SCRIPTS, "fuzz_api.py")!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.argv = ["fuzz_api.py", "500"]
mod.main()
""")
    try:
        r = _run([wrapper])
        assert r.returncode == 1, f"静默接受未被报出: {r.stdout[-300:]}"
        assert "problems: 0" not in r.stdout, r.stdout[-300:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── regression_compare ────────────────────────────────────────────

_MSH_QUAD = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
2
1 2 2 1 1 1 2 3
2 2 2 1 1 1 3 4
$EndElements
"""


def _regression_args(msh):
    return ["scripts/regression_compare.py", msh,
            "2.1e11", "0.3", "0.01", "1e6", "0.0", "elimination"]


def test_regression_compare_runs_clean():
    d = tempfile.mkdtemp(prefix="fem2d_smoke_")
    try:
        msh = os.path.join(d, "quad.msh")
        with open(msh, "w", encoding="utf-8") as fh:
            fh.write(_MSH_QUAD)
        r = _run(_regression_args(msh))
        assert r.returncode == 0, r.stderr[-500:]
        assert "max|u|=" in r.stdout, r.stdout[-300:]
        assert "eta=" in r.stdout, r.stdout[-300:]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_regression_compare_fails_on_broken_msh():
    """断裂 .msh 输入必须非零退出 — 脚本不能静默输出结果."""
    d = tempfile.mkdtemp(prefix="fem2d_smoke_")
    try:
        msh = os.path.join(d, "broken.msh")
        with open(msh, "w", encoding="utf-8") as fh:
            fh.write("$MeshFormat\n2.2 0 8\n")  # 截断, 无网格
        r = _run(_regression_args(msh))
        assert r.returncode != 0, f"断裂 .msh 竟成功: {r.stdout[-300:]}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ── perf_benchmark ────────────────────────────────────────────────

def test_perf_benchmark_runs_clean():
    d = tempfile.mkdtemp(prefix="fem2d_smoke_")
    try:
        out = os.path.join(d, "bench.json")
        r = _run(["scripts/perf_benchmark.py", "--scale", "1000", "--out", out])
        assert r.returncode == 0, r.stderr[-500:]
        assert os.path.isfile(out), "基准 JSON 未写出"
        with open(out, encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["micro_scale_smoke_1e-150"] == "ok"
        assert data["scales"], "基准无档位数据"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_perf_benchmark_reports_inner_crash():
    """内核异常必须非零退出 — 基准不能吞错误假装成功."""
    wrapper, d = _write_wrapper(f"""\
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
import importlib.util
import fem2d as F
F.solve = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("injected"))
spec = importlib.util.spec_from_file_location(
    "perf", {os.path.join(SCRIPTS, "perf_benchmark.py")!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.argv = ["perf_benchmark.py", "--scale", "1000", "--out", "perf_smoke.json"]
mod.main()
""")
    try:
        r = _run([wrapper])
        assert r.returncode != 0, f"注入崩溃未被报出: {r.stdout[-300:]}"
    finally:
        shutil.rmtree(d, ignore_errors=True)
