"""S-α 轮判别性测试 — 外部审查 4 项必修 + 检查器卫生 2 项.

分组 (每组判别性 = 回退旧行为必红):
  A. 表达式 DoS: '9**9**9**9+x*0' 旧代码编译成功求值永久挂起 (外部
     审查 timeout 10s EXIT=124); 修复后 int 常量 → float, 2s 内抛
     OverflowError。既有合法表达式逐位不变 (旧路径参照逐点比对)。
  B. exp 词法误判: 'exp(1)' 旧代码被 'x' in p 子串误判为 callable,
     与 sin(pi/2) 报错契约自相矛盾; 修复后同路径同消息。
  C. 复数 nodes: 旧代码 np.asarray(dtype=float) 只发 ComplexWarning
     静默丢虚部; 修复后构造/replace_nodes/setter 三路 TypeError。
  D. 发布包卫生: .fem2d-msh-* / .fem2d-write-probe-* 曾混入发布 zip
     (models/ 泄漏 4.1MB); 修复后排除规则 + 24h 陈旧启动清扫。
  E. check_dead_code noqa: F401: 注册副作用导入 (boundary/__init__.py)
     与实例形式调用方法曾误报; 修复后 noqa 行免报, 真实死代码仍报。
  F. \\s 转义: 非 raw 字符串 invalid escape (3.14 变 SyntaxError),
     -W error::SyntaxWarning 下全仓库零警告。

红侧演示约定: 本文件各断言即绿侧; 将对应修复改回旧行为后, 对应
分组测试必红 (DoS 组演示"旧代码挂起": 线程 2s 超时即失败)。
"""
import ast
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from fem2d import input_source
from fem2d.loads_core import _compile_expr, parse_vec2
from fem2d.mesh import Mesh
from scripts import make_release_zip as mrz

ROOT = Path(__file__).resolve().parents[1]

# ═══════════════════════════════════════════════════════════════
# A. 表达式 DoS (外部审查 #1, 最高优先)
# ═══════════════════════════════════════════════════════════════

DOS_EXPR = "9**9**9**9+x*0,0"


def test_a_dos_power_chain_raises_within_2s():
    """'9**9**9**9+x*0' 编译+求值必须在 2s 内抛异常.

    红侧: 旧代码 (无 int→float 变换) 编译成功, 求值在 int 域逐级
    膨胀到 10^92 位, 永久挂起 → 线程 2s 超时, 本测试必红.
    绿侧: float 域 9.0**9.0**9.0**9.0 微秒级抛 OverflowError.
    """
    outcome = {}

    def worker():
        try:
            f, _ = parse_vec2(DOS_EXPR)
            outcome["value"] = f(1.0, 1.0)
            outcome["done"] = True
        except Exception as exc:  # OverflowError / ValueError 等均算安全拒绝
            outcome["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(2.0)
    assert not thread.is_alive(), (
        f"DoS 未修复: 表达式 {DOS_EXPR!r} 求值挂起超过 2s")
    assert "error" in outcome, (
        f"DoS 未修复: 表达式求值未抛异常, 返回 {outcome!r}")


def test_a_huge_int_literal_rejected_at_compile():
    """超大整数字面量 (float() 溢出) 编译期拒绝, 秒级而非挂起.

    红侧: 旧代码编译成功 (int 域无界), 求值才爆炸.
    """
    huge = "9" * 400
    with pytest.raises((OverflowError, ValueError)):
        _compile_expr(huge)


def test_a_huge_int_literal_rejected_by_parse_vec2():
    """parse_vec2 入口同样秒级拒绝 (float() 溢出 → 非有限数值)."""
    huge = "9" * 400
    with pytest.raises(ValueError, match="不是有限数值"):
        parse_vec2(huge + ",0")


# ── 逐位不变证明: 既有合法表达式全量行为不变 ──
# 参照 = S-α 前编译路径复刻 (AST 白名单校验 + 编译, 无 int→float
# 变换)。对同一表达式, 新路径与旧路径在网格点上必须逐位相等。
_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "sqrt": math.sqrt, "log": math.log,
    "abs": abs, "pi": math.pi,
}


def _old_path_lambda(expr):
    """S-α 前 _compile_expr 编译路径复刻 (无 int→float 变换).

    校验器只查节点类型不查常量值, int→float 不改变节点类型 —
    语料均为合法表达式, 参照路径省略校验器不改变行为。
    """
    tree = ast.parse(expr.strip(), mode="eval")
    code = compile(tree, "<expr>", "eval")
    return lambda x, y: eval(code, {"__builtins__": {}},
                             {"x": x, "y": y, **_FUNCS})


_BITWISE_CORPUS = [
    "2*x", "10*y", "x**2", "3*(x+y)", "x/2", "-7*y",
    "100*x+50", "sin(pi*x/2)", "exp(x)", "sqrt(x*x+y*y)",
    "log(1+x*y)", "abs(x-3)", "x**(1/2)", "2.5*y+1e3",
    "x**2+y**2", "1/(x+2)", "(x-1)*(y+1)",
]
_GRID = [0.0, 0.5, 1.0, 2.0, -1.0, -2.5, 1e-3, 1e3]


def test_a_existing_expressions_bitwise_unchanged():
    """既有合法表达式与 S-α 前路径逐位一致 (== 即 bitwise).

    红侧: 若变换引入任何舍入差异 (或校验顺序变化), 首个不等点即红.
    """
    for expr in _BITWISE_CORPUS:
        old_f = _old_path_lambda(expr)
        new_f = _compile_expr(expr)
        for x in _GRID:
            for y in _GRID:
                try:
                    expected = old_f(x, y)
                    error = None
                except Exception as exc:
                    expected = None
                    error = exc
                try:
                    actual = new_f(x, y)
                except Exception as exc:
                    actual = None
                    new_error = exc
                else:
                    new_error = None
                if error is not None:
                    # 旧路径异常 → 新路径必须同异常 (不吞不换)
                    assert new_error is not None and (
                        type(new_error) is type(error)), (
                        f"{expr} at ({x},{y}): 旧路径 {error!r}, "
                        f"新路径 {new_error!r}")
                    continue
                assert actual is not None, (
                    f"{expr} at ({x},{y}): 旧路径 {expected!r}, "
                    f"新路径异常 {new_error!r}")
                assert actual == expected, (
                    f"{expr} at ({x},{y}): 旧路径 {expected!r} != "
                    f"新路径 {actual!r} (逐位漂移)")


def test_a_int_literal_vs_float_literal_bitwise():
    """'2*x' 与 '2.0*x' 浮点逐位相同 (int→float 转换不改变数值)."""
    xs = [0.0, 0.5, 1.0, -1.5, 2.0, 1e-3, 1e3, 1e-16]
    for expr in ("2*x", "2*x+y", "x/2", "x**2"):
        f_int = _compile_expr(expr)
        f_float = _compile_expr(expr.replace("2", "2.0", 1))
        for x in xs:
            for y in xs:
                assert f_int(x, y) == f_float(x, y), (
                    f"{expr} at ({x},{y}): {f_int(x, y)} != {f_float(x, y)}")


def test_a_complex_result_safety_path_unchanged():
    """负数指数/小数次幂返回 complex 的既安全路径保持 (不抛不改).

    _load_component_ok 拒绝 complex 是既有契约, 本测试锁定解析层
    行为不变 (求值返回 complex, 由载荷应用层拒绝)。
    """
    f = _compile_expr("x**(-0.5)")
    assert isinstance(f(-1.0, 0.0), complex)


# ═══════════════════════════════════════════════════════════════
# B. exp 词法误判 (外部审查 #2)
# ═══════════════════════════════════════════════════════════════

def test_b_exp1_unified_with_sin_pi2():
    """'exp(1)' 与 'sin(pi/2)' 行为统一 — 同为常数表达式, 同路径报错.

    红侧: 旧代码 'exp(1)' 被 'x' in p 子串误判为 callable
    ('exp' 含字母 x), 本测试必红。
    """
    with pytest.raises(ValueError, match="无法解析"):
        parse_vec2("exp(1),0")
    with pytest.raises(ValueError, match="无法解析"):
        parse_vec2("sin(pi/2),0")


def test_b_exp_spatial_still_callable():
    """含 x/y 变量的函数表达式仍是空间函数 (callable)."""
    assert callable(parse_vec2("exp(x),0")[0])
    assert callable(parse_vec2("exp(x/2),0")[0])
    assert callable(parse_vec2("sin(pi*x/2),0")[0])
    assert callable(parse_vec2("2*x,0")[0])
    assert callable(parse_vec2("x,0")[0])
    assert parse_vec2("1e6,0") == (1e6, 0.0)


# ═══════════════════════════════════════════════════════════════
# C. 复数 nodes 拒绝 (外部审查 #3)
# ═══════════════════════════════════════════════════════════════

_COMPLEX_NODES = np.array([[0, 0], [1, 0], [1 + 5j, 1]])
_REAL_NODES = np.array([[0, 0], [1, 0], [0, 1]])
_ELEMS = np.array([[0, 1, 2]])


def test_c_constructor_rejects_complex_nodes():
    """Mesh 构造复数 nodes → TypeError (旧: 仅 ComplexWarning 静默丢虚部).

    红侧: 回退为 np.asarray(dtype=float) 后本测试必红 (无异常抛出).
    """
    with pytest.raises(TypeError, match="complex"):
        Mesh(_COMPLEX_NODES, _ELEMS)


def test_c_replace_nodes_rejects_complex():
    """replace_nodes 复数 → TypeError."""
    mesh = Mesh(_REAL_NODES.copy(), _ELEMS)
    with pytest.raises(TypeError, match="complex"):
        mesh.replace_nodes(np.array([[0, 0], [1, 0], [0, 1j]]))


def test_c_nodes_setter_rejects_complex():
    """nodes property setter (路由到 replace_nodes) 同样拒绝."""
    mesh = Mesh(_REAL_NODES.copy(), _ELEMS)
    with pytest.raises(TypeError, match="complex"):
        mesh.nodes = np.array([[0, 0], [1, 0], [0 + 1j, 1]])


def test_c_real_nodes_no_complex_warning():
    """实数 nodes 正常构造, 无 ComplexWarning (复数拒绝不误伤实数).

    复数拒绝发生在 np.asarray(dtype=float) 之前 — 实数路径根本
    到不了强制转换, 不产生 ComplexWarning。
    """
    import warnings

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        mesh = Mesh(_REAL_NODES.copy(), _ELEMS)
        mesh.replace_nodes(np.array([[0, 0], [1.5, 0], [0, 1]]))
    complex_warn = np.exceptions.ComplexWarning if hasattr(
        np, "exceptions") else np.ComplexWarning
    assert not [w for w in records
                if issubclass(w.category, complex_warn)]
    assert mesh.nodes[1, 0] == 1.5


# ═══════════════════════════════════════════════════════════════
# D. 发布包卫生 (外部审查 #4)
# ═══════════════════════════════════════════════════════════════

def _tree_with_fem2d_temps(root: Path) -> None:
    """标准树 + .fem2d-msh-* / .fem2d-write-probe-* 泄漏文件."""
    for rel in [
        "fem2d/__init__.py",
        "models/demo.geo",
        "models/.fem2d-msh-45brgl7c.msh",
        "out/.fem2d-msh-4tjeef01.msh",
        "scripts/.fem2d-write-probe-12345",
        "models/.fem2d-write-probe-67890",
    ]:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('version = "1.2.3"\n',
                                         encoding="utf-8")


def test_d_release_zip_excludes_fem2d_temps(tmp_path):
    """collect_files 排除 .fem2d-msh-* / .fem2d-write-probe-* (任意层级).

    红侧: 回退 _ALWAYS_EXCLUDE 后泄漏文件进清单, 本测试必红.
    """
    _tree_with_fem2d_temps(tmp_path)
    files = mrz.collect_files(tmp_path, include_tools=False)
    rels = set(files)
    assert "fem2d/__init__.py" in rels
    assert "models/demo.geo" in rels
    assert not any(rel.startswith("models/.fem2d")
                   or "/.fem2d" in rel for rel in rels), (
        f".fem2d-* 临时文件泄漏进发布清单: "
        f"{[r for r in rels if '.fem2d' in r]}")


def test_d_split_conservation_holds_with_fem2d_temps(tmp_path):
    """split 守恒契约不破坏: 并集 == 单包清单, 且无 .fem2d-* 文件."""
    _tree_with_fem2d_temps(tmp_path)
    files = mrz.collect_files(tmp_path, include_tools=False)
    packages = mrz.split_manifests(files)
    union = sorted({rel for rels in packages.values() for rel in rels})
    assert union == sorted(files), (
        "split 并集 != 单包清单 — 守恒契约被破坏")
    assert not any(".fem2d" in rel for rel in union)


def test_d_stale_msh_sweep_removes_24h_old_only(tmp_path):
    """启动清扫: 24h 前创建的 .fem2d-msh-* 删除, 新鲜文件保留.

    时间戳用相对时间 (now-25h) 构造 — 禁绝对阈值.
    """
    stale = tmp_path / ".fem2d-msh-stale.msh"
    stale.write_text("x", encoding="utf-8")
    fresh = tmp_path / ".fem2d-msh-fresh.msh"
    fresh.write_text("x", encoding="utf-8")
    other = tmp_path / "normal.msh"
    other.write_text("x", encoding="utf-8")
    old = time.time() - 25 * 3600
    os.utime(stale, (old, old))

    removed = input_source._sweep_stale_msh_temps(str(tmp_path))

    assert removed == 1
    assert not stale.exists(), "24h 前临时网格应被清扫"
    assert fresh.exists(), "新鲜临时网格不得误删"
    assert other.exists(), "非 .fem2d-msh-* 前缀不得误删"
    assert input_source._sweep_stale_msh_temps(str(tmp_path / "nope")) == 0


def test_d_generate_geo_sweeps_stale_on_startup(monkeypatch, tmp_path):
    """generate_geo_with_topology 入口触发启动清扫 (无 gmsh 环境)."""
    stale = tmp_path / ".fem2d-msh-stale.msh"
    stale.write_text("x", encoding="utf-8")
    fresh = tmp_path / ".fem2d-msh-fresh.msh"
    fresh.write_text("x", encoding="utf-8")
    old = time.time() - 25 * 3600
    os.utime(stale, (old, old))
    geo = tmp_path / "m.geo"
    geo.write_text("Point(1) = {0, 0, 0, 0.1};", encoding="utf-8")

    class _Stub:
        """gmsh 运行桩: 未生成 (None) → generate 提前返回, 清扫已执行."""

        @staticmethod
        def run_gmsh(*_args, **_kwargs):
            return None

        @staticmethod
        def temp_copy_dir(*_args, **_kwargs):
            return str(tmp_path)

    monkeypatch.setattr(input_source, "_import_scripts",
                        lambda _name: _Stub())
    result = input_source.generate_geo_with_topology(
        str(geo), output_path=str(tmp_path / "m.msh"))
    assert result == (None, None)
    assert not stale.exists(), "生成入口应清扫 24h 前遗留临时网格"
    assert fresh.exists(), "新鲜临时文件不得被启动清扫误删"


# ═══════════════════════════════════════════════════════════════
# E. check_dead_code noqa: F401 (检查器卫生 #5)
# ═══════════════════════════════════════════════════════════════

def _run_dead_code(paths):
    proc = subprocess.run(
        [sys.executable, "scripts/check_dead_code.py", *map(str, paths)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_e_noqa_f401_suppresses_unused_import(tmp_path):
    """行尾 `# noqa: F401` → 未使用导入免报; 真实死代码仍报.

    红侧: 无 noqa 的未使用导入必须被报 (删一个真实未用导入仍红);
    绿侧: 加 noqa 注释后该行免报, 同文件零引用函数仍报 —
    真实死代码不因 noqa 机制豁免。
    """
    module = tmp_path / "s_alpha_dead_mod.py"
    # 红侧: 无 noqa → 未使用导入 os 必须被报
    module.write_text("import os\n", encoding="utf-8")
    out = _run_dead_code([module])
    assert "s_alpha_dead_mod.py:1: os" in out, (
        f"未使用导入未报 (红侧失效): {out}")
    # 绿侧: noqa: F401 → os 免报; 零引用函数仍报 — 真实死代码不豁免
    module.write_text(
        "import os  # noqa: F401 — 测试豁免\n"
        "def _never_called():\n    return 1\n",
        encoding="utf-8")
    out = _run_dead_code([module])
    assert "os" not in out, f"noqa: F401 未生效: {out}"
    assert "_never_called" in out, (
        f"真实死代码未被报 (豁免过宽): {out}")


def test_e_repo_candidates_cleared(tmp_path):
    """全仓库报告: 两处已知误报 (注册副作用导入/实例形式方法) 已消除.

    红侧: 回退 noqa 识别后 plugins/metadata_for_edges 重新出现必红.
    """
    out = _run_dead_code([ROOT / "fem2d"])
    assert "boundary/__init__.py:27: plugins" not in out, (
        f"注册副作用导入仍被误报: {out}")
    assert "metadata_for_edges" not in out, (
        f"实例形式调用方法仍被误报: {out}")


# ═══════════════════════════════════════════════════════════════
# F. \s 转义 (检查器卫生 #6 — 3.14 会变 SyntaxError)
# ═══════════════════════════════════════════════════════════════

def test_f_no_invalid_escape_syntaxwarnings():
    """全仓库 .py 在 -W error::SyntaxWarning 下零警告编译.

    红侧: test_fix_b2_format_anchor.py:3 的 ``^\\s*`` 未转义 →
    编译报 SyntaxError 必红 (3.14 起是 SyntaxError, 当前是警告).
    """
    import warnings

    py_files = []
    for base in ("fem2d", "scripts", "tests"):
        py_files.extend(Path(base).rglob("*.py"))
    for extra in ("run.py", "run_demo.py"):
        if (ROOT / extra).exists():
            py_files.append(ROOT / extra)
    bad = []
    for path in py_files:
        if "__pycache__" in str(path):
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                compile(src, str(path), "exec")
            except SyntaxError as exc:
                bad.append(f"{path}: SyntaxError {exc}")
                continue
        for w in caught:
            if issubclass(w.category, SyntaxWarning):
                bad.append(f"{path}:{w.lineno} {w.message}")
    assert not bad, "存在 invalid escape SyntaxWarning:\n" + "\n".join(bad[:10])
