"""P-κ 统一验收入口 — 四件套 + lint 全命令 + 无 gmsh 模拟, 一条命令完成.

用法 (从任意目录调用, cwd 固定为仓库根):
  python scripts/verify_all.py [--fast | --full | --no-gmsh | --drift]
                               [--list-stages | --stage <阶段名> | --fake-fail]

模式:
  --fast        常规轮次验收: pytest 全量 + 探针 + 漂移门 (不含 fuzz/lint/无 gmsh)
  --full        默认: 四件套 (pytest + 探针 + fuzz 500 + 漂移门) + lint 全命令
  --no-gmsh     无 gmsh 模拟全量: 当前 worktree 临时目录放假 gmsh.py
                (raise ImportError) + PYTHONPATH 前置跑 pytest 全量, 跑完清理。
                验收纪律: "干净 worktree" 是验收方的动作 (无 tools/gmsh.exe —
                交接文档教训 5), 脚本只提供该模式。
  --drift       仅漂移门 (regression_compare vs 钉死基线, 6 值 rel < 1e-9)
  --list-stages 打印阶段清单, 不执行任何阶段
  --stage <名>  只跑指定阶段 (名称见 --list-stages)
  --fake-fail   红侧自证: 跑一个必然失败的假命令阶段 (不引入任何坏文件)

失败语义: 任何阶段失败不中断 — 继续跑完全部选中阶段再汇总 (一次验收看到
全部红点)。退出码: 0 = 全部 PASS; 1 = 命令行用法错误; 2 = test 组任一阶段
FAIL (pytest/probe/fuzz/drift/no-gmsh/fake-fail); 3 = lint 组任一阶段 FAIL
(compileall/ruff/mypy/vulture/self-test, 优先级高于 2)。

来源同步 (改动需两边同步):
  - lint 各阶段命令与 .github/workflows/ci.yml lint job 保持一致
  - 漂移门 (_ci_drift.msh 内容 / regression_compare 参数 / 钉死基线 /
    rel < 1e-9) 与 .github/workflows/ci.yml test-full job 漂移门步骤保持一致

设计约束: 本脚本只执行子进程命令, 不 import fem2d 业务代码; 自身无 gmsh
依赖; 仅用标准库。
"""
import argparse
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, List, NoReturn, Optional, Tuple

# 脚本位于 scripts/ 下 — 仓库根固定由文件位置推导, 从任意目录调用都可跑
REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
_NOT_FOUND = -2  # 命令不存在 (未 pip install -e .[dev]) 的归一返回码

# --- 漂移门常量: 与 ci.yml test-full job 漂移门步骤保持同步 ---
_DRIFT_MSH = (
    "$MeshFormat\n"
    "2.2 0 8\n"
    "$EndMeshFormat\n"
    "$Nodes\n"
    "4\n"
    "1 0 0 0\n"
    "2 1 0 0\n"
    "3 1 1 0\n"
    "4 0 1 0\n"
    "$EndNodes\n"
    "$Elements\n"
    "2\n"
    "1 2 2 1 1 1 2 3\n"
    "2 2 2 1 1 1 3 4\n"
    "$EndElements\n"
)
_DRIFT_RE = re.compile(
    r"max\|u\|=([0-9.eE+-]+) energy=([0-9.eE+-]+) eta=([0-9.eE+-]+) "
    r"sxmax=([0-9.eE+-]+) symax=([0-9.eE+-]+) txymax=([0-9.eE+-]+)")
_DRIFT_BASELINE = [4.8551138165e-06, 2.2668188447e-02, 1.5523972662e+01,
                   1.0279627164e+06, 2.9161118509e+05, 2.7962716378e+04]
_DRIFT_REL_TOL = 1e-9
# --- 漂移门常量结束 ---


@dataclass
class Stage:
    """阶段注册表条目: 名称 / 分组 (test|lint) / 说明 / 执行函数."""
    name: str
    group: str
    help: str
    run: Callable[[], Tuple[bool, str]]  # (ok, detail)


@dataclass
class StageResult:
    name: str
    group: str
    ok: bool
    detail: str


def _run(cmd: List[str], *, capture: bool = False,
         env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """执行子进程 (cwd 固定为仓库根), 命令回显; OSError (命令不存在)
    归一为 returncode=_NOT_FOUND, 便于汇总为 FAIL."""
    print(f"$ {' '.join(cmd)}", flush=True)
    kwargs: dict = {"cwd": str(REPO_ROOT), "env": env}
    if capture:
        kwargs.update(text=True, encoding="utf-8", errors="replace",
                      capture_output=True)
    try:
        return subprocess.run(cmd, **kwargs)
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, _NOT_FOUND, stdout="",
                                           stderr=str(exc))


def _simple_stage(cmd: List[str], *, capture: bool = False) -> Tuple[bool, str]:
    """通用命令阶段: 退出码 0 → PASS, 其余 → FAIL."""
    proc = _run(cmd, capture=capture)
    if capture and proc.stdout:
        print(proc.stdout.strip(), flush=True)
    if proc.returncode == _NOT_FOUND:
        return False, f"命令未找到: {cmd[0]} (需 pip install -e .[dev])"
    if proc.returncode == 0:
        return True, "exit 0"
    return False, f"exit {proc.returncode}"


def _make_stage(name: str, group: str, help_text: str,
                cmd: List[str], *, capture: bool = False) -> Stage:
    """简单命令阶段的注册表构造 (退出码 0 → PASS)."""
    def run() -> Tuple[bool, str]:
        return _simple_stage(cmd, capture=capture)
    return Stage(name, group, help_text, run)


def _pytest_stage() -> Tuple[bool, str]:
    return _simple_stage([PY, "-m", "pytest", "-q"])


def _probe_stage() -> Tuple[bool, str]:
    cmd = [PY, "scripts/audit_contract_probe.py"]
    proc = _run(cmd, capture=True)
    if proc.stdout:
        print(proc.stdout.strip(), flush=True)
    if proc.returncode == _NOT_FOUND:
        return False, "命令未找到: audit_contract_probe.py 无法启动"
    m = re.search(r"FAILS: (\d+)", proc.stdout or "")
    if proc.returncode == 0 and m:
        return True, f"FAILS: {m.group(1)}"
    if m:
        return False, f"exit {proc.returncode}, FAILS: {m.group(1)}"
    return False, f"exit {proc.returncode}, 输出无 FAILS 标记"


def _fuzz_stage() -> Tuple[bool, str]:
    cmd = [PY, "scripts/fuzz_api.py", "500"]
    proc = _run(cmd, capture=True)
    if proc.stdout:
        print(proc.stdout.strip(), flush=True)
    if proc.returncode == _NOT_FOUND:
        return False, "命令未找到: fuzz_api.py 无法启动"
    m = re.search(r"problems: (\d+)", proc.stdout or "")
    if proc.returncode == 0 and m:
        return True, f"problems: {m.group(1)}"
    if m:
        return False, f"exit {proc.returncode}, problems: {m.group(1)}"
    return False, f"exit {proc.returncode}, 输出无 problems 标记"


def _drift_stage() -> Tuple[bool, str]:
    """漂移门: 写确定性 _ci_drift.msh → regression_compare (验收固定参数)
    → 6 值 vs 钉死基线 rel < 1e-9. 临时 msh 用完即删."""
    msh = REPO_ROOT / "_ci_drift.msh"
    try:
        msh.write_text(_DRIFT_MSH, encoding="ascii")
        proc = _run(
            [PY, "scripts/regression_compare.py", "_ci_drift.msh",
             "2.1e11", "0.3", "0.01", "1e6", "0.0", "elimination"],
            capture=True)
        if proc.stdout:
            print(proc.stdout.strip(), flush=True)
        if proc.returncode == _NOT_FOUND:
            return False, "命令未找到: regression_compare.py 无法启动"
        if proc.returncode != 0:
            return False, f"regression_compare exit {proc.returncode}"
        m = _DRIFT_RE.search(proc.stdout or "")
        if not m:
            return False, "regression_compare 输出无法解析 (6 值正则)"
        got = [float(v) for v in m.groups()]
        rel = max(abs(a - b) / abs(b) for a, b in zip(got, _DRIFT_BASELINE))
        if rel < _DRIFT_REL_TOL:
            return True, f"max rel = {rel:.3e} (< {_DRIFT_REL_TOL:.0e})"
        return False, f"数值漂移: max rel = {rel:.3e} (>= {_DRIFT_REL_TOL:.0e})"
    finally:
        msh.unlink(missing_ok=True)


@contextlib.contextmanager
def no_gmsh_pytest_env() -> Iterator[Tuple[Path, dict]]:
    """无 gmsh 模拟环境: 当前 worktree 内临时目录放假 gmsh.py
    (raise ImportError — 测试 skip 守卫只捕 ImportError, OSError 会导致
    1 个测试失败, 见 cae-lessons), PYTHONPATH 前置 (os.pathsep 兼容
    Windows ';'), yield (tmp_dir, env), 退出后清理临时目录."""
    tmp = Path(tempfile.mkdtemp(prefix=".verify_no_gmsh_", dir=str(REPO_ROOT)))
    (tmp / "gmsh.py").write_text(
        'raise ImportError("gmsh unavailable")\n', encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        yield tmp, env
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _no_gmsh_stage() -> Tuple[bool, str]:
    with no_gmsh_pytest_env() as (_tmp, env):
        proc = _run([PY, "-m", "pytest", "-q"], env=env)
    if proc.returncode == _NOT_FOUND:
        return False, "命令未找到: pytest"
    if proc.returncode == 0:
        return True, "0 FAILED (假 gmsh 模拟)"
    return False, f"exit {proc.returncode} (假 gmsh 模拟下存在 FAILED)"


def _fake_fail_stage() -> Tuple[bool, str]:
    """红侧自证: 必然失败的假命令 (不引入任何坏文件)."""
    proc = _run([PY, "-c", "import sys; sys.exit(3)"])
    if proc.returncode == 0:
        return True, "exit 0 (意外全绿)"
    return False, f"exit {proc.returncode} (红侧自证: 必然 FAIL)"


# 阶段注册表 (--list-stages 与 --stage 的唯一事实源)
STAGES = {
    "pytest": Stage("pytest", "test",
                    "pytest 全量 (python -m pytest -q)", _pytest_stage),
    "probe": Stage("probe", "test",
                   "契约探针 (python scripts/audit_contract_probe.py, 0 FAIL)",
                   _probe_stage),
    "fuzz": Stage("fuzz", "test",
                  "fuzz 500 轮固定种子 (python scripts/fuzz_api.py 500)",
                  _fuzz_stage),
    "drift": Stage("drift", "test",
                   "漂移门 (_ci_drift.msh + regression_compare, 6 值 rel<1e-9)",
                   _drift_stage),
    "no-gmsh": Stage("no-gmsh", "test",
                     "无 gmsh 模拟全量 (假 gmsh.py + PYTHONPATH 前置 pytest)",
                     _no_gmsh_stage),
    "fake-fail": Stage("fake-fail", "test",
                       "红侧自证: 必然失败的假命令 (仅 --fake-fail 用)",
                       _fake_fail_stage),
    # --- lint 组: 与 .github/workflows/ci.yml lint job 保持一致 ---
    "compileall": _make_stage(
        "compileall", "lint", "compileall 全源",
        [PY, "-m", "compileall", "-q", "fem2d", "scripts", "tests",
         "run.py", "run_demo.py"]),
    "ruff": _make_stage(
        "ruff", "lint", "ruff check",
        ["ruff", "check", "fem2d/", "scripts/", "tests/", "run.py",
         "run_demo.py"]),
    "mypy": _make_stage("mypy", "lint", "mypy fem2d/", ["mypy", "fem2d/"]),
    "vulture": _make_stage(
        "vulture", "lint", "vulture 死代码扫描",
        ["vulture", "fem2d", "scripts", "--min-confidence", "100"]),
    "self-test": _make_stage(
        "self-test", "lint", "python run.py --self-test",
        [PY, "run.py", "--self-test"]),
    # --- lint 组结束 ---
}

FAST_STAGES = ["pytest", "probe", "drift"]
FULL_STAGES = ["pytest", "probe", "fuzz", "drift",
               "compileall", "ruff", "mypy", "vulture", "self-test"]


class _Parser(argparse.ArgumentParser):
    """用法错误 → 退出码 1 (argparse 默认 2 会与 "test 组 FAIL = 2" 冲突)."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(1, f"verify_all: error: {message}\n")


def _build_parser() -> _Parser:
    parser = _Parser(
        prog="verify_all",
        description="FEM2D 统一验收入口: 四件套 + lint 全命令 + 无 gmsh 模拟.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fast", action="store_true",
                      help="pytest 全量 + 探针 + 漂移门 (常规轮次验收)")
    mode.add_argument("--full", action="store_true",
                      help="四件套 + lint 全命令 (默认)")
    mode.add_argument("--no-gmsh", action="store_true",
                      help="无 gmsh 模拟全量 (假 gmsh 模块 + PYTHONPATH 前置)")
    mode.add_argument("--drift", action="store_true", help="仅漂移门")
    mode.add_argument("--list-stages", action="store_true",
                      help="打印阶段清单, 不执行")
    mode.add_argument("--fake-fail", action="store_true",
                      help="红侧自证: 跑必然失败的假命令阶段")
    mode.add_argument("--stage", metavar="NAME",
                      help="只跑指定阶段 (--list-stages 查看名称)")
    return parser


def _print_stage_list() -> None:
    print("阶段清单 (python scripts/verify_all.py --stage <名> 单跑):")
    for group in ("test", "lint"):
        print(f"  {group} 组:")
        for stage in STAGES.values():
            if stage.group == group:
                print(f"    {stage.name:<11} {stage.help}")
    print("模式: --fast / --full / --no-gmsh / --drift / --stage / "
          "--fake-fail / --list-stages")


def _print_summary(results: List[StageResult]) -> None:
    name_w = max(len(r.name) for r in results)
    bar = "-" * (name_w + 24)
    print(bar)
    print(f"{'阶段':<{name_w}} | 结果 | 说明")
    print(bar)
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"{r.name:<{name_w}} | {mark} | {r.detail}")
    print(bar)


def _verdict(results: List[StageResult]) -> int:
    """任一 lint 组 FAIL → 3; 否则任一 test 组 FAIL → 2; 全绿 → 0."""
    if any(not r.ok and r.group == "lint" for r in results):
        return 3
    if any(not r.ok for r in results):
        return 2
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.list_stages:
        _print_stage_list()
        return 0
    if args.stage:
        if args.stage not in STAGES:
            print(f"verify_all: 未知阶段: {args.stage} "
                  "(--list-stages 查看全部)", file=sys.stderr)
            return 1
        names = [args.stage]
    elif args.fake_fail:
        names = ["fake-fail"]
    elif args.no_gmsh:
        names = ["no-gmsh"]
    elif args.drift:
        names = ["drift"]
    elif args.fast:
        names = FAST_STAGES
    else:
        names = FULL_STAGES  # --full 为默认

    print(f"== 运行 {len(names)} 个阶段: {', '.join(names)} ==", flush=True)
    results: List[StageResult] = []
    for name in names:
        stage = STAGES[name]
        print(f"\n--- 阶段 [{stage.group}] {stage.name}: {stage.help} ---",
              flush=True)
        ok, detail = stage.run()
        results.append(StageResult(name, stage.group, ok, detail))
    print()
    _print_summary(results)
    code = _verdict(results)
    print(f"VERDICT: {'ALL PASS' if code == 0 else 'FAIL'} → 退出码 {code}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
