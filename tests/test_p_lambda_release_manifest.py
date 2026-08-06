"""P-λ 发布拆包守恒判别性测试 (4 zip 并集 == 原单 zip 清单, 红侧锁定).

`scripts/make_release_zip.py --split` 把单 zip 拆为 4 包:
  source / runtime-win64 / models / testdata.
守恒契约与有意重叠两处 (运行必需代码 / tests/boundary_golden/) 见该脚本
模块 docstring。本文件用**精确集合相等**锁定:
  - 4 包清单并集 == 原单 zip 清单 (无文件丢失、无凭空新增);
  - 重叠恰为声明两处 (改拆包规则必须先改本文件, 否则红);
  - 4 个 zip 名带版本号 (与 pyproject 单一源一致) 与同一时间戳。

无真实 gmsh 依赖: 只做文件清单/zipfile/dry-run 断言; 运行包子集
self-test 冒烟也不需要 gmsh (verification 走 fem2d 内部解析)。
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts import make_release_zip as mrz

PACKAGES = ("source", "runtime-win64", "models", "testdata")


def _make_tree(root: Path) -> None:
    """确定性合成仓库: 代码/文档/测试/golden/models/tools/生成物全覆盖."""
    tracked = [
        "pyproject.toml", "README.md", "run.py", "run_demo.py", "LICENSE",
        "requirements.txt", ".github/workflows/ci.yml",
        "fem2d/__init__.py", "fem2d/runner.py", "fem2d/boundary/__init__.py",
        "scripts/__init__.py", "scripts/gmsh_runner.py", "scripts/geo_spec.py",
        "scripts/check_dead_code.py",
        "scripts/convergence/test_lc0.0100.geo",
        "docs/CODE_MAP.md", "tests/conftest.py", "tests/test_x.py",
        "tests/boundary_golden/a.json", "tests/boundary_golden/b.json",
        "models/demo.geo", "models/demo.spec",
    ]
    generated = [
        "tools/gmsh-4.15.2-Windows64/gmsh.exe",
        "tools/gmsh-4.15.2-Windows64/LICENSE.gmsh.txt",
        "models/demo.msh",
        "scripts/convergence/test_lc0.0100.msh",
        "scripts/convergence/convergence.png",
        "perf_results.json",
    ]
    for rel in tracked + generated:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('version = "1.2.3"\n', encoding="utf-8")


def _pkg_sets(packages: dict) -> dict:
    return {k: set(v) for k, v in packages.items()}


def test_split_union_equals_single_manifest_real_repo():
    """真实仓库 (标准模式): 4 包并集 == 单 zip 清单, 无丢失/无新增."""
    files = mrz.collect_files(mrz.REPO_ROOT, include_tools=False)
    packages = _pkg_sets(mrz.split_manifests(files))
    union = set().union(*packages.values())
    assert union == set(files)
    for name in PACKAGES:
        assert packages[name], f"{name} 包为空"


def test_split_overlap_contract_exact():
    """有意重叠必须恰为声明两处, 其余包两两不相交."""
    files = mrz.collect_files(mrz.REPO_ROOT, include_tools=False)
    packages = _pkg_sets(mrz.split_manifests(files))
    # 重叠1: runtime 恰是 source 的代码子集 (工具分支下 tools 除外)
    assert packages["runtime-win64"] <= packages["source"]
    assert all(mrz._is_runtime_code(r)
               for r in packages["runtime-win64"])
    # 重叠2: source ∩ testdata 恰为 boundary_golden 全量
    golden = {r for r in files if r.startswith("tests/boundary_golden/")}
    assert golden
    assert packages["source"] & packages["testdata"] == golden
    # 其余两两不相交
    assert packages["models"] & packages["source"] == set()
    assert packages["models"] & packages["runtime-win64"] == set()
    assert packages["models"] & packages["testdata"] == set()
    assert packages["testdata"] & packages["runtime-win64"] == set()


def test_split_full_mode_routes_tools_and_generated(tmp_path):
    """full 模式: tools/ 只进 runtime; 生成物按归属进 models/testdata."""
    _make_tree(tmp_path)
    files = mrz.collect_files(tmp_path, include_tools=True)
    packages = _pkg_sets(mrz.split_manifests(files))
    union = set().union(*packages.values())
    assert union == set(files)
    tools = {r for r in files if r.startswith("tools/")}
    assert tools and tools <= packages["runtime-win64"]
    assert packages["source"] & tools == set()
    # models/ 整目录进示例包 (含生成的 .msh 示例)
    assert packages["models"] == {r for r in files if r.startswith("models/")}
    assert "models/demo.msh" in packages["models"]
    # 可再生的基准产物进测试数据包
    for rel in ("perf_results.json",
                "scripts/convergence/test_lc0.0100.msh",
                "scripts/convergence/convergence.png"):
        assert rel in packages["testdata"], rel
    # 入库输入 (convergence .geo) 不进测试数据包
    assert "scripts/convergence/test_lc0.0100.geo" not in packages["testdata"]
    # 非运行必需脚本只进源码包
    assert "scripts/check_dead_code.py" in packages["source"]
    assert "scripts/check_dead_code.py" not in packages["runtime-win64"]


def test_split_builds_four_zips_versioned(tmp_path):
    """build_split_zips: 4 个 zip 名带版本与时间戳, 清单与并集守恒."""
    _make_tree(tmp_path)
    targets = mrz.build_split_zips(
        tmp_path, tmp_path / "out", include_tools=False,
        version="1.2.3", timestamp="20260805_000000")
    assert {t[0] for t in targets} == set(PACKAGES)
    single = mrz.build_zip(tmp_path, tmp_path / "out", include_tools=False,
                           version="1.2.3", timestamp="20260805_000000")
    with zipfile.ZipFile(single) as z:
        single_set = {n.split("/", 1)[1] for n in z.namelist()}
    expected = {k: set(v) for k, v in mrz.split_manifests(
        mrz.collect_files(tmp_path, include_tools=False)).items()}
    union = set()
    for name, target in targets:
        assert (f"FEM2D_project_Q4_1.2.3_20260805_000000_{name}.zip"
                == target.name), target.name
        with zipfile.ZipFile(target) as z:
            rels = {n.split("/", 1)[1] for n in z.namelist()}
        assert rels == expected[name], f"{name} 包清单与 split_manifests 不一致"
        union |= rels
    assert union == single_set


def test_split_list_cli_union_equals_single_list_cli():
    """CLI 红侧: --split --list 并集 == --list 单清单 (真实仓库)."""
    root = mrz.REPO_ROOT
    single = subprocess.run(
        [sys.executable, str(root / "scripts" / "make_release_zip.py"),
         "--list"], capture_output=True, text=True, check=True)
    split = subprocess.run(
        [sys.executable, str(root / "scripts" / "make_release_zip.py"),
         "--split", "--list"], capture_output=True, text=True, check=True)
    single_files = {l for l in single.stdout.splitlines()
                    if l and not l.startswith("#")}
    split_files = {l for l in split.stdout.splitlines()
                   if l and not l.startswith("#") and not l.startswith("==")}
    assert split_files == single_files  # 拆包前后文件清单并集零差


def test_runtime_subset_self_test_runs(tmp_path):
    """运行包 71 文件子集独立可 `python run.py --self-test` (承诺功能)."""
    files = mrz.collect_files(mrz.REPO_ROOT, include_tools=False)
    runtime = mrz.split_manifests(files)["runtime-win64"]
    assert runtime, "运行包清单为空"
    for rel in runtime:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mrz.REPO_ROOT / rel, dst)
    r = subprocess.run(
        [sys.executable, "run.py", "--self-test"], cwd=tmp_path,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300)
    assert r.returncode == 0, r.stdout[-500:] + r.stderr[-500:]
    assert "6 PASS" in r.stdout and "0 FAIL" in r.stdout, r.stdout[-300:]
