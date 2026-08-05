"""make_release_zip 排除规则单元测试 (发布链路轮 6a 新增).

构造临时目录树模拟仓库, 直接调用 collect_files/_excluded 验证:
  - 白名单文件必须保留 (.github 等点开头路径不得误排除)
  - 排除规则逐条生效, 标准模式不含 tools/, full 模式含 tools/
  - 端到端 build_zip: 产物无 *.egg-info/build/dist/.git, 有 .github/workflows/ci.yml
"""
import subprocess
import sys
import zipfile
from pathlib import Path

from scripts import make_release_zip as mrz

_KEEP = [
    "pyproject.toml",
    "README.md",
    ".github/workflows/ci.yml",
    ".gitattributes",
    "fem2d/__init__.py",
    "fem2d/boundary/detectors/__init__.py",
    "scripts/geo_spec.py",
    "models/demo.geo",
]

_DROP = [
    ".git/HEAD",
    ".git/config",
    ".coverage",
    "__pycache__/mod.pyc",
    ".pytest_cache/v/cache/nodeids",
    ".mypy_cache/3.13/x.meta.json",
    ".ruff_cache/x",
    ".venv/Scripts/python.exe",
    "wt_pkg_other/file.py",
    "build/lib/fem2d/x.py",
    "dist/fem2d_q4-9.24.0.whl",
    "fem2d_q4.egg-info/PKG-INFO",
    "tools/gmsh.exe",
    "models/tmp_scratch.geo",
    "models/a.zip",
    "PROMPT_9.md",
]


def _make_tree(root: Path) -> None:
    """标准树: _KEEP 全部保留 + _DROP 全部排除 + 一个对照文件."""
    for rel in _KEEP + _DROP:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    (root / "debug_notes.txt").write_text("x\n", encoding="utf-8")
    (root / "pyproject.toml").write_text('version = "1.2.3"\n', encoding="utf-8")


def test_standard_mode_keeps_whitelist_and_excludes_forbidden(tmp_path):
    _make_tree(tmp_path)
    files = mrz.collect_files(tmp_path, include_tools=False)
    rels = set(files)
    # 点开头路径不得误排除 (.github 曾漏进发布包)
    assert ".github/workflows/ci.yml" in rels
    for rel in _KEEP + ["debug_notes.txt"]:
        assert rel in rels, f"应保留却被排除: {rel}"
    for rel in _DROP:
        assert rel not in rels, f"应排除却保留: {rel}"


def test_full_mode_includes_tools_standard_not(tmp_path):
    _make_tree(tmp_path)
    standard = mrz.collect_files(tmp_path, include_tools=False)
    full = mrz.collect_files(tmp_path, include_tools=True)
    assert "tools/gmsh.exe" not in standard
    assert "tools/gmsh.exe" in full
    # full 模式不放松其他排除规则
    assert ".github/workflows/ci.yml" in full
    assert ".git/HEAD" not in full
    assert "fem2d_q4.egg-info/PKG-INFO" not in full


def test_build_zip_artifact_content(tmp_path):
    _make_tree(tmp_path)
    target = mrz.build_zip(tmp_path, tmp_path / "out", include_tools=False,
                           version="1.2.3", timestamp="20260805_000000")
    assert target.name == "FEM2D_project_Q4_1.2.3_20260805_000000_maintainable.zip"
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
    assert "FEM2D_project_Q4_maintainable/.github/workflows/ci.yml" in names
    joined = "/".join(names)
    for forbidden in ("egg-info", ".git/", "build/lib", "dist/fem2d", "tools/"):
        assert forbidden not in joined, f"zip 混入被排除内容: {forbidden}"
    assert "FEM2D_project_Q4_maintainable/models/demo.geo" in names


def test_version_is_single_source_from_pyproject():
    # 真实仓库的 pyproject.toml 是唯一版本源, 必须可解析
    version = mrz.load_version(mrz.REPO_ROOT / "pyproject.toml")
    assert __import__("re").fullmatch(r"\d+\.\d+\.\d+", version), version


def test_list_mode_on_real_repo():
    """--list 在真实仓库上: .github 在列, .git / egg-info / zip 不在列."""
    result = subprocess.run(
        [sys.executable, str(mrz.REPO_ROOT / "scripts" / "make_release_zip.py"),
         "--list"],
        capture_output=True, text=True, check=True)
    listed = result.stdout.splitlines()
    assert ".github/workflows/ci.yml" in listed
    for forbidden in (".git/", "egg-info", ".coverage"):
        assert not any(forbidden in line for line in listed), \
            f"--list 混入被排除路径: {[l for l in listed if forbidden in l]}"
