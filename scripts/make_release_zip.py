"""发布 zip 打包 — 把手工打包流程固化为可重复命令 (python zipfile).

用法:
  python scripts/make_release_zip.py             # 标准模式: 打包到 ~/Downloads
  python scripts/make_release_zip.py --list      # 只列将打包的文件, 不写 zip
  python scripts/make_release_zip.py --full      # full 模式: 含 tools/ (gmsh.exe)
  python scripts/make_release_zip.py --out-dir <目录>  # 覆盖输出目录 (测试用)

排除规则与 .gitignore/交接文档一致, 每条规则注明"为什么":
  .git / build / dist / *.egg-info   生成物与版本库不进发布包
                                     (过期 egg-info 曾带旧 PKG-INFO 版本号进包)
  __pycache__ / .pytest_cache / .mypy_cache / .ruff_cache / .venv*
                                     缓存与虚拟环境 (交接文档"缓存"项)
  .coverage / *.zip / tmp*           运行产物与临时文件
  wt_*                               分包 worktree (并行任务副本, 非当前代码)
  PROMPT_*                           派发任务书 (临时文档, 不入包)
  tools/ (标准模式)                  gmsh.exe 数据目录 — 分发时单独给, 保留 GPL
                                     v2+ 声明; full 模式才包含

⚠️ .github 必须保留 — 点开头路径不能无脑排除 (曾漏过 CI 配置, 见交接文档)。

版本单一源: 文件名中的版本号从 pyproject.toml 读取, 与打包声明永不漂移。
"""
import argparse
import fnmatch
import os
import re
import sys
import time
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# 所有模式都排除 (fnmatch 匹配任意路径层级的名字; "为什么"见模块 docstring)
_ALWAYS_EXCLUDE = [
    ".git",
    ".coverage",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv*",
    "wt_*",
    "*.zip",
    "tmp*",
    "PROMPT_*",
    "*.egg-info",
    "build",
    "dist",
]

# zip 内顶层目录 — 与历史发布包一致, 解压后得到干净的项目文件夹
_ARCHIVE_PREFIX = "FEM2D_project_Q4_maintainable"


def _excluded(rel: str, include_tools: bool) -> bool:
    """rel 为仓库根相对的正斜杠路径; 任一路径分量命中规则即排除."""
    parts = rel.split("/")
    for part in parts:
        if any(fnmatch.fnmatchcase(part, pattern) for pattern in _ALWAYS_EXCLUDE):
            return True
    if not include_tools and parts[0] == "tools":
        return True
    return False


def collect_files(root: Path, include_tools: bool = False) -> list:
    """返回按名排序的、规则过滤后的相对路径列表."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        if rel_dir != "." and _excluded(rel_dir, include_tools):
            dirnames[:] = []  # 排除的目录整棵剪枝
            continue
        kept = []
        for name in sorted(dirnames):
            rel = f"{rel_dir}/{name}" if rel_dir != "." else name
            if not _excluded(rel, include_tools):
                kept.append(name)
        dirnames[:] = kept
        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}" if rel_dir != "." else name
            if not _excluded(rel, include_tools):
                files.append(rel)
    return files


def load_version(pyproject: Path) -> str:
    """版本单一源 = pyproject.toml; 发布名里的版本号必须与打包声明一致."""
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        pyproject.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise SystemExit(f"pyproject.toml 找不到 version (单一版本源): {pyproject}")
    return match.group(1)


def build_zip(root, out_dir, include_tools=False, version=None, timestamp=None):
    """打包 root 到 out_dir, 返回 zip 路径. version/timestamp 可注入 (测试确定性)."""
    version = version or load_version(root / "pyproject.toml")
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    name = f"FEM2D_project_Q4_{version}_{timestamp}_maintainable.zip"
    files = collect_files(root, include_tools=include_tools)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel in files:
            archive.write(root / rel, f"{_ARCHIVE_PREFIX}/{rel}")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description="发布 zip 打包 (排除规则见模块 docstring)")
    parser.add_argument("--list", action="store_true",
                        help="只打印将打包的文件列表, 不写 zip (验证排除规则)")
    parser.add_argument("--full", action="store_true",
                        help="full 模式: 含 tools/ (gmsh.exe, 几十 MB 量级)")
    parser.add_argument("--out-dir", default=str(Path.home() / "Downloads"),
                        help="zip 输出目录 (默认 ~/Downloads)")
    args = parser.parse_args(argv)

    version = load_version(REPO_ROOT / "pyproject.toml")
    files = collect_files(REPO_ROOT, include_tools=args.full)
    mode = "full" if args.full else "standard"

    if args.list:
        for rel in files:
            print(rel)
        print(f"# {len(files)} files | version={version} | mode={mode}",
              file=sys.stderr)
        return 0

    target = build_zip(REPO_ROOT, args.out_dir,
                       include_tools=args.full, version=version)
    print(f"wrote {target} ({len(files)} files, {mode} mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
