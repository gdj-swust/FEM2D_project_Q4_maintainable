"""发布 zip 打包 — 把手工打包流程固化为可重复命令 (python zipfile).

用法:
  python scripts/make_release_zip.py             # 标准模式: 打包到 ~/Downloads
  python scripts/make_release_zip.py --list      # 只列将打包的文件, 不写 zip
  python scripts/make_release_zip.py --full      # full 模式: 含 tools/ (gmsh.exe)
  python scripts/make_release_zip.py --out-dir <目录>  # 覆盖输出目录 (测试用)
  python scripts/make_release_zip.py --split     # 拆 4 包 (P-λ 发布拆包, 见下)
  python scripts/make_release_zip.py --split --list   # 只列 4 包清单, 不写 zip
  python scripts/make_release_zip.py --root <目录>    # 仓库根覆盖 (守恒比对用)

排除规则与 .gitignore/交接文档一致 (点开头临时前缀是运行时产物,
.gitignore 以 models/*.msh 覆盖 models/ 内, 本表按任意路径层级兜底),
每条规则注明"为什么":
  .git / build / dist / *.egg-info   生成物与版本库不进发布包
                                     (过期 egg-info 曾带旧 PKG-INFO 版本号进包)
  __pycache__ / .pytest_cache / .mypy_cache / .ruff_cache / .venv*
                                     缓存与虚拟环境 (交接文档"缓存"项)
  .coverage / *.zip / tmp*           运行产物与临时文件
  wt_*                               分包 worktree (并行任务副本, 非当前代码)
  PROMPT_*                           派发任务书 (临时文档, 不入包)
  .fem2d-msh-* / .fem2d-write-probe-*
                                     fem2d 运行时临时产物 (崩溃/强制退出遗留,
                                     曾混入发布包 4.1MB — S-α 审查实证)
  tools/ (标准模式)                  gmsh.exe 数据目录 — 分发时单独给, 保留 GPL
                                     v2+ 声明; full 模式才包含

⚠️ .github 必须保留 — 点开头路径不能无脑排除 (曾漏过 CI 配置, 见交接文档)。

版本单一源: 文件名中的版本号从 pyproject.toml 读取, 与打包声明永不漂移。

--split 拆包设计 (P-λ, 划分以"文件是否可再生 / 是否运行必需"为准):
  source     源码包: 全量减 tools/、models/、可再生的生成物
             (脚本生成的 .msh / convergence.png / perf_results.json);
             保持可 `pip install .` 与 `pytest -q` 冒烟 (含 boundary_golden)。
  runtime    运行包: 运行必需代码 (run.py/fem2d/scripts 运行链子集) + tools/
             (Windows gmsh 4.15.2 可执行); 解压后 `python run.py ...` 端到端。
  models     示例模型包: models/ 全部 (.geo/.spec/.txt/.md + 生成的 .msh 示例)。
  testdata   测试/基准数据包: tests/boundary_golden/ 基准数据 + 可再生的
             收敛/性能基准产物 (scripts/convergence/*.msh、convergence.png、
             perf_results.json), 供验收复跑。

守恒契约: 4 包文件清单并集 == 原单 zip 清单 (无文件丢失、无凭空新增)。
有意重叠两处 (并集守恒下显式声明, 守恒测试红侧锁定):
  1. 运行必需代码 (fem2d/、run.py、run_demo.py、pyproject.toml、
     requirements.txt、README.md、LICENSE、scripts 运行链三文件)
     同时进 source 与 runtime — 源码包需代码做 pip install/pytest,
     运行包需代码做端到端运行;
  2. tests/boundary_golden/** 同时进 source (pytest 完整性, golden
     测试按相对路径读数据) 与 testdata (基准数据本体)。
4 个 zip 使用同一版本号与时间戳、同一顶层目录前缀, 解压到同一目录
即还原完整项目 (运行包可直接引用示例包的 models/)。
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
    ".fem2d-msh-*",
    ".fem2d-write-probe-*",
    "*.egg-info",
    "build",
    "dist",
]

# zip 内顶层目录 — 与历史发布包一致, 解压后得到干净的项目文件夹
_ARCHIVE_PREFIX = "FEM2D_project_Q4_maintainable"

# ── --split 拆包 (P-λ) ──────────────────────────────────────────────
# 4 包固定命名后缀 (守恒测试与 README 用法说明依赖, 改名须同步)
_SPLIT_SUFFIXES = ("source", "runtime-win64", "models", "testdata")

# 运行必需代码: 进 runtime, 同时进 source (有意重叠, 见模块 docstring)
_RUNTIME_CODE = {
    "run.py",
    "run_demo.py",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "scripts/__init__.py",      # runner 依赖 scripts 包声明
    "scripts/gmsh_runner.py",   # fem2d/gmsh_adapter.py 子进程 .geo → .msh
    "scripts/geo_spec.py",      # fem2d/wizard.py .spec/文本 → .geo
}
_RUNTIME_DIRS = ("fem2d/", "tools/")

# 可再生的基准产物 (脚本生成, 源码包排除; 进 testdata)
_GENERATED_ARTIFACTS = ("perf_results.json",)


def _is_runtime_code(rel: str) -> bool:
    """运行必需代码: 显式清单 + fem2d/ tools/ 整目录."""
    if rel in _RUNTIME_CODE:
        return True
    return any(rel.startswith(d) for d in _RUNTIME_DIRS)


def _is_generated_artifact(rel: str) -> bool:
    """脚本可再生的基准产物 (源码包不含; 进测试/基准数据包)."""
    if rel in _GENERATED_ARTIFACTS:
        return True
    if rel.startswith("scripts/convergence/"):
        # convergence_study.py 生成的网格与图; .geo 是入库输入, 留在源码包
        return rel.endswith(".msh") or rel.endswith(".png")
    return False


def split_manifests(files: list) -> dict:
    """单 zip 清单 → 4 包清单 dict.

    守恒契约: 4 包并集 == 输入清单 (无文件丢失、无凭空新增); 有意重叠
    两处 (运行必需代码 / boundary_golden) 见模块 docstring 与守恒测试。
    """
    packages = {name: [] for name in _SPLIT_SUFFIXES}
    for rel in files:
        if rel.startswith("tools/"):
            packages["runtime-win64"].append(rel)
        elif rel.startswith("models/"):
            packages["models"].append(rel)
        elif rel.startswith("tests/boundary_golden/"):
            # 有意重叠: 源码包 (pytest 完整性) + 测试数据包 (基准数据)
            packages["source"].append(rel)
            packages["testdata"].append(rel)
        elif _is_generated_artifact(rel):
            packages["testdata"].append(rel)
        elif _is_runtime_code(rel):
            # 有意重叠: 源码包 (pip install/pytest) + 运行包 (端到端运行)
            packages["runtime-win64"].append(rel)
            packages["source"].append(rel)
        else:
            packages["source"].append(rel)
    return {name: sorted(rels) for name, rels in packages.items()}


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


def _dir_bytes(root: Path, rels: list) -> int:
    """清单文件在磁盘上的字节合计 (报告/验收用, 文件缺失按 0 计)."""
    total = 0
    for rel in rels:
        try:
            total += (root / rel).stat().st_size
        except OSError:
            continue
    return total


def build_split_zips(root, out_dir, include_tools=False, version=None,
                     timestamp=None):
    """--split 模式: 4 包拆包, 返回 [(包名, zip 路径)] (守恒见 split_manifests).

    version/timestamp 可注入 (测试确定性); 4 个 zip 共用同一版本号与
    时间戳、同一顶层目录前缀 — 解压到同一目录即还原完整项目。
    """
    version = version or load_version(root / "pyproject.toml")
    timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
    files = collect_files(root, include_tools=include_tools)
    packages = split_manifests(files)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = []
    for name in _SPLIT_SUFFIXES:
        rels = packages[name]
        target = out_dir / (
            f"FEM2D_project_Q4_{version}_{timestamp}_{name}.zip")
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for rel in rels:
                archive.write(root / rel, f"{_ARCHIVE_PREFIX}/{rel}")
        targets.append((name, target))
    return targets


def main(argv=None):
    parser = argparse.ArgumentParser(description="发布 zip 打包 (排除规则见模块 docstring)")
    parser.add_argument("--list", action="store_true",
                        help="只打印将打包的文件列表, 不写 zip (验证排除规则)")
    parser.add_argument("--full", action="store_true",
                        help="full 模式: 含 tools/ (gmsh.exe, 几十 MB 量级)")
    parser.add_argument("--split", action="store_true",
                        help="拆 4 包: source/runtime-win64/models/testdata "
                             "(守恒并集 == 原单包清单, 见模块 docstring)")
    parser.add_argument("--root", default=None,
                        help="仓库根覆盖 (默认本脚本所在仓库; 守恒比对时指向 "
                             "含 tools/ 的另一检出)")
    parser.add_argument("--out-dir", default=str(Path.home() / "Downloads"),
                        help="zip 输出目录 (默认 ~/Downloads)")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else REPO_ROOT
    version = load_version(root / "pyproject.toml")
    files = collect_files(root, include_tools=args.full)
    mode = "full" if args.full else "standard"

    if args.list:
        if args.split:
            for name, rels in split_manifests(files).items():
                print(f"== {name} ==")
                for rel in rels:
                    print(rel)
                print(f"# {name}: {len(rels)} files", file=sys.stderr)
        else:
            for rel in files:
                print(rel)
        print(f"# {len(files)} files (union) | version={version} | "
              f"mode={mode} | split={bool(args.split)}", file=sys.stderr)
        return 0

    if args.split:
        targets = build_split_zips(root, args.out_dir,
                                   include_tools=args.full, version=version)
        for name, target in targets:
            n = len(split_manifests(files)[name])
            print(f"wrote {target} ({n} files, "
                  f"{_dir_bytes(root, split_manifests(files)[name])/1e6:.1f} MB, "
                  f"{name}, {mode} mode)")
        return 0

    target = build_zip(root, args.out_dir,
                       include_tools=args.full, version=version)
    print(f"wrote {target} ({len(files)} files, {mode} mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
