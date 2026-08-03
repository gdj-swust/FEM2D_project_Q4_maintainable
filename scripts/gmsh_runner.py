"""Safe Gmsh execution for FEM2D.

This module owns external mesh generation.  It never edits the source ``.geo``
file, writes Gmsh output (native ``.msh``) to a temporary file and only then
atomically publishes the requested output.  Topology validation is performed
by the importing process (``fem2d.gmsh_adapter.import_msh``).
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time

_EXPLICIT_SAVE_RE = re.compile(
    r"\bSave\s+\"[^\"]+\"\s*;",
    re.IGNORECASE,
)
_EXPLICIT_MESH_RE = re.compile(
    r"\bMesh\s+\d+\s*;",
    re.IGNORECASE,
)


def _bundled_gmsh_candidates():
    """Return project-relative standalone Gmsh candidates."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.name == "nt":
        return (
            os.path.join(
                project_root, "tools",
                "gmsh-4.15.2-Windows64", "gmsh.exe"),
            os.path.join(project_root, "tools", "gmsh", "gmsh.exe"),
        )
    return (os.path.join(project_root, "tools", "gmsh", "gmsh"),)


def find_gmsh():
    """Return the configured Gmsh executable, or ``None``.

    Search order:
      1. ``GMSH_PATH`` environment variable
      2. project-bundled native ``gmsh.exe``
      3. ``gmsh.exe``/``gmsh`` on PATH, rejecting Python wrappers
    """
    def usable(path):
        return (
            bool(path)
            and os.path.isfile(path)
            and not str(path).lower().endswith((".bat", ".cmd"))
        )

    configured = os.path.expandvars(
        os.path.expanduser(os.environ.get("GMSH_PATH", "").strip().strip('"')))
    if usable(configured):
        return configured
    for bundled in _bundled_gmsh_candidates():
        if usable(bundled):
            return bundled
    for name in ("gmsh.exe", "gmsh"):
        executable = shutil.which(name)
        if usable(executable):
            return executable
    return None


def build_gmsh_command(
        gmsh_exe, geo_path, output_path, quad=False, threads=None,
        fmt="msh"):
    """Build a deterministic Gmsh command without modifying ``geo_path``.

    ``fmt`` 输出格式 — 原生 Gmsh ``msh`` (默认): .msh 含 $PhysicalNames/
    $Entities, 主进程用 Gmsh API 读回即可恢复 CAD 与物理组语义, 不再需要
    Abaqus .inp 中间格式 (2026-08: 移除 Abaqus 输入口)。
    """
    command = [str(gmsh_exe), "-v", "2"]
    if threads is not None:
        command.extend(["-nt", str(max(1, int(threads)))])
    if quad:
        # Set options before parsing the input file.  Algorithm 8 is the
        # frontal-Delaunay quadrilateral algorithm; RecombineAll also covers
        # existing unstructured surfaces.
        command.extend([
            "-setnumber", "Mesh.RecombineAll", "1",
            "-setnumber", "Mesh.Algorithm", "8",
        ])
    command.extend([
        os.path.abspath(geo_path),
        "-2",
        "-o", os.path.abspath(output_path),
        "-format", fmt,
    ])
    return command


def sanitize_geo_source(source):
    """剥离 .geo 脚本中的 Save / Mesh 命令 / Mesh.Format 赋值 (纯文本).

    唯一实现 — gmsh_adapter._safe_geo_source (API 路径) 与子进程路径
    共用, 曾双实现且正则常量各一份, 语义靠人工对齐 (审计 2026-08-03)。

    - ``Save`` 可绕过验证后的 ``-o`` 目标;
    - ``Mesh 2;`` 会在解析期 + 命令行 ``-2`` 各网格化一次;
    - 脚本内 ``Mesh.Format`` (如 39=Abaqus) 会覆盖命令行输出格式。
    """
    sanitized = _EXPLICIT_SAVE_RE.sub(
        "// Save removed by FEM2D; FEM2D owns publication", source)
    sanitized = _EXPLICIT_MESH_RE.sub(
        "// Mesh command removed by FEM2D; FEM2D meshes once",
        sanitized,
    )
    sanitized = re.sub(
        r"^\s*Mesh\s*\.\s*Format\s*=\s*[^;]*;",
        "// Mesh.Format removed by FEM2D; output is native .msh",
        sanitized,
        flags=re.MULTILINE,
    )
    return sanitized


def _geometry_without_explicit_save(geo_path):
    """Return a source-safe geometry path and an optional cleanup path.

    A scripted ``Save`` can bypass the validated ``-o`` target. A scripted
    ``Mesh 2;`` would also mesh once while parsing and again for the command
    line ``-2`` option. Strip both standalone directives in a temporary
    sibling copy, leaving the user's geometry untouched and relative includes
    anchored to the same directory.
    """
    with open(geo_path, "r", encoding="utf-8", errors="ignore") as stream:
        source = stream.read()
    # 始终生成临时副本并末尾追加 ``Mesh.SaveAll = 1;`` — 1-D 边界单元
    # (Physical Curve) 必须写入 .msh。零复制优化不可行: 多重 SaveAll
    # 赋值 (先 1 后 0, 生效的是 0) 无法通过正则静态判定, 曾导致边界
    # 单元丢失 (复测 2026-08-02); 一次小文件复制的成本远低于解析风险。
    sanitized = sanitize_geo_source(source)
    # The subprocess fallback must export 1-D boundary elements as well as the
    # 2-D displacement mesh. Append the option after user commands so an
    # earlier ``Mesh.SaveAll = 0`` cannot silently discard Physical Curves.
    sanitized = sanitized.rstrip() + (
        "\n\n// Enforced by FEM2D subprocess fallback\n"
        "Mesh.SaveAll = 1;\n")

    descriptor, temporary_geo = tempfile.mkstemp(
        prefix=".fem2d-gmsh-source-", suffix=".geo",
        dir=os.path.dirname(os.path.abspath(geo_path)))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(sanitized)
    except Exception:
        if os.path.isfile(temporary_geo):
            os.unlink(temporary_geo)
        raise
    return temporary_geo, temporary_geo


def run_gmsh(
        geo_path, quad=False, output_path=None, gmsh_exe=None,
        threads=None, defer_publish=False):
    """Generate and atomically publish a Gmsh ``.msh`` mesh.

    Returns the absolute output path on success and ``None`` on failure.
    An existing output file is preserved when Gmsh fails.

    ``defer_publish=True``: 不发布正式文件, 返回临时网格路径 — 由调用方
    (import_msh 拓扑验证通过后) 负责原子替换, 避免"验证失败时旧文件已被
    覆盖" (评审发现: 原流程先发布后验证)。
    """
    gmsh_exe = gmsh_exe or find_gmsh()
    if not gmsh_exe:
        suffix = " with quad recombination" if quad else ""
        print(f"[ERROR] Gmsh executable not found{suffix}: {geo_path}")
        print("        设 GMSH_PATH 环境变量, 或将 gmsh.exe 放入 "
              "tools/gmsh-4.15.2-Windows64/ (或 PATH)")
        return None
    if not os.path.isfile(geo_path):
        print(f"[Gmsh] geometry file not found: {geo_path}")
        return None

    final_path = os.path.abspath(
        output_path or os.path.splitext(geo_path)[0] + ".msh")
    output_dir = os.path.dirname(final_path)
    os.makedirs(output_dir, exist_ok=True)

    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".fem2d-gmsh-", suffix=".msh", dir=output_dir)
    os.close(descriptor)
    os.unlink(temporary_path)
    command_geo = geo_path
    temporary_geo = None
    handed_off = False  # defer_publish 成功移交调用方 — 临时文件不再由本函数清理
    try:
        command_geo, temporary_geo = _geometry_without_explicit_save(geo_path)
    except OSError as error:
        print(f"[Gmsh] cannot prepare geometry: {error}")
        return None
    if threads is None:
        configured_threads = os.environ.get(
            "FEM2D_GMSH_THREADS", "").strip()
        try:
            threads = (
                int(configured_threads)
                if configured_threads
                else min(os.cpu_count() or 1, 8)
            )
        except ValueError:
            print(
                "[Gmsh] invalid FEM2D_GMSH_THREADS="
                f"{configured_threads!r}; using automatic thread count")
            threads = min(os.cpu_count() or 1, 8)
    command = build_gmsh_command(
        gmsh_exe, command_geo, temporary_path,
        quad=quad, threads=threads)
    print(f"[Gmsh] {shlex.join(command)}")

    try:
        started = time.perf_counter()
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False,
            timeout=300,
            errors="replace")  # 畸形 .geo / Gmsh 卡死时快速失败; 中文路径
            # 下 gmsh 输出非 ASCII 字节曾 UnicodeDecodeError 冒泡掩盖真实
            # 失败信息 (审计 2026-08-03)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            print(f"[Gmsh] failed (exit {completed.returncode}):\n{detail}")
            return None
        if not os.path.isfile(temporary_path) or os.path.getsize(
                temporary_path) == 0:
            print("[Gmsh] failed: output file was not created or is empty")
            return None
        if defer_publish:
            # 调用方负责 import_msh 验证 + 原子发布 — 移交临时文件所有权
            handed_off = True
            print(
                f"[Gmsh] -> {temporary_path} (待验证发布)  "
                f"({time.perf_counter() - started:.2f} s)")
            return temporary_path
        os.replace(temporary_path, final_path)
        print(
            f"[Gmsh] -> {final_path}  "
            f"({time.perf_counter() - started:.2f} s)")
        return final_path
    except subprocess.TimeoutExpired:
        print("[Gmsh] timed out after 300 s — .geo 可能畸形或 Gmsh 卡死")
        return None
    except OSError as error:
        print(f"[Gmsh] execution failed: {error}")
        return None
    finally:
        # 只有"成功移交 defer_publish"或"已发布"两种情况才不清理;
        # Gmsh 部分写入后失败 (非零退出) 时临时文件必须删除 (评审发现)
        if not handed_off and os.path.isfile(temporary_path):
            os.unlink(temporary_path)
        if temporary_geo and os.path.isfile(temporary_geo):
            os.unlink(temporary_geo)
