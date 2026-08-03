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

# 本程序生成物的 .msh 标记 — 同名 .msh 覆盖保护据此识别 (gmsh 读回时
# 忽略 $Comments 段, 2026-08 实测确认 MSH 4.x 实体/物理组完整恢复).
_MSH_MARKER = "// FEM2D-generated-mesh"
_MSH_MARKER_BYTES = _MSH_MARKER.encode("ascii")


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
    共用, 曾双实现且正则常量各一份, 语义靠人工对齐。

    - ``Save`` 可绕过验证后的 ``-o`` 目标;
    - ``Mesh 2;`` 会在解析期 + 命令行 ``-2`` 各网格化一次;
    - 脚本内 ``Mesh.Format`` (如 39=Abaqus) 会覆盖命令行输出格式;
    - 脚本内 ``Mesh.MshFileVersion`` (如 2) 会让输出变成 v2.2 — 本程序
      v2.2 无法注入生成物标记 ($Comments 仅 4.x 支持), 覆盖保护失效。
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
    sanitized = re.sub(
        r"^\s*Mesh\s*\.\s*MshFileVersion\s*=\s*[^;]*;",
        "// Mesh.MshFileVersion removed by FEM2D; output version is 4.x",
        sanitized,
        flags=re.MULTILINE,
    )
    return sanitized


def stamp_generated_msh(path):
    """向生成物 .msh 注入本程序标记 ($Comments 段, gmsh 读回时忽略).

    仅当文件为 MSH 4.x ($MeshFormat 首行版本 4.x 且存在 $EndMeshFormat)
    时注入 — 缺失任一段说明文件残缺或非本程序产物, 不碰 (防御: 残缺
    文件不改写). 幂等: 已含标记直接返回.
    标记是"本程序生成物"的唯一可靠识别信号: 同名 .msh 覆盖保护据此
    决定覆盖 (带标记) 或 WARN + 临时副本 (无标记, 来源不明).
    字节级改写 (不按文本解码) — 物理组名含中文等非 ASCII 时无损.
    """
    try:
        with open(path, "rb") as stream:
            data = stream.read()
    except OSError:
        return False
    if _MSH_MARKER_BYTES in data:
        return False
    # 版本检查容忍 \r\n (Windows 文本模式写出的文件) — 曾 LF 严格导致
    # 全部 Windows 生成物无法识别
    if not re.match(
            rb"^\$MeshFormat\s*\r?\n4\.\d", data.lstrip()):
        return False
    header_end = data.find(b"$EndMeshFormat")
    if header_end < 0:
        return False
    line_end = data.find(b"\n", header_end) + 1
    marker = (b"$Comments\n" + _MSH_MARKER_BYTES
              + b"\n$EndComments\n")
    try:
        with open(path, "wb") as stream:
            stream.write(data[:line_end] + marker + data[line_end:])
    except OSError:
        return False
    return True


def is_program_generated_msh(path):
    """目标 .msh 是否本程序生成 (带 $Comments 标记) — 覆盖保护判定.

    标记位于文件头 $MeshFormat 段之后, 读前 128 KiB 足够.
    """
    try:
        with open(path, "rb") as stream:
            head = stream.read(1 << 17)
    except OSError:
        return False
    return _MSH_MARKER_BYTES in head


_INCLUDE_RE = re.compile(r'^\s*Include\s+"', re.MULTILINE)


def _has_relative_include(geo_path):
    """源 .geo 是否含相对 Include — 相对引用以 .geo 所在目录解析, 临时
    副本移到其他目录会断裂; 绝对 Include (盘符或 / 开头) 不受影响."""
    try:
        with open(geo_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except OSError:
        return False
    # 正则已消费开引号 — rest 从路径首字符开始
    for match in _INCLUDE_RE.finditer(source):
        rest = source[match.end():]
        if not rest:
            return True  # 无路径的残缺 Include — 视为相对 (副本会断裂)
        char = rest[0]
        if char in ("/", "\\"):
            continue  # 绝对路径 (POSIX / UNC)
        if char.isalpha() and len(rest) > 1 and rest[1] == ":":
            continue  # Windows 盘符绝对路径
        return True  # 相对路径
    return False


def temp_copy_dir(geo_path, requested_dir):
    """临时几何副本目录: --output-dir 指定时用该目录; 源 .geo 含相对
    Include 时必须留在源目录 (相对引用以所在目录解析, 曾因副本移位导致
    Include 断裂). 返回实际使用的目录."""
    base = os.path.dirname(os.path.abspath(geo_path))
    if requested_dir is None:
        return base
    requested = os.path.abspath(requested_dir)
    if requested == base:
        return base
    if _has_relative_include(geo_path):
        print(
            "  [WARN] .geo 含相对 Include — 临时几何副本必须留在源目录, "
            "--output-dir 仅作用于 .msh 输出 (请用绝对 Include 路径)")
        return base
    return requested


def _geometry_without_explicit_save(geo_path, temp_dir=None):
    """Return a source-safe geometry path and an optional cleanup path.

    A scripted ``Save`` can bypass the validated ``-o`` target. A scripted
    ``Mesh 2;`` would also mesh once while parsing and again for the command
    line ``-2`` option. Strip both standalone directives in a temporary
    sibling copy, leaving the user's geometry untouched and relative includes
    anchored to the same directory. ``temp_dir`` (--output-dir) 移动副本
    位置; 含相对 Include 时由 :func:`temp_copy_dir` 强制留在源目录.
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

    copy_dir = temp_copy_dir(geo_path, temp_dir)
    descriptor, temporary_geo = tempfile.mkstemp(
        prefix=".fem2d-gmsh-source-", suffix=".geo",
        dir=copy_dir)
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
        threads=None, defer_publish=False, temp_dir=None):
    """Generate and atomically publish a Gmsh ``.msh`` mesh.

    Returns the absolute output path on success and ``None`` on failure.
    An existing output file is preserved when Gmsh fails.

    ``defer_publish=True``: 不发布正式文件, 返回临时网格路径 — 由调用方
    (import_msh 拓扑验证通过后) 负责原子替换, 避免"验证失败时旧文件已被
    覆盖" (评审发现: 原流程先发布后验证)。

    ``temp_dir`` (--output-dir): 临时几何副本的位置 — 默认与源 .geo 同
    目录 (相对 Include 锚定); 指定时移入该目录 (含相对 Include 的 .geo
    由 :func:`temp_copy_dir` 强制留在源目录)。生成的 .msh 在发布前注入
    本程序生成物标记 (``stamp_generated_msh``), 供同名覆盖保护识别。
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
        command_geo, temporary_geo = _geometry_without_explicit_save(
            geo_path, temp_dir=temp_dir)
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
            # 失败信息
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            print(f"[Gmsh] failed (exit {completed.returncode}):\n{detail}")
            return None
        if not os.path.isfile(temporary_path) or os.path.getsize(
                temporary_path) == 0:
            print("[Gmsh] failed: output file was not created or is empty")
            return None
        # 生成物标记: 覆盖保护 (is_program_generated_msh) 的唯一识别信号。
        # 只处理 MSH 4.x — 残缺/低版本文件不改写 (stamp 内部防御).
        stamp_generated_msh(temporary_path)
        if defer_publish:
            # 调用方负责 import_msh 验证 + 原子发布 — 移交临时文件所有权
            handed_off = True
            print(
                f"[Gmsh] -> {temporary_path} (待验证发布)  "
                f"({time.perf_counter() - started:.2f} s)")
            return temporary_path
        try:
            os.replace(temporary_path, final_path)
        except OSError as error:
            # 只读目录/目标文件 — 裸 PermissionError 会掩盖"输出位置"这一
            # 真实根因, 用户无从得知可用 --output-dir 解决
            print(
                f"[ERROR] 输出目录不可写: {output_dir} — "
                f"请用 --output-dir 指定可写位置 ({error})")
            return None
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
