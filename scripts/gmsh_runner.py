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

from fem2d.errors import GeoScriptRejected

_EXPLICIT_SAVE_RE = re.compile(
    r"\bSave\s+\"[^\"]+\"\s*;",
    re.IGNORECASE,
)
_EXPLICIT_MESH_RE = re.compile(
    r"\bMesh\s+\d+\s*;",
    re.IGNORECASE,
)
# SystemCall 是 Gmsh 脚本中唯一能执行任意系统命令的指令 — 黑名单拦截。
# 检测在"注释剥离后"的任意位置匹配 — 行首正则 (r"^\s*SystemCall\b") 曾可被
# 同行多语句 ('x = 1; SystemCall "whoami";') 或块注释后紧跟
# ('/* c */ SystemCall "whoami";') 绕过。注释先剥离再匹配 → 注释内的
# SystemCall 天然不触发 (Gmsh 解析器忽略注释, 行为保持)。
_SYSTEMCALL_TOKEN_RE = re.compile(r"\bSystemCall\b", re.IGNORECASE)
# Include 指令识别 (只认代码位 — 注释/字符串内的 Include 不是指令;
# 字符串无转义, 引号即字符串边界)
_INCLUDE_DIRECTIVE_RE = re.compile(r'\bInclude\s*"([^"]*)"', re.IGNORECASE)
# Merge "*.geo" 与 Include 同为"解析并执行被引用 .geo 脚本"的执行面 —
# 审查实证: main.geo 写 ``Merge "evil.geo";`` 时 gmsh 真实调用 evil.geo
# 内的 SystemCall, 拦截被完整绕过。扫描必须沿 Merge 树同样展开。
# 只追 .geo 目标: Merge 对 .step/.brep 等按扩展名走 CAD 导入, 不执行
# 脚本 — 扫描它们反而会把 CAD 文本里 (如 STEP 产品名) 的 SystemCall
# 字样误拒为 RCE (扩展名分发与 Gmsh 一致; 一个 Merge 语句一个文件)。
_MERGE_DIRECTIVE_RE = re.compile(r'\bMerge\s*"([^"]*)"', re.IGNORECASE)


def _mask_geo_comments(source):
    """逐字符词法扫描: 注释 (// 行注释与 /* */ 块注释) → 空格, 双引号
    字符串内容保留原样。换行保留 → 输出与原文等长、换行位置一致 (匹配
    位置可直接反推原文行号)。

    逐字符而非正则: 双引号字符串内的 // 与 /* 是普通字符, 正则无法区分
    (必须维护 in_string 状态)。Gmsh 字符串无转义字符; 未闭合字符串按
    "到行尾结束"处理 (保守, 不猜错注释边界)。

    字符串内容保留: SystemCall 检测要求字符串参数含 SystemCall 字样也
    命中 (保守拒绝 — 安全优先, 无法可靠区分"字符串参数"与真正调用,
    误拒纯字符串场景 (echo "SystemCall") 的代价远低于漏放 RCE);
    Include 检测同样依赖字符串内容保留 (指令的双引号是语法一部分,
    屏蔽后指令本身会被吞掉 — 曾导致全部 Include 检测失配)。
    """
    out = []
    state = "code"  # code / line (//) / block (/* */) / string ("...")
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        consumed = 1
        if state == "line":
            out.append(ch if ch == "\n" else " ")
            if ch == "\n":
                state = "code"
        elif state == "block":
            if ch == "*" and nxt == "/":
                out.append("  ")
                consumed = 2
                state = "code"
            else:
                out.append(ch if ch == "\n" else " ")
        elif state == "string":
            if ch == "\n":
                out.append("\n")  # 未闭合字符串 → 到行尾结束 (保守)
                state = "code"
            else:
                out.append(ch)
        elif ch == '"':
            out.append('"')
            state = "string"
        elif ch == "/" and nxt == "/":
            out.append("  ")
            consumed = 2
            state = "line"
        elif ch == "/" and nxt == "*":
            out.append("  ")
            consumed = 2
            state = "block"
        else:
            out.append(ch)
        i += consumed
    return "".join(out)


def _check_systemcall_text(source, label):
    """注释剥离后任意位置搜 ``\\bSystemCall\\b`` — 命中抛 GeoScriptRejected.

    ``label`` 带文件定位 (如 ".geo" 或 ".geo (Include 链: a.geo → b.geo)");
    行号由屏蔽文本位置反推 (等长等换行 = 原文行号)。
    """
    match = _SYSTEMCALL_TOKEN_RE.search(_mask_geo_comments(source))
    if match is None:
        return
    line_no = source.count("\n", 0, match.start()) + 1
    lines = source.splitlines()
    line_text = lines[line_no - 1].strip() if lines else ""
    raise GeoScriptRejected(
        f"{label} 第 {line_no} 行含被禁止的 SystemCall 指令: "
        f"{line_text!r} — SystemCall 会执行任意系统命令. "
        ".geo 是可信、可执行式输入, 只应运行自己编写的文件 "
        "(含 Include 引用的文件).")


def _iter_geo_includes(source, geo_path):
    """识别代码位 ``Include "path"`` / ``Merge "path.geo"`` 指令 → 绝对路径。

    在注释剥离文本上识别 (注释里的引用不是指令)。字符串内容保留且 Gmsh
    字符串无转义、不可能内含双引号 — 字符串内的 ``Include "`` 只会产生
    空路径 (跳过); 真正带路径的引用必在代码位。相对路径基于 ``geo_path``
    所在目录解析; 空路径引用跳过 (交给 Gmsh 解析时报错)。Merge 只对
    .geo 目标执行脚本 (见 _MERGE_DIRECTIVE_RE 的"为什么")。
    """
    base_dir = os.path.dirname(geo_path)
    masked = _mask_geo_comments(source)
    for match in _INCLUDE_DIRECTIVE_RE.finditer(masked):
        target = match.group(1).strip()
        if target:
            yield _resolve_geo_ref(base_dir, target)
    for match in _MERGE_DIRECTIVE_RE.finditer(masked):
        target = match.group(1).strip()
        if not target or os.path.splitext(target)[1].lower() != ".geo":
            continue
        yield _resolve_geo_ref(base_dir, target)


def _resolve_geo_ref(base_dir, target):
    if os.path.isabs(target):
        return target
    return os.path.normpath(os.path.join(base_dir, target))


def _scan_include_tree(geo_path, *, done=None, active=None, chain=()):
    """递归扫描引用树 (Include + Merge "*.geo") — 每个文件执行同样的
    "剥离注释 + SystemCall 检测", 命中即拒绝。

    - 相对引用基于被引文件所在目录解析;
    - ``active`` = 当前递归链: 命中 = 循环引用 → 拒绝并报引用链
      (如 "Include/Merge 循环引用: a.geo → b.geo → a.geo");
    - ``done`` = 已扫描完成: 钻石形共享引用 (a→b, a→c, b→d, c→d) 只扫一次;
    - 引用目标文件不存在 → 跳过不报错 (保持现有行为: Gmsh 解析时
      报错; test_output_dir_policy 依赖"缺失 Include 不拦截")。
    """
    geo_path = os.path.abspath(geo_path)
    key = os.path.normcase(geo_path)  # Windows 路径大小写不敏感
    if done is None:
        done, active = set(), set()
    if key in active:
        loop = " → ".join(chain + (os.path.basename(geo_path),))
        raise GeoScriptRejected(f"Include/Merge 循环引用: {loop}")
    if key in done:
        return
    active.add(key)
    chain = chain + (os.path.basename(geo_path),)
    try:
        with open(geo_path, "r", encoding="utf-8", errors="ignore") as stream:
            source = stream.read()
    except FileNotFoundError:
        active.discard(key)
        return  # 引用目标不存在 → 跳过不报错 (现有行为保持: Gmsh 解析时报错)
    except OSError:
        active.discard(key)
        raise GeoScriptRejected(
            f"引用文件无法读取: {geo_path}") from None
    _check_systemcall_text(
        source, f".geo (引用链: {' → '.join(chain)})")
    for child in _iter_geo_includes(source, geo_path):
        _scan_include_tree(child, done=done, active=active, chain=chain)
    active.discard(key)
    done.add(key)

# 本程序生成物的 .msh 标记 — 同名 .msh 覆盖保护据此识别 (gmsh 读回时
# 忽略 $Comments 段, 2026-08 实测确认 MSH 4.x 实体/物理组完整恢复).
# 标记行可带 "lc=<值>" 密度元数据: 复用判据 (input_source._try_reuse_msh)
# 据此拒绝"同名同 mtime 但密度不符"的过期网格; 前缀扩展向后兼容 —
# is_program_generated_msh 用子串匹配, 带/不带 lc 的标记都命中.
_MSH_MARKER = "// FEM2D-generated-mesh"
_MSH_MARKER_BYTES = _MSH_MARKER.encode("ascii")
# 标记行中的密度值 (lc=...) — 旧版无密度标记/外来无标记 msh 不匹配 → None
_MSH_LC_RE = re.compile(rb"// FEM2D-generated-mesh lc=([\d.eE+\-]+)")
# 几何脚本首条 lc 赋值 (同 input_source._LC_PATTERN — 脚本层不复用
# fem2d 模块, 避免 scripts→fem2d 反向依赖)
_MSH_SRC_LC_RE = re.compile(r'^\s*lc\s*=\s*([\d.eE+\-]+)', re.MULTILINE)


def _geo_lc_from(geo_path):
    """几何脚本首条 lc 赋值 (无/不可读 → None) — 写入生成物密度标记.

    解析实际交给 gmsh 运行的脚本 (含 lc 覆盖临时副本), 标记值即
    本次网格化的真实密度 — CLI --lc 覆盖自动反映, 无需穿透调用层.
    """
    try:
        with open(geo_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    m = _MSH_SRC_LC_RE.search(text)
    return float(m.group(1)) if m else None


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


def sanitize_geo_source(source, *, geo_path=None):
    """剥离 .geo 脚本中的 Save / Mesh 命令 / Mesh.Format 赋值 (纯文本).

    唯一实现 — gmsh_adapter._safe_geo_source (API 路径) 与子进程路径
    共用, 曾双实现且正则常量各一份, 语义靠人工对齐。

    - ``Save`` 可绕过验证后的 ``-o`` 目标;
    - ``Mesh 2;`` 会在解析期 + 命令行 ``-2`` 各网格化一次;
    - 脚本内 ``Mesh.Format`` (如 39=Abaqus) 会覆盖命令行输出格式;
    - 脚本内 ``Mesh.MshFileVersion`` (如 2) 会让输出变成 v2.2 — 本程序
      v2.2 无法注入生成物标记 ($Comments 仅 4.x 支持), 覆盖保护失效;
    - ``SystemCall`` 可执行任意系统命令 — 黑名单拒绝整个脚本 (RCE 面;
      .geo 是"可信、可执行式输入", 只应运行自己编写的文件)。

    ``geo_path`` (源文件路径, 可选): 提供时连同 Include/Merge 引用树一起
    递归扫描 (SystemCall 检测 + 循环引用检测); 仅文本调用时只查
    ``source`` 本身 (相对路径无法解析)。
    """
    # 顺序: 先 SystemCall 安全检查 (拒绝即抛, 不再往下), 再做替换 —
    # 替换注入的是注释文本且不含 SystemCall 字样, 顺序对检测结果无影响。
    _check_systemcall_text(source, ".geo")
    if geo_path is not None:
        _scan_include_tree(geo_path)
    sanitized = _EXPLICIT_SAVE_RE.sub(
        "// Save removed by FEM2D; FEM2D owns publication", source)
    sanitized = _EXPLICIT_MESH_RE.sub(
        "// Mesh command removed by FEM2D; FEM2D meshes once",
        sanitized,
    )
    # Mesh.Format / Mesh.MshFileVersion 剥离无行首锚点 — 旧 ``^\s*`` 可被
    # 同行前导语句 ('x = 1; Mesh.Format = 39;') 完全绕过, 与 Save/Mesh 2
    # 的 ``\b`` 匹配不一致, 同类指令应同款处理。值部分排除引号/换行:
    # [^;]* 会把字符串 'Print("Mesh.Format =");' 的闭合引号吞进值里造成
    # 引号失衡 (Gmsh 字符串无转义, 值表达式不会含引号)。大小写不敏感 —
    # Gmsh 关键字不区分大小写 (与 SystemCall 拦截同策略)。
    sanitized = re.sub(
        r"\bMesh\s*\.\s*Format\s*=\s*[^;\"\n]*;",
        "// Mesh.Format removed by FEM2D; output is native .msh",
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"\bMesh\s*\.\s*MshFileVersion\s*=\s*[^;\"\n]*;",
        "// Mesh.MshFileVersion removed by FEM2D; output version is 4.x",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


def stamp_generated_msh(path, lc=None):
    """向生成物 .msh 注入本程序标记 ($Comments 段, gmsh 读回时忽略).

    ``lc``: 本次网格化的密度 (None = 不写密度元数据) — 写入标记行
    "lc=<值>", 复用判据据此拒绝"同名同 mtime 但密度不符"的过期网格.
    仅当文件为 MSH 4.x ($MeshFormat 首行版本 4.x 且存在 $EndMeshFormat)
    时注入 — 缺失任一段说明文件残缺或非本程序产物, 不碰 (防御: 残缺
    文件不改写). 幂等: 已含标记直接返回 (含旧版无密度标记 — 生成时
    只写一次, 不做补写升级).
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
    marker_line = _MSH_MARKER_BYTES + (
        f" lc={lc:g}".encode("ascii") if lc is not None else b"")
    marker = b"$Comments\n" + marker_line + b"\n$EndComments\n"
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
# Merge 的相对引用同 Include 会因副本移位断裂 — 同判据纳入
# (不区分扩展名: .step 等非脚本目标的相对 Merge 同样依赖所在目录)
_MERGE_REF_RE = re.compile(r'\bMerge\s*"', re.IGNORECASE)


def _has_relative_include(geo_path):
    """源 .geo 是否含相对 Include/Merge 引用 — 相对引用以 .geo 所在目录
    解析, 临时副本移到其他目录会断裂; 绝对引用 (盘符或 / 开头) 不受影响.
    副本目录若落有同名文件, 未扫描的相对引用可能指向它 — 因此相对引用
    强制留在源目录 (引用解析到已扫描的原始文件)."""
    try:
        with open(geo_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except OSError:
        return False
    for regex in (_INCLUDE_RE, _MERGE_REF_RE):
        for match in regex.finditer(source):
            if _is_relative_ref(source[match.end():]):
                return True
    return False


def _is_relative_ref(rest):
    """正则已消费开引号 — rest 从路径首字符开始."""
    if not rest:
        return True  # 无路径的残缺引用 — 视为相对 (副本会断裂)
    char = rest[0]
    if char in ("/", "\\"):
        return False  # 绝对路径 (POSIX / UNC)
    if char.isalpha() and len(rest) > 1 and rest[1] == ":":
        return False  # Windows 盘符绝对路径
    return True  # 相对路径


def temp_copy_dir(geo_path, requested_dir):
    """临时几何副本目录: --output-dir 指定时用该目录; 源 .geo 含相对
    Include/Merge 引用时必须留在源目录 (相对引用以所在目录解析, 曾因
    副本移位导致 Include 断裂). 返回实际使用的目录."""
    base = os.path.dirname(os.path.abspath(geo_path))
    if requested_dir is None:
        return base
    requested = os.path.abspath(requested_dir)
    if requested == base:
        return base
    if _has_relative_include(geo_path):
        print(
            "  [WARN] .geo 含相对 Include/Merge 引用 — 临时几何副本必须留在源目录, "
            "--output-dir 仅作用于 .msh 输出 (请用绝对路径引用)")
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
    sanitized = sanitize_geo_source(source, geo_path=geo_path)
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
        # lc 密度元数据解析自实际运行的脚本 (含 CLI 覆盖临时副本) —
        # 复用判据据此拒绝"同名同 mtime 但密度不符"的过期网格.
        stamp_generated_msh(temporary_path, lc=_geo_lc_from(command_geo))
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
