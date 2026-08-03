"""输入源解析 — 把任意输入 (.spec / .geo / .txt / .msh) 统一为网格导入链路。

原 run.py 的"拿到网格文件"阶段: 处理 .spec 配置覆盖、.geo 的 lc 交互
与临时副本 (不修改原始文件)、.txt 中文描述 → .geo 生成, 以及子进程
Gmsh 路径下 --force Physical Point 的坐标回退。语义保持与 run.py 一致:
参数优先级 程序默认 < 配置 < CLI 显式参数。

gmsh 调用 (run_gmsh) 是唯一的第三方工具依赖, 位于 ``scripts/`` 工具层。
"""
import atexit
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import AnalysisConfig
from .errors import CliError

# scripts/ 是工具脚本层 (geo_spec / gmsh_runner), 不在 fem2d 包内。
# 注入项目根, 使 ``import scripts.geo_spec`` (namespace package) 在
# 库方式调用下也可用 —— 与原 run.py 顶部的路径注入行为一致。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


@dataclass
class ResolvedInput:
    """输入解析结果 — 求解链路只需要这几个字段。"""

    fp: str                              # 最终 .msh 路径
    gmsh_import: Optional[object]        # Gmsh API 导入结果 (无 → None)
    source_geo_path: Optional[str]       # 源 .geo 路径 (无 → None)
    quad_applied: bool                   # 是否已执行四边形重组
    geo_config_applied: bool = False     # @FEM 配置是否已在输入解析阶段合并


def generate_geo_with_topology(
        geo_path, *, quad=False, output_path=None, plane_type="stress"):
    """Mesh one ``.geo`` file with the native Gmsh executable.

    The standalone process avoids the Windows stack limit that can affect
    high-density quadrilateral recombination through the Python API. The
    generated ``.msh`` (原生 Gmsh 格式, 含 $PhysicalNames/$Entities) 由
    ``import_msh`` 用 Gmsh API 读回, CAD 与物理组语义完整恢复 —
    不再需要 Abaqus .inp 中间格式 (2026-08: 移除 Abaqus 输入口)。

    发布顺序: 临时生成 → import_msh 拓扑验证 → 原子替换正式文件.
    验证失败 (混合三角/四边、quad 重组不完整) 时不发布, 旧文件保留
    (评审发现: 原流程先发布后验证, 非法新网格会覆盖旧文件)。

    异常契约 (外部审查, 2026-08-03): gmsh 生成失败 (可执行文件缺失/
    退出码非零) → 返回 ``(None, None)``; 拓扑验证失败 (网格生成成功但
    非法) → 抛异常。调用方必须同时处理 None 与异常 — 测试层应
    检查 None 后 skip/报错, 不应在 None 上解包。
    """
    # quad 重组在 gmsh Blossom 下非确定性 (8 线程实测 ~75% 成功, 其余
    # 产出混合三角/四边) — 拓扑验证失败自动重试, 重试 2 次成功率 >98%
    # (高强度审计 2026-08-02)。
    max_attempts = 3 if quad else 1
    last_error = None
    for attempt in range(max_attempts):
        from scripts.geo_spec import run_gmsh
        generated = run_gmsh(
            geo_path, quad=quad, output_path=output_path, defer_publish=True)
        if not generated:
            return None, None
        from fem2d.gmsh_adapter import import_msh
        try:
            gmsh_import = import_msh(
                generated, require_quads=quad, plane_type=plane_type)
        except Exception as error:
            last_error = error
            # 拓扑验证失败 — 不发布, 清理临时网格, 保留旧文件
            try:
                os.unlink(generated)
            except OSError:
                pass
            if attempt < max_attempts - 1:
                print(f"[Gmsh] quad 重组验证失败 (第 {attempt + 1} 次) — 重试...")
                continue
            raise
        final_path = os.path.abspath(
            output_path or os.path.splitext(os.path.abspath(geo_path))[0]
            + ".msh")
        os.replace(generated, final_path)
        print(f"[Gmsh] published -> {final_path}")
        return final_path, gmsh_import
    raise last_error  # 理论不可达 (循环内已 raise)


def physical_point_from_geo(geo_path, name, mesh):
    """子进程 Gmsh 路径的 --force Physical Point 回退.

    默认 CLI 走子进程 gmsh (gmsh_runner), 该路径不恢复 Physical Point
    regions — 名称只在 Gmsh API 路径可用。这里用 Gmsh API 只读一次
    ``.geo``, 取同名 Physical Point 的 CAD 坐标并匹配最近的网格节点。
    注意: 边线中间的 Point 不保证有网格节点 (偏差可达单元尺寸量级),
    偏差由调用方打印警告。

    域判定 (审计 2026-08): 仅 AABB 包含不足以排除凹域/孔洞内但不属于
    实体的 construction point — 点必须落在某个单元内才接受, 且最近节点
    距离不得超过典型单元边长的 3 倍 (再远说明该点不属于此网格)。

    返回 ``(nid, label, dist, reason)``; 成功时 reason=None。
    失败时 nid/label/dist 为 None, reason 区分失败原因
    (not_found / ambiguous / outside_domain / too_far / gmsh_unavailable
     / no_geo_source) — 曾统一 (None,None,None) 让调用方把歧义/超距也
    报成"未找到" (审计 2026-08-03)。
    """
    if not geo_path:
        # .msh 直接输入无源 .geo — 曾 _safe_geo_source(None) 抛 TypeError
        # 被宽 except 吞掉, 误报 "Gmsh API 不可用" (审计 2026-08-03)
        return None, None, None, "no_geo_source"
    try:
        from fem2d.gmsh_adapter import _load_gmsh_module, _safe_geo_source
        gmsh_module = _load_gmsh_module()
    except Exception:
        return None, None, None, "gmsh_unavailable"
    initialized = bool(
        gmsh_module.isInitialized()
        if hasattr(gmsh_module, "isInitialized") else False)
    owns_session = not initialized
    command_geo = None
    temporary_geo = None
    try:
        if owns_session:
            gmsh_module.initialize()
        # 用 stripped 副本打开: .geo 里的 Mesh/Save 命令若直接执行会
        # 静默覆盖同名 .inp 并丢失 source binding (曾复现: 读 l_bracket
        # 的 .inp 时 sibling .geo 被 API 打开, l_bracket.inp 被重写).
        command_geo, temporary_geo = _safe_geo_source(geo_path)
        gmsh_module.open(command_geo)
        gmsh_module.model.geo.synchronize()
        found = []
        for dim, tag in gmsh_module.model.getPhysicalGroups():
            if int(dim) != 0:
                continue
            if str(gmsh_module.model.getPhysicalName(0, int(tag))).strip() != name:
                continue
            for entity in gmsh_module.model.getEntitiesForPhysicalGroup(
                    0, int(tag)):
                try:
                    coords = gmsh_module.model.getValue(0, int(entity), [])
                    found.append(np.asarray(coords[:2], dtype=float))
                except Exception:  # nosec B112 — 坏实体坐标读取失败, 跳过 (循环继续)
                    continue
        if len(found) != 1:
            return None, None, None, (
                "not_found" if not found else "ambiguous")
        nodes = np.asarray(mesh.nodes, dtype=float)
        # 一级过滤: AABB 包围盒 (快筛, 拒掉明显域外点)
        span = max(
            float(np.ptp(nodes[:, 0])), float(np.ptp(nodes[:, 1])),
            np.finfo(float).tiny)
        slack = span * 1e-6
        inside = (
            nodes[:, 0].min() - slack <= found[0][0] <= nodes[:, 0].max() + slack
            and nodes[:, 1].min() - slack <= found[0][1] <= nodes[:, 1].max() + slack)
        if not inside:
            return None, None, None, "outside_domain"
        # 二级过滤: 真实域包含 — 点必须落在某个单元内。凹域/孔洞内但
        # 不属于实体的 construction point 会在 AABB 内却不属于任何单元,
        # 旧实现会把它施加到任意最近节点 (审计 2026-08)。
        from fem2d.stress import point_in_element
        if point_in_element(mesh, found[0][0], found[0][1]) < 0:
            return None, None, None, "outside_domain"
        dist = np.linalg.norm(nodes - found[0], axis=1)
        nid = int(np.argmin(dist))
        # 三级过滤: 距离阈值与局部单元尺寸关联 — 边中点 Physical Point
        # 偏差可达 ~1 单元尺寸, 3 倍为安全裕量; 超过则拒绝报错。
        # 特征尺寸 = 每单元 x/y 跨度的最大值的中位数 (axis=1 沿节点轴,
        # axis=2 会得到 |x−y| 而非跨度, 旋转细长单元上被严重低估)
        elem_span = np.ptp(nodes[mesh.elements], axis=1)   # (n_elem, 2)
        h_char = float(np.median(np.max(elem_span, axis=1)))
        if dist[nid] > max(3.0 * h_char, np.finfo(float).tiny):
            return None, None, None, "too_far"
        return nid, f"Physical Point '{name}'", float(dist[nid]), None
    except Exception:
        return None, None, None, "gmsh_unavailable"
    finally:
        if owns_session:
            gmsh_module.finalize()
        if temporary_geo is not None and os.path.isfile(temporary_geo):
            try:
                os.unlink(temporary_geo)
            except OSError:
                pass


# .spec 键 → AnalysisConfig 字段映射 (单一映射表 — 新增字段只需加一行).
# "未指定"语义: 数值/字符串字段 config 为 None 表示未指定; 布尔字段为 False.
_SPEC_FIELD_MAP = {
    "lc": "lc", "E": "E", "nu": "nu", "t": "thickness",
    "plane": "plane", "body": "body", "fix": "fix",
    "fix_ux": "fix_ux", "fix_uy": "fix_uy",
    "traction": "traction", "force": "force",
    "save": "save", "no_plot": "no_plot",
}
_SPEC_FLOAT_FIELDS = {"lc", "E", "nu", "t"}


def resolve_spec_overrides(fp, config):
    """.spec → 解析后取 mesh 路径 + 参数覆盖 (相对路径以 .spec 所在目录为基准).

    .spec 值仅覆盖 CLI 未显式指定的参数 (config 字段保持程序默认
    表示未指定)。
    优先级: 程序默认 < .spec < CLI 显式参数。返回最终的网格文件路径。
    """
    from fem2d.preprocess import parse_spec_config
    spec = parse_spec_config(fp)
    spec_dir = os.path.dirname(os.path.abspath(fp))
    mesh_raw = spec.get('mesh', '')
    if os.path.isabs(mesh_raw):
        mesh_fp = mesh_raw
    else:
        mesh_fp = os.path.join(spec_dir, mesh_raw)
        if not os.path.isfile(mesh_fp):
            mesh_fp = mesh_raw  # 回退到 CWD 相对路径

    defaults = AnalysisConfig()
    explicit = getattr(config, "_explicit", frozenset())
    for spec_key, field in _SPEC_FIELD_MAP.items():
        if spec_key not in spec:
            continue
        if field in explicit:
            continue  # CLI 显式优先 — .spec 不覆盖
        if getattr(config, field) != getattr(defaults, field):
            continue  # 程序默认已被覆盖 (直接构造的显式赋值)
        value = spec[spec_key]
        if field == "plane" and value not in ("stress", "strain"):
            raise CliError(
                f"  [FATAL] .spec 中 plane='{value}' — 仅支持 stress 或 strain",
                exit_code=1)
        if field == "no_plot":
            # 接受 1/0/true/false/yes/no (曾只认字面 "true"; 其他字符串
            # 静默当 False, 拼写错误不易发现 — 审计 2026-08-03 白名单)
            text = str(value).strip().lower()
            if text in ("1", "true", "yes"):
                value = True
            elif text in ("0", "false", "no", ""):
                value = False
            else:
                raise ValueError(
                    f".spec no_plot = {value!r} — 仅接受 "
                    "1/0/true/false/yes/no")
        elif spec_key in _SPEC_FLOAT_FIELDS:
            try:
                value = float(value)
            except ValueError:
                # 'E = 2,1e11' 曾报无键名上下文的裸 ValueError
                # (审计 2026-08-03)
                raise ValueError(
                    f".spec 键 '{spec_key}' 值 {value!r} 无法解析为数值 — "
                    f"期望十进制数字 (如 2.1e11), 不能用逗号作小数分隔符") from None
        setattr(config, field, value)

    for spec_key in spec:
        if spec_key not in _SPEC_FIELD_MAP and spec_key != "mesh":
            # 未知键曾静默忽略: 拼错 traction/force 等键名时载荷不生效,
            # 求解照常"成功" (审计 2026-08-03)
            print(f"  [WARN] .spec 键 '{spec_key}' 不被识别, 已忽略 — "
                  f"可用键: {sorted(_SPEC_FIELD_MAP) + ['mesh']}")

    # 合并发生在 AnalysisConfig 构造之后 — 重新校验, 防 .spec 非法值
    # (负 E/非法 ν/t) 绕过构造校验进入求解。
    config.validate()

    # .spec 解析完成后检查目标网格文件是否存在
    if not os.path.isfile(mesh_fp):
        raise CliError(
            f'[FATAL] .spec 指定的网格文件不存在: {mesh_fp}\n'
            f'  .spec 目录: {spec_dir}',
            exit_code=1)
    return mesh_fp


# lc 赋值行: 允许前导空白 (" lc = 0.5" 也必须命中 — 曾只匹配行首紧贴,
# 导致报告"已修改"而临时文件实际未变)。
_LC_PATTERN = r'^\s*lc\s*=\s*([\d.eE+\-]+)'
_LC_SUB_PATTERN = r'^\s*lc\s*=\s*[\d.eE+\-]+'


def _resolve_geo_lc(fp, config, ask):
    """.geo 网格密度: CLI --lc > 交互输入 > .geo 当前值.

    修改 lc 时创建临时副本 (不碰原始 .geo), 退出时自动清理。
    找不到 lc 变量时明确警告 (曾静默无效); 替换未生效时不创建副本。
    返回 (实际使用的 .geo 路径, 临时副本路径或 None)。
    """
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        geo_text = f.read()
    from .cli import is_batch_mode
    batch = is_batch_mode(config)
    m = re.search(_LC_PATTERN, geo_text, re.MULTILINE)
    if m is None:
        # 无 lc 变量 — 覆盖请求无法兑现, 明确告知而非假装修改。
        # 交互判定与 bc_apply 统一用 is_batch_mode — 曾手写条件漏掉
        # fix_ux/fix_uy, 批处理下静默不提示 (审计 2026-08-03)
        if config.lc is not None or not batch:
            print(
                "  [WARN] .geo 中未找到 'lc' 赋值行 (例如 'lc = 0.1;'), "
                "无法覆盖网格密度 — 使用文件内默认值。")
        return fp, None
    current_lc = float(m.group(1))

    if config.lc is not None:
        new_lc = config.lc
    else:
        lc_str = "" if batch else ask(
            f"  网格密度 lc [当前={current_lc}]: ")
        new_lc = float(lc_str) if lc_str else None

    # 精确相等才视为未更改 — 曾 1e-15 容差使微尺度 lc (如 2e-16 vs 1e-16)
    # 的覆盖被静默忽略 (审计 2026-08-03)
    if new_lc is None or new_lc == current_lc:
        return fp, None
    # 只替换第一个 lc 赋值 (多 lc 变量几何曾被全文替换一起覆盖;
    # 高强度审计 2026-08-02), 其余 lc 行保留原值并警告
    lc_lines = re.findall(_LC_PATTERN, geo_text, re.MULTILINE)
    if len(lc_lines) > 1:
        print(
            f"  [WARN] .geo 含 {len(lc_lines)} 个 'lc' 赋值 — 仅覆盖第一个, "
            "其余保持原值")
    new_text = re.sub(
        _LC_SUB_PATTERN, f'lc = {new_lc}', geo_text, count=1,
        flags=re.MULTILINE)
    if new_text == geo_text:
        # 理论上不可达 (模式已匹配), 防御: 不创建无意义的副本
        return fp, None
    orig_dir = os.path.dirname(os.path.abspath(fp))
    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.geo', delete=False,
        encoding='utf-8', dir=orig_dir)
    tmp.write(new_text)
    tmp.close()
    atexit.register(lambda p=tmp.name: os.path.isfile(p) and os.unlink(p))
    print(f"  lc: {current_lc} → {new_lc} (临时副本: {tmp.name})")
    return tmp.name, tmp.name


def resolve_geo(fp, config, ask=None):
    """.geo → 询问网格密度 (CLI 全参数时跳过), 跑 Gmsh 生成 .msh.

    返回 (msh 路径, gmsh_import, 源 .geo 路径)。
    """
    if ask is None:
        from fem2d.cli import ask as default_ask
        ask = default_ask
    source_geo_path = os.path.abspath(fp)

    # 先解析 @FEM: 注解 — 合并逻辑与 runner._apply_geo_fem_config 共用
    # (CLI 显式参数 > 配置; 配置内部 traction+pressure 可并存)。
    # 在此一次性合并并打印 [auto]; runner 对 .geo/.txt 输入不再二次解析
    # (曾二次解析把自动写入的载荷误判为 CLI 显式, 产生错误 WARN)。
    from fem2d.preprocess import merge_geo_fem_config, parse_geo_fem_config
    merge_geo_fem_config(parse_geo_fem_config(fp), config, verbose=True)

    gmsh_geo, temp_geo = _resolve_geo_lc(fp, config, ask)
    target_msh = os.path.splitext(os.path.abspath(fp))[0] + '.msh'
    msh = None
    try:
        msh, gmsh_import = generate_geo_with_topology(
            gmsh_geo,
            quad=config.quad,
            output_path=target_msh,
            plane_type=config.plane or 'stress',
        )
    finally:
        if temp_geo is not None and os.path.isfile(temp_geo):
            try:
                os.unlink(temp_geo)
            except OSError:
                pass
    if not msh:
        raise CliError(
            f"Gmsh 网格生成失败: {fp} — 退出码非零或产物缺失",
            exit_code=1)
    return msh, gmsh_import, source_geo_path


def resolve_txt(fp, config):
    """.txt 中文描述 → .geo → Gmsh 生成 .msh. 返回 (msh, gmsh_import, geo 路径)."""
    from scripts.geo_spec import generate_geo, parse_spec
    geo_p = os.path.splitext(fp)[0] + '.geo'
    if os.path.isfile(geo_p):
        is_generated = False
        try:
            with open(geo_p, encoding='utf-8', errors='ignore') as fh:
                is_generated = 'Auto-generated by geo_spec.py' in fh.read(256)
        except OSError:
            pass
        if is_generated:
            # 同名 .geo 是以前 .txt 的生成物 — 覆盖无损失, 但明确提示
            print(f"  [WARN] 覆盖已生成的 {geo_p} — 内容以 .txt 为准")
        else:
            # 手写 .geo: 生成到临时副本, 原始文件不碰 — 曾静默覆盖导致
            # 用户手写几何永久丢失 (审计 2026-08-03 实测复现)
            fd, tmp_geo = tempfile.mkstemp(
                prefix='.fem2d-txt-', suffix='.geo',
                dir=os.path.dirname(os.path.abspath(geo_p)))
            os.close(fd)
            os.unlink(tmp_geo)   # generate_geo 以 "w" 打开
            print(f"  [INFO] {geo_p} 是手写 .geo — 生成到临时副本 {tmp_geo}, "
                  "原始文件未修改")
            geo_p = tmp_geo
            atexit.register(
                lambda p=tmp_geo: os.path.isfile(p) and os.unlink(p))
    try:
        generate_geo(parse_spec(fp), geo_p, quad=config.quad)
    except (ValueError, IndexError) as error:
        # IndexError 防御: 曾 '内孔 圆 x=' 透传裸 'list index out of range'
        # (修复后 parse_spec 不再抛, 双保险) (审计 2026-08-03)
        raise CliError(
            f"  [FATAL] 几何生成失败: {error}",
            exit_code=1)
    source_geo_path = os.path.abspath(geo_p)
    msh, gmsh_import = generate_geo_with_topology(
        geo_p,
        quad=config.quad,
        output_path=os.path.splitext(os.path.abspath(fp))[0] + '.msh',
        plane_type=config.plane or 'stress',
    )
    if not msh:
        raise CliError(
            f"Gmsh 网格生成失败: {geo_p} — 退出码非零或产物缺失",
            exit_code=1)
    return msh, gmsh_import, source_geo_path


def resolve_input_file(fp, config, ask=None):
    """输入源总入口: 把 .spec/.geo/.txt/.msh 统一解析为 Gmsh 网格导入.

    返回 :class:`ResolvedInput`。失败时抛 :class:`CliError` (CLI 层转退出码)。
    """
    if ask is None:
        from fem2d.cli import ask as default_ask
        ask = default_ask

    # .spec → 解析后取 mesh 路径 (相对路径以 .spec 所在目录为基准)
    if fp.endswith('.spec'):
        fp = resolve_spec_overrides(fp, config)

    # .inp (Abaqus) 输入口已移除 (2026-08) — 网格唯一来源为 .geo/.txt/.msh
    if fp.endswith('.inp'):
        raise CliError(
            '[ERROR] Abaqus .inp 输入已移除 — 请提供 .geo (Gmsh 几何)、'
            '.msh (Gmsh 网格) 或 .spec。',
            exit_code=2)

    # .geo → 询问网格密度（CLI全参数时跳过), 跑 Gmsh 生成 .msh
    quad_applied = False
    gmsh_import = None
    source_geo_path = None
    geo_config_applied = False
    if fp.endswith('.geo'):
        fp, gmsh_import, source_geo_path = resolve_geo(fp, config, ask=ask)
        quad_applied = config.quad
        geo_config_applied = True  # @FEM 已在 resolve_geo 内合并
    elif fp.endswith('.txt'):
        fp, gmsh_import, source_geo_path = resolve_txt(fp, config)
        quad_applied = config.quad
        # .txt 生成的 .geo 含 @FEM 注解 — 由 runner._apply_geo_fem_config 合并
        if config.lc is not None:
            # --lc 只对 .geo 输入生效 (.txt 用自身的 网格 行) — 曾静默
            # 忽略, 用户以为加密了实际没有 (审计 2026-08-03)
            print("  [WARN] --lc 只对 .geo 输入生效 — .txt 用自身的"
                  " '网格' 行, --lc 已忽略 (请在 .txt 中修改 网格 值)")
    elif fp.endswith('.msh'):
        # 已生成的 Gmsh 网格直接导入 — 无需重新网格化 (评审建议)
        # 必须传 plane_type: .msh 不含平面态信息, 默认 stress 会让
        # --plane strain 的 CPE 判型与导入的 CPS 类型冲突 (评审发现)
        if config.quad:
            # 重组只在 .geo/.txt 生成阶段生效, 直接输入 .msh 时静默
            # 忽略会误导用户 (高强度审计 2026-08-02)
            print("  [WARN] --quad 只对 .geo/.txt 网格生成生效, "
                  ".msh 直接输入时忽略")
        if config.lc is not None:
            print("  [WARN] --lc 只对 .geo 输入生效, .msh 直接输入时忽略 "
                  "(请用 gmsh 重新网格化)")
        from fem2d.gmsh_adapter import import_msh
        fp = os.path.abspath(fp)
        gmsh_import = import_msh(
            fp, plane_type=config.plane or "stress")
        if not gmsh_import:
            raise CliError(
                f'[ERROR] .msh 导入失败: {fp}',
                exit_code=1)
    else:
        raise CliError(
            f'[ERROR] 不支持的输入: {fp} — 仅支持 .spec/.geo/.txt/.msh',
            exit_code=2)

    if not fp.endswith('.msh'):
        raise CliError(
            f'[ERROR] 最终需要 .msh 文件, 得到: {fp}',
            exit_code=1)

    return ResolvedInput(
        fp=fp, gmsh_import=gmsh_import,
        source_geo_path=source_geo_path, quad_applied=quad_applied,
        geo_config_applied=geo_config_applied)
