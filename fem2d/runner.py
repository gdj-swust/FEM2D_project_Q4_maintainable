"""求解主流程编排 — 原 run.py 的 ``__main__`` 拆分层。

把 cli (参数) / input_source (输入解析) / mesh / boundary / loads /
solver / error_est / visualize 按固定顺序串成一次分析。与 fem2d/ 其他
模块一致, 本层不含 FEM 数值算法, 只做流程编排与交互输入。
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

from . import (
    Mesh,
    estimate_error,
    print_segments,
    run_patch_test,
    solve,
    verify_all_elements,
)
from .bc_apply import apply_bcs
from .verification import run_plane_verification
from .boundary import (
    BoundaryDiagnostics,
    build_boundary_segments,
    semantic_coverage,
)
from .cli import _resolve_plane_type, ask, is_batch_mode, parse_args
from .config import AnalysisConfig
from .element import get_element_kernel
from .errors import CliError, reconfigure_streams
from .input_source import resolve_input_file
from .preprocess import merge_geo_fem_config, parse_geo_fem_config, validate_mesh
from .quality import report as report_mesh_quality
from .reporting import (
    bending_heuristics,
    displacement_scale,
    print_result_summary,
)
from .wizard import run_wizard

# ── 分片检验 (每个 Python 进程跑一次) ──
_patch_checked = set()


# ═══════════════════════════════════════════════════════════════
# 自检
# ═══════════════════════════════════════════════════════════════

def _ensure_patch_test(elem_type="CPS3", plane="stress"):
    """每个进程对每种 (单元, 平面) 组合运行一次分片检验.

    契约: 通过 → 缓存该组合 (同进程后续调用直接返回);
    失败 → 抛 CliError **且不缓存** — 同进程下一次调用必须重新
    运行并再次失败, 不能静默放行 (失败只属于本次运行, 不是可复用的结果).
    """
    global _patch_checked
    key = (elem_type, plane)
    if key in _patch_checked:
        return
    print(f"\n{'='*55}")
    print(f"  {elem_type} Patch Test — Bathe §5.3.3 "
          f"(自动, plane={plane})")
    print(f"{'='*55}")
    r = run_patch_test(verbose=True, plane=plane, elem_type=elem_type)
    if not r["all_passed"]:
        raise CliError(
            "[FATAL] Patch test failed! Fix element code before solving.",
            exit_code=1)
    _patch_checked.add(key)


def _standalone_self_test() -> int:
    """``python run.py --self-test`` (不带网格): 全单元 patch test + 材料验证."""
    print("\n=== CST + Q4 + Q4R + Q4I Patch Tests (standalone) ===\n")
    results = [
        run_patch_test(verbose=True, elem_type=elem_type)
        for elem_type in ("CPS3", "CPS4", "CPS4R", "CPS4I")
    ]
    print("\n=== D-matrix / Plane Stress-Strain Verification ===\n")
    plane_pass, plane_fail = run_plane_verification()
    print(f"  → {plane_pass} PASS, {plane_fail} FAIL")
    success = (
        all(result["all_passed"] for result in results)
        and plane_fail == 0)
    return 0 if success else 1


# ═══════════════════════════════════════════════════════════════
# 网格导入
# ═══════════════════════════════════════════════════════════════

def _import_mesh(resolved):
    """读取网格数据 — 统一来自 Gmsh 导入结果 (.msh 由 import_msh 提取).

    返回 ``(coords, elems, node_id_map, elem_type, region_registry)``。
    edge_labels 与 sibling_geo 曾作为恒 None 的死输出字段保留
    (源 .geo 路径由 ``ResolvedInput.source_geo_path`` 提供; 边标签
    语义已由 region_registry/Physical Curve 通道承担)。
    """
    g = resolved.gmsh_import
    if g is None:
        raise RuntimeError(
            "No Gmsh import result — .msh 输入必须在 input_source 阶段完成"
            " import_msh (内部状态错误, 请报告).")
    return (g.nodes, g.elements, g.node_tag_to_index,
            g.elem_type, g.regions)


def _build_mesh(config, resolved, coords, elems, mesh_elem_type,
                region_registry):
    """网格校验 + 平面态判型 + Mesh 构造 + region 面积报告."""
    # ── 2a. 网格导入后校验 (Gmsh checkMeshCoherence 模式) ──
    # 传入 mesh_elem_type — 校验网格声明的类型与节点数一致 (2026-08:
    # 参数名清理, .inp 输入口已移除)
    vreport = validate_mesh(coords, elems, mesh_elem_type)
    if vreport["warnings"]:
        for w in vreport["warnings"]:
            print(f"  [WARN] {w}")
    if not vreport["ok"]:
        for e in vreport["errors"]:
            print(f"  [ERROR] {e}")
        raise RuntimeError("Mesh validation failed — aborting.")

    # ── 平面应力/应变判型: 必须在 Mesh 创建前完成 ──
    requested_plane = config.plane
    try:
        if config.elem_type:
            # --elem-type 覆写内核 (CST/Q4/Q4R/Q4I) 平面无关 — 曾按网格原码
            # 做冲突检查: CPS4 网格 + --elem-type Q4R --plane strain 误报
            # "cannot use --plane strain"
            config.plane = _resolve_plane_type(mesh_elem_type, None)
            if requested_plane is not None:
                config.plane = requested_plane
        else:
            config.plane = _resolve_plane_type(
                mesh_elem_type, requested_plane)
    except ValueError as error:
        # 用户参数与网格单元码冲突 (CPE 网格 + --plane stress) — 用户错误
        # 退出码 1; 曾裸 ValueError 冒泡到顶层 except Exception 被归为
        # 内部错误 2, 违反退出码矩阵
        raise CliError(f"  [FATAL] {error}", exit_code=1) from error
    if requested_plane is None and config.plane == "strain":
        print(f"  [auto] {mesh_elem_type} 网格 → 默认 plane=strain")

    # ── 单元类型覆盖预检 (在 Mesh 构造器抛裸 ValueError 前给友好信息) ──
    elem_type = config.elem_type if config.elem_type else mesh_elem_type
    if config.elem_type:
        expected_npe = get_element_kernel(config.elem_type).nodes_per_element
        if elems.shape[1] != expected_npe:
            raise CliError(
                f"  [FATAL] --elem-type {config.elem_type} 需要 "
                f"{expected_npe} 节点/单元, 但当前网格是 {elems.shape[1]} "
                f"节点/单元 ({mesh_elem_type}) — 单元类型与网格拓扑不兼容。",
                exit_code=1)
        print(f"  [elem] override: {mesh_elem_type} → {config.elem_type}  "
              f"({len(coords)} nodes, {len(elems)} elems)")

    print(f"\n{'='*55}")
    print(f"  FEM2D | {elem_type} | {config.plane}")
    print(f"{'='*55}")
    print(f"  网格: {os.path.basename(resolved.fp)}")
    print(f"  E={config.E:.2e}  nu={config.nu}  t={config.thickness}")

    mesh = Mesh(nodes=coords, elements=elems, E=config.E, nu=config.nu,
                thickness=config.thickness, plane_type=config.plane,
                elem_type=elem_type)

    if region_registry is not None:
        region_registry.validate_against_mesh(mesh)
        for surface in region_registry.surfaces:
            area = region_registry.surface_area(surface.name, mesh)
            print(
                f"  [region] Surface '{surface.name}': "
                f"{len(surface.element_ids)} elements, area={area:.6g}")
    return mesh


# ═══════════════════════════════════════════════════════════════
# 边界与 BC 装配
# ═══════════════════════════════════════════════════════════════

def _apply_geo_fem_config(config, geo_path):
    """从 .geo 的 @FEM: 注释自动读取 BC — 委托 preprocess.merge_geo_fem_config.

    优先级: CLI 显式参数 > .geo 配置 (与 input_source.resolve_geo 共用
    同一合并逻辑, 曾各自实现导致分叉)。
    """
    if not geo_path:
        return
    merge_geo_fem_config(
        parse_geo_fem_config(geo_path), config, verbose=True)


def _build_boundary(mesh, config, region_registry, edge_labels, geo_path):
    """联合边界模型: 网格拓扑/几何 + Gmsh Physical Curve 语义.

    返回 (segs, diagnostics)。语义恢复失败时按 ``--require-physical-groups``
    / ``--strict-boundary`` 决定是警告还是致命退出。
    """
    boundary_diagnostics = BoundaryDiagnostics()
    try:
        segs = build_boundary_segments(
            mesh,
            registry=region_registry,
            edge_labels=edge_labels,
            geo_path=geo_path,
            diagnostics=boundary_diagnostics,
            strict=config.strict_boundary,
        )
    except ValueError as error:
        raise CliError(f"  [FATAL] {error}", exit_code=1) from error
    print_segments(segs)

    for issue in boundary_diagnostics.issues:
        level = "ERROR" if issue.severity == "error" else "WARN"
        print(f"  [{level}] boundary/{issue.code}: {issue.message}")

    # ── Physical Curve 语义恢复报告 ──
    semantic_report = semantic_coverage(
        mesh, segs, diagnostics=boundary_diagnostics)
    if semantic_report["physical_names"]:
        print(
            "  [boundary semantics] "
            f"{len(semantic_report['mapped_physical_names'])}/"
            f"{len(semantic_report['declared_physical_names'])} "
            "Physical Curves mapped, "
            f"{semantic_report['covered_edges']}/"
            f"{semantic_report['total_boundary_edges']} boundary edges "
            "mapped")
    else:
        if geo_path or region_registry is not None:
            print(
                "  [WARN] 未恢复任何 Physical Curve；当前仅使用网格"
                "拓扑/几何识别。")
        if config.require_physical_groups:
            raise CliError(
                "  [FATAL] --require-physical-groups 已启用，但没有可用的 "
                "Physical Curve 语义。",
                exit_code=1)
    if (
            config.require_physical_groups
            and boundary_diagnostics.dropped_physical_names):
        dropped = ", ".join(
            boundary_diagnostics.dropped_physical_names)
        raise CliError(
            "  [FATAL] --require-physical-groups 已启用，但以下 "
            f"Physical Curve 未映射到外边界: {dropped}",
            exit_code=1)
    if config.require_physical_groups and boundary_diagnostics.errors:
        codes = ", ".join(sorted({
            issue.code for issue in boundary_diagnostics.errors
        }))
        raise CliError(
            "  [FATAL] --require-physical-groups 已启用，但边界语义存在"
            f"不可安全使用的问题: {codes}",
            exit_code=1)
    return segs, boundary_diagnostics


def _print_boundaries(config, mesh, segs):
    """``--list-boundaries``: 仅列出边界后退出."""
    print("\n  [INFO] --list-boundaries: 仅列出边界, 不求解")
    print("\n  CLI 快捷名称 (用于 --fix/--traction):")
    print(f"  {'边编号':<8} {'类型':<8} {'节点数':<6} {'标签':<20} {'尺寸'}")
    print(f"  {'-'*60}")
    for i, s in enumerate(segs):
        n_nodes = len(s['nodes'])
        label = s.get('label', '')
        if s['type'] == 'arc':
            dim = f"R={s['info'].get('radius', 0):.4f}"
        else:
            coords_arr = np.array([mesh.nodes[int(n)] for n in s['nodes']])
            seg_len = np.sum(np.linalg.norm(np.diff(coords_arr, axis=0), axis=1))
            dim = f"L={seg_len:.4f}"
        print(f"  {i+1:<8} {s['type']:<8} {n_nodes:<6} {label:<20} {dim}")
    # 曾用 config.mesh: 交互选文件路径时 config.mesh 为 None →
    # os.path.basename(None) TypeError, 边界已列出却报错退出码 2
    #
    demo_fp = config.mesh or "模型文件"
    print("\n  用法示例:")
    print(f"    python run.py {os.path.basename(demo_fp)} --fix left --body 0,-78000")
    print(f"    python run.py {os.path.basename(demo_fp)} --fix left --traction right:1e6,0")


# ═══════════════════════════════════════════════════════════════
# 求解与后处理
# ═══════════════════════════════════════════════════════════════

def _run_solve_time_self_test(mesh, plane):
    """``--self-test`` (带网格): 用实际求解的单元类型跑 patch test + 验证."""
    # 用实际求解的单元类型 (--elem-type 覆写后应与 mesh 一致),
    # 而非 .inp 文件里声明的原始单元码
    _ensure_patch_test(mesh.elem_type, plane)
    verify_all_elements(mesh, verbose=True)
    print(f"\n{'='*55}")
    print("  D-matrix / Plane Stress-Strain Verification")
    print(f"{'='*55}")
    vp_pass, vp_fail = run_plane_verification()
    print(f"  → {vp_pass} PASS, {vp_fail} FAIL")
    if vp_fail > 0:
        raise CliError(
            "[FATAL] Plane stress/strain verification failed!",
            exit_code=1)


def _analyze(mesh, config):
    """网格质量 → 求解 → 误差估计. 返回 ``(result, z2, q)``."""
    q = report_mesh_quality(mesh)
    result = solve(
        mesh,
        check_condition=config.check_cond,
        linear_solver=config.linear_solver,
    )
    error_method = config.error_method
    if error_method == 'auto':
        error_method = 'weighted' if mesh.n_elements >= 50000 else 'SPR'
        if error_method == 'weighted':
            print(
                "[Error] 大网格自动使用 weighted 恢复；"
                "如需完整 SPR，请加 --error-method spr")
    elif error_method == 'spr':
        error_method = 'SPR'
    elif error_method == 'l2':
        error_method = 'L2'
    z2 = estimate_error(mesh, result, method=error_method)
    return result, z2, q


def _plot(config, mesh, result, scale):
    """Isoband 云图绘制 (--no-plot 只抑制交互窗口).

    Isoband 参数校验已集中到 AnalysisConfig.validate() (入口统一拦截,
    曾在此处与配置层各校验一次 — 行为分叉), 此处只做层数计算.
    """
    # ── Isoband 固定带宽: 层数计算 (校验已由 config.validate 完成) ──
    band_args = (config.band_min, config.band_max, config.band_step)
    specified = [x is not None for x in band_args]

    isoband_levels = None
    if all(specified):
        # config.validate 已保证 (max-min)/step 在 1e-9 相对容差内整除 —
        # 但浮点除法 (如 0.3/0.1=2.9999999999999996) 常落在整数下方,
        # floor 截断会少一个带、末带静默加宽一倍 (0..0.3@0.1 曾生成 [0, 0.1, 0.3], 末带宽 0.2)。
        ratio = (config.band_max - config.band_min) / config.band_step
        n_bands = int(round(ratio))
        isoband_levels = (
            config.band_min + np.arange(n_bands + 1) * config.band_step)
        isoband_levels[-1] = config.band_max   # 尾部浮点误差归位
        print(f"  [Isoband] fixed levels: {isoband_levels[0]:.3g} to "
              f"{isoband_levels[-1]:.3g}, step={config.band_step:.3g} "
              f"({len(isoband_levels)-1} bands)")
        if (config.band_tag is not None
                and config.band_tag != "vm"):
            # levels 只应用于匹配 band_tag 的图; 初始/保存图恒为 vm —
            # 曾日志宣称生效而图中未应用
            print(f"  [WARN] --band-tag {config.band_tag} ≠ vm — "
                  "初始/保存图为 vm, 固定带宽不应用; "
                  "交互模式下切换到对应 tag 后才生效")

    # matplotlib 导入成本 ~1-2s — --no-plot 路径不付此成本, 保持局部
    from .visualize import PLOTS, interactive_plot, plot_three
    isoband_tag = config.band_tag if config.band_tag else 'vm'
    sigma_ref = config.jump_ref
    # 曾内联重写批处理判定 (漏 save/no_plot): 交互终端 + CLI BC 参数时,
    # 输入提示已跳过 (is_batch_mode=True) 但 interactive_plot 仍在
    # input() 阻塞挂起 — 统一用唯一批处理判定
    batch_mode = is_batch_mode(config) or bool(config.save) or config.no_plot
    plot_three(mesh, result, tag='vm', scale=scale,
               save=config.save if config.save else None,
               isoband_levels=isoband_levels, isoband_tag=isoband_tag,
               sigma_ref=sigma_ref)
    # 成功文案在绘图之后打印 — 曾先打印"云图已生成"再实际绘制,
    # --save 失败时输出"成功后再失败"的矛盾信息
    print("\n  [Plot] 云图已生成 — 可用按键:")
    for key, (_, label) in sorted(PLOTS.items(), key=lambda x: int(x[0])):
        print(f"     [{key:>2}] {label}")
    if not batch_mode:
        try:
            interactive_plot(mesh, result, scale,
                             isoband_levels=isoband_levels,
                             isoband_tag=isoband_tag,
                             sigma_ref=sigma_ref)
        except EOFError:
            # 终端关闭 stdin 时 input() 抛 EOFError — 求解已成功, 曾报
            # [ERROR] 退出码 2 误导脚本调用 (run_demo 已优雅处理)
            print("\n[INFO] 非交互环境 (stdin 不可用), 跳过交互绘图")


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

@dataclass
class _Model:
    """一次分析在流程中累积的模型状态 (阶段间传递)."""

    mesh: object
    segs: list
    geo_path: object
    region_registry: object
    node_id_map: dict
    resolved: object
    bfx: object = None
    bfy: object = None


def _resolve_input(config):
    """阶段 1: 输入文件选择 + 解析 (.spec/.geo/.txt/.msh).

    无 mesh 参数时: --wizard 或终端可用 → 交互建模向导 (替代裸
    "输入文件:" 提问, 用户钦定方向); 否则提问文件路径。
    返回 ResolvedInput; 无有效输入时返回 None (main 退出码 1).
    """
    fp = config.mesh
    if not fp:
        if config.wizard or sys.stdin.isatty():
            fp = run_wizard(config)
        else:
            fp = ask("  输入文件 (.geo / .txt / .msh / .spec): ")
    if not fp:
        print('[ERROR] 需要指定输入文件')
        return None
    if not os.path.isfile(fp):
        print(f'[ERROR] 文件不存在: {fp}')
        return None
    return resolve_input_file(fp, config)


def _build_model(config, resolved) -> _Model:
    """阶段 2: 网格导入校验 + @FEM 配置合并 + 联合边界模型."""
    (coords, elems, node_id_map, elem_type,
     region_registry) = _import_mesh(resolved)

    mesh = _build_mesh(config, resolved, coords, elems, elem_type,
                       region_registry)

    # 从 .geo 的 @FEM: 注释自动读取 BC (geo_spec.py 生成)。
    # .geo/.txt 输入已在 input_source 阶段合并 (geo_config_applied=True),
    # 避免二次解析把自动载荷误判为 CLI 显式 (错误 WARN)。
    geo_path = resolved.source_geo_path
    if not resolved.geo_config_applied:
        _apply_geo_fem_config(config, geo_path)

    segs, _ = _build_boundary(
        mesh, config, region_registry, None, geo_path)
    return _Model(mesh=mesh, segs=segs, geo_path=geo_path,
                  region_registry=region_registry, node_id_map=node_id_map,
                  resolved=resolved)


def _apply_conditions(config, model) -> None:
    """阶段 3: BC/载荷 (CLI 预设或交互) + 模型总表打印."""
    model.bfx, model.bfy = apply_bcs(
        config, model.mesh, model.segs, model.region_registry,
        model.node_id_map, model.resolved.source_geo_path)

    mesh = model.mesh
    nf = len(mesh.fixed_dofs)
    nt = len(mesh.surface_tractions)
    nc = len(mesh.concentrated_forces)
    print(f"\n{'='*55}")
    print(f"  {mesh.n_nodes} nodes  {mesh.n_elements} elems  {mesh.n_dof} DOFs")
    print(f"  E={config.E:.2e}  nu={config.nu}  t={config.thickness}  {config.plane}")
    body_str = (f"({model.bfx:.3e},{model.bfy:.3e})"
                if not callable(model.bfx) and not callable(model.bfy)
                else "f(x,y)")
    print(f"  fixed: {nf} DOFs  surf: {nt} segs  body: {body_str}  conc: {nc}")
    print(f"{'='*55}")


def _analyze_and_report(config, model):
    """阶段 4: 求解 + 误差估计 + 中文总结报告.

    返回 ``(result, z2, q, scale)`` 供绘图阶段使用.
    """
    if config.self_test:
        _run_solve_time_self_test(model.mesh, config.plane)
    result, z2, q = _analyze(model.mesh, config)

    scale = displacement_scale(model.mesh, result['u'])
    bending_stiff, n_through_x, n_through_y = bending_heuristics(model.mesh)
    print_result_summary(config, model.mesh, result, z2, q, scale,
                         bending_stiff, n_through_x, n_through_y)
    return result, z2, q, scale


def main(argv=None) -> int:
    """一次完整分析: 输入 → 网格 → 边界 → BC → 求解 → 报告 → 云图.

    执行层只消费 AnalysisConfig (类型化配置对象) — 不感知 argparse
    参数表; CLI/.spec/.geo 配置统一由 from_args + 合并逻辑汇入 config。
    流程按 5 个阶段函数组织 (resolve → model → conditions → analyze →
    plot), 每个阶段返回显式产物, 便于单独测试与替换.
    """
    # CLI 入口职责: 项目根置于 sys.path, 使 scripts 工具层可导入。
    # run.py 脚本方式运行 sys.path[0] 已是项目根; 此处兜底
    # python -m fem2d.runner / console 场景。曾由 fem2d.input_source
    # 模块顶层注入 (import 副作用污染库用户进程)。
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)
    # 编码安全网 (与 run_demo.py 共用 reconfigure_streams): 非中文
    # Windows (cp1252 等) 下中文输出不 UnicodeEncodeError 崩溃
    reconfigure_streams()
    try:
        config = AnalysisConfig.from_args(parse_args(argv))
    except ValueError as error:
        # 配置校验失败 (非法参数组合, 如 --band-min 缺 --band-step) — 用户
        # 错误退出码 1; 曾归入内部错误 2, 与退出码矩阵不一致 (此阶段 config
        # 尚未构建, 无 --debug 概念)
        print(f"[ERROR] {error}")
        return 1

    # ── 独立自检: Patch Test + plane stress/strain 材料验证 ──
    if config.self_test and not config.mesh:
        # 曾静默吞掉非法 BC 参数 (--force garbage!! 与 --self-test 组合
        # 退出 0, 载荷从未生效也无提示)
        for _key, _val in (("--fix", config.fix), ("--fix-ux", config.fix_ux),
                           ("--fix-uy", config.fix_uy),
                           ("--traction", config.traction),
                           ("--force", config.force),
                           ("--body", config.body)):
            if _val:
                print(f"  [WARN] {_key} 在独立自检模式下不生效 — "
                      "已忽略 (自检只跑单元数学验证)")
        if config.band_min is not None:
            # 与 --no-plot 同模式: band 参数已通过 config 校验但独立自检
            # 不绘图 — 曾静默忽略
            print("  [WARN] --band-min/max/step 在独立自检模式下不生效 — "
                  "已忽略 (自检只跑单元数学验证)")
        if config.list_boundaries:
            # 曾声称"自检不执行, 仅列出边界"但随后仍执行自检 — 文案撒谎
            # 行为: 自检照常, list-boundaries 无输入文件无从列出
            print("  [WARN] --self-test 与 --list-boundaries 组合: "
                  "--list-boundaries 需要输入文件, 独立自检模式下不生效 — "
                  "自检照常执行")
        return _standalone_self_test()

    try:
        resolved = _resolve_input(config)
    except CliError as error:
        # CLI 输入/交互错误 (原 sys.exit 语义) — 消息已带 [ERROR]/[FATAL]
        # 前缀, 不再包一层; 保留各自退出码
        print(error)
        return error.exit_code
    except KeyboardInterrupt:
        # 求解/解析中途 Ctrl-C (ask 内的 Ctrl-C 已是 CliError) — 向导
        # banner 承诺"随时 Ctrl-C 退出", 曾泄漏整段 traceback
        print("\n  [INFO] 已中断 (Ctrl-C)")
        return 130
    except Exception as error:
        # 输入解析阶段的任意异常 (含 gmsh 依赖缺失时的 ImportError/OSError
        # 与 Gmsh API 抛出的裸 Exception) — 友好报错而非整段 traceback;
        # --debug 恢复完整 traceback。
        if getattr(config, "debug", False):
            raise
        print(f"[ERROR] {error}")
        return 2
    if resolved is None:
        return 1

    try:
        model = _build_model(config, resolved)

        # ── --list-boundaries: 仅列出边界后退出 ──
        if config.list_boundaries:
            _print_boundaries(config, model.mesh, model.segs)
            return 0

        _apply_conditions(config, model)
        result, _z2, _q, scale = _analyze_and_report(config, model)

        if not config.no_plot or config.save:
            # --no-plot 只抑制交互窗口, --save 仍应生成文件
            _plot(config, model.mesh, result, scale)
        elif any(x is not None for x in (config.band_min, config.band_max,
                                         config.band_step)):
            # band 参数已通过 config.validate 校验但 --no-plot 无绘图 —
            # 曾静默忽略 (参数看似生效实未生效, 教学用户困惑)
            print("  [INFO] --band-min/max/step 仅在绘图时生效 — "
                  "--no-plot 抑制绘图, 参数已忽略")
    except CliError as error:
        print(error)
        return error.exit_code
    except KeyboardInterrupt:
        print("\n  [INFO] 已中断 (Ctrl-C)")
        return 130
    except Exception as error:
        # 顶层异常边界: 求解/模型/绘图阶段的任意领域异常
        # (含 Gmsh 依赖缺失时的 ImportError/OSError) 输出错误摘要而非整段
        # Python traceback; --debug 恢复完整 traceback 便于诊断。
        if getattr(config, "debug", False):
            raise
        print(f"[ERROR] {error}")
        return 2
    return 0
