"""BC/载荷装配 — 固定边/面力/压力/集中力/体力 (CLI 与交互模式).

从 runner.py 拆出 (曾与主流程编排混在一个 700+ 行文件)。
交互提问仅在此处; runner 只编排流程。

2026-08: apply_bcs 拆分 — 每个载荷类型一个独立阶段函数
(可单测), apply_bcs 只编排。圈复杂度 80 → ~10。
"""
import numpy as np

from .boundary import _resolve_edge_indices
from .cli import ask, is_batch_mode
from .errors import CliError
from .input_source import physical_point_from_geo
from .loads import make_edge_profile_func, parse_traction, parse_vec2
from .mesh import _check_load_pair
from .regions import canonical_edge, ordered_edge_chains


def _fmt_comp(value):
    """格式化载荷分量 — callable (表达式面力) 显示为 f(x,y), 避免 :.3e 崩溃."""
    return "f(x,y)" if callable(value) else f"{value:.3e}"


def _resolve_boundary_selection(selection, segs, *, fatal):
    """Resolve one CLI/interactive selector with ambiguity diagnostics."""
    try:
        return _resolve_edge_indices(selection, segs)
    except ValueError as error:
        if fatal:
            raise CliError(f"  [FATAL] {error}", exit_code=1)
        print(f"  [WARN] {error}")
        return []


def _each_edge(ns, apply):
    """对段内每条边 (a,b) 调用 apply(a, b) — 统一边施加循环."""
    for a, b in zip(ns, ns[1:]):
        apply(int(a), int(b))


def _interactive_edge_index(segs):
    """交互收集边编号 (回车结束), 逐个产出匹配的边索引.

    fix 与 traction 交互共用 (曾各自实现相同的提问循环).
    """
    while True:
        inp = ask("  边编号 (回车结束): ")
        if not inp:
            return
        indices = _resolve_boundary_selection(inp, segs, fatal=False)
        if not indices:
            print("    ? 无效编号")
            continue
        if len(indices) > 1:
            # 曾 yield indices[0] 静默丢弃其余匹配: "left" 命中 3 段时
            # 只施加第 1 段, 打印却显示成功
            print(f"    {len(indices)} 个匹配段 — 逐段处理: "
                  f"{', '.join(str(i + 1) for i in indices)}")
        for index in indices:
            yield index


def _print_segment_menu(segs):
    """交互式边菜单: 段类型 (直边/圆弧/椭圆/曲线) + 标签."""
    print(f"\n  {'='*45}")
    for i, s in enumerate(segs):
        tp = s['type']
        label = s['label']
        if tp == 'line':
            kind = '直边'
        elif tp == 'arc':
            R = s['info'].get('radius', 0)
            kind = f'圆弧 R={R:.6g}'
        elif tp == 'ellipse':
            a = s['info'].get('semi_major', 0)
            b = s['info'].get('semi_minor', 0)
            kind = f'椭圆 a={a:.6g} b={b:.6g}'
        else:
            kind = '曲线'
        print(f"  [{i+1}] {kind} | {label}")


def _apply_fix_bcs(config, mesh, segs, batch_mode):
    """边界约束 (CLI 预设 fix/fix_ux/fix_uy 或 交互逐边).

    batch_mode: 批处理判定与 traction/body 分支一致 (曾只按 config.fix
    是否为空分流 — --body/--save 时 fix 仍交互提问阻塞)。
    """
    print("\n  --- 边界约束 (Ux=?, Uy=?) ---")
    if config.fix or config.fix_ux or config.fix_uy:
        # CLI 模式 — fix/fix_ux/fix_uy 参数化合并 (曾三段重复循环)
        _FIX_LABELS = {"both": "Ux=0, Uy=0", "x": "Ux=0", "y": "Uy=0"}
        for dof, spec in (("both", config.fix),
                          ("x", config.fix_ux), ("y", config.fix_uy)):
            edges = [e.strip() for e in
                     spec.replace(',', ';').split(';') if e.strip()]
            for e in edges:
                matched = _resolve_boundary_selection(e, segs, fatal=True)
                if not matched:
                    raise CliError(
                        f"  [FATAL] 未找到边 '{e}' — 批处理模式终止. "
                        f"可用: {', '.join(str(i+1) for i in range(len(segs)))}",
                        exit_code=1)
                for idx in matched:
                    for n in segs[idx]['nodes']:
                        mesh.fix_node(int(n), dof, 0.0)
                    print(f"  边{idx+1}: {_FIX_LABELS[dof]}")
    elif batch_mode:
        # 批处理: 已判定不交互, 无约束可施 — 明确告知而非静默
        print("  [INFO] 批处理模式, 未指定 --fix — 无边界约束 "
              "(模型可能欠约束)")
    else:
        # 交互模式: 逐条添加
        for idx in _interactive_edge_index(segs):
            ux_str = ask(f"    边{idx+1} Ux位移 [默认0]: ")
            uy_str = ask(f"    边{idx+1} Uy位移 [默认0]: ")
            ux = float(ux_str) if ux_str else 0.0
            uy = float(uy_str) if uy_str else 0.0
            ux_given = ux_str.strip() != ''
            uy_given = uy_str.strip() != ''
            for n in segs[idx]['nodes']:
                if ux_given:
                    mesh.fix_node(int(n), 'x', ux)
                if uy_given:
                    mesh.fix_node(int(n), 'y', uy)
                if not ux_given and not uy_given:
                    mesh.fix_node(int(n), 'both', 0.0)
            parts = []
            if ux_given:
                parts.append(f'Ux={ux}')
            if uy_given:
                parts.append(f'Uy={uy}')
            if not ux_given and not uy_given:
                parts = ['Ux=0', 'Uy=0']
            print(f"    边{idx+1}: {', '.join(parts)}")


def _apply_traction_profile(mesh, segs, matched, edge_str, tx, ty, profile):
    """剖面面力 (:p 抛物线 / :l 线性, 按弧长) — 链式施加."""
    selected_edges = set()
    edge_directions = {}
    for idx in matched:
        ns = list(map(int, segs[idx]['nodes']))
        for a, b in zip(ns, ns[1:]):
            key = canonical_edge(a, b)
            selected_edges.add(key)
            edge_directions[key] = (a, b)

    chains = ordered_edge_chains(selected_edges)
    for raw_chain in chains:
        chain = list(map(int, raw_chain))
        first_key = canonical_edge(chain[0], chain[1])
        if edge_directions.get(first_key) != (chain[0], chain[1]):
            chain.reverse()
        is_closed = len(chain) >= 4 and chain[0] == chain[-1]
        if profile == 'l' and is_closed:
            raise CliError(
                "  [FATAL] 线性面力 :l 不能施加到闭合边界 "
                f"'{edge_str}'；闭环起点处会产生不唯一值。"
                "请拆成开放 Physical Curves 或使用 :p/常量压力。",
                exit_code=1)
        chain_coords = mesh.nodes[chain]
        edge_lengths = np.linalg.norm(
            np.diff(chain_coords, axis=0), axis=1)
        total_length = float(np.sum(edge_lengths))
        arc_start = 0.0
        for edge_index, (a, b) in enumerate(zip(chain, chain[1:])):
            pf = make_edge_profile_func(
                tx, ty, profile,
                mesh.nodes[a], mesh.nodes[b],
                arc_start, total_length)
            mesh.add_traction(int(a), int(b), pf[0], pf[1])
            arc_start += float(edge_lengths[edge_index])
    profile_label = {
        'p': 'parabolic arc-length',
        'l': 'linear arc-length',
    }[profile]
    print(
        f"  面力: {edge_str}  t=({_fmt_comp(tx)}, {_fmt_comp(ty)})  "
        f"[{profile_label}, {len(chains)} component(s)]")


def _apply_tractions(config, mesh, segs, batch_mode):
    """面力/压力 (CLI 预设 或 交互, 逐边添加)."""
    print("\n  --- 面力 (tx, ty [Pa]) ---")
    if config.traction:
        traction_specs = [t.strip() for t in
                          config.traction.split(';') if t.strip()]
    elif batch_mode:
        traction_specs = []  # 批处理模式: 跳过交互式面力提问
    else:
        traction_specs = []
        for idx in _interactive_edge_index(segs):
            tx_str = ask(f"    边{idx+1} tx [Pa]: ")
            if not tx_str:
                break
            ty_str = ask(f"    边{idx+1} ty [Pa]: ")
            if not ty_str:
                break
            traction_specs.append(f"{idx+1}:{tx_str},{ty_str}")
            print(f"    边{idx+1}: tx={tx_str}, ty={ty_str}")

    is_batch_traction = bool(config.traction)  # CLI/.spec 模式, 错误边名应终止
    for t_spec in traction_specs:
        edge_str, tx, ty, profile = parse_traction(t_spec)
        if edge_str is None:
            # 无 ':' 前缀 (如 --traction "1e6,0"): 曾报"未找到边 'None'"
            # 把用户引向检查边名, 真正问题是缺 edge: 前缀
            msg = (f"面力规格 '{t_spec}' 缺少边前缀 — 正确格式: "
                   f"edge:tx,ty (例: right:1e6,0) 或 edge:p:n")
            if is_batch_traction:
                raise CliError(f"  [FATAL] {msg}", exit_code=1)
            else:
                print(f"  [WARN] {msg}")
            continue
        matched = _resolve_boundary_selection(
            edge_str, segs, fatal=is_batch_traction)
        if not matched:
            msg = (f"未找到边 '{edge_str}' — 面力未施加. "
                   f"可用边: {', '.join(str(i+1) for i in range(len(segs)))}")
            if is_batch_traction:
                raise CliError(f"  [FATAL] {msg}", exit_code=1)
            else:
                print(f"  [WARN] {msg}")
            continue
        if profile in ('p', 'l'):
            _apply_traction_profile(
                mesh, segs, matched, edge_str, tx, ty, profile)
            continue
        for idx in matched:
            if idx >= len(segs):
                continue
            ns = segs[idx]['nodes']
            if profile == 'n':
                # 法向压力: 使用 mesh.add_pressure(), 自动计算 t = -p·n
                p = tx  # tx 存储压力幅值
                _each_edge(ns, lambda a, b: mesh.add_pressure(a, b, p))
                print(f"  压力: [{idx+1}] {segs[idx]['label']}  "
                      f"p={p:.3e} Pa (法向, t=-p·n)")
            else:
                _each_edge(ns, lambda a, b: mesh.add_traction(a, b, tx, ty))
                kind = ("t=f(x,y)" if (callable(tx) or callable(ty))
                        else f"t=({tx:.3e}, {ty:.3e})  [constant]")
                print(f"  面力: [{idx+1}] {segs[idx]['label']}  {kind}")


def _apply_concentrated_forces(config, mesh, region_registry, node_id_map,
                               source_geo_path):
    """集中力: --force node_id,fx,fy 或 physical_point_name,fx,fy."""
    if not config.force:
        return
    parts = config.force.split(',')
    if len(parts) != 3:
        raise CliError(
            f"  [FATAL] --force 需要恰好 3 个字段 (节点号,fx,fy), "
            f"得到: '{config.force}'",
            exit_code=1)
    target = parts[0].strip()
    try:
        fx, fy = float(parts[1]), float(parts[2])
    except ValueError:
        # 曾 float() 裸 ValueError 由顶层兜底 → [ERROR] 退出码 2,
        # 与字段数错误的 [FATAL] 退出码 1 风格分裂
        raise CliError(
            f"  [FATAL] --force 载荷分量无法解析: '{parts[1]},{parts[2]}' "
            f"— 需要两个数值 (例: --force 5,1e6,0)",
            exit_code=1)
    point_regions = (
        region_registry.by_name(target, dimension=0)
        if region_registry is not None else [])
    if point_regions:
        target_nodes = sorted({
            node for region in point_regions for node in region.node_ids
        })
        if len(target_nodes) != 1:
            raise CliError(
                f"  [FATAL] Physical Point '{target}' 映射到 "
                f"{len(target_nodes)} 个网格节点；集中力目标必须唯一。",
                exit_code=1)
        nid = target_nodes[0]
        source_label = f"Physical Point '{target}'"
    else:
        # 子进程 gmsh 路径不恢复 Physical Point regions — 用 Gmsh API
        # 读 .geo 同名 Point 坐标回退匹配最近节点
        nid, source_label, _point_dist, point_reason = (
            physical_point_from_geo(source_geo_path, target, mesh))
        if nid is not None and _point_dist > 0.0:
            print(f"  [WARN] Physical Point '{target}' 未落在网格节点上, "
                  f"已施加到最近节点 #{nid} (偏差 {_point_dist:.3g} m)")
        if nid is None:
            try:
                orig_nid = int(target)
            except ValueError:
                # 失败原因曾统一报"未找到" — 歧义/超距是不同问题,
                # 排查方向完全不同
                reason_hint = {
                    "ambiguous": "该名称映射到多个 Physical Point — 请确认"
                                 ".geo 中同名点唯一",
                    "outside_domain": "该点不在位移网格域内 (孔内/凹域/域外"
                                      "构造点) — 请检查 .geo 坐标",
                    "too_far": "该点距离最近网格节点超过 3 倍单元尺寸 — "
                               "请缩小网格或检查坐标",
                    "gmsh_unavailable": "Gmsh API 不可用, 无法解析 Physical "
                                        "Point — 请改用节点编号",
                    "no_geo_source": ".msh 直接输入没有源 .geo — 名称映射"
                                     "不可用, 请改用 Gmsh 节点编号",
                }.get(point_reason, "")
                raise CliError(
                    f"  [FATAL] 未找到 Physical Point '{target}'；"
                    "--force 第一项应为 Gmsh 节点号或 Physical Point 名称。"
                    + (f"\n        原因: {reason_hint}" if reason_hint else ""),
                    exit_code=1)
            # 将用户输入的 Gmsh 原始节点号映射到内部编号
            if node_id_map and orig_nid in node_id_map:
                nid = node_id_map[orig_nid]
            elif node_id_map:
                raise CliError(
                    f"  [FATAL] 节点 {orig_nid} 不在网格中 (有效范围: "
                    f"{min(node_id_map.keys())}~{max(node_id_map.keys())})",
                    exit_code=1)
            else:
                nid = orig_nid  # 非 Gmsh 网格, 直接使用
            source_label = f"Gmsh节点{orig_nid}"
    mesh.add_force(nid, fx, fy)
    print(f"  集中力: {source_label} → 内部节点{nid} "
          f"F=({fx:.3e}, {fy:.3e})")


def _apply_body_force(config, mesh, batch_mode):
    """体力 (bx, by [N/m^3], 支持含x/y表达式/元组/callable).

    返回 (bfx, bfy) 二元组供总表打印; 分量可为 callable (打印层显示
    f(x,y)), 整体 callable 时返回分量 lambda.
    """
    print("\n  --- 体力 (bx, by [N/m^3], 支持含x/y表达式) ---")
    if config.body is not None:
        body_input = config.body
        print(f"  {body_input}")
    elif batch_mode:
        body_input = ""  # 批处理: 无体力
    else:
        body_input = ask("  bx,by (回车跳过): ")
    # 跳过判据只对字符串做真值/等值判断 — ndarray/tuple 的 bool() 或
    # 与 '0' 比较会抛裸 ValueError (ambiguous truth value)
    zero_str = isinstance(body_input, str) and body_input.strip() in ("", "0")
    if body_input is not None and not zero_str:
        if callable(body_input):
            # 整体 callable: 契约 body(x, y) → (bx, by), 与 mesh.body_force
            # 的 callable 语义一致, 直接透传。不在此预调用 — 节点形心对
            # 带孔/凹域可能在材料域外 (环域形心在孔洞内), 合法体力会被
            # 误拒; 校验 (长度/数值/有限性) 由 evaluate_vector_field 在
            # 真实 Gauss 点完成 (复测 2026-08-02)。
            mesh.body_force = body_input
            print("  体力: 已设置 (callable)")
            # 返回契约必须是二元组 (runner 固定 bfx, bfy = apply_bcs(...)):
            # 分量独立求值, callable 时打印层显示 f(x,y)
            return (lambda x, y: body_input(x, y)[0],
                    lambda x, y: body_input(x, y)[1])
        if isinstance(body_input, (tuple, list, np.ndarray)):
            # 程序化配置 (AnalysisConfig(body=(bx, by))) — 字段声明为
            # object, 曾仅按字符串处理而抛 TypeError; 形状校验收敛到
            # 载荷 schema (mesh._check_load_pair) — 1/3 分量、NaN 带字段名
            bfx, bfy = _check_load_pair(body_input, "body")
        else:
            # 单个数字补逗号 (如 "0" "78000" 等交互快捷输入)
            if ',' not in body_input:
                body_input = '0,' + body_input  # 单个数字 = y分量 (如 -78000 = 重力)
            bfx, bfy = parse_vec2(body_input)
    else:
        bfx, bfy = 0.0, 0.0
    if callable(bfx) or callable(bfy) or bfx != 0.0 or bfy != 0.0:
        # 曾用 abs>1e-30 阈值: 微尺度模型合法小体力被静默丢弃
        mesh.body_force = (bfx, bfy)
        print("  体力: 已设置")
    return bfx, bfy


def apply_bcs(config, mesh, segs, region_registry, node_id_map,
               source_geo_path):
    """施加全部 BC/载荷: 固定边 (CLI/交互) → 面力 → 集中力 → 体力.

    每个载荷类型独立阶段函数 (可单测), 本函数只编排.
    返回 ``(bfx, bfy)`` 供总表打印 (callable 时返回原值).

    ``config`` 为 AnalysisConfig (类型化配置, 非 argparse Namespace).
    """
    batch_mode = is_batch_mode(config)
    _print_segment_menu(segs)
    _apply_fix_bcs(config, mesh, segs, batch_mode)
    _apply_tractions(config, mesh, segs, batch_mode)
    _apply_concentrated_forces(
        config, mesh, region_registry, node_id_map, source_geo_path)
    return _apply_body_force(config, mesh, batch_mode)
