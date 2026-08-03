"""网格导入：Gmsh 3 节点三角形与 4 节点四边形 (2026-08: Abaqus 输入口已移除)。"""
import os
import re

import numpy as np


def read_geo_groups(geo_path, *, gmsh_module=None):
    """从 Gmsh .geo 文件提取 Physical Curve 名称和最终 entity ID.

    优先让 Gmsh API 解析命令语言、跨行定义、注释、宏和 CAD 重编号，
    但不生成网格。API 不可用时退回兼容旧环境的文本解析。

    Returns: {"name1":[ids], "name2":[ids], ...}  或 None
    """
    if not os.path.isfile(geo_path):
        return None
    from .gmsh_adapter import read_geo_curve_groups
    api_groups = read_geo_curve_groups(
        geo_path, gmsh_module=gmsh_module)
    if api_groups:
        # API 返回空 dict (例如 Physical Curve 引用了未定义曲线, Gmsh 只
        # 警告并跳过) 时不能视为"无定义" — 落入下方文本解析回退。
        return api_groups

    groups = {}
    with open(geo_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # Physical Curve("name", ID) = {1,2,3} 或 {300:319}
    for m in re.finditer(r'Physical\s+Curve\s*\(\s*"([^"]+)"[^)]*\)\s*=\s*\{([^}]+)\}', content):
        name = m.group(1)
        ids = []
        for part in m.group(2).split(','):
            part = part.strip()
            if ':' in part:
                a, b = part.split(':', 1)
                ids.extend(range(int(a), int(b) + 1))
            elif part.isdigit():
                ids.append(int(part))
        groups.setdefault(name, []).extend(ids)
        groups[name] = sorted(set(groups[name]))
    return groups if groups else None

# ═══════════════════════════════════════════════════════════════
# .spec / .geo 配置解析 (从 run.py 提取)
# ═══════════════════════════════════════════════════════════════

def parse_spec_config(filepath):
    """解析 .spec 键值配置文件 → dict (与 argparse namespace 兼容)

    格式: key = value
    支持键: mesh, E, nu, t, plane, fix, traction, body, save, no-plot

    注意与 scripts/geo_spec.parse_spec (.txt 中文几何描述) 区分 —
    两者同名曾是输入端分叉温床 (2026-08-03 审计后改名消歧)。

    格式错误 (漏写 = / 空值) 必须响亮报错 — 曾静默丢键, 约束/载荷
    无声消失, 与 .txt 入口的缺值报错行为分叉。
    """
    spec = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if lineno == 1:
                # Windows 记事本 UTF-8 BOM 曾把首行键变成 '﻿mesh',
                # mesh 键丢失 → FATAL 报空路径
                line = line.lstrip("﻿")
            # 剥离行尾注释 (例: mesh = test.txt  # 说明)
            line = line.split('#')[0].strip()
            if not line:
                continue
            if '=' not in line:
                raise ValueError(
                    f"{filepath}:{lineno} 缺少 '=' — 格式: 键 = 值 "
                    f"(如 'fix = left'), 得到: '{line}'")
            key, val = line.split('=', 1)
            key = key.strip().replace('-', '_')  # no-plot → no_plot
            val = val.strip()
            if not val:
                raise ValueError(
                    f"{filepath}:{lineno} 键 '{key}' 空值 — 格式: "
                    f"{key} = <值> (如 '{key} = left')")
            spec[key] = val
    return spec


def parse_geo_fem_config(geo_path: str):
    """从 .geo 文件读取 @FEM: 注释 → BC 配置

    格式错误必须响亮失败 (带文件名/行号/原始内容) — 曾 traction 字段
    不足静默丢弃、空 @FEM:pressure= 连正则都不匹配, 求解成功但工况
    已变: 静默丢载荷是最危险的错误类型。
    """
    if not geo_path or not os.path.isfile(geo_path):
        return {"fix": [], "traction": [], "pressure": [], "body": None}
    config = {"fix": [], "traction": [], "pressure": [], "body": None}
    with open(geo_path, 'r', encoding='utf-8', errors='ignore') as f:
        for lineno, line in enumerate(f, 1):
            m = re.search(r'@FEM:(\w+)=(.*)', line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip()
            if not val:
                raise ValueError(
                    f".geo @FEM: 载荷配置错误 — {geo_path}:{lineno}: "
                    f"'{line.strip()}' — '{key}' 缺少值")
            if key == 'fix':
                # 格式: 单个边/物理组名 (无逗号)
                if ',' in val:
                    raise ValueError(
                        f".geo @FEM: 载荷配置错误 — {geo_path}:{lineno}: "
                        f"'{line.strip()}' — fix 需要单个边名, 含逗号")
                config['fix'].append(val)
            elif key == 'traction':
                # 格式: edge,tx,ty 或 edge,tx,ty,profile — 字段过多曾
                # 静默丢弃, 载荷消失无提示
                parts = val.split(',')
                if len(parts) < 3 or len(parts) > 4:
                    raise ValueError(
                        f".geo @FEM: 载荷配置错误 — {geo_path}:{lineno}: "
                        f"'{line.strip()}' — traction 需要 edge,tx,ty "
                        f"(可加 ,profile), 实际 {len(parts)} 个字段")
                if len(parts) == 4:
                    config['traction'].append(f"{parts[0]}:{parts[1]},{parts[2]}:{parts[3]}")
                else:
                    config['traction'].append(f"{parts[0]}:{parts[1]},{parts[2]}")
            elif key == 'pressure':
                # 格式: edge,p (法向压力, t = -p·n, profile='n')
                parts = val.split(',')
                if len(parts) != 2:
                    raise ValueError(
                        f".geo @FEM: 载荷配置错误 — {geo_path}:{lineno}: "
                        f"'{line.strip()}' — pressure 需要 edge,p, "
                        f"实际 {len(parts)} 个字段")
                config['pressure'].append(f"{parts[0]}:{parts[1]}:n")
            elif key == 'body':
                # 格式: bx,by (体力两分量)
                parts = val.split(',')
                if len(parts) != 2:
                    raise ValueError(
                        f".geo @FEM: 载荷配置错误 — {geo_path}:{lineno}: "
                        f"'{line.strip()}' — body 需要 bx,by, "
                        f"实际 {len(parts)} 个字段")
                config['body'] = val
            else:
                # 未知 @FEM: 键 (如 geo_spec 曾生成的 bc=) 曾静默丢弃,
                # 载荷消失无提示
                print(f"  [WARN] .geo @FEM: 未知键 '{key}' 已忽略 — "
                      "支持: fix/traction/pressure/body")
    return config


def merge_geo_fem_config(geo_fem, config, *, verbose=True):
    """按优先级把 @FEM 配置合并进 config — CLI 显式参数 > .geo 配置.

    配置内部 traction+pressure 可并存 (同一配置源); CLI 已显式指定
    ``--traction`` 时配置载荷整体跳过并告警。input_source.resolve_geo
    与 runner._apply_geo_fem_config 共用 (曾各自实现, 逻辑分叉)。

    ``verbose=True`` 时打印 [auto]/[WARN] 信息。返回是否应用了配置。
    """
    applied = False
    if not config.fix and geo_fem['fix']:
        config.fix = ';'.join(geo_fem['fix'])
        applied = True
        if verbose:
            print(f"  [auto] fix: {config.fix}")
    elif geo_fem['fix'] and config.fix and verbose:
        # 与 traction 分支一致: .geo @FEM:fix 被 CLI --fix 覆盖须提示 —
        # 曾静默替换, 用户以为 .geo 约束已生效
        print(
            "  [WARN] .geo 配置含 fix 但 CLI 已显式指定 --fix — "
            "按 CLI 优先, .geo 约束未施加")
    config_loads = list(geo_fem['traction']) + list(geo_fem['pressure'])
    if config_loads and not config.traction:
        config.traction = ';'.join(config_loads)
        applied = True
        if verbose:
            print(f"  [auto] traction/pressure: {config.traction}")
    elif config_loads and config.traction and verbose:
        print(
            "  [WARN] .geo 配置含面力/压力但 CLI 已显式指定 --traction — "
            "按 CLI 优先, 配置载荷未施加 (如需叠加请合并到 --traction)。")
    if config.body is None and geo_fem['body']:
        config.body = geo_fem['body']
        applied = True
        if verbose:
            print(f"  [auto] body: {config.body}")
    elif geo_fem['body'] and config.body is not None and verbose:
        print(
            "  [WARN] .geo 配置含体力但 CLI 已显式指定 --body — "
            "按 CLI 优先, .geo 体力未施加")
    return applied


# ═══════════════════════════════════════════════════════════════
# 网格导入后校验 — Gmsh checkMeshCoherence() 模式
# ═══════════════════════════════════════════════════════════════

class MeshValidationError(Exception):
    """网格校验致命错误 — 无法继续求解."""


# ── validate_mesh 的独立校验步骤 (每步可单测, 2026-08 拆分) ──

_LOCAL_EDGES = {
    3: ((0, 1), (1, 2), (2, 0)),
    4: ((0, 1), (1, 2), (2, 3), (3, 0)),
}


def _coordinate_ulp(nodes):
    """坐标 ULP 容差 — 基于网格局部坐标尺度, 不强制下限 1.0.

    曾用 max(..., 1.0) 下限: 微米/纳米模型 (坐标 1e-16) 的容差被抬到
    ~1.4e-14, 正常节点被判重复、正常边被判退化。
    tiny 下限仅防纯零坐标时出现 0 容差, 不再放大微尺度模型。
    """
    return 64.0 * np.finfo(float).eps * max(
        float(np.max(np.abs(nodes))), np.finfo(float).tiny)


def _validate_element_type(elements, elem_type):
    """elem_type 声明的节点数与网格拓扑一致, 不一致则致命."""
    if elem_type is not None:
        from .element import get_element_kernel
        expected = get_element_kernel(elem_type).nodes_per_element
        if elements.shape[1] != expected:
            raise MeshValidationError(
                f"elem_type={elem_type} requires {expected} nodes per "
                f"element, but mesh has {elements.shape[1]}.")


def _validate_node_indices(elements, n_node):
    """节点索引合法性 (负值/越界) — 非法则无法继续校验."""
    errors = []
    neg_mask = elements < 0
    if np.any(neg_mask):
        bad = np.unique(elements[neg_mask])
        errors.append(f"负节点索引: {bad.tolist()}")
    out_mask = elements >= n_node
    if np.any(out_mask):
        bad = np.unique(elements[out_mask])
        errors.append(f"节点索引越界: {bad.tolist()} (n_nodes={n_node})")
    if errors:
        raise MeshValidationError("; ".join(errors))


def _element_span(nodes, elements):
    """每单元 x/y 跨度的最大值 (重复节点容差与零边容差共用)."""
    return np.max(np.ptp(nodes[elements], axis=1), axis=1)


def _resolve_duplicate_tolerance(h_elem, tol, coord_ulp):
    """重复节点容差: 最小非零单元跨度×1e-6 + ULP — 全局 span×1e-8
    在多尺度网格 (1 m + 1 nm) 会把纳米正常节点误报为重复."""
    if tol is not None:
        return tol
    h_pos = h_elem[h_elem > 0]
    h_min = float(np.min(h_pos)) if h_pos.size else 0.0
    return max(h_min * 1e-6, coord_ulp)


def _find_duplicate_nodes(nodes, tol):
    """cKDTree 全量空间查询重复节点 — 不漏非相邻近点."""
    from scipy.spatial import cKDTree
    tree = cKDTree(nodes)
    pairs = tree.query_pairs(tol, output_type='ndarray')
    return (f"{len(pairs)} 对重复节点 (tol={tol:.1e})"
            if len(pairs) > 0 else None, len(pairs))


def _find_duplicate_elements(elements):
    """节点号排序后判重的拓扑重复单元."""
    sorted_elems = np.sort(elements, axis=1)
    seen = set()
    dup_elem_ids = []
    for eid in range(elements.shape[0]):
        key = tuple(sorted_elems[eid])
        if key in seen:
            dup_elem_ids.append(eid)
        seen.add(key)
    return (f"{len(dup_elem_ids)} 个重复单元: {dup_elem_ids[:10]}"
            if dup_elem_ids else None, len(dup_elem_ids))


def _find_zero_edges(nodes, elements, h_elem, coord_ulp):
    """退化边: 逐单元局部容差 (1e-8×单元跨度 + ULP)."""
    edge_tol = np.maximum(1e-8 * h_elem, coord_ulp)
    zero_edges = []
    for eid, conn in enumerate(elements):
        for ia, ib in _LOCAL_EDGES[elements.shape[1]]:
            a, b = int(conn[ia]), int(conn[ib])
            if a == b or np.linalg.norm(nodes[a] - nodes[b]) < edge_tol[eid]:
                zero_edges.append((eid, ia, ib))
    return (f"{len(zero_edges)} 条退化边 (边长 < 局部尺度×1e-8)"
            if zero_edges else None, len(zero_edges))


def _signed_area(coords):
    """单元有向面积: 三角形 = 单三角, 四边形 = 两三角之和."""
    a1 = 0.5 * ((coords[1, 0] - coords[0, 0]) *
                (coords[2, 1] - coords[0, 1]) -
                (coords[2, 0] - coords[0, 0]) *
                (coords[1, 1] - coords[0, 1]))
    if coords.shape[0] == 3:
        return a1
    a2 = 0.5 * ((coords[3, 0] - coords[2, 0]) *
                (coords[0, 1] - coords[2, 1]) -
                (coords[0, 0] - coords[2, 0]) *
                (coords[3, 1] - coords[2, 1]))
    return a1 + a2


def _find_degenerate_elements(nodes, elements):
    """退化单元 (零面积/负面积/自交四边形)."""
    degenerate = []
    for eid, conn in enumerate(elements):
        area = _signed_area(nodes[conn])
        bad = area <= 0
        if not bad and len(conn) == 4:
            # 蝴蝶形 (自交) 四边形: 两片三角有向面积异号, 净面积仍为正
            # 而漏检 — 求解器 Jacobian 检查会以 inverted 拒绝, 导入校验
            # 必须给出同样诊断
            c = nodes[conn]
            a1 = 0.5 * ((c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) -
                        (c[2, 0] - c[0, 0]) * (c[1, 1] - c[0, 1]))
            a2 = 0.5 * ((c[3, 0] - c[2, 0]) * (c[0, 1] - c[2, 1]) -
                        (c[0, 0] - c[2, 0]) * (c[3, 1] - c[2, 1]))
            bad = a1 * a2 < 0
        if bad:
            degenerate.append((eid, area))
    return (f"{len(degenerate)} 个退化单元 (面积 ≤ 0 或自交)"
            if degenerate else None, len(degenerate))


def _find_orphan_nodes(elements, n_node):
    """孤立节点 (不被任何单元引用)."""
    referenced = np.unique(elements)
    orphan_mask = np.ones(n_node, dtype=bool)
    orphan_mask[referenced] = False
    orphan_nodes = np.flatnonzero(orphan_mask)
    return (f"{len(orphan_nodes)} 个孤立节点 (不被任何单元引用)"
            if len(orphan_nodes) > 0 else None, orphan_nodes)


def _edge_to_elem_map(elements):
    """边 → 单元列表映射 (非流形边/顶点共用)."""
    from collections import defaultdict
    edge_to_elems = defaultdict(list)
    for eid, conn in enumerate(elements):
        for ia, ib in _LOCAL_EDGES[elements.shape[1]]:
            a, b = int(conn[ia]), int(conn[ib])
            edge_to_elems[(min(a, b), max(a, b))].append(eid)
    return edge_to_elems


def _find_non_manifold_edges(edge_to_elems):
    """非流形边 (被 ≥3 个单元共享)."""
    non_manifold = [(e, eids) for e, eids in edge_to_elems.items()
                    if len(eids) >= 3]
    return (f"{len(non_manifold)} 条非流形边 (被 ≥3 个单元共享)"
            if non_manifold else None, len(non_manifold))


def _find_non_manifold_nodes(elements, edge_to_elems):
    """非流形顶点: 顶点周围单元不连通 (BFS)."""
    from collections import defaultdict as dd
    from collections import deque
    elem_adj = dd(set)
    for _, eids in edge_to_elems.items():
        if len(eids) == 2:
            e1, e2 = eids
            elem_adj[e1].add(e2)
            elem_adj[e2].add(e1)
    node_to_elems = dd(list)
    for eid, conn in enumerate(elements):
        for nid in conn:
            node_to_elems[int(nid)].append(eid)
    non_manifold_nodes = []
    for nid, eids in node_to_elems.items():
        # 去重: list 的 `in` 是 O(n), 且连接表重复引用同一单元时
        # len(list) 虚高会误报非流形
        eid_set = set(eids)
        if len(eid_set) <= 1:
            continue  # 边界节点, 跳过
        start = next(iter(eid_set))
        visited = {start}
        q = deque([start])
        while q:
            cur = q.popleft()
            for nb in elem_adj[cur]:
                if nb in eid_set and nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        if len(visited) != len(eid_set):
            non_manifold_nodes.append(nid)
    return (f"{len(non_manifold_nodes)} 个非流形顶点"
            if non_manifold_nodes else None, len(non_manifold_nodes))


def validate_mesh(nodes, elements, elem_type=None, tol=None):
    """导入后网格全量校验 (Bathe Table 4.3 + Gmsh checkMeshCoherence).

    校验项:
      1. 节点索引合法性 (负值 / 越界)
      2. 重复节点 (空间距离 < tol)
      3. 重复单元 (拓扑相同)
      4. 退化边 (边长为零)
      5. 退化单元 (零面积 / 负面积)
      6. 孤立节点 (不被任何单元引用)
      7. 非流形边 (边被 3+ 个单元共享)
      8. 非流形顶点 (顶点周围单元不连通)

    参数
    ----
    nodes : (n_nodes, 2) ndarray
    elements : (n_elem, npe) ndarray of int
    elem_type : str — "CPS3" (三角形) or "CPS4" (四边形)
    tol : float or None — 重复节点容差; None 则 auto = min(edge_length)*1e-3

    返回
    ----
    report : dict
        {"ok": bool, "errors": [...], "warnings": [...],
         "stats": {...}, "orphan_nodes": ndarray}
    """
    nodes = np.asarray(nodes, dtype=float)
    if not np.all(np.isfinite(nodes)):
        # cKDTree 对 NaN/Inf 抛裸 scipy ValueError, 与网格无关的底层报错
        # 会把用户引向错误方向
        raise MeshValidationError(
            "节点坐标含 NaN/Inf — 网格导入前必须清理")
    # 先验证连接关系的有限性/整数值, 再转换 — 曾先 dtype=int 把浮点
    # 1.9 静默截断成 1, 校验错误通过
    elems_raw = np.asarray(elements)
    if not np.issubdtype(elems_raw.dtype, np.integer):
        if not np.all(np.isfinite(elems_raw)):
            raise MeshValidationError(
                "Element node indices must be finite integers — "
                f"NaN/Inf found (e.g. {elems_raw.flat[0]})")
        # 拓扑编号必须严格整数: np.allclose 的相对容差会把 1.000001
        # 当整数, 静默改变拓扑
        bad = elems_raw != np.rint(elems_raw)
        if np.any(bad):
            first_bad = elems_raw[bad].flat[0]
            raise MeshValidationError(
                "Element node indices must be integers — "
                f"non-integer value found: {first_bad} "
                f"(位置 {int(np.flatnonzero(bad)[0])})")
        elems_raw = np.rint(elems_raw)
    elements = elems_raw.astype(np.int64, copy=False)
    if elements.shape[1] not in _LOCAL_EDGES:
        # 5/6 列连接曾在 _find_zero_edges 抛裸 KeyError
        raise MeshValidationError(
            f"单元连接宽度 {elements.shape[1]} 不受支持 — 仅支持 "
            f"{sorted(_LOCAL_EDGES)} 节点 (三角/四边形)")
    if elements.shape[0] == 0:
        # 空单元集让所有校验循环空转, ok=True 误导下游
        raise MeshValidationError("网格不包含任何单元")
    n_node = nodes.shape[0]
    n_elem = elements.shape[0]
    errors, warnings = [], []
    coord_ulp = _coordinate_ulp(nodes)

    _validate_element_type(elements, elem_type)
    _validate_node_indices(elements, n_node)

    h_elem = _element_span(nodes, elements)
    tol = _resolve_duplicate_tolerance(h_elem, tol, coord_ulp)

    # 每步返回 (消息 or None, 计数/数据); 消息非 None 才记入报告
    msg, n_dup_nodes = _find_duplicate_nodes(nodes, tol)
    if msg: warnings.append(msg)
    msg, n_dup_elems = _find_duplicate_elements(elements)
    if msg: errors.append(msg)
    msg, n_zero_edges = _find_zero_edges(nodes, elements, h_elem, coord_ulp)
    if msg: errors.append(msg)
    msg, n_degenerate = _find_degenerate_elements(nodes, elements)
    if msg: errors.append(msg)
    msg, orphan_nodes = _find_orphan_nodes(elements, n_node)
    if msg: warnings.append(msg)
    edge_to_elems = _edge_to_elem_map(elements)
    msg, n_non_manifold_edges = _find_non_manifold_edges(edge_to_elems)
    if msg: errors.append(msg)
    msg, n_non_manifold_nodes = _find_non_manifold_nodes(
        elements, edge_to_elems)
    if msg: errors.append(msg)

    # ── 9. 全局连通性: 互不接触的多个域
    # 通过全部校验, 但求解时刚体检查会以奇异拒绝 — 提前给出诊断
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    lo = np.array([min(e) for e in edge_to_elems], dtype=int)
    hi = np.array([max(e) for e in edge_to_elems], dtype=int)
    adj = csr_matrix(
        (np.ones(len(lo)), (lo, hi)), shape=(n_node, n_node))
    n_comp, _ = connected_components(adj, directed=False)
    if n_comp > 1:
        warnings.append(
            f"{n_comp} 个互不连通的网格分量 — 请确认是否是有意的多域模型"
            " (每个分量需独立满足刚体约束)")

    ok = len(errors) == 0
    return {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "stats": {
            "n_nodes": n_node, "n_elements": n_elem,
            "orphan_nodes": len(orphan_nodes),
            "degenerate_elems": n_degenerate,
            "zero_edges": n_zero_edges,
            "duplicate_elems": n_dup_elems,
            "duplicate_nodes": n_dup_nodes,
            "non_manifold_edges": n_non_manifold_edges,
            "non_manifold_nodes": n_non_manifold_nodes,
        },
        "orphan_nodes": orphan_nodes,
    }
