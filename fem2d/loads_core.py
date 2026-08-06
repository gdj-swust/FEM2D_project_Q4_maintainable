"""等效节点载荷 — Bathe §4.2.1: 体力 + 面力 → 等效节点力向量

Bathe Eq 4.17: R = R_c + R_s + R_b
  R_c: 集中力 (直接加在节点上)
  R_s: 面力 → ∫ Nᵀ·t·dS  (3 点 Gauss-Legendre 线积分)
  R_b: 体力 → 委托当前 element kernel 的一致积分

数值积分:
  线 3 点 Gauss-Legendre (精度 degree 5): 标准 [-1,1] 区间
    xi = 0, ±√(3/5) ≈ ±0.7746, 权 5/9, 8/9, 5/9
    精度 degree 5 → 精确积分 ≤ 5 次多项式。
    CST/Q4 形函数沿直边为线性，面力常数/线性时 2 点即可精确;
    保留 3 点为兼容将来二次面力分布 (如流体压力沿高度线性变化)。
"""
import ast
import math as _m

import numpy as np

from .element import evaluate_vector_field
from .loads_schema import _load_component_ok

# 标准 Gauss-Legendre 3 点 ([-1, 1], 精度 degree 5)
# 格式: (weight, xi)
LINE_GAUSS = [
    (5/9, -0.774596669241483),   # xi = -√(3/5)
    (8/9,  0.0),                 # xi = 0
    (5/9,  0.774596669241483),   # xi = +√(3/5)
]

# 批量 3 点 Gauss 常量 (P-δ 向量化) — 由 LINE_GAUSS 派生, 与逐点路径
# 的标量运算逐位一致: Ni = 0.5·(1−xi), Nj = 0.5·(1+xi)
_LGAUSS_W = np.array([w for w, _ in LINE_GAUSS])
_LGAUSS_NI = 0.5 * (1.0 - np.array([xi for _, xi in LINE_GAUSS]))
_LGAUSS_NJ = 0.5 * (1.0 + np.array([xi for _, xi in LINE_GAUSS]))

# 退化边判据常量 — 同 mesh.boundary_outward_normal 的 64·eps·max(|坐标|)
# (有界于 tiny)。模块级常量避免逐边 np.finfo(float) 构造 (cProfile:
# 10000 次 __new__ ≈ 0.007s), 数值逐位相同。
_F_EPS = np.finfo(float).eps
_F_TINY = np.finfo(float).tiny

def assemble(mesh, n_dof):
    """组装全局等效节点力向量 F (Bathe §4.2.1)

    F = R_c + Σ R_s + Σ R_b
      R_c: 集中力 (直接加在节点 DOF)
      R_s: 面力 ∫ Nᵀ·t·dS (3 点 Gauss-Legendre 线积分)
      R_b: 体力 ∫ Nᵀ·f^B·dV (由 element kernel 积分)

    参数
    ----
    mesh : Mesh — 包含载荷定义的网格
    n_dof : int — 总自由度数

    返回
    ----
    F : (n_dof,) ndarray — 等效节点力向量
    """
    mesh.build_connectivity()
    if n_dof != 2 * mesh.n_nodes:
        # 独立调用传错 n_dof 会裸 IndexError (集中力越界写) — 契约前置
        raise ValueError(
            f"assemble_loads: n_dof={n_dof} 必须等于 2×节点数 "
            f"({2 * mesh.n_nodes})")
    F = np.zeros(n_dof)

    # (1) 集中力
    for cf in mesh.concentrated_forces:
        nid=cf["node"]; fx,fy=cf["force"]
        F[2*nid]+=fx; F[2*nid+1]+=fy

    # (2) 体力
    if mesh.body_force is not None:
        fe_batch = mesh.element_kernel.body_force_batch(
            mesh, mesh.body_force)
        if fe_batch is not None:
            fe_batch = np.asarray(fe_batch, dtype=float)
            if fe_batch.shape != mesh.element_dofs.shape:
                raise RuntimeError(
                    f"{mesh.element_kernel.name} kernel returned body-force "
                    f"shape {fe_batch.shape}; expected "
                    f"{mesh.element_dofs.shape}.")
            if not np.all(np.isfinite(fe_batch)):
                raise ValueError("Body-force integration returned NaN/Inf")
            F += np.bincount(
                mesh.element_dofs.ravel(),
                weights=fe_batch.ravel(),
                minlength=n_dof,
            )
        else:
            for eid in range(mesh.n_elements):
                fe = np.asarray(mesh.element_kernel.body_force_vector(
                    mesh, eid, mesh.body_force), dtype=float)
                dofs = mesh.element_dofs[eid]
                if fe.shape != dofs.shape:
                    raise RuntimeError(
                        f"{mesh.element_kernel.name} kernel returned body-force "
                        f"shape {fe.shape}; expected {dofs.shape}.")
                np.add.at(F, dofs, fe)

    # ── 面力/法向压力 — P-δ 向量化 ──
    # 记录序循环只做: 逐边校验 (节点/退化边/法向错误) + callable 逐点
    # 求值, 每条记录的 3 点 (tx, ty) 写入数组; 循环后一次性批量累积 —
    # np.add.at 按 (记录序, Gauss 点序, DOF 序) 展开, 与逐点路径的
    # 浮点求和顺序完全一致 (逐位等价)。边分类 (非法节点 id / 非网格边 /
    # 内部边) 与压力外法向 (Bathe §5.3.2, 与逐边 boundary_outward_normal
    # 代数等价) 批量计算, 校验错误按记录序延迟到该记录处理时抛出
    # (类型/消息不变; 面力路径同语义 — 逐边 _validate_boundary_edge 即
    # 此校验)。
    all_pairs = [st["nodes"] for st in mesh.surface_tractions]
    class_errors, class_eids = _classify_edges_batch(mesh, all_pairs)
    press_pos = [r for r, st in enumerate(mesh.surface_tractions)
                 if st.get("is_pressure")]
    press_errors = [class_errors[r] for r in press_pos]
    press_normals = _pressure_normals_batch(
        mesh, [all_pairs[r] for r in press_pos],
        press_errors, class_eids[press_pos])
    n_rec = len(mesh.surface_tractions)
    if n_rec:
        rec_lo = np.empty(n_rec, dtype=np.int64)
        rec_hi = np.empty(n_rec, dtype=np.int64)
        rec_L = np.empty(n_rec)
        rec_tx = np.empty((n_rec, 3))
        rec_ty = np.empty((n_rec, 3))
    else:
        rec_lo = rec_hi = rec_L = None
        rec_tx = rec_ty = None
    pr = 0  # 压力记录游标 (与 press_pairs 同序)

    # (3) 面力 / 法向压力
    for r, st in enumerate(mesh.surface_tractions):
        ni,nj = st["nodes"]; trac = st["traction"]
        xi_c,yi_c=mesh.nodes[ni]; xj_c,yj_c=mesh.nodes[nj]
        dx, dy = xj_c - xi_c, yj_c - yi_c
        L = float(np.hypot(dx, dy))
        # 零长判据基于该边端点的局部坐标尺度 — max(全局节点, 1.0)
        # 下限会让微米/纳米模型 (边长 1e-16) 全被判退化
        edge_ulp = 64.0 * _F_EPS * max(
            float(max(abs(xi_c), abs(xj_c), abs(yi_c), abs(yj_c))),
            _F_TINY)
        if L <= edge_ulp:
            raise ValueError(
                f"边 ({ni},{nj}) 长度 {L:.3e} 低于端点坐标 ULP "
                f"({edge_ulp:.3e}) — 节点重合或退化, 面力无法积分")
        is_pressure = st.get("is_pressure", False)

        if is_pressure:
            # 法向压力: t = -p·n — 外法向由批量路径实时计算 (与节点
            # 顺序无关)。trac = (p,) 为压力幅值。
            p_raw = trac[0]
            err = press_errors[pr]
            if err is not None:
                raise err
            nx, ny = press_normals[pr]
            pr += 1
            if callable(p_raw):
                # callable 契约逐点求值 (每 Gauss 点一次); 失败按序抛
                # 第一个 Gauss 点的错误
                for k in range(3):
                    xg = _LGAUSS_NI[k] * xi_c + _LGAUSS_NJ[k] * xj_c
                    yg = _LGAUSS_NI[k] * yi_c + _LGAUSS_NJ[k] * yj_c
                    try:
                        p_val = p_raw(xg, yg)
                    except Exception as error:
                        # 1/x 除零等表达式错误 — 裸异常无载荷上下文,
                        # 面力路径已有包装, 压力路径补齐
                        raise ValueError(
                            f"边 ({ni},{nj}) 压力表达式在 Gauss 点 "
                            f"({xg:.4g},{yg:.4g}) 求值失败: {error}") from error
                    if callable(p_val) or not _load_component_ok(p_val):
                        # callable 返回 str/序列/None/NaN 会裸 TypeError/ValueError
                        # (np.isfinite 真值判定) 无载荷上下文 — 统一走 loads_schema
                        # 标量校验, 与面力路径契约一致
                        raise ValueError(
                            f"边 ({ni},{nj}) 压力在 Gauss 点 "
                            f"({xg:.4g},{yg:.4g}) "
                            f"处非法值 {p_val!r} — 压力必须是单个有穷数值 "
                            f"(NaN/Inf/字符串/序列均拒绝)")
                    rec_tx[r, k] = -float(p_val) * nx
                    rec_ty[r, k] = -float(p_val) * ny
            else:
                p_val = float(p_raw)
                if not _load_component_ok(p_val):
                    # 常数非法值在第一个 Gauss 点被拒 (与逐点路径一致)
                    xg0 = _LGAUSS_NI[0] * xi_c + _LGAUSS_NJ[0] * xj_c
                    yg0 = _LGAUSS_NI[0] * yi_c + _LGAUSS_NJ[0] * yj_c
                    raise ValueError(
                        f"边 ({ni},{nj}) 压力在 Gauss 点 "
                        f"({xg0:.4g},{yg0:.4g}) "
                        f"处非法值 {p_val!r} — 压力必须是单个有穷数值 "
                        f"(NaN/Inf/字符串/序列均拒绝)")
                rec_tx[r, :] = -p_val * nx
                rec_ty[r, :] = -p_val * ny
        else:
            # 全局坐标面力 (tx, ty), 3 点 Gauss 积分
            # 边界边校验 (批量分类结果延迟抛出) — replace_elements 使
            # 该边变内部边后同样抛错, 与压力路径一致 (消息/类型同
            # 逐边 _validate_boundary_edge)
            err = class_errors[r]
            if err is not None:
                raise err
            if callable(trac) or (
                    isinstance(trac, (tuple, list))
                    and any(callable(c) for c in trac)):
                # callable 分量逐点求值 (每 Gauss 点一次)
                for k in range(3):
                    xg = _LGAUSS_NI[k] * xi_c + _LGAUSS_NJ[k] * xj_c
                    yg = _LGAUSS_NI[k] * yi_c + _LGAUSS_NJ[k] * yj_c
                    try:
                        tx_k, ty_k = evaluate_vector_field(trac, xg, yg)
                    except Exception as error:
                        # 1/x 除零 / sqrt(x-10) 域错误无载荷上下文裸抛
                        #
                        raise ValueError(
                            f"边 ({ni},{nj}) 面力表达式在 Gauss 点 "
                            f"({xg:.4g},{yg:.4g}) 求值失败: {error}") from error
                    rec_tx[r, k] = tx_k
                    rec_ty[r, k] = ty_k
            else:
                # 常数面力: 求值不依赖 Gauss 点坐标, 一次求值广播 3 点
                # (值与逐点求值逐位一致; 非法值在第一个 Gauss 点被拒)
                xg0 = _LGAUSS_NI[0] * xi_c + _LGAUSS_NJ[0] * xj_c
                yg0 = _LGAUSS_NI[0] * yi_c + _LGAUSS_NJ[0] * yj_c
                try:
                    tx0, ty0 = evaluate_vector_field(trac, xg0, yg0)
                except Exception as error:
                    raise ValueError(
                        f"边 ({ni},{nj}) 面力表达式在 Gauss 点 "
                        f"({xg0:.4g},{yg0:.4g}) 求值失败: {error}") from error
                rec_tx[r, :] = tx0
                rec_ty[r, :] = ty0
            # evaluate_vector_field 自检 NaN/Inf 先抛 — 复检为死防御
            # (与逐点路径一致, 消息格式保留)
            bad = ~(np.isfinite(rec_tx[r]) & np.isfinite(rec_ty[r]))
            if np.any(bad):
                k = int(np.flatnonzero(bad)[0])
                xgk = _LGAUSS_NI[k] * xi_c + _LGAUSS_NJ[k] * xj_c
                ygk = _LGAUSS_NI[k] * yi_c + _LGAUSS_NJ[k] * yj_c
                raise ValueError(
                    f"Traction callable returned NaN/Inf at "
                    f"Gauss point ({xgk:.4g},{ygk:.4g}) on edge ({ni},{nj})")
        rec_lo[r] = ni
        rec_hi[r] = nj
        rec_L[r] = L

    # 全量批量累积 — 求和顺序与逐点路径逐位一致
    if n_rec:
        _accumulate_surface_batch(F, mesh.thickness,
                                  rec_lo, rec_hi, rec_L, rec_tx, rec_ty)

    return F


def _accumulate_surface_batch(F, thickness, lo, hi, L, tx, ty):
    """全量面力/压力记录批量累积 (P-δ 向量化).

    求和顺序 = 逐点路径: 每记录按 (Gauss 点序, DOF 序 2ni, 2ni+1,
    2nj, 2nj+1) 展开, 记录间按 surface_tractions 序 — np.add.at 按
    索引数组顺序处理重复索引 (读-加-写无缓冲), 逐位一致 (由
    tests/test_p_delta_loads_batch.py 的逐位断言锁定)。
    """
    dofs = np.stack([2 * lo, 2 * lo + 1, 2 * hi, 2 * hi + 1], axis=1)
    fe = thickness * _LGAUSS_W[None, :] * L[:, None] / 2.0
    contrib = np.stack([
        fe * _LGAUSS_NI[None, :] * tx,
        fe * _LGAUSS_NI[None, :] * ty,
        fe * _LGAUSS_NJ[None, :] * tx,
        fe * _LGAUSS_NJ[None, :] * ty,
    ], axis=2)
    np.add.at(F, np.repeat(dofs, 3, axis=0).ravel(), contrib.ravel())


def _classify_edges_batch(mesh, pairs):
    """批量边分类 — 逐边 _validate_boundary_edge 语义, 单次向量化边表查询.

    压力与面力路径共用同一校验: 非网格边/内部边的错误类型与消息逐字
    同逐边 _validate_boundary_edge (内部边恒报告 2 个相邻单元 —
    build_edge_table 在构建时即拒绝非流形边); 非法节点 id 的 TypeError
    同 boundary_outward_normal 的 _validate_node_id。校验错误按记录序
    收集为异常对象, 由调用方在该记录处理时抛出 — 跨记录错误优先级
    与逐边路径一致 (先节点/边校验, 后 Gauss 点求值)。

    边表查询: edge_to_elems 是 topology_core 惰性视图, 逐键 .get() 为
    两次 searchsorted (cProfile: 5000 次 __getitem__ ≈ 0.044s, 面力组装
    热点)。这里把 (lo, hi) 编码为与 build_edge_table 相同的 int64 键
    (lo·n_nodes+hi, 节点 id < n_nodes 无碰撞), 对已按该键排序的边表做
    一次批量 searchsorted; 键编码在 n_nodes > 3e9 时溢出 → 退回逐键
    查询 (同 build_edge_table 的界, 该量级实际不可达)。

    参数
    ----
    mesh : Mesh
    pairs : list[(ni, nj)] — 记录节点对 (surface_tractions 记录序)

    返回
    ----
    (errors, eids) : errors[r] 为应抛出的异常对象或 None; eids 为
    (n,) int64 数组 — 边界边的唯一相邻单元 id, 其余为 -1。
    """
    n = len(pairs)
    errors = [None] * n
    eids = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return errors, eids
    mesh.build_connectivity()
    lo = np.empty(n, dtype=np.int64)
    hi = np.empty(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    for r, pair in enumerate(pairs):
        try:
            ni, nj = pair
            lo[r] = mesh._validate_node_id(ni)
            hi[r] = mesh._validate_node_id(nj)
        except Exception as err:
            errors[r] = err
            continue
        ok[r] = True
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return errors, eids
    table = mesh.edge_to_elems.table
    n_edges = int(table.lo.size)
    if n_edges == 0 or mesh.n_nodes > 3_000_000_000:
        # 空网格无边; 或键编码溢出风险 — 均退回逐键查询 (逐边路径原语)
        for r in idx:
            a, b = int(lo[r]), int(hi[r])
            key = (a, b) if a <= b else (b, a)
            edge_list = mesh.edge_to_elems.get(key, [])
            if len(edge_list) == 0:
                ni, nj = pairs[r]
                errors[r] = ValueError(
                    f"Edge ({ni},{nj}) is not a mesh edge — "
                    f"nodes must be connected by an element side.")
            elif len(edge_list) > 1:
                ni, nj = pairs[r]
                errors[r] = ValueError(
                    f"Edge ({ni},{nj}) is an interior edge shared by "
                    f"{len(edge_list)} elements. Surface tractions only "
                    f"supported on boundary edges.")
            else:
                eids[r] = edge_list[0]
        return errors, eids
    # 批量查询: 记录键与边表同编码, 一次 searchsorted
    lo_b, hi_b = lo[idx], hi[idx]
    qkey = (np.minimum(lo_b, hi_b) * np.int64(mesh.n_nodes)
            + np.maximum(lo_b, hi_b))
    tkey = table.lo * np.int64(mesh.n_nodes) + table.hi
    row = np.searchsorted(tkey, qkey, side="left")
    row_c = np.minimum(row, n_edges - 1)
    found = (row < n_edges) & (tkey[row_c] == qkey)
    interior = found & (table.counts[row_c] == 2)
    boundary = found & ~interior
    miss = np.flatnonzero(~found)
    for j in miss:
        r = int(idx[j])
        ni, nj = pairs[r]
        errors[r] = ValueError(
            f"Edge ({ni},{nj}) is not a mesh edge — "
            f"nodes must be connected by an element side.")
    inter = np.flatnonzero(interior)
    for j in inter:
        r = int(idx[j])
        ni, nj = pairs[r]
        errors[r] = ValueError(
            f"Edge ({ni},{nj}) is an interior edge shared by 2 elements. "
            f"Surface tractions only supported on boundary edges.")
    eids[idx[boundary]] = table.owners[row_c[boundary], 0]
    return errors, eids


def _pressure_normals_batch(mesh, pairs, errors, eids):
    """压力边批量外法向 — 与逐边 Mesh.boundary_outward_normal 代数等价.

    边分类 (节点 id / 非网格边 / 内部边) 由 _classify_edges_batch 完成;
    本函数对分类通过的边界边批量匹配单元局部 CCW 边 (local_edges),
    外法向 n = (dy/L, -dx/L) (Bathe §5.3.2: CCW 域外法向 = 切向顺时针
    90°)。退化边/匹配失败错误并入 errors (消息逐字同逐边路径), 由
    调用方在该记录处理时抛出。

    参数
    ----
    mesh : Mesh
    pairs : list[(ni, nj)] — 压力记录的节点对 (surface_tractions 记录序)
    errors : list[None | Exception] — _classify_edges_batch 的对应切片;
        退化/匹配失败错误就地写入
    eids : (len(pairs),) int64 — 边界边唯一相邻单元 id, 其余为 -1

    返回
    ----
    normals : list[None | (nx, ny)] — 按 pairs 序; 失败记录为 None
    """
    n = len(pairs)
    normals = [None] * n
    if n == 0:
        return normals
    b = np.flatnonzero(eids >= 0)
    if b.size == 0:
        return normals
    lo = np.array([int(pairs[r][0]) for r in b])
    hi = np.array([int(pairs[r][1]) for r in b])
    conn = mesh.elements[eids[b]]
    nrm = np.zeros((b.size, 2))
    matched = np.zeros(b.size, dtype=bool)
    deg = np.zeros(b.size, dtype=bool)
    Ls = np.zeros(b.size)
    ulps = np.zeros(b.size)
    for ia, ib in mesh.element_kernel.local_edges:
        ca, cb = conn[:, ia], conn[:, ib]
        slot = ((ca == lo) & (cb == hi)) | ((ca == hi) & (cb == lo))
        sel = slot & ~matched
        j = np.flatnonzero(sel)
        if len(j) == 0:
            continue
        xa = mesh.nodes[ca[j], 0]
        xb = mesh.nodes[cb[j], 0]
        ya = mesh.nodes[ca[j], 1]
        yb = mesh.nodes[cb[j], 1]
        dx, dy = xb - xa, yb - ya
        Lb = np.hypot(dx, dy)
        # 退化边 ULP 判据同 boundary_outward_normal (逐边坐标, 非全局)
        eulp = 64.0 * _F_EPS * np.maximum(
            np.maximum(np.abs(xa), np.abs(xb)),
            np.maximum(np.abs(ya), np.abs(yb)))
        eulp = np.maximum(eulp, _F_TINY)
        bad = Lb <= eulp
        if np.any(bad):
            jb = j[bad]
            deg[jb] = True
            Ls[jb] = Lb[bad]
            ulps[jb] = eulp[bad]
        g = j[~bad]
        nrm[g, 0] = dy[~bad] / Lb[~bad]
        nrm[g, 1] = -dx[~bad] / Lb[~bad]
        matched[sel] = True
    for jj, rr in enumerate(b):
        r = int(rr)
        if deg[jj]:
            errors[r] = ValueError(
                f"Zero-length edge ({lo[jj]},{hi[jj]}) "
                f"(L={Ls[jj]:.3e} <= ULP {ulps[jj]:.3e}).")
        elif not matched[jj]:
            errors[r] = RuntimeError(
                f"Boundary edge ({lo[jj]},{hi[jj]}) not found in "
                f"adjacent element {eids[r]}. This should not happen "
                f"— check mesh consistency.")
        else:
            normals[r] = (float(nrm[jj, 0]), float(nrm[jj, 1]))
    return normals


# ═══════════════════════════════════════════════════════════════
# 面力解析 & 剖面函数 — Bathe §4.2.1 (从 run.py 提取)
# ═══════════════════════════════════════════════════════════════

def parse_traction(s: str):
    """解析面力规格 → (edge_name, tx, ty, profile)

    格式:
      right:1e6,0        → 常数面力
      right:1e6,0:p      → 抛物线分布 (中心最大, 两端为零)
      right:1e6,0:l      → 线性分布 (一端最大, 另一端为零)
      right:1e6:n        → 法向压力 (t = -p·n)

    返回: (edge_name, tx, ty, profile)  其中 profile ∈ {None, 'p', 'l', 'n'}
    """
    if not isinstance(s, str):
        # 非 str (int/None) 会冒裸 TypeError ('in' 判据) — 类型契约前置
        raise ValueError(
            f"parse_traction: 需要面力规格字符串 (如 'right:1e6,0'), "
            f"got {type(s).__name__}: {s!r}")
    if ':' not in s:
        return None, 0, 0, None
    parts = s.split(':')
    # 分布类型只允许出现在第三段 (parts[2])。此前校验 parts[-1] 但取
    # parts[2], "edge:tx,ty:x:p" 这类畸形输入会静默接受 parts[2]='x'
    # 而忽略 parts[-1]='p' — 下游 _profile_factor('x') 直接返回 1.0,
    # 用户以为加了分布载荷, 实际是常数 (静默错误载荷)。
    if len(parts) == 3:
        if parts[2] not in ('p', 'l', 'n'):
            raise ValueError(
                f"面力分布类型 '{parts[2]}' 无效 — 仅支持 'p' (抛物线), 'l' (线性), "
                f"或 'n' (法向压力). 格式: edge:tx,ty[:p|l] 或 edge:p[:n]")
        edge = parts[0].strip()
        profile = parts[2]
        if profile == 'n':
            # 法向压力: edge:p:n → tx = 压力值, ty = 0 (占位, 实际按法向计算)
            try:
                p_val = float(parts[1].strip())
            except ValueError:
                raise ValueError(f"法向压力值无效: '{parts[1]}' — 需要单个数值, 如 right:1e6:n")
            tx, ty = p_val, 0.0
        else:
            tx, ty = parse_vec2(parts[1])
    elif len(parts) == 2:
        edge = parts[0].strip()
        tx, ty = parse_vec2(parts[1])
        profile = None
    else:
        raise ValueError(
            f"面力格式无效: '{s}'. 正确格式: edge:tx,ty 或 edge:tx,ty:p")
    return edge, tx, ty, profile


def _profile_factor(profile, coordinate):
    coordinate = min(max(float(coordinate), 0.0), 1.0)
    if profile == 'p':
        return 1.0 - (2.0 * coordinate - 1.0) ** 2
    if profile == 'l':
        return coordinate
    return 1.0


def make_edge_profile_func(
        tx, ty, profile, edge_start, edge_end,
        arc_start, total_length):
    """Build an O(1) arc-length traction profile for one polyline edge."""
    start = np.asarray(edge_start, dtype=float)
    end = np.asarray(edge_end, dtype=float)
    tangent = end - start
    length_squared = float(np.dot(tangent, tangent))
    edge_length = float(np.sqrt(length_squared))
    total_length = float(total_length)
    if (
            profile is None
            or length_squared <= np.finfo(float).tiny
            or total_length <= np.finfo(float).tiny):
        return tx, ty

    def _coordinate(x, y):
        local = float(np.dot(
            np.array([x, y], dtype=float) - start,
            tangent) / length_squared)
        local = min(max(local, 0.0), 1.0)
        return (float(arc_start) + local * edge_length) / total_length

    def fx(x, y, _tx=tx):
        # 坐标函数 (表达式面力) 与弧长分布的合法组合: f = tx(x,y)·s(arc)
        # 直接 _tx * factor 会对 callable 抛 TypeError
        value = _tx(x, y) if callable(_tx) else _tx
        return value * _profile_factor(profile, _coordinate(x, y))

    def fy(x, y, _ty=ty):
        value = _ty(x, y) if callable(_ty) else _ty
        return value * _profile_factor(profile, _coordinate(x, y))

    return fx, fy


# ═══════════════════════════════════════════════════════════════
# 表达式解析 — AST 白名单编译 (从 run.py 提取)
# ═══════════════════════════════════════════════════════════════

class _IntToFloat(ast.NodeTransformer):
    """整数字面量 → float (表达式 DoS 防护).

    Python 大整数运算不是常量时间: ``9**9**9**9`` 在 int 域逐级膨胀到
    10^92 位, 编译成功但求值永久挂起 (外部审查实测 timeout 10s 仍无
    结果)。整数字面量转 float 后 ``9.0**9.0**9.0**9.0`` 微秒级抛
    OverflowError — 资源耗尽变为响亮失败。

    转换在 AST 白名单校验之后、编译之前执行: 校验器仍按原始常量
    判定 (语义不变); 小整数 → float 精确, 大整数字面量的 float()
    舍入与 int×float 运算的隐式 int→float 转换逐位一致 → 既有合法
    表达式数值逐位不变 (test_s_alpha_security 逐位证明锁定)。
    """

    def visit_Constant(self, node):  # noqa: F401 — ast.NodeTransformer 框架 dispatch 调用
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(float(node.value)), node)
        return node


def _expr_has_spatial_names(p: str) -> bool:
    """含 x/y 变量 → 空间函数表达式 (AST Name 检查, 非 'x' in p 子串).

    ``'exp(1)'`` 里的 'x' 是函数名字符串而非空间变量 — 子串匹配把
    常数函数表达式误判为 callable, 与 sin(pi/2) 报错的既有契约自相
    矛盾 (外部审查实证)。语法错误时退回子串路由: 报错路径交
    _compile_expr 带表达式上下文 (消息不变)。
    """
    try:
        tree = ast.parse(p.strip(), mode="eval")
    except SyntaxError:
        return "x" in p or "y" in p
    return any(isinstance(n, ast.Name) and n.id in ("x", "y")
               for n in ast.walk(tree))


def _compile_expr(expr: str):
    """AST 白名单编译: x,y 空间表达式 → lambda x,y: <expr>

    仅允许: 变量 x/y, 数字, 算术运算, sin/cos/exp/sqrt/log/abs/tan/pi.
    禁止属性访问、下标、lambda、列表推导、任意函数调用。
    """
    _FUNCS = {
        'sin': _m.sin, 'cos': _m.cos, 'tan': _m.tan,
        'exp': _m.exp, 'sqrt': _m.sqrt, 'log': _m.log,
        'abs': abs, 'pi': _m.pi,
    }
    _ALLOWED_NAMES = {'x', 'y'} | set(_FUNCS.keys())
    _ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
    _ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    class _Validator(ast.NodeVisitor):
        def visit(self, node):
            if isinstance(node, ast.Expression):
                return self.visit(node.body)
            if isinstance(node, ast.BinOp):
                if type(node.op) not in _ALLOWED_BINOPS:
                    raise ValueError(
                        f"表达式 '{expr}' 不允许该运算符. "
                        f"仅允许: +, -, *, /, **.")
                self.visit(node.left); self.visit(node.right)
            elif isinstance(node, ast.UnaryOp):
                if type(node.op) not in _ALLOWED_UNARYOPS:
                    raise ValueError(
                        f"表达式 '{expr}' 不允许该一元运算符. 仅允许: +, -.")
                self.visit(node.operand)
            elif isinstance(node, ast.Name):
                if node.id not in _ALLOWED_NAMES:
                    raise ValueError(
                        f"表达式 '{expr}' 中不允许使用 '{node.id}'. "
                        f"仅允许: {sorted(_ALLOWED_NAMES)}.")
            elif isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                    raise ValueError(
                        f"表达式 '{expr}' 中仅允许数字常量, "
                        f"不允许: {type(node.value).__name__}.")
            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                    raise ValueError(
                        f"表达式 '{expr}' 中不允许该函数调用. "
                        f"仅允许: {sorted(_FUNCS.keys())}.")
                if node.keywords:
                    raise ValueError(f"表达式 '{expr}' 中不允许关键字参数.")
                for arg in node.args:
                    self.visit(arg)
            else:
                raise ValueError(
                    f"表达式 '{expr}' 包含不支持的操作 ({type(node).__name__}). "
                    f"仅允许: 算术运算 + 数学函数 {sorted(_FUNCS.keys())}.")

    try:
        tree = ast.parse(expr.strip(), mode='eval')
    except SyntaxError as error:
        # '1e6x,0' 等语法错误抛裸 SyntaxError 无表达式上下文
        #
        raise ValueError(
            f"表达式语法错误: {error.msg} — 表达式: '{expr}' "
            "(仅支持 数字/运算符/x/y/sin/cos/tan/exp/sqrt/log/abs/pi)") from None
    _Validator().visit(tree)

    # DoS 防护: int 常量 → float (见 _IntToFloat docstring)。fix_missing_locations
    # 补齐替换节点的行/列信息 — 替换后的常量节点缺少 location 会导致
    # compile() 抛 "PyCF_ONLY_AST" 类位置错误。
    tree = ast.fix_missing_locations(_IntToFloat().visit(tree))

    code = compile(tree, '<expr>', 'eval')
    return lambda x, y: eval(code, {"__builtins__": {}}, {"x": x, "y": y, **_FUNCS})  # nosec B307 — AST 白名单校验后执行, 见 ast_whitelist


def parse_vec2(s: str):
    """解析 "1e6,0" 或 "0,-1000*(1-y/2)" → (float|callable, float|callable)

    含 x/y → 编译为 lambda x,y: expr; 纯数字 → float
    """
    if not isinstance(s, str):
        # 非 str (None/int) 会冒裸 AttributeError (.replace) — 类型契约前置
        raise ValueError(
            f"parse_vec2: 需要载荷分量字符串 (如 '1e6,0'), "
            f"got {type(s).__name__}: {s!r}")
    # 全角逗号报"需要两个分量"会误导用户以为少写逗号
    parts = s.replace("，", ",").split(',')
    if len(parts) != 2:
        raise ValueError(
            f"需要恰好两个分量 (逗号分隔), 得到 {len(parts)} 个: '{s}'. "
            f"例: '1e6,0' 或 '0,-1000*(1-y/2)'")
    results = []
    for p in parts[:2]:
        p = p.strip()
        if p.lower() in ("nan", "+nan", "-nan", "inf", "+inf", "-inf",
                         "infinity", "+infinity", "-infinity"):
            # CLI 的 NaN/Inf 体力/面力会静默不施加载荷 (bc_apply 的
            # abs(bfx) > 1e-30 对 NaN 恒 False)
            raise ValueError(
                f"载荷分量 {p!r} 不是有限数值 — NaN/Inf 会被静默忽略")
        if not p:
            results.append(0.0)
            continue
        if _expr_has_spatial_names(p):
            results.append(_compile_expr(p))
        else:
            try:
                value = float(p)
            except ValueError:
                raise ValueError(
                    f"无法解析 '{p}' — 纯数字或含 x/y 的表达式 "
                    f"(例: 1e6 / 0,-1000*(1-y/2) / sin(pi*x/2),0). "
                    f"注意: 不含 x/y 变量的函数表达式 (如 sin(pi/2)、"
                    f"exp(1)) 不会被识别为空间函数, 请直接写数值 (如 1.0).")
            if not np.isfinite(value):
                # 数值溢出 (如 1e999 → inf) — CLI 静默忽略
                raise ValueError(
                    f"载荷分量 {p!r} 不是有限数值 ({value})")
            results.append(value)
    return results[0], results[1]
