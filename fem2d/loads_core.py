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
        # 独立调用传错 n_dof 曾裸 IndexError (集中力越界写) — 契约前置
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

    # (3) 面力 / 法向压力
    for st in mesh.surface_tractions:
        ni,nj = st["nodes"]; trac = st["traction"]
        xi_c,yi_c=mesh.nodes[ni]; xj_c,yj_c=mesh.nodes[nj]
        dx, dy = xj_c - xi_c, yj_c - yi_c
        L = float(np.hypot(dx, dy))
        # 零长判据基于该边端点的局部坐标尺度 — 曾用 max(全局节点, 1.0)
        # 下限, 微米/纳米模型 (边长 1e-16) 全被判退化
        edge_ulp = 64.0 * np.finfo(float).eps * max(
            float(max(abs(xi_c), abs(xj_c), abs(yi_c), abs(yj_c))),
            np.finfo(float).tiny)
        if L <= edge_ulp:
            raise ValueError(
                f"边 ({ni},{nj}) 长度 {L:.3e} 低于端点坐标 ULP "
                f"({edge_ulp:.3e}) — 节点重合或退化, 面力无法积分")
        is_pressure = st.get("is_pressure", False)

        if is_pressure:
            # 法向压力: t = -p·n — 外法向由当前网格几何实时计算
            # (Mesh.boundary_outward_normal, 与节点顺序无关; 不缓存,
            # 几何变更后自动跟随)。trac = (p,) 为压力幅值。
            p_raw = trac[0]
            nx, ny = mesh.boundary_outward_normal(ni, nj)
            for w, xi_g in LINE_GAUSS:
                Ni = 0.5*(1-xi_g); Nj = 0.5*(1+xi_g)
                xg = Ni*xi_c + Nj*xj_c; yg = Ni*yi_c + Nj*yj_c
                if callable(p_raw):
                    try:
                        p_val = p_raw(xg, yg)
                    except Exception as error:
                        # 1/x 除零等表达式错误 — 曾裸异常无载荷上下文,
                        # 面力路径已有包装, 压力路径补齐
                        raise ValueError(
                            f"边 ({ni},{nj}) 压力表达式在 Gauss 点 "
                            f"({xg:.4g},{yg:.4g}) 求值失败: {error}") from error
                else:
                    p_val = p_raw
                if callable(p_val) or not _load_component_ok(p_val):
                    # callable 返回 str/序列/None/NaN 曾裸 TypeError/ValueError
                    # (np.isfinite 真值判定) 无载荷上下文 — 统一走 loads_schema
                    # 标量校验, 与面力路径契约一致
                    raise ValueError(
                        f"边 ({ni},{nj}) 压力在 Gauss 点 ({xg:.4g},{yg:.4g}) "
                        f"处非法值 {p_val!r} — 压力必须是单个有穷数值 "
                        f"(NaN/Inf/字符串/序列均拒绝)")
                p_val = float(p_val)
                tx = -p_val * nx
                ty = -p_val * ny
                fe = mesh.thickness * w * L / 2.0
                F[2*ni] += fe * Ni * tx; F[2*ni+1] += fe * Ni * ty
                F[2*nj] += fe * Nj * tx; F[2*nj+1] += fe * Nj * ty
        else:
            # 全局坐标面力 (tx, ty), 3 点 Gauss 积分
            # 边界边校验 — 曾缺失: replace_elements 使该边变内部边后,
            # 压力路径抛错而面力路径静默施加到内部边
            mesh._validate_boundary_edge(ni, nj)
            for w, xi_g in LINE_GAUSS:
                Ni=0.5*(1-xi_g); Nj=0.5*(1+xi_g)
                xg=Ni*xi_c+Nj*xj_c; yg=Ni*yi_c+Nj*yj_c
                try:
                    tx,ty = evaluate_vector_field(trac, xg, yg)
                except Exception as error:
                    # 1/x 除零 / sqrt(x-10) 域错误曾无载荷上下文裸抛
                    #
                    raise ValueError(
                        f"边 ({ni},{nj}) 面力表达式在 Gauss 点 "
                        f"({xg:.4g},{yg:.4g}) 求值失败: {error}") from error
                if not (np.isfinite(tx) and np.isfinite(ty)):
                    raise ValueError(
                        f"Traction callable returned NaN/Inf at "
                        f"Gauss point ({xg:.4g},{yg:.4g}) on edge ({ni},{nj})")
                fe = mesh.thickness*w*L/2.0
                F[2*ni]+=fe*Ni*tx; F[2*ni+1]+=fe*Ni*ty
                F[2*nj]+=fe*Nj*tx; F[2*nj+1]+=fe*Nj*ty

    return F


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
        # 非 str (int/None) 曾冒裸 TypeError ('in' 判据) — 类型契约前置
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
        # 直接 _tx * factor 会对 callable 抛 TypeError (曾静默失败路径)
        value = _tx(x, y) if callable(_tx) else _tx
        return value * _profile_factor(profile, _coordinate(x, y))

    def fy(x, y, _ty=ty):
        value = _ty(x, y) if callable(_ty) else _ty
        return value * _profile_factor(profile, _coordinate(x, y))

    return fx, fy


# ═══════════════════════════════════════════════════════════════
# 表达式解析 — AST 白名单编译 (从 run.py 提取)
# ═══════════════════════════════════════════════════════════════

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
        # '1e6x,0' 等语法错误曾抛裸 SyntaxError 无表达式上下文
        #
        raise ValueError(
            f"表达式语法错误: {error.msg} — 表达式: '{expr}' "
            "(仅支持 数字/运算符/x/y/sin/cos/tan/exp/sqrt/log/abs/pi)") from None
    _Validator().visit(tree)

    code = compile(tree, '<expr>', 'eval')
    return lambda x, y: eval(code, {"__builtins__": {}}, {"x": x, "y": y, **_FUNCS})  # nosec B307 — AST 白名单校验后执行, 见 ast_whitelist


def parse_vec2(s: str):
    """解析 "1e6,0" 或 "0,-1000*(1-y/2)" → (float|callable, float|callable)

    含 x/y → 编译为 lambda x,y: expr; 纯数字 → float
    """
    if not isinstance(s, str):
        # 非 str (None/int) 曾冒裸 AttributeError (.replace) — 类型契约前置
        raise ValueError(
            f"parse_vec2: 需要载荷分量字符串 (如 '1e6,0'), "
            f"got {type(s).__name__}: {s!r}")
    # 全角逗号曾报"需要两个分量"误导用户以为少写逗号
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
            # CLI 的 NaN/Inf 体力/面力曾静默不施加载荷 (bc_apply 的
            # abs(bfx) > 1e-30 对 NaN 恒 False)
            raise ValueError(
                f"载荷分量 {p!r} 不是有限数值 — NaN/Inf 会被静默忽略")
        if not p:
            results.append(0.0)
            continue
        if 'x' in p or 'y' in p:
            results.append(_compile_expr(p))
        else:
            try:
                value = float(p)
            except ValueError:
                raise ValueError(
                    f"无法解析 '{p}' — 纯数字或含 x/y 的表达式 "
                    f"(例: 1e6 / 0,-1000*(1-y/2) / sin(pi*x/2),0). "
                    f"注意: 不含 x/y 的函数表达式 (如 sin(pi/2)) 不会被识别为空间函数, "
                    f"请直接写数值 (如 1.0).")
            if not np.isfinite(value):
                # 数值溢出 (如 1e999 → inf) — CLI 曾静默忽略
                raise ValueError(
                    f"载荷分量 {p!r} 不是有限数值 ({value})")
            results.append(value)
    return results[0], results[1]
