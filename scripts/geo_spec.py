"""几何描述器 — 中文文本 → Gmsh .geo → .msh 网格

不需要学 Gmsh 脚本，几行中文就能生成网格。

支持:
  矩形板          宽 高
  矩形板+内孔     内孔 圆 x= y= r=
  圆板            外径 [内径]

用法:
  python geo_spec.py 描述文件.txt

示例 — 带孔矩形板 (test_spec.txt):
  类型 矩形板
  宽 3.0
  高 2.0
  内孔 圆 x=0.8 y=0.3 r=0.3
  网格 0.1
  边界 左 固定
  边界 右 拉力 1e6
"""
import math
import os
import sys

try:
    from .gmsh_runner import run_gmsh
except ImportError:  # ``python scripts/geo_spec.py`` / run.py path import
    from gmsh_runner import run_gmsh


# ══════════════════════════════════════════════════════
# 解析器
# ══════════════════════════════════════════════════════

_VALID_KEYS = {"类型", "宽", "高", "外径", "内径", "外半径", "内半径",
               "网格", "网格尺寸", "内孔", "边界", "体力"}
_TYPE_MAP = {"矩形板": "rect", "矩形": "rect", "圆板": "circle",
             "圆环": "annulus", "圆": "circle"}
_BC_TYPES = {"固定", "压力", "拉力", "面力"}


def parse_spec(filepath):
    """解析中文几何描述 → spec dict"""
    import unicodedata
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    spec = {
        "type": None, "params": {}, "holes": [],
        "mesh_size": 0.1, "boundaries": [],
        "body_force": None,
    }

    def require_positive(value, what, lineno):
        try:
            v = float(value)
        except ValueError:
            raise ValueError(
                f"{filepath}:{lineno} {what} 值无法解析: '{value}'") from None
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError(
                f"{filepath}:{lineno} {what} 必须为正数且有限, 得到: {v!r} — "
                "负值/零/NaN 曾静默生成镜像几何或空网格")
        return v

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if lineno == 1:
            line = line.lstrip("﻿")  # Windows 记事本 UTF-8 BOM (曾吞掉首行 类型)
        line = unicodedata.normalize("NFKC", line)  # 全角＝，３．０ → 半角
        parts = line.replace("=", " ").split()
        if not parts:
            continue
        key, rest = parts[0], parts[1:]

        if key not in _VALID_KEYS:
            # 拼错/未知键曾整行静默丢弃 — .spec 路径有 WARN, 此处补上
            # (厚度 0.5 曾无痕消失, 用户以为已设置) (审计 2026-08-03)
            key_safe = "".join(c for c in key if c.isprintable())
            print(f"  [WARN] {filepath}:{lineno} 未知键 '{key_safe}' 已忽略 — "
                  f"可用键: {sorted(_VALID_KEYS)}")
            continue
        if not rest:
            # 合法键无值曾静默丢弃, 用户以为已设置 (审计 2026-08-03)
            raise ValueError(
                f"{filepath}:{lineno} 键 '{key}' 缺少数值 — 格式: "
                f"{key} <数值> (如 '{key} 0.5')")
        if key == "类型":
            if rest[0] not in _TYPE_MAP:
                raise ValueError(
                    f"{filepath}:{lineno} 未知类型 '{rest[0]}' — 支持: "
                    f"{sorted(_TYPE_MAP)}")
            spec["type"] = _TYPE_MAP[rest[0]]
            if len(rest) > 1:
                raise ValueError(
                    f"{filepath}:{lineno} '类型' 多余参数: "
                    f"{' '.join(rest[1:])}")
        elif key in ("宽", "高", "外径", "内径", "外半径", "内半径",
                     "网格", "网格尺寸"):
            if len(rest) > 1:
                # 多余 token 曾静默丢弃 (宽 3 4 取 3 丢 4) (审计 2026-08-03)
                raise ValueError(
                    f"{filepath}:{lineno} '{key}' 多余参数: "
                    f"{' '.join(rest[1:])} — 每个键只需一个数值")
            if key == "宽":
                spec["params"]["width"] = require_positive(rest[0], "宽", lineno)
            elif key == "高":
                spec["params"]["height"] = require_positive(rest[0], "高", lineno)
            elif key == "外径":
                spec["params"]["outer_r"] = require_positive(rest[0], "外径", lineno) / 2.0
            elif key == "内径":
                spec["params"]["inner_r"] = require_positive(rest[0], "内径", lineno) / 2.0
            elif key in ("外半径", "outer_r"):
                spec["params"]["outer_r"] = require_positive(rest[0], "外半径", lineno)
            elif key in ("内半径", "inner_r"):
                spec["params"]["inner_r"] = require_positive(rest[0], "内半径", lineno)
            else:  # 网格 / 网格尺寸
                # 负值曾使 Gmsh 无限细化挂起, 0 曾静默用默认密度
                spec["mesh_size"] = require_positive(rest[0], "网格尺寸", lineno)
        elif key == "内孔":
            hole = _parse_hole(" ".join(rest), filepath, lineno)
            if hole: spec["holes"].append(hole)
        elif key == "边界":
            bc = _parse_bc(" ".join(rest), filepath, lineno)
            for prev in spec["boundaries"]:
                if prev["edge"] == bc["edge"]:
                    # 同一边多类型 (固定+拉力) 曾并存: 约束把载荷吞掉,
                    # 求解成功但载荷静默消失 (审计 2026-08-03)
                    raise ValueError(
                        f"{filepath}:{lineno} 边界 '{bc['edge']}' 重复声明 "
                        f"(已声明 '{prev['bc']}') — 同一边只能声明一种载荷, "
                        "固定/压力/拉力 互斥")
            spec["boundaries"].append(bc)
        elif key == "体力":
            raw = "".join(rest).replace("，", ",")
            values = [x.strip() for x in raw.split(",")]
            if len(values) != 2:
                raise ValueError(
                    f"{filepath}:{lineno} 体力格式必须为 bx,by "
                    f"(如 0,-78000)，得到: '{' '.join(rest)}'")
            try:
                body_values = [float(x) for x in values]
            except ValueError:
                raise ValueError(
                    f"{filepath}:{lineno} 体力值无法解析: '{' '.join(rest)}' "
                    "— 需要两个数值") from None
            if not all(math.isfinite(v) for v in body_values):
                # NaN/Inf 曾延迟到求解阶段才被拒 (审计 2026-08-03)
                raise ValueError(
                    f"{filepath}:{lineno} 体力值必须为有限数值: "
                    f"'{' '.join(rest)}'")
            spec["body_force"] = body_values
    return spec


def _parse_hole(text, filepath="", lineno=0):
    """内孔 圆 x=1.5 y=1.0 r=0.3 — 参数缺失/无法解析必须报错.

    曾静默回落 r=0.1 于原点 (用户毫不知情的孔), 尾部缺值还抛裸
    IndexError (审计 2026-08-03).
    """
    parts = text.replace("=", " ").split()
    hole = {"type": None}
    i = 0
    while i < len(parts):
        kw = parts[i].lower()
        if kw in ("圆", "circle"):
            hole["type"] = "circle"
        elif kw in ("矩形", "rect"):
            hole["type"] = "rect"
        elif kw in ("x", "y", "r", "w", "h"):
            if i + 1 >= len(parts):
                raise ValueError(
                    f"{filepath}:{lineno} 内孔参数 '{kw}' 缺少数值 "
                    "(如 '内孔 圆 x=' — 曾 IndexError 崩溃)")
            try:
                value = float(parts[i + 1])
            except ValueError:
                raise ValueError(
                    f"{filepath}:{lineno} 内孔参数 '{kw}' 值无法解析: "
                    f"'{parts[i + 1]}'") from None
            if not math.isfinite(value) or (kw in ("r", "w", "h") and value <= 0):
                # NaN 坐标曾生成含字面 nan 的 .geo, Gmsh 报错不可读
                # (审计 2026-08-03)
                raise ValueError(
                    f"{filepath}:{lineno} 内孔参数 '{kw}' 必须为"
                    + ("有限正数" if kw in ("r", "w", "h") else "有限数值")
                    + f", 得到: {value!r}")
            hole[kw] = value
            i += 1
        i += 1
    if not hole["type"]:
        raise ValueError(
            f"{filepath}:{lineno} 内孔缺少类型 — 格式: "
            "内孔 圆 x=.. y=.. r=.. 或 内孔 矩形 x=.. y=.. w=.. h=..")
    if hole["type"] == "circle" and "r" not in hole:
        raise ValueError(
            f"{filepath}:{lineno} 圆形内孔必须指定 r — 曾静默默认 r=0.1")
    if hole["type"] == "rect" and ("w" not in hole or "h" not in hole):
        raise ValueError(
            f"{filepath}:{lineno} 矩形内孔必须指定 w 和 h")
    return hole


def _parse_bc(text, filepath="", lineno=0):
    """边界 左 固定  or  边界 右 拉力 1e6 — 未知类型/缺值必须报错.

    曾写 @FEM:bc= 死注解被下游静默丢弃 (载荷消失), 或静默施加载荷 0
    (审计 2026-08-03).
    """
    parts = text.split()
    if len(parts) < 2:
        raise ValueError(
            f"{filepath}:{lineno} 边界格式: 边名 + 类型 (+ 数值), "
            f"如 '边界 右 拉力 1e6'")
    edge, bc = parts[0], parts[1]
    if bc not in _BC_TYPES:
        raise ValueError(
            f"{filepath}:{lineno} 未知边界类型 '{bc}' — 支持: "
            f"{sorted(_BC_TYPES)} (拼错曾静默丢失整个边界载荷)")
    val = None
    if bc != "固定" and len(parts) < 3:
        raise ValueError(
            f"{filepath}:{lineno} 边界类型 '{bc}' 需要数值 "
            f"(如 '边界 右 拉力 1e6') — 曾静默施加载荷 0")
    if bc == "固定" and len(parts) > 2:
        # 固定约束无幅值 — 曾接受值并写进物理曲线名 (审计 2026-08-03)
        raise ValueError(
            f"{filepath}:{lineno} 边界类型 '固定' 不接受数值: "
            f"{' '.join(parts[2:])} — 固定约束无幅值")
    val = None
    if len(parts) >= 3:
        raw_val = parts[2]
        if bc in ("拉力", "面力") and ("," in raw_val or "，" in raw_val):
            # 双分量面力: 拉力 1e6,2e6 → (tx,ty). 曾只能单值, ty 恒 0
            # (审计 2026-08-03 输入端整改)
            comps = [c.strip() for c in raw_val.replace("，", ",").split(",")]
            if len(comps) != 2:
                raise ValueError(
                    f"{filepath}:{lineno} 面力 '{bc}' 需 1 或 2 个分量 "
                    f"(如 '拉力 1e6' 或 '拉力 1e6,2e6'), 得到: '{raw_val}'")
            try:
                vals = [float(c) for c in comps]
            except ValueError:
                raise ValueError(
                    f"{filepath}:{lineno} 面力分量无法解析: '{raw_val}'"
                ) from None
            if not all(math.isfinite(v) for v in vals):
                raise ValueError(
                    f"{filepath}:{lineno} 面力分量必须为有限数值: '{raw_val}'")
            val = f"{vals[0]},{vals[1]}"
        else:
            try:
                val = float(raw_val)
            except ValueError:
                raise ValueError(
                    f"{filepath}:{lineno} 边界数值无法解析: '{raw_val}'"
                ) from None
            if not math.isfinite(val):
                # NaN/Inf 曾延迟到求解阶段才被拒 (审计 2026-08-03)
                raise ValueError(
                    f"{filepath}:{lineno} 边界数值必须为有限数值: '{raw_val}'")
    if len(parts) > 3:
        raise ValueError(
            f"{filepath}:{lineno} 边界 '{bc}' 多余参数: "
            f"{' '.join(parts[3:])} (曾静默丢弃)")
    return {"edge": edge, "bc": bc, "value": val}


def _check_hole_separation(holes, lc, margin_factor=0.5):
    """拒绝孔间过近/重叠 (在 Gmsh 看到几何之前)。

    孔与孔重叠时 Gmsh 会静默丢弃孔 loop (用户拿到实心板无提示);
    近相切时产生 sliver 或 "intersections in the 1D mesh" 并导致
    面网格失败。两类都是生成器应在生成阶段报告的几何错误。
    """
    margin = margin_factor * lc
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            a, b = holes[i], holes[j]
            if a["type"] != "circle" or b["type"] != "circle":
                continue
            dx = a.get("x", 0.0) - b.get("x", 0.0)
            dy = a.get("y", 0.0) - b.get("y", 0.0)
            gap = math.hypot(dx, dy)
            need = a.get("r", 0.1) + b.get("r", 0.1) + margin
            if gap < need:
                raise ValueError(
                    f"内孔{i + 1} 与内孔{j + 1} 距离过近: 圆心距 {gap:.4g} "
                    f"< 半径和 + 留量 {need:.4g} "
                    f"(留量 {margin:.4g} = {margin_factor}×lc) — "
                    f"孔与孔之间必须保留足够间距。")


# ══════════════════════════════════════════════════════
# .geo 生成 — 内置几何内核 (tutorial t1/t4)
# ══════════════════════════════════════════════════════

def _require_param(spec, key, label):
    """几何参数缺失必须报错 — 漏参数 + 静默默认值 = 看似正常但错误的几何."""
    if key not in spec["params"]:
        raise ValueError(
            f"几何类型 {spec['type']} 缺少参数 '{label}' — "
            f"请在描述中指定 (如 '{label} 3.0')")


def generate_geo(spec, output_path, quad=False):
    """spec → Gmsh .geo 文件.

    ``quad=True`` 把重组选项写入新生成的几何文件；已有 ``.geo`` 的
    四边形选项由 :mod:`gmsh_runner` 通过命令行传入，源文件不会被修改。
    """
    L = []  # lines accumulator
    W = lambda s: L.append(s)

    lc = spec.get("mesh_size", 0.1)

    W('// Auto-generated by geo_spec.py')
    W(f'lc = {lc};')
    W('')

    # ── 几何 ──
    edges = {}
    if spec["type"] == "rect":
        _require_param(spec, "width", "宽")
        _require_param(spec, "height", "高")
        edges = _geo_rect(L, spec)
        W('')
    elif spec["type"] in ("circle", "annulus"):
        _require_param(spec, "outer_r", "外径/外半径")
        if spec["type"] == "annulus":
            # 缺内径曾静默按 0 处理 → 圆环变成实心圆, 几何语义根本改变
            _require_param(spec, "inner_r", "内径/内半径")
        inner = spec["params"]["inner_r"] if spec["type"] == "annulus" else 0
        edges = _geo_circle(L, spec["params"]["outer_r"], inner, spec["holes"], lc)
        W('')
    else:
        raise ValueError(f"Unknown type: {spec['type']}")

    # ── 物理组 ──
    W('// ---- Boundary Physical Groups ----')
    W('Mesh.SaveAll = 1;')
    W('Physical Surface("domain", 200) = {1};')
    bc_id = 100
    physical_names = []
    for b in spec.get("boundaries", []):
        bc_id += 1
        edge_name = b["edge"]
        eids = _edge_ids(edge_name, edges)
        if eids:
            tag = f'{edge_name}_{b["bc"]}'
            if b.get("value") is not None:
                # 双分量面力 (1e6,2e6): 逗号会打断 @FEM 逗号分隔往返,
                # 标签内用下划线编码 (审计 2026-08-03 输入端整改)
                tag += f'_{str(b["value"]).replace(",", "_")}'
            W(f'Physical Curve("{tag}", {bc_id}) = {{{eids}}};')
            physical_names.append(tag)
        else:
            # 该边名对当前几何不可用 (圆板无 左/右/上/下, 无孔时无 内孔) —
            # 曾静默写注释 + @FEM 注解照常输出: 圆板"左固定"只钉住极区
            # 小段弧, 载荷静默退化; 无孔时"内孔"到求解期才 FATAL
            # (审计 2026-08-03)
            available = [k for k in edges if edges.get(k)]
            raise ValueError(
                f"边界 '{edge_name}' 无法映射到当前几何的边 — "
                f"可用边名: {available or ['(无)']} "
                f"(圆板/圆环仅支持 外边/内孔/内孔N)")
    W('')

    # ── FEM 配置 (run.py 自动读取) ──
    W('// ---- FEM Config (auto-read by run.py) ----')
    for b, physical_name in zip(
            spec.get("boundaries", []), physical_names):
        edge_name = physical_name
        bc_type = b["bc"]
        val = b.get("value")
        if bc_type == "固定":
            W(f'// @FEM:fix={edge_name}')
        elif bc_type == "压力":
            # 压力 = 法向面力 (t = -p·n), 与全局坐标面力区分
            v_str = f'{val}' if val else '0'
            W(f'// @FEM:pressure={edge_name},{v_str}')
        elif bc_type in ("拉力", "面力"):
            if isinstance(val, str) and "," in val:
                v_str = val            # 双分量 (tx,ty) 直写
            else:
                v_str = f'{val},0' if val else '0,0'
            W(f'// @FEM:traction={edge_name},{v_str}')
        else:
            # @FEM:bc= 是死注解, parse_geo_fem_config 无此分支, 载荷会
            # 静默消失 — 解析阶段已拦截未知类型, 此处防御 (审计 2026-08-03)
            raise ValueError(
                f"未知边界类型 '{bc_type}' — 支持: {sorted(_BC_TYPES)}")
    if spec.get("body_force"):
        bf = spec["body_force"]
        W(f'// @FEM:body={bf[0]},{bf[1]}')
    W('')
    # ── 导出 ──
    # 输出格式由 gmsh_runner 统一强制为原生 .msh (Mesh.Format 曾=39/Abaqus,
    # 2026-08 移除 Abaqus 输入口后不再需要).
    W('// ---- Output ----')
    W('Mesh.SaveGroupsOfNodes = 1;')
    W('Mesh.SaveGroupsOfElements = 1;')
    if quad:
        W('// Q4 mesh requested by FEM2D')
        W('Mesh.RecombineAll = 1;')
        W('Mesh.Algorithm = 8;')
    W('Mesh 2;')

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return output_path


def _edge_ids(edge_name, edges):
    """将中文边名映射到线 ID 字符串"""
    m = {
        "左": edges.get("左"), "右": edges.get("右"),
        "下": edges.get("下"), "上": edges.get("上"),
        "底": edges.get("下"), "顶": edges.get("上"),
        "外边": ",".join(str(x) for x in edges.get("外边", [])),
        "内孔": ",".join(str(x) for x in edges.get("内孔", [])),
    }
    # 内孔1, 内孔2 等编号
    for k, v in edges.items():
        if k.startswith("内孔") and k != "内孔":
            m[k] = ",".join(str(x) for x in (v if isinstance(v, list) else [v]))
    ids = m.get(edge_name)
    if ids and isinstance(ids, list):
        ids = ",".join(str(x) for x in ids)
    return ids


# ══════════════════════════════════════════════════════
# 矩形
# ══════════════════════════════════════════════════════

def _geo_rect(L, spec):
    p = spec["params"]
    hw, hh = p["width"] / 2.0, p["height"] / 2.0
    holes = spec.get("holes", [])
    n_holes = len(holes)
    lc = spec.get("mesh_size", 0.1)  # 校验留量用 (几何文件中的 lc 是字面文本)

    L.append(f'// Rectangle: {hw*2} x {hh*2}' + (f' + {n_holes} hole(s)' if holes else ''))

    # 外边界点 (CCW)
    L.append(f'Point(1) = {{ {-hw}, {-hh}, 0, lc}};')
    L.append(f'Point(2) = {{  {hw}, {-hh}, 0, lc}};')
    L.append(f'Point(3) = {{  {hw},  {hh}, 0, lc}};')
    L.append(f'Point(4) = {{ {-hw},  {hh}, 0, lc}};')
    L.append('Line(1) = {1, 2};  // bottom')
    L.append('Line(2) = {2, 3};  // right')
    L.append('Line(3) = {3, 4};  // top')
    L.append('Line(4) = {4, 1};  // left')

    edges = {"左": [4], "右": [2], "上": [3], "下": [1],
             "外边": [1, 2, 3, 4]}

    if not holes:
        L.append('Curve Loop(1) = {1, 2, 3, 4};')
        L.append('Plane Surface(1) = {1};')
        return edges

    # 外边界 Curve Loop
    L.append('Curve Loop(1) = {1, 2, 3, 4};')

    # ── 内孔 (Gmsh tutorial t4: 多 Curve Loop → Plane Surface) ──
    # 预校验 (生成几何前完成): 半径为正、孔在矩形内且与四边保持留量、
    # 孔与孔间距足够。域外/重叠孔会被 Gmsh 静默丢弃或导致面网格失败,
    # 生成器必须在 Gmsh 看到几何之前报告。
    margin = 0.5 * lc
    for hi, hole in enumerate(holes):
        if hole["type"] != "circle":
            continue  # 非圆孔由下方几何循环显式报错
        cx, cy, r = hole.get("x", 0), hole.get("y", 0), hole.get("r", 0.1)
        if r <= 0.0:
            raise ValueError(f"内孔{hi + 1} 半径 r={r} 必须为正数")
        if (cx - r < -hw + margin or cx + r > hw - margin
                or cy - r < -hh + margin or cy + r > hh - margin):
            raise ValueError(
                f"内孔{hi + 1} 圆心 ({cx:.4g},{cy:.4g}) r={r:.4g} 超出矩形可布"
                f"区域 [{-hw + margin:.4g},{hw - margin:.4g}]"
                f"×[{-hh + margin:.4g},{hh - margin:.4g}] "
                f"(四边留量 {margin:.4g} = 0.5×lc) — "
                f"孔必须完全位于板内, 否则 Gmsh 会静默丢弃该孔。")
    _check_hole_separation(holes, lc)

    hole_loops = []
    pid, lid = 5, 5
    for hi, hole in enumerate(holes):
        if hole["type"] == "circle":
            cx, cy, r = hole.get("x", 0), hole.get("y", 0), hole.get("r", 0.1)
            n = 8  # 用 8 段圆弧逼近圆
            # 圆上 8 点 CCW (内孔需要 CW 方向 — 在 Curve Loop 中取负号)
            p0 = pid
            for j in range(n):
                ang = 2 * math.pi * j / n
                px, py = cx + r * math.cos(ang), cy + r * math.sin(ang)
                L.append(f'Point({pid}) = {{ {px:.17g}, {py:.17g}, 0, lc}};')
                pid += 1
            # 圆弧段
            l0 = lid
            for j in range(n):
                a, b = p0 + j, p0 + (j + 1) % n
                center_pt = pid
                L.append(f'Point({center_pt}) = {{ {cx:.17g}, {cy:.17g}, 0, lc}};')
                pid += 1
                L.append(f'Circle({lid}) = {{{a}, {center_pt}, {b}}};')
                lid += 1
            ll_id = lid
            hole_line_ids = list(range(l0, lid))
            # CCW 方向
            L.append(f'Curve Loop({ll_id}) = {{{",".join(str(x) for x in hole_line_ids)}}};')
            hole_loops.append(ll_id)
            lid += 1
            # 将孔洞曲线 ID 加入 edges, 支持 "内孔" / "内孔1" 等命名
            hole_name = f"内孔{hi + 1}" if n_holes > 1 else "内孔"
            if n_holes == 1:
                edges["内孔"] = hole_line_ids
            else:
                edges[hole_name] = hole_line_ids
                edges.setdefault("内孔", []).extend(hole_line_ids)
            # 孔洞曲线不加入 "外边"
            all_hole_ids = set()
            for v in edges.get("内孔", []):
                if isinstance(v, list):
                    all_hole_ids.update(v)
                else:
                    all_hole_ids.add(v)
            # 从外边中移除属于内孔的曲线
            edges["外边"] = [x for x in edges.get("外边", [1,2,3,4])
                            if x not in all_hole_ids]
        else:
            raise ValueError(
                f"Unsupported hole type '{hole['type']}'. "
                f"Only '圆'/'circle' is currently implemented for geometry generation. "
                f"Rectangular/elliptical holes must be defined directly in a .geo file.")

    # Plane Surface: 外边界 + 内孔 (hole loops 前不需要负号, 因为是内孔 loop)
    # Gmsh: Plane Surface(n) = {outer_loop, hole_loop1, hole_loop2, ...}
    all_loops = [1] + hole_loops
    L.append(f'Plane Surface(1) = {{{", ".join(str(x) for x in all_loops)}}};')

    return edges


# ══════════════════════════════════════════════════════
# 圆板
# ══════════════════════════════════════════════════════

def _geo_circle(L, outer_r, inner_r, holes, lc):
    """圆板/圆环 — 用 Circle 弧段"""
    n = 16  # 16 段逼近圆
    edges = {}

    # 外圆 CCW
    L.append(f'// Circle: outer R={outer_r}' + (f' inner R={inner_r}' if inner_r > 0 else ''))
    p0, l0 = 1, 1
    for j in range(n):
        ang = 2 * math.pi * j / n
        px, py = outer_r * math.cos(ang), outer_r * math.sin(ang)
        L.append(f'Point({p0+j}) = {{ {px:.17g}, {py:.17g}, 0, lc}};')
    # 圆心 (多用于 arc 中心)
    L.append(f'Point({p0+n}) = {{ 0, 0, 0, lc}};')
    center_id = p0 + n

    outer_lines = []
    for j in range(n):
        L.append(f'Circle({l0+j}) = {{{p0+j}, {center_id}, {p0+(j+1)%n}}};')
        outer_lines.append(l0 + j)

    L.append(f'Curve Loop(1) = {{{",".join(str(x) for x in outer_lines)}}};')

    edges["外边"] = outer_lines

    # 修复: 此前内径 ≥ 外径无校验, 内圆落在域外被 Gmsh 静默丢弃
    # (用户拿到实心圆板无提示); holes 参数也被完全忽略。
    if outer_r <= 0.0:
        raise ValueError(f"外半径 R={outer_r} 必须为正数")
    if inner_r >= outer_r:
        raise ValueError(
            f"内径/内半径 {inner_r} 必须小于外半径 {outer_r}")

    hole_loops = []
    if inner_r > 0:
        # 内圆 CW (与外圆反向)
        pid = p0 + n + 1
        lid = l0 + n
        for j in range(n):
            ang = 2 * math.pi * (n - 1 - j) / n  # CW
            px, py = inner_r * math.cos(ang), inner_r * math.sin(ang)
            L.append(f'Point({pid+j}) = {{ {px:.17g}, {py:.17g}, 0, lc}};')
        inner_lines = []
        for j in range(n):
            L.append(f'Circle({lid+j}) = {{{pid+j}, {center_id}, {pid+(j+1)%n}}};')
            inner_lines.append(lid + j)
        # 带内孔的 Curve Loop
        iloop_id = lid + n
        L.append(f'Curve Loop({iloop_id}) = {{{",".join(str(x) for x in inner_lines)}}};')
        hole_loops.append(iloop_id)
        edges["内孔"] = inner_lines

    # 圆板/圆环上的内孔 (用户可写多个 内孔 圆 x= y= r=)
    # 校验: 孔在外圆内 (留量)、与环内边界保持间距、孔与孔间距足够 —
    # 与 _geo_rect 同约定, 在 Gmsh 看到几何之前报告。
    margin = 0.5 * lc
    for hi, hole in enumerate(holes):
        if hole["type"] != "circle":
            raise ValueError(
                f"圆板/圆环目前仅支持圆形内孔 ('圆'), "
                f"不支持的孔类型 '{hole['type']}'")
        cx, cy, r = hole.get("x", 0), hole.get("y", 0), hole.get("r", 0.1)
        if r <= 0.0:
            raise ValueError(f"内孔{hi + 1} 半径 r={r} 必须为正数")
        if math.hypot(cx, cy) + r > outer_r - margin:
            raise ValueError(
                f"内孔{hi + 1} 圆心 ({cx:.4g},{cy:.4g}) r={r:.4g} "
                f"距外圆边界过近/越界 (允许最大半径 {outer_r - margin:.4g}, "
                f"留量 {margin:.4g} = 0.5×lc) — 孔必须完全位于板内。")
        if inner_r > 0 and math.hypot(cx, cy) - r < inner_r + margin:
            raise ValueError(
                f"内孔{hi + 1} 圆心 ({cx:.4g},{cy:.4g}) r={r:.4g} "
                f"与环内边界过近/重叠 (圆心距需 ≥ {inner_r + r + margin:.4g}, "
                f"留量 {margin:.4g} = 0.5×lc) — 孔必须完全位于环形域内。")
    _check_hole_separation(holes, lc)

    next_pt = p0 + 2 * n + 1
    next_ln = l0 + 2 * n
    next_loop = iloop_id + 1 if inner_r > 0 else 2
    for hi, hole in enumerate(holes):
        cx, cy, r = hole.get("x", 0), hole.get("y", 0), hole.get("r", 0.1)
        pts = []
        for j in range(n):
            ang = 2 * math.pi * j / n
            px, py = cx + r * math.cos(ang), cy + r * math.sin(ang)
            L.append(f'Point({next_pt + j}) = {{ {px:.17g}, {py:.17g}, 0, lc}};')
            pts.append(next_pt + j)
        L.append(f'Point({next_pt + n}) = {{ {cx:.17g}, {cy:.17g}, 0, lc}};')
        hc = next_pt + n
        hole_lines = []
        for j in range(n):
            a, b = pts[j], pts[(j + 1) % n]
            L.append(f'Circle({next_ln + j}) = {{{a}, {hc}, {b}}};')
            hole_lines.append(next_ln + j)
        L.append(f'Curve Loop({next_loop}) = {{{",".join(str(x) for x in hole_lines)}}};')
        hole_loops.append(next_loop)
        # 圆板的 内孔N 编号 — 曾与矩形不一致 (圆板多孔时 内孔2 无法
        # 引用, 求解期才 FATAL) (审计 2026-08-03)
        edges[f"内孔{hi + 1}"] = hole_lines
        prev = edges.get("内孔", [])
        edges["内孔"] = (prev + hole_lines
                         if isinstance(prev, list) else [prev] + hole_lines)
        next_pt += n + 1
        next_ln += n
        next_loop += 1

    all_loops = [1] + hole_loops
    L.append(f'Plane Surface(1) = {{{", ".join(str(x) for x in all_loops)}}};')

    return edges


# ══════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("geo_spec.py — 中文几何描述 → Gmsh .geo → .msh")
        print("用法: python geo_spec.py <描述文件.txt>")
        print()
        print("示例描述文件:")
        print("  类型 矩形板")
        print("  宽 3.0")
        print("  高 2.0")
        print("  内孔 圆 x=0.8 y=0.3 r=0.3")
        print("  网格 0.1")
        print("  边界 左 固定")
        print("  边界 右 拉力 1e6")
        return 2

    spec_path = sys.argv[1]
    if not os.path.exists(spec_path):
        print(f"文件不存在: {spec_path}")
        return 1

    try:
        print(f"[Spec] {spec_path}")
        spec = parse_spec(spec_path)
        if not spec["type"]:
            print("[ERROR] 缺少 '类型' 定义 (矩形板 / 圆板)")
            return 1

        print(f"  类型={spec['type']}  参数={spec['params']}  网格={spec['mesh_size']}")
        if spec["holes"]: print(f"  内孔: {spec['holes']}")
        if spec["boundaries"]: print(f"  边界: {spec['boundaries']}")

        base = os.path.splitext(spec_path)[0]
        geo_path = base + ".geo"
        print(f"\n[Geo] -> {geo_path}")
        generate_geo(spec, geo_path)
    except ValueError as error:
        # 解析/生成错误曾裸 traceback, 行号信息已在错误消息内
        # (审计 2026-08-03); 退出码曾恒 0, 脚本化调用无法感知失败
        print(f"[ERROR] {error}")
        return 1

    try:
        inp_path = run_gmsh(geo_path)
    except Exception as error:
        print(f"[ERROR] Gmsh 运行失败: {error}")
        return 1
    if inp_path:
        print(f"\n[Done] 可直接求解: python run.py {inp_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
