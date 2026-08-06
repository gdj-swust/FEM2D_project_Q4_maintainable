"""交互式输入向导 — 终端问答建模, 用户无需写任何文件.

与 .txt 描述器同内核: 问答结果构建 spec dict → scripts.geo_spec.generate_geo
→ 临时 .geo (含 @FEM 注解) → 走现有 resolve_geo 管线。不新增解析器,
不改变 .txt/.geo/.spec/.msh 四条既有输入路径。

交互纪律 (沿项目教训):
- 每个回答即时校验, 非法立即重问, 不静默降级
- 边名白名单按形状 — 向导只问合法边 (圆板永不问 左/右), 消灭
  "圆板 左 固定" 类错误
- EOF (回车) 用默认值, 顶层 EOF 干净退出
- 临时 .geo 退出时自动清理, 原始文件永不修改
"""
import atexit
import math
import os
import tempfile

from .cli import ask
from .errors import CliError

# ── 形状 → 可用边名白名单 (与 geo_spec._edge_ids 的命名一致) ──
_SHAPE_EDGES = {
    "rect": ["左", "右", "上", "下"],
    "circle": ["外边"],
    "annulus": ["外边", "内孔"],
}
_SHAPE_NAMES = {"矩形板": "rect", "圆板": "circle", "圆环": "annulus"}
_BC_TYPES = ["固定", "压力", "拉力"]
_BC_HINT = {
    "固定": "Ux=Uy=0 (无数值)",
    "压力": "法向压力 p, 正值=压向板内 (如 1e6)",
    "拉力": "面力 tx[,ty] (如 1e6 或 1e6,2e6)",
}


# ═══════════════════════════════════════════════════════════════
# 提问原语 (EOF 安全 + 默认值 + 非法重问)
# ═══════════════════════════════════════════════════════════════

def _ask_choice(prompt, options, default=1):
    """选项选择: 显示编号列表, 回车 → default."""
    lines = [f"  {i + 1}. {opt}" for i, opt in enumerate(options)]
    while True:
        print(prompt)
        for line in lines:
            print(line)
        raw = ask(f"  选择 [1-{len(options)}, 回车={default}]: ")
        if not raw:
            return default - 1
        try:
            idx = int(raw)
        except ValueError:
            print(f"    ? 请输入编号 1-{len(options)}")
            continue
        if 1 <= idx <= len(options):
            return idx - 1
        print(f"    ? 编号超出范围 1-{len(options)}")


def _ask_number(prompt, default=None, *, positive=False, what="数值"):
    """数值输入; 非法/非有限立即重问, 不静默. EOF → 默认值或干净退出."""
    while True:
        raw = ask(prompt)
        if not raw and default is not None:
            return float(default)
        if not raw:
            # EOF (标准输入关闭) 且无默认值 — 否则在此死循环
            raise CliError("\n  [INFO] 未提供数值 — 退出向导", exit_code=0)
        try:
            value = float(raw)
        except ValueError:
            print(f"    ? '{raw}' 不是 {what} — 请重新输入")
            continue
        if not math.isfinite(value):
            print(f"    ? {what}必须有限 (NaN/Inf 曾静默产生错误结果)")
            continue
        if positive and value <= 0.0:
            print(f"    ? {what}必须为正数, 得到 {value!r}")
            continue
        return value


def _ask_vec2(prompt, default=None):
    """两个分量 (逗号分隔); 非法重问. 返回 (bx, by) 浮点元组."""
    while True:
        raw = ask(prompt)
        if not raw and default is not None:
            raw = default
        if not raw:
            return None
        parts = [p.strip() for p in raw.replace("，", ",").split(",")]
        if len(parts) != 2:
            print(f"    ? 需要恰好两个分量 (逗号分隔), 得到 {len(parts)} 个")
            continue
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            print(f"    ? 分量无法解析: '{raw}'")
            continue
        if not all(math.isfinite(v) for v in vals):
            print("    ? 分量必须有限 (NaN/Inf 会被静默忽略)")
            continue
        return vals[0], vals[1]


def _ask_yes(prompt, default="y"):
    """y/n 确认; 非法重问."""
    while True:
        raw = ask(prompt).strip().lower()
        if not raw:
            raw = default
        if raw in ("y", "yes", "是"):
            return True
        if raw in ("n", "no", "否"):
            return False
        print("    ? 请输入 y/n")


# ═══════════════════════════════════════════════════════════════
# 向导各阶段
# ═══════════════════════════════════════════════════════════════

def _ask_geometry():
    """S1 几何: 类型 → 尺寸 → 孔. 返回 (spec_type, params, holes)."""
    print("\n  ── 几何 ──")
    shape, with_holes = _ask_shape()
    params = {}
    if shape == "rect":
        params["width"] = _ask_number("  宽 [m]: ", positive=True, what="宽")
        params["height"] = _ask_number("  高 [m]: ", positive=True, what="高")
    elif shape == "circle":
        params["outer_r"] = _ask_number(
            "  外半径 [m]: ", positive=True, what="外半径")
    elif shape == "annulus":
        params["outer_r"] = _ask_number(
            "  外半径 [m]: ", positive=True, what="外半径")
        params["inner_r"] = _ask_number(
            "  内半径 [m]: ", positive=True, what="内半径")
    holes = []
    if with_holes or shape in ("circle", "annulus"):
        while True:
            if not _ask_yes(f"  添加内孔? [{len(holes)} 个已添加]", default="n"):
                break
            cx = _ask_number("    孔心 x [m]: ", default=0.0)
            cy = _ask_number("    孔心 y [m]: ", default=0.0)
            r = _ask_number("    孔半径 [m]: ", positive=True, what="孔半径")
            holes.append({"type": "circle", "x": cx, "y": cy, "r": r})
    return shape, params, holes


def _ask_shape():
    """几何类型单选 — 独立函数避免上面重复提问."""
    options = ["矩形板", "圆板", "圆环", "带孔矩形板"]
    idx = _ask_choice("  几何类型:", options, default=1)
    if idx == 3:
        return "rect", True
    return _SHAPE_NAMES[options[idx]], False


def _ask_boundaries(shape, n_holes):
    """S4 边界: 白名单边名循环; 回车结束. 返回 boundaries 列表."""
    print("\n  ── 边界约束与载荷 ──")
    print("  可用边:", " ".join(_available_edges(shape, n_holes, [])))
    boundaries = []
    used = set()
    while True:
        edges = _available_edges(shape, n_holes, used)
        if not edges:
            print("  (所有边已配置)")
            break
        print(f"\n  可用边 (未配置): {', '.join(edges)}")
        edge = ask("  边名 (回车结束): ").strip()
        if not edge:
            break
        if edge not in edges:
            print(f"    ? 边 '{edge}' 不可用 — 可选: {', '.join(edges)}")
            continue
        print(f"  边 {edge} 的载荷类型:")
        for i, bc in enumerate(_BC_TYPES):
            print(f"    {i + 1}. {bc} — {_BC_HINT[bc]}")
        raw = ask("  类型 [1-3]: ").strip()
        try:
            bc = _BC_TYPES[int(raw) - 1]
        except (ValueError, IndexError):
            print("    ? 请输入 1-3")
            continue
        value = None
        if bc == "压力":
            value = _ask_number(
                f"  边 {edge} 压力 p [Pa]: ", positive=False, what="压力")
        elif bc == "拉力":
            vec = _ask_vec2(
                f"  边 {edge} 面力 tx,ty [Pa]: ", default="1e6,0")
            if vec is None:
                continue
            if vec[1] == 0.0:
                value = vec[0]
            else:
                value = f"{vec[0]},{vec[1]}"
        boundaries.append({"edge": edge, "bc": bc, "value": value})
        used.add(edge)
        print(f"  已配置: {edge} {bc}"
              + (f" {value}" if value is not None else ""))
    return boundaries


def _available_edges(shape, n_holes, used):
    """当前形状可用的边名 (排除已配置).

    与 geo_spec._edge_ids 的命名严格对齐: 单孔只生成 "内孔" (无编号),
    多孔生成 "内孔1..N" + "内孔" 聚合 — 向导问的边必须生成器可映射。

    聚合/子边互斥 (与 parse_spec 拒绝重复边配置同族): 选过 "内孔"
    (聚合) 后子边全部不可选, 选过任一 "内孔i" 后聚合不可选 — 否则
    两轮配置落到同一批曲线, .geo 生成重叠 Physical Curve, 同边
    固定+载荷 并存 ("约束吞载荷" 同族历史教训)。
    """
    edges = list(_SHAPE_EDGES[shape])
    if n_holes == 1:
        edges.append("内孔")
    elif n_holes > 1:
        edges.append("内孔")            # 聚合: 全部孔一起施加
        edges.extend(f"内孔{i + 1}" for i in range(n_holes))
    sub_edges = [f"内孔{i + 1}" for i in range(n_holes)]
    if "内孔" in used:
        edges = [e for e in edges if e not in sub_edges]
    elif any(e in used for e in sub_edges):
        edges = [e for e in edges if e != "内孔"]
    return [e for e in edges if e not in used]


def _ask_material(config):
    """材料 E/nu/t — CLI 已显式指定的字段不重复提问 (CLI > 向导).

    曾无条件提问并覆盖 config: --wizard --E 5e7 被向导默认值覆盖,
    优先级分叉。
    """
    explicit = getattr(config, "_explicit", frozenset())
    material = {}
    for key, prompt, default, positive in (
            ("E", "  弹性模量 E [Pa, 默认 2.1e11]: ", 2.1e11, True),
            ("nu", "  泊松比 nu [默认 0.3]: ", 0.3, False),
            ("thickness", "  厚度 [m, 默认 0.01]: ", 0.01, True)):
        if key in explicit:
            value = getattr(config, key)
            print(f"  (使用 CLI --{key} {value:.3g})")
            material[key] = value
        else:
            material[key] = _ask_number(
                prompt, default=default, positive=positive, what=key)
    return material


def _ask_boundaries_cli_aware(config, shape, n_holes):
    """边界 — CLI 已提供 fix/traction 时跳过提问 (批处理安全)."""
    explicit = getattr(config, "_explicit", frozenset())
    provided = [k for k in ("fix", "fix_ux", "fix_uy", "traction", "force")
                if k in explicit and getattr(config, k)]
    if provided:
        print("\n  ── 边界约束与载荷 ──")
        print(f"  (使用 CLI 参数: {', '.join('--' + p for p in provided)} — "
              "边界阶段跳过)")
        return []
    return _ask_boundaries(shape, n_holes)


def _ask_body(config):
    """体力 — CLI 已提供 --body 时跳过提问."""
    explicit = getattr(config, "_explicit", frozenset())
    if "body" in explicit and config.body is not None:
        print("  (使用 CLI --body — 体力阶段跳过)")
        return None
    # N/m^3 而非 N/m³ — U+00B3 不在 GBK/cp936, 未重配 stdout 的
    # 直接 API/嵌入调用路径下 input() 抛 UnicodeEncodeError 硬崩溃
    return _ask_vec2("  体力 bx,by [N/m^3, 回车跳过]: ")


def _print_summary(shape, params, holes, mesh_size, material, boundaries,
                   body):
    """S6 总览表."""
    shape_cn = {v: k for k, v in _SHAPE_NAMES.items()}[shape]
    if shape == "rect" and holes:
        shape_cn = "带孔矩形板"
    print("\n" + "=" * 50)
    print("  模型总览")
    print("=" * 50)
    print(f"  几何: {shape_cn}")
    if shape == "rect":
        print(f"    宽 {params['width']} m × 高 {params['height']} m")
    else:
        print(f"    外半径 {params['outer_r']} m"
              + (f", 内半径 {params['inner_r']} m"
                 if shape == "annulus" else ""))
    for i, h in enumerate(holes):
        print(f"    内孔{i + 1}: 圆心 ({h['x']}, {h['y']}) r={h['r']}")
    print(f"  网格密度: {mesh_size}")
    print(f"  材料: E={material['E']:.3g} Pa, "
          f"nu={material['nu']}, 厚 {material['thickness']} m")
    if boundaries:
        for b in boundaries:
            val = f" {b['value']}" if b["value"] is not None else ""
            print(f"  边界: {b['edge']} {b['bc']}{val}")
    else:
        print("  边界: (无 — 求解将因欠约束被拒绝)")
    if body:
        print(f"  体力: ({body[0]:.3g}, {body[1]:.3g})")
    print("=" * 50)


def _spec_to_txt(spec):
    """spec dict → .txt 描述文件行 (与 geo_spec.parse_spec 格式互逆)."""
    name = {v: k for k, v in _SHAPE_NAMES.items()}[spec["type"]]
    lines = [f"类型 {name}"]
    p = spec["params"]
    if spec["type"] == "rect":
        lines.append(f"宽 {p['width']}")
        lines.append(f"高 {p['height']}")
    elif spec["type"] == "circle":
        lines.append(f"外半径 {p['outer_r']}")
    elif spec["type"] == "annulus":
        lines.append(f"外半径 {p['outer_r']}")
        lines.append(f"内半径 {p['inner_r']}")
    for h in spec["holes"]:
        lines.append(f"内孔 圆 x={h['x']} y={h['y']} r={h['r']}")
    lines.append(f"网格 {spec['mesh_size']}")
    for b in spec["boundaries"]:
        val = f" {b['value']}" if b["value"] is not None else ""
        lines.append(f"边界 {b['edge']} {b['bc']}{val}")
    if spec["body_force"]:
        lines.append(f"体力 {spec['body_force'][0]},{spec['body_force'][1]}")
    return "\n".join(lines) + "\n"


# ═══════════════════════════════════════════════════════════════
# 向导入口
# ═══════════════════════════════════════════════════════════════

def run_wizard(config):
    """运行交互向导, 返回网格输入文件路径 (.geo 临时文件).

    产物含 @FEM 注解 → resolve_input_file 走 .geo 分支自动合并配置,
    与 .txt 描述器完全同内核 (校验/边名/BC 语义零分叉)。
    """
    print("\n" + "=" * 50)
    print("  FEM2D 交互建模向导")
    print("  每步回车使用默认值; 随时 Ctrl-C 退出")
    print("=" * 50)

    # 入口选项: 交互建模 或 使用已有文件 (_ask_choice 返回 0-based 索引)
    options = ["交互建模 (几何/网格/材料/边界)", "使用已有文件 (.geo/.txt/.msh/.spec)"]
    if _ask_choice("  建模方式:", options, default=1) == 0:
        fp = _build_and_generate(config)
    else:
        fp = ask("  文件路径: ").strip()
        if not fp:
            raise CliError("  [INFO] 未输入路径 — 退出", exit_code=0)
        if not os.path.isfile(fp):
            raise CliError(f"  [FATAL] 文件不存在: {fp}", exit_code=1)
    return fp


def _build_and_generate(config):
    """问答构建 spec → 总览确认 → 生成临时 .geo → (可选) 保存 .txt."""
    while True:
        shape, params, holes = _ask_geometry()
        mesh_size = _ask_number(
            "  网格密度 lc [m, 默认 0.1]: ", default=0.1,
            positive=True, what="网格密度")
        material = _ask_material(config)
        boundaries = _ask_boundaries_cli_aware(config, shape, len(holes))
        body = _ask_body(config)
        _print_summary(shape, params, holes, mesh_size, material,
                       boundaries, body)
        if _ask_yes("  确认建模? [Y/n]", default="y"):
            break
        print("\n  — 重新开始 —")
        if not _ask_yes("  重新建模? [y/N]", default="n"):
            raise CliError("  [INFO] 取消建模 — 退出", exit_code=0)

    spec = {
        "type": shape,
        "params": params,
        "holes": holes,
        "mesh_size": mesh_size,
        "boundaries": boundaries,
        "body_force": [body[0], body[1]] if body else None,
    }
    # 材料写入 config — 向导产物走 resolve_geo 管线时 E/nu/t 生效
    config.E = material["E"]
    config.nu = material["nu"]
    config.thickness = material["thickness"]

    # 生成临时 .geo (含 @FEM 注解 → 边界自动合并)
    from scripts.geo_spec import generate_geo
    fd, geo_p = tempfile.mkstemp(
        prefix=".fem2d-wizard-", suffix=".geo", dir=os.getcwd())
    os.close(fd)
    os.unlink(geo_p)   # generate_geo 以 "w" 打开
    try:
        generate_geo(spec, geo_p, quad=config.quad)
    except (ValueError, IndexError) as error:
        raise CliError(
            f"  [FATAL] 几何生成失败: {error}", exit_code=1) from error
    atexit.register(lambda p=geo_p: os.path.isfile(p) and os.unlink(p))

    # 可选: 保存为 .txt (用户资产格式, 以后可直接 run.py xxx.txt)
    if _ask_yes("  保存模型为 .txt 文件? [y/N]", default="n"):
        txt_path = ask("  保存路径 [model.txt]: ").strip() or "model.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(_spec_to_txt(spec))
            print(f"  [OK] 已保存: {txt_path} (下次: python run.py {txt_path})")
        except OSError as error:
            print(f"  [WARN] 保存失败: {error}")

    print("\n  [OK] 建模完成 — 开始求解...")
    return geo_p
