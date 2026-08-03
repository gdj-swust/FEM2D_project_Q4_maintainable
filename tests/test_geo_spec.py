"""Generator-side geometry validation (scripts.geo_spec).

Out-of-domain, tangency and overlapping holes are rejected before Gmsh
sees them — otherwise Gmsh silently drops the invalid loop and the user
gets a solid plate with no error.
"""
import os

import pytest

from scripts.geo_spec import generate_geo, parse_spec


def _generate(tmp_path, text):
    spec_path = tmp_path / "model.txt"
    spec_path.write_text(text, encoding="utf-8")
    spec = parse_spec(str(spec_path))
    out = str(tmp_path / "out.geo")
    generate_geo(spec, out)
    return spec, out


def test_rect_missing_width_rejected(tmp_path):
    """矩形缺宽 — 曾静默默认 1.0, 用户以为按描述建模实际 1×1 (审查 P2-3)."""
    with pytest.raises(ValueError, match="缺少参数.*宽"):
        _generate(tmp_path, "类型 矩形板\n高 2.0\n")


def test_rect_missing_height_rejected(tmp_path):
    with pytest.raises(ValueError, match="缺少参数.*高"):
        _generate(tmp_path, "类型 矩形板\n宽 3.0\n")


def test_circle_missing_outer_r_rejected(tmp_path):
    """圆板缺外径 — 曾静默默认 1.0."""
    with pytest.raises(ValueError, match="缺少参数.*外径"):
        _generate(tmp_path, "类型 圆板\n")


def test_annulus_missing_inner_r_rejected(tmp_path):
    """圆环缺内径 — 曾静默生成实心圆, 几何语义根本改变 (审查 P2-3)."""
    with pytest.raises(ValueError, match="缺少参数.*内径"):
        _generate(tmp_path, "类型 圆环\n外径 4.0\n")


def test_annulus_with_inner_r_ok(tmp_path):
    """合法圆环不受影响."""
    spec, out = _generate(tmp_path, "类型 圆环\n外径 4.0\n内径 2.0\n")
    assert spec["type"] == "annulus"
    assert "Circle: outer R=2.0 inner R=1.0" in open(
        out, encoding="utf-8").read()


def test_rect_hole_outside_rejected(tmp_path):
    with pytest.raises(ValueError, match="矩形可布区域"):
        _generate(tmp_path, "类型 矩形板\n宽 3.0\n高 2.0\n"
                            "内孔 圆 x=5 y=0.3 r=0.3\n")


def test_rect_hole_tangent_boundary_rejected(tmp_path):
    # 圆心距右边界 0.05 < 0.5×lc 留量 → 拒绝
    with pytest.raises(ValueError, match="矩形可布区域"):
        _generate(tmp_path, "类型 矩形板\n宽 3.0\n高 2.0\n"
                            "内孔 圆 x=1.45 y=0 r=0.3\n网格 0.1\n")


def test_rect_holes_overlapping_rejected(tmp_path):
    with pytest.raises(ValueError, match="距离过近"):
        _generate(tmp_path, "类型 矩形板\n宽 6.0\n高 3.0\n"
                            "内孔 圆 x=-0.5 y=0 r=0.5\n"
                            "内孔 圆 x=0.4 y=0 r=0.5\n")


def test_rect_multi_hole_valid(tmp_path):
    """多孔必须全部写入 .geo — 曾只断言文件存在, 第二个孔静默丢失也
    通过."""
    _, out = _generate(tmp_path, "类型 矩形板\n宽 6.0\n高 3.0\n"
                                 "内孔 圆 x=-1.0 y=0.5 r=0.4\n"
                                 "内孔 圆 x=1.2 y=0.2 r=0.5\n")
    assert os.path.isfile(out)
    text = open(out, encoding="utf-8").read()
    # 两孔各 16 个点 + 1 个孔心点, 且两孔 x 坐标都在 .geo 文本中
    assert text.count("Point(") >= 2 * 17, "孔点数量不足 (第二孔可能丢失)"
    assert "-1" in text and "1.2" in text, "第二孔坐标未写入"


def test_circle_hole_outside_rejected(tmp_path):
    with pytest.raises(ValueError, match="外圆"):
        _generate(tmp_path, "类型 圆板\n外径 2.0\n"
                            "内孔 圆 x=1.5 y=0 r=0.8\n")


def test_circle_hole_valid(tmp_path):
    _, out = _generate(tmp_path, "类型 圆板\n外径 2.0\n"
                                 "内孔 圆 x=0.5 y=0.3 r=0.3\n")
    assert os.path.isfile(out)
    text = open(out, encoding="utf-8").read()
    assert "0.5" in text, "孔坐标未写入 (曾只断言文件存在, 审计 2026-08-03)"


def test_annulus_inner_radius_greater_than_outer_rejected(tmp_path):
    with pytest.raises(ValueError, match="内径"):
        _generate(tmp_path, "类型 圆环\n外径 2.0\n内径 3.0\n")


def test_annulus_hole_overlapping_inner_boundary_rejected(tmp_path):
    with pytest.raises(ValueError, match="环内边界"):
        _generate(tmp_path, "类型 圆环\n外径 4.0\n内径 2.0\n"
                            "内孔 圆 x=1.45 y=0 r=0.5\n")


def test_annulus_hole_valid(tmp_path):
    _, out = _generate(tmp_path, "类型 圆环\n外径 2.0\n内径 1.0\n"
                                 "内孔 圆 x=0.75 y=0 r=0.15\n")
    assert os.path.isfile(out)
    text = open(out, encoding="utf-8").read()
    assert "0.75" in text, "孔坐标未写入 (曾只断言文件存在, 审计 2026-08-03)"


# ═══════════════════════════════════════════════════════════════
# 输入端审计 2026-08-03 — 静默错误家族 (.txt 描述解析器)
# ═══════════════════════════════════════════════════════════════

def test_spec_bom_first_line_accepted(tmp_path):
    """UTF-8 BOM 曾吞掉首行 类型 → type=None 全部参数静默丢失."""
    path = tmp_path / "bom.txt"
    path.write_bytes(
        b"\xef\xbb\xbf" + "类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n".encode("utf-8"))
    spec = parse_spec(str(path))
    assert spec["type"] == "rect"


def test_spec_fullwidth_normalized(tmp_path):
    """全角＝，３．０ －７８０００ 必须归一化 (曾静默丢行/崩溃)."""
    path = tmp_path / "full.txt"
    path.write_text("类型 矩形板\n宽 ３．０\n高 2.0\n网格 0.5\n"
                    "体力 ０，－７８０００\n", encoding="utf-8")
    spec = parse_spec(str(path))
    assert spec["params"]["width"] == 3.0
    assert spec["body_force"] == [0.0, -78000.0]


def test_spec_unknown_key_warns(tmp_path, capsys):
    """拼错键名 (厚度) 必须 WARN — 曾整行静默丢弃, 用户以为已设置."""
    path = tmp_path / "unk.txt"
    path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n厚度 0.5\n网格 0.5\n",
                    encoding="utf-8")
    spec = parse_spec(str(path))
    out = capsys.readouterr().out
    assert "厚度" in out and "WARN" in out, f"未警告: {out!r}"
    assert "thickness" not in spec  # .txt 格式无厚度键, 但必须提示


@pytest.mark.parametrize("line", [
    "网格 -0.5\n", "网格 0\n", "网格 nan\n",
    "宽 -3.0\n", "高 0\n",
])
def test_spec_nonpositive_dimensions_rejected(tmp_path, line):
    """负/零/NaN 尺寸必须拒绝 — 曾静默镜像几何/空网格/挂起."""
    path = tmp_path / "neg.txt"
    path.write_text(f"类型 矩形板\n宽 3.0\n高 2.0\n{line}", encoding="utf-8")
    with pytest.raises(ValueError, match="正数"):
        parse_spec(str(path))


def test_spec_unknown_bc_type_rejected(tmp_path):
    """未知边界类型必须拒绝 — 曾写 @FEM:bc= 死注解被下游静默丢弃."""
    path = tmp_path / "badbc.txt"
    path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                    "边界 右 拉屎 1e6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="拉屎"):
        parse_spec(str(path))


def test_spec_bc_missing_value_rejected(tmp_path):
    """边界缺数值必须拒绝 — 曾静默施加载荷 0."""
    path = tmp_path / "bcnoval.txt"
    path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                    "边界 右 拉力\n", encoding="utf-8")
    with pytest.raises(ValueError, match="需要数值"):
        parse_spec(str(path))


def test_spec_duplicate_bc_rejected(tmp_path):
    """同一边重复声明必须拒绝 — 曾静默双倍加载."""
    path = tmp_path / "dupbc.txt"
    path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                    "边界 右 拉力 1e6\n边界 右 拉力 1e6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        parse_spec(str(path))


def test_spec_hole_missing_r_rejected(tmp_path):
    """孔缺 r 必须拒绝 — 曾静默默认 r=0.1 于 (0,0) 生成未知孔."""
    path = tmp_path / "nohole_r.txt"
    path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                    "内孔 圆 x=0.8 y=0.3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="r"):
        parse_spec(str(path))


def test_spec_hole_trailing_value_rejected(tmp_path):
    """尾部缺值 (内孔 圆 x=) 必须报错 — 曾裸 IndexError 崩溃."""
    path = tmp_path / "holeeq.txt"
    path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                    "内孔 圆 x=\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少"):
        parse_spec(str(path))


def test_spec_unknown_type_rejected(tmp_path):
    """未知类型必须解析期拒绝 (曾推迟到 generate_geo 且无支持列表)."""
    path = tmp_path / "badtype.txt"
    path.write_text("类型 三角板\n宽 3.0\n高 2.0\n网格 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="三角板"):
        parse_spec(str(path))


def test_parse_geo_fem_config_unknown_key_warns(capsys, tmp_path):
    """@FEM: 未知键必须 WARN — 曾静默丢弃 (如 geo_spec 的 bc= 死注解)."""
    from fem2d.preprocess import parse_geo_fem_config
    geo = tmp_path / "unk.geo"
    geo.write_text("// @FEM:bc=右_拉屎_1e6,拉屎,1e6\n"
                   "// @FEM:fix=左\n", encoding="utf-8")
    cfg = parse_geo_fem_config(str(geo))
    out = capsys.readouterr().out
    assert cfg["fix"] == ["左"], "合法 fix 仍须生效"
    assert "bc" in out and "WARN" in out, f"未知键未警告: {out!r}"


# ═══════════════════════════════════════════════════════════════
# P0-1 回归补充 (2026-08-03) — NaN 家族 / 键缺值 / 多余 token / 圆板边名
# ═══════════════════════════════════════════════════════════════

def test_spec_nan_hole_params_rejected(tmp_path):
    """内孔 NaN 坐标/半径必须解析期拒绝 — 曾生成含字面 nan 的 .geo,
    Gmsh 报错不可读."""
    for hole_line in ("内孔 圆 x=nan y=0.3 r=0.3\n",
                      "内孔 圆 x=0.8 y=nan r=0.3\n",
                      "内孔 圆 x=0.8 y=0.3 r=nan\n"):
        path = tmp_path / "nanhole.txt"
        path.write_text(f"类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n{hole_line}",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="有限"):
            parse_spec(str(path))


def test_spec_nan_bc_value_rejected(tmp_path):
    """边界数值 NaN/Inf 必须解析期拒绝 — 曾延迟到求解阶段."""
    for bc_line in ("边界 右 拉力 nan\n", "边界 右 压力 inf\n"):
        path = tmp_path / "nanbc.txt"
        path.write_text(f"类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n{bc_line}",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="有限"):
            parse_spec(str(path))


def test_spec_nan_body_force_rejected(tmp_path):
    """体力 NaN/Inf 必须解析期拒绝 — 曾延迟到求解阶段."""
    for body_line in ("体力 nan,0\n", "体力 0,inf\n"):
        path = tmp_path / "nanbody.txt"
        path.write_text(f"类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n{body_line}",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="有限"):
            parse_spec(str(path))


def test_spec_key_missing_value_rejected(tmp_path):
    """合法键缺数值 (宽 单独一行) 必须报错 — 曾整行静默丢弃."""
    path = tmp_path / "nokeyval.txt"
    path.write_text("类型 矩形板\n宽\n高 2.0\n网格 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少"):
        parse_spec(str(path))


def test_spec_extra_token_rejected(tmp_path):
    """多余 token (宽 3 4) 必须报错 — 曾取 3 丢 4 静默."""
    path = tmp_path / "extratok.txt"
    path.write_text("类型 矩形板\n宽 3.0 4.0\n高 2.0\n网格 0.5\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="多余"):
        parse_spec(str(path))


def test_spec_bc_extra_token_rejected(tmp_path):
    """边界多余 token (拉力 1e6 2e6) 必须报错 — 曾静默丢弃."""
    path = tmp_path / "bcxtra.txt"
    path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                    "边界 右 拉力 1e6 2e6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="多余"):
        parse_spec(str(path))


def test_spec_fix_with_value_rejected(tmp_path):
    """固定约束不接受数值 (固定 5) — 曾把值写进物理曲线名."""
    path = tmp_path / "fixval.txt"
    path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                    "边界 左 固定 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不接受数值"):
        parse_spec(str(path))


def test_circle_invalid_edge_name_rejected(tmp_path):
    """圆板 左 固定 必须生成期拒绝 — 曾静默降级为极区小段弧."""
    spec_path = tmp_path / "c.txt"
    spec_path.write_text("类型 圆板\n外径 2.0\n网格 0.5\n"
                         "边界 左 固定\n", encoding="utf-8")
    spec = parse_spec(str(spec_path))
    with pytest.raises(ValueError, match="无法映射"):
        generate_geo(spec, str(tmp_path / "out.geo"))


def test_circle_holeN_edge_mappable(tmp_path):
    """圆板多孔 内孔2 必须生成期可映射 — 曾与矩形不一致, 求解期才 FATAL
   ."""
    spec_path = tmp_path / "c2.txt"
    spec_path.write_text("类型 圆板\n外径 4.0\n网格 0.5\n"
                         "内孔 圆 x=-1.0 y=0 r=0.3\n"
                         "内孔 圆 x=1.0 y=0 r=0.3\n"
                         "边界 内孔2 固定\n", encoding="utf-8")
    spec = parse_spec(str(spec_path))
    out = str(tmp_path / "out.geo")
    generate_geo(spec, out)
    text = open(out, encoding="utf-8").read()
    assert "内孔2_固定" in text, "内孔2 Physical Curve 未生成"


# ═══════════════════════════════════════════════════════════════
# 输入端整改 2026-08-03 — .txt 面力双分量 (曾只能单值, ty 恒 0)
# ═══════════════════════════════════════════════════════════════

def test_spec_traction_two_components_roundtrip(tmp_path):
    """.txt 拉力 1e6,2e6 → spec → .geo @FEM → geo_fem 配置全程往返.

    逗号会打断 @FEM 逗号分隔编码, 物理曲线名内用下划线表示;
    parse_traction 必须还原 (tx,ty) = (1e6, 2e6).
    """
    from fem2d.loads import parse_traction
    from fem2d.preprocess import parse_geo_fem_config
    spec_path = tmp_path / "t2.txt"
    spec_path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                         "边界 右 拉力 1e6,2e6\n", encoding="utf-8")
    spec = parse_spec(str(spec_path))
    assert spec["boundaries"][0]["value"] == "1000000.0,2000000.0"
    out = str(tmp_path / "out.geo")
    generate_geo(spec, out)
    text = open(out, encoding="utf-8").read()
    assert "右_拉力_1000000.0_2000000.0" in text, \
        "双分量标签应以下划线编码 (逗号会打断 @FEM 往返)"
    cfg = parse_geo_fem_config(out)
    assert len(cfg["traction"]) == 1
    edge, tx, ty, profile = parse_traction(cfg["traction"][0])
    assert (tx, ty) == (1e6, 2e6), f"往返丢失分量: ({tx}, {ty})"
    assert profile is None


def test_spec_traction_single_component_backward_compat(tmp_path):
    """.txt 拉力 1e6 (单值) 保持旧行为: (tx, 0)."""
    spec_path = tmp_path / "t1.txt"
    spec_path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                         "边界 右 拉力 1e6\n", encoding="utf-8")
    spec = parse_spec(str(spec_path))
    assert spec["boundaries"][0]["value"] == 1e6
    out = str(tmp_path / "out.geo")
    generate_geo(spec, out)
    text = open(out, encoding="utf-8").read()
    assert "右_拉力_1000000.0" in text, "单值标签保持旧格式"
    assert "@FEM:traction=右_拉力_1000000.0,1000000.0,0" in text, \
        "单值应补 ty=0"


def test_spec_traction_fullwidth_comma_components(tmp_path):
    """.txt 拉力 1e6，2e6 (全角逗号) 必须解析为双分量."""
    spec_path = tmp_path / "t3.txt"
    spec_path.write_text("类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n"
                         "边界 右 拉力 1e6，2e6\n", encoding="utf-8")
    spec = parse_spec(str(spec_path))
    assert spec["boundaries"][0]["value"] == "1000000.0,2000000.0"


def test_spec_traction_bad_components_rejected(tmp_path):
    """.txt 拉力 1e6,2e6,3e6 与 NaN 分量必须拒绝 (曾只拒单值 NaN)."""
    for bc_line in ("边界 右 拉力 1e6,2e6,3e6\n",
                    "边界 右 拉力 1e6,nan\n",
                    "边界 右 拉力 inf,2e6\n"):
        spec_path = tmp_path / "bad.txt"
        spec_path.write_text(f"类型 矩形板\n宽 3.0\n高 2.0\n网格 0.5\n{bc_line}",
                             encoding="utf-8")
        with pytest.raises(ValueError, match="分量"):
            parse_spec(str(spec_path))
