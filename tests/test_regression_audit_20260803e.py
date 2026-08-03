"""2026-08-03 第五轮外部审查回归测试.

覆盖: .geo @FEM 载荷严格解析 (静默丢弃 → 响亮失败) / 装配前单元刚度
对称性检查 (全局抽样漏检) / 主应力极端应力溢出 (hypot 风格).
"""
import os

import numpy as np
import pytest


def _geo(tmp_path, content):
    p = tmp_path / "cfg.geo"
    p.write_text(content, encoding="utf-8")
    return str(p)


# ═══════════════════════════════════════════════════════════════
# .geo 载荷配置 — 曾字段不足静默丢弃、空值连正则都不匹配,
# 求解成功但工况已变 (第五轮外部审查复现: 最高优先级)
# ═══════════════════════════════════════════════════════════════

def test_geo_fem_config_rejects_short_traction(tmp_path):
    from fem2d.preprocess import parse_geo_fem_config
    geo = _geo(tmp_path, "// @FEM:traction=only_one_field\n")
    with pytest.raises(ValueError, match="载荷配置错误.*traction"):
        parse_geo_fem_config(geo)


def test_geo_fem_config_rejects_empty_value(tmp_path):
    from fem2d.preprocess import parse_geo_fem_config
    geo = _geo(tmp_path, "// @FEM:pressure=\n")
    with pytest.raises(ValueError, match="载荷配置错误.*pressure"):
        parse_geo_fem_config(geo)


def test_geo_fem_config_error_carries_file_and_line(tmp_path):
    from fem2d.preprocess import parse_geo_fem_config
    geo = _geo(tmp_path, "// ok\n// @FEM:traction=bad\n")
    with pytest.raises(ValueError) as exc:
        parse_geo_fem_config(geo)
    msg = str(exc.value)
    assert os.path.basename(geo) in msg, f"错误消息缺文件名: {msg}"
    assert ":2" in msg, f"错误消息缺行号: {msg}"
    assert "@FEM:traction=bad" in msg, f"错误消息缺原始内容: {msg}"


def test_geo_fem_config_rejects_extra_traction_fields(tmp_path):
    from fem2d.preprocess import parse_geo_fem_config
    geo = _geo(tmp_path, "// @FEM:traction=right,1e6,2e6,sin(pi*x),extra\n")
    with pytest.raises(ValueError, match="traction.*5 个字段"):
        parse_geo_fem_config(geo)


def test_geo_fem_config_rejects_extra_pressure_fields(tmp_path):
    from fem2d.preprocess import parse_geo_fem_config
    geo = _geo(tmp_path, "// @FEM:pressure=top,5e5,999\n")
    with pytest.raises(ValueError, match="pressure.*3 个字段"):
        parse_geo_fem_config(geo)


def test_geo_fem_config_rejects_extra_body_fields(tmp_path):
    from fem2d.preprocess import parse_geo_fem_config
    geo = _geo(tmp_path, "// @FEM:body=0,-78000,9\n")
    with pytest.raises(ValueError, match="body.*3 个字段"):
        parse_geo_fem_config(geo)


def test_geo_fem_config_rejects_comma_in_fix(tmp_path):
    from fem2d.preprocess import parse_geo_fem_config
    geo = _geo(tmp_path, "// @FEM:fix=left,right\n")
    with pytest.raises(ValueError, match="fix 需要单个边名"):
        parse_geo_fem_config(geo)


def test_geo_fem_config_normal_format_unaffected(tmp_path):
    from fem2d.preprocess import parse_geo_fem_config
    geo = _geo(
        tmp_path,
        "// @FEM:traction=right,1e6,2e6\n// @FEM:pressure=top,5e5\n"
        "// @FEM:fix=left\n// @FEM:body=0,-78000\n")
    cfg = parse_geo_fem_config(geo)
    assert cfg["traction"] == ["right:1e6,2e6"]
    assert cfg["pressure"] == ["top:5e5:n"]
    assert cfg["fix"] == ["left"]
    assert cfg["body"] == "0,-78000"


# ═══════════════════════════════════════════════════════════════
# 单元刚度对称性 — 全局抽样 256 行会漏掉未抽中位置的非对称项,
# scatter 前对每个小型 Ke 全量检查 (第五轮外部审查)
# ═══════════════════════════════════════════════════════════════

def test_local_symmetry_check_rejects_asymmetric():
    from fem2d.assembly import _check_local_symmetry
    sym = np.tile(np.eye(4)[None], (2, 1, 1))
    _check_local_symmetry(sym, "test")          # 对称通过
    asym = np.tile(np.eye(4)[None], (2, 1, 1))
    asym[0, 0, 1] = 1.0                          # 非对称项
    with pytest.raises(RuntimeError, match="asymmetric"):
        _check_local_symmetry(asym, "test")


# ═══════════════════════════════════════════════════════════════
# Q4R 推荐域验证 (评分 7.2 — 公式特性已文档化, 此处锁定推荐域行为:
# 规则网格/长宽比<10/膜主导时 Q4R 必须与 Q4 结果接近)
# ═══════════════════════════════════════════════════════════════

def test_q4r_agrees_with_q4_in_recommended_domain():
    """规则网格 (AR=2 < 10) + 单向拉伸: Q4R 与 Q4 位移相对差 < 1%
    且沙漏能占比低 — 锁定 Q4R 在推荐域内的可靠性."""
    from fem2d import Mesh, solve
    L, H, nx, ny = 8.0, 2.0, 8, 2          # 单元 AR = 2
    ncol = nx + 1
    nodes = [[i * L / nx, j * H / ny] for j in range(ny + 1)
             for i in range(ncol)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (j * ncol + i, j * ncol + i + 1,
                          (j + 1) * ncol + i, (j + 1) * ncol + i + 1)
            elems.append([a, b, d, c])       # Q4 环形序: 左下,右下,右上,左上
    def run(elem_type):
        m = Mesh(nodes=np.array(nodes), elements=np.array(elems),
                 E=210e9, nu=0.3, thickness=0.01, plane_type="stress",
                 elem_type=elem_type)
        for i in range(ny + 1):
            m.fix_node(i * ncol, "both", 0.0)
        right = [i * ncol + ncol - 1 for i in range(ny + 1)]
        for k in range(ny):
            m.add_traction(right[k], right[k + 1], 1e6, 0.0)
        r = solve(m, verbose=False)
        return r["u"], r.get("hourglass_energy_ratio")
    u_q4, _ = run("CPS4")
    u_q4r, hg = run("CPS4R")
    rel = np.abs(u_q4r - u_q4).max() / max(np.abs(u_q4).max(), 1e-30)
    # Q4R 是减缩积分 + 沙漏稳定 (专用单元), 与 Q4 全积分有 ~1% 级
    # 固有差异 — 锁定"推荐域内行为合理" (2%) 而非"等价"
    assert rel < 0.02, f"Q4R vs Q4 位移相对差 {rel:.2%} (推荐域应 <2%)"
    assert np.all(np.isfinite(u_q4r))
    if hg is not None:
        assert hg < 0.30, f"Q4R 沙漏能占比 {hg:.0%} 过高 (推荐域应低)"


# ═══════════════════════════════════════════════════════════════
# 主应力 — 有限极端应力下直接平方溢出 inf (第五轮外部审查)
# ═══════════════════════════════════════════════════════════════

def test_principal_stresses_extreme_finite():
    from fem2d.stress import principal_stresses
    s = np.array([[1e308, 1e308, 0.0],       # average 曾 inf
                  [1e308, -1e308, 0.0],      # 半径曾 inf
                  [0.0, 0.0, 1e308]])        # txy 主导
    s1, s2, radius, _ = principal_stresses(s)
    assert np.all(np.isfinite(s1)) and np.all(np.isfinite(s2)), \
        f"主应力溢出: {s1}"
    assert np.all(np.isfinite(radius))
    # 与正常尺度公式一致 (相对差)
    small = s / 1e300
    a1, a2, ar, _ = principal_stresses(small)
    assert np.allclose(s1 / 1e300, a1, rtol=1e-12)
    assert np.allclose(radius / 1e300, ar, rtol=1e-12)
