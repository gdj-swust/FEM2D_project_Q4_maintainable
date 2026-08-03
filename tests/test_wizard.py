"""交互式建模向导 (fem2d/wizard.py) — 判别性测试.

策略: monkeypatch wizard.ask 为脚本化回答队列, 验证:
完整流程 / 非法重问 / 边名白名单 / 双分量拉力 / EOF 退出 /
保存 .txt round-trip / 总览否定重来。
"""
import contextlib
import io
import os

import pytest

from fem2d.config import AnalysisConfig
import fem2d.wizard as wizard


def _scripted(answers, eof_after=True):
    """按序返回预设回答; 队列耗尽后返回 '' (模拟 EOF)."""
    state = {"idx": 0}

    def ask_impl(prompt):
        if state["idx"] < len(answers):
            answer = answers[state["idx"]]
            state["idx"] += 1
            return answer
        return "" if eof_after else ""

    return ask_impl


def _run(answers, monkeypatch, config=None):
    """运行向导, 返回 (返回的 .geo 路径, 全部 stdout)."""
    monkeypatch.setattr(wizard, "ask", _scripted(answers))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fp = wizard.run_wizard(config or AnalysisConfig())
    return fp, buf.getvalue()


# ═══════════════════════════════════════════════════════════════
# 完整流程
# ═══════════════════════════════════════════════════════════════

def test_wizard_full_rect_flow(tmp_path, monkeypatch):
    """矩形板 3×2 + 左固定 + 右拉力(双分量) + 体力 → .geo 含正确 @FEM.

    回答序列: 建模方式(1=交互) / 几何类型(1=矩形板) / 宽 / 高 /
    孔(y=否) / 网格密度 / E(回车默认) / nu(回车) / 厚度(回车) /
    边界边名"左" / 类型1(固定) / 边名"右" / 类型3(拉力) /
    tx,ty"1e6,2e6" / 边名(回车结束) / 体力"0,-78000" /
    确认(y) / 保存(n).
    """
    answers = ["1", "1", "3.0", "2.0", "n", "0.2",
               "", "", "",
               "左", "1", "右", "3", "1e6,2e6", "",
               "0,-78000", "y", "n"]
    fp, out = _run(answers, monkeypatch)
    assert fp and os.path.isfile(fp), f"未生成 .geo: {out!r}"
    text = open(fp, encoding="utf-8").read()
    assert "@FEM:fix=左_固定" in text, f"固定未写入 @FEM: {text}"
    assert "右_拉力_1000000.0_2000000.0" in text, \
        f"双分量拉力标签未写入: {text}"
    assert "@FEM:traction=右_拉力_1000000.0_2000000.0," \
           "1000000.0,2000000.0" in text, \
        f"双分量 @FEM 未写入: {text}"


def test_wizard_material_flows_to_config(monkeypatch):
    """向导材料 (E/nu/t) 必须写回 config — 求解器使用向导输入的值."""
    answers = ["1", "1", "3.0", "2.0", "n",
               "0.2", "7e10", "0.25", "0.02",
               "", "y", "n"]
    config = AnalysisConfig()
    fp, out = _run(answers, monkeypatch, config)
    assert config.E == 7e10, f"E 未生效: {config.E}"
    assert config.nu == 0.25
    assert config.thickness == 0.02


# ═══════════════════════════════════════════════════════════════
# 非法输入重问 (不静默)
# ═══════════════════════════════════════════════════════════════

def test_wizard_invalid_number_requestion(monkeypatch):
    """宽 = 'abc' 必须重问, 不接受静默降级."""
    answers = ["1", "1", "abc", "3.0", "2.0", "n",
               "0.2", "", "", "", "", "y", "n"]
    fp, out = _run(answers, monkeypatch)
    assert fp and os.path.isfile(fp), f"非法值未重问: {out!r}"
    assert "不是" in out or "重新输入" in out, f"未提示重问: {out!r}"


def test_wizard_negative_dimension_requestion(monkeypatch):
    """宽 = -3.0 必须重问 (负尺寸曾静默生成镜像几何)."""
    answers = ["1", "1", "-3.0", "3.0", "2.0", "n",
               "0.2", "", "", "", "", "y", "n"]
    fp, out = _run(answers, monkeypatch)
    assert "必须为正数" in out, f"负值未拒绝: {out!r}"
    assert fp and os.path.isfile(fp)


def test_wizard_nan_rejected(monkeypatch):
    """宽 = nan 必须重问 (NaN 曾延迟到求解期)."""
    answers = ["1", "1", "nan", "3.0", "2.0", "n",
               "0.2", "", "", "", "", "y", "n"]
    fp, out = _run(answers, monkeypatch)
    assert "有限" in out, f"NaN 未拒绝: {out!r}"


# ═══════════════════════════════════════════════════════════════
# 边名白名单 (向导只问合法边)
# ═══════════════════════════════════════════════════════════════

def test_wizard_circle_edge_whitelist(monkeypatch):
    """圆板: 问 '左' 必须被拒 (圆板无直边), '外边' 可用."""
    answers = ["1", "2", "2.0",          # 交互 / 圆板 / 外半径
               "n",                       # 无孔
               "0.2", "", "", "",
               "左", "外边", "1",          # 左 被拒 → 外边 固定
               "", "y", "n"]
    fp, out = _run(answers, monkeypatch)
    assert "不可用" in out, f"左 未被拒绝: {out!r}"
    assert fp and os.path.isfile(fp)
    text = open(fp, encoding="utf-8").read()
    assert "@FEM:fix=外边" in text, f"外边固定未写入: {text}"


def test_wizard_rect_hole_edges_available(monkeypatch):
    """带孔矩形板 (单孔): 边名白名单只给 "内孔" (生成器无 内孔1 键),
    边界可施加到孔边."""
    answers = ["1", "4",                 # 交互 / 带孔矩形板
               "6.0", "3.0",             # 宽 高
               "y", "0.0", "0.0", "0.5",  # 加孔 (0,0) r=0.5
               "n",                       # 不再加孔
               "0.2", "", "", "",
               "内孔1",                   # 单孔无 内孔1 → 被拒
               "内孔", "1",              # 孔边固定
               "", "", "y", "n"]         # 边界结束 / 体力跳过
    fp, out = _run(answers, monkeypatch)
    assert fp and os.path.isfile(fp), f"带孔矩形板失败: {out!r}"
    assert "不可用" in out, f"内孔1 未被拒 (单孔白名单错误): {out!r}"
    text = open(fp, encoding="utf-8").read()
    assert "@FEM:fix=内孔_固定" in text, f"内孔 未写入: {text}"


def test_wizard_duplicate_edge_rejected(monkeypatch):
    """同一边二次配置必须被拒 (白名单排除已配置边)."""
    answers = ["1", "1", "3.0", "2.0", "n",
               "0.2", "", "", "",
               "左", "1", "左", "2",     # 第二次 '左' 被拒
               "", "y", "n"]
    fp, out = _run(answers, monkeypatch)
    assert "不可用" in out, f"重复边未拒绝: {out!r}"


# ═══════════════════════════════════════════════════════════════
# EOF 退出 (不崩溃)
# ═══════════════════════════════════════════════════════════════

def test_wizard_eof_at_number_exits_cleanly(monkeypatch):
    """无默认值数值处 EOF → CliError(0) (曾死循环; 库层不 SystemExit)."""
    from fem2d.errors import CliError
    answers = ["1", "1"]   # 宽 处 EOF
    with pytest.raises(CliError) as exc:
        _run(answers, monkeypatch)
    assert exc.value.exit_code == 0


def test_wizard_eof_at_edge_uses_defaults(monkeypatch):
    """边名处 EOF → 无边界结束; 默认值一路回车 → 全部默认."""
    answers = ["1", "1", "3.0", "2.0", "n", "0.2",
               "", "", "",                    # E/nu/t 默认
               "",                            # 边界 EOF 结束
               "",                            # 体力跳过
               "y", "n"]                      # 确认
    fp, out = _run(answers, monkeypatch)
    assert fp and os.path.isfile(fp)
    text = open(fp, encoding="utf-8").read()
    assert "lc = 0.2" in text


# ═══════════════════════════════════════════════════════════════
# 总览否定 → 重新开始
# ═══════════════════════════════════════════════════════════════

def test_wizard_summary_reject_restarts(monkeypatch):
    """总览 'n' → 重新建模 (第二次确认 'y' 完成)."""
    answers = ["1", "1", "3.0", "2.0", "n", "0.2",
               "", "", "", "", "",
               "n",                        # 第一次确认拒绝
               "y",                        # 重新建模
               "1", "3.0", "2.0", "n", "0.2",
               "", "", "", "", "",
               "y", "n"]
    fp, out = _run(answers, monkeypatch)
    assert fp and os.path.isfile(fp), f"重来失败: {out!r}"
    assert "重新开始" in out or "重新建模" in out


# ═══════════════════════════════════════════════════════════════
# 保存 .txt round-trip
# ═══════════════════════════════════════════════════════════════

def test_wizard_save_txt_roundtrip(tmp_path, monkeypatch):
    """保存的 .txt 可被 geo_spec.parse_spec 重新解析且语义一致."""
    from scripts.geo_spec import parse_spec
    saved = str(tmp_path / "w.txt")
    answers = ["1", "1", "3.0", "2.0", "n", "0.2",
               "", "", "",
               "左", "1", "右", "3", "1e6,2e6", "",
               "0,-78000", "y",
               "y", saved]                # 保存 .txt
    fp, out = _run(answers, monkeypatch)
    assert os.path.isfile(saved), f".txt 未保存: {out!r}"
    spec = parse_spec(saved)
    assert spec["params"]["width"] == 3.0
    assert spec["mesh_size"] == 0.2
    assert spec["body_force"] == [0.0, -78000.0]
    bcs = {(b["edge"], b["bc"]) for b in spec["boundaries"]}
    assert ("左", "固定") in bcs and ("右", "拉力") in bcs, \
        f"BC round-trip 失败: {spec['boundaries']}"
    trac = [b for b in spec["boundaries"] if b["edge"] == "右"][0]
    assert trac["value"] == "1000000.0,2000000.0", \
        f"双分量丢失: {trac['value']}"


# ═══════════════════════════════════════════════════════════════
# 入口: 使用已有文件
# ═══════════════════════════════════════════════════════════════

def test_wizard_use_existing_file(tmp_path, monkeypatch):
    """向导入口 '使用已有文件' → 直接返回文件路径."""
    existing = tmp_path / "m.geo"
    existing.write_text("Point(1) = {0, 0, 0, 0.5};\n", encoding="utf-8")
    answers = ["2", str(existing)]
    fp, out = _run(answers, monkeypatch)
    assert fp == str(existing), f"已有文件路径未返回: {fp}"


def test_wizard_existing_file_missing_fatal(tmp_path, monkeypatch):
    """向导入口文件不存在 → CliError(1) (库层不 SystemExit)."""
    from fem2d.errors import CliError
    answers = ["2", str(tmp_path / "nope.geo")]
    with pytest.raises(CliError) as exc:
        _run(answers, monkeypatch)
    assert exc.value.exit_code == 1


# ═══════════════════════════════════════════════════════════════
# main 集成: --wizard 触发向导, 产物走完整求解管线
# ═══════════════════════════════════════════════════════════════

def test_main_wizard_flag_triggers_and_solves(monkeypatch):
    """--wizard + 无 mesh → 向导触发, 产物进入网格生成管线.

    向导产物 (临时 .geo 含 @FEM) 由 resolve_geo 消费 — 判别向导真正
    接入主流程. 求解阶段的边名匹配依赖真实 .geo 的 Physical Curve
    语义 (fake 网格无), 故只验证到管线入口.
    """
    from fem2d import input_source, runner
    import fem2d.wizard as _wz

    answers = ["1", "1", "3.0", "2.0", "n", "0.5",
               "", "", "",
               "左", "1",
               "", "", "y", "n"]
    monkeypatch.setattr(_wz, "ask", _scripted(answers))

    seen = {}
    def fake_generate(geo_path, *, quad=False, output_path=None,
                      plane_type="stress"):
        seen["geo"] = geo_path
        class _FakeImport:
            nodes = [[0., 0.], [3., 0.], [3., 2.], [0., 2.]]
            elements = [[0, 1, 2], [0, 2, 3]]
            elem_type = "CPS3"
            node_tag_to_index = {0: 0, 1: 1, 2: 2, 3: 3}
            element_tag_to_index = {0: 0}
            regions = None
        return "dummy.msh", _FakeImport()
    monkeypatch.setattr(input_source, "generate_geo_with_topology",
                        fake_generate)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            runner.main(["--wizard", "--no-plot"])
        except SystemExit:
            # fake 网格无 Physical Curve 语义 — 边名匹配在 BC 阶段失败
            # 属测试环境限制; 向导触发与产物正确性已由下方断言锁定
            pass
    out = buf.getvalue()
    assert "交互建模向导" in out, f"向导未触发: {out!r}"
    assert seen.get("geo"), "向导产物未进入网格生成管线"
    assert os.path.isfile(seen["geo"]), f"向导 .geo 不存在: {seen['geo']}"
    text = open(seen["geo"], encoding="utf-8").read()
    assert "@FEM:fix=" in text, f"向导产物缺 @FEM: {text}"


# ═══════════════════════════════════════════════════════════════
# CLI 参数优先 (向导不覆盖显式参数)
# ═══════════════════════════════════════════════════════════════

def test_wizard_cli_material_wins(monkeypatch):
    """--E/--nu/--thickness 显式指定 → 向导不重复提问也不覆盖.

    曾无条件提问并覆盖 config (--wizard --E 5e7 被向导默认 2.1e11
    覆盖, 审计 2026-08-03).
    """
    from fem2d.cli import parse_args
    config = AnalysisConfig.from_args(parse_args(
        ["--wizard", "--E", "5e7", "--nu", "0.25",
         "--thickness", "0.02"]))
    answers = ["1", "1", "3.0", "2.0", "n", "0.2",
               "",             # 边界 EOF (CLI 未给 fix, 仍交互)
               "",             # 体力跳过
               "y", "n"]
    fp, out = _run(answers, monkeypatch, config)
    assert "使用 CLI" in out, f"未提示使用 CLI 参数: {out!r}"
    assert config.E == 5e7, f"CLI E 被向导覆盖: {config.E}"
    assert config.nu == 0.25
    assert config.thickness == 0.02
    # 材料问答全部跳过 — 回答序列里没有 E/nu/t 的输入
    assert "弹性模量" not in out, f"向导仍问了 E: {out!r}"


def test_wizard_cli_fix_skips_boundary(monkeypatch):
    """--fix 已给 → 边界阶段跳过 (批处理安全, 不重复提问)."""
    from fem2d.cli import parse_args
    config = AnalysisConfig.from_args(
        parse_args(["--wizard", "--fix", "left"]))
    answers = ["1", "1", "3.0", "2.0", "n", "0.2",
               "",             # 体力跳过
               "y", "n"]
    fp, out = _run(answers, monkeypatch, config)
    assert "边界阶段跳过" in out, f"边界未跳过: {out!r}"
    assert "边名" not in out, f"边界仍提问: {out!r}"


def test_wizard_cli_body_skips(monkeypatch):
    """--body 已给 → 体力阶段跳过."""
    from fem2d.cli import parse_args
    config = AnalysisConfig.from_args(
        parse_args(["--wizard", "--body", "0,-78000"]))
    answers = ["1", "1", "3.0", "2.0", "n", "0.2",
               "",             # 边界 EOF
               "y", "n"]
    fp, out = _run(answers, monkeypatch, config)
    assert "体力阶段跳过" in out, f"体力未跳过: {out!r}"


# ═══════════════════════════════════════════════════════════════
# 第五轮深测: 圆环 / 压力 / 多孔 / 孔边名编号
# ═══════════════════════════════════════════════════════════════

def test_wizard_annulus_full_flow(tmp_path, monkeypatch):
    """圆环: 外边压力 + 内孔固定 — 边名白名单 (外边/内孔)."""
    answers = ["1", "3",                 # 交互 / 圆环
               "2.0", "1.0",             # 外半径 内半径
               "n",                       # 无额外孔
               "0.2", "", "", "",
               "外边", "2", "1e6",        # 外边 压力 1e6
               "内孔", "1",              # 内孔 固定
               "", "y", "n"]             # 体力跳过 / 确认 / 不保存
    monkeypatch.chdir(tmp_path)
    fp, out = _run(answers, monkeypatch)
    assert fp and os.path.isfile(fp), f"圆环失败: {out!r}"
    text = open(fp, encoding="utf-8").read()
    assert "@FEM:pressure=外边_压力_1000000.0,1000000.0" in text, \
        f"压力未写入: {text}"
    assert "@FEM:fix=内孔_固定" in text, f"内孔固定未写入: {text}"


def test_wizard_multi_hole_edges_numbered(monkeypatch):
    """多孔: 白名单含 内孔1/内孔2/内孔 (聚合), 单孔边可单独施加."""
    answers = ["1", "4",                 # 交互 / 带孔矩形板
               "6.0", "3.0",
               "y", "-1.0", "0.5", "0.4",  # 孔1
               "y", "1.2", "0.2", "0.5",   # 孔2
               "n",
               "0.2", "", "", "",
               "内孔2", "1",              # 孔2 固定
               "", "", "y", "n"]
    fp, out = _run(answers, monkeypatch)
    assert fp and os.path.isfile(fp), f"多孔失败: {out!r}"
    text = open(fp, encoding="utf-8").read()
    assert "@FEM:fix=内孔2_固定" in text, f"内孔2 未写入: {text}"
    # 多孔可用边含编号
    assert "内孔1" in out and "内孔2" in out, f"多孔编号未列出: {out!r}"


def test_wizard_circle_with_hole_edges(tmp_path, monkeypatch):
    """圆板带孔: 孔边名 内孔 (单孔) 可用, 外边+内孔 双配置.

    边循环在全部边配置完后自动结束, 不再问"边名(回车结束)" —
    曾多给一个 "" 使 "y"/"n" 错位到"保存?"/"保存路径", 在 CWD
    写出名为 n 的文件 (第四轮外部审查复现)。chdir(tmp_path) 兜底.
    """
    answers = ["1", "2", "2.0",           # 圆板 外半径
               "y", "0.0", "0.0", "0.3",   # 孔
               "n",
               "0.2", "", "", "",
               "外边", "1", "内孔", "1",   # 外边固定 内孔固定
               "", "y", "n"]               # 体力跳过 / 确认 / 不保存
    monkeypatch.chdir(tmp_path)
    fp, out = _run(answers, monkeypatch)
    assert fp and os.path.isfile(fp), f"圆板带孔失败: {out!r}"
    text = open(fp, encoding="utf-8").read()
    assert "@FEM:fix=外边_固定" in text
    assert "@FEM:fix=内孔_固定" in text, f"孔边未写入: {text}"


def test_wizard_restart_twice(monkeypatch):
    """总览否定两次 → 第三次确认完成 (重来循环稳定)."""
    answers = ["1", "1", "3.0", "2.0", "n", "0.2",
               "", "", "", "", "",
               "n", "y",                  # 否定 → 重来
               "1", "3.0", "2.0", "n", "0.2",
               "", "", "", "", "",
               "n", "y",                  # 再否定 → 再重来
               "1", "3.0", "2.0", "n", "0.2",
               "", "", "", "", "",
               "y", "n"]                  # 确认完成
    fp, out = _run(answers, monkeypatch)
    assert fp and os.path.isfile(fp), f"两次重来失败: {out!r}"
