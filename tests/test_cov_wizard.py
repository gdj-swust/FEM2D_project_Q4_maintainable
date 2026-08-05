"""覆盖轮 C1 — wizard.py 缺口行 (29 行).

交互向导: 提问原语 (非法重问/EOF 默认值) 用 monkeypatch input 驱动;
集成路径 (_build_and_generate) 走完整问答脚本, 不依赖真实 gmsh
(generate_geo 只写 .geo 文本).
"""
import builtins
import contextlib
import io

import pytest

import fem2d.wizard as W
from fem2d.config import AnalysisConfig
from fem2d.errors import CliError


def _drive(monkeypatch, inputs):
    """input 序列驱动器: 依次消费, 耗尽后抛 (防测试脚本长度失配)."""
    it = iter(inputs)

    def ask(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"input 脚本耗尽 (还差: {prompt!r})")
    monkeypatch.setattr("builtins.input", ask)
    return it


def _quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


# ── 提问原语: 非法重问 / EOF 默认值 / 范围检查 ──────────────────────────────

def test_ask_choice_empty_uses_default(monkeypatch):
    """空输入 → 返回 default-1 (0-based)."""
    _drive(monkeypatch, [""])
    assert _quiet(W._ask_choice, "p:", ["a", "b"], default=1) == 0


def test_ask_choice_invalid_then_valid(monkeypatch):
    """非数字 → 重问; 合法编号 → 0-based 索引."""
    _drive(monkeypatch, ["abc", "2"])
    assert _quiet(W._ask_choice, "p:", ["a", "b", "c"], default=1) == 1


def test_ask_choice_out_of_range(monkeypatch):
    """编号超范围 → 重问."""
    _drive(monkeypatch, ["9", "3"])
    assert _quiet(W._ask_choice, "p:", ["a", "b", "c"], default=1) == 2


def test_ask_vec2_empty_uses_default(monkeypatch):
    """空输入 + default → 解析默认串."""
    _drive(monkeypatch, [""])
    assert _quiet(W._ask_vec2, "p:", default="1e6,0") == (1e6, 0.0)


def test_ask_vec2_unparseable_retries(monkeypatch):
    """非数值分量 → 重问."""
    _drive(monkeypatch, ["abc", "1,2"])
    assert _quiet(W._ask_vec2, "p:") == (1.0, 2.0)


def test_ask_vec2_nonfinite_rejected(monkeypatch):
    """NaN/Inf 分量 → 重问 (禁止静默忽略)."""
    _drive(monkeypatch, ["nan,1", "1,2"])
    assert _quiet(W._ask_vec2, "p:") == (1.0, 2.0)


def test_ask_yes_invalid_retries(monkeypatch):
    """非 y/n → 重问."""
    _drive(monkeypatch, ["x", "y"])
    assert _quiet(W._ask_yes, "继续?") is True


# ── _ask_boundaries: 面力向量分支 ───────────────────────────────────────────

def test_ask_boundaries_traction_with_ty(monkeypatch):
    """拉力 ty≠0 → 字符串 'tx,ty'; 回车结束循环."""
    _drive(monkeypatch, ["下", "3", "2e6,5e5", ""])
    got = _quiet(W._ask_boundaries, "rect", 0)
    assert got == [{"edge": "下", "bc": "拉力",
                    "value": "2000000.0,500000.0"}]


def test_ask_boundaries_traction_ty_zero(monkeypatch):
    """ty=0 → 标量 tx (line 198)."""
    _drive(monkeypatch, ["下", "3", "1e6,0", ""])
    got = _quiet(W._ask_boundaries, "rect", 0)
    assert got == [{"edge": "下", "bc": "拉力", "value": 1000000.0}]


def test_ask_boundaries_vec_none_skips_edge(monkeypatch):
    """向量回答 None (default 恒存在时仅 monkeypatch 可达) → 跳过该边."""
    monkeypatch.setattr(W, "_ask_vec2", lambda *a, **k: None)
    _drive(monkeypatch, ["下", "3", "上", "1", ""])
    got = _quiet(W._ask_boundaries, "rect", 0)
    assert got == [{"edge": "上", "bc": "固定", "value": None}]


# ── _spec_to_txt 形状分支 ───────────────────────────────────────────────────

def _spec(stype, params, holes=()):
    return {"type": stype, "params": params, "holes": list(holes),
            "mesh_size": 0.1,
            "boundaries": [{"edge": "左", "bc": "固定", "value": None}],
            "body_force": None}


def test_spec_to_txt_circle():
    lines = W._spec_to_txt(_spec("circle", {"outer_r": 2.0}))
    assert "外半径 2.0" in lines and "圆板" in lines


def test_spec_to_txt_annulus_with_hole():
    lines = W._spec_to_txt(_spec(
        "annulus", {"outer_r": 2.0, "inner_r": 1.0},
        holes=[{"type": "circle", "x": 0.5, "y": 0.0, "r": 0.1}]))
    assert "外半径 2.0" in lines and "内半径 1.0" in lines
    assert "内孔 圆 x=0.5 y=0.0 r=0.1" in lines


# ── run_wizard / _build_and_generate 集成路径 ───────────────────────────────

def test_run_wizard_manual_path_empty_exits(monkeypatch):
    """手动输入路径 + 空输入 → CliError 退出码 0 (干净退出)."""
    _drive(monkeypatch, ["2", ""])
    with pytest.raises(CliError) as excinfo:
        _quiet(W.run_wizard, AnalysisConfig())
    assert excinfo.value.exit_code == 0


def test_build_and_generate_cancel_on_redo(monkeypatch):
    """确认建模 N → 重新建模 N → CliError 取消."""
    _drive(monkeypatch, [
        "1",          # 矩形板
        "1", "2",     # 宽/高
        "n",          # 不加孔
        "0.1",        # 网格密度
        "", "", "",   # E/nu/t 默认
        "",           # 边界回车结束
        "",           # 体力回车跳过
        "n",          # 确认建模 → N
        "n",          # 重新建模 → N → 取消
    ])
    with pytest.raises(CliError) as excinfo:
        _quiet(W._build_and_generate, AnalysisConfig())
    assert excinfo.value.exit_code == 0


def test_build_and_generate_geo_failure(monkeypatch):
    """generate_geo 抛 ValueError → CliError 退出码 1."""
    import scripts.geo_spec as GS
    monkeypatch.setattr(GS, "generate_geo",
                        lambda spec, path, quad=False: (_ for _ in ()).throw(
                            ValueError("bad geometry")))
    _drive(monkeypatch, [
        "1", "1", "2", "n", "0.1", "", "", "", "", "", "y",
    ])
    with pytest.raises(CliError) as excinfo:
        _quiet(W._build_and_generate, AnalysisConfig())
    assert excinfo.value.exit_code == 1
    assert "几何生成失败" in str(excinfo.value)


def test_build_and_generate_save_txt_oserror(monkeypatch, tmp_path):
    """保存 .txt 失败 (只读目录) → WARN 而非崩溃."""
    orig_open = builtins.open

    def fake_open(path, *a, **kw):
        if str(path).endswith(".txt"):
            raise OSError("read-only")
        return orig_open(path, *a, **kw)
    monkeypatch.setattr("builtins.open", fake_open)
    _drive(monkeypatch, [
        "1", "1", "2", "n", "0.1", "", "", "", "", "", "y",
        "y", str(tmp_path / "m.txt"),
    ])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        W._build_and_generate(AnalysisConfig())
    assert "保存失败" in out.getvalue()
