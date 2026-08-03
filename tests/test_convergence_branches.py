"""convergence.py 分支补测 — 包 2 覆盖率任务.

未覆盖行集中: 三角形网格生成、verbose 输出块、采样点回退、Richardson
短序列、__main__ 入口分发。

判别性: 断言具体数组形状、收敛率数值区间、退出行为。
"""
from pathlib import Path

import pytest

import fem2d.convergence as conv_mod


def test_gen_cantilever_mesh_triangle_branch():
    """CST → 每格 2 三角, 形状 (2*nx*ny, 3)."""
    nodes, elems = conv_mod._gen_cantilever_mesh(5.0, 1.0, 2, 1, "CPS3")
    assert elems.shape == (4, 3)
    assert len(nodes) == (2 + 1) * (1 + 1)


def test_gen_cantilever_mesh_quad_branch():
    """Q4 → 每格 1 四边形."""
    nodes, elems = conv_mod._gen_cantilever_mesh(5.0, 1.0, 2, 1, "CPS4")
    assert elems.shape == (2, 4)


def test_richardson_extrapolation_and_tip_theory():
    """2 级细分的 Richardson 外推 ≈ 最细网格值 (平滑收敛)."""
    res = conv_mod.run_cantilever_convergence(
        refinements=2, verbose=False, elem_type="CPS3")
    assert len(res["h"]) == 2
    uy = res["uy_tip"]
    assert res["uy_richardson"] == pytest.approx(
        uy[-1] + (uy[-1] - uy[-2]) / (2.0**2 - 1), rel=1e-12)
    assert res["uy_tip_theory"] > 0.0          # 理论值方向正确 (P>0 下弯)


def test_verbose_prints_and_convergence_rates(capsys):
    """verbose 输出完整报告; CST 收敛率符合 u~O(h²), eta~O(h)."""
    res = conv_mod.run_cantilever_convergence(
        refinements=3, verbose=True, elem_type="CPS3")
    out = capsys.readouterr().out
    assert "Timoshenko Parabolic Shear Cantilever" in out
    assert "Per-level convergence rates (Richardson ref)" in out
    assert "Asymptotic convergence rates" in out
    assert 1.5 < res["uy_rate"] < 2.5, f"uy_rate={res['uy_rate']:.2f}"
    assert 0.5 < res["e_rate"] < 1.5, f"e_rate={res['e_rate']:.2f}"
    assert 0.5 < res["s_rate"] < 1.5, f"s_rate={res['s_rate']:.2f}"


def test_sampling_fallback_when_point_not_in_element(monkeypatch):
    """采样点不在单元内 (x=L/2 落在孔/间隙) → 回退 max|σ_xx|."""
    monkeypatch.setattr(conv_mod, "point_in_element", lambda *a, **k: -1)
    res = conv_mod.run_cantilever_convergence(
        refinements=1, verbose=False, elem_type="CPS3")
    s_ref = res["sigma_sample"][0]
    assert s_ref > 0.0
    # 单级序列: Richardson 参考 = 唯一最细网格值
    assert res["uy_richardson"] == res["uy_tip"][-1]


def test_module_main_entrypoint_dispatches_three_elements(monkeypatch):
    """``python -m fem2d.convergence`` → 依次跑 CPS3/CPS4/CPS4R."""
    calls = []
    monkeypatch.setattr(conv_mod, "run_cantilever_convergence",
                        lambda **k: calls.append(k))
    src = Path(conv_mod.__file__).read_text(encoding="utf-8")
    # 仅替换 __main__ 保护块内的调用 (def 行含同名子串, 不能全文替换),
    # 覆盖 3 元素分发而不重跑真实收敛
    src = src.replace(
        "run_cantilever_convergence(\n"
        "            refinements=5, verbose=True, elem_type=_elem_type)",
        "FAKE_RUN(refinements=5, verbose=True, elem_type=_elem_type)")
    globs = {
        "__name__": "__main__",
        "__package__": conv_mod.__package__,
        "__file__": conv_mod.__file__,
        "FAKE_RUN": lambda **k: calls.append(k),
    }
    exec(compile(src, str(conv_mod.__file__), "exec"), globs)
    assert [c["elem_type"] for c in calls] == ["CPS3", "CPS4", "CPS4R"]
    assert all(c["refinements"] == 5 for c in calls)
