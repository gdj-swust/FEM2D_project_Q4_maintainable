"""CST Patch Test — Bathe §5.3.3"""
from fem2d.patch_test import run_patch_test


def test_cst_patch_all_pass():
    result = run_patch_test(verbose=False)
    assert result["all_passed"], f"Patch test failed: {result}"


def test_cst_patch_three_tests_run():
    result = run_patch_test(verbose=False)
    assert len(result["tests"]) == 3, f"Expected 3 patch tests, got {len(result['tests'])}"
    for t in result["tests"]:
        assert t.get("passed", False), f"Patch test failed: {t.get('name', '?')}"

def test_patch_error_is_relative_at_micro_displacement_scale():
    """分片检验误差必须无条件相对化 — 曾 u_ref<1e-15 时退回绝对误差,
    微尺度位移模型下单元相对误差 50% 也能通过 (静默错误, 审计 2026-08-03).

    E=2.1e17 → 位移 ~1e-18 量级: 修复前 u_error≈1e-34 (绝对), 修复后
    ≈1e-16 (相对). 断言 >1e-20 判别绝对分支."""
    result = run_patch_test(E=2.1e17, verbose=False)
    assert result["all_passed"]
    for t in result["tests"]:
        assert t["u_error"] > 1e-20, \
            f"{t['test']}: u_error={t['u_error']:.3e} — 仍落入绝对误差分支"
        assert t["u_error"] < 1e-10, \
            f"{t['test']}: u_error={t['u_error']:.3e} — 相对误差异常"
