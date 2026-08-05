"""覆盖轮 C1 — patch_test.py 缺口行 (子进程不计 coverage 的分支).

CST 无 stress_qp 的 qp_rel 回退与失败检测分支 — 主进程直调 run_patch_test
(子进程 __main__ 测试只验退出码, 不计覆盖率).
"""
from fem2d.patch_test import run_patch_test


def test_patch_test_cst_stress_qp_single_sample():
    """CST 单响应点 → stress_qp 与代表应力逐位一致 (单样本等价)."""
    r = run_patch_test(tol=1e-10, verbose=False, elem_type="CPS3")
    assert r["all_passed"] is True
    # qp 相对误差与代表应力误差同一尺度 (solve 恒产出 stress_qp)
    for t in r["tests"]:
        assert t["stress_qp_error"] <= 10.0 * t["s_error"]


def test_patch_test_failure_detected(capsys):
    """过严容差 → all_passed=False + [FAIL] 报告 (lines 189/211)."""
    r = run_patch_test(tol=1e-30, verbose=True, elem_type="CPS3")
    assert r["all_passed"] is False
    assert "[FAIL] SOME PATCH TESTS FAILED" in capsys.readouterr().out


def test_patch_test_plane_strain():
    """平面应变位移场分支 (line 92 区域) 全过."""
    r = run_patch_test(plane="strain", verbose=False, elem_type="CPS3")
    assert r["all_passed"] is True
