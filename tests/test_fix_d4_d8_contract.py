"""R-δ 轮判别性测试 — D4-D8 契约表修正锁定.

判别性 (回滚契约必须红): 断言 docs/api_contract.md 含修正后的表述 —
  D4: E 行 unsupported ext/.inp → CliError(exit 1) (曾记 exit 2)
  D5: G 行 validate_boundary_segments 缺段 → ValueError 带统计
  D6: B 行 apply_penalty 补记 OverflowError
  D7: B 行 solve 补记 penalty 仅接受 auto/direct
  D8: 探针数字 151 项 + (3,) 行改为如实锁定声明
契约侧的可执行判别性由 scripts/audit_contract_probe.py 承担
(exit_code 断言 / (3,) 正路径断言) — 本文件锁文档文本。
"""
import re
from pathlib import Path

CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "api_contract.md"


def _text() -> str:
    return CONTRACT.read_text(encoding="utf-8")


def test_d4_contract_e_row_exit_1():
    """E 行: 不支持的扩展名/.inp → exit 1 (用户错误归 1).

    旧表述 "扩展名不支持 → CliError(exit 2)" 必须消失; 状态列的
    历史注记 ("曾记 exit 2") 保留无妨。
    """
    text = _text()
    assert re.search(r"扩展名不支持 → CliError\(exit 1\)", text)
    assert re.search(r"\.inp → CliError\(exit 1", text)
    assert "扩展名不支持 → CliError(exit 2)" not in text, \
        "契约仍残留 exit 2 旧表述"


def test_d5_contract_g_row_throws_valueerror():
    """G 行: 缺段 → ValueError 带统计 (硬校验, 不再记"不抛")."""
    text = _text()
    assert "缺段/闭合缺失 → ValueError 带统计" in text
    assert "missing/extra/duplicated" in text
    assert "诊断报告 (不抛" not in text
    # 旧行文本 "缺段/闭合缺失 → 诊断报告 (不抛, 返回诊断)" — 逗号区分
    # 新契约的历史注记 ("曾记不抛返回诊断")
    assert "不抛, 返回诊断" not in text, "契约仍残留'不抛返回诊断'旧表述"


def test_d6_contract_overflowerror_recorded():
    """B 行: apply_penalty 自动罚因子溢出 → OverflowError 已记载.

    锁定行级表述 (表头注记也含 OverflowError 字样, 须以行级
    独有文本 "→ OverflowError (极端 corner" 区分).
    """
    assert "→ OverflowError (极端 corner" in _text()


def test_d7_contract_penalty_solver_limit():
    """B 行: penalty 方法仅接受 auto/direct 已记载."""
    assert "penalty 方法仅接受 auto/direct" in _text()


def test_d8_contract_numbers_and_3_path():
    """探针数字回填真实值 (151); (3,) 行改为如实锁定声明."""
    text = _text()
    assert text.count("151 项") >= 2, "表头与 M1 节均应为 151 项"
    assert "100 项" not in text, "探针计数陈旧数字 100 项未清零"
    assert "探针 (3,) 正路径断言" in text, "(3,) 行缺探针正路径断言声明"
    assert "锁不抛异常" in text, "(3,) 行缺 fuzz 职责如实声明"
