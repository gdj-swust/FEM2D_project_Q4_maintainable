"""D-α 轮判别性测试 — docs 数字漂移守卫 (README/CODE_MAP 硬编码锁定).

判别性 (回退必须红):
  - README 测试体系数字回退 (134 文件 / 1628 测试) → 红
  - CODE_MAP 测试体系数字回退 (134 文件 / ~1552 函数) → 红
  - 仓库实际测试文件数 / 测试函数数与文档声明不符 → 红
  - 新增测试文件/函数而不同步文档 → 红 (动态口径核对)

⚠️ 本文件内禁止出现 def 测试函数前缀的连续字样 (含注释/docstring) —
文档计数口径是 grep 全行匹配, 该字样会把本文件自身计入函数数, 污染
口径。断言所需字样用 _DF 拼接构造。

P-λ 规则: 本文件锁定值 = docs 数字同步例行范围内的判别性语义
(见 task_sheets_doc/PROMPT_D-alpha.md)。数字更新流程 (P-λ "顺序注意"):
分支实测 → 同步 README/CODE_MAP/ARCH/performance 四处 → 本文件 +
test_fix_d11_doc_map.py 锁定值同步 → 合入前复测同批数字。

口径与 d11 一致:
  - 文件数 = tests/test_*.py 数量 (含本守卫自身, 全仓库口径)
  - 函数数 = grep def 测试函数前缀 tests/test_*.py 行计数 (含本守卫
    自身; 本文件避免该前缀字样 → 行计数 == 真实函数数)
  - 收集数 = pytest --collect-only -q 每文件计数求和
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CODE_MAP = ROOT / "docs" / "CODE_MAP.md"

# 硬编码锁定值 (2026-08-06, D-α 实测 @ main 810b7bd + 本守卫文件计入):
# P轮Ⅱ 七包合入前 134 文件 / ~1552 函数 / 1628 收集 → 合入后
# 141 文件 / 1627 函数 / 1790 收集 → 本包新增守卫文件 → 142/1630/1793
# → S-α 新增 test_s_alpha_security.py (19 函数) → 143/1649/1812
# → GUI v12 新增 test_gui_logic.py (26 收集) → 144/1667/1838
# → msh 复用新增 5 (test_resolve_geo_*) + GUI 组选择新增 3 (test_apply_identification_*/test_add_bc_*) → 144/1675/1846.
# → SPR-BC-2026-001 新增 test_spr_boundary.py (16 函数/24 收集) → 145/1699/1878.
# → 状态栏读值/blit 刷新 (16 函数/17 收集) → 145/1715/1895.
# → 流畅性备忘 (SPR/traction 缓存, 2 函数/2 收集) → 145/1717/1897.
# → SPR 边界恢复会话 (stress 探针单次定位 2 + msh 密度元数据 4 + 生成标记 2
#   + visualize 分支 11, 共 19 函数/19 收集) → 145/1736/1916.
TEST_FILES = 145
TEST_FUNCTIONS = 1736
COLLECTED = 1916

# 前缀字样由拼接构造 — 禁止在本文件出现连续序列
_DF = "def test" + "_"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _repo_test_file_count() -> int:
    """tests/test_*.py 全仓库计数 (含本守卫自身 — 口径与 ls 实测一致)."""
    return len(list(ROOT.glob("tests/test_*.py")))


def _repo_test_function_count() -> int:
    """文档口径: grep 全行匹配 tests/test_*.py 的 def 函数行计数.

    用 _DF in line 与 grep 等价 (跨平台, 无需 subprocess);
    本文件自身不含该前缀, 计数 == 真实函数数.
    """
    total = 0
    for f in ROOT.glob("tests/test_*.py"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if _DF in line:
                total += 1
    return total


def test_guard_readme_test_suite_numbers():
    """README 测试体系数字锁定: 142→143 文件 / 1793→1812 收集.

    回退 (134 文件 / 1628 测试 重现) 必须红 — 硬编码判别性语义.
    """
    text = _read(README)
    assert f"{TEST_FILES} 个测试文件 + conftest.py" in text, \
        f"README 测试文件数未同步 ({TEST_FILES})"
    assert f"{COLLECTED} 测试" in text, \
        f"README 测试收集数未同步 ({COLLECTED})"
    # 旧值回归即红 (P轮Ⅱ 前数字不得复现)
    assert "134 个测试文件" not in text, "README 回退到 134 个测试文件"
    assert "1628 测试" not in text, "README 回退到 1628 测试"


def test_guard_code_map_test_suite_numbers():
    """CODE_MAP 测试体系数字锁定: 143 文件 / ~1649 函数.

    d11 已锁 CODE_MAP, 本测试覆盖 README + 全仓库口径 — 双锁防漏.
    """
    text = _read(CODE_MAP)
    assert f"{TEST_FILES} 文件" in text, \
        f"CODE_MAP 测试文件数未同步 ({TEST_FILES})"
    assert f"~{TEST_FUNCTIONS} test 函数" in text, \
        f"CODE_MAP test 函数数未同步 (~{TEST_FUNCTIONS})"
    assert f"{_DF}: {TEST_FUNCTIONS}" in text, \
        f"CODE_MAP 函数数括号注记未同步 ({_DF}: {TEST_FUNCTIONS})"
    # 旧值回归即红 (P轮Ⅱ 前数字不得复现)
    assert "134 文件" not in text, "CODE_MAP 回退到 134 文件"
    assert "1552" not in text, "CODE_MAP 回退到 ~1552 函数"


def test_guard_repo_actual_counts_match_docs():
    """全仓库数字口径动态核对: 实际文件数/函数数 == 文档锁定值.

    未来任何测试文件增删或测试函数增删而不同步文档 → 红.
    (收集数 1793 由 pytest 收集生成, 无轻量静态口径, 仅文档文本锁定;
    文件数/函数数对齐后收集数漂移概率极低.)
    """
    files = _repo_test_file_count()
    funcs = _repo_test_function_count()
    assert files == TEST_FILES, (
        f"tests/test_*.py 实际 {files} 个 != 锁定 {TEST_FILES} — "
        f"新增/删除测试文件后需同步 README/CODE_MAP 数字")
    assert funcs == TEST_FUNCTIONS, (
        f"{_DF} 行实际 {funcs} 个 != 锁定 {TEST_FUNCTIONS} — "
        f"测试函数增减后需同步 README/CODE_MAP 数字")
