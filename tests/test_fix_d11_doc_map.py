"""R-δ 轮判别性测试 — D11 docs/CODE_MAP.md 符号引用 + 真实计数.

判别性 (回滚必须红):
  - 行号引用 (module.py:NNN / func:NNN) 重新出现 → 红
  - 计数回退 (35 顶层模块 / ~90 文件 / ~1060 函数) → 红
  - 文档标注的 module.symbol 引用不可解析 → 红
"""
import importlib
import re
from pathlib import Path

CODE_MAP = Path(__file__).resolve().parents[1] / "docs" / "CODE_MAP.md"

# 模块 → 可能所在包 (b 轮拆分后 boundary/ 子包)
_MODULE_CANDIDATES = {
    "boundary": ("fem2d",),  # 包级 re-export (build_boundary_segments)
    "runner": ("fem2d",),
    "input_source": ("fem2d",),
    "gmsh_adapter": ("fem2d",),
    "preprocess": ("fem2d",),
    "bc_apply": ("fem2d",),
    "bc": ("fem2d",),
    "solver": ("fem2d",),
    "error_est": ("fem2d",),
    "loads_core": ("fem2d",),
    "loads": ("fem2d",),
    "convergence": ("fem2d",),
    "verification": ("fem2d",),
    "stress": ("fem2d",),
    "spr": ("fem2d",),
    "visualize": ("fem2d",),
    "reporting": ("fem2d",),
    "quality": ("fem2d",),
    "wizard": ("fem2d",),
    "regions": ("fem2d",),
    "topology_core": ("fem2d",),
    "checks": ("fem2d",),
    "errors": ("fem2d",),
    "assembly": ("fem2d",),
    "material": ("fem2d",),
    "config": ("fem2d",),
    "patch_test": ("fem2d",),
    "mesh": ("fem2d",),
    "naming": ("fem2d.boundary",),
    "topology": ("fem2d.boundary",),
    "geometry": ("fem2d.boundary",),
    "physical_mapping": ("fem2d.boundary",),
    "registry_mapping": ("fem2d.boundary",),
    "selectors": ("fem2d.boundary",),
    "conic_merge": ("fem2d.boundary",),
    "base": ("fem2d.element",),
    "cst": ("fem2d.element",),
    "q4": ("fem2d.element",),
    "q4i": ("fem2d.element",),
    "q4r": ("fem2d.element",),
}


def _text() -> str:
    return CODE_MAP.read_text(encoding="utf-8")


def test_d11_no_line_number_references():
    """文档不得再含行号引用 — 拆分后必然漂移 (曾 12 处失效)."""
    text = _text()
    stale = re.findall(
        r"\b[a-zA-Z_][\w]*\.py:[0-9]+\b|"     # module.py:NNN
        r"\b[a-zA-Z_][\w]*:[0-9]{3}\b",       # func:NNN (三位数行号)
        text)
    assert not stale, f"CODE_MAP 仍含行号引用 (回滚 D11): {stale}"


def test_d11_counts_current():
    """计数回填真实值: 30 顶层模块 / 116 测试文件 / ~1400 函数."""
    text = _text()
    assert "30 顶层模块" in text, "顶层模块计数未回填"
    assert "116 文件" in text, "测试文件计数未回填"
    assert "1401" in text, "test 函数计数未回填 (grep def test_ 计数)"
    assert "~1400 函数" in text, "测试体系节标题计数未回填"


def test_d11_symbols_resolve():
    """文档中 module.symbol() 引用必须全部可解析 (符号漂移 → 红)."""
    unresolved = []
    for mod, sym in re.findall(
            r"\b([a-z_][\w]*)\.([a-zA-Z_][\w]*)\(\)", _text()):
        mod_attr = None
        for pkg in _MODULE_CANDIDATES.get(mod, ()):
            try:
                mod_attr = importlib.import_module(f"{pkg}.{mod}")
                break
            except ImportError:
                continue
        if mod_attr is None:
            unresolved.append(f"{mod}.{sym}: 模块不可解析")
            continue
        if not hasattr(mod_attr, sym):
            unresolved.append(f"{mod}.{sym}: 符号不存在")
    assert not unresolved, f"CODE_MAP 符号引用失效: {unresolved}"
