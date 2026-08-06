"""R-δ 轮判别性测试 — D1 GBK 提示 / D2 interactive_plot EOF / D3 聚合边互斥.

判别性 (回滚必须红):
  D1: GBK (cp936) 流下向导提示可打印 — 回滚 (N/m³, U+00B3) → 裸
      UnicodeEncodeError
  D2: 关闭 stdin 调 interactive_plot → 干净返回 — 回滚 (裸 input()) →
      裸 EOFError 泄漏
  D3: 选过聚合边 "内孔" 后子边不再列出 — 回滚 → 仍同时列出 内孔1..N
"""
import io

import numpy as np

from fem2d.config import AnalysisConfig
from fem2d.mesh import Mesh
import fem2d.wizard as wizard

# ═══════════════════════════════════════════════════════════════
# D1: 向导提示串在 GBK 流 (中文 Windows 默认代码页) 上可打印
# ═══════════════════════════════════════════════════════════════


def test_d1_body_prompt_gbk_stream(monkeypatch):
    """GBK 流下 _ask_body 提示串可编码 — U+00B3 回归必 UnicodeEncodeError.

    未重配 stdout 的直接 API/嵌入调用路径 (无 reconfigure 兜底) 下,
    input() 提示串含非 GBK 字符即硬崩溃 — 提示串必须全 ASCII 可编码。
    """
    stream = io.TextIOWrapper(io.BytesIO(), encoding="gbk")

    def ask_impl(prompt):
        stream.write(prompt)  # GBK 严格编码 — 含 ³ 在此抛错
        return ""

    monkeypatch.setattr(wizard, "ask", ask_impl)
    assert wizard._ask_body(AnalysisConfig()) is None

# ═══════════════════════════════════════════════════════════════
# D2: interactive_plot 关闭 stdin 不泄漏裸 EOFError
# ═══════════════════════════════════════════════════════════════


def test_d2_interactive_plot_eof_clean_exit(monkeypatch):
    """stdin 关闭 (EOF) → ask 返回空串 → 退出键, 干净返回不抛异常."""
    mesh = Mesh(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                np.array([[0, 1, 2]]), E=1e6, nu=0.3, thickness=1.0)
    # 空 StringIO = 立即 EOF — 裸 input() 在此抛 EOFError, ask 转空串
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    from fem2d.visualize import interactive_plot
    interactive_plot(mesh, {"u": np.zeros(6)})  # 不应抛

# ═══════════════════════════════════════════════════════════════
# D3: 聚合边 "内孔" 与子边 "内孔1..N" 互斥 (与 parse_spec 同族)
# ═══════════════════════════════════════════════════════════════


def test_d3_aggregate_used_excludes_subedges():
    """选过聚合边后子边必须全部排除 — 回滚 (无互斥) → 内孔1/2 仍在列."""
    edges = wizard._available_edges("rect", 2, {"内孔"})
    assert "内孔" not in edges                      # 已配置
    assert not [e for e in edges if e.startswith("内孔")], \
        f"聚合边已选, 子边仍可配置: {edges}"


def test_d3_subedge_used_excludes_aggregate():
    """选过任一子边后聚合边必须排除 (反向互斥)."""
    edges = wizard._available_edges("rect", 2, {"内孔1"})
    assert "内孔" not in edges, f"子边已选, 聚合边仍可配置: {edges}"
    assert "内孔2" in edges                         # 其余子边不受影响


def test_d3_single_hole_unaffected():
    """单孔: 无子边, 行为不变."""
    assert "内孔" in wizard._available_edges("rect", 1, set())
