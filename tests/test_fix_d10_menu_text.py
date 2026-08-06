"""R-δ 轮判别性测试 — D10 交互菜单文案含 @段名 通道.

判别性 (回滚必须红): 捕获输出断言 "@段名" 出现 — 回滚旧文案
("输入 @组名 整组选择" 无 @段名) → 红。

只锁文案; @段名/@组名 解析逻辑由 test_usability_round_d1 等锁定。
"""
import numpy as np

from fem2d import Mesh
from fem2d.bc_apply import _print_segment_menu
from fem2d.config import AnalysisConfig
from fem2d.runner import _print_boundaries


def _mesh():
    return Mesh(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                np.array([[0, 1, 3], [1, 2, 3]]), E=1e6, nu=0.3,
                thickness=1.0)


def test_d10_segment_menu_hint_mentions_segment_name(capsys):
    """交互边菜单提示行含 @段名 通道 (与 @组名 并列)."""
    segs = [{"type": "line", "label": "椭圆孔", "nodes": [0, 1],
             "info": {}}]
    _print_segment_menu(segs)
    out = capsys.readouterr().out
    assert "输入编号 12,13 精细选择" in out
    assert "@组名" in out, "菜单提示仍缺 @组名"
    assert "@段名" in out, "菜单提示未提 @段名 通道 (D10 未修)"


def test_d10_list_boundaries_hint_mentions_segment_name(capsys):
    """--list-boundaries 用法示例含 @段名 通道."""
    segs = [{"type": "line", "label": "底边", "nodes": [0, 1],
             "info": {}}]
    _print_boundaries(AnalysisConfig(), _mesh(), segs)
    out = capsys.readouterr().out
    assert "@组名" in out
    assert "@段名" in out, "--list-boundaries 示例未提 @段名 (D10 未修)"
