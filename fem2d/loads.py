"""Public equivalent-load API — 兼容层 facade.

纯 re-export fem2d.loads_core 的公共符号:
保留本层以稳定 ``from fem2d.loads import ...`` 公共导入路径;
实现唯一在 loads_core.py, 本文件不应新增逻辑。
"""
from .loads_core import (
    LINE_GAUSS,
    assemble,
    make_edge_profile_func,
    parse_traction,
    parse_vec2,
)

__all__ = [
    "LINE_GAUSS",
    "assemble",
    "make_edge_profile_func",
    "parse_traction",
    "parse_vec2",
]
