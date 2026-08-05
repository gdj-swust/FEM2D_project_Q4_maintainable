"""FEM2D — 可扩展二维有限元分析 (CLI 入口)

所有逻辑在 ``fem2d/runner.py`` (主流程) 与 ``fem2d/cli.py`` (参数) 中,
本文件只转发到 ``runner.main`` (编码安全网在 runner.main 内, 供
``fem2d`` console script 与源码运行共用同一份)。

用法:
  python run.py <几何.geo>                     # 交互模式
  python run.py <几何.geo> --fix left --body 0,-78000    # 命令行模式
  python run.py <几何.geo> --fix left --traction right:1e6,0  # 右端受拉
  python run.py <几何.geo> --fix left --traction right:2e6,0:p  # 抛物线面力

面力格式:
  right:1e6,0      → 常数分布 (单位: Pa)
  right:2e6,0:p    → 抛物线分布 (tx,ty 为峰值, 合力 = 2/3 × 峰值 × 边长)
  right:1e6,0:l    → 线性分布 (tx,ty 为最大值, 一端=0)

边名称: left/right/top/bottom/hole 或数字(1-based) 或 @段名 (如 @底部)

入口选择指南 (什么场景用哪条, 一行一条 — 详见 docs/input_entries.md):
  python run.py <文件.geo/.msh/.txt/.spec>  # 标准入口: 任意模型文件, 完整求解
  python run_demo.py                        # 演示: 内置示例, 免模型文件
  fem2d <文件>                              # 已安装包: 与 run.py 等价的命令行
  python run.py <文件.spec>                 # 参数化: 载荷/材料写进 .spec 批处理
  gmsh <文件.geo> -2 -o <文件.msh>          # 只用 Gmsh 出网格, 再交给 run.py
"""
import sys

from fem2d.runner import main

ENTRY_GUIDE = """\

入口选择指南 (什么场景用哪条 — 详见 docs/input_entries.md):
  python run.py <文件.geo/.msh/.txt/.spec>  # 标准入口: 任意模型文件, 完整求解
  python run_demo.py                        # 演示: 内置示例, 免模型文件
  fem2d <文件>                              # 已安装包: 与 run.py 等价的命令行
  python run.py <文件.spec>                 # 参数化: 载荷/材料写进 .spec 批处理
  gmsh <文件.geo> -2 -o <文件.msh>          # 只用 Gmsh 出网格, 再交给 run.py
"""


def print_help_with_entry_guide(argv):
    """--help/-h: argparse 帮助后追加入口选择指南.

    cli.py 属包内参数层 (跨轮边界), 指南由脚本入口追加 — argparse
    打印帮助后抛 SystemExit(0), 捕获后追加再退出.
    """
    if "-h" not in argv and "--help" not in argv:
        return
    from fem2d.cli import parse_args
    try:
        parse_args(argv)
    except SystemExit:
        pass
    print(ENTRY_GUIDE)
    raise SystemExit(0)


if __name__ == "__main__":
    print_help_with_entry_guide(sys.argv[1:])
    sys.exit(main())
