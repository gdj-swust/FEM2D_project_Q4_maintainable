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

边名称: left/right/top/bottom/hole 或数字(1-based)
"""
import sys

from fem2d.runner import main

if __name__ == "__main__":
    sys.exit(main())
