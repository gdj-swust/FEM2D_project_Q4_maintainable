# 三孔平板 — 多孔应力集中干涉
# 左边固定, 右端 1MPa 拉伸
# 三个不同大小/间距的孔 → 观察应力集中因子 K_t 和孔间干涉
# 理论 (Kirsch): 单孔 K_t ≈ 3.0; 多孔干涉会使 K_t 进一步升高
#
# 用法: python run.py models/multi_hole_plate.spec

mesh     = multi_hole_plate.geo
lc       = 0.4
E        = 2.10e11
nu       = 0.3
t        = 0.01
plane    = stress
fix      = ~左边
traction = ~右边:1e6,0
