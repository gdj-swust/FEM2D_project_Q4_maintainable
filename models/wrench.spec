# 扳手 — 混合曲边界工程件
# 圆头端固定, 手柄端受向下力 → 弯剪组合 + 孔边应力集中
#
# 用法: python run.py models/wrench.spec

mesh     = wrench.geo
lc       = 0.8
E        = 2.10e11
nu       = 0.3
t        = 0.005
plane    = stress
fix      = ~头部
traction = 手柄端:0,-3e6
