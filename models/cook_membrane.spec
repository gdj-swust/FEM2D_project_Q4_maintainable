# Cook's Membrane — FEM 经典基准测试
# 梯形板左端固定，右端受均匀剪切 τ_y = 1.0
# 注意: 本模型右下角为 (48,0) (右端全高 60、底边水平), 与标准 Cook 膜
# (右下角 (48,44)) 几何不同 — 位移不可与标准参考值 23.96 直接对比,
# 结果仅用于单元/网格间的相对比较。
#
# 用法: python run.py models/cook_membrane.spec

mesh     = cook_membrane.geo
lc       = 0.8
E        = 1.0
nu       = 0.333333
t        = 1.0
plane    = stress
fix      = 左
traction = 右:0,1.0
