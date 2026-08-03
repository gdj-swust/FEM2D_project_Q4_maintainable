# L 形支架 — 凹角应力奇异性测试
# 左边固定, 右侧下半段受 1MPa 水平拉力
# 内凹角 (0.5, 0.5) 处应力理论值 → ∞
# 适合做网格加密收敛性研究和自适应误差估计
#
# 用法: python run.py models/l_bracket.spec

mesh     = l_bracket.geo
lc       = 0.04
E        = 2.10e11
nu       = 0.3
t        = 0.01
plane    = stress
fix      = ~左边
traction = ~右边:1e6,0
