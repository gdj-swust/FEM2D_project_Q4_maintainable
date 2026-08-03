# 曲梁纯弯 — 90° 圆弧梁 Timoshenko 验证
# 一端固定, 自由端受竖向力 (近似纯弯)
# 解析解: 环向应力 σ_θ 沿径向呈双曲线分布
#   最大 σ_θ (内弧) ≈ σ_avg × (4r_outer² / (r_outer² - r_inner²) - 1)
#
# 用法: python run.py models/curved_beam.spec

mesh     = curved_beam.geo
lc       = 0.8
E        = 2.10e11
nu       = 0.3
t        = 0.01
plane    = stress
fix      = 固定端
traction = 自由端:0,10.0
