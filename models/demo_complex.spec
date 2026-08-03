# FEM2D — demo_complex 复杂几何分析
# 用法: python run.py models/demo_complex.spec

mesh     = demo_complex.geo  # .spec 同目录
lc       = 0.03
E        = 2.10e11
nu       = 0.3
t        = 0.01
plane    = stress

# 约束: 左边 + 底部直边
fix      = left, 底部

# 面力: 右端 1MPa 拉伸 + 椭圆孔内压 1MPa
traction = right:1e6,0;hole:1e6:n

# 体力: 自重
body     = 0,-78000
