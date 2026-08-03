# FEM2D 输入文件
# 用法: python run.py demo.spec
#
# 悬臂梁: 左边固定, 右端 1MPa 向右拉伸, 自重
# 运行前请确认边界编号与实际网格一致 (python run.py test_spec.txt 查看)

mesh     = test_spec.txt  # 相对路径以 .spec 所在目录为基准
E        = 2.10e11
nu       = 0.3
t        = 0.01
plane    = stress
fix      = left
traction = right:1e6,0
body     = 0,-78000
