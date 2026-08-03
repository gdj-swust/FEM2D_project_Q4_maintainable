// Q4 四边形示例: 悬臂板 2.0×0.5m
// 用法: python run.py models/plate_q4.geo --quad --fix left --traction right:1e6,0
lc = 0.1;

Point(1) = {0.0, 0.0, 0, lc};
Point(2) = {2.0, 0.0, 0, lc};
Point(3) = {2.0, 0.5, 0, lc};
Point(4) = {0.0, 0.5, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Curve("左端", 100) = {4};
Physical Curve("右端", 101) = {2};
Physical Curve("底边", 102) = {1};
Physical Curve("顶边", 103) = {3};
Physical Surface("板", 200) = {1};
Mesh.SaveAll = 1;
