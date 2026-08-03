// L 形支架 — 凹角应力奇异性经典考题
// 顶部固定, 右侧受水平拉力
// 内凹角处应力理论上趋向无穷 → 适合测试 h-adaptivity 和误差估计
//
// 几何: 单位正方形挖去右上1/4 → 标准 L 形域
//     固定
//    ┌─────┐
//    │     │
//    │  ┌──┘
//    │  │     ← 凹角 (0.5, 0.5) — 应力奇异点
//    │  │
//    └──┴────→ 拉力

lc = 0.04;

// ── 外边界: L 形 (从凹角出发, CCW) ──
// 左下
Point(1) = { 0.0, 0.0, 0, lc};
// 右下
Point(2) = { 1.0, 0.0, 0, lc};
// 凹角右下
Point(3) = { 1.0, 0.5, 0, lc};
// 凹角
Point(4) = { 0.5, 0.5, 0, lc*0.5};  // 细化凹角区域
// 凹角左上
Point(5) = { 0.5, 1.0, 0, lc};
// 左上
Point(6) = { 0.0, 1.0, 0, lc};

Line(1) = {1, 2};  // 底边
Line(2) = {2, 3};  // 右边下半
Line(3) = {3, 4};  // 凹角水平边
Line(4) = {4, 5};  // 凹角垂直边
Line(5) = {5, 6};  // 顶边
Line(6) = {6, 1};  // 左边 — 固定

Curve Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};

// ── Physical Groups ──
Mesh.SaveAll = 1;
Physical Surface("domain", 200) = {1};
Physical Curve("左边_固定", 101) = {6};
Physical Curve("右边_拉力", 102) = {2};

// @FEM:fix=左边_固定
// @FEM:traction=右边_拉力,1000000.0,0

Mesh.Format = 39;
Mesh.SaveGroupsOfNodes = 1;
Mesh.SaveGroupsOfElements = 1;
Mesh 2;
Save "l_bracket.msh";
