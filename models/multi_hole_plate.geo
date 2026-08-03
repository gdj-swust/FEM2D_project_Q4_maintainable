// 三孔平板 — 多孔应力集中 + 孔间干涉
// 左边固定, 右端均匀拉伸 → 经典应力集中因子验证
//
// 三个孔水平排列, 间距递减 → 观察孔间应力干涉效应
//   固定                         拉力
//   ┌─────────────────────────┐
//   │   ⭕    ⭕   ⭕          │ → 1MPa
//   │  d=12  d=8  d=10       │
//   └─────────────────────────┘
//       ← 100mm →  厚10mm

lc = 0.4;

W = 100.0;  H = 40.0;

// ── 外矩形 ──
Point(1) = { 0,    0,    0, lc};
Point(2) = { W,    0,    0, lc};
Point(3) = { W,    H,    0, lc};
Point(4) = { 0,    H,    0, lc};

Line(1) = {1, 2};  // 底
Line(2) = {2, 3};  // 右
Line(3) = {3, 4};  // 顶
Line(4) = {4, 1};  // 左 — 固定

// ── 孔1: 中心 (20, 20), 半径 6 — 离左端最近 ──
R1 = 6.0;  cx1 = 20.0;  cy1 = 20.0;  n = 24;
For i In {0:n-1}
  ang = 2*Pi*i/n;
  Point(100 + i) = { cx1 + R1*Cos(ang), cy1 + R1*Sin(ang), 0, lc*0.6};
EndFor
Point(199) = { cx1, cy1, 0, lc*0.3};
For i In {0:n-1}
  Circle(200 + i) = { 100+i, 199, 100+((i+1)%n) };
EndFor
Curve Loop(301) = { 200:200+n-1 };

// ── 孔2: 中心 (45, 20), 半径 4 — 最小孔, 靠近孔1 ──
R2 = 4.0;  cx2 = 45.0;  cy2 = 20.0;
For i In {0:n-1}
  ang = 2*Pi*i/n;
  Point(400 + i) = { cx2 + R2*Cos(ang), cy2 + R2*Sin(ang), 0, lc*0.5};
EndFor
Point(499) = { cx2, cy2, 0, lc*0.3};
For i In {0:n-1}
  Circle(500 + i) = { 400+i, 499, 400+((i+1)%n) };
EndFor
Curve Loop(501) = { 500:500+n-1 };

// ── 孔3: 中心 (70, 20), 半径 5 — 离右端最近 ──
R3 = 5.0;  cx3 = 70.0;  cy3 = 20.0;
For i In {0:n-1}
  ang = 2*Pi*i/n;
  Point(700 + i) = { cx3 + R3*Cos(ang), cy3 + R3*Sin(ang), 0, lc*0.6};
EndFor
Point(799) = { cx3, cy3, 0, lc*0.3};
For i In {0:n-1}
  Circle(800 + i) = { 700+i, 799, 700+((i+1)%n) };
EndFor
Curve Loop(801) = { 800:800+n-1 };

// ── 外边界 + 带孔面 ──
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1, 301, 501, 801};

// ── Physical Groups ──
Mesh.SaveAll = 1;
Physical Surface("domain", 200) = {1};
Physical Curve("左边_固定", 101) = {4};
Physical Curve("右边_拉力", 102) = {2};

// @FEM:fix=左边_固定
// @FEM:traction=右边_拉力,1e6,0

Mesh.Format = 39;
Mesh.SaveGroupsOfNodes = 1;
Mesh.SaveGroupsOfElements = 1;
Mesh 2;
Save "multi_hole_plate.msh";
