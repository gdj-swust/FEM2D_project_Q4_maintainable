// 曲梁纯弯 — 90° 圆弧梁
// 一端固定, 自由端受纯弯矩 M → 验证曲边界 + 弯曲响应精度
//
// 几何: 内半径 40mm, 外半径 60mm, 90° 扇区
// 解析解 (Timoshenko): 环向应力 σ_θ 沿径向呈双曲线分布
//
//       固定端
//       ╱
//      ╱  60mm
//     │   ← 40mm
//     │
//      ╲
//       ╲____→ 自由端 (纯弯矩 M)
//
// 内弧圆心角90°, 理论最大 σ_θ ≈ 67.5 MPa (M=1000 N·mm)

lc = 0.8;

R_inner  = 40.0;
R_outer  = 60.0;
n_circle = 18;  // 90° 用 18 段

// ── 外弧 (从0°到-90°) — 圆心原点 ──
Point(100) = { 0.0, 0.0, 0, lc};

// 外弧: 从 (R_outer, 0) 到 (0, -R_outer)
// 内弧: 从 (R_inner, 0) 到 (0, -R_inner)

For i In {0:n_circle}
  ang = -Pi/2 * i / n_circle;  // 从 0 到 -π/2
  Point(1 + i) = { R_outer*Cos(ang), R_outer*Sin(ang), 0, lc};
  Point(101 + i) = { R_inner*Cos(ang), R_inner*Sin(ang), 0, lc};
EndFor

// 外弧: CCW (从右上到左下)
For i In {0:n_circle-1}
  Circle(200 + i) = { 1+i, 100, 1+(i+1) };
EndFor

// 内弧: 从固定端(θ=-90°)回到自由端(θ=0°), 与外弧反向
For i In {0:n_circle-1}
  Circle(300 + i) = { 101+n_circle-i, 100, 101+n_circle-1-i };
EndFor

// 径向连接边
Line(400) = { 101, 1 };                      // 自由端内→外 (θ=0)
Line(401) = { 1+n_circle, 101+n_circle };    // 固定端外→内 (θ=-90°)

// 闭环: 外弧 (1→19) → 径向 (19→119) → 内弧 (119→101) → 径向 (101→1)
Curve Loop(500) = { 200:200+n_circle-1, 401, 300:300+n_circle-1, 400 };
Plane Surface(1) = {500};

// ── Physical Groups ──
Mesh.SaveAll = 1;
Physical Surface("domain", 200) = {1};
Physical Curve("固定端", 101) = {401};
Physical Curve("自由端", 102) = {400};

// @FEM:fix=固定端
// @FEM:traction=自由端,0,10.0

Mesh.Format = 39;
Mesh.SaveGroupsOfNodes = 1;
Mesh.SaveGroupsOfElements = 1;
Mesh 2;
Save "curved_beam.msh";
