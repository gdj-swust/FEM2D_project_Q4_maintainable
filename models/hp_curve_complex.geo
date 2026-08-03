// 跑道形外边 + 4孔(两圆+三角+方)
// 各孔与外边界保留明确间隙，避免近切几何导致 Gmsh 2-D 网格失败。
lc = 0.25;
R = 2.0; L = 3.0; n = 20;

// 外边: 左半圆+右半圆+上下直线
For i In {0:n}
  // 修复: 左弧应扫左半圆 (π/2 → 3π/2, 经 (-3.5,0)), 与右弧对称构成
  // 体育场形。原 π/2 → -π/2 扫的是右半圆, 边界向右鼓进域内, 使
  // 中高度处域仅 x∈[0.5,3.5], 孔1/孔3 整体落到域外、孔4 左边缘
  // 穿出边界, 面网格因 1D 网格自交失败 ("No elements in surface 1")。
  ang = Pi/2 + Pi*i/n;
  Point(1+i) = {-L/2+R*Cos(ang), R*Sin(ang), 0, lc*0.7};
EndFor
For i In {0:n}
  ang = -Pi/2 + Pi*i/n;
  Point(100+i) = {L/2+R*Cos(ang), R*Sin(ang), 0, lc*0.7};
EndFor
Line(1) = {1+n, 100};
Line(2) = {100+n, 1};
Point(200) = {-L/2, 0, 0, lc*0.5};
For i In {0:n-1}
  Circle(300+i) = {1+i, 200, 1+i+1};
EndFor
Point(201) = {L/2, 0, 0, lc*0.5};
For i In {0:n-1}
  Circle(400+i) = {100+i, 201, 100+i+1};
EndFor
Curve Loop(1) = {300:319, 1, 400:419, 2};

// 孔1: 圆 R=0.7 @ (0, 0.5)
Point(500) = {0, 0.5, 0, lc*0.3};
For i In {0:19}
  ang = 2*Pi*i/20;
  Point(510+i) = {0.7*Cos(ang), 0.5+0.7*Sin(ang), 0, lc*0.3};
EndFor
For i In {0:19}
  Circle(600+i) = {510+i, 500, 510+((i+1)%20)};
EndFor
Curve Loop(2) = {600:619};

// 孔2: 圆 R=0.45 @ (2, -0.7)
Point(700) = {2, -0.7, 0, lc*0.3};
For i In {0:15}
  ang = 2*Pi*i/16;
  Point(710+i) = {2+0.45*Cos(ang), -0.7+0.45*Sin(ang), 0, lc*0.3};
EndFor
For i In {0:15}
  Circle(800+i) = {710+i, 700, 710+((i+1)%16)};
EndFor
Curve Loop(3) = {800:815};

// 孔3: 三角（远离左侧圆弧，避免近切）
Point(900) = {-2.0, -0.1, 0, lc*0.3};
Point(901) = {-2.55, -1.25, 0, lc*0.3};
Point(902) = {-1.35, -1.25, 0, lc*0.3};
Line(900) = {900, 901};
Line(901) = {901, 902};
Line(902) = {902, 900};
Curve Loop(4) = {900, 901, 902};

// 孔4: 方 0.6x0.6 @ (0.5, -1.2)
Point(1000) = {0.2, -1.5, 0, lc*0.25};
Point(1001) = {0.8, -1.5, 0, lc*0.25};
Point(1002) = {0.8, -0.9, 0, lc*0.25};
Point(1003) = {0.2, -0.9, 0, lc*0.25};
Line(1000) = {1000, 1001};
Line(1001) = {1001, 1002};
Line(1002) = {1002, 1003};
Line(1003) = {1003, 1000};
Curve Loop(5) = {1000, 1001, 1002, 1003};

Plane Surface(1) = {1, 2, 3, 4, 5};

Physical Curve("outer_racetrack", 101) = {300:319, 1, 400:419, 2};
Physical Curve("round_hole_1", 102) = {600:619};
Physical Curve("round_hole_2", 103) = {800:815};
Physical Curve("triangle_hole", 104) = {900, 901, 902};
Physical Curve("square_hole", 105) = {1000, 1001, 1002, 1003};
Physical Surface("domain", 201) = {1};

Mesh.SaveAll = 1;
Mesh.Format = 39;
Mesh 2;
Save "hp_curve_complex.msh";
