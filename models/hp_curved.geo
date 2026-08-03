// 椭圆外边 + 曲线孔(椭圆/半圆) + 直线孔(三角/方)
lc = 0.15;
n_outer = 40;
a_outer = 5.0;  b_outer = 3.0;

// 外边: 椭圆 (多边形近似, 40段直线)
For i In {0:n_outer-1}
  ang = 2*Pi*i/n_outer;
  Point(100+i) = {a_outer*Cos(ang), b_outer*Sin(ang), 0, lc};
EndFor
For i In {0:n_outer-1}
  Line(200+i) = {100+i, 100+((i+1)%n_outer)};
EndFor
Curve Loop(1) = {200:239};

// 孔1: 椭圆 a=1.2 b=0.5 @ (2, 1.2), n=28, 直线近似
For i In {0:27}
  ang = 2*Pi*i/28;
  Point(400+i) = {2+1.2*Cos(ang), 1.2+0.5*Sin(ang), 0, lc*0.4};
EndFor
For i In {0:27}
  Line(500+i) = {400+i, 400+((i+1)%28)};
EndFor
Curve Loop(2) = {500:527};

// 孔2: 半圆槽 R=0.7 @ (-2, 1.5) — 15段半圆弧+直径
Point(600) = {-2, 1.5, 0, lc*0.3};
For i In {0:15}
  ang = Pi - Pi*i/15;
  Point(610+i) = {-2+0.7*Cos(ang), 1.5+0.7*Sin(ang), 0, lc*0.3};
EndFor
For i In {0:14}
  Circle(1300+i) = {610+i, 600, 610+i+1};
EndFor
Line(1315) = {610+15, 610};
Curve Loop(3) = {1300:1314, 1315};

// 孔3: 三角 @ (-3, -1)
Point(800) = {-3, -0.3, 0, lc*0.3};
Point(801) = {-3.8, -1.5, 0, lc*0.3};
Point(802) = {-2.2, -1.5, 0, lc*0.3};
Line(800) = {800, 801};
Line(801) = {801, 802};
Line(802) = {802, 800};
Curve Loop(4) = {800, 801, 802};

// 孔4: 方 0.7x0.7 @ (0.5, -1.5)
Point(900) = {0.15, -1.85, 0, lc*0.3};
Point(901) = {0.85, -1.85, 0, lc*0.3};
Point(902) = {0.85, -1.15, 0, lc*0.3};
Point(903) = {0.15, -1.15, 0, lc*0.3};
Line(900) = {900, 901};
Line(901) = {901, 902};
Line(902) = {902, 903};
Line(903) = {903, 900};
Curve Loop(5) = {900, 901, 902, 903};

// 孔5: 圆 R=0.35 @ (3.5, -1)
Point(1000) = {3.5, -1, 0, lc*0.25};
For i In {0:15}
  ang = 2*Pi*i/16;
  Point(1010+i) = {3.5+0.35*Cos(ang), -1+0.35*Sin(ang), 0, lc*0.25};
EndFor
For i In {0:15}
  Circle(1100+i) = {1010+i, 1000, 1010+((i+1)%16)};
EndFor
Curve Loop(6) = {1100:1115};

Plane Surface(1) = {1, 2, 3, 4, 5, 6};

Physical Curve("外边", 101) = {200:239};
Physical Curve("孔1_椭圆", 102) = {500:527};
Physical Curve("孔2_半圆槽", 103) = {1300:1314, 1315};
Physical Curve("孔3_三角", 104) = {800, 801, 802};
Physical Curve("孔4_方", 105) = {900, 901, 902, 903};
Physical Curve("孔5_圆", 106) = {1100:1115};

Mesh.SaveAll = 1;
Mesh.Format = 39;
Mesh 2;
Save "hp_curved.msh";
