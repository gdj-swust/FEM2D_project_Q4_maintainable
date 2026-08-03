// 不规则六边形 + 椭圆孔 + 三角孔 + 圆孔
lc = 0.12;

// 外边: 不规则六边形 (非90°)
Point(1) = {0, 0, 0, lc};
Point(2) = {8, 0, 0, lc};
Point(3) = {10, 3, 0, lc};
Point(4) = {7, 7, 0, lc};
Point(5) = {2, 8, 0, lc};
Point(6) = {-1, 4, 0, lc};
Line(1) = {1, 2};  Line(2) = {2, 3};  Line(3) = {3, 4};
Line(4) = {4, 5};  Line(5) = {5, 6};  Line(6) = {6, 1};
Curve Loop(100) = {1, 2, 3, 4, 5, 6};

// 椭圆孔 a=0.8 b=0.4 @ (3, 3), n=24
Point(10) = {3, 3, 0, lc*0.4};
For i In {0:23}
  ang = 2*Pi*i/24;
  Point(100+i) = {3+0.8*Cos(ang), 3+0.4*Sin(ang), 0, lc*0.4};
EndFor
For i In {0:23}
  Circle(200+i) = {100+i, 10, 100+((i+1)%24)};
EndFor
Curve Loop(200) = {200:223};

// 三角孔  @ (6, 2)
Point(300) = {6, 1.5, 0, lc*0.3};
Point(301) = {5.3, 2.8, 0, lc*0.3};
Point(302) = {6.7, 2.8, 0, lc*0.3};
Line(300) = {300, 301};
Line(301) = {301, 302};
Line(302) = {302, 300};
Curve Loop(300) = {300, 301, 302};

// 圆孔 R=0.5 @ (5, 5.5), n=20
Point(20) = {5, 5.5, 0, lc*0.3};
For i In {0:19}
  ang = 2*Pi*i/20;
  Point(400+i) = {5+0.5*Cos(ang), 5.5+0.5*Sin(ang), 0, lc*0.3};
EndFor
For i In {0:19}
  Circle(500+i) = {400+i, 20, 400+((i+1)%20)};
EndFor
Curve Loop(400) = {500:519};

Plane Surface(1) = {100, 200, 300, 400};
Mesh.SaveAll = 1;
Mesh.Format = 39;
Mesh 2;
Save "hp_hexagon.msh";
