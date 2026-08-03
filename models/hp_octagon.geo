// 正八边形 + 圆孔
lc = 0.15;
n_hole = 20;
R_hole = 0.8;

// 外边: 正八边形 R=3
For i In {0:7}
  ang = 2*Pi*i/8;
  Point(1+i) = {3*Cos(ang), 3*Sin(ang), 0, lc};
EndFor
For i In {0:7}
  Line(10+i) = {1+i, 1+((i+1)%8)};
EndFor
Curve Loop(100) = {10:17};

// 内孔: 圆 R=0.8 @ (0,0)
Point(50) = {0, 0, 0, lc*0.3};
For i In {0:n_hole-1}
  ang = 2*Pi*i/n_hole;
  Point(60+i) = {R_hole*Cos(ang), R_hole*Sin(ang), 0, lc*0.3};
EndFor
For i In {0:n_hole-1}
  Circle(200+i) = {60+i, 50, 60+((i+1)%n_hole)};
EndFor
Curve Loop(101) = {200:219};

Plane Surface(1) = {100, 101};

Physical Curve("octagon", 102) = {10:17};
Physical Curve("hole_round", 103) = {200:219};

Mesh.SaveAll = 1;
Mesh.Format = 39;
Mesh 2;
