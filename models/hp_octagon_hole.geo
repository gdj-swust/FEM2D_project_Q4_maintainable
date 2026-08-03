// 正八边形外边 + 正八边形内孔
lc = 0.12;

// 外边: 正八边形 R=3
For i In {0:7}
  ang = 2*Pi*i/8;
  Point(1+i) = {3*Cos(ang), 3*Sin(ang), 0, lc};
EndFor
For i In {0:7}
  Line(10+i) = {1+i, 1+((i+1)%8)};
EndFor
Curve Loop(100) = {10:17};

// 内孔: 正八边形 R=1.2, 旋转 22.5° 避免与外边平行
For i In {0:7}
  ang = 2*Pi*i/8 + Pi/8;
  Point(50+i) = {1.2*Cos(ang), 1.2*Sin(ang), 0, lc*0.5};
EndFor
For i In {0:7}
  Line(60+i) = {50+i, 50+((i+1)%8)};
EndFor
Curve Loop(101) = {60:67};

Plane Surface(1) = {100, 101};
Recombine Surface{1};

Physical Curve("outer", 200) = {10:17};
Physical Curve("inner_octagon", 201) = {60:67};

Mesh.SaveAll = 1;
Mesh.Format = 39;
Mesh 2;
