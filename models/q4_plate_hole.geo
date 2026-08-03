lc = 0.08;
Point(1) = {-1.0, -1.0, 0, lc};
Point(2) = { 1.0, -1.0, 0, lc};
Point(3) = { 1.0,  1.0, 0, lc};
Point(4) = {-1.0,  1.0, 0, lc};
Line(1) = {1, 2};  Line(2) = {2, 3};
Line(3) = {3, 4};  Line(4) = {4, 1};
n = 24;
For i In {0:n-1}
  ang = 2*Pi*i/n;
  Point(10 + i) = {0.3*Cos(ang), 0.3*Sin(ang), 0, lc*0.1};
EndFor
Point(99) = {0, 0, 0, lc*0.1};
For i In {0:n-1}
  Circle(100 + i) = {10+i, 99, 10+((i+1)%n)};
EndFor
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {100:100+n-1};
Plane Surface(1) = {1, 2};
Recombine Surface{1};
Physical Surface("domain", 200) = {1};
Physical Curve("left",  101) = {4};
Physical Curve("right", 102) = {2};
Mesh.Format = 39; Mesh.SaveAll = 1; Mesh 2;
Save "q4_plate_hole.msh";
