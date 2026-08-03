// 五角星外边界(10尖角) + 5个圆孔在尖角处
lc = 0.08;
Ro = 3;  Ri = 1.2;  n_star = 5;

// 星形外边
For i In {0:n_star-1}
  ang = 2*Pi*i/n_star - Pi/2;
  Point(1+i*2) = {Ro*Cos(ang), Ro*Sin(ang), 0, lc};
  Point(2+i*2) = {Ri*Cos(ang+Pi/n_star), Ri*Sin(ang+Pi/n_star), 0, lc};
EndFor
For i In {0:2*n_star-1}
  Line(100+i) = {1+i, 1+((i+1)%(2*n_star))};
EndFor
Curve Loop(200) = {100:109};

// 5个圆孔 R=0.3 在尖角处
For k In {0:4}
  ang_k = 2*Pi*k/5 - Pi/2;
  cx = 1.9*Cos(ang_k);
  cy = 1.9*Sin(ang_k);
  Point(300+k) = {cx, cy, 0, lc*0.4};
  For i In {0:15}
    ang = 2*Pi*i/16;
    Point(400+k*20+i) = {cx+0.3*Cos(ang), cy+0.3*Sin(ang), 0, lc*0.4};
  EndFor
  For i In {0:15}
    Circle(500+k*20+i) = {400+k*20+i, 300+k, 400+k*20+((i+1)%16)};
  EndFor
  Curve Loop(600+k) = {500+k*20:500+k*20+15};
EndFor

Plane Surface(1) = {200, 600, 601, 602, 603, 604};
Mesh.SaveAll = 1;
Mesh.Format = 39;
Mesh 2;
Save "hp_star.msh";
