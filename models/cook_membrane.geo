// Cook's Membrane — 经典 FEM 基准测试 (Cook 1974)
// 梯形板: 左边固定, 右端均匀剪切力 τ=1.0
// 广泛用于验证单元在弯曲主导问题中的性能
// 参考: Cook, R.D. (1974) "Improved 2D finite element"

lc = 0.8;

// ── 梯形节点: 48×44 → 60×44 (左边高44, 右边高60) ──
// 顶点坐标
L = 48.0;    // 水平跨度
H_left  = 44.0;  // 左边高度
H_right = 60.0;  // 右边高度

// 左下
Point(1) = {  0,          0,         0, lc};
// 左上
Point(2) = {  0,          H_left,    0, lc};
// 右上
Point(3) = {  L,          H_right,   0, lc};
// 右下
Point(4) = {  L,          0,         0, lc};

// 四条外边
Line(1) = {1, 2};  // 左边 — 固定端
Line(2) = {2, 3};  // 顶边
Line(3) = {3, 4};  // 右边 — 受剪
Line(4) = {4, 1};  // 底边

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// ── Physical Groups ──
Mesh.SaveAll = 1;
Physical Surface("domain", 200) = {1};
Physical Curve("左_固定",  101) = {1};
Physical Curve("右_剪力",  102) = {3};

// @FEM:fix=左_固定
// @FEM:traction=右_剪力,0,1.0

Mesh.Format = 39;
Mesh.SaveGroupsOfNodes = 1;
Mesh.SaveGroupsOfElements = 1;
Mesh 2;
Save "cook_membrane.msh";
