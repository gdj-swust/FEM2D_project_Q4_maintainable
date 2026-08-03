# Q4 example models

These small files are intended for smoke tests and as input-format examples.

```bash
# CPS4 / plane stress — generate a new CPS4 mesh with Gmsh recombination
python run.py models/plate_q4.geo \
  --quad --fix left --traction right:1e6,0 --no-plot

# CPE4 / plane strain — 必须显式 --quad (生成四边形) + --plane strain
# (.geo/.msh 不含 CPE/CPS 单元码语义, 平面态由 --plane 声明)
python run.py models/plate_q4.geo --quad --plane strain \
  --fix left --traction right:1e6,0 --no-plot
```

`--quad` only applies while generating a mesh from `.geo` or `.txt`; it does
not reinterpret an existing mesh. FEM2D validates the Gmsh output and
rejects triangle/quad mixtures or incomplete recombination before solving.

Simple four-sided surfaces normally recombine directly. Models with holes or
complex boundaries may need surface partitioning or a transfinite mesh in
Gmsh. The solver intentionally keeps one homogeneous displacement-element
topology per analysis.

When a `.geo` file is the source, define semantic Physical Groups and use
their names in the FEM command:

```c
Physical Curve("fixed") = {4};
Physical Curve("traction") = {2};
Physical Surface("domain") = {1};
```

```bash
python run.py model.geo \
  --fix fixed --traction traction:1e6,0 --no-plot
```

The Gmsh API path maps these groups to generated nodes, boundary edges and
elements before publishing the archival `.msh`; arbitrary CAD curves therefore
do not need to be rediscovered from coordinates.
