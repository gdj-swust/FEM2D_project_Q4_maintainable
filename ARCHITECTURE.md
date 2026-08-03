# FEM2D element architecture

`Mesh` selects one homogeneous `ElementKernel` from
`fem2d/element/registry.py`.
Assembly, solving, loads, point location, stress recovery and error estimation
consume the kernel contract and do not branch on CST/Q4 element names.

## Kernel contract

Each kernel owns:

- `nodes_per_element`, `dofs_per_element`, and `local_edges`
- geometry caches and Jacobian validation samples
- batched and scalar element stiffness
- representative and integration-point stress/strain response
- consistent body-force integration
- inverse mapping / shape values for physical points
- recovery quadrature and element verification

Built-in registrations:

| Kernel | Aliases | Integration |
|---|---|---|
| `CSTElement` | `CST`, `CPS3`, `CPE3`, `C2D3` | exact constant-strain formulas |
| `Q4Element` | `Q4`, `CPS4`, `CPE4` | 2×2 Gauss full integration |
| `Q4RElement` | `Q4R`, `CPS4R`, `CPE4R` | one-point + affine-projector hourglass stabilization |
| `Q4IElement` | `Q4I`, `CPS4I`, `CPE4I` | incompatible modes (QM6) with static condensation |

`material.py` contains topology-independent constitutive and von Mises
operations. The `element/` package registers the kernels; `element/cst.py`
also re-exports the legacy CST public formula API (B-matrix, shape values,
verification) and the common material functions for compatibility.

## Adding another element

1. Implement a new `ElementKernel` module.
2. Register its instance and aliases.
3. Add input-code parsing and element-specific verification tests.

No changes should be needed in `mesh.py`, `assembly.py`, `solver.py`,
`loads_core.py`, or the generic L2/Z2 recovery pipeline.

Optional hooks: `degeneracy_measure(mesh)` returns a unitless shape-
degeneracy score (area / longest-side²) so the Jacobian check can flag
slender elements regardless of absolute size; `response_at_quadrature`
must be overridden (or `compute_response`) — the defaults call each other
and a kernel overriding neither gets a clear `NotImplementedError`.

The current mesh container is deliberately homogeneous. Mixed element blocks
would require a higher-level block mesh/assembly abstraction rather than
adding scattered type conditionals.

## Gmsh topology boundary

`fem2d/gmsh_adapter.py` is the primary `.geo` adapter. Gmsh parses its own
command language and builds the BRep; FEM2D never tries to reproduce Boolean
operations, loops, CAD tags or spline semantics. Before the Gmsh model is
discarded, the adapter creates a `RegionRegistry`:

| Gmsh dimension | Registry record | FEM target |
|---|---|---|
| 0, Physical Point | `PointRegion` | node IDs |
| 1, Physical Curve | `CurveRegion` | boundary edge pairs and node chains |
| 1, every CAD boundary entity | `CadCurveRegion` | hard segment partition |
| 2, Physical Surface | `SurfaceRegion` | element IDs and integrated area |

The registry is validated against the final `Mesh` before any BC or load is
assembled. Curve regions may overlap semantically, while the integration
segments still cover every boundary edge exactly once. This lets `fixed`,
`traction`, `hole_1` and `all_holes` coexist without duplicating force.
`CadCurveRegion` is independent of Physical Groups: every curve on an active
surface is retained with its entity tag/type and surface occurrence.  The
joint key is therefore `(loop, CAD entity, Physical membership set)`.
The adapter marks this registry as complete only after excluding CAD
construction surfaces with no active displacement elements.

For a complete registry, CAD and mesh topology form a hard contract:

- every external mesh edge has exactly one external CAD curve owner;
- a CAD curve bounding one active surface cannot be internal in the mesh;
- a CAD curve shared by two surfaces cannot appear as an exposed edge;
- empty, missing, overlapping, or non-manifold CAD curves are fatal.

These contradictions fail even without `--strict-boundary`; automatically
falling back to geometric classification would destroy the `.geo` semantics.

The adapter also:

- passes quad recombination options without editing the source `.geo`;
- accepts only homogeneous first-order CST or Q4 displacement elements;
- requires Q4 when quad mode is requested;
- writes the archival native `.msh` atomically; and
- exposes original Gmsh node/element tags only through explicit maps.

`scripts/gmsh_runner.py` is the primary CLI generation path (subprocess gmsh executable with stripped-copy, topology validation and atomic publish); the Python API path (`gmsh_adapter.generate_from_geo`) is used for semantic recovery and tests. The API path cannot preserve the full point/surface
entity graph; `--require-physical-groups` prevents an unnoticed geometry-only
fallback.
`--strict-boundary` additionally rejects internal/missing Physical Curves and
conflicting CAD provenance.

The mesh importer uses a dict-compatible edge-label map that also retains
unmapped T3D2/C2D2 records. Consequently orphan/undefined curve nodes,
interior edges and partially missing groups are diagnosed by the same
`BoundaryDiagnostics` contract as the API path. Coverage is reported as
mapped versus declared names, not merely as “at least one group survived”.

## Boundary subsystem

The boundary package separates these concerns (mirrors the subpackage
docstring):

| Module | Responsibility |
|---|---|
| `boundary/topology.py` | mesh edges → validated/nested/oriented loops |
| `boundary/geometry.py` | curvature and primitive classifiers |
| `boundary/predicates.py` | adaptive-precision orientation predicate |
| `boundary/physical_mapping.py` | Gmsh Physical Group semantic mapping |
| `boundary/registry_mapping.py` | exact Gmsh CAD/Physical registry mapping |
| `boundary/segment_builder.py` | ordered chains → public segment dictionaries |
| `boundary/conic_merge.py` | conservative CAD conic presentation merge |
| `boundary/selectors.py` | exact CLI boundary-name resolution |
| `boundary/naming.py` | public orchestration and reporting facade |
| `boundary/model.py` | structured boundary diagnostics |
| `boundary/validation.py` | per-name/group consistency checks |
| `boundary/segment_utils.py` | shared segment metadata helpers |

`topology.detect()` never branches on CST/Q4. It consumes
`mesh.boundary_edges`, whose local topology comes from the active element
kernel.

Whole-loop conic fitting is not allowed to bypass segmentation blindly.
Turning-angle analysis first finds long collinear runs and their tangent
endpoints. A rounded rectangle is consequently retained as line/arc
primitives, while a polygonized circle can still be recovered when its global
conic residual is substantially smaller. Chord junctions are used for a second
fit so subdivision nodes do not bias the radius.

Gmsh CAD entity transitions are hard segmentation boundaries before Physical
membership or geometric classification. This keeps an arbitrary spline,
parabola, or variable-curvature CAD edge intact and prevents adjacent CAD
entities from being glued together. The geometry-only fallback lacks
that entity graph, so it conservatively subdivides semantic chains only at
structural corners or sustained straight runs. Smooth curvature extrema and
inflection points are not segmentation events.

Physical Curve selectors are exact and case-insensitive by default. Implicit
substring expansion is prohibited because `load` must not select both
`load_a` and `load_b`. An explicit `~query` compatibility search is accepted
only when it resolves to one semantic name or label. Numeric-only names,
control characters, CLI delimiters and case-fold collisions are rejected
before a BC or load selector is evaluated. A Physical Curve that contains only
some meshed CAD entities is reported as partially unmapped.

Primitive labels are conservative. Open circular arcs use a normalized
least-squares fit with a strict all-point radial residual, and partial ellipse
fits have an additional dimensionless residual gate. A failed primitive model
becomes a generic ordered `curve` carrying length, curvature statistics, and
inflection count instead of being forced into the nearest circle or ellipse.

Loop containment uses adaptive `orient2d` proper intersections with a
scale-aware ray selected in general position. Duplicate vertices and exact
boundary points have explicit behavior, and a half-open crossing rule is the
degenerate fallback. Closed output follows the Gmsh convention: outer loops
CCW and holes CW.

Before nesting, every boundary node must have degree two. A sweep-line
candidate filter plus robust orientation predicates rejects self-crossings,
non-adjacent touches, collinear overlaps, and contacts between distinct loops.
`loop_id`, `loop_depth`, and `is_outer` are explicit metadata; presentation
labels never drive algorithms.

Zero/numerically degenerate loop area is rejected using a translated
shoelace sum, preserving both microscopic models and small shapes at large
global coordinates. Physical names that collide case-insensitively or contain
CLI grammar delimiters are rejected in strict semantic mode.

Distributed `:l`/`:p` tractions use normalized cumulative polyline arc length.
The CLI first joins all selected CAD partitions into connected chains, then
creates an O(1) local profile callable for each mesh edge. A linear profile on
a closed chain is rejected because its seam value would be non-unique.
Connected edge chains are reconstructed with an adjacency walk and
deterministic O(E log E) ordering. Every component edge must be consumed
exactly once; self-loops and branches fail.
