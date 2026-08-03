# Boundary recognition upgrade — 2026-07-29

This revision strengthens the joint Gmsh-CAD / mesh-topology boundary contract
without changing the CST or Q4 element formulations.

## Changes

- A `RegionRegistry` extracted from the Gmsh Python API now explicitly records
  that its active CAD boundary inventory is complete.
- CAD construction surfaces with no active 2-D displacement elements are
  excluded from that inventory.
- Every external mesh edge in a complete registry must have exactly one
  external CAD curve owner.
- A CAD curve bounding one active surface must be external in the mesh; a
  curve shared by two surfaces must remain internal.
- Empty, missing, overlapping and non-manifold CAD curve mappings are fatal,
  even when `--strict-boundary` is not requested.
- Physical Curves report CAD entities that generated no usable line mesh
  instead of accepting the mapped subset silently.
- Numeric-only, whitespace/control-character, delimiter-conflicting and
  case-fold-colliding Physical Curve names are diagnosed before selection.
- Boundary chains reject self-loop edges, branches and incomplete traversal.
- Chain reconstruction now uses a deterministic adjacency walk rather than
  repeated global edge scans.
- Topology context must exist and agree on loop ID, depth and outer/hole role
  for every edge of a semantic segment.

## Verification

- 162 function tests passed.
- 4 executable/API integration modules were skipped because Gmsh is not
  installed in the verification environment.
- A 12,000-edge complete CAD boundary passed exact edge-coverage validation.
- A 120,000-edge chain reconstructed in about 0.6 seconds.
- 50,000 disconnected two-node components reconstructed in about 0.3 seconds.
- `models/hp_curved.inp` passed strict Physical Curve coverage:
  6/6 names and 618/618 boundary edges.

The remaining geometric limitation is intentional: CST/Q4 are first-order
straight-edge elements, so curve membership is exact but length, normal and
pressure integration remain discrete mesh approximations.
