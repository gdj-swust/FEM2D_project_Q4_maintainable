"""Traction jump consistency tests"""
import numpy as np

from fem2d import Mesh
from fem2d.visualize import compute_traction_jumps


def _make_test_mesh():
    L, H = 1.0, 1.0
    nx = ny = 3
    nodes = np.array([[i*L/nx, j*H/ny] for j in range(ny+1) for i in range(nx+1)], dtype=float)
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i; b = a+1; c = a+nx+1; d = c+1
            elems.extend([[a,b,c],[b,d,c]])
    return Mesh(nodes=nodes, elements=np.array(elems, dtype=int),
                E=210e9, nu=0.3, thickness=0.01)


def test_constant_stress_has_zero_jump():
    """All elements have identical stress → all traction jumps must be zero."""
    mesh = _make_test_mesh()
    mesh.build_connectivity()
    s = np.ones((mesh.n_elements, 3)) * 5e6
    jumps = compute_traction_jumps(mesh, s)
    max_jump = max(j['jump_abs'] for j in jumps) if jumps else 0.0
    assert max_jump < 1e-12, f"Constant stress should give zero jump, got {max_jump:.2e}"


def test_traction_jump_rotation_invariance():
    """Rotating both mesh and stress tensor must preserve traction jumps."""
    mesh = _make_test_mesh()
    # Apply non-uniform stress
    mesh.build_connectivity()
    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]
    s = np.column_stack([3*cx - 2*cy, cx + 4*cy, 0.5*cx - 0.5*cy])

    jumps_orig = compute_traction_jumps(mesh, s)
    orig_vals = np.array([j['jump_abs'] for j in jumps_orig])

    # Rotate mesh 45° CCW
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    nodes_rot = mesh.nodes @ R.T
    mesh_rot = Mesh(nodes=nodes_rot, elements=mesh.elements.copy(),
                    E=210e9, nu=0.3, thickness=0.01)
    mesh_rot.build_connectivity()

    # Rotate stress tensors: σ' = R σ Rᵀ
    s_rot = np.zeros_like(s)
    for eid in range(mesh.n_elements):
        sigma = np.array([[s[eid,0], s[eid,2]],
                          [s[eid,2], s[eid,1]]])
        sigma_rot = R @ sigma @ R.T
        s_rot[eid, 0] = sigma_rot[0, 0]
        s_rot[eid, 1] = sigma_rot[1, 1]
        s_rot[eid, 2] = sigma_rot[0, 1]

    jumps_rot = compute_traction_jumps(mesh_rot, s_rot)
    rot_vals = np.array([j['jump_abs'] for j in jumps_rot])

    diff = np.max(np.abs(orig_vals - rot_vals))
    assert diff < 1e-12, f"Rotation invariance violated: max diff={diff:.2e}"