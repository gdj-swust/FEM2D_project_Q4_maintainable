"""SPR linear consistency test"""
import numpy as np

from fem2d import Mesh
from fem2d.spr import spr_recovery


def _make_test_mesh():
    """Generate a regular 4x4 quad mesh (split to triangles)."""
    L, H = 1.0, 1.0
    nx = ny = 4
    nodes = np.array([[i*L/nx, j*H/ny] for j in range(ny+1) for i in range(nx+1)], dtype=float)
    elems = []
    for j in range(ny):
        for i in range(nx):
            a = j*(nx+1)+i; b = a+1; c = a+nx+1; d = c+1
            elems.extend([[a,b,c],[b,d,c]])
    return Mesh(nodes=nodes, elements=np.array(elems, dtype=int),
                E=210e9, nu=0.3, thickness=0.01)


def test_spr_recovers_linear_field_exactly():
    """SPR must recover a strictly linear 3-component stress field to ~machine precision."""
    mesh = _make_test_mesh()
    mesh.build_connectivity()

    # Strictly linear stress: σ(x,y) = a + bx + cy
    cx, cy = mesh.centroids[:, 0], mesh.centroids[:, 1]
    s_elem = np.column_stack([
        2.0 + 3.0*cx - 4.0*cy,       # σ_x
        1.0 + 0.5*cx + 2.0*cy,       # σ_y
        0.1*cx - 0.3*cy,             # τ_xy
    ])

    s_node = spr_recovery(mesh, s_elem)

    max_err = 0.0
    for nid, (x, y) in enumerate(mesh.nodes):
        exact = np.array([2.0+3.0*x-4.0*y, 1.0+0.5*x+2.0*y, 0.1*x-0.3*y])
        max_err = max(max_err, np.max(np.abs(s_node[nid] - exact)))

    assert max_err < 1e-12, f"SPR linear recovery error: {max_err:.2e}"
