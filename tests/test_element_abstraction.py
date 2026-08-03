"""Architecture tests for topology-independent solver code."""
import unittest

import numpy as np

from fem2d import (
    ElementKernel,
    Mesh,
    assemble_sparse,
    compute_stresses,
    get_element_kernel,
    register_element,
)


class _TestQuadKernel(ElementKernel):
    """Minimal non-physical four-node kernel used only to test dispatch."""

    name = "TEST4"
    aliases = ("TST4",)
    nodes_per_element = 4
    local_edges = ((0, 1), (1, 2), (2, 3), (3, 0))

    def build_geometry(self, nodes, elements):
        coords = nodes[elements]
        x, y = coords[:, :, 0], coords[:, :, 1]
        signed = 0.5 * np.sum(
            x * np.roll(y, -1, axis=1)
            - y * np.roll(x, -1, axis=1), axis=1)
        return {
            "areas": np.abs(signed),
            "signed_areas": signed,
            "centroids": coords.mean(axis=1),
        }

    def stiffness_batch(self, mesh):
        return np.broadcast_to(
            np.eye(8), (mesh.n_elements, 8, 8)).copy()

    def compute_response(self, mesh, u_e):
        shape = (mesh.n_elements, 3)
        return np.zeros(shape), np.zeros(shape), np.zeros(mesh.n_elements)

    def jacobian_determinants(self, mesh):
        return mesh.signed_areas[:, None]

    def body_force_vector(self, mesh, eid, body_force):
        return np.zeros(8)

    def shape_values_at(self, coords, x, y, tol=1e-12):
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        if np.all(np.array([x, y]) >= lo - tol) and np.all(
                np.array([x, y]) <= hi + tol):
            return np.full(4, 0.25)
        return None

    def verify_mesh(self, mesh, verbose=True):
        return True


def _ensure_test_kernel():
    try:
        return get_element_kernel("TEST4")
    except ValueError:
        return register_element(_TestQuadKernel())


class ElementAbstractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_test_kernel()

    def test_cst_aliases_resolve_to_one_kernel(self):
        kernels = [
            get_element_kernel(name)
            for name in ("CST", "CPS3", "CPE3", "C2D3")
        ]
        self.assertTrue(all(kernel is kernels[0] for kernel in kernels))

    def test_unknown_type_fails_at_mesh_boundary(self):
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "Unsupported element type"):
            Mesh(nodes, np.array([[0, 1, 2]]), elem_type="UNKNOWN")

    def test_four_node_topology_and_dofs_are_generic(self):
        nodes = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ])
        mesh = Mesh(
            nodes, np.array([[0, 1, 2, 3]]), elem_type="TEST4")
        mesh.build_connectivity()
        np.testing.assert_array_equal(
            mesh.element_dofs[0], np.arange(8))
        self.assertEqual(
            set(mesh.boundary_edges),
            {(0, 1), (1, 2), (2, 3), (0, 3)})
        self.assertEqual(mesh.elem_neighbors, [[]])
        np.testing.assert_allclose(mesh.centroids, [[0.5, 0.5]])

    def test_assembly_and_stress_delegate_to_four_node_kernel(self):
        nodes = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
        ])
        mesh = Mesh(
            nodes, np.array([[0, 1, 2, 3]]), elem_type="TST4")
        np.testing.assert_allclose(assemble_sparse(mesh).toarray(), np.eye(8))
        stress, strain, vm = compute_stresses(mesh, np.zeros(8))
        self.assertEqual(stress.shape, (1, 3))
        self.assertEqual(strain.shape, (1, 3))
        self.assertEqual(vm.shape, (1,))


if __name__ == "__main__":
    unittest.main()


# ═══════════════════════════════════════════════════════════════
# 第六轮覆盖率补测: 单元验证路径 (verify_all_elements)
# ═══════════════════════════════════════════════════════════════

def test_elem_type_is_read_only_after_construction():
    """构造后修改 elem_type 必须拒绝 — 曾只改字符串不改 kernel,
    显示 CPS4I 实际仍按原单元计算 (外部审查复现)."""
    import pytest
    from fem2d import Mesh
    nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    elems = np.array([[0, 1, 2, 3]])
    m = Mesh(nodes=nodes, elements=elems, elem_type="CPS4")
    assert m.elem_type == "CPS4"
    with pytest.raises(AttributeError, match="read-only"):
        m.elem_type = "CPS4I"
    # 即使私有字段被外部直接破坏 (仿真分叉), 求解前状态校验必须拒绝
    m._elem_type = "CPS4I"
    with pytest.raises(ValueError, match="不一致"):
        m.validate_state()


def test_verify_all_elements_pass_regular_mesh():
    """verify_all_elements 对合法网格必须全部通过 (完备性 + 刚体模态).

    曾无测试触发 — cst._verify_element / q4r.verify_mesh 覆盖率 0
    (第六轮覆盖率分析, 2026-08-03).
    """
    from fem2d import Mesh, verify_all_elements
    for elem_type, npe in (("CPS3", 3), ("CPS4", 4), ("CPS4R", 4),
                           ("CPS4I", 4)):
        if npe == 3:
            nodes = np.array([[0., 0.], [1., 0.], [0., 1.]])
            elems = np.array([[0, 1, 2]])
        else:
            nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
            elems = np.array([[0, 1, 2, 3]])
        m = Mesh(nodes=nodes, elements=elems,
                 E=210e9, nu=0.3, thickness=0.01, elem_type=elem_type)
        ok = verify_all_elements(m, verbose=False)
        assert ok, f"{elem_type} 合法网格验证失败"


def test_verify_all_elements_detects_degenerate():
    """verify_all_elements 对退化 (共线) CST 单元必须报告失败."""
    from fem2d import Mesh, verify_all_elements
    m = Mesh(nodes=np.array([[0., 0.], [1., 0.], [2., 0.]]),
             elements=np.array([[0, 1, 2]]),
             E=210e9, nu=0.3, thickness=0.01, elem_type="CPS3")
    ok = verify_all_elements(m, verbose=False)
    assert not ok, "共线退化单元不应通过验证"


def test_cst_verify_element_math():
    """CST 单单元验证: 完备性误差与刚体模态误差必须 < 1e-12."""
    from fem2d.element.cst import _verify_element
    r = _verify_element(0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    assert r["all_ok"], f"标准 CST 验证失败: {r}"
    assert r["completeness_err"] < 1e-12
    assert r["rigid_body_err"] < 1e-12
    assert abs(r["area"] - 0.5) < 1e-12
