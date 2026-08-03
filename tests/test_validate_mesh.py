"""网格校验器测试 — validate_mesh() 应正确拦截各种坏网格."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from fem2d.preprocess import MeshValidationError, validate_mesh


def _good_mesh():
    """2×2 三角形网格: 矩形拆成 4 个三角."""
    nodes = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1], [2, 1],
    ], dtype=float)
    elems = np.array([
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
    ], dtype=int)
    return nodes, elems


def test_good_mesh_passes():
    nodes, elems = _good_mesh()
    r = validate_mesh(nodes, elems)
    assert r["ok"], f"Good mesh should pass, got: {r['errors']}"


def test_negative_index_rejected():
    nodes, elems = _good_mesh()
    elems[0, 0] = -1
    try:
        validate_mesh(nodes, elems)
        assert False, "Should have raised MeshValidationError"
    except MeshValidationError:
        pass


def test_out_of_range_rejected():
    nodes, elems = _good_mesh()
    elems[1, 2] = 999
    try:
        validate_mesh(nodes, elems)
        assert False, "Should have raised"
    except MeshValidationError:
        pass


def test_duplicate_elements_detected():
    nodes, elems = _good_mesh()
    elems[2] = elems[0].copy()  # duplicate of element 0
    r = validate_mesh(nodes, elems)
    assert not r["ok"]
    assert r["stats"]["duplicate_elems"] >= 1


def test_zero_area_element_detected():
    """三个共线节点 → 退化三角."""
    nodes = np.array([
        [0, 0], [0.5, 0], [1, 0],   # 共线!
        [0, 1], [1, 1],
    ], dtype=float)
    elems = np.array([[0, 1, 2]], dtype=int)
    r = validate_mesh(nodes, elems)
    assert not r["ok"]
    assert r["stats"]["degenerate_elems"] >= 1


def test_negative_area_element_detected():
    """CW 三角形 → 有向面积 < 0"""
    nodes = np.array([
        [0, 0], [1, 0],
        [0, 1],
    ], dtype=float)
    elems = np.array([[0, 2, 1]], dtype=int)  # CW
    r = validate_mesh(nodes, elems)
    assert not r["ok"]
    assert r["stats"]["degenerate_elems"] >= 1


def test_zero_length_edge_detected():
    nodes, elems = _good_mesh()
    nodes[1] = nodes[0].copy()  # node 1 == node 0
    r = validate_mesh(nodes, elems)
    assert not r["ok"]
    assert r["stats"]["zero_edges"] >= 1


def test_orphan_node_detected():
    nodes = np.array([
        [0, 0], [1, 0], [2, 0],
        [0, 1], [1, 1],
        [99, 99],  # 孤立节点
    ], dtype=float)
    elems = np.array([[0, 1, 4], [0, 4, 3]], dtype=int)
    r = validate_mesh(nodes, elems)
    assert r["stats"]["orphan_nodes"] >= 1


def test_non_manifold_edge_detected():
    """一个边被 3 个单元共享."""
    nodes = np.array([
        [0, 0], [1, 0], [0.5, 0.8],
        [0.5, -0.8],
    ], dtype=float)
    # 三条边都共享 (0,1) 边 → 非流形
    elems = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 1],  # 第三个单元共享 (0,1)
    ], dtype=int)
    r = validate_mesh(nodes, elems)
    assert not r["ok"]
    assert r["stats"]["non_manifold_edges"] >= 1
