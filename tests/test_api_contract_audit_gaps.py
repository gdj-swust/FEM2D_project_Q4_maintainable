"""复查轮审计 2 — 测试盲区补齐 (行为锁定测试).

契约表 ✅ 项中此前无判别性测试锁定的误用路径:
check_jacobian (0 测试) / estimate_condition 直接调用 (仅 monkeypatch 间接) /
nodes_on_edge 误用 / replace_nodes/replace_elements 误用。
"""
import numpy as np
import pytest

from fem2d.mesh import Mesh
from fem2d.solver import estimate_condition

NODES = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])


def _mesh():
    return Mesh(NODES, ELEMS, E=1e6, nu=0.3, thickness=1.0)


# ── check_jacobian: 契约 A3 (此前 0 测试) ──

def test_check_jacobian_ok_mesh():
    ok, bad = _mesh().check_jacobian()
    assert ok is True
    assert bad == []


def test_check_jacobian_cw_element_detected():
    # CW (顺时针) 单元有向面积 < 0 → 必须被判 bad
    m = Mesh(NODES, np.array([[0, 2, 1]]), E=1e6, nu=0.3, thickness=1.0)
    ok, bad = m.check_jacobian()
    assert ok is False
    assert bad == [0]


def test_check_jacobian_degenerate_detected():
    # 共线三点 → 零面积
    m = Mesh(np.array([[0., 0.], [1., 0.], [2., 0.]]),
             np.array([[0, 1, 2]]), E=1e6, nu=0.3, thickness=1.0)
    ok, bad = m.check_jacobian()
    assert ok is False


# ── estimate_condition: 直接调用契约 (B 组, 此前仅 monkeypatch 间接) ──

def test_estimate_condition_dense_ok():
    info = estimate_condition(np.diag([1.0, 2.0, 4.0, 8.0]), method="dense")
    assert info["condition_number"] == pytest.approx(8.0)
    assert info["status"] == "GOOD"
    assert info["lambda_min"] > 0


def test_estimate_condition_bogus_method_rejected():
    # 复查轮审计发现: 非法 method 曾静默降级 sparse 路径并成功返回
    # GOOD (拼错无察觉) — 必须响亮失败
    with pytest.raises(ValueError, match="auto/dense/sparse"):
        estimate_condition(np.diag([1.0, 2.0, 3.0]), method="bogus")
    with pytest.raises(ValueError, match="auto/dense/sparse"):
        estimate_condition(np.diag([1.0, 2.0, 3.0]), method=None)


def test_estimate_condition_singular_returns_singular():
    info = estimate_condition(np.diag([1.0, 0.0, 0.0]), method="dense")
    assert info["condition_number"] == float("inf")
    assert info["status"] == "CRITICAL"


def test_estimate_condition_non_square_ndarray_rejected():
    # 非方阵 ndarray → 前置 ValueError (曾走到特征值求解再报 SINGULAR?)
    with pytest.raises(ValueError, match="方阵"):
        estimate_condition(np.ones((2, 3)), method="dense")


def test_estimate_condition_non_array_k_rejected():
    # fuzz 发现: tuple/list/标量 曾冒裸 AttributeError ('.shape')
    with pytest.raises(ValueError, match="方阵"):
        estimate_condition((0.5, 1.5))
    with pytest.raises(ValueError, match="方阵"):
        estimate_condition([1.0])
    with pytest.raises(ValueError, match="方阵"):
        estimate_condition(5.0)
    with pytest.raises(ValueError, match="方阵"):
        estimate_condition(None)
    with pytest.raises(ValueError, match="方阵"):
        estimate_condition(np.ones((2, 3)))


def test_estimate_condition_sparse_ok():
    from scipy.sparse import eye
    info = estimate_condition(eye(4), method="dense")
    assert info["condition_number"] == pytest.approx(1.0)
    assert info["status"] == "GOOD"


# ── nodes_on_edge 误用路径 (契约 A2, 此前无 raises 断言) ──

def test_nodes_on_edge_invalid_axis():
    with pytest.raises(ValueError, match="axis"):
        _mesh().nodes_on_edge("z", "min")


def test_nodes_on_edge_invalid_edge():
    with pytest.raises(ValueError, match="edge"):
        _mesh().nodes_on_edge("x", "mid")


def test_nodes_on_edge_invalid_tol():
    m = _mesh()
    with pytest.raises(ValueError, match="tol"):
        m.nodes_on_edge("x", "min", tol=-1.0)
    with pytest.raises(ValueError, match="tol"):
        m.nodes_on_edge("x", "min", tol=float("nan"))


def test_nodes_on_edge_valid():
    m = _mesh()
    left = m.nodes_on_edge("x", "min")
    assert set(left) == {0, 2}  # x=0 的节点: (0,0) 与 (0,1)


# ── replace_nodes / replace_elements 误用路径 (契约 A2) ──

def test_replace_nodes_shape_mismatch():
    with pytest.raises(ValueError, match="形状"):
        _mesh().replace_nodes(NODES[:3])


def test_replace_nodes_nan():
    with pytest.raises(ValueError, match="NaN/Inf"):
        _mesh().replace_nodes(np.full((4, 2), np.nan))


def test_replace_elements_float_index():
    with pytest.raises(ValueError, match="整数"):
        _mesh().replace_elements(np.array([[0.5, 1.0, 3.0]]))


def test_replace_elements_out_of_range():
    with pytest.raises(ValueError, match="越界"):
        _mesh().replace_elements(np.array([[0, 1, 9]]))


def test_replace_elements_duplicate():
    with pytest.raises(ValueError, match="重复"):
        _mesh().replace_elements(np.array([[0, 1, 3], [0, 1, 3]]))


def test_replace_elements_negative():
    with pytest.raises(ValueError, match="越界"):
        _mesh().replace_elements(np.array([[-1, 1, 3]]))
