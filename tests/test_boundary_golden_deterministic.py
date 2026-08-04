"""边界金标准快照 — 确定性手建网格层 (无 gmsh 依赖, 全部环境运行).

覆盖: CST/Q4/Q4R/Q4I 网格、嵌套孔板、整圆/椭圆闭环检测.
金标准逐位对比入库文件 (tests/boundary_golden/), 且 build 路径输出
必须与 detect 路径逐位一致 (无注册表时两者是同一语义).

判别性: 金标准为显式入库文本, 非运行时自派生 — 识别器重排/翻转/
改标签 → 对比必红. 生成器: FEM2D_UPDATE_GOLDEN=1.
"""
import numpy as np

from fem2d import Mesh, build_boundary_segments, detect_boundaries

from tests.boundary_snapshot import (
    build_snapshot,
    compare_golden,
    edge_coverage_check,
    render,
)


def _check(mesh, name):
    """自动检测路径 → 金标准; build 路径必须与 detect 逐位一致."""
    segs = detect_boundaries(mesh)
    assert edge_coverage_check(mesh, segs)
    snapshot = build_snapshot(mesh, segs)
    compare_golden(f"{name}.json", snapshot)

    built = build_boundary_segments(mesh)
    assert edge_coverage_check(mesh, built)
    assert render(build_snapshot(mesh, built)) == render(snapshot), (
        f"{name}: build_boundary_segments 与 detect 输出不一致")


def _cst_triangle():
    """单三角形 CST — 全部直边."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return Mesh(nodes=nodes, elements=np.array([[0, 1, 2]], dtype=int),
                E=1e6, nu=0.3, thickness=1.0, elem_type="CPS3")


def _cst_square():
    """2×2 方板 CST (2 三角形) — 4 直边."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
    return Mesh(nodes=nodes, elements=elems,
                E=1e6, nu=0.3, thickness=1.0, elem_type="CPS3")


def _q4_square(elem_type):
    """2×2 方板 Q4 单单元 — 4 直边 (CPS4/CPS4R/CPS4I 同拓扑)."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.array([[0, 1, 2, 3]], dtype=int)
    return Mesh(nodes=nodes, elements=elems,
                E=1e6, nu=0.3, thickness=1.0, elem_type=elem_type)


def _holed_plate():
    """8×8 板带 4×4 孔 (二级嵌套): 外环 0-3 + 内环 4-7."""
    nodes = np.array([[0, 0], [8, 0], [8, 8], [0, 8],
                      [2, 2], [6, 2], [6, 6], [2, 6]], dtype=float)
    elems = np.array([[0, 1, 4], [0, 4, 7], [1, 2, 5], [1, 5, 4],
                      [2, 3, 6], [2, 6, 5], [3, 0, 7], [3, 7, 6]], dtype=int)
    return Mesh(nodes=nodes, elements=elems,
                E=1e6, nu=0.3, thickness=1.0, elem_type="CPS3")


def _circle_fan(n=32, radius=1.5):
    """扇形 CST 网格: 中心 + 32 边正圆 — 闭环整圆检测."""
    angle = 2.0 * np.pi * np.arange(n) / n
    rim = radius * np.column_stack([np.cos(angle), np.sin(angle)])
    nodes = np.vstack([[0.0, 0.0], rim])
    elems = np.array([
        [0, 1 + i, 1 + (i + 1) % n]
        for i in range(n)
    ], dtype=int)
    return Mesh(nodes=nodes, elements=elems,
                E=1e6, nu=0.3, thickness=1.0, elem_type="CPS3")


def _ellipse_fan(n=40, a=2.0, b=1.0):
    """扇形 CST 网格: 40 边椭圆 (2:1) — 闭环椭圆检测."""
    angle = 2.0 * np.pi * np.arange(n) / n
    rim = np.column_stack([a * np.cos(angle), b * np.sin(angle)])
    nodes = np.vstack([[0.0, 0.0], rim])
    elems = np.array([
        [0, 1 + i, 1 + (i + 1) % n]
        for i in range(n)
    ], dtype=int)
    return Mesh(nodes=nodes, elements=elems,
                E=1e6, nu=0.3, thickness=1.0, elem_type="CPS3")


def test_cst_triangle_golden():
    _check(_cst_triangle(), "cst_triangle")


def test_cst_square_golden():
    _check(_cst_square(), "cst_square")


def test_q4_square_golden():
    _check(_q4_square("CPS4"), "q4_square")


def test_q4r_square_golden():
    _check(_q4_square("CPS4R"), "q4r_square")


def test_q4i_square_golden():
    _check(_q4_square("CPS4I"), "q4i_square")


def test_holed_plate_golden():
    _check(_holed_plate(), "holed_plate")


def test_circle_fan_golden():
    _check(_circle_fan(), "circle_fan")


def test_ellipse_fan_golden():
    _check(_ellipse_fan(), "ellipse_fan")


def test_discriminative_hardcoded_expectations():
    """判别性证据: 期望值显式硬编码 (不取自运行输出) — 识别器重排/
    翻转/改标签, 或段序 sort 键变化, 此处必红 (金标准之外的独立锁)."""
    q4 = detect_boundaries(_q4_square("CPS4"))
    assert [s["label"] for s in q4] == [
        "外边 直边 (0,0)→(1,0)",
        "外边 直边 (1,0)→(1,1)",
        "外边 直边 (1,1)→(0,1)",
        "外边 直边 (0,1)→(0,0)",
    ]
    assert [s["type"] for s in q4] == ["line"] * 4

    circle = detect_boundaries(_circle_fan())
    assert len(circle) == 1
    assert circle[0]["type"] == "arc"
    assert circle[0]["closed"] is True
    assert circle[0]["label"] == "外边 整圆 R=1.5"
    # 环分解自最小节点 1 起 CW 走序, _orient_loops 反转成 CCW 后
    # 起点位移到 2 — 确定性行为, 金标准已锁定 (改动此处必红).
    assert circle[0]["nodes"] == [
        *range(2, 33), 1, 2]

    ellipse = detect_boundaries(_ellipse_fan())
    assert len(ellipse) == 1
    assert ellipse[0]["type"] == "ellipse"
    assert ellipse[0]["label"] == "外边 椭圆 a=2 b=1"
