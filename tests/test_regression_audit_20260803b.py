"""2026-08-03 全链路复扫回归测试 (4 agent 并行审查发现).

覆盖: 边界微尺度分段 / _validate_nodes 微尺度 / geo_spec 精度塌缩 /
.spec 格式错误 / solver trivial 误标 / area_cv 微尺度 / nodes_on_edge
微尺度 / 压力 callable 上下文。
"""
import contextlib
import io
import math

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════
# 边界: 微尺度模型分段必须保持直边识别 (曾绝对 1e-15 → 全判 arc)
# ═══════════════════════════════════════════════════════════════

def _square_loop_coords(scale):
    # CCW 方形: 底 → 右 → 顶 → 左 (每条边 10 个采样点, 不重复闭合点 —
    # 与 Gmsh 曲线网格一致, 闭合靠 wrap 索引)
    t = np.linspace(0, 1, 40, endpoint=False)
    x = np.where(t < 0.25, t * 4,
        np.where(t < 0.5, 1.0,
            np.where(t < 0.75, 1.0 - (t - 0.5) * 4, 0.0)))
    y = np.where(t < 0.25, 0.0,
        np.where(t < 0.5, (t - 0.25) * 4,
            np.where(t < 0.75, 1.0, 1.0 - (t - 0.75) * 4)))
    return np.column_stack([scale * x, scale * y])


def test_boundary_micro_scale_square_still_lines():
    """1e-16 尺度方形边界必须仍识别为直边 — 曾 1e-15 绝对零长判据
    使每条边都被判退化, 整环合并成 arc."""
    from fem2d.boundary.detectors import LineDetector
    from fem2d.boundary.geometry import sharp_corner_indices
    for scale in (1.0, 1e-9, 1e-16):
        coords = _square_loop_coords(scale)
        # 闭合环的每条边: 取角点切分 (方形角 = 90° 转角)
        corners = sharp_corner_indices(coords)
        assert len(corners) == 4, \
            f"scale={scale}: 角点 {len(corners)} ≠ 4 (曾微尺度全跳过)"
        # 角点间段应分类为 line (闭合环切分需 wrap 拼接);
        # 旧 _classify_line 私有函数已迁移为 detectors.LineDetector
        line_detector = LineDetector()
        n_line = 0
        for i, c in enumerate(corners):
            nxt = corners[(i + 1) % len(corners)]
            if nxt > c:
                chain = coords[c:nxt + 1]
            else:
                chain = np.concatenate([coords[c:], coords[:nxt + 1]])
            cls = line_detector.detect(
                chain, scale=1.0, is_outer=True, closed=False)
            n_line += 1 if cls else 0
        assert n_line == 4, f"scale={scale}: 仅 {n_line}/4 段识别为直边"


# ═══════════════════════════════════════════════════════════════
# 边界: _validate_nodes 微尺度不误拒 (曾 max(...,1.0) 容差 > 整个模型)
# ═══════════════════════════════════════════════════════════════

def test_validate_nodes_micro_scale_not_rejected():
    """微尺度物理曲线节点链不得被误判零长边 — 曾 1.0 下限使容差
    ~7e-15 > 整个 1e-16 模型, 每条边都 ValueError."""
    from fem2d.boundary.segment_builder import BoundarySegmentBuilder
    from fem2d import Mesh
    s = 1e-16
    nodes = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=float)
    mesh = Mesh(nodes=nodes, elements=np.array([[0, 1, 2], [0, 2, 3]]),
                E=210e9, nu=0.3, thickness=1.0)
    builder = BoundarySegmentBuilder(mesh, {}, {}, s)
    builder._validate_nodes(np.array([0, 1, 2]), "微尺度边")


# ═══════════════════════════════════════════════════════════════
# 输入端: .txt 微尺度孔坐标不得塌缩 (曾 {:.6f} → 0.000000)
# ═══════════════════════════════════════════════════════════════

def test_geo_spec_micro_hole_full_precision(tmp_path):
    """微尺度孔坐标必须全精度写入 .geo — 曾 6 位小数把 1e-7 孔塌缩成
    0.000000, Gmsh 得到零半径圆弧."""
    from scripts.geo_spec import parse_spec, generate_geo
    spec_path = tmp_path / "micro.txt"
    spec_path.write_text("类型 矩形板\n宽 1e-6\n高 2e-6\n"
                         "内孔 圆 x=1e-7 y=2e-7 r=1e-7\n网格 1e-7\n",
                         encoding="utf-8")
    spec = parse_spec(str(spec_path))
    out = str(tmp_path / "out.geo")
    generate_geo(spec, out)
    text = open(out, encoding="utf-8").read()
    assert "0.000000" not in text, \
        "存在 6 位小数塌缩点 (微尺度坐标被截断)"
    assert "1e-07" in text, "微尺度孔坐标未写入"


# ═══════════════════════════════════════════════════════════════
# 输入端: .spec 格式错误必须响亮 (曾无 = 行 / 空值静默丢键)
# ═══════════════════════════════════════════════════════════════

def test_spec_config_missing_equals_rejected(tmp_path):
    """.spec 漏写 = 必须报错 — 曾静默丢键, 约束消失无提示, 与 .txt
    缺值报错行为分叉."""
    from fem2d.preprocess import parse_spec_config
    p = tmp_path / "bad.spec"
    p.write_text("mesh = m.geo\nfix left\nE = 2.1e11\n", encoding="utf-8")
    with pytest.raises(ValueError, match="="):
        parse_spec_config(str(p))


def test_spec_config_empty_value_rejected(tmp_path):
    """.spec 空值 (fix =) 必须报错 — 曾静默丢键."""
    from fem2d.preprocess import parse_spec_config
    p = tmp_path / "empty.spec"
    p.write_text("mesh = m.geo\nfix =\n", encoding="utf-8")
    with pytest.raises(ValueError, match="空值"):
        parse_spec_config(str(p))


# ═══════════════════════════════════════════════════════════════
# 求解器: 微尺度非平凡解不得标 "trivial solution"
# ═══════════════════════════════════════════════════════════════

def test_solver_micro_nontrivial_not_labeled_trivial():
    """微尺度载荷的非平凡解 (max|u| 显著非零) 不得打印 "trivial solution"
    — 曾 denom<1e-15 绝对判据误标."""
    from fem2d import Mesh, solve
    s = 1e-16
    m = Mesh(nodes=np.array([[0, 0], [s, 0], [0, s]], dtype=float),
             elements=np.array([[0, 1, 2]]), E=210e9, nu=0.3,
             thickness=1e-9, plane_type="stress", elem_type="CPS3")
    m.fix_node(0, "both", 0.0)
    m.fix_node(1, "both", 0.0)
    m.add_force(2, 1e-16, 0.0)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = solve(m, method="elimination", verbose=True)
    out = buf.getvalue()
    assert "trivial solution" not in out, \
        f"非平凡解被误标 trivial: {out!r}"
    assert np.abs(r["u"]).max() > 0, "微尺度载荷应产生非零位移"


# ═══════════════════════════════════════════════════════════════
# 网格质量: area_cv 微尺度不失真 (曾 1e-30 地板主导分母)
# ═══════════════════════════════════════════════════════════════

def test_quality_area_cv_micro_scale_realistic():
    """微尺度网格 (面积 ~1e-32) 的 area_cv 必须反映真实离散 — 曾
    1e-30 绝对地板把 CV 从 ~27% 压到 ~0.4%."""
    from fem2d.quality import evaluate
    from fem2d import Mesh
    s = 1e-16
    nodes = np.array([[0, 0], [s, 0], [s, s], [0, 4 * s]], dtype=float)
    # 面积比 1:4 的两个三角形 → 面积 {0.5s², 2s²}, 真实 CV ≈ 0.6
    elems = np.array([[0, 1, 2], [0, 2, 3]])
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=1.0)
    q = evaluate(m)
    assert q["area_cv"] > 0.01, \
        f"area_cv={q['area_cv']:.4f} — 微尺度 CV 被 1e-30 地板压平"


# ═══════════════════════════════════════════════════════════════
# 网格: nodes_on_edge 微尺度正确 (曾 span<1e-30 → 1.0, 返回全部节点)
# ═══════════════════════════════════════════════════════════════

def test_nodes_on_edge_micro_scale_selective():
    """1e-32 跨度模型 nodes_on_edge('x','min') 必须只返回左边界节点 —
    曾 1.0 地板使 tol=1e-8 覆盖全部节点."""
    from fem2d import Mesh
    s = 1e-32
    m = Mesh(nodes=np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=float),
             elements=np.array([[0, 1, 2], [0, 2, 3]]), E=210e9, nu=0.3,
             thickness=1.0)
    left = m.nodes_on_edge("x", "min")
    right = m.nodes_on_edge("x", "max")
    assert set(left) == {0, 3}, f"左边界应 {0,3}, 得到 {set(left)}"
    assert set(right) == {1, 2}, f"右边界应 {1,2}, 得到 {set(right)}"


# ═══════════════════════════════════════════════════════════════
# 载荷: 压力 callable 异常带上下文 (曾裸异常, 面力路径已有包装)
# ═══════════════════════════════════════════════════════════════

def test_pressure_callable_error_has_edge_context():
    """压力表达式在 Gauss 点抛异常必须带边/点上下文 — 曾裸
    ValueError (math domain) 无载荷上下文, 面力路径已有包装
   ."""
    from fem2d import Mesh, solve
    nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=float)
    elems = np.array([[0, 1, 2], [1, 3, 2]], dtype=int)
    m = Mesh(nodes=nodes, elements=elems, E=210e9, nu=0.3, thickness=0.01)
    for n in [0, 2]:
        m.fix_node(n, "both", 0.0)
    # x<1 处 x-1<0 → math.log 抛 domain error; 压力路径应包装上下文
    m.add_pressure(1, 3, lambda x, y: math.log(x - 1.0))
    with pytest.raises(ValueError, match="压力表达式"):
        solve(m, method="elimination", verbose=False)


# ═══════════════════════════════════════════════════════════════
# 第二轮外部审查修复 (2026-08-03): error_est 尺度不变 / 椭圆轴比 /
# BC 共享校验器 / 装配极端尺度
# ═══════════════════════════════════════════════════════════════

def test_error_est_eta_invariant_at_extreme_loads():
    """Z2 eta 在载荷 1e-152 量级必须保持尺度不变 — 曾能量二次型下溢,
    eta 从 65.7% 塌缩到 0.07% (第二轮外部审查复现)."""
    from fem2d import Mesh, solve
    from fem2d.error_est import estimate
    L, H, t, nx, ny = 5.0, 1.0, 0.1, 8, 4
    ncol = nx + 1
    nodes = [[i * L / nx, j * H / ny] for j in range(ny + 1)
             for i in range(nx + 1)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (j * ncol + i, j * ncol + i + 1,
                          (j + 1) * ncol + i, (j + 1) * ncol + i + 1)
            elems.append([a, b, d]); elems.append([a, d, c])
    def run(tau):
        m = Mesh(nodes=np.array(nodes), elements=np.array(elems),
                 E=210e9, nu=0.3, thickness=t, plane_type="stress")
        for i in range(ny + 1):
            m.fix_node(i * ncol, "both", 0.0)
        right = [i * ncol + ncol - 1 for i in range(ny + 1)]
        for k in range(ny):
            m.add_traction(right[k], right[k + 1], 0.0, tau)
        r = solve(m, method="elimination", verbose=False)
        return estimate(m, r, method="SPR", verbose=False)["eta"]
    base = run(1e5)
    for tau in (1e-100, 1e-150, 1e-152):
        assert abs(run(tau) - base) < 0.05, \
            f"tau={tau}: eta={run(tau):.4f} vs 基准 {base:.4f}"


def test_ellipse_aspect_ratio_scale_invariant():
    """2:1 椭圆在微尺度 (1e-32) 必须保持轴比 2:1 —
    曾轴比分母 1e-30 下限使微尺度轴比失真 (第二轮外部审查复现)."""
    from fem2d.boundary.geometry import fit_ellipse
    t = np.linspace(0, 2 * np.pi, 41)
    for scale in (1.0, 1e-16, 1e-32):
        coords = np.column_stack([scale * 2.0 * np.cos(t),
                                  scale * 1.0 * np.sin(t)])
        ellipse = fit_ellipse(coords)
        assert ellipse is not None, f"scale={scale}: 椭圆拟合失败"
        _, _, semi_major, semi_minor, _ = ellipse
        ratio = semi_major / semi_minor
        assert 1.5 < ratio < 2.5, \
            f"scale={scale}: 轴比 {ratio:.3f} 失真 (应 2:1)"


def test_bc_partition_validator_rejects_overlap_and_gaps():
    """apply_elimination 必须拒绝 free/fixed 重叠与遗漏 DOF —
    曾约束覆盖自由解、遗漏 DOF 静默设 0 (第二轮外部审查复现)."""
    from scipy.sparse import csr_matrix
    from fem2d.bc import apply_elimination
    K = csr_matrix(np.eye(6))
    F = np.zeros(6)
    with pytest.raises(ValueError, match="重叠"):
        apply_elimination(K, F, [0, 1, 2, 3, 4], [0, 5], [0.0, 0.0])
    with pytest.raises(ValueError, match="未覆盖"):
        apply_elimination(K, F, [1, 2, 3], [4, 5], [0.0, 0.0])
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        # 纯 Dirichlet 分支曾绕过 solver 名称检查
        apply_elimination(K, F, [], [0, 1, 2, 3, 4, 5], np.zeros(6),
                          linear_solver="bogus")


def test_assembly_weighted_extreme_scale_no_overflow():
    """微尺度几何 (1e-150) 刚度组装不得溢出 — 曾 BᵀDB ~ E/L² 中间量 Inf,
    装配后误报 'Factor is exactly singular' (第二轮外部审查复现)."""
    from fem2d import Mesh
    s = 1e-150
    for et in ("CPS4", "CPS4I"):
        m = Mesh(nodes=np.array([[0., 0.], [s, 0.], [s, s], [0, s]]),
                 elements=np.array([[0, 1, 2, 3]]), E=1e9, nu=0.3,
                 thickness=1e-9, elem_type=et)
        m.build_connectivity()
        Ke = m.element_kernel.stiffness_batch(m)[0]
        assert np.all(np.isfinite(Ke)), \
            f"{et} @ 1e-150 刚度溢出"


def test_error_est_elem_contrib_scale_invariant():
    """Z2 elem_contrib 在微尺度载荷必须 sum=100% 且 worst 稳定 —
    曾归一化改动残留: 乘回 s_scale 后的 elem_err² (~1e-308 次正规)
    求和失真, sum 只到 0.46% (回归审计 2026-08-03 发现并修复)."""
    from fem2d import Mesh, solve
    from fem2d.error_est import estimate
    L, H, t, nx, ny = 5.0, 1.0, 0.1, 8, 4
    ncol = nx + 1
    nodes = [[i * L / nx, j * H / ny] for j in range(ny + 1)
             for i in range(nx + 1)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (j * ncol + i, j * ncol + i + 1,
                          (j + 1) * ncol + i, (j + 1) * ncol + i + 1)
            elems.append([a, b, d]); elems.append([a, d, c])
    def run(tau):
        m = Mesh(nodes=np.array(nodes), elements=np.array(elems),
                 E=210e9, nu=0.3, thickness=t, plane_type="stress")
        for i in range(ny + 1):
            m.fix_node(i * ncol, "both", 0.0)
        right = [i * ncol + ncol - 1 for i in range(ny + 1)]
        for k in range(ny):
            m.add_traction(right[k], right[k + 1], 0.0, tau)
        r = solve(m, method="elimination", verbose=False)
        return estimate(m, r, method="SPR", verbose=False)
    ref = run(1e5)
    for tau in (1e-150, 1e-155):
        z = run(tau)
        ec = z["elem_contrib"]
        assert abs(ec.sum() - 100.0) < 0.5, \
            f"tau={tau}: elem_contrib sum={ec.sum():.2f}% (应 100)"
        assert z["worst_elem"] == ref["worst_elem"], "worst_elem 漂移"


# ═══════════════════════════════════════════════════════════════
# 第三轮外部审查修复 (2026-08-03): 椭圆链路 / elimination / penalty /
# eta 1e-300 / Q4R 单警告
# ═══════════════════════════════════════════════════════════════

def test_ellipse_full_chain_micro_scale(tmp_path):
    """完整 detect_boundaries 链路在微尺度必须识别 2:1 椭圆 —
    曾 topology.py 复制旧 1e-30 轴比逻辑, 1e-32 误判整圆 (第三轮审查)."""
    from fem2d import Mesh, detect_boundaries
    from scipy.spatial import Delaunay
    for scale in (1.0, 1e-16, 1e-32):
        rng = np.random.RandomState(7)
        pts = []
        while len(pts) < 40:
            x, y = rng.rand(2) * 2 - 1
            if (x / 2.0) ** 2 + y ** 2 < 0.9:
                pts.append([x * 2.0, y])
        theta = np.linspace(0, 2 * np.pi, 33, endpoint=False)
        edge = np.column_stack(
            [2.0 * np.cos(theta), np.sin(theta)])
        all_pts = np.vstack([np.array(pts), edge]) * scale
        tri = Delaunay(all_pts)
        cent = all_pts[tri.simplices].mean(axis=1)
        inside = (cent[:, 0] / (2 * scale)) ** 2 + (
            cent[:, 1] / scale) ** 2 < 1.0
        m = Mesh(nodes=all_pts, elements=tri.simplices[inside],
                 E=2.1e11, nu=0.3, thickness=0.01)
        segs = detect_boundaries(m)
        assert segs and segs[0]["type"] == "ellipse", \
            f"scale={scale}: 误判为 {segs[0]['type'] if segs else None}"
        # 标签必须保留精度 (科学计数) — 曾 .3f 塌缩成 a=0.000
        assert "a=" in segs[0]["label"] \
            and "a=0.000" not in segs[0]["label"], \
            f"scale={scale}: 标签精度丢失 {segs[0]['label']}"


def test_elimination_rejects_free_duplicates_and_f_nan():
    """apply_elimination 必须拒绝 free 自身重复与 F 含 NaN —
    曾重复延迟到 SuperLU 奇异、NaN 返回 NaN 位移 (第三轮审查)."""
    from scipy.sparse import csr_matrix
    from fem2d.bc import apply_elimination
    K = csr_matrix(np.eye(6))
    with pytest.raises(ValueError, match="duplicates"):
        apply_elimination(K, np.zeros(6), [1, 1, 2, 3, 4, 5], [0], [0.0])
    with pytest.raises(ValueError, match="NaN"):
        apply_elimination(
            K, np.array([0., 1, np.nan, 0, 0, 0]),
            [1, 2, 3, 4, 5], [0], [0.0])
    with pytest.raises(ValueError, match="square"):
        apply_elimination(
            csr_matrix(np.eye(6)[:5]), np.zeros(6), [1, 2, 3, 4], [0, 5],
            [0., 0.])


def test_penalty_relative_scale_validation():
    """penalty 必须按刚度尺度校验 — 曾绝对 >1.0: K=1e-12 时 penalty=1e-4
    (相对 1e8 倍, 有效) 被误拒, K=1e12 时 penalty=2 (无效) 被接受
    (第三轮审查)."""
    from scipy.sparse import csr_matrix
    from fem2d.bc import apply_penalty
    # 有效: 罚刚度 >= max|K_ii| 即接受
    K_small = csr_matrix(np.eye(6) * 1e-12)
    apply_penalty(K_small, np.zeros(6), [0], penalty=1e-4)
    # 无效: 罚刚度 < max|K_ii| 拒绝
    K_big = csr_matrix(np.eye(6) * 1e12)
    with pytest.raises(ValueError, match="max|K_ii|"):
        apply_penalty(K_big, np.zeros(6), [0], penalty=2.0)


def test_eta_invariant_at_1e300():
    """Z2 eta 在载荷 1e-300 量级必须仍精确 — 曾 1e-160 塌缩、1e-310
    NaN (第三轮审查). 归一化空间比值彻底尺度无关."""
    from fem2d import Mesh, solve
    from fem2d.error_est import estimate
    L, H, t, nx, ny = 5.0, 1.0, 0.1, 8, 4
    ncol = nx + 1
    nodes = [[i * L / nx, j * H / ny] for j in range(ny + 1)
             for i in range(nx + 1)]
    elems = []
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = (j * ncol + i, j * ncol + i + 1,
                          (j + 1) * ncol + i, (j + 1) * ncol + i + 1)
            elems.append([a, b, d]); elems.append([a, d, c])
    def run(tau):
        m = Mesh(nodes=np.array(nodes), elements=np.array(elems),
                 E=210e9, nu=0.3, thickness=t, plane_type="stress")
        for i in range(ny + 1):
            m.fix_node(i * ncol, "both", 0.0)
        right = [i * ncol + ncol - 1 for i in range(ny + 1)]
        for k in range(ny):
            m.add_traction(right[k], right[k + 1], 0.0, tau)
        r = solve(m, method="elimination", verbose=False)
        return estimate(m, r, method="SPR", verbose=False)["eta"]
    base = run(1e5)
    for tau in (1e-150, 1e-200, 1e-300):
        assert abs(run(tau) - base) < 0.05, f"tau={tau} eta 失真"
    # 1e-310 必须有限 (双精度极限, 允许 0)
    assert np.isfinite(run(1e-310))
