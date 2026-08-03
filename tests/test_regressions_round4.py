"""第四轮审查修复的回归锁定 — 2026-08 审查发现的缺陷.

每个测试对应一个已修复缺陷, 防止回归:
  1. 压力法向实时计算 (replace_nodes 后自动跟随几何)
  2. Q4R/Q4I 材料指纹含 plane_type (切平面态后缓存失效)
  3. ILU 预条件 (drop_tol 1e-6) 在 300+ DOF 上收敛
  4. 奇异条件数不二次崩溃 (SINGULAR? 状态)
  5. 误差指标对载荷拆分方式不变
  6. @FEM 配置单次合并 (无错误 WARN)
  7. 装配 rows/cols 标准 COO 语义 (非对称内核不转置)
"""
import os
import tempfile
import warnings

import numpy as np
import pytest

from fem2d import Mesh, solve
from fem2d.convergence import _gen_cantilever_mesh


def _cantilever(nx=8, ny=4, code="CPS4", E=3e7):
    nodes, elements = _gen_cantilever_mesh(8.0, 1.0, nx, ny, elem_type=code)
    mesh = Mesh(nodes=nodes, elements=elements, E=E, nu=0.3,
                thickness=1.0, elem_type=code)
    for n in mesh.nodes_on_edge("x", "min", tol=1e-6):
        mesh.fix_node(int(n), "both", 0.0)
    right = sorted(mesh.nodes_on_edge("x", "max", tol=1e-6),
                   key=lambda n: mesh.nodes[int(n), 1])
    I = 1.0 / 12
    for k in range(len(right) - 1):
        a, b = int(right[k]), int(right[k + 1])
        mesh.add_traction(a, b, 0.0,
                          lambda x, y: -1.0 / (2 * I) * (0.25 - y**2))
    return mesh


class _AsymKernel:
    """非对称刚度内核 — 装配 rows/cols 方向测试专用."""

    nodes_per_element = 4
    local_edges = ((0, 1), (1, 2), (2, 3), (3, 0))

    def __init__(self):
        self.dofs_per_element = 8

    def build_geometry(self, n, e):
        return {"areas": np.ones(len(e)),
                "centroids": np.zeros((len(e), 2)),
                "signed_areas": np.ones(len(e))}

    def stiffness_batch(self, mesh, element_slice=None):
        sel = slice(None) if element_slice is None else element_slice
        n = len(mesh.elements[sel])
        return np.tile(np.arange(64).reshape(8, 8), (n, 1, 1)).astype(float)

    def jacobian_determinants(self, mesh):
        return np.ones((mesh.n_elements, 1))

    def body_force_vector(self, mesh, eid, bf):
        return np.zeros(8)

    def shape_values_at(self, coords, x, y, tol=1e-12):
        return None

    def verify_mesh(self, mesh, verbose=True):
        return True

    def recovery_quadrature(self, mesh, eid):
        return None

    def recovery_shape_matrix(self, mesh):
        return None

    def recovery_weights(self, mesh):
        return None

    def compute_response(self, mesh, u_e):
        return None

    def response_at_quadrature(self, mesh, u_e):
        return None


def test_pressure_normal_follows_replaced_geometry():
    """replace_nodes 后压力法向必须按当前几何重算 (曾缓存旧法向)."""
    from fem2d.loads import assemble as assemble_loads

    nodes = np.array([[1.0, -1.0], [2.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    mesh = Mesh(nodes=nodes, elements=np.array([[0, 1, 2, 3]]),
                E=2.1e11, nu=0.3, thickness=0.01, elem_type="CPS4")
    mesh.add_pressure(0, 1, 1e5)
    n0 = mesh.boundary_outward_normal(0, 1)
    F0 = assemble_loads(mesh, mesh.n_dof)
    L0 = float(np.linalg.norm(mesh.nodes[1] - mesh.nodes[0]))
    assert np.allclose([F0[0::2].sum(), F0[1::2].sum()],
                       [-1e5 * 0.01 * L0 * n0[0], -1e5 * 0.01 * L0 * n0[1]],
                       rtol=1e-9)

    mesh.replace_nodes(np.array([[1.0, -1.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0]]))
    n1 = mesh.boundary_outward_normal(0, 1)
    F1 = assemble_loads(mesh, mesh.n_dof)
    L1 = float(np.linalg.norm(mesh.nodes[1] - mesh.nodes[0]))
    assert np.allclose([F1[0::2].sum(), F1[1::2].sum()],
                       [-1e5 * 0.01 * L1 * n1[0], -1e5 * 0.01 * L1 * n1[1]],
                       rtol=1e-9)
    assert not np.allclose(n0, n1), "几何变了但法向没变"


@pytest.mark.parametrize("code", ["CPS4I", "CPS4R"])
def test_plane_switch_invalidates_material_cache(code):
    """stress→strain 切换后 Q4R/Q4I 的应力必须与全新构造一致 (曾用旧 D)."""
    nodes, elements = _gen_cantilever_mesh(8.0, 1.0, 8, 4, elem_type=code)
    right = sorted([n for n in range(len(nodes)) if nodes[n, 0] > 8.0 - 1e-6],
                   key=lambda n: nodes[int(n), 1])
    I = 1.0 / 12

    def build(plane):
        m = Mesh(nodes=nodes, elements=elements, E=3e7, nu=0.3,
                 thickness=1.0, elem_type=code, plane_type=plane)
        for n in m.nodes_on_edge("x", "min", tol=1e-6):
            m.fix_node(int(n), "both", 0.0)
        for k in range(len(right) - 1):
            a, b = int(right[k]), int(right[k + 1])
            m.add_traction(a, b, 0.0,
                           lambda x, y: -1.0 / (2 * I) * (0.25 - y**2))
        return m

    m = build("stress")
    solve(m, verbose=False)
    m.plane_type = "strain"
    res = solve(m, verbose=False)
    ref = solve(build("strain"), verbose=False)
    s = m.element_kernel.response_at_quadrature(
        m, res["u"][m.element_dofs])[0]
    rel = np.abs(s - ref["stress_qp"]).max() / np.abs(ref["stress_qp"]).max()
    assert rel < 1e-12


def test_ilu_preconditioner_converges_medium_mesh():
    """ILU-CG 在 300+ DOF 上必须收敛 (drop_tol 1e-4 曾产生非正定因子)."""
    mesh = _cantilever(nx=16, ny=8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = solve(mesh, verbose=False, linear_solver="ilu")
    assert result["residual"] < 1e-8
    ref = solve(mesh, verbose=False, linear_solver="direct")
    rel = np.abs(result["u"] - ref["u"]).max() / np.abs(ref["u"]).max()
    assert rel < 1e-8


def test_singular_condition_does_not_format_crash(monkeypatch):
    """SINGULAR? 状态 (condition_number=None) 不能触发 :.2e 二次崩溃.

    通过 monkeypatch estimate_condition 返回 SINGULAR?, 真正调用
    solve(check_condition=True) 的打印路径 (曾只手工重写格式化分支).
    """
    import fem2d.solver as solver_mod

    mesh = _cantilever(nx=4, ny=2)
    # 约束 K_aa 的奇异估计 — 让 solve 走到条件数打印分支
    monkeypatch.setattr(
        solver_mod, "estimate_condition",
        lambda K_aa, method="auto": {
            "condition_number": None,
            "status": "SINGULAR?",
            "error": "RuntimeError: Factor is exactly singular",
        })
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 不崩溃即通过 — 修复前 :.2e 会对 None 抛 TypeError
        result = solve(mesh, verbose=False, check_condition=True)
    assert result.get("condition_info", {}).get("status") == "SINGULAR?"


def test_error_indicator_invariant_to_load_splitting():
    """同载荷不同拆分方式 → 残差指标一致 (曾拆分敏感)."""
    from fem2d.error_est import element_refinement_indicator

    nodes, elements = _gen_cantilever_mesh(8.0, 1.0, 8, 4, elem_type="CPS4")
    right = sorted([n for n in range(len(nodes)) if nodes[n, 0] > 8.0 - 1e-6],
                   key=lambda n: nodes[int(n), 1])
    edges = [(int(right[k]), int(right[k + 1])) for k in range(len(right) - 1)]

    def run(specs):
        m = Mesh(nodes=nodes, elements=elements, E=3e7, nu=0.3,
                 thickness=1.0, elem_type="CPS4")
        for n in m.nodes_on_edge("x", "min", tol=1e-6):
            m.fix_node(int(n), "both", 0.0)
        for a, b, tx, ty in specs:
            m.add_traction(a, b, tx, ty)
        return element_refinement_indicator(m, solve(m, verbose=False)).sum()

    single = run([(a, b, 1e6, 0.0) for a, b in edges])
    split = run([(a, b, 5e5, 0.0) for a, b in edges]
                + [(a, b, 5e5, 0.0) for a, b in edges])
    rel = abs(single - split) / max(single, 1e-30)
    assert rel < 1e-10


def test_geo_config_merged_once_no_false_warning(monkeypatch):
    """@FEM 配置经真实 .geo → resolve_input → build_model 双阶段只合并一次.

    曾只直接调 merge_geo_fem_config() — 不穿生产流程, 二次合并的
    "错误 WARN" 回归无法被捕获. 这里 monkeypatch 网格生成, 其余走
    runner.main 完整生产路径.
    """
    import contextlib
    import io

    from fem2d import input_source
    from fem2d.runner import main

    mesh = tempfile.NamedTemporaryFile(
        mode="w", suffix=".msh", delete=False, encoding="utf-8")
    mesh.close()
    geo = tempfile.NamedTemporaryFile(
        mode="w", suffix=".geo", delete=False, encoding="utf-8")
    geo.write("lc = 0.1;\n"
              "// @FEM:fix=左_固定\n"
              "// @FEM:traction=右_拉力_1000000.0,1000000.0,0\n"
              "// @FEM:pressure=顶_压力,500000.0\n")
    geo.close()

    class _FakeGmshImport:
        """替代 import_msh 结果 — runner 只消费 nodes/elements/elem_type 等."""
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        elements = np.array([[0, 1, 2, 3]])
        elem_type = "CPS4"
        node_tag_to_index = {0: 0, 1: 1, 2: 2, 3: 3}
        element_tag_to_index = {0: 0}
        regions = None

    def fake_generate(geo_path, *, quad=False, output_path=None, plane_type="stress"):
        # 生产 resolve_geo 的生成步骤 — 返回 .msh 路径 + 假导入结果 (不调 gmsh)
        msh_path = os.path.splitext(mesh.name)[0] + ".msh"
        return msh_path, _FakeGmshImport()

    monkeypatch.setattr(input_source, "generate_geo_with_topology", fake_generate)
    try:
        # 场景 A: CLI 显式 traction → 配置整体跳过, 恰好 1 个 WARN, 无二次合并
        # (临时网格无语义边名, fix 用数字索引 "1")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main([geo.name, "--fix", "1", "--traction", "1:1e6,0",
                         "--no-plot"])
        out = buf.getvalue()
        assert code == 0, out[-300:]
        # 配置跳过 WARN 恰好 1 次 (二次合并曾产生第 2 个);
        # 边界语义的 "未恢复 Physical Curve" WARN 与配置合并无关, 不计数
        assert out.count("配置含面力/压力") == 1, (
            f"配置 WARN 次数 {out.count('配置含面力/压力')}: {out[-400:]}")

        # 场景 B: CLI 无 traction, 且 .geo 只含 fix (避免 @FEM 边名在
        # 临时网格上 FATAL) → 配置合并一次, 无配置 WARN, fix 生效
        geo2 = tempfile.NamedTemporaryFile(
            mode="w", suffix=".geo", delete=False, encoding="utf-8")
        # 临时网格无语义边名 — @FEM fix 用数字索引 "1" 保证可匹配
        geo2.write("lc = 0.1;\n// @FEM:fix=1\n")
        geo2.close()
        try:
            buf2 = io.StringIO()
            with contextlib.redirect_stdout(buf2):
                code2 = main([geo2.name, "--no-plot"])
            out2 = buf2.getvalue()
            assert code2 == 0, out2[-300:]
            assert out2.count("配置含面力/压力") == 0
            assert "[auto] fix" in out2
        finally:
            os.unlink(geo2.name)
    finally:
        os.unlink(geo.name)
        os.unlink(mesh.name)


def test_asymmetric_kernel_scatter_semantics(monkeypatch):
    """装配 rows/cols 必须按标准 COO: K[dofs[j], dofs[k]] = Ke[j,k].

    通过 monkeypatch 内核刚度注入非对称 Ke, 真正调用生产的
    assemble_sparse_vectorized (曾只手工构造 COO, 不穿生产路径).
    """
    from fem2d.assembly import assemble_sparse_vectorized
    from fem2d.bc import apply_elimination  # noqa: F401 — 触发 _check_symmetry 前置条件

    nodes2 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    mesh = Mesh(nodes=nodes2, elements=np.array([[0, 1, 2, 3]]),
                E=1.0, nu=0.0, thickness=1.0, elem_type="CPS4")
    monkeypatch.setattr(mesh, "element_kernel", _AsymKernel())
    mesh.build_connectivity()
    with pytest.raises(RuntimeError, match="asymmetric"):
        # 非对称 Ke 会被 _check_symmetry 守卫拦截 — 先证明生产路径真的
        # 被调用 (直接手工 COO 无法触发此守卫).
        assemble_sparse_vectorized(mesh)
    # 语义验证: 手动模拟生产 rows/cols (与 assembly.py 一致) — 但由上面的
    # 守卫证明生产路径已执行, 语义断言保持精确.
    dofs = mesh.element_dofs
    nldof = dofs.shape[1]
    rows = np.broadcast_to(dofs[:, :, None], (1, nldof, nldof)).reshape(-1)
    cols = np.broadcast_to(dofs[:, None, :], (1, nldof, nldof)).reshape(-1)
    Ke = np.arange(64).reshape(1, 8, 8)
    from scipy.sparse import coo_matrix
    K = coo_matrix((Ke.ravel(), (rows, cols)), shape=(8, 8)).toarray()
    assert K[0, 1] == Ke[0, 0, 1] == 1
    assert K[1, 0] == Ke[0, 1, 0] == 8
    assert K[2, 3] == Ke[0, 2, 3] == 19
    assert K[3, 2] == Ke[0, 3, 2] == 26


def test_assembly_rows_cols_direction_locked_with_guard_bypassed(monkeypatch):
    """评审要求: 绕过对称性守卫后, 直接断言生产函数返回矩阵的元素方向.

    上面的测试只断言"守卫会抛" — 若生产代码再次交换 rows/cols, 守卫照样
    抛, 测试仍然通过, 方向不被锁定. 本测试 monkeypatch _check_symmetry
    为 no-op, 使生产路径 assemble_sparse_vectorized 返回真实矩阵, 逐元素
    断言 COO 方向: K[dofs[j], dofs[k]] = Ke[j, k].
    """
    import fem2d.assembly as assembly_mod
    from fem2d.assembly import assemble_sparse_vectorized

    nodes2 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    mesh = Mesh(nodes=nodes2, elements=np.array([[0, 1, 2, 3]]),
                E=1.0, nu=0.0, thickness=1.0, elem_type="CPS4")
    monkeypatch.setattr(mesh, "element_kernel", _AsymKernel())
    monkeypatch.setattr(assembly_mod, "_check_symmetry", lambda K: K)
    # 本地 Ke 对称守卫 (第五轮外部审查新增) 同样绕过 —
    # 本测试只锁定 scatter 方向, 守卫本身由专测覆盖
    monkeypatch.setattr(
        assembly_mod, "_check_local_symmetry", lambda *a, **k: None)
    mesh.build_connectivity()

    Kd = assemble_sparse_vectorized(mesh).toarray()
    # Ke = arange(64) → Ke[j, k] = 8j + k; 单单元 dofs = [0..7], 全 8×8 锁定.
    for j in range(8):
        for k in range(8):
            assert Kd[j, k] == 8 * j + k, f"K[{j},{k}]={Kd[j,k]}, 期望 {8*j+k}"
