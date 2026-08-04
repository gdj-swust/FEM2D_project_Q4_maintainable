"""CST/Q4/Q4R Patch Test — Bathe §5.3.3 单元收敛性基本检验

Irons & Razzaque 分片检验:
  用不规则网格 (至少一个内部节点) 施加恒定应力场对应的位移边界条件。
  如果单元能精确恢复恒定应力，则通过检验。
  → 证明单元同时满足完备性和兼容性要求。

Bathe §5.3.3 (line 19754-19896):
  完备性: Σh_i = 1 ∀ (r,s)  →  刚体位移 + 常应变可精确表示
  兼容性: 邻接单元在公共边上位移连续

对平面应力问题需要测试三个独立常应力状态 (Bathe line 13673-13676):
  Test 1: σ_xx = 1, σ_yy = 0, τ_xy = 0  (单向拉伸)
  Test 2: σ_xx = 0, σ_yy = 1, τ_xy = 0  (垂直拉伸)
  Test 3: σ_xx = 0, σ_yy = 0, τ_xy = 1  (纯剪)

运行: python -m fem2d.patch_test
"""
import numpy as np

from .element import get_element_kernel
from .mesh import Mesh
from .solver import solve


def _gen_irregular_patch():
    r"""生成一个 4 单元的非规则 patch 网格 (含 1 个内部节点)

    节点布局 (类似 Bathe Fig 4.17 的不规则 patch):
           3
          /|\
         / | \
        4  0  2
         \ | /
          \|/
           1

    共 5 节点, 4 三角形单元, 内部节点 0。
    边界: 1-2, 2-3, 3-4, 4-1
    内部: 节点 0 被 4 个单元环绕
    """
    nodes = np.array([
        [0.3,  0.2],   # 0 (内部节点 — 偏离中心)
        [0.0, -1.0],   # 1
        [1.2, -0.3],   # 2
        [0.8,  1.2],   # 3
        [-0.9,  0.4],  # 4
    ], dtype=float)

    elements = np.array([
        [0, 1, 2],   # 下-右
        [0, 2, 3],   # 右-上
        [0, 3, 4],   # 上-左
        [0, 4, 1],   # 左-下
    ], dtype=int)

    return nodes, elements


def _gen_irregular_q4_patch():
    """生成 4 个不规则 Q4 单元组成的 patch，节点 4 为内部节点。"""
    nodes = np.array([
        [-1.0, -1.0], [0.05, -1.10], [1.10, -0.85],
        [-1.10, 0.05], [0.12, 0.08], [1.18, 0.18],
        [-0.92, 1.08], [0.02, 1.18], [1.08, 1.02],
    ], dtype=float)
    elements = np.array([
        [0, 1, 4, 3],
        [1, 2, 5, 4],
        [3, 4, 7, 6],
        [4, 5, 8, 7],
    ], dtype=int)
    return nodes, elements


def _identify_boundary_nodes(mesh):
    """识别边界节点 (至少属于一条边界边的节点)"""
    mesh.build_connectivity()
    bdy_nodes = set()
    for a, b in mesh.boundary_edges:
        bdy_nodes.add(a)
        bdy_nodes.add(b)
    return sorted(bdy_nodes)


def run_patch_test(E=210e9, nu=0.3, plane="stress", tol=1e-10,
                   verbose=True, elem_type="CPS3"):
    """运行 CST、Q4 或 Q4R 单元的分片检验 (Bathe §5.3.3)

    测试三个独立常应力状态，验证单元能够精确恢复。

    参数
    ----
    E : float — 杨氏模量
    nu : float — 泊松比
    plane : str — "stress" or "strain"
    tol : float — 允许容差
    verbose : bool — 打印详细结果

    返回
    ----
    dict: 包含各测试的通过状态和最大误差
    """
    is_q4 = get_element_kernel(elem_type).nodes_per_element == 4
    nodes, elements = (
        _gen_irregular_q4_patch() if is_q4 else _gen_irregular_patch())
    mesh = Mesh(nodes=nodes, elements=elements, E=E, nu=nu,
                thickness=1.0, plane_type=plane, elem_type=elem_type)

    # ── 三个常应力测试 ──
    # 对平面应力:
    #   ε_xx = (σ_xx - ν·σ_yy)/E,  ε_yy = (σ_yy - ν·σ_xx)/E
    #   γ_xy = τ_xy/G,  G = E/(2(1+ν))
    # 位移场: u_x = ε_xx·x + ½γ_xy·y,  u_y = ½γ_xy·x + ε_yy·y
    # + 刚体位移 (设为 0)

    if plane == "stress":
        G = E / (2 * (1 + nu))
        def _displacement(sx, sy, txy):
            ex = (sx - nu * sy) / E
            ey = (sy - nu * sx) / E
            gxy = txy / G
            return lambda x, y: (ex * x + 0.5 * gxy * y,
                                  0.5 * gxy * x + ey * y)
    else:  # plane strain
        # ε_zz = 0 → σ_zz = ν(σ_xx+σ_yy)
        # ε_xx = [(1-ν²)σ_xx - ν(1+ν)σ_yy]/E
        # ε_yy = [(1-ν²)σ_yy - ν(1+ν)σ_xx]/E
        # γ_xy = τ_xy/G
        G = E / (2 * (1 + nu))
        def _displacement(sx, sy, txy):
            ex = ((1 - nu**2) * sx - nu * (1 + nu) * sy) / E
            ey = ((1 - nu**2) * sy - nu * (1 + nu) * sx) / E
            gxy = txy / G
            return lambda x, y: (ex * x + 0.5 * gxy * y,
                                  0.5 * gxy * x + ey * y)

    tests = [
        ("σ_xx = 1 (uniaxial)",    1.0, 0.0, 0.0),
        ("σ_yy = 1 (transverse)",  0.0, 1.0, 0.0),
        ("τ_xy = 1 (pure shear)",  0.0, 0.0, 1.0),
    ]

    results = []
    all_passed = True

    for label, sx, sy, txy in tests:
        # 建立网格 (每次重置 BC)
        mesh = Mesh(nodes=nodes, elements=elements, E=E, nu=nu,
                    thickness=1.0, plane_type=plane, elem_type=elem_type)
        mesh.build_connectivity()
        bdy_nodes = _identify_boundary_nodes(mesh)

        # 施加精确位移 BC
        u_exact = _displacement(sx, sy, txy)
        for n in bdy_nodes:
            ux, uy = u_exact(*mesh.nodes[n])
            mesh.fix_node(n, "x", ux)
            mesh.fix_node(n, "y", uy)

        # 求解
        result = solve(mesh, method="elimination", verbose=verbose)
        stress_exact = np.array([sx, sy, txy])

        # ── 验证 ──
        u_num = result["u"].reshape(-1, 2)
        u_ana = np.array([u_exact(*mesh.nodes[n]) for n in range(mesh.n_nodes)])
        u_ref = np.max(np.abs(u_ana))  # 位移参考尺度
        u_error = np.max(np.abs(u_num - u_ana))
        # 无条件相对化 — 曾 u_ref<1e-15 时退回绝对误差 (1e-15 是绝对阈值),
        # 微尺度位移模型下单元相对误差 50% 也能通过分片检验 (静默错误)。常应力非零 → 参考尺度恒 > 0, tiny 仅防除零。
        u_rel = u_error / (u_ref + np.finfo(float).tiny)

        s_ref = np.max(np.abs(stress_exact))  # 应力参考尺度
        s_error = np.max(np.abs(result["stress"] - stress_exact))
        s_rel = s_error / (s_ref + np.finfo(float).tiny)

        # Q4 的代表应力可能掩盖单个积分点错误；分片检验必须覆盖全部
        # 2×2 Gauss 点。CST 也走同一检查（只有一个响应点时等价）。
        stress_qp = result.get("stress_qp")
        if stress_qp is None:
            qp_rel = s_rel
        else:
            qp_error = np.max(np.abs(stress_qp - stress_exact))
            qp_rel = qp_error / (s_ref + np.finfo(float).tiny)

        passed = u_rel < tol and s_rel < tol and qp_rel < tol
        if not passed:
            all_passed = False

        results.append({
            "test": label,
            "u_error": u_rel,
            "s_error": s_rel,
            "stress_qp_error": qp_rel,
            "passed": passed,
        })

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(
                f"  {status}  {label}:  |u|_rel={u_rel:.2e}  "
                f"|σ|_rel={s_rel:.2e}  |σ_qp|_rel={qp_rel:.2e}")

    if verbose:
        print(f"\n{'='*55}")
        if all_passed:
            print(f"  [OK] ALL PATCH TESTS PASSED — {elem_type} element "
                  f"verified (Bathe §5.3.3)")
        else:
            print("  [FAIL] SOME PATCH TESTS FAILED — check element formulation!")
        print(f"{'='*55}")

    return {"all_passed": all_passed, "tests": results, "tol": tol}


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  CST + Q4 + Q4R Patch Tests — Bathe §5.3.3")
    print(f"{'='*55}\n")
    for _elem_type in ("CPS3", "CPS4", "CPS4R"):
        print(f"\n  --- {_elem_type} ---")
        run_patch_test(verbose=True, elem_type=_elem_type)
