"""
综合测试: 圆环 + 离心力 + 重力 + 内孔压力 + 外边界径向位移
================================================================
用法:
  python scripts/test_complex.py <几何.geo>
  python scripts/test_complex.py <your_mesh.geo>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import numpy as np

from fem2d import (
    Mesh,
    print_segments,
    solve,
)
from fem2d import (
    detect_boundaries as detect,
)
from fem2d import (
    estimate_error as ee,
)
from fem2d import (
    report_mesh_quality as qr,
)
from fem2d.gmsh_adapter import generate_from_geo
from fem2d.visualize import interactive_plot


def main():
    parser = argparse.ArgumentParser(description='FEM2D complex test: ring + loads')
    parser.add_argument('mesh', help='几何文件路径 (.geo)')
    args = parser.parse_args()

    # ═══ 网格: .geo → Gmsh API 网格化 (2026-08: 无 Abaqus 中间格式) ═══
    fp = args.mesh
    g = generate_from_geo(fp)
    coords, elems = g.nodes, g.elements
    mesh = Mesh(nodes=coords, elements=elems,
                E=210e9, nu=0.3, thickness=0.01, plane_type="stress",
                elem_type=g.elem_type)

    # ═══ 自动识别边界 ═══
    segs = detect(mesh)
    print_segments(segs)

    # ═══ 位移约束: 外边界径向扩 0.001m ═══
    # A2修复: 按 label 前缀筛选, 不再假设 segs[0]=外边 segs[1]=内孔
    outer_segs = [s for s in segs if s['label'].startswith('外边')]
    inner_segs = [s for s in segs if s['label'].startswith('内孔')]
    if not outer_segs or not inner_segs:
        print("  [ERROR] 无法通过 label 前缀识别外边/内孔 — 请检查边界检测输出")
        print(f"  可用 labels: {[s['label'] for s in segs]}")
        sys.exit(1)
    outer = outer_segs[0]
    inner = inner_segs[0]

    mesh.fix_nodes_func(outer["nodes"],
        lambda x, y: (0.001*x/np.sqrt(x**2+y**2), 0.001*y/np.sqrt(x**2+y**2)))
    print(f"  BC: 外边 径向+0.001m ({len(outer['nodes'])}节点)")

    # ═══ 面力: 内孔施加 1MPa 向外压力 ═══
    for a, b in zip(inner["nodes"], inner["nodes"][1:]):
        mx = (coords[a, 0]+coords[b, 0])/2
        my = (coords[a, 1]+coords[b, 1])/2
        r = np.sqrt(mx*mx+my*my)
        mesh.add_traction(int(a), int(b), 1e6*mx/r, 1e6*my/r)
    print(f"  面力: 内孔 1MPa向外 ({len(inner['nodes'])-1}段)")

    # ═══ 体力: 离心力(ω=50) + 重力 ═══
    def body_force(x, y):
        bx = 7800 * 50**2 * x
        by = 7800 * 50**2 * y - 78000
        return bx, by
    mesh.body_force = body_force
    print("  体力: 离心(ω=50) + 重力")

    # ═══ 集中力: 外边界顶部中点 1000N 向右 ═══
    top_node = outer["nodes"][len(outer["nodes"])//4]
    mesh.add_force(int(top_node), 1000.0, 0.0)
    print(f"  集中力: node{top_node} Fx=1000N")

    # ═══ 求解 ═══
    print(f"\n{mesh.info()}")
    qr(mesh)
    result = solve(mesh)
    ee(mesh, result)

    u2 = result["u"].reshape(-1, 2)
    mag = np.sqrt(u2[:, 0]**2+u2[:, 1]**2)
    vm = result["vm_stress"]

    print(f"\n  Max|U|={mag.max():.6e} m  Max vm={vm.max():.3e} Pa")
    print(f"  外边界|U|: {mag[outer['nodes']].mean():.6e} m (理论~0.001)")

    interactive_plot(mesh, result, 500)


if __name__ == "__main__":
    main()
