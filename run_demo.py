"""demo_complex — 椭圆孔内压 + 顶部压力 + 自重 (薄示例: 复用正式 API)

⚠️ 工况注意: 本脚本施加的是【顶部 1MPa 下压】, 与 models/demo_complex.spec
的【右端 1MPa 拉伸】是同一模型的不同工况 — 两者 max|u| 相差约 7.5%
(1.63e-5 vs 1.75e-5) 属正常差异。

边界识别直接复用正式 API (fem2d.boundary.build_boundary_segments),
不重复实现 Physical Curve 重建与孔洞角色推断 (曾各自实现, 且用
名称猜测"拓扑角色"违背架构原则)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 编码安全网: 非中文 Windows 下中文输出会 UnicodeEncodeError, 替换为 ?
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass
import numpy as np

from fem2d import (
    Mesh,
    print_segments,
    solve,
)
from fem2d.boundary import BoundaryDiagnostics, build_boundary_segments
from fem2d.errors import CliError
from fem2d.input_source import generate_geo_with_topology
from fem2d.visualize import interactive_plot

# ── 读网格: .geo → Gmsh → .msh → Gmsh API 导入 (2026-08: 无 Abaqus 中间格式) ──
GEO_PATH = "models/demo_complex.geo"
MSH_PATH = "models/demo_complex.msh"

try:
    msh_path, gmsh_import = generate_geo_with_topology(
        GEO_PATH, output_path=MSH_PATH, plane_type="stress")
except CliError as error:
    print(f"[ERROR] {error}")
    sys.exit(error.exit_code)
if not msh_path:
    print("[FATAL] Gmsh 网格化失败")
    sys.exit(1)

coords = gmsh_import.nodes
elems = gmsh_import.elements
elem_type = gmsh_import.elem_type or "CPS3"
mesh = Mesh(nodes=coords, elements=elems, E=2.1e11, nu=0.3,
            thickness=0.01, plane_type="stress", elem_type=elem_type)

# ── 正式边界模型 (物理组语义 + 拓扑角色, 与 run.py 同路径) ──
geo_path = GEO_PATH
segs = build_boundary_segments(
    mesh,
    registry=gmsh_import.regions,
    edge_labels=None,
    geo_path=geo_path,
    diagnostics=BoundaryDiagnostics(),
)
print_segments(segs)

# ── 约束: 左边 + 底部 (按标签匹配, 不硬编码索引) ──
FIX_NAMES = ['left', '左', '底部', 'bottom']
fixed_count = 0
for i, s in enumerate(segs):
    label_lower = s.get('label', '').lower()
    if any(name.lower() in label_lower for name in FIX_NAMES):
        for n in s["nodes"]:
            mesh.fix_node(int(n), "both", 0.0)
        print(f"  固定 [{i+1}] {s['label']}")
        fixed_count += 1
if fixed_count == 0:
    # 曾 WARN 后继续求解: 欠约束模型要么奇异矩阵报错 (引向错误方向),
    # 要么部分约束时静默给出错误结果 — 演示脚本是新手入口, 必须响亮
    # 失败
    print("  [FATAL] 未找到左边/底部边界 — 约束未施加. 可用分段:")
    for i, s in enumerate(segs):
        print(f"    [{i+1}] {s['type']} | {s.get('label', '')}")
    sys.exit(1)

# ── 载荷: 顶部 1MPa 向下 ──
for s in segs:
    if '顶部' in s.get('label', ''):
        ns = s["nodes"]
        for a, b in zip(ns, ns[1:]):
            mesh.add_traction(int(a), int(b), 0.0, -1e6)
        print("  面力 顶部: 0, -1e6")
        break
else:
    # 曾 WARN 后继续: 顶部载荷静默缺失, 求解成功但工况不是文档声明的
    # 工况
    print("  [FATAL] 未找到顶部边 — 面力未施加. 可用分段:")
    for i, s in enumerate(segs):
        print(f"    [{i+1}] {s['type']} | {s.get('label', '')}")
    sys.exit(1)

# ── 载荷: 椭圆孔内压 1MPa (法向压力, add_pressure 自动计算 t=-p·n) ──
pressure_found = False
for s in segs:
    if '椭圆孔' in s.get('label', '') or '内孔' in s.get('label', ''):
        pressure_found = True
        ns = s["nodes"]
        for a, b in zip(ns, ns[1:]):
            mesh.add_pressure(int(a), int(b), 1e6)
        print(f"  压力 椭圆孔: p=1e6 Pa, 法向 ({len(ns)} 节点)")
if not pressure_found:
    # 标签生成变化曾导致孔压静默消失, 求解成功但工况不是文档声明的
    # 工况
    print("  [FATAL] 未找到椭圆孔/内孔边界 — 孔压未施加. 可用分段:")
    for i, s in enumerate(segs):
        print(f"    [{i+1}] {s['type']} | {s.get('label', '')}")
    sys.exit(1)

# ── 体力: 自重 ──
mesh.body_force = (0.0, -78000.0)
print("  体力: 0, -78000")

# ── 求解 ──
print(f"\n{mesh.n_nodes} nodes  {mesh.n_elements} elems  {mesh.n_dof} DOFs")
result = solve(mesh)

u = result["u"]
vm = result["vm_stress"]
print(f"  最大位移: {np.max(np.abs(u)):.6e} m")
print(f"  最大 von Mises: {vm.max():.3e} Pa  ({vm.max()*1e-6:.2f} MPa)")

try:
    interactive_plot(mesh, result, scale=500)
except EOFError:
    print("\n[INFO] 非交互环境 (stdin 不可用), 跳过交互绘图")
