"""Abaqus 对比探针 — 固定坐标输出应力分量/主应力/von Mises.

用途: 云图对比看不出头绪时, 抛开显示口径直接比数字。
复用 runner 完整管线 (与终端 run.py 同一路径), 求解后在固定坐标
输出两种口径:
  element   单元代表应力 (对应本项目 isoband 栏 / Abaqus 单元值)
  recovered SPR 恢复点插值 (对应本项目 Gouraud 栏 / Abaqus 节点插值)

Abaqus 侧: Tools → Query → Probe values, 读同一坐标的
S11/S22/S12 (节点插值), 或查积分点值 (对应 element 列)。

用法 (与 run.py 相同的参数):
  python scripts/probe_abq_compare.py models/demo_complex.geo \
      --fix "@底部,@left" --body 0,-77000 \
      --traction "@顶部:0,2e6;@椭圆孔:1e6:n" --lc 0.008 --elem-type Q4I
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fem2d.runner import (
    _analyze_and_report,
    _apply_conditions,
    _build_model,
    _parse_cli_config,
    _resolve_input_guarded,
)
from fem2d.material import von_mises
from fem2d.spr import spr_recovery
from fem2d.stress import (
    principal_stresses,
    stress_probe,
)

# 探针点: 中域平滑区 + 孔/凹口附近 + 受压区 — 坐标经几何校验 (圆角矩形
# 2.0×1.5 圆角 0.3, 椭圆孔 a=0.4/b=0.2 中心 (0.3,0.15), 凹口 R=0.35 心 (0.8,-1.5))
PROBE_POINTS = [
    (-1.6, 0.5), (-1.2, 0.5), (-0.8, 0.5), (-0.4, 0.5), (0.0, 0.5),
    (0.4, 0.5), (0.8, 0.5), (1.2, 0.5), (1.6, 0.5),   # y=0.5 中线
    (0.72, 0.15), (0.72, 0.30), (0.58, 0.35),         # 孔右侧
    (0.30, 0.40), (0.30, -0.10),                      # 孔上/下方
    (0.55, -1.2), (1.2, -1.2),                        # 凹口附近
    (-1.5, -1.2), (-1.9, 0.5),                        # 受压/左固支附近
    (0.0, 1.45), (1.0, 1.45),                         # 顶边附近
]


def _fmt_row(x, y, e, r):
    """一行两口径: (x, y, e_σx, e_σy, e_τxy, e_s1, e_s2, e_vm, r_σx, ...)."""
    cells = [f"{x:.4f}", f"{y:.4f}"]
    for vec in (e, r):
        cells += [f"{v:.4e}" for v in vec]
    return "  ".join(cells)


def main(argv):
    config, exit_code = _parse_cli_config(argv)
    if config is None:
        return exit_code
    resolved, exit_code = _resolve_input_guarded(config)
    if resolved is None:
        return exit_code
    model = _build_model(config, resolved)
    _apply_conditions(config, model)
    result, _, _, _ = _analyze_and_report(config, model)
    mesh = model.mesh

    print("\n" + "=" * 120)
    print("  探针表 — element 列对应本项目 isoband / Abaqus 单元(积分点)值; "
          "recovered 列对应本项目 Gouraud / Abaqus 节点插值")
    print("  x  y  |  e_sx e_sy e_txy e_s1 e_s2 e_vm |  r_sx r_sy r_txy r_s1 r_s2 r_vm")
    print("=" * 120)
    # 行装配来自 fem2d.stress.stress_probe — 与交互探针同一来源,
    # 口径语义改动只改 stress.py 一处
    for x, y in PROBE_POINTS:
        try:
            e_row, r_row = stress_probe(mesh, result, x, y)
        except ValueError as err:
            print(f"  ({x:.4f}, {y:.4f}) 跳过: {err}")
            continue
        print(_fmt_row(x, y, e_row, r_row))

    # 全局最值 — 供与 Abaqus legend 最值对数; 节点场报节点坐标,
    # 单元场报质心 (mesh.nodes[单元号] 取到的是任意节点, 曾误报位置)
    stress_qp = result.get("stress_qp")
    spr = spr_recovery(mesh, stress_qp if stress_qp is not None else result["stress"])
    vm_n = von_mises(spr, mesh.plane_type, mesh.nu)
    s_n = principal_stresses(spr)
    e_s = principal_stresses(result["stress"])
    e_vm = np.asarray(result["vm_stress"])

    print("\n" + "=" * 120)
    print("  全局最值 (r_* 节点恢复场@节点; e_* 单元场@质心)")
    print("=" * 120)
    for name, arr, kind in (("r_vm", vm_n, "node"), ("r_s1", s_n[0], "node"),
                            ("r_s2", s_n[1], "node"), ("e_vm", e_vm, "centroid"),
                            ("e_s1", e_s[0], "centroid"), ("e_s2", e_s[1], "centroid")):
        is_max = "s1" in name or "vm" in name
        i = int(np.argmax(arr)) if is_max else int(np.argmin(arr))
        pos = mesh.nodes[i] if kind == "node" else mesh.centroids[i]
        where = f"@ 节点#{i}" if kind == "node" else f"@ 单元#{i} 质心"
        print(f"  {name}: {'max' if is_max else 'min'} {arr[i]:.4e} "
              f"{where} ({pos[0]:.4f}, {pos[1]:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
