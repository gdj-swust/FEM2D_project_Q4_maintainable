"""边界金标准快照共享工具 (阶段 1 行为锁定基线).

快照内容 = 载荷输入链的全部依赖面:
  segments — 段集合 (type/label/nodes/info 规范化)
  edges    — 边界边集合 (canonical 排序)
  normals  — 段定向外法向 vs mesh.boundary_outward_normal 一致性
             (段方向翻转/重排 → 此处必红)
  print    — print_segments 打印输出 (CLI 用户看到的文本)

金标准文件提交入库 (tests/boundary_golden/)。测试运行时重新生成并
对比 (浮点圆整到 12 位有效数字 — 结构/标签/边集合逐位一致, 拟合
浮点值免疫跨平台 LAPACK 末位噪声); 环境变量 FEM2D_UPDATE_GOLDEN=1
重写金标准 (仅阶段 3 的快照更新步骤使用, 常规 CI 不设置)。
"""
from __future__ import annotations

import io
import json
import math
import os
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from fem2d.boundary.naming import print_segments

GOLDEN_DIR = Path(__file__).resolve().parent / "boundary_golden"
UPDATE_GOLDEN = os.environ.get("FEM2D_UPDATE_GOLDEN") == "1"


def canonical_edges(segments):
    """边界边集合的规范形式: 排序的 [a, b] 列表 (端点排序)."""
    edges = set()
    for seg in segments:
        for a, b in zip(seg["nodes"], seg["nodes"][1:]):
            edges.add(tuple(sorted((int(a), int(b)))))
    return sorted(edges)


def _canonical_value(value):
    """JSON 安全规范化: numpy 标量 → Python 标量; tuple/list 保序转 list;
    嵌套 dict 排序键; 浮点圆整到 12 位有效数字."""
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (np.ndarray,)):
        return [_canonical_value(item) for item in value.tolist()]
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        # 圆锥拟合 (lstsq) 跨平台 LAPACK 有末位 (1 ULP) 舍入差 —
        # 位级比较在 Linux CI vs Windows 本地必红 (实测 CI 首轮).
        # 12 位有效数字 = 语义精度 1e-12, 远超任何行为级差异
        # (段类型/标签/边集合变化在第 2-6 位), 同时免疫末位噪声.
        if math.isfinite(value):
            return float(f"{value:.12g}")
        return value
    return str(value)


def segments_snapshot(segments, *, include_nodes=True):
    """段集合快照: 排序键后的 JSON 安全列表."""
    result = []
    for seg in segments:
        entry = {
            "type": str(seg["type"]),
            "closed": bool(seg.get("closed", False)),
            "label": str(seg["label"]),
            "n_nodes": int(len(seg["nodes"])),
            "info": _canonical_value(seg.get("info", {})),
        }
        if include_nodes:
            entry["nodes"] = [int(node) for node in seg["nodes"]]
        result.append(entry)
    return result


def normals_snapshot(mesh, segments):
    """段定向外法向 vs mesh API 外法向一致性.

    段序定向下, 边 (a→b) 的外法向 = 切向顺时针 90° = (dy, -dx)/L
    (与 mesh.boundary_outward_normal 同一公式, 方向取段序). 逐位一致
    要求两者完全相同 — 段方向被翻转/重排时此处必红.

    返回 (entries, consistent): entries 每段一条
    {"edge": [a, b], "segment_normal": [nx, ny], "api_normal": [nx, ny]};
    consistent = 全部边逐位一致.
    """
    entries = []
    consistent = True
    for seg in segments:
        nodes = seg["nodes"]
        for a_raw, b_raw in zip(nodes, nodes[1:]):
            a, b = int(a_raw), int(b_raw)
            xa, ya = mesh.nodes[a]
            xb, yb = mesh.nodes[b]
            dx, dy = float(xb - xa), float(yb - ya)
            length = math.hypot(dx, dy)
            if length <= 0.0:
                raise AssertionError(
                    f"快照前置: 段 {seg['label']!r} 含零长边 ({a},{b})")
            seg_normal = [dy / length, -dx / length]
            api_normal = list(map(float, mesh.boundary_outward_normal(a, b)))
            # numpy 除法与 math.hypot 路径存在 1 ULP 级舍入差 → 位级比较
            # 过严 (实测 ellipse_fan 5/40 边末位差 1)。方向翻转则是 2 个
            # 量级的差异, 容差只容忍末位噪声.
            tol = 8.0 * np.finfo(float).eps
            match = all(
                abs(seg_value - api_value) <= tol
                for seg_value, api_value in zip(seg_normal, api_normal))
            consistent = consistent and match
            entries.append({
                "edge": [a, b],
                "segment_normal": seg_normal,
                "api_normal": api_normal,
            })
    return entries, consistent


def print_snapshot(segments):
    """print_segments 的完整打印输出 (CLI 用户可见文本)."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        print_segments(segments)
    return buffer.getvalue()


def build_snapshot(mesh, segments, *, include_nodes=True):
    """一个 (网格, 段集合) 对的完整金标准快照.

    include_nodes=False (gmsh 层) 时法向逐边明细含节点 ID, 跨 gmsh
    版本可漂移 → 只留计数 + 一致性 (一致性断言本身每次测试都实时执行).
    """
    edges = canonical_edges(segments)
    normal_entries, normals_consistent = normals_snapshot(mesh, segments)
    return {
        "segments": segments_snapshot(segments, include_nodes=include_nodes),
        "n_segments": len(segments),
        "boundary_edges": edges if include_nodes else len(edges),
        "n_boundary_edges": len(edges),
        "normals_consistent": normals_consistent,
        "normals": (
            normal_entries
            if include_nodes else len(normal_entries)),
        "print_output": print_snapshot(segments),
    }


def golden_path(name):
    """金标准文件路径 — name 形如 "demo_complex.registry.json"."""
    return GOLDEN_DIR / name


def render(data):
    """确定性 JSON 渲染 (排序键 + 中文直写 + 缩进)."""
    return json.dumps(
        data, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def compare_golden(name, data):
    """与入库金标准逐位对比; 不一致抛 AssertionError (判别性)."""
    path = golden_path(name)
    rendered = render(data)
    if UPDATE_GOLDEN:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        return
    if not path.exists():
        raise AssertionError(
            f"金标准缺失: {path} — 请先以 FEM2D_UPDATE_GOLDEN=1 生成")
    expected = path.read_text(encoding="utf-8")
    if expected != rendered:
        raise AssertionError(
            f"金标准不一致: {name} — 边界输出已漂移. "
            "修复实现或确认后以 FEM2D_UPDATE_GOLDEN=1 重写.")


def edge_coverage_check(mesh, segments):
    """每条网格边界边恰出现一次 (validate 之外的独立校验)."""
    from fem2d.regions import canonical_edge

    mesh.build_connectivity()
    expected = {
        canonical_edge(a, b) for a, b in mesh.boundary_edges
    }
    found = set()
    for seg in segments:
        for a, b in zip(seg["nodes"], seg["nodes"][1:]):
            edge = canonical_edge(a, b)
            if edge in found:
                raise AssertionError(f"边界边重复出现: {edge}")
            found.add(edge)
    if found != expected:
        raise AssertionError(
            f"边界边覆盖漂移: 多 {len(found - expected)} 缺 {len(expected - found)}")
    return True
