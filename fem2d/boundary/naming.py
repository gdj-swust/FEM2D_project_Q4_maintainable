"""边界命名 — 边名称解析、Physical Curve 映射、自动分类、验证"""
from collections import Counter, defaultdict

import numpy as np

from ..regions import canonical_edge, ordered_edge_chains
from .conic_merge import merge_compatible_cad_conics
from .model import BoundaryDiagnostics
from .physical_mapping import map_physical_edges
from .registry_mapping import map_region_registry
from .segment_builder import BoundarySegmentBuilder
from .segment_utils import (
    mesh_scale,
    segment_edges,
    segment_is_outer,
    segment_physical_names,
    segment_sort_key,
)
from .selectors import resolve_boundary_selector
from .topology import detect

_FATAL_CAD_CODES = frozenset({
    "cad_curve_empty",
    "cad_curve_unowned",
    "cad_curve_nonmanifold",
    "cad_curve_missing_edges",
    "cad_boundary_internal_mismatch",
    "cad_interface_exposed",
    "cad_entity_overlap",
    "cad_boundary_coverage_gap",
})
# 模块内使用的别名 (segment_is_outer 在下方多处引用)
_segment_is_outer = segment_is_outer

# ═══════════════════════════════════════════════════════════════
# 边界完整性验证
# ═══════════════════════════════════════════════════════════════

def validate_boundary_segments(mesh, segments):
    """验证边界分段覆盖了所有边界边, 每条边恰好一次."""
    mesh.build_connectivity()
    expected = {canonical_edge(a, b) for a, b in mesh.boundary_edges}

    counts = Counter()
    for seg in segments:
        ns = seg["nodes"]
        for a, b in zip(ns, ns[1:]):
            counts[canonical_edge(a, b)] += 1

    found = set(counts.keys())
    missing = expected - found
    extra = found - expected
    duplicated = {e: n for e, n in counts.items() if n > 1}

    if missing or extra or duplicated:
        raise ValueError(
            f"边界分段不完整: "
            f"missing={len(missing)} edges, "
            f"extra={len(extra)} edges, "
            f"duplicated={len(duplicated)} edges. "
            f"Missing: {sorted(missing)[:5]}... "
            f"Extra: {sorted(extra)[:5]}...")
    return True


def semantic_coverage(mesh, segments, diagnostics=None):
    """Summarize Physical Curve names and exact boundary-edge coverage."""
    mesh.build_connectivity()
    boundary_edges = {
        canonical_edge(a, b) for a, b in mesh.boundary_edges
    }
    names = set()
    covered_edges = set()
    for segment in segments:
        info = segment.get("info", {})
        raw_names = info.get("physical_names", ())
        physical_names = (
            (raw_names,) if isinstance(raw_names, str)
            else tuple(raw_names)
        )
        if not physical_names and info.get("physical_name"):
            physical_names = (info["physical_name"],)
        if not physical_names:
            continue
        names.update(str(name) for name in physical_names)
        covered_edges.update(
            canonical_edge(a, b)
            for a, b in zip(segment["nodes"], segment["nodes"][1:])
        )
    declared_names = (
        set(diagnostics.declared_physical_names)
        if diagnostics is not None else set(names))
    mapped_names = (
        set(diagnostics.mapped_physical_names)
        if diagnostics is not None else set(names))
    return {
        "physical_names": tuple(sorted(names, key=str.casefold)),
        "declared_physical_names": tuple(sorted(
            declared_names, key=str.casefold)),
        "mapped_physical_names": tuple(sorted(
            mapped_names, key=str.casefold)),
        "dropped_physical_names": (
            diagnostics.dropped_physical_names
            if diagnostics is not None else ()),
        "covered_edges": len(covered_edges & boundary_edges),
        "total_boundary_edges": len(boundary_edges),
    }


# ═══════════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════════

def print_segments(segments):
    """打印边界段摘要"""
    if not segments:
        print("  (未检测到边界)")
        return

    geo_desc = describe_geometry(segments)
    print(f"\n  [{geo_desc}]")
    print(f"  {len(segments)} 条边界:")

    for i, seg in enumerate(segments):
        tag = {"line": "━", "arc": "⌒", "curve": "~", "ellipse": "O"}.get(seg["type"], "?")
        info_str = ""
        if seg["type"] == "arc":
            cx, cy = seg["info"].get("center", (None, None))
            R = seg["info"].get("radius", 0)
            ang = seg["info"].get("angle", 0)
            if cx is not None and cy is not None:
                info_str = f"  |  中心: ({cx:.6g}, {cy:.6g})  R={R:.6g}  角度={np.degrees(ang):.1f}°"
        elif seg["type"] == "ellipse":
            cx, cy = seg["info"].get("center", (None, None))
            a = seg["info"].get("semi_major", 0)
            b = seg["info"].get("semi_minor", 0)
            if cx is not None and cy is not None:
                info_str = f"  |  中心: ({cx:.6g}, {cy:.6g})  a={a:.6g}  b={b:.6g}"
        elif seg["type"] == "curve":
            km = seg["info"].get("curvature_mean", 0)
            cv = seg["info"].get("curvature_cv", 0)
            Req = seg["info"].get("equivalent_radius", None)
            Rmin = seg["info"].get("R_min", None)
            Rmax = seg["info"].get("R_max", None)
            if Req is not None:
                info_str = f"  |  k_mean={km:.6g}  R_eq={Req:.6g}  CV={cv:.2f}"
            elif Rmin is not None and Rmax is not None:
                info_str = f"  |  k_mean={km:.6g}  R=[{Rmin:.6g},{Rmax:.6g}]  CV={cv:.2f}"
            else:
                info_str = f"  |  k_mean={km:.6g}  k_std={seg['info'].get('curvature_std',0):.6g}  CV={cv:.2f}"
        print(f"  [{i+1}] {tag} {seg['label']}  ({len(seg['nodes'])}点){info_str}")

    arcs = [i for i, s in enumerate(segments) if s["type"] == "arc"]
    lines = [i for i, s in enumerate(segments) if s["type"] == "line"]
    curves = [i for i, s in enumerate(segments) if s["type"] == "curve"]
    outer = [
        i for i, segment in enumerate(segments)
        if _segment_is_outer(segment)]
    inner = [
        i for i, segment in enumerate(segments)
        if not _segment_is_outer(segment)]

    if outer: print(f"  外边: {','.join(str(i+1) for i in outer)}")
    if inner: print(f"  内孔: {','.join(str(i+1) for i in inner)}")
    if lines: print(f"  直边: {','.join(str(i+1) for i in lines)}")
    if arcs:  print(f"  圆弧: {','.join(str(i+1) for i in arcs)}")
    if curves: print(f"  曲线: {','.join(str(i+1) for i in curves)}")


def describe_geometry(segments):
    """根据边界特征自动识别几何类型"""
    n = len(segments)
    lines = [s for s in segments if s["type"] == "line"]
    arcs = [s for s in segments if s["type"] == "arc"]
    outer = [s for s in segments if _segment_is_outer(s)]
    holes = [s for s in segments if not _segment_is_outer(s)]

    desc = []

    if n == 2 and len(arcs) == 2:
        desc.append("同心圆环")
    elif n == 4 and len(lines) == 4:
        desc.append("矩形板")
    elif n >= 4 and len(lines) >= 3 and len(arcs) >= 1:
        desc.append("带孔/槽矩形板")
    elif len(arcs) == 1 and len(lines) == 0:
        desc.append("圆板")
    elif len(outer) == 1 and len(holes) >= 1:
        desc.append("带孔板")
    elif len(lines) >= 6:
        desc.append("多边形板")
    else:
        desc.append("一般二维截面")

    if holes:
        desc.append("含内孔")
    if arcs:
        arc_info = []
        for s in arcs:
            R = s['info'].get('radius', 0)
            cx, cy = s['info'].get('center', (0, 0))
            tag = '外' if _segment_is_outer(s) else '内'
            arc_info.append(f'{tag}R{R:.2f}@{cx:.2f},{cy:.2f}')
        desc.append(' | '.join(arc_info))

    all_nodes = np.concatenate([s["coords"] for s in segments])
    x_r = all_nodes[:, 0].max() - all_nodes[:, 0].min()
    y_r = all_nodes[:, 1].max() - all_nodes[:, 1].min()
    desc.append(f"{x_r:.2f}×{y_r:.2f}m")

    return " | ".join(desc)


# ═══════════════════════════════════════════════════════════════
# 边名称解析 — CLI 用户输入 → 段索引
# ═══════════════════════════════════════════════════════════════

def parse_edge_name(name: str, segs: list):
    """Resolve a CLI boundary name to zero-based segment indices.

    Exact Physical Curve names always win.  Fuzzy lookup is opt-in through a
    leading ``~`` and rejects ambiguous names instead of widening a load or
    displacement constraint silently.
    """
    return resolve_boundary_selector(name, segs)


def _resolve_edge_indices(edge_str, segs):
    """解析边编号/别名 → 段索引列表"""
    if edge_str is None:
        return []
    s = str(edge_str).strip()
    if not s:
        return []
    if s.isdigit():
        idx = int(s) - 1
        return [idx] if 0 <= idx < len(segs) else []
    return parse_edge_name(s, segs)


# ═══════════════════════════════════════════════════════════════
# Physical Curve → 边界段重建
# ═══════════════════════════════════════════════════════════════

def segments_from_physical_curves(
        mesh, edge_labels, geo_path=None, *, edge_partitions=None,
        edge_metadata=None, diagnostics=None):
    """从边标签信息重建边界段

    两路数据源 (原 read_inp_edgesets 路径已随 Abaqus 输入口移除):
      1. *ELEMENT 段的 ELSET=Line{cid}    — 初始名称
      2. *ELSET 段的 Physical Curve 名称  — 覆盖前者 (主路径)

    Physical Curve 是 CAD 语义边界，优先级高于几何猜测。每个同名边组
    先按连通性重建成完整链，再整体分类；样条或渐变曲率曲线不会因为
    自动检测中的局部曲率极值而被切碎。
    """
    if edge_labels is None:
        edge_labels = {}
    edge_partitions = edge_partitions or {}
    edge_metadata = edge_metadata or {}
    diagnostics = (
        diagnostics if diagnostics is not None
        else BoundaryDiagnostics())
    if (
            not edge_labels
            and not edge_partitions
            and not getattr(edge_labels, "unmapped_records", ())):
        return None

    mapped = map_physical_edges(
        mesh,
        edge_labels,
        edge_partitions,
        geo_path,
        diagnostics,
    )
    named_edges = mapped.named_edges
    combo_names = mapped.combined_names
    boundary_edges = mapped.boundary_edges
    if not named_edges and not edge_partitions:
        return None

    scale = mesh_scale(mesh.nodes)
    auto_segs = detect(mesh)
    edge_context = {}
    for auto in auto_segs:
        auto_info = auto.get("info", {})
        is_outer = bool(auto_info.get("is_outer", False))
        for a, b in zip(auto['nodes'], auto['nodes'][1:]):
            edge = canonical_edge(a, b)
            edge_context[edge] = {
                'is_outer': is_outer,
                'loop_depth': int(auto_info.get(
                    "loop_depth", 0 if is_outer else 1)),
                'loop_id': int(auto_info.get("loop_id", -1)),
                'direction': (int(a), int(b)),
            }

    segments = []
    builder = BoundarySegmentBuilder(
        mesh,
        edge_context,
        edge_metadata,
        scale,
        segments,
    )

    covered = set()
    for name, partition in sorted(
            named_edges, key=lambda item: (item[0].casefold(), item[1])):
        edges = named_edges[(name, partition)]
        source_info = {
            "physical_names": combo_names[name],
            **builder.metadata_for_edges(edges),
        }
        for chain in ordered_edge_chains(edges):
            builder.append_chain(
                chain, name,
                source_info,
                # A real Gmsh entity is already the authoritative CAD split.
                allow_geometric_split=not bool(partition))
        covered.update(edges)

    # Preserve every unlabelled CAD entity as an exact hard segment too.
    cad_remainder = defaultdict(set)
    for edge, partition_value in edge_partitions.items():
        edge = canonical_edge(*edge)
        partition = tuple(sorted(map(int, partition_value)))
        if edge in boundary_edges and edge not in covered and partition:
            cad_remainder[partition].add(edge)
    for partition in sorted(cad_remainder):
        edges = cad_remainder[partition]
        source_info = {
            "source": "gmsh_cad+geometry",
            **builder.metadata_for_edges(edges),
        }
        for chain in ordered_edge_chains(edges):
            builder.append_chain(
                chain, "", source_info,
                allow_geometric_split=False)
        covered.update(edges)

    # Preserve normal automatic segmentation for boundary edges that have no
    # semantic Physical Curve. Only partially covered automatic segments need
    # to be rebuilt into residual chains.
    for auto in auto_segs:
        auto_edges = {
            canonical_edge(a, b)
            for a, b in zip(auto['nodes'], auto['nodes'][1:])
        }
        residual = auto_edges - covered
        if not residual:
            continue
        if residual == auto_edges:
            copied = dict(auto)
            copied['info'] = dict(auto.get('info', {}))
            copied['info']['source'] = 'automatic'
            segments.append(copied)
            continue
        for chain in ordered_edge_chains(residual):
            builder.append_chain(
                chain, "", {"source": "automatic"})

    segments.sort(key=segment_sort_key)
    for segment in segments:
        diagnostics.register_mapped(
            segment_physical_names(segment))
    return segments


def segments_from_region_registry(mesh, registry, diagnostics=None):
    """Build segments from exact Gmsh CAD and Physical Curve metadata.

    The registry mapper validates CAD ownership and reduces the Gmsh model to
    edge labels, hard entity partitions and provenance.  The generic chain
    builder then performs topology-aware ordering and geometry classification.
    """
    if registry is None or (
            not registry.curves
            and not getattr(registry, "cad_curves", ())):
        return None

    diagnostics = (
        diagnostics if diagnostics is not None
        else BoundaryDiagnostics())
    mapped = map_region_registry(mesh, registry, diagnostics)
    segments = segments_from_physical_curves(
        mesh,
        mapped.edge_labels,
        geo_path=None,
        edge_partitions=mapped.edge_partitions,
        edge_metadata=mapped.edge_metadata,
        diagnostics=diagnostics,
    )
    if segments is None:
        return None

    for segment in segments:
        info = segment.setdefault("info", {})
        names = segment_physical_names(segment)
        if not names:
            info.setdefault("source", "automatic")
            continue
        edges = segment_edges(segment)
        info.update({
            "source": "gmsh_api+cad+geometry",
            "physical_names": names,
            "physical_tags": tuple(sorted({
                tag for name in names
                for tag in mapped.physical_metadata[name][
                    "physical_tags"]
            })),
            "entity_tags": tuple(sorted({
                tag for edge in edges
                for tag in mapped.edge_entity_tags[edge]
            })),
            "cad_entity_types": tuple(sorted({
                kind for edge in edges
                for kind in mapped.edge_entity_types[edge]
            })),
        })
    return merge_compatible_cad_conics(mesh, segments)


def build_boundary_segments(
        mesh, *, registry=None, edge_labels=None, geo_path=None,
        diagnostics=None, strict=False):
    """Build one validated boundary model from topology and Gmsh semantics.

    The mesh topology is always authoritative for edge existence, connectivity,
    loop orientation, inner/outer depth and geometric classification. Gmsh
    Physical Curves are authoritative for names and exact edge membership.

    Priority:
      1. in-memory Gmsh ``RegionRegistry`` (full command-language semantics);
      2. edge-label map (原 ELSET 语义的等价接口, 已随 Abaqus 输入口移除);
      3. topology/geometry-only detection for unlabelled external meshes.

    Every returned boundary edge occurs exactly once. Overlapping Physical
    Curves are stored in ``info["physical_names"]`` instead of duplicating the
    integration edge.
    """
    diagnostics = (
        diagnostics if diagnostics is not None
        else BoundaryDiagnostics())
    segments = None
    if registry is not None and (
            registry.curves or getattr(registry, "cad_curves", ())):
        segments = segments_from_region_registry(
            mesh, registry, diagnostics=diagnostics)
    elif edge_labels is not None:
        segments = segments_from_physical_curves(
            mesh, edge_labels, geo_path=geo_path,
            diagnostics=diagnostics)
    if segments is None:
        segments = detect(mesh)
    validate_boundary_segments(mesh, segments)
    for segment in segments:
        diagnostics.register_mapped(
            segment.get("info", {}).get("physical_names", ()))
    # A complete API registry is a hard contract: every external mesh edge
    # must agree with exactly one active CAD boundary entity, while shared CAD
    # interfaces must remain internal.  Continuing after such a contradiction
    # would silently replace CAD semantics with geometric guessing.
    if any(
            issue.code in _FATAL_CAD_CODES
            for issue in diagnostics.errors):
        diagnostics.raise_for_errors()
    if strict:
        diagnostics.raise_for_errors()
    return segments
