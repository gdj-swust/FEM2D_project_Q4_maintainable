"""Gmsh ``.geo`` execution with exact entity-to-mesh region mapping.

The adapter lets Gmsh parse its own command language. It never attempts to
reimplement loops, macros, Boolean operations, or CAD entity numbering.
Physical Points, Curves and Surfaces are converted directly into semantic
node, edge and element regions.
"""
from __future__ import annotations

import importlib
import os
import tempfile
from dataclasses import dataclass

import numpy as np

from .regions import (
    CadCurveRegion,
    CurveRegion,
    PointRegion,
    RegionRegistry,
    SurfaceRegion,
    canonical_edge,
)


class GmshUnavailableError(RuntimeError):
    """Raised when the optional Gmsh Python API cannot be imported."""


class GmshTopologyError(ValueError):
    """Raised when a generated mesh cannot be represented by FEM2D."""


@dataclass
class GmshImportResult:
    nodes: np.ndarray
    elements: np.ndarray
    elem_type: str
    node_tag_to_index: dict[int, int]
    element_tag_to_index: dict[int, int]
    regions: RegionRegistry
    output_path: str | None = None


def _load_gmsh_module():
    try:
        return importlib.import_module("gmsh")
    except (ImportError, OSError) as error:
        raise GmshUnavailableError(
            "The Gmsh Python API is not available. Install the `gmsh` "
            "package to recover full CAD semantics. Native executable mesh "
            "generation (import_msh) requires the API to read back .msh."
        ) from error


def _safe_geo_source(geo_path):
    """Strip standalone scripted Save + Mesh commands in a copy.

    消毒规则唯一实现在 scripts.gmsh_runner.sanitize_geo_source —
    曾双实现分叉。延迟 import: 避免包初始化顺序
    依赖项目根在 sys.path。
    """
    from scripts.gmsh_runner import sanitize_geo_source
    with open(geo_path, "r", encoding="utf-8", errors="ignore") as stream:
        original = stream.read()
    source = sanitize_geo_source(original)
    if source == original:
        return os.path.abspath(geo_path), None
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".fem2d-gmsh-api-source-",
        suffix=".geo",
        dir=os.path.dirname(os.path.abspath(geo_path)),
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(source)
    except Exception:
        if os.path.isfile(temporary_path):
            os.unlink(temporary_path)
        raise
    return temporary_path, temporary_path


def read_geo_curve_groups(geo_path, *, gmsh_module=None):
    """Return ``Physical Curve name -> final CAD entity tags`` via Gmsh.

    This deliberately reads only CAD/Physical Group metadata. Mesh generation
    and mesh connectivity remain owned by the generated ``.msh`` import.
    ``None`` means the optional API path was unavailable and the caller should
    use its text-parser fallback.
    """
    if not os.path.isfile(geo_path):
        return None
    temporary_geo = None
    owns_session = False
    try:
        gmsh_module = gmsh_module or _load_gmsh_module()
        initialized = bool(
            gmsh_module.isInitialized()
            if hasattr(gmsh_module, "isInitialized") else False)
        owns_session = not initialized
        if owns_session:
            gmsh_module.initialize()
        command_geo, temporary_geo = _safe_geo_source(geo_path)
        gmsh_module.open(command_geo)
        gmsh_module.model.geo.synchronize()

        groups = {}
        for dimension, physical_tag in gmsh_module.model.getPhysicalGroups():
            if int(dimension) != 1:
                continue
            name = str(gmsh_module.model.getPhysicalName(
                1, int(physical_tag))).strip()
            if not name:
                continue
            entities = gmsh_module.model.getEntitiesForPhysicalGroup(
                1, int(physical_tag))
            groups.setdefault(name, set()).update(
                int(entity) for entity in entities)
        return {
            name: sorted(entities)
            for name, entities in groups.items()
        }
    except Exception as error:
        import sys
        print(
            f"[Gmsh] CAD group read failed ({type(error).__name__}: {error}); "
            "falling back to .geo regex parser.", file=sys.stderr)
        return None
    finally:
        if temporary_geo and os.path.isfile(temporary_geo):
            os.unlink(temporary_geo)
        if owns_session and gmsh_module is not None:
            gmsh_module.finalize()


def _entity_type(model, dimension, tag):
    getter = getattr(model, "getEntityType", None)
    if getter is None:
        getter = model.getType
    return str(getter(int(dimension), int(tag)))


def _element_properties(mesh_api, element_type):
    properties = mesh_api.getElementProperties(int(element_type))
    if len(properties) < 6:
        raise GmshTopologyError(
            f"Gmsh returned incomplete properties for element type "
            f"{element_type}.")
    name, dimension, order, node_count, _, primary_node_count = properties[:6]
    return (
        str(name), int(dimension), int(order),
        int(node_count), int(primary_node_count),
    )


def _map_node_tags(node_tags, node_tag_to_index, context):
    mapped = []
    missing = []
    for tag in np.asarray(node_tags, dtype=np.int64).reshape(-1):
        index = node_tag_to_index.get(int(tag))
        if index is None:
            missing.append(int(tag))
        else:
            mapped.append(index)
    if missing:
        raise GmshTopologyError(
            f"{context} references Gmsh nodes absent from the 2-D mesh: "
            f"{missing[:8]}.")
    return mapped


def _physical_node_ids(
        mesh_api, dimension, physical_tag, node_tag_to_index):
    getter = getattr(mesh_api, "getNodesForPhysicalGroup", None)
    if getter is None:
        return []
    result = getter(int(dimension), int(physical_tag))
    node_tags = result[0] if isinstance(result, tuple) else result
    # A .geo file may contain Physical Groups on construction geometry that
    # does not participate in the 2-D displacement mesh. Preserve the group,
    # but only attach active displacement nodes to it.
    return [
        node_tag_to_index[int(tag)]
        for tag in np.asarray(node_tags, dtype=np.int64).reshape(-1)
        if int(tag) in node_tag_to_index
    ]


def _curve_edges_for_entity(
        mesh_api, entity_tag, node_tag_to_index):
    element_types, _, node_blocks = mesh_api.getElements(
        1, int(entity_tag))
    edges = []
    for element_type, node_block in zip(element_types, node_blocks):
        name, dimension, _, node_count, primary_count = _element_properties(
            mesh_api, element_type)
        if dimension != 1 or primary_count < 2:
            raise GmshTopologyError(
                f"Curve entity {entity_tag} contains unsupported "
                f"element type {name}.")
        connectivity = np.asarray(
            node_block, dtype=np.int64).reshape(-1, node_count)
        for element_nodes in connectivity:
            endpoints = [
                node_tag_to_index.get(int(tag))
                for tag in element_nodes[:2]
            ]
            if any(endpoint is None for endpoint in endpoints):
                continue
            a, b = endpoints
            edges.append(canonical_edge(a, b))
    return edges


def _surface_elements_for_entity(
        mesh_api, entity_tag, element_tag_to_index):
    _, element_blocks, _ = mesh_api.getElements(2, int(entity_tag))
    result = []
    missing = []
    for element_block in element_blocks:
        for tag in np.asarray(element_block, dtype=np.int64).reshape(-1):
            index = element_tag_to_index.get(int(tag))
            if index is None:
                missing.append(int(tag))
            else:
                result.append(index)
    if missing:
        raise GmshTopologyError(
            f"Surface entity {entity_tag} references elements absent from "
            f"the displacement mesh: {missing[:8]}.")
    return result


def _extract_regions(
        gmsh_module, node_tag_to_index, element_tag_to_index, coords=None,
        elements=None, elem_type=None):
    model = gmsh_module.model
    mesh_api = model.mesh
    registry = RegionRegistry(cad_boundary_complete=True)

    surface_candidates = set()
    curve_surface_occurrences = {}

    for dimension, physical_tag in model.getPhysicalGroups():
        dimension = int(dimension)
        physical_tag = int(physical_tag)
        if dimension not in (0, 1, 2):
            continue
        name = str(model.getPhysicalName(dimension, physical_tag)).strip()
        if not name:
            name = f"physical_{dimension}_{physical_tag}"
        entity_tags = tuple(sorted(
            int(tag) for tag in
            model.getEntitiesForPhysicalGroup(dimension, physical_tag)
        ))
        physical_nodes = set(_physical_node_ids(
            mesh_api, dimension, physical_tag, node_tag_to_index))

        if dimension == 0:
            node_ids = tuple(sorted(physical_nodes))
            if not node_ids and coords is not None and entity_tags:
                # 域内 Point 的构造节点被 _extract_mesh 剔除 (不在任何 2D
                # 单元内, 如孔心) — node_ids 曾恒空且无提示, 与边界点
                # 行为不一致; 回退最近位移节点并警告。
                # 域外 Physical Point 曾同样回退到最近节点, 集中力施加到
                # 完全错误的位置 — 域外必须拒绝 (node_ids 留空, 下游报错)
                try:
                    coords_arr = np.asarray(coords, dtype=float)
                    fallback = []
                    for entity_tag in entity_tags:
                        point_xy = np.asarray(
                            model.getValue(0, entity_tag, []),
                            dtype=float)[:2]
                        span = max(
                            float(np.ptp(coords_arr[:, 0])),
                            float(np.ptp(coords_arr[:, 1])),
                            np.finfo(float).tiny)
                        slack = span * 1e-6
                        inside = (
                            coords_arr[:, 0].min() - slack
                            <= point_xy[0]
                            <= coords_arr[:, 0].max() + slack
                            and coords_arr[:, 1].min() - slack
                            <= point_xy[1]
                            <= coords_arr[:, 1].max() + slack)
                        if not inside:
                            print(
                                f"  [WARN] Physical Point '{name}' 位于"
                                f"网格包围盒外 ({point_xy[0]:.6g}, "
                                f"{point_xy[1]:.6g}) — 拒绝映射, 施加该点"
                                f"集中力将报错")
                            continue
                        # 单元级判域 (与 input_source 一致): 孔心/凹域
                        # 缺口点在 AABB 内但不属于任何单元 — 曾回退到
                        # 最近节点, 集中力施加到材料域外位置 (静默错)
                        if elements is not None:
                            from .mesh import Mesh
                            from .stress import point_in_element
                            tmp = Mesh(
                                coords_arr, elements, elem_type=elem_type)
                            if point_in_element(
                                    tmp, point_xy[0], point_xy[1]) < 0:
                                print(
                                    f"  [WARN] Physical Point '{name}' 不在"
                                    f"材料域内 ({point_xy[0]:.6g}, "
                                    f"{point_xy[1]:.6g}) — 拒绝映射, 施加该"
                                    f"点集中力将报错 (如需载荷请选边界曲线)")
                                continue
                        dist2 = np.sum(
                            (coords_arr - point_xy) ** 2, axis=1)
                        fallback.append(int(np.argmin(dist2)))
                    if fallback:
                        node_ids = tuple(sorted(set(fallback)))
                        print(
                            f"  [WARN] Physical Point '{name}' 的构造节点不在"
                            f"位移网格中, 已回退到最近节点 {list(node_ids)}")
                except Exception:  # nosec B110 — 回退失败保持空, 由下游处理 (故意吞异常)
                    pass
            registry.points.append(PointRegion(
                name=name,
                physical_tag=physical_tag,
                entity_tags=entity_tags,
                node_ids=node_ids,
            ))
            continue

        entity_types = tuple(
            _entity_type(model, dimension, tag)
            for tag in entity_tags
        )
        if dimension == 1:
            edge_entities = []
            for entity_tag, entity_type in zip(
                    entity_tags, entity_types):
                for edge in _curve_edges_for_entity(
                        mesh_api, entity_tag, node_tag_to_index):
                    edge_entities.append((
                        int(edge[0]), int(edge[1]),
                        int(entity_tag), str(entity_type),
                    ))
            edges = {
                canonical_edge(a, b)
                for a, b, _, _ in edge_entities
            }
            physical_nodes.update(
                node for edge in edges for node in edge)
            registry.curves.append(CurveRegion(
                name=name,
                physical_tag=physical_tag,
                entity_tags=entity_tags,
                entity_types=entity_types,
                node_ids=tuple(sorted(physical_nodes)),
                edge_pairs=tuple(sorted(edges)),
                edge_entities=tuple(sorted(set(edge_entities))),
            ))
            continue

        element_ids = {
            element
            for entity_tag in entity_tags
            for element in _surface_elements_for_entity(
                mesh_api, entity_tag, element_tag_to_index)
        }
        oriented_boundaries = []
        for entity_tag in entity_tags:
            surface_candidates.add(int(entity_tag))
            for boundary_dimension, boundary_tag in model.getBoundary(
                    [(2, int(entity_tag))],
                    combined=False, oriented=True, recursive=False):
                if int(boundary_dimension) == 1:
                    oriented_boundaries.append(int(boundary_tag))
        registry.surfaces.append(SurfaceRegion(
            name=name,
            physical_tag=physical_tag,
            entity_tags=entity_tags,
            entity_types=entity_types,
            element_ids=tuple(sorted(element_ids)),
            oriented_boundary_entities=tuple(oriented_boundaries),
        ))

    # Physical Surfaces are optional in Gmsh.  Enumerate all active 2-D CAD
    # entities as a fallback so unlabelled curves still retain exact entity
    # boundaries and CAD types.
    get_entities = getattr(model, "getEntities", None)
    if get_entities is not None:
        surface_candidates.update(
            int(tag) for dimension, tag in get_entities(2)
            if int(dimension) == 2)

    for surface_tag in sorted(surface_candidates):
        # ``getEntities(2)`` is a CAD inventory, not an active-mesh
        # inventory.  Hidden/excluded construction surfaces may exist in the
        # model but have no displacement elements and must not create phantom
        # external boundaries.
        if not _surface_elements_for_entity(
                mesh_api, surface_tag, element_tag_to_index):
            continue
        boundary = model.getBoundary(
            [(2, int(surface_tag))],
            combined=False, oriented=True, recursive=False)
        for boundary_dimension, boundary_tag in boundary:
            if int(boundary_dimension) != 1:
                continue
            curve_surface_occurrences.setdefault(
                abs(int(boundary_tag)), set()).add(
                    (int(surface_tag), int(boundary_tag)))

    for entity_tag in sorted(curve_surface_occurrences):
        edges = tuple(sorted(set(_curve_edges_for_entity(
            mesh_api, entity_tag, node_tag_to_index))))
        # Keep empty records.  ``entity_tag`` came from the boundary of an
        # active 2-D surface, so silently dropping it would make the adapter
        # claim complete CAD coverage when Gmsh supplied no usable line mesh.
        # Boundary validation will report this as a semantic/topology error.
        registry.cad_curves.append(CadCurveRegion(
            entity_tag=int(entity_tag),
            entity_type=_entity_type(model, 1, entity_tag),
            node_ids=tuple(sorted({
                node for edge in edges for node in edge
            })),
            edge_pairs=edges,
            surface_occurrences=tuple(sorted(
                curve_surface_occurrences[entity_tag])),
        ))

    # 物理组/CAD 成功提取才宣称边界完整 — 曾无条件 True, MSH 2.x / 裸
    # 网格输入下完整性校验误判为覆盖完整, 掩盖静默降级。
    # 只用 registry.curves 会把"只有 Physical Surface、无 Physical Curve
    # 组"的常见用法静默降级: cad_curves 已由 getEntities(2)+getBoundary
    # 完整枚举, 完整性校验却被关掉 
    registry.cad_boundary_complete = bool(
        registry.curves or registry.cad_curves)
    return registry


def normalize_element_orientation(nodes, elements):
    """单元方向归一化: 全部翻转为 CCW (有向面积 > 0).

    Gmsh 单元方向跟随 Curve Loop 方向 — 顺时针几何 (cook_membrane /
    curved_beam 等随包 demo) 产出全部 CW 单元, 被校验误报"退化"、
    求解误报"inverted"。边界映射用无序边键, 节点顺序反转不影响
   。三角/四边均用前 3 节点有向面积判向
    (凸四边形两三角同号)。返回新数组, 不原地修改。
    """
    elements = np.asarray(elements)
    if elements.size == 0:
        return elements
    s = 0.5 * (
        (nodes[elements[:, 1], 0] - nodes[elements[:, 0], 0])
        * (nodes[elements[:, 2], 1] - nodes[elements[:, 0], 1])
        - (nodes[elements[:, 2], 0] - nodes[elements[:, 0], 0])
        * (nodes[elements[:, 1], 1] - nodes[elements[:, 0], 1]))
    flip = s < 0
    if np.any(flip):
        flipped = elements.copy()
        flipped[flip] = flipped[flip][:, ::-1]
        return flipped
    return elements


def _extract_mesh(gmsh_module, require_quads=False, plane_type="stress"):
    mesh_api = gmsh_module.model.mesh
    node_tags, coordinates, _ = mesh_api.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1, 3)
    if len(node_tags) == 0:
        raise GmshTopologyError("Gmsh generated no mesh nodes.")
    if len(node_tags) != len(coordinates):
        raise GmshTopologyError(
            "Gmsh node tags and coordinates have inconsistent lengths.")
    node_tag_to_index = {
        int(tag): index for index, tag in enumerate(node_tags)
    }
    nodes = coordinates[:, :2].copy()
    z_vals = coordinates[:, 2].copy() if coordinates.shape[1] >= 3 else None

    element_types, element_tag_blocks, node_blocks = mesh_api.getElements(
        2, -1)
    connectivities = []
    element_tag_to_index = {}
    nodes_per_element = set()
    next_element = 0
    for element_type, element_tags, node_block in zip(
            element_types, element_tag_blocks, node_blocks):
        name, dimension, order, node_count, primary_count = (
            _element_properties(mesh_api, element_type))
        if dimension != 2:
            continue
        if (
                order != 1 or node_count != primary_count
                or primary_count not in (3, 4)):
            raise GmshTopologyError(
                f"Unsupported Gmsh displacement element {name}: "
                f"order={order}, nodes={node_count}, "
                f"primary_nodes={primary_count}.")
        connectivity_tags = np.asarray(
            node_block, dtype=np.int64).reshape(-1, node_count)
        element_tags = np.asarray(
            element_tags, dtype=np.int64).reshape(-1)
        if len(connectivity_tags) != len(element_tags):
            raise GmshTopologyError(
                f"Element type {name} has inconsistent tag/connectivity "
                "counts.")
        mapped = np.array([
            _map_node_tags(
                row, node_tag_to_index, f"Element type {name}")
            for row in connectivity_tags
        ], dtype=int)
        connectivities.append(mapped)
        nodes_per_element.add(node_count)
        for offset, tag in enumerate(element_tags):
            element_tag_to_index[int(tag)] = next_element + offset
        next_element += len(element_tags)

    if not connectivities:
        raise GmshTopologyError(
            "Gmsh generated no supported first-order 2-D elements.")
    if len(nodes_per_element) != 1:
        raise GmshTopologyError(
            "Gmsh generated a mixed triangle/quad mesh; FEM2D requires "
            "one homogeneous element topology.")
    node_count = next(iter(nodes_per_element))
    if require_quads and node_count != 4:
        raise GmshTopologyError(
            "Quad mode requested, but triangle elements remain after "
            "recombination.")
    elements = np.vstack(connectivities)
    elements = normalize_element_orientation(nodes, elements)

    # getNodes() can include meshed construction entities that are unrelated
    # to the displacement domain. Keep only nodes referenced by 2-D elements
    # so they cannot introduce disconnected DOFs into the linear system.
    active_old_indices = np.unique(elements.reshape(-1))
    if len(active_old_indices) != len(nodes):
        old_to_new = np.full(len(nodes), -1, dtype=int)
        old_to_new[active_old_indices] = np.arange(
            len(active_old_indices), dtype=int)
        elements = old_to_new[elements]
        nodes = nodes[active_old_indices]
        node_tag_to_index = {
            tag: int(old_to_new[old_index])
            for tag, old_index in node_tag_to_index.items()
            if old_to_new[old_index] >= 0
        }
    # Z 检查: 仅验证活动2D节点的z坐标, 而非所有构造节点
    # 容差 = 相对模型跨度 ×1e-8 + 坐标 ULP 兜底 — 固定 1.0 下限在微米/纳米
    # 模型 (跨度 1e-9) 下放大到 1e-8, 翘曲 5 倍于模型的非平面网格被
    # 静默投影。
    if z_vals is not None and len(active_old_indices) > 0:
        active_z = z_vals[active_old_indices]
        z_range = float(np.ptp(active_z))
        span = max(float(np.ptp(nodes[:, 0])),
                   float(np.ptp(nodes[:, 1])))
        coord_ulp = 64.0 * np.finfo(float).eps * float(
            np.max(np.abs(nodes)))
        z_tol = max(span * 1e-8, coord_ulp)
        if z_range > z_tol:
            raise GmshTopologyError(
                "Active 2-D mesh nodes contain non-constant z-coordinates "
                f"(z range = {z_range:.3e}, model span = {span:.3e}, "
                f"tolerance = {z_tol:.3e}). "
                "FEM2D is a strict 2-D solver; project or re-export the mesh.")

    prefix = "CPE" if str(plane_type).lower() == "strain" else "CPS"
    elem_type = f"{prefix}{node_count}"
    regions = _extract_regions(
        gmsh_module, node_tag_to_index, element_tag_to_index, coords=nodes,
        elements=elements, elem_type=elem_type)
    regions.validate_indices(len(nodes), len(elements))
    return GmshImportResult(
        nodes=nodes,
        elements=elements,
        elem_type=elem_type,
        node_tag_to_index=node_tag_to_index,
        element_tag_to_index=element_tag_to_index,
        regions=regions,
    )


def _file_declares_physical_names(msh_path):
    """.msh 文件是否声明了 $PhysicalNames 段 (用于检测物理组静默丢失)."""
    try:
        with open(msh_path, "r", encoding="ascii", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped == "$PhysicalNames":
                    return True
                if stripped.startswith("$") and stripped != "$MeshFormat":
                    continue
        return False
    except OSError:
        return False


def import_msh(msh_path, *, require_quads=False, plane_type="stress"):
    """从 Gmsh 原生 .msh 文件导入网格 + CAD/物理组语义 (无需 Abaqus 中间格式).

    子进程 gmsh 网格化 .geo 后输出 .msh; 本函数用 Gmsh API 打开文件并复用
    ``_extract_mesh`` 提取节点/单元/物理组 — 与 ``generate_from_geo`` 的
    API 路径共享同一提取逻辑, 保证两条路径语义一致。
    """
    if not os.path.isfile(msh_path):
        # 文件不存在曾透传 gmsh 底层 "Unable to open file" — 前置明确
        # 报错 (与 generate_from_geo 的 FileNotFoundError 契约一致)
        raise FileNotFoundError(f"Mesh file not found: {msh_path}")
    gmsh_module = _load_gmsh_module()
    initialized = bool(
        gmsh_module.isInitialized()
        if hasattr(gmsh_module, "isInitialized") else False)
    owns_session = not initialized
    try:
        if owns_session:
            gmsh_module.initialize()
        gmsh_module.open(os.path.abspath(msh_path))
        result = _extract_mesh(
            gmsh_module, require_quads=require_quads, plane_type=plane_type)
        if (
                not result.regions.curves
                and not result.regions.surfaces
                and _file_declares_physical_names(msh_path)):
            # MSH 2.x 的 $Elements physical 字段与 $PhysicalNames 标签自相
            # 矛盾, 4.1 缺 $Entities 段同样使 gmsh 读回后物理组为空 —
            # 文件里声明的边名 (bottom/左端 等) 全部不可用, 曾静默丢失
            #
            print(
                "  [WARN] .msh 声明了 $PhysicalNames 但 gmsh 未能恢复物理组 — "
                "边名 BC (--fix 左端 / --traction bottom:...) 将不可用。"
                "请用 Gmsh 4.1 格式重新导出 (python gmsh.write('x.msh') "
                "或 CLI 不加 -format msh2)。")
        return result
    finally:
        if owns_session:
            gmsh_module.finalize()


def _write_atomic(gmsh_module, output_path):
    final_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(final_path)
    os.makedirs(output_dir, exist_ok=True)
    # 临时文件后缀必须与最终输出一致 — Gmsh 按扩展名推断格式,
    # .inp 后缀曾把原生 .msh 内容写成 Abaqus 格式 (评审发现).
    suffix = os.path.splitext(final_path)[1] or ".msh"
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".fem2d-gmsh-api-", suffix=suffix, dir=output_dir)
    os.close(descriptor)
    os.unlink(temporary_path)
    try:
        gmsh_module.option.setNumber("Mesh.SaveAll", 1)
        gmsh_module.write(temporary_path)
        if (
                not os.path.isfile(temporary_path)
                or os.path.getsize(temporary_path) == 0):
            raise GmshTopologyError(
                "Gmsh API did not create a non-empty mesh output file.")
        os.replace(temporary_path, final_path)
        return final_path
    finally:
        if os.path.isfile(temporary_path):
            os.unlink(temporary_path)


def generate_from_geo(
        geo_path, *, quad=False, output_path=None, plane_type="stress",
        gmsh_module=None):
    """Execute ``geo_path`` and return mesh arrays plus semantic regions.

    The optional ``gmsh_module`` argument is intended for deterministic tests.
    Normal callers leave it unset and use the installed Gmsh Python package.
    """
    if not os.path.isfile(geo_path):
        raise FileNotFoundError(f"Geometry file not found: {geo_path}")
    gmsh_module = gmsh_module or _load_gmsh_module()
    initialized = bool(
        gmsh_module.isInitialized()
        if hasattr(gmsh_module, "isInitialized") else False)
    owns_session = not initialized
    command_geo = os.path.abspath(geo_path)
    temporary_geo = None
    try:
        if owns_session:
            gmsh_module.initialize()
        command_geo, temporary_geo = _safe_geo_source(geo_path)
        gmsh_module.open(command_geo)
        # Existing project .geo files often contain `Mesh 2;`. Clear that
        # early mesh and regenerate once so API options and extracted regions
        # always refer to the same final mesh.
        clear_mesh = getattr(gmsh_module.model.mesh, "clear", None)
        if clear_mesh is not None:
            clear_mesh()
        if quad:
            gmsh_module.option.setNumber("Mesh.RecombineAll", 1)
            gmsh_module.option.setNumber("Mesh.Algorithm", 8)
        gmsh_module.model.mesh.generate(2)
        result = _extract_mesh(
            gmsh_module, require_quads=quad, plane_type=plane_type)
        if output_path:
            result.output_path = _write_atomic(
                gmsh_module, output_path)
        return result
    finally:
        if temporary_geo and os.path.isfile(temporary_geo):
            os.unlink(temporary_geo)
        if owns_session:
            gmsh_module.finalize()


