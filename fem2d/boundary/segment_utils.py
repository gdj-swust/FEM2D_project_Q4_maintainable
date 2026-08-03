"""Small, side-effect-free helpers for dictionary-backed boundary segments.

Boundary segments intentionally remain dictionaries for public compatibility.
This module centralizes the repetitive metadata access that previously spread
through topology, naming, reporting and CAD merging code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..regions import canonical_edge


def tuple_value(value):
    """Normalize a scalar/iterable metadata value to a tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def segment_info(segment):
    """Return a segment's metadata dictionary without mutating the segment."""
    return segment.get("info", {})


def segment_is_outer(segment):
    """Return the topology role; never infer it from presentation text."""
    return bool(segment_info(segment).get("is_outer", False))


def segment_loop_depth(segment):
    """Return explicit nesting depth with a compatibility fallback."""
    default = 0 if segment_is_outer(segment) else 1
    return int(segment_info(segment).get("loop_depth", default))


def segment_physical_names(segment):
    """Return exact Physical Curve memberships in deterministic order."""
    info = segment_info(segment)
    names = tuple_value(info.get("physical_names", ()))
    if not names and info.get("physical_name"):
        names = (info["physical_name"],)
    return tuple(sorted(map(str, names), key=str.casefold))


def segment_edges(segment):
    """Return the canonical mesh-edge set represented by one segment."""
    nodes = segment["nodes"]
    return {
        canonical_edge(a, b)
        for a, b in zip(nodes, nodes[1:])
    }


def segment_sort_key(segment):
    """Canonical user-facing segment order."""
    info = segment_info(segment)
    return (
        segment_loop_depth(segment),
        int(info.get("loop_id", -1)),
        -len(segment["nodes"]),
        segment.get("label", ""),
    )


def mesh_scale(nodes):
    """Characteristic in-plane span used by geometry classifiers."""
    nodes = np.asarray(nodes, dtype=float)
    return (
        float(np.ptp(nodes[:, 0]))
        + float(np.ptp(nodes[:, 1]))
    ) / 2.0


def deduplicate_consecutive_nodes(nodes):
    """Return integer node IDs with only consecutive duplicates removed."""
    result = []
    for raw_node in nodes:
        node = int(raw_node)
        if not result or node != result[-1]:
            result.append(node)
    return result


@dataclass(frozen=True)
class LoopContext:
    """Topology metadata shared by every edge in one semantic segment."""

    loop_id: int
    loop_depth: int
    is_outer: bool

    @classmethod
    def from_edges(cls, name, edges, edge_context):
        """Validate and combine per-edge contexts into one segment context."""
        missing = [edge for edge in edges if edge not in edge_context]
        if missing:
            raise ValueError(
                f"Boundary segment {name!r} contains edges with no topology "
                f"loop context: {missing[:5]}.")

        contexts = [edge_context[edge] for edge in edges]
        loop_ids = {int(item["loop_id"]) for item in contexts}
        loop_depths = {int(item["loop_depth"]) for item in contexts}
        outer_roles = {bool(item["is_outer"]) for item in contexts}
        if (
                len(loop_ids) != 1
                or len(loop_depths) != 1
                or len(outer_roles) != 1):
            raise ValueError(
                f"Boundary segment {name!r} crosses inconsistent topology "
                f"contexts: loops={sorted(loop_ids)}, "
                f"depths={sorted(loop_depths)}, "
                f"outer_roles={sorted(outer_roles)}.")
        return cls(
            loop_id=next(iter(loop_ids)),
            loop_depth=next(iter(loop_depths)),
            is_outer=next(iter(outer_roles)),
        )
