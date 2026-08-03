"""Vectorized mesh-topology primitives (element-count independent of Python).

``Mesh.build_connectivity`` previously walked every element and every local
edge in interpreted Python, which dominated preprocessing time on meshes above
a few thousand elements.  All structures below are built with NumPy sorting and
segment reductions instead, so cost is one ``argsort`` per relation rather than
``O(n_element x nodes_per_element)`` interpreter steps.

The public containers keep list/dict semantics because the boundary subsystem,
stress recovery and the error estimator all consume them as ordinary Python
sequences.  ``CSRLists`` provides that interface over two flat arrays so the
per-node Python lists are never materialized at all.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class CSRLists:
    """Immutable ``list[list[int]]`` view over CSR-style offset/index arrays.

    ``lists[i]`` materializes one small Python list on demand; ``lists.ids(i)``
    returns the underlying integer array without a copy for numeric code.
    """

    __slots__ = ("_len", "flat", "ptr")

    def __init__(self, ptr: np.ndarray, flat: np.ndarray):
        self.ptr = np.asarray(ptr, dtype=np.int64)
        self.flat = np.asarray(flat, dtype=np.int64)
        self._len = int(self.ptr.size - 1)

    def __len__(self) -> int:
        return self._len

    def ids(self, index: int) -> np.ndarray:
        """Return the raw index array for row ``index`` (no copy)."""
        if index < 0:
            index += self._len
        if not 0 <= index < self._len:
            raise IndexError(index)
        return self.flat[self.ptr[index]:self.ptr[index + 1]]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._len))]
        return self.ids(int(index)).tolist()

    def __iter__(self):
        ptr, flat = self.ptr, self.flat
        for i in range(self._len):
            yield flat[ptr[i]:ptr[i + 1]].tolist()

    def __eq__(self, other):
        if isinstance(other, CSRLists):
            return (np.array_equal(self.ptr, other.ptr)
                    and np.array_equal(self.flat, other.flat))
        if isinstance(other, (list, tuple)):
            return len(other) == self._len and all(
                list(row) == self[i] for i, row in enumerate(other))
        return NotImplemented

    def __repr__(self) -> str:
        return f"CSRLists({self._len} rows, {self.flat.size} entries)"


def _group_by(keys: np.ndarray, n_rows: int, values: np.ndarray):
    """Return CSR (ptr, sorted values) grouping ``values`` by ``keys``."""
    counts = np.bincount(keys, minlength=n_rows)
    ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    order = np.lexsort((values, keys))
    return ptr, values[order]


def node_element_table(elements: np.ndarray, n_nodes: int) -> CSRLists:
    """Return node -> element ids, sorted ascending inside each node."""
    elements = np.asarray(elements, dtype=np.int64)
    n_elem, npe = elements.shape
    flat_nodes = elements.ravel()
    elem_ids = np.repeat(np.arange(n_elem, dtype=np.int64), npe)
    ptr, flat = _group_by(flat_nodes, n_nodes, elem_ids)
    return CSRLists(ptr, flat)


class EdgeTable:
    """Edge -> element incidence for a homogeneous element block.

    Attributes
    ----------
    lo, hi : (n_edge,) int64
        Canonical (min, max) node pair of every distinct mesh edge.
    counts : (n_edge,) int64
        Number of incident elements (1 = boundary, 2 = interior).
    owners : (n_edge, 2) int64
        Incident element ids; the second column is ``-1`` for boundary edges.
    """

    __slots__ = ("counts", "hi", "lo", "owners")

    def __init__(self, lo, hi, counts, owners):
        self.lo, self.hi = lo, hi
        self.counts, self.owners = counts, owners

    @property
    def n_edges(self) -> int:
        return int(self.lo.size)

    def boundary_mask(self) -> np.ndarray:
        return self.counts == 1

    def interior_mask(self) -> np.ndarray:
        return self.counts == 2

    def index_of(self, a: int, b: int) -> int:
        """Return the row index of edge ``(a, b)`` or ``-1``."""
        lo, hi = (a, b) if a <= b else (b, a)
        left = int(np.searchsorted(self.lo, lo, side="left"))
        right = int(np.searchsorted(self.lo, lo, side="right"))
        if left == right:
            return -1
        sub = self.hi[left:right]
        pos = int(np.searchsorted(sub, hi))
        if pos >= sub.size or sub[pos] != hi:
            return -1
        return left + pos

    def as_mapping(self) -> EdgeIncidence:
        """Return a lazy ``{(a, b): [element ids]}`` mapping view."""
        return EdgeIncidence(self)

    def as_dict(self) -> dict:
        """Return an eager ``{(a, b): [element ids]}`` dictionary."""
        return dict(self.as_mapping().items())


class EdgeIncidence(Mapping):
    """Read-only ``{(a, b): [element ids]}`` view over an :class:`EdgeTable`.

    Materializing the dictionary costs one Python list per mesh edge, which on
    large meshes is more expensive than the vectorized table itself.  Lookups
    here are two binary searches over the already-sorted canonical pairs, so
    the dictionary is never built unless a caller explicitly asks for it.
    """

    __slots__ = ("_table",)

    def __init__(self, table: EdgeTable):
        self._table = table

    @property
    def table(self) -> EdgeTable:
        return self._table

    def _owners(self, index: int) -> list:
        owners = self._table.owners[index]
        if self._table.counts[index] == 1:
            return [int(owners[0])]
        return [int(owners[0]), int(owners[1])]

    def __getitem__(self, key):
        try:
            a, b = key
        except (TypeError, ValueError):
            raise KeyError(key) from None
        index = self._table.index_of(int(a), int(b))
        if index < 0:
            raise KeyError(key)
        return self._owners(index)

    def __contains__(self, key):
        try:
            a, b = key
        except (TypeError, ValueError):
            return False
        return self._table.index_of(int(a), int(b)) >= 0

    def __len__(self):
        return self._table.n_edges

    def __iter__(self):
        lo = self._table.lo.tolist()
        hi = self._table.hi.tolist()
        return iter(zip(lo, hi))

    def items(self):
        counts = self._table.counts.tolist()
        first = self._table.owners[:, 0].tolist()
        second = self._table.owners[:, 1].tolist()
        for index, key in enumerate(self):
            yield key, ([first[index]] if counts[index] == 1
                        else [first[index], second[index]])

    def __repr__(self):
        return f"EdgeIncidence({self._table.n_edges} edges)"


def build_edge_table(elements: np.ndarray, local_edges, n_nodes: int) -> EdgeTable:
    """Build the canonical edge/element incidence table without Python loops."""
    elements = np.asarray(elements, dtype=np.int64)
    n_elem = elements.shape[0]
    local = np.asarray(local_edges, dtype=np.int64)
    if local.ndim != 2 or local.shape[1] != 2:
        raise ValueError(
            "local_edges must be a sequence of two-node index pairs; "
            f"got shape {local.shape}.")
    a = elements[:, local[:, 0]].ravel()
    b = elements[:, local[:, 1]].ravel()
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    elem_ids = np.repeat(np.arange(n_elem, dtype=np.int64), local.shape[0])

    # A single int64 key keeps the sort one-dimensional.  n_nodes is bounded by
    # the mesh validation in Mesh.__post_init__, so lo*n_nodes+hi cannot wrap.
    if n_nodes > 3_000_000_000:
        order = np.lexsort((hi, lo))
        key_sorted_changes = np.flatnonzero(
            (np.diff(lo[order]) != 0) | (np.diff(hi[order]) != 0)) + 1
    else:
        key = lo * np.int64(n_nodes) + hi
        order = np.argsort(key, kind="stable")
        key_sorted_changes = np.flatnonzero(np.diff(key[order]) != 0) + 1

    lo_s, hi_s, eid_s = lo[order], hi[order], elem_ids[order]
    starts = np.concatenate(([0], key_sorted_changes))
    stops = np.concatenate((key_sorted_changes, [lo_s.size]))
    counts = stops - starts

    over = np.flatnonzero(counts > 2)
    if over.size:
        index = int(over[0])
        edge = (int(lo_s[starts[index]]), int(hi_s[starts[index]]))
        raise ValueError(
            f"Non-manifold edge {edge} shared by {int(counts[index])} "
            "elements. 2D mesh edges must belong to exactly 1 or 2 elements.")

    owners = np.full((starts.size, 2), -1, dtype=np.int64)
    owners[:, 0] = eid_s[starts]
    two = counts == 2
    owners[two, 1] = eid_s[starts[two] + 1]
    # Deterministic ordering inside an interior edge.
    swap = two & (owners[:, 1] < owners[:, 0])
    owners[swap] = owners[swap][:, ::-1]
    return EdgeTable(lo_s[starts], hi_s[starts], counts.astype(np.int64), owners)


def element_neighbor_table(edges: EdgeTable, n_elem: int) -> CSRLists:
    """Return element -> face-adjacent element ids as a lazy list view."""
    interior = edges.interior_mask()
    e1 = edges.owners[interior, 0]
    e2 = edges.owners[interior, 1]
    rows = np.concatenate([e1, e2])
    vals = np.concatenate([e2, e1])
    ptr, flat = _group_by(rows, n_elem, vals)
    return CSRLists(ptr, flat)


class ElementLocator:
    """Uniform-bucket spatial index over element bounding boxes.

    The generic kernel fallback tested every element with an inverse isoparam
    mapping, i.e. ``O(n_element)`` per query.  Bucketing the axis-aligned
    bounding boxes reduces a query to the handful of elements whose box covers
    the point, which is what makes point probes, stress queries and
    convergence sampling usable on large meshes.
    """

    __slots__ = (
        "_tol",
        "elements",
        "flat",
        "inv_cell",
        "nodes",
        "origin",
        "ptr",
        "shape",
    )

    def __init__(self, nodes: np.ndarray, elements: np.ndarray,
                 max_cells: int = 4_000_000):
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=np.int64)
        coords = self.nodes[self.elements]
        lower = coords.min(axis=1)
        upper = coords.max(axis=1)

        low = self.nodes.min(axis=0)
        span = self.nodes.max(axis=0) - low
        # 尺度下限取坐标 ULP — 固定 1.0 会让纳米网格的容差 (1e-9) 远超
        # 单元尺寸, 包围盒 padding 吞掉整个网格, 点定位退化为全量搜索
        coord_ulp = 64.0 * np.finfo(float).eps * float(
            np.max(np.abs(self.nodes)))
        # 每轴 ULP — 零跨度轴 (共线/退化网格) 的 extent 曾回落绝对 1.0,
        # 微尺度退化网格分桶数随 1.0 发散 (审计 2026-08-03)
        axis_ulp = 64.0 * np.finfo(float).eps * np.max(
            np.abs(self.nodes), axis=0)
        scale = float(max(span.max(), coord_ulp))
        self._tol = 1e-9 * scale
        n_elem = self.elements.shape[0]

        # The cell size is strictly larger than the padded bounding box of the
        # largest element, so every element box straddles at most one cell
        # boundary per axis and therefore touches at most 2x2 cells.  Insertion
        # is then a fixed four-offset scatter with no per-element Python work.
        # The strict inequality matters: with cell == element extent, a box
        # whose face lies exactly on a cell boundary spans three cells once the
        # tolerance padding is applied.
        extent = np.where(span > 0.0, span, axis_ulp) + 4.0 * self._tol
        largest = (upper - lower).max(axis=0) + 4.0 * self._tol
        target = extent / max(np.sqrt(n_elem), 1.0)
        cell = np.maximum(np.maximum(largest, target), extent / 1.0e6)
        cell = np.where(cell > 0.0, cell, extent) * (1.0 + 1e-9) + self._tol
        counts = np.maximum(np.ceil(extent / cell).astype(np.int64), 1)
        while int(counts[0]) * int(counts[1]) > max_cells:
            cell *= 2.0
            counts = np.maximum(np.ceil(extent / cell).astype(np.int64), 1)
        nx, ny = int(counts[0]), int(counts[1])
        self.shape = (nx, ny)
        self.origin = low - 2.0 * self._tol
        self.inv_cell = 1.0 / cell

        limit = np.array([nx - 1, ny - 1], dtype=np.int64)
        i0 = np.clip(((lower - self._tol - self.origin)
                      * self.inv_cell).astype(np.int64), 0, limit)
        i1 = np.clip(((upper + self._tol - self.origin)
                      * self.inv_cell).astype(np.int64), 0, limit)
        gx = np.stack([i0[:, 0], i1[:, 0]], axis=1)
        gy = np.stack([i0[:, 1], i1[:, 1]], axis=1)
        cell_ids = (gx[:, :, None] * ny + gy[:, None, :]).reshape(n_elem, 4)
        elem_ids = np.repeat(np.arange(n_elem, dtype=np.int64), 4)
        flat_cells = cell_ids.ravel()
        order = np.lexsort((elem_ids, flat_cells))
        cells_sorted = flat_cells[order]
        elems_sorted = elem_ids[order]
        keep = np.ones(cells_sorted.size, dtype=bool)
        if cells_sorted.size > 1:
            keep[1:] = ~((np.diff(cells_sorted) == 0)
                         & (np.diff(elems_sorted) == 0))
        cells_sorted = cells_sorted[keep]
        elems_sorted = elems_sorted[keep]
        bucket_counts = np.bincount(cells_sorted, minlength=nx * ny)
        self.ptr = np.zeros(nx * ny + 1, dtype=np.int64)
        np.cumsum(bucket_counts, out=self.ptr[1:])
        self.flat = elems_sorted

    def candidates(self, x: float, y: float) -> np.ndarray:
        """Return element ids whose bucket covers ``(x, y)``.

        The bucket alone is sufficient given the cell-size guarantee above.
        An empty bucket for a point inside the global bounding box still
        triggers a 3x3 neighbourhood sweep, so a degenerate mesh can never turn
        a location query into a silent miss.
        """
        nx, ny = self.shape
        gx = int((x - self.origin[0]) * self.inv_cell[0])
        gy = int((y - self.origin[1]) * self.inv_cell[1])
        if 0 <= gx < nx and 0 <= gy < ny:
            cell = gx * ny + gy
            found = self.flat[self.ptr[cell]:self.ptr[cell + 1]]
            if found.size:
                return found
        blocks = []
        for ix in range(max(gx - 1, 0), min(gx + 2, nx)):
            for iy in range(max(gy - 1, 0), min(gy + 2, ny)):
                cell = ix * ny + iy
                blocks.append(self.flat[self.ptr[cell]:self.ptr[cell + 1]])
        if not blocks:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(blocks))

    def max_bucket_span(self) -> int:
        """Return the largest number of cells any element box occupies.

        Used by the test-suite to assert the 2x2 insertion invariant.
        """
        coords = self.nodes[self.elements]
        lower = coords.min(axis=1)
        upper = coords.max(axis=1)
        limit = np.array(self.shape, dtype=np.int64) - 1
        i0 = np.clip(((lower - self._tol - self.origin)
                      * self.inv_cell).astype(np.int64), 0, limit)
        i1 = np.clip(((upper + self._tol - self.origin)
                      * self.inv_cell).astype(np.int64), 0, limit)
        spans = (i1 - i0 + 1)
        return int((spans[:, 0] * spans[:, 1]).max())
