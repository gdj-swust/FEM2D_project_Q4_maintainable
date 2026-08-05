"""Finite-element kernel protocol and registry.

The solver operates on a homogeneous element block.  A kernel owns all
topology-specific and interpolation-specific operations; Mesh, assembly and
stress recovery only consume this protocol.

Adding a new displacement element therefore requires:

1. implement :class:`ElementKernel` in a new module;
2. register one kernel instance with :func:`register_element`.

The built-in registry contains CST (CPS3/CPE3/C2D3), full-integration Q4
(CPS4/CPE4), stabilized one-point Q4R (CPS4R/CPE4R), and incompatible-mode
Q4I (CPS4I/CPE4I) kernels. Further homogeneous element types can be added
without element-type branches in Mesh, assembly.py or stress.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .. import material

if TYPE_CHECKING:
    from ..mesh import Mesh


@dataclass(frozen=True)
class JacobianReport:
    """Summary of the Jacobian checks at every element integration point."""

    bad: np.ndarray
    inverted: int
    degenerate: int
    tolerance: float

    @property
    def ok(self) -> bool:
        return self.bad.size == 0


class ElementKernel(ABC):
    """Protocol implemented by a homogeneous 2-D displacement element."""

    name: str
    aliases: tuple[str, ...]
    nodes_per_element: int
    local_edges: tuple[tuple[int, int], ...]
    recovery_family: str | None = None

    @property
    def dofs_per_element(self) -> int:
        return 2 * self.nodes_per_element

    def matches(self, elem_type: str) -> bool:
        key = str(elem_type).strip().upper()
        return key == self.name.upper() or key in {
            alias.upper() for alias in self.aliases
        }

    @abstractmethod
    def build_geometry(self, nodes: np.ndarray,
                       elements: np.ndarray) -> dict[str, np.ndarray]:
        """Return standard and element-specific geometry caches."""

    @abstractmethod
    def stiffness_batch(self, mesh, element_slice=None) -> np.ndarray:
        """Return local stiffness matrices as ``(ne_batch, ndofe, ndofe)``.

        ``element_slice=None`` means all elements. Kernels should honor a
        slice/index array so large meshes can be assembled in bounded-memory
        chunks.
        """

    def stiffness(self, mesh, eid: int) -> np.ndarray:
        """Return one local stiffness matrix.

        Kernels may override this for a cheaper single-element implementation.
        """
        return self.stiffness_batch(mesh)[eid]

    def compute_response(self, mesh, u_e: np.ndarray):
        """Return representative ``(stress, strain, von_mises)`` arrays.

        Default: dA-weighted mean of the integration-point response
        (shared by the Q4-family kernels); single-sample kernels like CST
        override this with their direct evaluation.
        """
        # 防互递归: 默认实现与 response_at_quadrature 默认实现互相调用,
        # 第三方内核若两个都未覆盖会 RecursionError — 显式检测并报可读错误
        if type(self).response_at_quadrature is ElementKernel.response_at_quadrature:
            raise NotImplementedError(
                f"{type(self).__name__} must override response_at_quadrature "
                "or compute_response")
        stress_qp, strain_qp, dA = self.response_at_quadrature(mesh, u_e)
        area = np.sum(dA, axis=1)
        stress = np.sum(stress_qp * dA[:, :, None], axis=1) / area[:, None]
        strain = np.sum(strain_qp * dA[:, :, None], axis=1) / area[:, None]
        vm = material.von_mises(stress, mesh.plane_type, mesh.nu)
        return stress, strain, vm

    @abstractmethod
    def jacobian_determinants(self, mesh) -> np.ndarray:
        """Return det(J) samples shaped ``(n_element, n_sample)``."""

    def degeneracy_measure(self, mesh):
        """Optional per-element shape-degeneracy measure (unitless).

        0 = degenerate; larger = healthier. Convention: area / (longest
        side)² — scale-invariant. Return ``None`` to fall back to the
        detJ-scale test only (per-element detJ scaling is vacuous for
        single-sample kernels: ``|detJ| <= 1e-15*|detJ|`` never holds for
        nonzero detJ, so slender elements pass).
        """
        del mesh

    @abstractmethod
    def body_force_vector(self, mesh, eid: int,
                          body_force) -> np.ndarray:
        """Return the local consistent body-force vector."""

    def body_force_batch(self, mesh, body_force, element_slice=None):
        """Return batched consistent body forces, or ``None``.

        The default keeps compatibility with coordinate-dependent loads.
        Kernels override this for vectorized constant body forces.
        """
        del mesh, body_force, element_slice

    def _constant_body_force(self, body_force):
        """Return ``(bx, by)`` for constant body forces, or ``None``.

        Shared guard used by every batched kernel: coordinate-dependent
        loads (callables or mixed tuples) fall back to the per-element path.
        """
        if callable(body_force):
            return None
        if (
                not isinstance(body_force, (tuple, list))
                or len(body_force) != 2
                or any(callable(value) for value in body_force)):
            return None
        bx, by = float(body_force[0]), float(body_force[1])
        if not (np.isfinite(bx) and np.isfinite(by)):
            raise ValueError("Body force contains NaN/Inf")
        return bx, by

    @abstractmethod
    def shape_values_at(self, coords: np.ndarray, x: float, y: float,
                        tol: float = 1e-12) -> np.ndarray | None:
        """Return shape values for an interior point, or ``None`` if outside."""

    @abstractmethod
    def verify_mesh(self, mesh, verbose: bool = True) -> bool:
        """Run element completeness/rigid-body verification."""

    def recovery_quadrature(self, mesh, eid: int):
        """Return ``(N, dA)`` samples for recovery/error integration.

        ``N`` has shape ``(n_sample, nodes_per_element)`` and ``dA`` contains
        the corresponding physical-area weights. Kernels should override this
        when their interpolation is not represented by the centroid fallback.
        """
        N = np.full((1, self.nodes_per_element),
                    1.0 / self.nodes_per_element)
        return N, np.array([mesh.areas[eid]], dtype=float)

    def recovery_shape_matrix(self, mesh):
        """Return an element-independent recovery shape matrix, or ``None``.

        Kernels with the same recovery points in every element should
        override this hook so recovery can map every sample position in one
        vectorized operation. Third-party kernels with element-dependent
        rules remain supported through :meth:`recovery_quadrature`.
        """
        del mesh

    def recovery_weights(self, mesh):
        """Return all physical recovery weights ``(ne,n_sample)``, or ``None``.

        This is the batched counterpart of the ``dA`` returned by
        :meth:`recovery_quadrature`.
        """
        del mesh

    def response_at_quadrature(self, mesh, u_e: np.ndarray):
        """Return ``(stress_qp, strain_qp, dA_qp)``.

        Default: one representative centroid sample. Kernels whose stress
        varies internally must override this. (compute_response 默认实现
        与此方法互为引用 — 未覆盖任一者会由 compute_response 的互递归
        检测报 NotImplementedError, 而非 RecursionError。)
        """
        stress, strain, _ = self.compute_response(mesh, u_e)
        return (
            stress[:, None, :],
            strain[:, None, :],
            np.asarray(mesh.areas, dtype=float)[:, None],
        )

    def jacobian_report(self, mesh) -> JacobianReport:
        """Classify elements using all Jacobian samples provided by a kernel."""
        det_j = np.asarray(self.jacobian_determinants(mesh), dtype=float)
        if det_j.ndim == 1:
            det_j = det_j[:, None]
        if not np.all(np.isfinite(det_j)):
            # 超大坐标浮点溢出会产生 NaN/Inf — report.ok=True 后给出
            # 误导性刚体约束错误
            bad = np.flatnonzero(~np.isfinite(det_j))
            raise RuntimeError(
                f"{self.name} Jacobian determinants contain NaN/Inf "
                f"(element {int(bad[0]) // max(det_j.shape[1], 1)}) — "
                "坐标超出数值范围, 检查模型尺度")
        if det_j.shape[0] != mesh.n_elements:
            raise RuntimeError(
                f"{self.name} kernel returned Jacobians with shape "
                f"{det_j.shape}; expected first dimension {mesh.n_elements}.")
        # 逐单元尺度容差 (全局 max 容差会在多尺度模型误杀小单元)
        scale_per = np.max(np.abs(det_j), axis=1)
        tol_per = np.maximum(1e-15 * scale_per, np.finfo(float).tiny)
        inverted_mask = np.any(det_j < -tol_per[:, None], axis=1)
        degenerate_mask = (~inverted_mask
                           & np.any(np.abs(det_j) <= tol_per[:, None], axis=1))
        # 形状退化检查: 内核可选提供无量纲指标 (面积/最长边²),
        # 数学塌缩级 (< 1e-8) 判退化, 与单元绝对尺寸无关
        measure = self.degeneracy_measure(mesh)
        if measure is not None:
            measure = np.asarray(measure, dtype=float)
            if measure.shape[0] != mesh.n_elements:
                raise RuntimeError(
                    f"{self.name} kernel degeneracy_measure returned shape "
                    f"{measure.shape}; expected first dimension "
                    f"{mesh.n_elements}.")
            # 与 inverted 分类互斥: 反向单元只计入 inverted
            degenerate_mask |= (~inverted_mask & (measure < 1e-8))
        bad_mask = inverted_mask | degenerate_mask
        return JacobianReport(
            bad=np.flatnonzero(bad_mask),
            inverted=int(np.sum(inverted_mask)),
            degenerate=int(np.sum(degenerate_mask)),
            tolerance=float(np.max(tol_per)) if det_j.size else 0.0,
        )

    def find_containing_element(self, mesh: Mesh, x: float,
                                y: float) -> int:
        """Generic point location using the kernel's shape-function inverse."""
        mesh.build_connectivity()
        # 自然坐标容差 (无量纲) — 与模型尺度无关。传 1e-12×全局
        # span 时, 对 1e12 尺度的网格容差放大到 1, 重心坐标 [-0.5,0.75,0.75]
        # 的域外点被误判在单元内。需要物理残差容差的内核
        # (如 Q4 的牛顿迭代) 自己内部乘以局部单元尺度 — 单元内判断
        # 只用局部几何, 不随整个模型的跨度放宽。
        tol = 1e-10
        candidates = (
            mesh.locator.candidates(x, y)
            if mesh.locator is not None
            else np.arange(mesh.n_elements, dtype=np.int64)
        )
        for eid in candidates:
            conn = mesh.elements[eid]
            if self.shape_values_at(mesh.nodes[conn], x, y, tol) is not None:
                return int(eid)
        return -1


_REGISTRY: dict[str, ElementKernel] = {}


def register_element(kernel: ElementKernel) -> ElementKernel:
    """Register a kernel and all of its aliases.

    Duplicate keys are rejected unless they point to the same kernel instance;
    this prevents import order from silently changing element behavior.
    """
    if not isinstance(kernel, ElementKernel):
        raise TypeError("register_element expects an ElementKernel instance")
    keys: Iterable[str] = (kernel.name, *kernel.aliases)
    for raw_key in keys:
        key = str(raw_key).strip().upper()
        if not key:
            raise ValueError("Element type names cannot be empty")
        existing = _REGISTRY.get(key)
        if existing is not None and existing is not kernel:
            raise ValueError(
                f"Element type '{key}' is already registered by "
                f"{existing.__class__.__name__}.")
        _REGISTRY[key] = kernel
    return kernel


def get_element_kernel(elem_type: str) -> ElementKernel:
    """Resolve an Abaqus/internal element name to its registered kernel."""
    key = str(elem_type).strip().upper()
    kernel = _REGISTRY.get(key)
    if kernel is None:
        supported = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unsupported element type '{elem_type}'. "
            f"Registered element types: {supported or '(none)'}.")
    return kernel


def registered_element_types() -> tuple[str, ...]:
    """Return registered names and aliases in deterministic order."""
    return tuple(sorted(_REGISTRY))


def evaluate_vector_field(field, x: float, y: float) -> tuple[float, float]:
    """Evaluate a constant/callable two-component vector field.

    统一校验 (体力/面力/压力共用): callable 返回值必须为 (bx, by)
    二元组; 分量必须可转 float 且有限。错误带求值点坐标 —
    曾裸 IndexError/TypeError/静默 NaN (复测 2026-08-02)。
    """
    if callable(field):
        value = field(x, y)
        if isinstance(value, np.ndarray):
            ok = value.ndim == 1 and value.shape == (2,)
        else:
            ok = isinstance(value, (tuple, list)) and len(value) == 2
        if not ok:
            raise ValueError(
                f"载荷 callable 必须返回 (bx, by) 二元组, 在 "
                f"({x:.4g},{y:.4g}) 得到 {value!r}")
        bx, by = value[0], value[1]
    elif isinstance(field, (tuple, list)) and len(field) == 2:
        bx = field[0](x, y) if callable(field[0]) else field[0]
        by = field[1](x, y) if callable(field[1]) else field[1]
    else:
        bx, by = field[0], field[1]
    try:
        bx, by = float(bx), float(by)
    except (TypeError, ValueError):
        raise ValueError(
            f"载荷分量非数值: ({bx!r}, {by!r}) 在 ({x:.4g},{y:.4g})") from None
    if not (np.isfinite(bx) and np.isfinite(by)):
        raise ValueError(
            f"载荷分量 NaN/Inf 在 ({x:.4g},{y:.4g})")
    return bx, by
