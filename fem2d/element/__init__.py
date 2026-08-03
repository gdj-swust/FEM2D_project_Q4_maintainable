"""Element subpackage — kernel protocol, built-in types, and shape utilities.

Adding a new element type::

    1. Implement :class:`ElementKernel` in a new module (e.g. ``fem2d/element/q8.py``).
    2. Register it at module scope: ``MyElem = register_element(MyElemKernel())``.
    3. Import the module in this ``__init__.py`` so the kernel is available at import time.
"""

# ── Material helpers (backward compat — were in old element.py) ──
from ..material import D_matrix, von_mises

# ── Kernel protocol & registry ──
from .base import (
    ElementKernel,
    JacobianReport,
    evaluate_vector_field,
    get_element_kernel,
    register_element,
    registered_element_types,
)

# ── CST (constant-strain triangle) ──
from .cst import CST, CSTElement

# ── Q4 (bilinear quadrilateral) ──
from .q4 import Q4, Q4Element

# ── Q4I (incompatible modes / QM6, Taylor-Beresford-Wilson) ──
from .q4i import Q4I, Q4IElement

# ── Q4R (one-point integration + affine-projector hourglass control) ──
from .q4r import Q4R, Q4RElement

# ── Kernel dispatch ──
from .registry import verify_all_elements

__all__ = [
    # Material
    "D_matrix",
    "von_mises",
    # Kernel protocol & registry
    "ElementKernel",
    "JacobianReport",
    "evaluate_vector_field",
    "get_element_kernel",
    "register_element",
    "registered_element_types",
    # CST
    "CST",
    "CSTElement",
    # Q4
    "Q4",
    "Q4Element",
    # Q4R
    "Q4R",
    "Q4RElement",
    # Q4I
    "Q4I",
    "Q4IElement",
    # Dispatch
    "verify_all_elements",
]
