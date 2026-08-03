"""Element-kernel registry and built-in registrations."""
from .base import (
    ElementKernel,
    JacobianReport,
    get_element_kernel,
    register_element,
    registered_element_types,
)
from .cst import CST, CSTElement
from .q4 import Q4, Q4Element
from .q4i import Q4I, Q4IElement
from .q4r import Q4R, Q4RElement


def verify_all_elements(mesh, verbose=True):
    """Dispatch mesh verification to its active element kernel."""
    mesh.build_connectivity()
    return mesh.element_kernel.verify_mesh(mesh, verbose=verbose)


__all__ = [
    "CST",
    "Q4",
    "Q4I",
    "Q4R",
    "CSTElement",
    "ElementKernel",
    "JacobianReport",
    "Q4Element",
    "Q4IElement",
    "Q4RElement",
    "get_element_kernel",
    "register_element",
    "registered_element_types",
    "verify_all_elements",
]
