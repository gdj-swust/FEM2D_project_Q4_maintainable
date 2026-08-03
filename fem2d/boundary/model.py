"""Boundary-model metadata and structured diagnostics.

Boundary labels are presentation text.  Algorithms must instead use the
explicit metadata carried by every segment and the issues collected here.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BoundaryIssue:
    """One recoverable or fatal boundary-import problem."""

    code: str
    severity: str
    message: str
    physical_name: str | None = None
    entity_tag: int | None = None
    edge_count: int = 0


@dataclass
class BoundaryDiagnostics:
    """Issues produced while joining FEM topology and Gmsh CAD semantics."""

    issues: list[BoundaryIssue] = field(default_factory=list)
    declared_physical_names: set[str] = field(default_factory=set)
    mapped_physical_names: set[str] = field(default_factory=set)

    def register_declared(self, names):
        self.declared_physical_names.update(
            str(name) for name in names if str(name))

    def register_mapped(self, names):
        self.mapped_physical_names.update(
            str(name) for name in names if str(name))

    def add(
            self, code, severity, message, *, physical_name=None,
            entity_tag=None, edge_count=0):
        severity = str(severity).lower()
        if severity not in {"warning", "error"}:
            raise ValueError(
                f"Unsupported boundary issue severity {severity!r}.")
        issue = BoundaryIssue(
            code=str(code),
            severity=severity,
            message=str(message),
            physical_name=(
                None if physical_name is None else str(physical_name)),
            entity_tag=(
                None if entity_tag is None else int(entity_tag)),
            edge_count=int(edge_count),
        )
        if issue not in self.issues:
            self.issues.append(issue)
        return issue

    @property
    def warnings(self):
        return tuple(
            issue for issue in self.issues
            if issue.severity == "warning")

    @property
    def errors(self):
        return tuple(
            issue for issue in self.issues
            if issue.severity == "error")

    @property
    def dropped_physical_names(self):
        issue_names = {
            issue.physical_name
            for issue in self.issues
            if issue.physical_name
            and issue.code in {
                "physical_curve_internal",
                "physical_curve_partly_internal",
                "physical_curve_empty",
                "physical_curve_unmeshed_entity",
                "physical_curve_missing_edges",
                "physical_curve_unmapped_nodes",
            }
        }
        return tuple(sorted(
            issue_names | (
                self.declared_physical_names
                - self.mapped_physical_names),
            key=str.casefold))

    def raise_for_errors(self):
        if not self.errors:
            return
        details = "\n".join(
            f"  - [{issue.code}] {issue.message}"
            for issue in self.errors)
        raise ValueError(
            "Boundary semantic validation failed:\n" + details)

    def summary(self):
        """返回诊断统计 (公开 API, 供集成方汇总报告)."""
        return {
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "declared_physical_names": tuple(sorted(
                self.declared_physical_names, key=str.casefold)),
            "mapped_physical_names": tuple(sorted(
                self.mapped_physical_names, key=str.casefold)),
            "dropped_physical_names": self.dropped_physical_names,
        }

