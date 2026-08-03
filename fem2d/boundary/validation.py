"""Validation rules shared by boundary import and semantic mapping."""
from collections import defaultdict


def validate_physical_curve_names(names, diagnostics):
    """Reject names that the boundary selector grammar cannot represent."""
    spellings_by_fold = defaultdict(set)
    for raw_name in names:
        name = str(raw_name)
        spellings_by_fold[name.casefold()].add(name)

    for spellings in spellings_by_fold.values():
        if len(spellings) > 1:
            choices = ", ".join(sorted(
                spellings, key=str.casefold))
            diagnostics.add(
                "physical_name_case_collision",
                "error",
                "Physical Curve names differ only by case and cannot be "
                f"selected unambiguously: {choices}.",
            )

    for raw_name in names:
        _validate_one_name(str(raw_name), diagnostics)


def _validate_one_name(name, diagnostics):
    if not name or name != name.strip():
        diagnostics.add(
            "physical_name_whitespace",
            "error",
            f"Physical Curve name {name!r} is empty or has leading/"
            "trailing whitespace.",
            physical_name=name,
        )
    if name.isdecimal():
        diagnostics.add(
            "physical_name_numeric",
            "error",
            f"Physical Curve {name!r} is numeric and conflicts with the "
            "1-based boundary-segment selector.",
            physical_name=name,
        )
    if any(
            ord(character) < 32 or ord(character) == 127
            for character in name):
        diagnostics.add(
            "physical_name_control_character",
            "error",
            f"Physical Curve {name!r} contains a control character and "
            "cannot be selected safely.",
            physical_name=name,
        )
    if any(separator in name for separator in (",", ";", ":")):
        diagnostics.add(
            "physical_name_cli_delimiter",
            "error",
            f"Physical Curve {name!r} contains ',', ';' or ':' and "
            "cannot be represented safely by the current CLI boundary/"
            "traction grammar.",
            physical_name=name,
        )
