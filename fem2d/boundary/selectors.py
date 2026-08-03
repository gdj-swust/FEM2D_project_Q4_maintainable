"""Resolve CLI boundary selectors without mixing them with display code."""
from __future__ import annotations

import numpy as np

from .segment_utils import (
    segment_is_outer,
    segment_physical_names,
)

GEOMETRIC_ALIASES = {
    "左": "left",
    "右": "right",
    "上": "top",
    "顶": "top",
    "下": "bottom",
    "底": "bottom",
    "孔": "hole",
    "内孔": "hole",
}


class BoundarySelector:
    """Apply the documented selector precedence to boundary segments."""

    def __init__(self, segments):
        self.segments = segments

    def resolve(self, raw_name):
        """Return selected zero-based segment indices."""
        needle = str(raw_name).strip()
        if not needle:
            return []

        exact = self._exact_physical_names(needle)
        if exact:
            return exact
        exact = self._exact_visible_labels(needle)
        if exact:
            return exact
        if needle.startswith("~") and len(needle) > 1:
            return self._explicit_fuzzy_match(needle)

        alias = GEOMETRIC_ALIASES.get(
            needle, needle.lower())
        geometric = self._geometric_match(alias)
        if geometric:
            return geometric
        shortcut = self._shortcut_match(alias)
        if shortcut:
            return shortcut
        return self._numeric_match(alias)

    def _exact_physical_names(self, needle):
        if needle.isdigit():
            return []
        folded = needle.casefold()
        return [
            index
            for index, segment in enumerate(self.segments)
            if any(
                candidate.casefold() == folded
                for candidate in segment_physical_names(segment))
        ]

    def _exact_visible_labels(self, needle):
        if needle.isdigit():
            return []
        folded = needle.casefold()
        return [
            index
            for index, segment in enumerate(self.segments)
            if (
                segment.get("label", "").strip().casefold()
                == folded)
        ]

    def _explicit_fuzzy_match(self, needle):
        query = needle[1:].strip().casefold()
        matches = []
        candidate_names = set()
        for index, segment in enumerate(self.segments):
            names = segment_physical_names(segment)
            matching_names = {
                candidate
                for candidate in names
                if query in candidate.casefold()
            }
            label = segment.get("label", "")
            if matching_names or query in label.casefold():
                matches.append(index)
                candidate_names.update(matching_names or {label})
        if len(candidate_names) > 1:
            choices = ", ".join(sorted(
                candidate_names, key=str.casefold))
            raise ValueError(
                f"边界名称 '{needle}' 匹配到多个候选: {choices}. "
                "请使用完整 Physical Curve 名称。")
        return matches

    def _geometric_match(self, alias):
        if not self.segments:
            return []
        x_min, x_max, y_min, y_max = self._coordinate_bounds()
        results = []
        for index, segment in enumerate(self.segments):
            coordinates = segment["coords"]
            xs = coordinates[:, 0]
            ys = coordinates[:, 1]
            mean_x = float(np.mean(xs))
            mean_y = float(np.mean(ys))
            horizontal = (
                abs(np.ptp(ys)) < abs(np.ptp(xs)) * 0.3)
            vertical = (
                abs(np.ptp(xs)) < abs(np.ptp(ys)) * 0.3)
            if self._matches_position(
                    alias,
                    mean_x,
                    mean_y,
                    horizontal,
                    vertical,
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    segment,
            ):
                results.append(index)
        return results

    def _coordinate_bounds(self):
        coordinates = np.concatenate([
            segment["coords"] for segment in self.segments
        ])
        return (
            float(np.min(coordinates[:, 0])),
            float(np.max(coordinates[:, 0])),
            float(np.min(coordinates[:, 1])),
            float(np.max(coordinates[:, 1])),
        )

    @staticmethod
    def _matches_position(
            alias, mean_x, mean_y, horizontal, vertical,
            x_min, x_max, y_min, y_max, segment):
        if alias == "left":
            return (
                vertical
                and mean_x < x_min + 0.1 * (x_max - x_min))
        if alias == "right":
            return (
                vertical
                and mean_x > x_max - 0.1 * (x_max - x_min))
        if alias == "bottom":
            return (
                horizontal
                and mean_y < y_min + 0.1 * (y_max - y_min))
        if alias == "top":
            return (
                horizontal
                and mean_y > y_max - 0.1 * (y_max - y_min))
        if alias == "hole":
            return not segment_is_outer(segment)
        return False

    def _shortcut_match(self, alias):
        if alias == "a":
            return [
                index
                for index, segment in enumerate(self.segments)
                if segment.get("type") == "arc"
            ]
        if alias == "l":
            return [
                index
                for index, segment in enumerate(self.segments)
                if segment.get("type") == "line"
            ]
        if alias == "o":
            return [
                index
                for index, segment in enumerate(self.segments)
                if segment_is_outer(segment)
            ]
        return []

    def _numeric_match(self, alias):
        if not alias.isdigit():
            return []
        index = int(alias) - 1
        return (
            [index]
            if 0 <= index < len(self.segments)
            else []
        )


def resolve_boundary_selector(name, segments):
    """Functional façade for callers that do not need a selector object."""
    return BoundarySelector(segments).resolve(name)
