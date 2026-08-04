"""Validation rules shared by boundary import and semantic mapping."""
from collections import defaultdict

import numpy as np

# Segment 数据结构稳定化 (阶段 2): 段 schema 的必要键 —
# 类型 / 节点链 / 坐标 / 标签 / 参数 (info). closed 为可选键 (部分
# 路径不写).
_SEGMENT_REQUIRED_KEYS = ("type", "nodes", "coords", "label", "info")
_SEGMENT_TYPES = frozenset({"line", "arc", "ellipse", "curve"})


def validate_segment_schema(segments):
    """段 schema 稳定化校验 — 每个段必有 type/nodes/coords/label/info,
    type 为受控枚举, 边集合可经 nodes 推导.

    任何新识别器/合并器输出不满足此 schema → 立即报错, 不允许静默
    生成结构残缺的段 (识别输出是载荷输入链, 残缺段会被下游静默消费).
    """
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TypeError(
                f"边界段 {index} 不是 dict, 收到 "
                f"{type(segment).__name__}")
        missing = [
            key for key in _SEGMENT_REQUIRED_KEYS
            if key not in segment
        ]
        if missing:
            raise ValueError(
                f"边界段 {index} 缺 schema 键 {missing} — 段必须含 "
                f"type/nodes/coords/label/info")
        if segment["type"] not in _SEGMENT_TYPES:
            raise ValueError(
                f"边界段 {index} type={segment['type']!r} 不在 "
                f"受控枚举 {sorted(_SEGMENT_TYPES)}")
        if not isinstance(segment["info"], dict):
            raise TypeError(
                f"边界段 {index} info 必须是 dict, 收到 "
                f"{type(segment['info']).__name__}")
        nodes = segment["nodes"]
        if (
                len(nodes) < 2
                or not all(
                    isinstance(node, (int, np.integer))
                    for node in nodes)):
            raise ValueError(
                f"边界段 {index} nodes 必须为 ≥2 个整数节点 ID")
        if len(segment["coords"]) != len(nodes):
            raise ValueError(
                f"边界段 {index} coords 行数 {len(segment['coords'])} "
                f"≠ nodes 数 {len(nodes)}")
    return True


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
