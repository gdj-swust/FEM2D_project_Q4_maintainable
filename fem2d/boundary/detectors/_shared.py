"""探测器共用判定原语 — 标签前缀/段标签/闭合圆锥曲线拟合门槛."""
from __future__ import annotations

from ..geometry import (
    _closed_conic_turn_limit_deg,
    fit_closed_ellipse,
    sharp_corner_indices,
)


def _prefix(is_outer: bool) -> str:
    return "外边 " if is_outer else "内孔 "


def _segment_label(
        coords, prefix: str, kind: str, extra: str = "") -> str:
    """构建稳定的用户可见基元标签 (与旧 classify 文本逐位一致)."""
    x0, y0 = map(float, coords[0])
    x1, y1 = map(float, coords[-1])
    base = f"{prefix}{kind}"
    if extra:
        base += f" {extra}"
    return f"{base} ({x0:.6g},{y0:.6g})→({x1:.6g},{y1:.6g})"


# ═══════════════════════════════════════════════════════════════
# 闭合圆锥曲线共用判定 (旧 _classify_closed_conic 的拟合+门槛部分)
# ═══════════════════════════════════════════════════════════════

def _fit_closed_conic(coords):
    """闭合链的整环圆锥拟合 + 平滑门槛 — 通过返回 (ellipse, fit_info),
    Circle/Ellipse 探测器按轴比各自接管. 门槛与旧实现逐位一致."""
    if len(coords) < 8:
        return None
    ellipse, fit_info = fit_closed_ellipse(coords)
    if not ellipse:
        return None
    if sharp_corner_indices(
            coords,
            angle_threshold_deg=_closed_conic_turn_limit_deg(
                ellipse, fit_info)):
        return None
    return ellipse, fit_info
