"""载荷 schema — 四种载荷的合法形状集中定义, 统一校验 (契约清账阶段 3 拆分).

   body_force:          None | callable | 恰好 2 个分量 (bx, by)
   surface_tractions:   普通面力 (tx, ty) 恰好 2 个分量
                        压力 (p,) 恰好 1 个标量 (is_pressure=True)
   concentrated_forces: (fx, fy) 恰好 2 个数值分量
分量 = 有限数值或 callable。整体 callable 的返回契约 (f(x,y)→(bx,by))
在真实 Gauss 点检查 — 预调用会误拒形心在材料域外的合法体力
(带孔/凹域模型), 见 element.base.evaluate_vector_field。

调用方: mesh.Mesh._validate_loads_state (求解前校验) 与
bc_apply._apply_body_force (程序化体力配置)。
"""
import numpy as np


def _load_component_ok(value):
    """载荷分量合法: callable 或可转 float 且有限."""
    if callable(value):
        return True
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _check_load_pair(value, field, allow_callable=True):
    """校验二元载荷 (体力/面力/集中力) 的容器形状与分量.

    多余/缺失分量、标量、任意非序列容器 → 带字段名和原始值的
    ValueError (禁止裸 IndexError/TypeError 冒出)。
    """
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(
                f"{field} must have exactly 2 components "
                f"(like (bx, by)), got {len(value)}: {value!r}")
        comps = tuple(value)
    elif isinstance(value, np.ndarray):
        # 0-d ndarray 的 len() 抛裸 TypeError — 先查 ndim 再查长度
        if value.ndim != 1 or value.shape[0] != 2:
            raise ValueError(
                f"{field} must have exactly 2 components "
                f"(like (bx, by)), got shape {value.shape}: {value!r}")
        comps = tuple(value)
    else:
        raise ValueError(
            f"{field} must be a 2-component tuple/list, "
            f"got {type(value).__name__}: {value!r}")
    for i, comp in enumerate(comps):
        if callable(comp):
            if not allow_callable:
                raise ValueError(
                    f"{field}[{i}] is a callable — force components must be "
                    f"finite numbers, got {comp!r}")
            continue
        if not _load_component_ok(comp):
            raise ValueError(
                f"{field}[{i}] = {comp!r} — must be a finite number or callable")
    return comps


def _check_load_scalar(value, field):
    """校验单分量载荷 (压力幅值) — 接受标量或 1 元素序列, 返回规范化 1 元组."""
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError(
                f"{field} must have exactly 1 component (pressure magnitude p), "
                f"got {len(value)}: {value!r}")
        comp = value[0]
    elif isinstance(value, np.ndarray):
        # 0-d ndarray 的 len() 抛裸 TypeError — 先查 ndim 再查长度
        if value.ndim != 1 or value.shape[0] != 1:
            raise ValueError(
                f"{field} must have exactly 1 component (pressure magnitude p), "
                f"got shape {value.shape}: {value!r}")
        comp = value[0]
    else:
        comp = value
    if not _load_component_ok(comp):
        raise ValueError(
            f"{field} = {comp!r} — must be a finite number or callable "
            f"(pressure magnitude)")
    return (comp,)
