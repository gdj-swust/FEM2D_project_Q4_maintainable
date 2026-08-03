"""共享数值校验 — 契约清账阶段 2 收敛点.

四种载荷/BC/材料参数校验模式的唯一实现:
  * 标量有限性 (E/nu/t/力值/位移值/压力/容差)
  * 正数 (E/thickness/lc/jump_ref/tol)
  * 开区间 (nu ∈ (-1, 0.5))

错误消息格式统一: ``<name>=<value> — 原因, 期望``。
禁止在调用方重复 np.isfinite — 非数值类型 (str/容器/None) 曾冒裸
TypeError ("ufunc 'isfinite' not supported"), 拒绝 NaN/Inf 与拒绝
非数值类型是同一契约的两面。
"""
import numbers

import numpy as np


def require_finite_scalar(value, name):
    """数值必须为有限标量 — 返回 float 规范化值.

    TypeError: 非数值类型 (str/容器/None/complex — bool 是 numbers.Real
    子类, 与历史 np.isfinite 语义一致接受 0.0/1.0)
    ValueError: NaN/Inf
    """
    if not isinstance(value, numbers.Real):
        raise TypeError(
            f"{name}={value!r} — must be a finite real number, "
            f"got {type(value).__name__}")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(
            f"{name}={value!r} — must be finite (NaN/Inf rejected)")
    return v


def require_finite_positive(value, name):
    """有限且 > 0 — 弹性模量/厚度/网格密度/参考应力/容差."""
    v = require_finite_scalar(value, name)
    if v <= 0.0:
        raise ValueError(f"{name}={value!r} — must be > 0")
    return v


def require_nu_valid(value, name="nu"):
    """泊松比开区间 (-1, 0.5) — 各向同性材料稳定性 (Bathe Table 4.3)."""
    v = require_finite_scalar(value, name)
    if not (-1.0 < v < 0.5):
        raise ValueError(
            f"{name}={value!r} — must be in (-1, 0.5) for "
            "isotropic material stability")
    return v
