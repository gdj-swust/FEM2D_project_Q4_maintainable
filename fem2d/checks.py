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


def require_dof_index_array(arr, name, n_dof=None, bool_error=TypeError):
    """DOF 索引数组 — 1-D 整数索引, 拒绝布尔掩码/NaN/非整数/越界/标量.

    布尔掩码曾 asarray(float) 折叠成 {0,1} 约束错 DOF。异常类型按
    调用方既有锁定行为: mesh → TypeError, bc → ValueError
    (``bool_error`` 参数; 两边都有判别性测试锁定, 不可统一)。
    其余错误一律 ValueError 带参数名。返回规范化 int64 数组;
    不查重复 (mesh 与 bc 的重复语义不同, 各自保留)。
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.dtype == np.bool_ or arr.dtype.kind == "b":
        raise bool_error(
            f"{name} must be integer DOF indices, not a boolean mask — "
            "pass np.flatnonzero(mask) or explicit integer indices")
    # str/object 数组尝试数值化 (mesh 旧行为 astype(float) 带上下文报错)
    if arr.dtype.kind not in ("i", "u", "f"):
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{name} must be numeric DOF indices, got dtype "
                f"{arr.dtype}: {error}") from None
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contain NaN/Inf")
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(arr == np.rint(arr)):
            raise ValueError(f"{name} must be integers, got {arr.tolist()}")
        arr = np.rint(arr).astype(np.int64)
    else:
        arr = arr.astype(np.int64)
    if n_dof is not None and np.any((arr < 0) | (arr >= n_dof)):
        raise ValueError(f"{name} out of range [0, {n_dof - 1}]")
    return arr
