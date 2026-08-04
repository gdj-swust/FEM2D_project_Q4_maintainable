"""公共 API 误用输入 fuzz (复查轮, 可复用).

随机类型/形状/越界值/NaN/Inf/布尔/多余分量喂给全部公共 API:
  - 预期异常 (ValueError/TypeError/CliError) 带非空消息 → 正确行为
  - 预期异常空消息 / 其他异常 (裸 IndexError/KeyError/RuntimeError
    /OverflowError 等) 冒出 → bug (契约禁止)
  - 静默成功但输入非法 (已知误用模式) → bug
静默豁免按生成值过滤: 只有该值"确实合法" (契约允许) 才允许静默成功 —
complex/NaN/str/容器等非法类别照常断言必须抛异常 (曾把值类别参数
整体 silent_ok=True, 非法输入被静默接受也查不出来)。
用法: python scripts/fuzz_api.py [轮数=500]
退出码: 抓到 bug → 1。
"""
import os
import random
import sys

import numpy as np

# 脚本位于 scripts/ 下 — 审计必须针对本项目代码。editable install 指向
# 其他 worktree 时 sys.path 无 cwd, `python scripts/xxx.py` 会 import 到
# 外部 fem2d 副本 (曾静默测到旧实现, 数据失真)。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import fem2d as F
from fem2d.bc import apply_elimination, apply_penalty
from fem2d.config import AnalysisConfig
from fem2d.error_est import estimate as estimate_error
from fem2d.errors import CliError
from fem2d.loads_core import parse_traction, parse_vec2
from fem2d.material import D_matrix, von_mises
from fem2d.mesh import Mesh
from fem2d.solver import estimate_condition, solve
from fem2d.spr import spr_recovery
from fem2d.stress import (
    compute_stresses,
    nodal_L2_projection,
    nodal_average,
    nodal_simple,
    point_in_element,
    principal_stresses,
    stress_at_point,
)

# 预期异常: 契约允许的输入拒绝方式 (带诊断消息). 其他一律 unexpected —
# 曾把全部非 BARE 异常当成功忽略, RuntimeError/OverflowError 被静默放过.
EXPECTED = (ValueError, TypeError, CliError)


def _classify(exc):
    """异常分类: 预期类型带非空消息 → None (正确行为); 其余 → 报告文本."""
    if isinstance(exc, EXPECTED):
        if str(exc).strip():
            return None
        return f"{type(exc).__name__} 空消息 (无诊断)"
    return f"unexpected {type(exc).__name__}: {str(exc)[:60]}"

NODES = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
ELEMS = np.array([[0, 1, 3], [0, 3, 2]])


def _mesh():
    return Mesh(NODES, ELEMS, E=1e6, nu=0.3, thickness=1.0)


def _solved():
    m = _mesh()
    for i in range(4):
        m.fix_node(i, "both", 0.0)
    m.add_force(2, 1.0, 0.0)
    return m, solve(m, verbose=False)


def _rand_value(rng):
    """随机标量/容器值."""
    kind = rng.randrange(14)
    if kind == 0:
        return None
    if kind == 1:
        return rng.choice([True, False])
    if kind == 2:
        return rng.choice([0, 1, -1, 5, -5, 100])
    if kind == 3:
        return rng.choice([0.0, 1.0, -1.0, 1e-310, 1e150, -1e150])
    if kind == 4:
        return float("nan")
    if kind == 5:
        return float("inf")
    if kind == 6:
        return -float("inf")
    if kind == 7:
        return rng.choice(["abc", "1e6,0", "", "1e6x,0", "nan,0"])
    if kind == 8:
        return 1 + 2j
    if kind == 9:
        return [1.0]
    if kind == 10:
        return (0.5, 1.5)
    if kind == 11:
        return np.array([1.0, 2.0, 3.0])
    if kind == 12:
        return np.array([True, False, True])
    if kind == 13:
        return np.array([0.5])
    return 0


def _is_real(v):
    """有限实数标量 — bool 是数值 (值参数接受 True/False 为 0/1)."""
    return isinstance(v, (int, float)) and np.isfinite(v)


def _valid_nid(v):
    """合法节点索引: 非 bool 的整数 0..3 (网格 4 节点, "恰为整数"
    浮点接受; 数组内 nid 2/3 也是合法节点, 不能只认 {0,1})."""
    return (not isinstance(v, bool) and isinstance(v, (int, float))
            and float(v).is_integer() and 0 <= float(v) < 4)


def _valid_nid_list(v):
    """合法节点列表: 非空 1-D 容器且元素全部是合法 nid."""
    if isinstance(v, np.ndarray):
        if v.ndim != 1:
            return False
        values = v.tolist()
    elif isinstance(v, (list, tuple)):
        values = list(v)
    else:
        return False
    return bool(values) and all(_valid_nid(e) for e in values)


def _flat2(v):
    """(2,) 一维数值容器 — spr_recovery 单分量恢复的合法输入."""
    if isinstance(v, np.ndarray):
        return v.ndim == 1 and v.shape[0] == 2
    return isinstance(v, (list, tuple)) and len(v) == 2


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rng = random.Random(20260803)
    bugs = []
    calls = 0

    def feed(name, fn, silent_ok=False):
        """非法输入必须报领域错误 — 正常返回且 silent_ok=False 记为
        "静默成功" bug (曾只查异常, 非法输入被静默接受不报告).
        silent_ok 由调用方按生成值过滤: 仅该值确实合法时放行."""
        nonlocal calls
        calls += 1
        try:
            fn()
        except Exception as exc:
            issue = _classify(exc)
            if issue:
                bugs.append(f"{name}: {issue}")
        else:
            if not silent_ok:
                bugs.append(f"{name}: 非法输入静默成功 (无异常) — 应为领域错误")

    for _ in range(rounds):
        v = _rand_value(rng)
        i = rng.randrange(30)
        if i == 0:
            feed(f"fix_node({v!r})", lambda v=v: _mesh().fix_node(v, "both", 0.0),
                 silent_ok=_valid_nid(v))  # 仅 0/1 是合法 nid
        elif i == 1:
            feed(f"fix_node value({v!r})", lambda v=v: _mesh().fix_node(0, "both", v),
                 silent_ok=_is_real(v))  # 有限实数 (含负/零/bool) 合法
        elif i == 2:
            feed(f"add_force({v!r})", lambda v=v: _mesh().add_force(v, 1.0, 0.0),
                 silent_ok=_valid_nid(v))  # 仅 0/1 是合法 nid
        elif i == 3:
            feed(f"add_pressure({v!r})", lambda v=v: _mesh().add_pressure(0, 1, v),
                 silent_ok=_is_real(v))  # 有限实数 (负=反向) 合法
        elif i == 4:
            feed(f"fix_nodes_func({v!r})", lambda v=v: _mesh().fix_nodes_func(v, 0.0),
                 silent_ok=_valid_nid_list(v))  # 元素全为合法 nid 的列表
        elif i == 5:
            feed(f"nodes_on_edge({v!r})", lambda v=v: _mesh().nodes_on_edge(v, "min"),
                 silent_ok=False)  # 生成值恒非 "x"/"y", 全部必须报错
        elif i == 6:
            feed(f"solve({v!r})", lambda v=v: solve(v, verbose=False))
        elif i == 7:
            feed(f"estimate_condition({v!r})", lambda v=v: estimate_condition(np.eye(4), method=v))
        elif i == 8:
            feed(f"estimate_error({v!r})", lambda v=v: estimate_error(_mesh(), v))
        elif i == 9:
            feed(f"principal({v!r})", lambda v=v: principal_stresses(v))
        elif i == 10:
            feed(f"von_mises({v!r})", lambda v=v: von_mises(v))
        elif i == 11:
            feed(f"D_matrix({v!r})", lambda v=v: D_matrix(v, 0.3),
                 silent_ok=_is_real(v) and v > 0)  # 合法 E: 有限正数
        elif i == 12:
            feed(f"compute_stresses({v!r})", lambda v=v: compute_stresses(_mesh(), v))
        elif i == 13:
            feed(f"nodal_average({v!r})", lambda v=v: nodal_average(_mesh(), v))
        elif i == 14:
            feed(f"spr_recovery({v!r})", lambda v=v: spr_recovery(_mesh(), v),
                 silent_ok=_flat2(v))  # (2,) 一维 → 单分量恢复是设计
        elif i == 15:
            feed(f"parse_vec2({v!r})", lambda v=v: parse_vec2(v),
                 silent_ok=isinstance(v, str) and v == "1e6,0")  # 唯一合法格式
        elif i == 16:
            feed(f"parse_traction({v!r})", lambda v=v: parse_traction(v),
                 silent_ok=isinstance(v, str))  # 无冒号 → (None,0,0,None) 契约
        elif i == 17:
            feed(f"apply_penalty penalty({v!r})",
                 lambda v=v: apply_penalty(np.eye(4), np.zeros(4),
                                           np.array([0]), penalty=v),
                 silent_ok=v is None or (_is_real(v) and v >= 1e4))  # None=自动; 阈值相对 max|K_ii|=1
        elif i == 18:
            feed(f"stress_at_point({v!r})", lambda v=v: stress_at_point(_mesh(), {"stress": np.ones((2, 3))}, v, 0.5),
                 silent_ok=_is_real(v))  # x 坐标: 有限实数 (域外报错亦正确)
        elif i == 19:
            feed(f"get_element_kernel({v!r})", lambda v=v: F.get_element_kernel(v))
        elif i == 20:
            feed(f"Mesh nodes({v!r})", lambda v=v: Mesh(v, ELEMS))
        elif i == 21:
            feed(f"Mesh elems({v!r})", lambda v=v: Mesh(NODES, v))
        elif i == 22:
            feed(f"apply_elim free({v!r})", lambda v=v: apply_elimination(np.eye(4), np.zeros(4), v, [0, 1], [0., 0.]))
        elif i == 23:
            feed(f"nodal_L2({v!r})", lambda v=v: nodal_L2_projection(_mesh(), v))
        elif i == 24:
            feed(f"nodal_simple({v!r})", lambda v=v: nodal_simple(_mesh(), v))
        elif i == 25:
            feed(f"AnalysisConfig({v!r})", lambda v=v: AnalysisConfig(E=v),
                 silent_ok=v is None or (_is_real(v) and v > 0))  # None=未指定; 合法 E: 有限正数
        elif i == 26:
            feed(f"point_in_element({v!r})", lambda v=v: point_in_element(_mesh(), v, 0.5),
                 silent_ok=_is_real(v))  # x 坐标: 有限实数 (域外返回 -1 是设计)
        elif i == 27:
            feed(f"replace_nodes({v!r})", lambda v=v: _mesh().replace_nodes(v))
        elif i == 28:
            feed(f"replace_elements({v!r})", lambda v=v: _mesh().replace_elements(v))
        elif i == 29:
            feed(f"estimate_condition K({v!r})", lambda v=v: estimate_condition(v))

    print(f"calls={calls}")
    print(f"problems: {len(bugs)}")
    for b in bugs:
        print("  -", b)
    if not bugs:
        print("OK: 无意外异常冒出")
    sys.exit(1 if bugs else 0)


if __name__ == "__main__":
    main()
