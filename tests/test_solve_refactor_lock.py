"""solver.solve 重构行为锁定测试 (包 5 任务 2).

GOLDEN 是重构前 (2026-08-03 基线) 对代表性模型 (单元族 x 求解方法 x
微尺度 x 大坐标 x 纯 Dirichlet) 抓取的精确输出: stdout 日志序列 + 完整
result dict。重构只允许等价拆分, 不允许任何数值/日志/键集合变化 --
逐值精确比较 (非 allclose)。生成脚本: 仓库外 gen_solve_golden.py。

环境敏感例外 (2026-08-04 CI 首跑确认): Residual / ΣF / ΣM 行与 result
的 residual / force_balance / moment_balance 键是 backward-error 与
力/力矩平衡残差 — 浮点舍入噪声级 (Linux OpenBLAS 与 Windows 求和序
不同 → 最后一位不同, 观测 4.15e-17 vs 2.07e-17, 恰 2×)。这些行/键
改为相对容差比较 (rtol=2.0 覆盖 2× 观测 + atol=1e-12 覆盖近零分量,
如 ΣF -5.68e-14 vs 0.0); 其余 stdout 行与 result 键保持逐字符/逐值。
"""
import io
import re
import warnings
from contextlib import redirect_stdout

import numpy as np
import pytest

# 环境敏感诊断行 — 结构必须逐字符匹配 (regex 锚定整行), 数值容差比较
_RESID_RE = re.compile(
    r"\[OK\] Residual = ([0-9.eE+-]+) \(backward error\)\n")
_SIGF_RE = re.compile(
    r"\[Solver\] ΣF = \(([0-9.eE+-]+), ([0-9.eE+-]+)\) N  "
    r"\(rel: ([0-9.eE+-]+)\)\n")
_SIGM_RE = re.compile(
    r"\[Solver\] ΣM = ([0-9.eE+-]+) N·m  \(rel: ([0-9.eE+-]+)\)\n")
_DIAG_RE = (_RESID_RE, _SIGF_RE, _SIGM_RE)
_BALANCE_KEYS = ("residual", "force_balance", "moment_balance")
# rtol=2.0: 观测到 2× 舍入差异 (4.15e-17 vs 2.07e-17); atol=1e-12:
# 覆盖近零分量 (如 0.0 vs -5.68e-14) — 微尺度模型 (1e-166) 亦远小于此
_DIAG_RTOL, _DIAG_ATOL = 2.0, 1e-12


def _compare_stdout(name, out, golden_out):
    """逐字符比较, 环境敏感诊断行剥离后容差比较."""
    for pattern in _DIAG_RE:
        om, gm = pattern.search(out), pattern.search(golden_out)
        assert om is not None or gm is None, (
            f"[{name}] 金标准无 {pattern.pattern[:24]}... 行但当前输出有")
        assert gm is not None or om is None, (
            f"[{name}] 金标准有 {pattern.pattern[:24]}... 行但当前输出缺失")
        if om is None:
            continue
        for a, g in zip(om.groups(), gm.groups()):
            assert np.isclose(float(a), float(g), rtol=_DIAG_RTOL,
                              atol=_DIAG_ATOL), (
                f"[{name}] {pattern.pattern[:16]}: 数值漂移 "
                f"{a!r} vs 金标准 {g!r}")
        out = out[:om.start()] + out[om.end():]
        golden_out = golden_out[:gm.start()] + golden_out[gm.end():]
    assert out == golden_out, (
        "[" + name + "] stdout 与金标准不一致 (诊断行已剥离), 前 400 字符: "
        + repr(out[:400]))


def _compare_result(name, result, golden_result):
    assert sorted(result) == sorted(golden_result), (
        "[" + name + "] result 与金标准不一致, 键差: "
        + repr(sorted(result) ^ sorted(golden_result)))
    for key, gv in golden_result.items():
        av = result[key]
        if key in _BALANCE_KEYS:
            ga, aa = np.asarray(gv, dtype=float), np.asarray(av, dtype=float)
            assert ga.shape == aa.shape, (
                f"[{name}] result[{key}] 形状漂移: {aa.shape} vs {ga.shape}")
            assert np.all(np.isclose(aa, ga, rtol=_DIAG_RTOL,
                                     atol=_DIAG_ATOL)), (
                f"[{name}] result[{key}] 数值漂移: {aa!r} vs {ga!r}")
        else:
            assert av == gv, f"[{name}] result[{key}] 与金标准不一致"

from fem2d import Mesh
from fem2d.solver import solve


def _plate(elem_type, scale=1.0, origin=(0.0, 0.0), forces=-500.0):
    x0, y0 = origin
    if elem_type == "CPS3":
        nodes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        elements = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    else:
        nodes = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    nodes = nodes * scale + np.array([x0, y0])
    mesh = Mesh(nodes=nodes, elements=elements, E=2.1e11, nu=0.3,
                thickness=0.01, plane_type="stress", elem_type=elem_type)
    for node in (0, 1):
        mesh.fix_node(node, "both", 0.0)
    for node in (2, 3):
        mesh.add_force(node, 0.0, forces)
    return mesh


def _dirichlet_only():
    mesh = Mesh(
        nodes=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        E=2.1e11, nu=0.3, thickness=0.01, plane_type="stress",
        elem_type="CPS4")
    for node in range(4):
        mesh.fix_node(node, "both", 1e-3)
    return mesh


MODELS = [
    ("cst", _plate("CPS3"), {}),
    ("cst_penalty", _plate("CPS3"), {"method": "penalty"}),
    ("q4", _plate("CPS4"), {}),
    ("q4_cg", _plate("CPS4"), {"linear_solver": "cg"}),
    ("q4_cond", _plate("CPS4"), {"check_condition": True}),
    ("q4r", _plate("CPS4R"), {}),
    ("dirichlet_only", _dirichlet_only(), {}),
    ("micro", _plate("CPS3", scale=1e-150, forces=-5e-150), {}),
    ("large_coord", _plate("CPS4", origin=(1e12, 1e12)), {}),
]


def _np(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _np(v) for k, v in value.items()}
    return value


def _run(mesh, kwargs):
    buf = io.StringIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(buf):
            result = solve(mesh, **kwargs)
    return buf.getvalue(), _np(result)


GOLDEN = {'cst': {'stdout': '[Solver] 约束: 4 DOFs, 4 free DOFs\n[Solver] 组装总刚 K (8×8) ...\n[Solver] 组装等效载荷 F ...\n[Solver] 消去法 + SuperLU ...\n  [OK] Residual = 4.15e-17 (backward error)\n[Solver] max|u| = 4.855114e-07\n[Solver] ΣF = (-5.684e-14, -2.274e-13) N  (rel: 3.16e-08)\n[Solver] ΣM = -5.684e-14 N·m  (rel: 6.77e-17)\n[Solver] max|reaction| = 5.000000e+02\n', 'result': {'u': [0.0, 0.0, 0.0, 0.0, -3.462050599201064e-08, -4.2121615623612943e-07, 9.891573140574466e-08, -4.855113816498634e-07], 'stress': [[-29161.118508655112, -97203.72836218371, -2796.2716378162436], [-2796.271637816237, -102796.27163781619, 2796.2716378162477]], 'strain': [[0.0, -4.2121615623612943e-07, -3.462050599201064e-08], [1.3353623739775531e-07, -4.855113816498634e-07, 3.462050599201069e-08]], 'vm_stress': [86532.21136172145, 101542.62013593818], 'stress_qp': [[[-29161.118508655112, -97203.72836218371, -2796.2716378162436]], [[-2796.271637816237, -102796.27163781619, 2796.2716378162477]]], 'strain_qp': [[[0.0, -4.2121615623612943e-07, -3.462050599201064e-08]], [[1.3353623739775531e-07, -4.855113816498634e-07, 3.462050599201069e-08]]], 'dA_qp': [[0.5], [0.5]], 'reactions': [159.78695073235684, 499.99999999999994, -159.7869507323569, 499.9999999999999], 'residual': np.float64(4.147910649287126e-17), 'small_deformation_ok': True, 'internal_energy': 0.0002266818844714982, 'hourglass_energy': 0.0, 'hourglass_energy_ratio': 0.0, 'linear_solver': {'name': 'direct', 'iterations': 1}, 'reaction_dofs': [0, 1, 2, 3], 'reaction_vector': [159.78695073235684, 499.99999999999994, -159.7869507323569, 499.9999999999999, 0.0, 0.0, 0.0, 0.0], 'external_force_vector': [0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, -500.0], 'force_balance': [-5.684341886080802e-14, -2.2737367544323206e-13], 'moment_balance': -5.684341886080802e-14, 'balance_ok': True}}, 'cst_penalty': {'stdout': '[Solver] 约束: 4 DOFs, 4 free DOFs\n[Solver] 组装总刚 K (8×8) ...\n[Solver] 组装等效载荷 F ...\n[Solver] 乘大数法 + spsolve ...\n  penalty = 1.56e+17 (k_penalty = max(|K_ii|) × 1e8, additive)\n  [OK] Residual = 1.50e-24 (backward error)\n[Solver] max|u| = 4.855114e-07\n[Solver] ΣF = (-5.684e-14, -1.137e-13) N  (rel: 1.71e-08)\n[Solver] ΣM = -5.684e-14 N·m  (rel: 6.77e-17)\n[Solver] max|reaction| = 5.000000e+02\n', 'result': {'u': [-1.0257927554021374e-15, -3.209876543209875e-15, 1.0257927554021376e-15, -3.209876543209875e-15, -3.4620506520614875e-08, -4.212161602354976e-07, 9.89157310109989e-08, -4.855113847258816e-07], 'stress': [[-29161.118089869415, -97203.72840234125, -2796.2715976587097], [-2796.2715976587, -102796.27159765866, 2796.271597658716]], 'strain': [[2.051585510804275e-15, -4.2121615702562104e-07, -3.462050549482212e-08], [1.3353623753161378e-07, -4.85511381516005e-07, 3.46205054948222e-08]], 'vm_stress': [86532.21149025825, 101542.62011174104], 'stress_qp': [[[-29161.118089869415, -97203.72840234125, -2796.2715976587097]], [[-2796.2715976587, -102796.27159765866, 2796.271597658716]]], 'strain_qp': [[[2.051585510804275e-15, -4.2121615702562104e-07, -3.462050549482212e-08]], [[1.3353623753161378e-07, -4.85511381516005e-07, 3.46205054948222e-08]]], 'dA_qp': [[0.5], [0.5]], 'reactions': [159.7869484376407, 499.99999999999994, -159.78694843764075, 499.99999999999994], 'residual': np.float64(1.503242609461277e-24), 'small_deformation_ok': True, 'internal_energy': 0.0002266818844714982, 'hourglass_energy': 0.0, 'hourglass_energy_ratio': 0.0, 'linear_solver': {'name': 'direct', 'iterations': 1}, 'reaction_dofs': [0, 1, 2, 3], 'reaction_vector': [159.7869484376407, 499.99999999999994, -159.78694843764075, 499.99999999999994, 0.0, 0.0, 0.0, 0.0], 'external_force_vector': [0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, -500.0], 'force_balance': [-5.684341886080802e-14, -1.1368683772161603e-13], 'moment_balance': -5.684341886080802e-14, 'balance_ok': True}}, 'q4': {'stdout': '[Solver] 约束: 4 DOFs, 4 free DOFs\n[Solver] 组装总刚 K (8×8) ...\n[Solver] 组装等效载荷 F ...\n[Solver] 消去法 + SuperLU ...\n  [OK] Residual = 5.86e-17 (backward error)\n[Solver] max|u| = 4.620098e-07\n[Solver] ΣF = (-1.421e-14, 1.137e-13) N  (rel: 1.59e-08)\n[Solver] ΣM = 2.842e-14 N·m  (rel: 3.16e-17)\n[Solver] max|reaction| = 5.000000e+02\n', 'result': {'u': [0.0, 0.0, 0.0, 0.0, 9.558823529411771e-08, -4.620098039215687e-07, -9.55882352941176e-08, -4.6200980392156864e-07], 'stress': [[-9926.47058823529, -100000.00000000001, 2.2737367544323206e-12]], 'strain': [[9.558823529411765e-08, -4.6200980392156864e-07, 2.6469779601696886e-23]], 'vm_stress': [95424.77539672583], 'stress_qp': [[[-22662.138290947623, -103820.70031081371, -4457.483695949312], [-22662.138290947623, -103820.70031081371, 4457.4836959493205], [2809.197114477043, -96179.29968918631, 4457.483695949317], [2809.197114477043, -96179.29968918631, -4457.483695949316]]], 'strain_qp': [[[4.040034191569753e-08, -4.6200980392156864e-07, -5.518789337842007e-08], [4.040034191569753e-08, -4.6200980392156864e-07, 5.518789337842016e-08], [1.5077612867253777e-07, -4.6200980392156864e-07, 5.518789337842012e-08], [1.5077612867253777e-07, -4.6200980392156864e-07, -5.518789337842011e-08]]], 'dA_qp': [[0.25, 0.25, 0.25, 0.25]], 'reactions': [99.26470588235296, 500.0, -99.26470588235297, 500.0000000000001], 'residual': np.float64(5.862145945076695e-17), 'small_deformation_ok': True, 'internal_energy': 0.0002310049019607843, 'hourglass_energy': 0.0, 'hourglass_energy_ratio': 0.0, 'linear_solver': {'name': 'direct', 'iterations': 1}, 'reaction_dofs': [0, 1, 2, 3], 'reaction_vector': [99.26470588235296, 500.0, -99.26470588235297, 500.0000000000001, 0.0, 0.0, 0.0, 0.0], 'external_force_vector': [0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, -500.0], 'force_balance': [-1.4210854715202004e-14, 1.1368683772161603e-13], 'moment_balance': 2.842170943040401e-14, 'balance_ok': True}}, 'q4_cg': {'stdout': '[Solver] 约束: 4 DOFs, 4 free DOFs\n[Solver] 组装总刚 K (8×8) ...\n[Solver] 组装等效载荷 F ...\n[Solver] 消去法 + Jacobi-PCG ...\n  PCG (jacobi) iterations = 2 (rtol=1.0e-10)\n  [OK] Residual = 4.40e-17 (backward error)\n[Solver] max|u| = 4.620098e-07\n[Solver] ΣF = (4.263e-14, 2.274e-13) N  (rel: 3.21e-08)\n[Solver] ΣM = 5.684e-14 N·m  (rel: 6.31e-17)\n[Solver] max|reaction| = 5.000000e+02\n', 'result': {'u': [0.0, 0.0, 0.0, 0.0, 9.55882352941177e-08, -4.6200980392156874e-07, -9.558823529411766e-08, -4.6200980392156864e-07], 'stress': [[-9926.470588235285, -100000.0, -3.183231456205249e-12]], 'strain': [[9.558823529411769e-08, -4.6200980392156864e-07, -4.301339185275744e-23]], 'vm_stress': [95424.77539672583], 'stress_qp': [[[-22662.138290947623, -103820.70031081371, -4457.483695949318], [-22662.138290947623, -103820.70031081371, 4457.483695949318], [2809.197114477054, -96179.2996891863, 4457.483695949311], [2809.1971144770578, -96179.2996891863, -4457.483695949324]]], 'strain_qp': [[[4.0400341915697545e-08, -4.6200980392156864e-07, -5.5187893378420135e-08], [4.0400341915697545e-08, -4.620098039215687e-07, 5.518789337842013e-08], [1.5077612867253782e-07, -4.620098039215687e-07, 5.518789337842005e-08], [1.5077612867253782e-07, -4.6200980392156864e-07, -5.5187893378420214e-08]]], 'dA_qp': [[0.25, 0.25, 0.25, 0.25]], 'reactions': [99.26470588235297, 500.00000000000006, -99.26470588235293, 500.0000000000001], 'residual': np.float64(4.39660945880752e-17), 'small_deformation_ok': True, 'internal_energy': 0.00023100490196078434, 'hourglass_energy': 0.0, 'hourglass_energy_ratio': 0.0, 'linear_solver': {'name': 'cg', 'iterations': 2, 'rtol': 1e-10, 'preconditioner': 'jacobi'}, 'reaction_dofs': [0, 1, 2, 3], 'reaction_vector': [99.26470588235297, 500.00000000000006, -99.26470588235293, 500.0000000000001, 0.0, 0.0, 0.0, 0.0], 'external_force_vector': [0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, -500.0], 'force_balance': [4.263256414560601e-14, 2.2737367544323206e-13], 'moment_balance': 5.684341886080802e-14, 'balance_ok': True}}, 'q4_cond': {'stdout': '[Solver] 约束: 4 DOFs, 4 free DOFs\n[Solver] 组装总刚 K (8×8) ...\n[Solver] 组装等效载荷 F ...\n[Solver] 消去法 + SuperLU ...\n  [OK] Residual = 5.86e-17 (backward error)\n[Solver] max|u| = 4.620098e-07\n[Solver] ΣF = (-1.421e-14, 1.137e-13) N  (rel: 1.59e-08)\n[Solver] ΣM = 2.842e-14 N·m  (rel: 3.16e-17)\n[Solver] max|reaction| = 5.000000e+02\n[Solver] cond(K_aa) = 1.01e+01 -> ~0.0 digits lost [GOOD]\n', 'result': {'u': [0.0, 0.0, 0.0, 0.0, 9.558823529411771e-08, -4.620098039215687e-07, -9.55882352941176e-08, -4.6200980392156864e-07], 'stress': [[-9926.47058823529, -100000.00000000001, 2.2737367544323206e-12]], 'strain': [[9.558823529411765e-08, -4.6200980392156864e-07, 2.6469779601696886e-23]], 'vm_stress': [95424.77539672583], 'stress_qp': [[[-22662.138290947623, -103820.70031081371, -4457.483695949312], [-22662.138290947623, -103820.70031081371, 4457.4836959493205], [2809.197114477043, -96179.29968918631, 4457.483695949317], [2809.197114477043, -96179.29968918631, -4457.483695949316]]], 'strain_qp': [[[4.040034191569753e-08, -4.6200980392156864e-07, -5.518789337842007e-08], [4.040034191569753e-08, -4.6200980392156864e-07, 5.518789337842016e-08], [1.5077612867253777e-07, -4.6200980392156864e-07, 5.518789337842012e-08], [1.5077612867253777e-07, -4.6200980392156864e-07, -5.518789337842011e-08]]], 'dA_qp': [[0.25, 0.25, 0.25, 0.25]], 'reactions': [99.26470588235296, 500.0, -99.26470588235297, 500.0000000000001], 'residual': np.float64(5.862145945076695e-17), 'small_deformation_ok': True, 'internal_energy': 0.0002310049019607843, 'hourglass_energy': 0.0, 'hourglass_energy_ratio': 0.0, 'linear_solver': {'name': 'direct', 'iterations': 1}, 'reaction_dofs': [0, 1, 2, 3], 'reaction_vector': [99.26470588235296, 500.0, -99.26470588235297, 500.0000000000001, 0.0, 0.0, 0.0, 0.0], 'external_force_vector': [0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, -500.0], 'force_balance': [-1.4210854715202004e-14, 1.1368683772161603e-13], 'moment_balance': 2.842170943040401e-14, 'balance_ok': True, 'condition_info': {'condition_number': np.float64(10.068143994182403), 'lambda_min': np.float64(183365856.43000287), 'lambda_max': np.float64(1846153846.153846), 'digits_lost': 0.0, 'status': 'GOOD', 'error': None}}}, 'q4r': {'stdout': '[Solver] 约束: 4 DOFs, 4 free DOFs\n[Solver] 组装总刚 K (8×8) ...\n[Solver] 组装等效载荷 F ...\n[Solver] 消去法 + SuperLU ...\n  [OK] Residual = 2.87e-17 (backward error)\n[Solver] max|u| = 4.746032e-07\n[Solver] Q4R hourglass energy = 7.642563e-07 (0.32% of internal energy)\n[Solver] ΣF = (8.882e-15, 0.000e+00) N  (rel: 1.26e-09)\n[Solver] ΣM = -2.842e-14 N·m  (rel: 2.87e-17)\n[Solver] max|reaction| = 5.000000e+02\n', 'result': {'u': [0.0, 0.0, 0.0, 0.0, 1.3756613756613675e-07, -4.746031746031738e-07, -1.3756613756613837e-07, -4.7460317460317534e-07], 'stress': [[-1111.1111111111059, -99999.99999999999, -3.206915605590199e-12]], 'strain': [[1.3756613756613755e-07, -4.7460317460317454e-07, -3.970466940254533e-23]], 'vm_stress': [99449.09982895832], 'stress_qp': [[[-1111.1111111111059, -99999.99999999999, -3.206915605590199e-12]]], 'strain_qp': [[[1.3756613756613755e-07, -4.7460317460317454e-07, -3.970466940254533e-23]]], 'dA_qp': [[1.0]], 'reactions': [11.111111111111095, 500.00000000000006, -11.111111111111086, 500.0], 'residual': np.float64(2.872952577802567e-17), 'small_deformation_ok': True, 'internal_energy': 0.0002373015873015873, 'hourglass_energy': 7.642563198118752e-07, 'hourglass_energy_ratio': 0.0032206119162640893, 'linear_solver': {'name': 'direct', 'iterations': 1}, 'hourglass_energy_elem': [7.642563198118752e-07], 'reaction_dofs': [0, 1, 2, 3], 'reaction_vector': [11.111111111111095, 500.00000000000006, -11.111111111111086, 500.0, 0.0, 0.0, 0.0, 0.0], 'external_force_vector': [0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, -500.0], 'force_balance': [8.881784197001252e-15, 0.0], 'moment_balance': -2.842170943040401e-14, 'balance_ok': True}}, 'dirichlet_only': {'stdout': '[Solver] 约束: 8 DOFs, 0 free DOFs\n[Solver] 全部 8 DOF 给定位移 — 直接计算反力\n[Solver] 组装总刚 K (8×8) ...\n[Solver] 组装等效载荷 F ...\n[Solver] max|u| = 1.000000e-03\n[Solver] ΣF = (-4.366e-11, -8.877e-10) N  (rel: 2.85e-03)\n[Solver] ΣM = -3.238e-10 N·m  (rel: 6.01e-01)\n[Solver] max|reaction| = 4.220055e-10\n', 'result': {'u': [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001], 'stress': [[9.382518800355326e-10, 3.127506266785109e-09, 1.094627193374788e-09]], 'strain': [[0.0, 1.3552527156068805e-20, 1.3552527156068805e-20]], 'vm_stress': [3.3647956077543547e-09], 'stress_qp': [[[0.0, 0.0, 0.0], [1.8765037600710652e-09, 6.255012533570218e-09, 2.189254386749576e-09], [1.8765037600710652e-09, 6.255012533570218e-09, 2.189254386749576e-09], [0.0, 0.0, 0.0]]], 'strain_qp': [[[0.0, 0.0, 0.0], [0.0, 2.710505431213761e-20, 2.710505431213761e-20], [0.0, 2.710505431213761e-20, 2.710505431213761e-20], [0.0, 0.0, 0.0]]], 'dA_qp': [[0.25, 0.25, 0.25, 0.25]], 'reactions': [-1.3460521586239338e-10, -3.4924596548080444e-10, -1.1641532182693481e-10, -1.1641532182693481e-10, 1.4915713109076023e-10, -4.220055416226387e-10, 5.820766091346741e-11, 0.0], 'residual': 0.0, 'small_deformation_ok': True, 'internal_energy': -4.656612873077393e-13, 'hourglass_energy': 0.0, 'hourglass_energy_ratio': 0.0, 'linear_solver': {'name': 'none', 'iterations': 0}, 'reaction_dofs': [0, 1, 2, 3, 4, 5, 6, 7], 'reaction_vector': [-1.3460521586239338e-10, -3.4924596548080444e-10, -1.1641532182693481e-10, -1.1641532182693481e-10, 1.4915713109076023e-10, -4.220055416226387e-10, 5.820766091346741e-11, 0.0], 'external_force_vector': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'force_balance': [-4.3655745685100555e-11, -8.87666828930378e-10], 'moment_balance': -3.2378011383116245e-10, 'balance_ok': True}}, 'micro': {'stdout': '[Solver] 约束: 4 DOFs, 4 free DOFs\n[Solver] 组装总刚 K (8×8) ...\n[Solver] 组装等效载荷 F ...\n[Solver] 消去法 + SuperLU ...\n  [OK] Residual = 0.00e+00 (backward error)\n[Solver] max|u| = 4.855114e-159\n[Solver] ΣF = (-8.140e-166, 0.000e+00) N  (rel: 0.00e+00)\n[Solver] ΣM = -3.316e-316 N·m  (rel: 3.95e-17)\n[Solver] max|reaction| = 5.000000e-150\n', 'result': {'u': [0.0, 0.0, 0.0, 0.0, -3.4620505992010614e-160, -4.2121615623612947e-159, 9.891573140574473e-160, -4.855113816498635e-159], 'stress': [[-291.61118508655113, -972.0372836218371, -27.962716378162416], [-27.962716378162437, -1027.9627163781622, 27.962716378162472]], 'strain': [[0.0, -4.2121615623612946e-09, -3.4620505992010625e-10], [1.3353623739775514e-09, -4.855113816498634e-09, 3.4620505992010625e-10]], 'vm_stress': [865.3221136172144, 1015.4262013593822], 'stress_qp': [[[-291.61118508655113, -972.0372836218372, -27.962716378162416]], [[-27.962716378162433, -1027.9627163781622, 27.962716378162472]]], 'strain_qp': [[[0.0, -4.2121615623612946e-09, -3.4620505992010614e-10]], [[1.3353623739775532e-09, -4.855113816498635e-09, 3.4620505992010687e-10]]], 'dA_qp': [[5e-301], [5e-301]], 'reactions': [1.5978695073235682e-150, 5e-150, -1.597869507323569e-150, 5e-150], 'residual': np.float64(0.0), 'small_deformation_ok': True, 'internal_energy': 2.2668188447149825e-308, 'hourglass_energy': 0.0, 'hourglass_energy_ratio': 0.0, 'linear_solver': {'name': 'direct', 'iterations': 1}, 'reaction_dofs': [0, 1, 2, 3], 'reaction_vector': [1.5978695073235682e-150, 5e-150, -1.597869507323569e-150, 5e-150, 0.0, 0.0, 0.0, 0.0], 'external_force_vector': [0.0, 0.0, 0.0, 0.0, 0.0, -5e-150, 0.0, -5e-150], 'force_balance': [-8.139985654852579e-166, 0.0], 'moment_balance': -3.3156184e-316, 'balance_ok': True}}, 'large_coord': {'stdout': '[Solver] 约束: 4 DOFs, 4 free DOFs\n[Solver] 组装总刚 K (8×8) ...\n[Solver] 组装等效载荷 F ...\n[Solver] 消去法 + SuperLU ...\n  [OK] Residual = 5.86e-17 (backward error)\n[Solver] max|u| = 4.620098e-07\n[Solver] ΣF = (-1.421e-14, 1.137e-13) N  (rel: 1.59e-08)\n[Solver] ΣM = 2.842e-14 N·m  (rel: 3.16e-17)\n[Solver] max|reaction| = 5.000000e+02\n', 'result': {'u': [0.0, 0.0, 0.0, 0.0, 9.558823529411771e-08, -4.620098039215687e-07, -9.55882352941176e-08, -4.6200980392156864e-07], 'stress': [[-9926.47058823529, -100000.00000000001, 2.2737367544323206e-12]], 'strain': [[9.558823529411765e-08, -4.6200980392156864e-07, 2.6469779601696886e-23]], 'vm_stress': [95424.77539672583], 'stress_qp': [[[-22662.138290947623, -103820.70031081371, -4457.483695949312], [-22662.138290947623, -103820.70031081371, 4457.4836959493205], [2809.197114477043, -96179.29968918631, 4457.483695949317], [2809.197114477043, -96179.29968918631, -4457.483695949316]]], 'strain_qp': [[[4.040034191569753e-08, -4.6200980392156864e-07, -5.518789337842007e-08], [4.040034191569753e-08, -4.6200980392156864e-07, 5.518789337842016e-08], [1.5077612867253777e-07, -4.6200980392156864e-07, 5.518789337842012e-08], [1.5077612867253777e-07, -4.6200980392156864e-07, -5.518789337842011e-08]]], 'dA_qp': [[0.25, 0.25, 0.25, 0.25]], 'reactions': [99.26470588235296, 500.0, -99.26470588235297, 500.0000000000001], 'residual': np.float64(5.862145945076695e-17), 'small_deformation_ok': True, 'internal_energy': 0.0002310049019607843, 'hourglass_energy': 0.0, 'hourglass_energy_ratio': 0.0, 'linear_solver': {'name': 'direct', 'iterations': 1}, 'reaction_dofs': [0, 1, 2, 3], 'reaction_vector': [99.26470588235296, 500.0, -99.26470588235297, 500.0000000000001, 0.0, 0.0, 0.0, 0.0], 'external_force_vector': [0.0, 0.0, 0.0, 0.0, 0.0, -500.0, 0.0, -500.0], 'force_balance': [-1.4210854715202004e-14, 1.1368683772161603e-13], 'moment_balance': 2.842170943040401e-14, 'balance_ok': True}}}


def test_solve_behavior_identical_to_golden():
    for name, mesh, kwargs in MODELS:
        out, result = _run(mesh, kwargs)
        _compare_stdout(name, out, GOLDEN[name]["stdout"])
        _compare_result(name, result, GOLDEN[name]["result"])


def test_residual_tolerance_accepts_2x_platform_noise():
    """容差逻辑判别性 (本地即生效): CI 观测的 2× 舍入差异必须接受,
    真漂移 (1e-10) 必须拒绝. 若改回逐字符比较, 2× 用例在此失败."""
    cst_out = GOLDEN["cst"]["stdout"]
    _compare_stdout("cst", cst_out.replace("4.15e-17", "2.07e-17"),
                    cst_out)                        # Linux 2× → 接受
    for drift in ("1.00e-10", "5.00e-12"):
        with pytest.raises(AssertionError):
            _compare_stdout("cst", cst_out.replace("4.15e-17", drift),
                            cst_out)                # 真漂移 → 拒绝


def test_residual_line_structure_still_locked():
    """诊断行结构仍必须逐字符锁定 — 行格式/单位/前缀变了必须红."""
    out, _ = _run(*MODELS[0][1:])
    assert "[OK] Residual = " in out and "backward error" in out
    assert "ΣF = (" in out and "N  (rel:" in out
    assert "ΣM = " in out and "N·m  (rel:" in out
