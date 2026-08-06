"""FEM2D 性能基准 — 各规模组装/求解/后处理时间表 (JSON 输出).

用法::

    python scripts/perf_benchmark.py                 # 默认 1k/10k/100k 单元
    python scripts/perf_benchmark.py --scale 300000  # 指定目标单元规模
    python scripts/perf_benchmark.py --all           # 1k/10k/100k/300k 全档
    python scripts/perf_benchmark.py --out bench.json
    python scripts/perf_benchmark.py --ci            # CI 门: 1k/10k 与基线比较
    python scripts/perf_benchmark.py --ci --all      # CI 门加 100k 档
    python scripts/perf_benchmark.py --update-baseline  # 重测基线打印新常量段

每档 (Q4 + CST 规则网格) 分阶段计时: 连接/几何构建、K 组装、求解
(solver 按规模 auto 选择)、应力、L2 恢复、Z2 误差估计; 输出 JSON 供
未来改动对比 (组装/求解时间不退化)。另跑 1e-150 微尺度冒烟确认
性能路径无绝对阈值 (有限结果即可, 不计时)。

--ci 模式是 CI 性能回归门 (workflow test-perf job): 跑 1k/10k 两档
(--all 加 100k), 每阶段与 CI_BASELINE 常量比较, 阈值
= max(基线 × CI_MULT, 基线 + CI_ADD_MS) — 宽松门防 flaky, 超阈值才红,
任一阶段超阈值退出码非 0; --mem 与 --ci 互斥 (tracemalloc 使计时失真)。
基线刷新: --update-baseline 每档两遍测量取中位数, 输出可粘贴常量段。

JSON 键 (可对比项):
    machine / platform / numpy / scipy — 环境指纹
    scales: [{scale, elem_type, n_elem, n_nodes, n_dof, nnz, solver,
              stages: {connect_ms, assemble_ms, solve_ms, stress_ms,
                       l2_ms, estimate_ms}}]
    ci (--ci): {baseline_meta, threshold_formula, stages, passed}
    baseline (--update-baseline): 新基线常量段 (含 meta 日期/commit)
"""
import argparse
import datetime
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc

import numpy as np

# 脚本位于 scripts/ 下 — 基准必须测量本项目代码。editable install 指向
# 其他 worktree 时 sys.path 无 cwd, `python scripts/perf_benchmark.py`
# 会 import 到外部 fem2d 副本 (曾静默测到旧实现, 数据失真)。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fem2d import Mesh, solve
from fem2d.assembly import assemble_sparse_vectorized
from fem2d.error_est import estimate
from fem2d.stress import nodal_L2_projection

# 目标单元规模档位 (实际网格为规则方形, 单元数 ≥ 目标)
DEFAULT_SCALES = (1000, 10_000, 100_000)
ALL_SCALES = (1000, 10_000, 100_000, 300_000)

# CI 门档位: --ci 默认 1k/10k (100k 对 CI 耗时过重, 默认不含),
# --ci --all 才加 100k。
CI_SCALES = (1000, 10_000)
CI_ALL_SCALES = (1000, 10_000, 100_000)

# ── CI 门阈值公式 (宽松门, 防 flaky) ──────────────────────────────
# 每阶段阈值 = max(基线 × CI_MULT, 基线 + CI_ADD_MS); 实测 ≤ 阈值 → 绿,
# 实测 > 阈值 → 红 (超阈值才红, 边界值不红)。
#   CI_MULT = 4.0 — 吸收跨机倍差 + 共享 runner 抖动: 本地开发机 →
#     ubuntu-latest 2 vCPU 标准 runner 对 BLAS 单线程路径典型 1.5-3×,
#     叠加 co-tenant 争用后 4× 以下均属"正常波动"。门只拦量级级回归
#     (如 L2 向量化回退、组装复杂度退化 — 历史优化回退均 ≥4-6×)。
#   CI_ADD_MS = 200 — 微阶段 (1k 应力 ~0.4ms) 的固定噪声地板: 乘性门
#     在绝对毫秒级阶段上对 runner 启动/调度抖动过敏感, 地板使正常
#     波动全绿; 0.4ms → 200ms 是 500× 裕度, 但微阶段绝对量无关紧要,
#     10k 档才是结构回归的检测面。
#   两式在基线 R = 200/3 ≈ 66.7ms 处相等: 大阶段乘性项主导, 小阶段
#   加性项主导。
CI_MULT = 4.0
CI_ADD_MS = 200.0

# 每档每阶段比较的阶段键 (顺序即表格列顺序)。
CI_STAGE_KEYS = ("connect_ms", "assemble_ms", "solve_ms",
                 "stress_ms", "l2_ms", "estimate_ms")

# ── CI 门基线常量 ─────────────────────────────────────────────────
# 2026-08-06 本机实测 (每档两遍测量取中位数, ms); 刷新方法:
#   python scripts/perf_benchmark.py --update-baseline
# 输出可粘贴常量段, 替换本常量后提交 (meta 记录日期/commit 即来源)。
# 注意: 基线以本常量为准 — docs/performance.md 的表只是人工可读记录,
# 不作为门的数据源 (解析 md 脆)。
CI_BASELINE = {
    "meta": {
        "recorded_on": "2026-08-06",
        "commit": "1902d6a",
        "platform": "Windows-11-10.0.26200-SP0",
        "machine": "AMD64",
        "python": "3.13.5",
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "method": "每档两遍测量取中位数",
    },
    "CPS4": {
        1000: {
            "connect_ms": 3.534,
            "assemble_ms": 5.414,
            "solve_ms": 24.779,
            "stress_ms": 0.425,
            "l2_ms": 3.376,
            "estimate_ms": 5.293,
        },
        10000: {
            "connect_ms": 29.139,
            "assemble_ms": 47.564,
            "solve_ms": 242.106,
            "stress_ms": 4.516,
            "l2_ms": 39.12,
            "estimate_ms": 43.825,
        },
        100000: {
            "connect_ms": 331.913,
            "assemble_ms": 476.701,
            "solve_ms": 918.513,
            "stress_ms": 44.355,
            "l2_ms": 710.468,
            "estimate_ms": 513.335,
        },
    },
    "CPS3": {
        1000: {
            "connect_ms": 1.05,
            "assemble_ms": 1.877,
            "solve_ms": 7.428,
            "stress_ms": 0.18,
            "l2_ms": 4.875,
            "estimate_ms": 2.663,
        },
        10000: {
            "connect_ms": 7.979,
            "assemble_ms": 11.731,
            "solve_ms": 75.566,
            "stress_ms": 2.176,
            "l2_ms": 50.898,
            "estimate_ms": 14.509,
        },
        100000: {
            "connect_ms": 94.033,
            "assemble_ms": 115.924,
            "solve_ms": 295.265,
            "stress_ms": 24.575,
            "l2_ms": 646.712,
            "estimate_ms": 158.32,
        },
    },
}


def _grid(nx, ny, code):
    """规则网格: Q4 为 nx×ny 四边形, CST 为 2×nx×ny 三角形."""
    gx = np.linspace(0.0, 1.0, nx + 1)
    gy = np.linspace(0.0, 1.0, ny + 1)
    xx, yy = np.meshgrid(gx, gy)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    if code == "CPS3":
        quads = []
        for j in range(ny):
            for i in range(nx):
                n0 = j * (nx + 1) + i
                quads.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
        tri = []
        for n0, n1, n2, n3 in quads:
            tri.append([n0, n1, n2])
            tri.append([n0, n2, n3])
        return nodes, np.array(tri, dtype=np.int64)
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elems.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    return nodes, np.array(elems, dtype=np.int64)


def _corner_dofs(nx, ny):
    """4 角节点 DOF (刚体模态约束充分)."""
    ncol = nx + 1
    corners = [0, nx, ny * ncol, ny * ncol + nx]
    dofs = []
    for n in corners:
        dofs.extend([2 * n, 2 * n + 1])
    return np.array(dofs, dtype=np.int64)


def _timeit(fn, track_mem):
    if track_mem:
        tracemalloc.reset_peak()
    t0 = time.perf_counter()
    out = fn()
    dt = (time.perf_counter() - t0) * 1000.0
    if track_mem:
        peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
        return out, dt, peak_mb
    return out, dt, None


def run_scale(scale, code, track_mem=False):
    """跑一个规模档位, 返回 stage 时间 (可选内存峰值) dict."""
    nx = int(np.ceil(np.sqrt(scale)))
    if code == "CPS3":
        nx = int(np.ceil(np.sqrt(scale / 2.0)))
    nodes, elems = _grid(nx, nx, code)
    n_dof = 2 * len(nodes)
    mesh = Mesh(nodes, elems, E=2.1e11, nu=0.3, plane_type="stress",
                fixed_dofs=_corner_dofs(nx, nx), elem_type=code)
    F = np.zeros(n_dof)
    F[2 * nx] = 1e6  # 右下角 x 向集中力

    if track_mem:
        tracemalloc.start()
    _, t_connect, m_connect = _timeit(
        lambda: mesh.build_connectivity(), track_mem)
    K, t_assemble, m_assemble = _timeit(
        lambda: assemble_sparse_vectorized(mesh), track_mem)
    nnz = int(K.nnz)
    res, t_solve, m_solve = _timeit(
        lambda: solve(mesh, verbose=False), track_mem)  # auto: ≥10 万 DOF 用 CG
    _, t_stress, m_stress = _timeit(
        lambda: mesh.element_kernel.compute_response(
            mesh, res["u"][mesh.element_dofs]), track_mem)
    _, t_l2, m_l2 = _timeit(
        lambda: nodal_L2_projection(mesh, res["stress"]), track_mem)
    _, t_est, m_est = _timeit(
        lambda: estimate(mesh, res, verbose=False), track_mem)
    if track_mem:
        tracemalloc.stop()
    stages = {
        "connect_ms": round(t_connect, 3),
        "assemble_ms": round(t_assemble, 3),
        "solve_ms": round(t_solve, 3),
        "stress_ms": round(t_stress, 3),
        "l2_ms": round(t_l2, 3),
        "estimate_ms": round(t_est, 3),
    }
    if track_mem:
        stages.update({
            "connect_peak_mb": round(m_connect, 1),
            "assemble_peak_mb": round(m_assemble, 1),
            "solve_peak_mb": round(m_solve, 1),
            "stress_peak_mb": round(m_stress, 1),
            "l2_peak_mb": round(m_l2, 1),
            "estimate_peak_mb": round(m_est, 1),
        })
    return {
        "scale": scale, "elem_type": code,
        "n_elem": int(mesh.n_elements), "n_nodes": int(mesh.n_nodes),
        "n_dof": n_dof, "nnz": nnz,
        "solver": res.get("linear_solver", "unknown")["name"],
        "stages": stages,
    }


def micro_scale_smoke():
    """1e-150 几何冒烟: 性能路径必须无绝对阈值 (全有限, 不崩溃)."""
    nx = 20
    nodes, elems = _grid(nx, nx, "CPS4")
    nodes = nodes * 1e-150
    mesh = Mesh(nodes, elems, E=2.1e11, nu=0.3, plane_type="stress",
                fixed_dofs=_corner_dofs(nx, nx), elem_type="CPS4")
    mesh.build_connectivity()
    assemble_sparse_vectorized(mesh)
    F = np.zeros(mesh.n_dof)
    F[2 * nx] = 1e-150
    res = solve(mesh, verbose=False)
    s = mesh.element_kernel.compute_response(
        mesh, res["u"][mesh.element_dofs])
    nodal_L2_projection(mesh, res["stress"])
    eta = estimate(mesh, res, verbose=False)["eta"]
    if not (np.all(np.isfinite(res["u"])) and np.all(np.isfinite(s[0]))
            and np.isfinite(eta)):
        raise RuntimeError("微尺度冒烟产生 NaN/Inf — 性能路径存在绝对阈值")


# ── CI 门: 阈值判定 / 基线校验 / 基线生成 ──────────────────────────


def ci_threshold(baseline_ms):
    """CI 门单阶段阈值: max(基线×CI_MULT, 基线+CI_ADD_MS).

    宽松门语义: 实测 ≤ 阈值全绿, 实测 > 阈值才红 — 正常波动 (跨机倍差
    + runner 抖动, 见 CI_MULT/CI_ADD_MS 注释) 不触发; 门只拦量级级回归。
    """
    return max(baseline_ms * CI_MULT, baseline_ms + CI_ADD_MS)


def judge_ci_stage(recorded_ms, measured_ms):
    """单阶段判定. 返回 (threshold_ms, is_fail): 实测 > 阈值 → FAIL,
    实测 == 阈值 (边界值) → 不红 (超阈值才红, 防 flaky)."""
    threshold_ms = ci_threshold(recorded_ms)
    return threshold_ms, measured_ms > threshold_ms


def validate_ci_baseline(baseline):
    """基线 schema 校验 — 损坏必须明确报错, 非静默: CI 门决不能用坏
    基线当"全绿" (曾有过把坏输入静默放行的教训, 见 knowledge base
    input-parsing-edge-cases)。"""
    if not isinstance(baseline, dict) or "meta" not in baseline:
        raise ValueError("顶层必须含 meta (记录日期/commit)")
    for code in ("CPS4", "CPS3"):
        by_scale = baseline.get(code)
        if not isinstance(by_scale, dict) or not by_scale:
            raise ValueError(f"{code} 档位缺失或为空")
        for scale, stages in by_scale.items():
            if not isinstance(stages, dict):
                raise ValueError(f"{code}/{scale} 不是 dict")
            missing = [k for k in CI_STAGE_KEYS if k not in stages]
            if missing:
                raise ValueError(f"{code}/{scale} 缺阶段键 {missing}")
            for key in CI_STAGE_KEYS:
                value = stages[key]
                if not isinstance(value, (int, float)) or not value > 0:
                    raise ValueError(
                        f"{code}/{scale}.{key}={value!r} 非正数")


def judge_ci(results, baseline):
    """逐阶段与基线比较. 返回 (rows, failed): rows 为判定行 dict 列表
    (含记录值/实测值/阈值/状态), failed 指示是否有任一阶段超阈值.
    无基线条目的 (档位, 类型) 组合跳过并 WARN (不判红)."""
    rows = []
    failed = False
    for r in results:
        code = r["elem_type"]
        scale = r["scale"]
        entry = baseline.get(code, {}).get(scale)
        if entry is None:
            print(f"[ci] WARN: {code} {scale} 无基线条目, 跳过判定",
                  flush=True)
            continue
        for key in CI_STAGE_KEYS:
            recorded = entry[key]
            measured = r["stages"][key]
            threshold, is_fail = judge_ci_stage(recorded, measured)
            failed = failed or is_fail
            rows.append({
                "elem_type": code,
                "scale": scale,
                "stage": key,
                "recorded_ms": recorded,
                "measured_ms": measured,
                "threshold_ms": round(threshold, 3),
                "ratio": round(measured / recorded, 3),
                "status": "FAIL" if is_fail else "PASS",
            })
    return rows, failed


def print_ci_table(rows):
    """stdout 阶段判定表 (记录值/实测值/阈值/比值) — CI 日志可 grep."""
    print(f"\n[ci] 阶段判定 (阈值 = max(基线×{CI_MULT:g}, 基线"
          f"+{CI_ADD_MS:g}ms); 实测 ≤ 阈值 → PASS)")
    print(f"{'scale':>7} {'type':<5} {'stage':<12} {'recorded':>9} "
          f"{'measured':>9} {'threshold':>10} {'ratio':>7}  status")
    for r in rows:
        print(f"{r['scale']:>7} {r['elem_type']:<5} {r['stage']:<12} "
              f"{r['recorded_ms']:>9.3f} {r['measured_ms']:>9.3f} "
              f"{r['threshold_ms']:>10.3f} {r['ratio']:>7.3f}  "
              f"{r['status']}")


def print_ci_failures(rows):
    """超阈值阶段逐条明细 (错误表) — CI 日志红侧可读."""
    n = 0
    for r in rows:
        if r["status"] != "FAIL":
            continue
        n += 1
        print(f"[ci] FAIL {r['elem_type']} {r['scale']} {r['stage']}: "
              f"记录 {r['recorded_ms']:.3f}ms 实测 {r['measured_ms']:.3f}ms "
              f"阈值 {r['threshold_ms']:.3f}ms "
              f"(比值 {r['ratio']:.3f})", flush=True)
    return n


def _git_short_head():
    """基线来源 commit (诊断用; git 不可用/无仓库时不阻塞)."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT,
            capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _baseline_meta(scipy_version):
    return {
        "recorded_on": datetime.date.today().isoformat(),
        "commit": _git_short_head(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "method": "每档两遍测量取中位数",
    }


def update_baseline(scales, scipy_version):
    """每档两遍测量取中位数 → 新基线 dict, 并打印可粘贴常量段."""
    baseline = {"meta": _baseline_meta(scipy_version),
                "CPS4": {}, "CPS3": {}}
    for scale in scales:
        for code in ("CPS4", "CPS3"):
            a = run_scale(scale, code)
            b = run_scale(scale, code)
            baseline[code][scale] = {
                key: round(statistics.median([a["stages"][key],
                                              b["stages"][key]]), 3)
                for key in CI_STAGE_KEYS
            }
            print(f"[baseline] {code} {scale} 中位数: "
                  f"{baseline[code][scale]}", flush=True)
    print("\n[baseline] 生成完成 — 用以下常量段替换 perf_benchmark.py "
          "的 CI_BASELINE (含 meta):", flush=True)
    print("CI_BASELINE = " + json.dumps(baseline, indent=4, ensure_ascii=False))
    return baseline


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", type=int, default=None,
                    help="单档单元规模 (如 300000)")
    ap.add_argument("--all", action="store_true", help="1k/10k/100k/300k 全档")
    ap.add_argument("--mem", action="store_true",
                    help="同时记录每阶段内存峰值 (tracemalloc, 时间略偏大)")
    ap.add_argument("--out", default="perf_results.json",
                    help="JSON 输出路径 (默认 perf_results.json)")
    ap.add_argument("--ci", action="store_true",
                    help="CI 门模式: 1k/10k 与基线常量比较 (--all 加 100k), "
                         "任一阶段超阈值退出码非 0")
    ap.add_argument("--update-baseline", action="store_true",
                    help="重测基线 (每档两遍取中位数) 并打印可粘贴常量段")
    args = ap.parse_args()
    if args.mem and (args.ci or args.update_baseline):
        ap.error("--ci/--update-baseline 禁止 --mem: tracemalloc 使计时失真, "
                 "与门语义冲突")
    if args.ci and args.update_baseline:
        ap.error("--ci 与 --update-baseline 互斥: 门判定与基线生成是两个动作")

    if args.scale:
        scales = (args.scale,)
    elif args.all:
        scales = ALL_SCALES if not args.ci else CI_ALL_SCALES
    else:
        scales = CI_SCALES if args.ci else DEFAULT_SCALES

    import scipy

    out = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scales": [],
    }

    if args.update_baseline:
        print("[bench] 微尺度冒烟 (1e-150 几何) ...", flush=True)
        micro_scale_smoke()
        out["micro_scale_smoke_1e-150"] = "ok"
        baseline = update_baseline(scales, scipy.__version__)
        out["baseline"] = baseline
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"[baseline] 结果写入 {args.out}")
        return 0

    # 门判定前先校验基线 — 损坏立刻非零退出, 不浪费一档测量。
    if args.ci:
        try:
            validate_ci_baseline(CI_BASELINE)
        except ValueError as exc:
            print(f"[ci] 基线损坏: {exc}", file=sys.stderr)
            return 1

    for scale in scales:
        for code in ("CPS4", "CPS3"):
            print(f"[bench] {code} ~{scale} 单元 ...", flush=True)
            out["scales"].append(run_scale(scale, code, track_mem=args.mem))

    print("[bench] 微尺度冒烟 (1e-150 几何) ...", flush=True)
    micro_scale_smoke()
    out["micro_scale_smoke_1e-150"] = "ok"

    exit_code = 0
    if args.ci:
        rows, failed = judge_ci(out["scales"], CI_BASELINE)
        print_ci_table(rows)
        n_fail = print_ci_failures(rows)
        out["ci"] = {
            "baseline_meta": CI_BASELINE.get("meta"),
            "threshold_formula": {"mult": CI_MULT, "add_ms": CI_ADD_MS},
            "stages": rows,
            "passed": not failed,
        }
        if failed:
            exit_code = 1
            print(f"[ci] {n_fail} 个阶段超阈值 → 红 (性能回归门, 见上表)",
                  flush=True)
        else:
            print("[ci] 全绿: 所有阶段 ≤ 阈值 (性能回归门通过)", flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[bench] 结果写入 {args.out}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
