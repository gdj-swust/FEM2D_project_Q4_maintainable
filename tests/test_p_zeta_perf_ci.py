"""P-ζ: perf_benchmark --ci 性能回归门单测 (慢值红/正常绿/边界值).

判别性红侧自证 (任务书验收 1):
  - 阈值判定函数: 实测 < 阈值 → PASS (正常绿); 实测 > 阈值 → FAIL
    (慢值红); 实测 == 阈值 → PASS (边界值不红 — 语义"超阈值才红",
    防 flaky)。
  - --ci 真实跑 1k 冒烟 (快) + 退出码断言 + 判定表完整 (12 行)。
  - 红侧集成 (确定性, 不依赖机器速度): 桩替换 run_scale 注入 10× 慢
    测量 → 全 FAIL → 退出码非 0 + 错误表。
  - 基线损坏 → 明确报错 (非静默): schema 校验 ValueError + --ci 启动
    即非零退出, 不静默当全绿。
  - --update-baseline: 两遍中位数生成可粘贴常量段 + JSON 含 baseline。
无 gmsh 依赖 (perf_benchmark 全路径无 gmsh import, 见 ci.yml test-perf
注释); --ci --scale 1000 冒烟 < 30s。
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(PROJECT_ROOT, "scripts")
PERF = os.path.join(SCRIPTS, "perf_benchmark.py")


def _run(args, timeout=300):
    # PYTHONIOENCODING=utf-8: Windows 本地控制台默认 GBK, 子进程管道输出
    # 编码与父侧 UTF-8 解码不一致 → 中文断言必挂; 强制子进程 UTF-8 输出
    # (与 CI Ubuntu 行为一致)。
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        [sys.executable] + args,
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
        env=env)


def _load_perf():
    """每次调用新建模块实例 — 桩/基线注入互不影响 (模块级状态隔离)."""
    sys.path.insert(0, PROJECT_ROOT)
    spec = importlib.util.spec_from_file_location("perf_benchmark", PERF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 阈值判定函数单测 (判别性: 慢值红 / 正常绿 / 边界值) ────────────


def test_ci_threshold_formula_floor_and_mult():
    """阈值 = max(基线×CI_MULT, 基线+CI_ADD_MS): 小阶段加性地板主导,
    大阶段乘性项主导."""
    mod = _load_perf()
    assert mod.ci_threshold(0.4) == pytest.approx(200.4)   # 1k stress 量级
    assert mod.ci_threshold(50.0) == pytest.approx(250.0)
    assert mod.ci_threshold(242.0) == pytest.approx(968.0)  # 10k solve 量级


def test_ci_threshold_formula_crossover():
    """两式在 R = 200/(4-1) ≈ 66.7ms 处相等 (公式交叉点)."""
    mod = _load_perf()
    r = mod.CI_ADD_MS / (mod.CI_MULT - 1.0)
    assert mod.ci_threshold(r) == pytest.approx(r * mod.CI_MULT, rel=1e-12)


def test_ci_judge_normal_green():
    """正常 (实测 < 阈值) → PASS."""
    mod = _load_perf()
    threshold, fail = mod.judge_ci_stage(242.0, 300.0)   # 300 < 968
    assert threshold == pytest.approx(968.0)
    assert fail is False


def test_ci_judge_slow_red():
    """慢值 (实测 > 阈值) → FAIL — 判别性红侧."""
    mod = _load_perf()
    _, fail = mod.judge_ci_stage(242.0, 1000.0)          # 1000 > 968
    assert fail is True


def test_ci_judge_boundary_value_green():
    """边界值: 实测 == 阈值 → 不红 (超阈值才红, 防 flaky);
    阈值 + ε → 红 (按系数公式预期)."""
    mod = _load_perf()
    threshold, fail = mod.judge_ci_stage(242.0, 242.0 * mod.CI_MULT)
    assert threshold == pytest.approx(968.0)
    assert fail is False
    _, fail2 = mod.judge_ci_stage(242.0, 242.0 * mod.CI_MULT + 1e-3)
    assert fail2 is True


# ── --ci 模式集成 (真实跑 1k 冒烟 + 退出码) ────────────────────────


def test_ci_mode_smoke_1k(tmp_path):
    """判别性: --ci --scale 1000 真实跑通 → 退出 0 + 判定表完整
    (2 类型 × 6 阶段全 PASS) + JSON ci 段."""
    out = tmp_path / "ci.json"
    r = _run(["scripts/perf_benchmark.py", "--ci", "--scale", "1000",
              "--out", str(out)])
    assert r.returncode == 0, r.stderr[-800:]
    assert "[ci]" in r.stdout
    # 表头说明行含 1 个 "PASS" + 12 行状态列 → ≥ 12 行全绿; 精确性由
    # JSON 的 stages 逐行断言兜底。
    assert r.stdout.count("PASS") >= 12, r.stdout[-1200:]
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["micro_scale_smoke_1e-150"] == "ok"
    assert data["ci"]["passed"] is True
    assert data["ci"]["threshold_formula"] == {"mult": 4.0, "add_ms": 200.0}
    assert len(data["ci"]["stages"]) == 12
    assert all(s["status"] == "PASS" for s in data["ci"]["stages"])
    assert data["ci"]["baseline_meta"]["commit"]  # 基线来源非空


# ── 红侧集成: 桩注入慢测量 → 全 FAIL → 非零退出 (确定性) ──────────


def test_ci_mode_red_on_slow_measurements(tmp_path):
    """判别性红侧 (桩替换 run_scale 注入"实测 = 1.5×阈值"慢测量, 不依赖
    机器速度): 每阶段必超阈值 (含 200ms 地板主导的微阶段) → 全 FAIL →
    退出码非 0 + 错误表 + JSON passed=False."""
    wrapper = tmp_path / "red_wrapper.py"
    wrapper.write_text(f"""\
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
import importlib.util
spec = importlib.util.spec_from_file_location(
    "perf", {PERF!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def slow(scale, code, track_mem=False):
    entry = mod.CI_BASELINE[code][scale]
    return {{
        "scale": scale, "elem_type": code,
        "n_elem": 0, "n_nodes": 0, "n_dof": 0, "nnz": 0,
        "solver": "stub",
        "stages": {{k: mod.ci_threshold(v) * 1.5
                   for k, v in entry.items()}},
    }}


mod.run_scale = slow
sys.argv = ["perf_benchmark.py", "--ci", "--scale", "1000",
            "--out", {str(tmp_path / "red.json")!r}]
sys.exit(mod.main())
""", encoding="utf-8")
    r = _run([str(wrapper)])
    assert r.returncode != 0, f"慢测量竟全绿: {r.stdout[-600:]}"
    assert r.stdout.count("FAIL") >= 12, r.stdout[-1200:]
    data = json.loads((tmp_path / "red.json").read_text(encoding="utf-8"))
    assert data["ci"]["passed"] is False
    assert len(data["ci"]["stages"]) == 12
    assert all(s["status"] == "FAIL" for s in data["ci"]["stages"])


# ── 基线损坏 → 明确报错 (非静默) ──────────────────────────────────


def test_ci_baseline_corruption_raises():
    """损坏基线 (缺阶段键/缺类型档位/非正数) → 明确 ValueError —
    判门决不能用坏基线当"全绿"."""
    mod = _load_perf()
    bad1 = json.loads(json.dumps(mod.CI_BASELINE))
    del bad1["CPS4"]["1000"]["stress_ms"]   # json 往返后键为字符串
    with pytest.raises(ValueError, match="阶段键"):
        mod.validate_ci_baseline(bad1)
    bad2 = {"meta": {"recorded_on": "x"}}
    with pytest.raises(ValueError, match="档位"):
        mod.validate_ci_baseline(bad2)
    bad3 = json.loads(json.dumps(mod.CI_BASELINE))
    bad3["CPS3"]["1000"]["solve_ms"] = -1.0
    with pytest.raises(ValueError, match="非正数"):
        mod.validate_ci_baseline(bad3)


def test_ci_mode_corrupt_baseline_exits_nonzero(tmp_path):
    """判别性: --ci 启动时基线损坏 → 非零退出 + 明确报错 (不静默全绿)."""
    wrapper = tmp_path / "corrupt_wrapper.py"
    wrapper.write_text(f"""\
import sys
sys.path.insert(0, {PROJECT_ROOT!r})
import importlib.util
spec = importlib.util.spec_from_file_location(
    "perf", {PERF!r})
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
del mod.CI_BASELINE["CPS4"][1000]["solve_ms"]
sys.argv = ["perf_benchmark.py", "--ci", "--scale", "1000",
            "--out", {str(tmp_path / "corrupt.json")!r}]
sys.exit(mod.main())
""", encoding="utf-8")
    r = _run([str(wrapper)])
    assert r.returncode != 0
    assert "基线损坏" in r.stderr


# ── 参数互斥与基线刷新 ────────────────────────────────────────────


def test_ci_mode_rejects_mem():
    """--ci 禁止 --mem (tracemalloc 与门语义冲突) — 非零退出 + 明确报错."""
    r = _run(["scripts/perf_benchmark.py", "--ci", "--mem"])
    assert r.returncode != 0
    assert "--mem" in r.stderr


def test_ci_and_update_baseline_mutually_exclusive():
    r = _run(["scripts/perf_benchmark.py", "--ci", "--update-baseline"])
    assert r.returncode != 0
    assert "互斥" in r.stderr


def test_update_baseline_generates_snippet(tmp_path):
    """--update-baseline: 两遍中位数 → 可粘贴常量段 + JSON baseline
    (JSON 键序列化为字符串)."""
    out = tmp_path / "bl.json"
    r = _run(["scripts/perf_benchmark.py", "--update-baseline",
              "--scale", "1000", "--out", str(out)])
    assert r.returncode == 0, r.stderr[-800:]
    assert "CI_BASELINE = {" in r.stdout
    assert '"CPS4"' in r.stdout
    data = json.loads(out.read_text(encoding="utf-8"))
    bl = data["baseline"]
    assert bl["CPS4"]["1000"]["solve_ms"] > 0
    assert bl["meta"]["commit"]  # 基线来源 commit 非空
