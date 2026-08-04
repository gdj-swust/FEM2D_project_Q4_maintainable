# FEM2D CI 行为说明

> 版本: 2026-08-03 (包 1/4 — CI 搭建)。工作流: `.github/workflows/ci.yml`。
> 本文件记录: 依赖矩阵决策依据、每个 job 的步骤与预期结果、clone 后本地复现路径。

## 1. 矩阵决策

### 最低 NumPy 版本: **2.0** (与 pyproject 一致, 已由 CI 实测)

| 问题 | 结论 | 依据 |
|------|------|------|
| 代码是否 numpy 2.x 兼容 | 是 | 全库 grep: 只用 `np.ptp(...)` **顶层函数** (numpy 2.0 只移除 `arr.ptp()` 方法并提示改用顶层函数, 顶层函数保留); `np.float_`/`np.NaN`/`np.Inf`/`np.alltrue` 等 2.0 删除别名 0 命中 |
| 最低版实测 | numpy 2.0.x | 本机 (numpy 2.5.1, 含 gmsh) 939 收集 0 失败; 3.13 上最老 2.x (2.1.0) 全量实测同样 0 失败; 3.11×numpy==2.0 CI 组合为声明地板 — numpy 2.0 最高支持 Python 3.12 (3.13 需 numpy>=2.1), 地板组合因此放 3.11 |
| 旧 CI 的 numpy==1.24 | 移除 | 与 pyproject `numpy>=2.0` 矛盾, 且 1.24 无 Python 3.13 wheel, 装不上 |
| 旧 CI 的 `numpy==latest` | 改为约束 `numpy>=2.0,<3` | `latest` 不是合法版本号, pip 直接报错 |
| scipy 下限 | 1.12 → **1.13.0** | 1.13.0 是首个支持 numpy 2.0 的 scipy (1.12 与 numpy 2.x 不兼容) |
| matplotlib 下限 | 3.5 → **3.8.4** | 3.8.4 是首个支持 numpy 2.0 的 matplotlib (3.5 与 numpy 2.x 不兼容) |
| Python 版本 | CI 测 3.11/3.12/3.13 | 建议矩阵; `requires-python >=3.9` 保留 (numpy 2.0/scipy 1.13/matplotlib 3.8.4 均支持 3.9+, 但 3.9 已 EOL 不测) |

### 测试矩阵 (test-core job)

| Python | numpy 约束 | 覆盖含义 |
|--------|-----------|---------|
| 3.11 | `==2.0` | **声明最低版** (numpy 2.0 最高支持 3.12) |
| 3.11 | `>=2.0,<3` | 2.x 最新 |
| 3.12 | `>=2.0,<3` | 2.x 最新 |
| 3.13 | `>=2.0,<3` | 最新栈 |

test-full job 用 3.13 × `>=2.0,<3` (与本地开发环境一致, 便于对照)。

## 2. 三个 job 与预期结果

### lint (一次)

checkout → setup-python 3.13 → `pip install -e ".[dev]"` → compileall → ruff → mypy → `python run.py --self-test`。

预期: 全绿 (本机 dry-run 实测: ruff 0 问题 / mypy 0 错误 / self-test 6 PASS 0 FAIL)。

### test-core (矩阵 4 组合, 无 gmsh)

与 lint 相同安装, 加一步**显式卸载 gmsh** 并断言 `import gmsh` 失败 — 与 runner 镜像
是否预装 libGLU 无关, 确定性构造"无 gmsh 环境"。然后 `python -m pytest -q`。

无 gmsh 时的 skip 行为 (已有守卫, 本文件第 3 节逐一列明; 2026-08-04 全新无 gmsh venv
实测, 与 runner 镜像无关; numpy 2.5.1 / scipy 1.18.0 / matplotlib 3.11.1 / pytest 9.1.1):
**918 collected → 898 passed + 20 skipped, 0 失败**。
无 collection error, 无 ImportError 崩溃。

### test-full (一次, 有 gmsh)

checkout → setup-python 3.13 → `apt-get install libglu1-mesa libgl1` (gmsh Python API
加载 libgmsh.so 需要 libGLU.so.1; libgl1 兜底 libGL.so.1) → `pip install -e ".[dev]"` →
`python -m pytest -q -rs`。

预期: **939 collected → 937 passed + 2 skipped** (2026-08-04 本机实测)。2 个 skip 恒定为
`test_geo_models.py` 依赖捆绑 gmsh.exe 的两个测试 (86MB 可执行文件 gitignore 不入库,
所有 job 都没有) — 与本地无 tools/ 检出实测一致。

## 3. 无 gmsh 环境的 skip 守卫清单 (全部已确认, 非 ImportError 崩溃)

无 gmsh 时实测的 20 个 skip 明细 (2026-08-04 全新无 gmsh venv 实测, `-rs` 输出;
复现: `python -m pip uninstall -y gmsh && python -m pytest -rs`, 数值以实测为准):

| 文件 | 数量 | 守卫方式 |
|------|------|---------|
| tests/test_boundary_gmsh.py | 1 | 模块级 `try: import gmsh` → 捕 `(ImportError, OSError)` → `pytest.skip(allow_module_level=True)` (OSError 是缺 libGLU 时的真实异常, importorskip 捕不到); pytest 汇总按 1 个模块事件计 |
| tests/test_boundary_complex.py | 1 | 同上 |
| tests/test_boundary_highpressure.py | 1 | 同上 |
| tests/test_boundary_stress.py | 1 | 同上 |
| tests/test_msh_import_audit_20260803.py | 7 | 模块级 import 捕 `(ImportError, OSError)` → `pytestmark = skipif(_gmsh is None)` |
| tests/test_physical_point_resolution.py | 5 | 同上 |
| tests/test_geo_models.py | 2 | `find_gmsh()` 返回 None (无 GMSH_PATH/无 tools/gmsh.exe/无 PATH 可执行文件) → `pytest.skip("bundled gmsh executable not available")`; 另一测试捕 `GmshUnavailableError` — **这两个 skip 在有无 gmsh 的 job 里恒有** |
| tests/test_regressions.py:205 | 1 | 显式 skipif: "gmsh Python API unavailable" |
| tests/test_output_dir_policy.py:439 | 1 | `gmsh` 不可用 (No module named 'gmsh') 的显式 skip (9.21.0 新增 --output-dir 用例) |

fem2d/scripts 全库无模块级 `import gmsh`; `fem2d/gmsh_adapter.py` 用
`_load_gmsh_module()` 懒加载, 失败包装为 `GmshUnavailableError` — 无 gmsh 时
import fem2d 正常。

## 4. 本地复现 (无 GitHub Actions 时的 dry-run 路径)

```bash
python -m venv .venv-ci                      # 本机 Python 3.13
python -m pip install -e ".[dev]"            # 与 CI 同一条命令 (2026-08-03 已验证)
python -m pytest                             # 有 gmsh: 939 collected → 937 passed + 2 skipped (2026-08-04 实测)
python -m pip uninstall -y gmsh && python -m pytest   # 无 gmsh: 918 collected → 898 passed + 20 skipped (2026-08-04 实测)
ruff check fem2d/ scripts/ tests/ run.py run_demo.py
mypy fem2d/
python -m compileall -q fem2d scripts tests run.py run_demo.py
python run.py --self-test
```

numpy 地板组合的本机近似 (3.13 装不了 numpy 2.0 — 它只发布到 Python 3.12 的
wheel): `python -m pip install -e ".[dev]" "numpy==2.1.0"` (3.13 上最老的 2.x),
2026-08-03 曾实测与当时最新栈逐项一致, 0 失败 — 数值以实测为准, 不再写死。

## 5. 常见问题

- **yaml 验证**: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml', encoding='utf-8'))"` — 通过 (`on` 键已加引号, PyYAML 1.1 下避免被解析为布尔 True)。
- **numpy 地板组合为什么是 3.11×2.0 而不是 3.13×2.0**: numpy 2.0 只发布到 cp312 的 wheel, 3.13 装不上 — 这正是旧 CI "Python 3.13 + numpy 1.24" 必挂的原因之一。
- **为什么 core job 卸载 gmsh 而不是依赖 libGLU 缺失**: ubuntu-latest 镜像内容不受我们控制, 预装 libGLU 后 gmsh 会 import 成功导致"该 skip 的没 skip"; 卸载是确定性构造。
- **为什么 gmsh 仍是硬依赖而非 optional**: 项目声明 `pip install -e ".[dev]"` 即含 gmsh; 卸载只在 CI 的 core job 构造测试环境, 不改变安装契约 (本机/用户安装仍自动带 gmsh)。
