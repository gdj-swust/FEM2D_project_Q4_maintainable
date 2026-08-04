# FEM2D 项目摘要（2026-08-03 状态）

## 是什么

二维线弹性有限元教学求解器（CST / Q4 / Q4R / Q4I 四种单元族），约 5.5 万行 Python。2026-08-03 起有 git 基线（主会话 init，含 @FEM 严格解析 / Physical Point 域外拒绝 / elem_type 只读 3 项已修）。
定位：教学与工程教学场景，非商业生产。

## 输入输出

- 输入四入口：`.txt` 中文描述（如 `models/test_spec.txt`）→ `.geo` → Gmsh → `.msh`；也可直接 `.geo` / `.spec` / `.msh`
- 交互向导：`python run.py` 无参数自动进入（fem2d/wizard.py）
- 输出：求解报告 + matplotlib 云图（应力/位移/误差带）；`--no-plot` 纯计算

## 核心能力

- 单元：CST（教学基础）、Q4（稳健）、Q4I/QM6（综合最佳）、Q4R（专用：规则网格/长宽比<10/膜主导，已有强警告）
- 求解：消去法（默认）/ 乘大数罚函数，稀疏直接/PCG/ILU
- 后处理：SPR/L2/加权应力恢复、Z2 误差估计（eta）、等应力带（Bathe §4.3.6）、应力跳跃、Kirsch 验证
- 微尺度稳健性：全尺度（1e-310 载荷/1e-150 几何）有限结果，绝对阈值已全部相对化

## 目录结构

```
FEM2D_project_Q4_maintainable/
├── run.py / run_demo.py      # CLI 入口 / 演示
├── fem2d/                    # 求解器包
│   ├── element/              # cst / q4 / q4r / q4i 单元内核
│   ├── boundary/             # 边界检测（几何/拓扑/命名/物理组）
│   ├── solver.py             # 消去/罚函数 + 奇异性守卫
│   ├── assembly.py           # 稀疏/向量化装配 + 参考实现
│   ├── error_est.py          # Z2 误差估计
│   ├── stress.py / spr.py    # 应力恢复 / 主应力
│   ├── mesh.py / config.py   # 网格模型 / 配置（校验严格）
│   ├── input_source.py       # 四入口解析（严格，错误响亮）
│   ├── runner.py             # 主流程编排（5 阶段）
│   ├── wizard.py / errors.py # 交互向导 / 领域异常（CliError）
├── scripts/                  # geo_spec / gmsh_runner / convergence 等工具
├── tests/                    # 74 测试文件 + conftest.py，939 测试
├── models/                   # 算例库（21 .geo + 7 .spec + 3 .txt）
└── tools/                    # gmsh-4.15.2 Windows 可执行文件（打包内含，GPL v2+）
```

## 当前状态

- **测试**：`python -m pytest` 本机实测 **939 collected → 0 失败**（2026-08-04 pkg7 复核，937 passed + 2 skipped；无 Gmsh 依赖环境的 collected/skip 数会减少，以 `pytest --collect-only` 实测为准）
- **静态**：ruff（E/F）干净、mypy 50 文件干净、死代码 0 候选
- **数值验证**：四单元族 patch test 机器精度、Kirsch K_t≈3.04、Cook 膜收敛、悬臂梁 vs Timoshenko、微尺度全链路
- **版本**：9.21.1；文档：README / CHANGELOG / COMMANDER_HANDOFF_20260804.md（唯一有效交接；旧交接已归档）
- **打包**（用户惯例）：`FEM2D_project_Q4_YYYYMMDD_HHMMSS_maintainable.zip` 放 Downloads，不含 `tools/`、缓存、`*.egg-info`（gmsh 可执行文件另行分发，分发时保留其 GPL v2+ 声明）

## 已知边界（诚实清单）

1. Q4R 为专用单元：沙漏稳定公式在长宽比 ≥50 / 单排细长网格不可靠（已警告 + 文档化）
2. 真实 Gmsh 端到端 CI 未搭建（本机已覆盖真实链路）
3. mypy 仅基础检查（--strict 有 1215 存量，缺类型标注为主）；ruff 只开 E/F
4. 高复杂度函数已拆分（`_point_in_loop` 27→11 / `solver.solve` 22→5，包 5 完成，行为锁定测试在场）
5. 2026-08-03 起有 git 基线（此前无历史）；历史注释的"审计/审查"叙事已清理（包 5），迁移至 CHANGELOG

## 上手

```bash
pip install -e .[dev]          # 安装
python -m pytest                # 全量测试
python run.py models/test_spec.txt --no-plot   # 冒烟
python run.py                   # 交互向导
python run_demo.py              # 演示（椭圆孔板）
```
