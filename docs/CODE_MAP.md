# FEM2D 全局代码地图（2026-08-05 摸底）

> 用途：给下一任总指挥/执行者快速进入状态。本文件是**静态结构地图**（main @ 788fce1），
> 数值行为由金标准/漂移门锁定，结构变动需同步更新本节。配套：`COMMANDER_HANDOFF_20260805c.md`。

## 1. 顶层结构

```
run.py                 纯转发 → runner.main（编码安全网在 main 内）
run_demo.py            demo_complex 示例（椭圆孔内压+顶部压力+自重），直接复用正式 API
fem2d/                 主包（35 顶层模块 + element/ + boundary/ 两个子包）
scripts/               工具层（打包/探针/fuzz/漂移门/geo_spec 等 12+ 脚本）
tests/                 ~90 文件，~1060 test 函数
docs/                  api_contract / boundary_plugins / ci / coverage / performance
models/                示例模型；tools/ 捆绑 Gmsh 4.15.2
```

## 2. 主线数据流（一次分析 = 5 阶段）

```
阶段1 _resolve_input        runner.py:416  — .spec/.geo/.txt/.msh 统一分派
   └─ input_source.resolve_input_file:563   .spec 键值 / .geo @FEM:注释 / .txt 中文几何 / .msh 直读
阶段2 _build_model           runner.py:438  — 网格导入校验 + @FEM 合并 + 边界模型
   ├─ gmsh_adapter.generate_from_geo:670   执行 .geo → 网格 + 语义区域（API 路径）
   ├─ gmsh_adapter.import_msh:613          打开已有 .msh（子进程路径共用 _extract_mesh:479）
   ├─ preprocess.validate_mesh:402         重复节点/单元、零边、退化、孤立、非流形 全家桶
   └─ boundary.build_boundary_segments     naming.py:484 — 优先级：RegionRegistry → edge_labels → 纯拓扑 detect
阶段3 _apply_conditions      runner.py:460  — bc_apply.apply_bcs:419（fix/traction/force/body 四类）
阶段4 _analyze_and_report    runner.py:480  — solver.solve + error_est.estimate + 中文报告
阶段5 _plot                 runner.py:334  — visualize（云图/isoband/交互）
```

## 3. 模块职责表

### element/（内核注册协议 — 行为冻结区）
| 模块 | 职责 |
|---|---|
| base.py | `ElementKernel` ABC（协议）+ `register_element`/`get_element_kernel` 注册表 + JacobianReport |
| cst.py | 常应变三角：显式形状函数/B 矩阵/刚度；自检 completeness + 刚体模态 |
| q4.py | 双线性四边形（2×2 积分），batch 运动学 |
| q4i.py | QM6 非协调模式：incompatible_derivatives + 静力凝聚（_condensation_blocks） |
| q4r.py | 单点积分稳定：affine complement + hourglass 系数 + hourglass_energy |

Mesh 构造时浅拷贝 kernel（Q4R hourglass_coefficient 是可变类属性，防跨 Mesh 污染）。

### boundary/（识别管线 — 识别算法冻结，只加插件/标签）
| 模块 | 职责 |
|---|---|
| topology.py | 邻接图 → 闭环分解(_decompose_loops) → 嵌套定向(_validate_and_nest_loops) → 曲率分割 → 自交检测 |
| geometry.py | 曲率(双边滤波)/锐角断点/直线椭圆拟合(fit_closed_ellipse) |
| naming.py | `build_boundary_segments`:484 总入口；段命名/边名解析(parse_edge_name) |
| detectors/ | 插件识别体系：Detector 基类 + Registry（首个非 None 胜出，GeneralCurveDetector 兜底恒返回） |
| plugins/ | 轮2 示例插件：arc_curvature / circle_label / ellipse_group_label |
| segment_builder/segment_utils | 段构建 + 纯函数助手（segment_sort_key 等） |
| physical_mapping.py | PhysicalEdgeMapper（A 轮已标记弃用） |
| registry_mapping.py | RegionBoundaryMapper：RegionRegistry → 边界映射 |
| selectors.py | BoundarySelector + resolve_boundary_selector |
| conic_merge.py | 兼容 CAD 圆锥曲线合并（组级椭圆标签） |
| model/validation/predicates | 诊断模型 / schema 验证 / orient2d |

### 核心数值（冻结区）
| 模块 | 职责 |
|---|---|
| mesh.py | Mesh 容器：数据不可变（只读 setter + replace_* + 缓存失效）；外法向由相邻单元 CCW 确定 |
| assembly.py | 4 条路径：assemble_sparse / assemble_sparse_vectorized / assemble_lil_reference(教学参考) / assemble_expand |
| solver.py | `solve`:597 完整流程（见 §4）；奇异守卫/平衡校验/沙漏监控 |
| bc.py | apply_elimination:26（Bathe §4.2.2 消去，direct/cg/ilu）+ apply_penalty:192 |
| bc_apply.py | apply_bcs:419 四类载荷施加（fix/traction/force/body） |
| loads_core.py | assemble:31（F=Rc+ΣRs+ΣRb）；parse_traction/parse_vec2/make_edge_profile_func（p/l 分布）；_compile_expr（AST 白名单） |
| loads.py | 兼容层 facade，纯 re-export loads_core |

### 后处理
| 模块 | 职责 |
|---|---|
| stress.py | compute_stresses / principal_stresses / nodal_average / nodal_L2_projection / stress_at_point |
| spr.py | SPR 恢复：节点补丁稀疏结构(_node_patch_csr) + 局部最小二乘(_fit_node_block) |
| error_est.py | estimate:143（SPR/L2/weighted）；compute_traction_jumps:359（牵引跳跃）；element_refinement_indicator:410（Z2 指示器） |
| visualize.py | plot_contour / plot_three / interactive_plot / isoband（Bathe 等应力带）/ Gouraud / traction jumps 图 |
| reporting.py | 中文结果摘要（位移模长 np.hypot 等） |

### 验证与辅助
| 模块 | 职责 |
|---|---|
| convergence.py | 收敛研究（run_cantilever_convergence 204 行，B 轮拆分对象） |
| verification.py | run_plane_verification:15（155 行，B 轮拆分对象） |
| patch_test.py | patch 测试 |
| quality.py | 网格质量指标（bad/warn/ok 互斥分类） |
| wizard.py | 交互建模向导（无参数+终端时自动） |
| regions.py / topology_core.py | 区域注册表 / 矢量化拓扑原语 |
| checks.py / errors.py | 预检项 / 错误类型体系 |

## 4. solver.solve 阶段细节（Bathe §4.2）

```
0. validate_state           构造后字段可重写 → 求解前快速失败
1. 拓扑 + Jacobian 报告     无效单元先于组装抛错（inverted/degenerate）
2. _partition_dofs          约束划分 + 逐连通分量刚体模态检查（rank<3 禁止求解）
   + Q4R 长宽比告警
3. assemble_sparse + assemble_loads
4. _solve_linear_system     纯 Dirichlet（空矩阵特判）/ 消去 / 罚 三分支
   _solve_with_singular_guard:140  秩警告→异常，非秩警告转发（轮3 修复）
   _check_solution_finite   NaN/Inf 快速失败
4'. 残差                    后向误差 ||r||∞/(||K||∞||u||∞+||F||∞)（Bathe §8.2.6）
5. 应力响应                 单元/积分点应变应力 + von Mises
5'. 沙漏监控                内能有限性 + Q4R 沙漏能占比
5.5 全局平衡检查            ΣR+ΣF/ΣM，tol_rel 相对尺度（无绝对阈值）
6. 小变形 + 结果有限性检查；条件数可选（默认关）
```

## 5. 测试体系（~1060 函数）

- **金标准锁定**：`tests/boundary_golden/*.json`（边界段 schema——不得新增键）+ `tests/test_solve_refactor_lock.py`（求解逐位锁）
- **漂移门**：`scripts/regression_compare.py`（基线 7ee65fc vs main，相对差必须 0.0）
- **探针**：`scripts/audit_contract_probe.py`（契约表一致性，0 FAIL）
- **fuzz**：`scripts/fuzz_api.py 500`（固定种子）+ `combo_fuzz.py`
- **分支锁**：test_{solver,runner,stress,error_est}_branches.py
- **分组**：contract / boundary / regression_audit_2026080{2,3a-e} / usability / analytic（E 轮）/ structure（B 轮）

## 6. 冻结区与契约（改动需特殊程序）

| 区域 | 规则 |
|---|---|
| element/ 内核公式、solver 数值逻辑、error_est 公式 | 只改入口校验 |
| 边界识别管线（boundary/） | 只加插件/标签/输入 |
| 段 schema | 金标准锁定，不得新增键 |
| von_mises/principal_stresses 对 (n,3) 输出 | 逐位不变 |
| apply_penalty 有限输入 | 数值逐位不变 |
| cg_rtol | 必须 1e-10（实测 1e-8 污染支反力） |
| DOF 约定 | x→2n、y→2n+1 |
| gmsh | 必须 -nt N -2；.geo 视为"可信可执行输入"（SystemCall 拦截在 scripts/gmsh_runner.sanitize_geo_source） |
| 禁绝对阈值 | 用相对尺度 / np.finfo(float).tiny（微尺度 1e-150 模型是硬指标） |
