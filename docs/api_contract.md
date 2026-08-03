# FEM2D API 契约表

> 版本: 9.18.1+ (2026-08-03, 契约清账阶段 1 交付物; 阶段 2/3 完成后已更新:
> K1-K11 全部修复并带判别性测试, 见第 K 节状态列)
> 范围: fem2d/ 全部顶层导出 (`fem2d/__init__.py` `__all__` + 惰性导出) +
> Mesh 全部公共方法 + 输入端入口 (input_source / gmsh_adapter / bc_apply)。
> 单元内核公式 (element/)、solver 数值逻辑、error_est 公式为行为冻结区,
> 本表只记录它们的**入口校验契约**, 不改内核。

## 0. 全局约定

| 约定 | 内容 |
|------|------|
| 错误类型 | 用户误用 → `ValueError` / `TypeError` / `CliError(exit_code)` 带参数上下文; 禁止裸 `IndexError` / `KeyError` / `AttributeError` 冒出 |
| 消息格式 | `函数名: 参数名=值 — 原因, 期望` (阶段 2 统一为共享 helper 的格式) |
| DOF 编号 | 节点 i → x=2i, y=2i+1 (Bathe Eq 4.18) |
| NaN/Inf | 一律拒绝 (有限性), 不允许静默传播或静默夹值 |
| 布尔掩码 | `[True, False]` 是布尔掩码, 一律拒绝 (禁止折叠成 DOF 0/1) |
| 多余分量 | 静默忽略 = 最危险错误; 必须报错 |
| 微尺度/大坐标 | 1e-150 几何 / 1e12 坐标 / 1e-310 载荷必须保持有限, 禁止绝对阈值 |
| 状态图例 | ✅ = 已达标 (全部误用路径有带上下文的领域错误) · ⚠️ = 有缺口 (列明) · ➖ = 不适用/内部 |

---

## A. Mesh 类 (fem2d/mesh.py)

### A0. 构造器 `Mesh(nodes, elements, thickness=1.0, E=210e9, nu=0.3, plane_type="stress", fixed_dofs=None, prescribed_vals=None, body_force=None, surface_tractions=None, concentrated_forces=None, elem_type="CPS3")`

| 参数 | 合法形状 (None 语义) | 误用清单 → 应有错误 | 现状 |
|------|---------------------|---------------------|------|
| nodes | (n_nodes, 2) 有限实数数组; 非空 | 标量 → ValueError(先验维度); 1-D/3-D → ValueError(形状); NaN/Inf → ValueError; 空 → ValueError | ✅ |
| elements | (n_elem, npe) 整数索引; 非空; npe 与 elem_type 匹配 | 标量 → ValueError; 形状错 → ValueError; 浮点索引 → ValueError(拒绝截断); 负/越界 → ValueError; NaN → ValueError; 重复单元 → ValueError; 空 → ValueError | ✅ |
| thickness/E/nu | 有限正标量 / nu∈(-1, 0.5) | NaN/Inf → ValueError; t≤0/E≤0 → ValueError; nu 越界 → ValueError; 非数值 (str) → np.isfinite TypeError 冒泡 | ⚠️ 非数值类型: `np.isfinite("a")` 抛裸 TypeError — 阶段 2 收敛为共享标量 helper |
| plane_type | "stress" 或 "strain" | 其他字符串 → 构造期未查 (仅 validate_state 查) → ⚠️ 构造器不校验, 求解前才报 | ⚠️ 构造期不校验 (文档化: 延迟到 validate_state) |
| fixed_dofs | (n_fixed,) 整数 DOF 索引; None = 空 | 布尔掩码 → TypeError(明示); 浮点非整数 → ValueError; 负/越界 → ValueError; 重复 → unique 去重 (幂等, 合法); list → 接受 | ✅ |
| prescribed_vals | dict{DOF int → 有限值}; None = {} | 键不在 fixed_dofs → ValueError; 值 NaN/Inf → validate_state 报 ValueError; 非 int 键 → ValueError(键集差) | ✅ |
| body_force | None \| callable \| (bx, by) 分量=有限值或 callable | 1/3 分量 → ValueError(形状); 分量 NaN/Inf → ValueError; 标量 → ValueError; 整体 callable 返回契约在真实 Gauss 点检查 (evaluate_vector_field) | ✅ |
| surface_tractions | list of dict{nodes:(ni,nj), traction: 2 分量或 (p,) if is_pressure}; None = [] | 非 dict → ValueError; 缺键 → ValueError; nodes 非二元组 → ValueError; 节点越界/非整数 → ValueError; 内部边 → ValueError(_validate_boundary_edge); traction 形状错 → ValueError; 分量 NaN → ValueError | ✅ (节点整数校验与 _validate_node_id 重复, 阶段 2 收敛) |
| concentrated_forces | list of dict{node:int, force:(fx,fy) 有限}; None = [] | 同上; force 分量 callable → ValueError(allow_callable=False) | ✅ (同上收敛点) |
| elem_type | 已注册单元类型字符串 (CPS3/CPS4/.../别名) | 未注册 → ValueError(带注册表); 构造后赋值 → AttributeError(明示只读) | ✅ |

### A1. 节点索引 API (所有 `nid`/`ni`/`nj` 参数共用契约)

**契约**: 整数 (接受"恰为整数"的浮点如 1.0, 拒绝 bool); 范围 [0, n_nodes); 越界/负/布尔 → 带函数名的 ValueError/TypeError。

| API | 签名 | 误用清单 → 应有错误 | 现状 |
|-----|------|---------------------|------|
| fix_node | (nid, dof="both", value=0.0) | nid 非整数/布尔 → TypeError; 越界/负 → ValueError; dof ∉ {x,y,both} → ValueError; value NaN/Inf → ValueError; 非数值 value (str) → np.isfinite 裸 TypeError | ⚠️ value 非数值类型冒裸 TypeError |
| fix_nodes_func | (node_list, func) | 单个节点数传入 (node_list=5) → ValueError(明示需列表); 越界 → ValueError(范围先于索引); func 非 callable 非数值 → 下游 TypeError/ValueError; func 返回 3+ 分量 → ValueError(明示); func 返回 NaN → ValueError; func 返回不可迭代对象 → TypeError(带坐标上下文) | ✅ (K2, 含 np 标量返回值支持) |
| add_force | (nid, fx=0.0, fy=0.0) | nid 非法/越界 → TypeError/ValueError; fx/fy NaN/Inf → ValueError; 非数值 → np.isfinite 裸 TypeError | ⚠️ 非数值分量冒裸 TypeError |
| add_traction | (ni, nj, tx, ty) | 同上; 内部边 → ValueError; 零长边 → ValueError(ULP 相对判据) | ⚠️ 同上 (非数值分量) |
| add_pressure | (ni, nj, p) | 同上; p NaN/Inf → ValueError; 非数值 p → 裸 TypeError | ⚠️ 同上 |
| boundary_outward_normal | (ni, nj) | 非法/越界 → TypeError/ValueError; 内部边 → ValueError; 零长边 → ValueError | ✅ |
| _validate_node_id | static (nid) | bool → TypeError; 非整数 → TypeError; "恰为整数"浮点 → 规范化 | ✅ (阶段 2 收敛的目标 helper) |

### A2. 结构修改 API

| API | 签名 | 误用清单 → 应有错误 | 现状 |
|-----|------|---------------------|------|
| replace_nodes | (new_nodes) | 形状 ≠ 原 → ValueError(保持节点数); NaN/Inf → ValueError; 标量 → ValueError(形状) | ✅ |
| replace_elements | (new_elements) | 形状/空/浮点索引/越界/重复 → ValueError (与构造器同族) | ✅ |
| nodes_on_edge | (axis, edge, tol=None) | axis ∉ {x,y} / edge ∉ {min,max} → ValueError; tol<0 / NaN / Inf → ValueError; tol=None → span×1e-8 自动 (微尺度 tiny 兜底) | ✅ |
| info | () | 无参数 — 内部 check_jacobian 异常可冒泡 (网格已校验) | ✅ |
| validate_state | () | 材料/BC/载荷/缓存一致性全查 → ValueError/TypeError | ✅ |

### A3. 几何/拓扑查询

| API | 签名 | 误用清单 → 应有错误 | 现状 |
|-----|------|---------------------|------|
| check_jacobian | () | 返回 (ok, bad) — 无输入 | ✅ |
| check_rigid_body_constraints | () | 返回 issues list — 无输入; 逐连通分量中心化+归一化 | ✅ |
| build_connectivity | () | 内部 — 无输入 | ✅ |

---

## B. 求解与 BC (fem2d/solver.py, fem2d/bc.py)

| API | 签名 | 参数合法形状 | 误用清单 → 应有错误 | 现状 |
|-----|------|-------------|---------------------|------|
| solve | (mesh, method="elimination", verbose=True, check_condition=False, linear_solver="auto") | mesh: Mesh (或等价协议对象); method ∈ {elimination, penalty}; linear_solver ∈ {auto,direct,cg,cg-block,ilu} | method 非法 → ValueError; linear_solver 非法 → ValueError(所有分支前校验); mesh 非 Mesh (dict/None) → 裸 AttributeError(n_dof); 网格非法 → validate_state ValueError; 欠约束 → RuntimeError(刚体模态, 带分量明细); 奇异 → RuntimeError(MatrixRankWarning 转); 残差大 → RuntimeError; NaN 解 → RuntimeError | ⚠️ mesh 类型未校验 — 非 Mesh 对象冒裸 AttributeError |
| estimate_condition | (K, method="auto") | K: 方阵 (csr/dense); method ∈ {auto,dense,sparse} | K 非方阵 → 下游 numpy/scipy 异常; method 非法 → 静默走 sparse 分支返回 SKIP dict (有 error 字段); 奇异 → dict{status:"SINGULAR?"} | ⚠️ method 非法值静默降级 (有 error 字段, 部分达标); K 形状由 eigsh 兜底 |
| apply_elimination | (K, F, free_dofs, fixed_dofs, prescribed_vals, linear_solver="direct", cg_rtol=1e-10, cg_maxiter=None, return_info=False) | K 方阵有限; F (n,); free/fixed 整数 DOF 数组; prescribed 长度 = fixed; 纯 Dirichlet (free 空) 合法 | K 非方阵/NaN → ValueError; F 形状/NaN → ValueError; free/fixed 布尔掩码 → ValueError; 重叠 → ValueError; 遗漏 DOF → ValueError; 自身重复 → ValueError; fixed/prescribed 长度不等 → ValueError; prescribed NaN → ValueError; 重复约束不同值 → ValueError; linear_solver 非法 → ValueError; cg 不收敛 → RuntimeError(建议 direct) | ✅ |
| apply_penalty | (K, F, fixed_dofs, prescribed_vals=None, penalty=None) | 同上; penalty None = 自动 max\|K_ii\|×1e8 | 同上; penalty NaN/Inf/< max\|K_ii\|×1e4 → ValueError(相对判据) | ✅ |
| estimate_error | (mesh, result, method="SPR", verbose=True) | result: solve() 输出 dict; method ∈ {SPR,L2,weighted} | method 非法 → ValueError(带可选值); result 缺 "stress" 键 → 裸 KeyError; result["stress"] 形状与 n_elem 不符 → 下游 IndexError; mesh 类型错 → AttributeError | ⚠️ result 契约未校验 — 缺键裸 KeyError |

---

## C. 载荷 (fem2d/loads_core.py, fem2d/loads/__init__)

| API | 签名 | 参数合法形状 | 误用清单 → 应有错误 | 现状 |
|-----|------|-------------|---------------------|------|
| assemble_loads | (mesh, n_dof) | mesh 已校验; n_dof=2×n_nodes | n_dof 与网格不符 → 集中力越界写 (numpy IndexError); 载荷记录非法 → mesh.validate_state 已在 solve 前拦截 (独立调用时未拦); callable 求值失败 → ValueError(带边/高斯点); 零长边 → ValueError(ULP) | ⚠️ 独立调用时 n_dof 不匹配 → 裸 IndexError (solve 路径已由 validate_state 防护) |
| parse_vec2 | (s: str) | "bx,by" 字符串; 分量=纯数字或含 x/y 表达式 | 分量数 ≠ 2 → ValueError; 全角逗号 → 归一化接受; NaN/Inf 字面 → ValueError; 溢出 (1e999) → ValueError; 语法错误 → ValueError(带表达式); 非 str (None/int) → AttributeError | ⚠️ 非 str 输入冒裸 AttributeError (.split) |
| parse_traction | (s: str) | "edge:tx,ty[:p\|l\|n]" | 无 ':' → (None,0,0,None) 由调用方处理; 3+ 段 → ValueError; profile 非法 → ValueError; 压力值非数字 → ValueError(带原因); 非 str → AttributeError | ⚠️ 同 parse_vec2 |
| make_edge_profile_func | (tx, ty, profile, edge_start, edge_end, arc_start, total_length) | 分量数值/callable; 边端点为 2 元可转数组; 零长边/零总长 → 返回常数 (退化安全) | 端点形状错 → np.dot ValueError; NaN 坐标 → 静默 NaN 剖面 | ➖ 内部面力链 (bc_apply 专用), 退化已安全 |

---

## D. 应力与误差估计 (fem2d/stress.py, fem2d/spr.py, fem2d/error_est.py)

| API | 签名 | 参数合法形状 | 误用清单 → 应有错误 | 现状 |
|-----|------|-------------|---------------------|------|
| compute_stresses | (mesh, u) | u: (n_dof,) 有限 | u 形状错 → ValueError(带期望); u NaN → 下游输出 NaN (静默) → ⚠️; mesh 非法 → validate_state 未调用 | ⚠️ u NaN 未校验 (静默 NaN 输出) |
| nodal_simple / nodal_weighted | (mesh, elem_stress) | elem_stress: (n_elem, n_comp) | 形状错 → ValueError; 孤立节点 → ValueError(与 L2 一致); weights 含 NaN → 静默 NaN | ⚠️ 权重 NaN 未校验 |
| nodal_L2_projection | (mesh, elem_stress) | (n_elem, n_comp) 或 (n_elem, nqp, n_comp) | ndim ∉ {2,3} → ValueError; 首维 ≠ n_elem → ValueError; 采样数 ≠ nqp → ValueError; 孤立节点 → ValueError(一致质量阵奇异前置); NaN → 静默 NaN | ⚠️ NaN 未校验 |
| principal_stresses | (stress) | (n, 3) 有限数组 [σx, σy, τxy] | 形状 (n,2)/(n,)/(标量) → 裸 IndexError(stress[:,2]); NaN/Inf 输入 → NaN/Inf 输出 (静默); 非数组 → AttributeError | ⚠️ **形状与有限性均未校验 — 裸 IndexError 冒出** |
| stress_at_point | (mesh, result, x, y, mode="element") | mode ∈ {element,sides,average,recovered} | mode 非法 → ValueError(带可选值); 点不在网格 → ValueError; result 缺 "stress" → KeyError; x/y NaN → point_in_element 返回 -1 → ValueError(带坐标) | ✅ (mode/域), ⚠️ result 缺键同 B 组 |
| point_in_element | (mesh, x, y) | x,y 有限标量 | 返回 -1 (不在网格) — 调用方契约; NaN → 返回 -1 | ✅ |
| spr_recovery | (mesh, elem_stress) | (n_elem, n_comp) 或 (n_elem, nqp, n_comp) | 形状错 → 下游 IndexError; NaN → 静默 NaN; 孤立节点 → 无样点节点回退平均 (设计行为) | ⚠️ 形状/NaN 未前置校验 |
| element_refinement_indicator | (mesh, result) | result: solve() dict | 缺键 → KeyError; 形状错 → IndexError | ⚠️ 同 estimate_error (模块级 API) |

---

## E. 输入链 (fem2d/input_source.py, fem2d/gmsh_adapter.py, fem2d/preprocess.py)

| API | 签名 | 参数合法形状 | 误用清单 → 应有错误 | 现状 |
|-----|------|-------------|---------------------|------|
| resolve_input_file | (fp, config, ask=None) | fp: 存在文件 (.spec/.geo/.txt/.msh); config: AnalysisConfig | 扩展名不支持 → CliError(exit 2); .inp → CliError(exit 2, 已移除提示); .msh 导入失败 → CliError(exit 1); 最终非 .msh → CliError(exit 1); .spec 指定网格不存在 → CliError(exit 1) | ✅ |
| resolve_geo / resolve_txt | (fp, config, ask=None) | fp: .geo/.txt 存在 | gmsh 生成失败 → CliError(exit 1); quad 验证失败 → 重试后抛 GmshTopologyError; .txt 解析失败 → CliError(几何生成失败); 手写 .geo 保护 (临时副本) | ✅ |
| resolve_spec_overrides | (fp, config) | fp: .spec 存在 | 格式错误 (缺 = / 空值) → ValueError(带行号); 键值无法转 float → ValueError(带键名); plane 非法 → CliError; no_plot 非法 → ValueError; 网格不存在 → CliError; 未知键 → WARN; 负 E/非法 nu → config.validate ValueError | ✅ |
| physical_point_from_geo | (geo_path, name, mesh) | geo_path: .geo 或 None; name: str | 无源 .geo → reason="no_geo_source"; gmsh 不可用 → "gmsh_unavailable"; 未找到/歧义/域外/超距 → 区分 reason; 成功 → (nid, label, dist, None) | ✅ |
| generate_geo_with_topology | (geo_path, *, quad=False, output_path=None, plane_type="stress") | geo_path: 存在 .geo | gmsh 缺失/失败 → (None, None) 契约; 拓扑验证失败 → 抛异常 (调用方须双处理); quad 混合网格 → GmshTopologyError | ✅ |
| generate_from_geo | (geo_path, *, quad=False, output_path=None, plane_type="stress", gmsh_module=None) | geo_path: 存在 | 文件不存在 → FileNotFoundError(明示); gmsh 不可用 → GmshUnavailableError; 无单元 → GmshTopologyError; 混合拓扑 → GmshTopologyError; 非平面 z → GmshTopologyError | ✅ |
| import_msh | (msh_path, *, require_quads=False, plane_type="stress") | msh_path: 存在 .msh | 文件不存在 → FileNotFoundError(前置, 与 generate_from_geo 一致); 无 $PhysicalNames 恢复 → WARN; 2-D 单元缺失 → GmshTopologyError | ✅ (K7) |
| read_geo_groups | (geo_path, *, gmsh_module=None) | geo_path: .geo 或 None | 文件不存在 → None (设计); API 失败 → 文本解析回退 | ✅ |
| parse_spec_config | (filepath) | 存在 .spec | 缺 = → ValueError(行号); 空值 → ValueError(行号); BOM → 自动剥离 | ✅ |
| parse_geo_fem_config | (geo_path) | .geo 或 None | 空值/字段数错 → ValueError(文件名+行号+原文); 未知键 → WARN | ✅ |
| validate_mesh | (nodes, elements, elem_type=None, tol=None) | 数组 + 可选类型/容差 | 负索引/重复单元/孤立节点/非流形边/退化单元/零长边 → MeshValidationError(分类报告) | ✅ |

---

## F. 材料与单元注册 (fem2d/material.py, fem2d/element/)

| API | 签名 | 参数合法形状 | 误用清单 → 应有错误 | 现状 |
|-----|------|-------------|---------------------|------|
| D_matrix | (E, nu, plane_type="stress") | E>0 有限; nu∈(-1,0.5); plane ∈ {stress,strain} | E≤0/NaN → ValueError; nu 越界 → ValueError; plane 非法 → ValueError; 非数值 → np.isfinite TypeError | ⚠️ 非数值类型裸 TypeError (同 A 组, 共享标量 helper 收敛) |
| von_mises | (stress, plane_type="stress", nu=0.3) | (..., 3) 有限数组 | plane 非法 → ValueError; 标量/1-D/末维≠3 → ValueError(形状); NaN → ValueError | ✅ (K12, fuzz 发现后修复) |
| get_element_kernel | (elem_type: str) | 注册过的类型/别名 | 未注册 → ValueError(带注册表列表); None → ValueError | ✅ |
| register_element | (kernel: ElementKernel) | ElementKernel 实例 | 非实例 → TypeError; 空名 → ValueError; 重复键不同内核 → ValueError(明示) | ✅ |
| registered_element_types | () | — | — | ✅ |
| verify_all_elements | (mesh, verbose=True) | Mesh | mesh 非法 → 下游异常 (ElementKernel 校验) | ✅ (委托内核) |

---

## G. 边界 (fem2d/boundary/)

| API | 签名 | 参数合法形状 | 误用清单 → 应有错误 | 现状 |
|-----|------|-------------|---------------------|------|
| detect_boundaries | (mesh, ...) | Mesh | 微尺度/大坐标已相对化; 椭圆轴比 tiny 兜底; 闭合环验证 | ✅ |
| build_boundary_segments | (mesh, ...) | Mesh | 无网格 → ValueError | ✅ |
| validate_boundary_segments | (mesh, segments) | Mesh + segments dict list | 缺段/闭合缺失 → 诊断报告 (不抛, 返回诊断) | ✅ (诊断型 API) |
| describe_geometry / print_segments / parse_edge_name | (segments) / (segments) / (name: str, segs) | 段字典列表 | 非法 name → ValueError (解析器) | ✅ |
| segments_from_physical_curves / segments_from_region_registry | (mesh, ...) | Mesh + 源 | 未映射 → 空/诊断 | ✅ |
| semantic_coverage | (mesh, segments, diagnostics=None) | — | 覆盖不完整 → 诊断对象 | ✅ |

---

## H. 配置与质量 (fem2d/config.py, fem2d/quality.py, fem2d/patch_test.py)

| API | 签名 | 参数合法形状 | 误用清单 → 应有错误 | 现状 |
|-----|------|-------------|---------------------|------|
| AnalysisConfig.validate | () | 字段语义 | E≤0/nu 越界/t≤0/lc≤0/jump_ref≤0 → ValueError; plane 非法 → ValueError; linear_solver/error_method/band_tag 非法 → ValueError; body 形状错 → ValueError; band 三参数不齐/非整除/超 10000 层 → ValueError | ✅ |
| AnalysisConfig.from_dict | (data: dict) | 键 ∈ 字段集 | 未知键 → WARN 忽略 (前向兼容, 文档化) | ✅ |
| AnalysisConfig.from_args | (args) | argparse Namespace | 无 args 字段 → 跳过 (保持默认) | ✅ |
| evaluate_mesh_quality / report_mesh_quality | (mesh) / (mesh) | Mesh | 空网格 → 除零风险 (n=0) → ⚠️ 低优先 (solve 前已拦) | ✅/⚠️ |
| run_patch_test | (E=210e9, nu=0.3, plane="stress", tol=1e-10, verbose=True, elem_type="CPS3") | E/nu/plane 语义同上; elem_type 注册过 | E/nu/plane 非法 → D_matrix ValueError; elem_type 非法 → ValueError | ✅ |

---

## I. 装配 (fem2d/assembly.py)

| API | 签名 | 参数合法形状 | 误用清单 → 应有错误 | 现状 |
|-----|------|-------------|---------------------|------|
| assemble_sparse | (mesh) | 已校验 Mesh | 非对称内核 → RuntimeError(对称性检查); 奇异/NaN → 诊断 | ✅ |
| assemble_sparse_vectorized | (mesh, ...) | 已校验 Mesh | 同上; 形状错 → ValueError | ✅ |

---

## J. 其他顶层导出

| API | 签名 | 误用清单 → 应有错误 | 现状 |
|-----|------|---------------------|------|
| run_cantilever_convergence | (L=..., H=..., nx=..., ny=..., elem_type="CPS3", ...) | elem_type 非法 → ValueError; 内部网格生成 | ✅ (惰性导出, 见 __init__ __getattr__) |
| run_patch_test / GmshImportResult / 区域类 | — | 数据类, 字段由构造链校验 | ✅ |
| GmshUnavailableError / GmshTopologyError | — | 领域异常类型 | ✅ |

---

## K. 缺口清单 (阶段 2 修复队列 — 状态列 = 修复 commit)

| # | 缺口 | 位置 | 状态 (修复 commit) |
|---|------|------|--------------------|
| K1 | 标量有限性检查对**非数值类型** (str/complex/容器) 冒裸 TypeError: `np.isfinite("a")` | mesh.fix_node / add_force / add_traction / add_pressure / __init__ 材料 / D_matrix / config | ✅ 已修 — `fem2d/checks.py` `require_finite_scalar/_positive/_nu_valid` (c3d14c5) |
| K2 | `fix_nodes_func(node_list, func)`: node_list 传标量 → 裸 TypeError; func 返回多余分量静默忽略 | mesh.py | ✅ 已修 (e367b19) |
| K3 | `principal_stresses`: 形状 (n,3) 未校验 → 裸 IndexError; NaN/Inf 静默传播 | stress.py | ✅ 已修 (f3d75b5) |
| K4 | `estimate_error`/`stress_at_point`: result 缺键 → 裸 KeyError | error_est.py, stress.py | ✅ 已修 (f3d75b5) |
| K5 | `solve(mesh, ...)`: mesh 非 Mesh → 裸 AttributeError | solver.py | ✅ 已修 (f3d75b5) |
| K6 | `parse_vec2`/`parse_traction`: 非 str 输入 → 裸 AttributeError/TypeError | loads_core.py | ✅ 已修 (f3d75b5) |
| K7 | `import_msh`: 文件不存在 → gmsh 异常冒泡 | gmsh_adapter.py | ✅ 已修 — 前置 FileNotFoundError (f3d75b5) |
| K8 | `compute_stresses`/`spr_recovery`/`nodal_*`: 输入数组 NaN → 静默 NaN 输出 | stress.py/spr.py | ✅ 已修 (c3d14c5) |
| K9 | DOF 数组校验三处重复 | mesh.py, bc.py | ✅ 已修 — `checks.require_dof_index_array` (mesh→TypeError / bc→ValueError 按既有锁定分流) (75bc6e5) |
| K10 | 载荷记录节点整数校验与 _validate_node_id 重复 | mesh.py | ✅ 已修 — 改调 _validate_node_id (e367b19) |
| K11 | 材料参数校验重复 (__post_init__ vs _validate_material_and_mesh) | mesh.py | ✅ 已修 — 共享标量 helper, 消息保留历史格式 (c3d14c5) |
| K12 | `von_mises`: 标量/1-D/末维≠3 → 裸 IndexError; NaN 静默 (fuzz 发现) | material.py | ✅ 已修 (自查轮, 判别性测试在场) |
| K13 | `bc_apply._apply_body_force`: 元组/ndarray 解包绕开 schema; ndarray 真值判据裸 ValueError | bc_apply.py | ✅ 已修 (23af2f1) |

## L. 阶段 3 架构决策记录

### L1. import 去环 (commit 059da97)

提升 11 处函数内 import 到模块顶部: predicates.warnings / topology.geometry 别名 /
gmsh_adapter.sys+Mesh+point_in_element / input_source.cli.is_batch_mode /
loads_core.ast+math / mesh.topology_core×2+warnings / preprocess.gmsh_adapter+element /
stress.spr / visualize.error_est / runner.element+quality+wizard / physical_mapping.preprocess
(后经测试发现是 patch 注入点, 已回退, 见下)。

**保留局部 import 及原因**:
| 位置 | 原因 |
|------|------|
| mesh.py `from .element import get_element_kernel` | 真环: element/base.py 局部 `from ..mesh import Mesh` — 提升任何一边都成环 |
| element/base.py `from ..mesh import Mesh` | 同上 (环的另一端) |
| runner.py `from .visualize import ...` | matplotlib 导入成本 ~1-2s, `--no-plot` 路径不付 (代价评估后保留) |
| physical_mapping.py `from ..preprocess import read_geo_groups` | 测试 patch 注入点: `patch("fem2d.preprocess.read_geo_groups")` 依赖调用时模块属性查找 — 直接名绑定复制旧引用使 patch 失效 (既有判别性测试锁定) |
| __init__.py `from .convergence import run_cantilever_convergence` | PEP 562 惰性导出 (避免 `python -m fem2d.convergence` runpy 警告, 注释在案) |

### L2. loads_schema.py 拆分 (commit 0b8985b)

`_load_component_ok` / `_check_load_pair` / `_check_load_scalar` 从 mesh.py 纯搬移到
`fem2d/loads_schema.py` (行为不变), mesh.py 引用, bc_apply 直接引用。
载荷形状校验 (体力 2 / 面力 2 / 压力 1 / 集中力 2) 收敛到一处。

> 行为冻结区 (element/ 内核公式、solver/error_est 数值逻辑) 全程未触碰。
> 阶段 2 每类校验 = 一个独立 commit, 每项附判别性测试 (放回旧实现必须失败),
> 每个 commit 前全量 pytest 0 失败 — 全程无红测试提交。
