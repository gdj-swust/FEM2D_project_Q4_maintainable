# FEM2D — CST + Q4 2-D Elasticity Solver

二维小变形、各向同性线弹性有限元程序，支持：

- `CPS3` / `CPE3` / `C2D3`：三节点常应变三角形（CST）
- `CPS4` / `CPE4`：四节点双线性等参四边形（Q4，2×2 full integration）
- 平面应力与平面应变
- 集中力、面力、压力和体力
- Z2/L2/SPR 应力恢复、牵引跳跃与残差型加密指标
- 三角形与四边形网格、云图、等应力带和变形图

## Quick start

```bash
# 依赖单一来源: pyproject.toml [project].dependencies / [project.optional-dependencies].dev
pip install -e .[dev]

# 需要 Gmsh 可执行文件 (CLI .geo 生成走子进程 gmsh.exe, 设 GMSH_PATH
# 或放 tools/ 下) + gmsh Python 包 (pip install gmsh — 子进程产物由
# Gmsh API 读回, 恢复物理组/CAD 语义):
python run.py models/plate_q4.geo --quad --fix left --traction right:1e6,0
# 输入统一走 Gmsh 几何 (.geo) — Abaqus .inp 输入口已移除 (2026-08):
python run.py models/plate_q4.geo --fix left --traction right:1e6,0
python run.py models/demo.spec --no-plot
python run.py --self-test

# 交互式建模向导 (2026-08 新增): 终端无参数直接进入, 问答完成
# 几何/网格/材料/边界/体力建模, 无需写任何文件:
python run.py            # 终端 → 自动进入向导
python run.py --wizard   # 显式触发
```

网格中的 `CPE3/CPE4/CPE4R/CPE4I` 单元会自动选择平面应变，`CPS3/CPS4/C2D3`
会自动选择平面应力（平面态可由 `--plane` 或 `.spec` 覆盖）。一个分析网格
必须使用同一种单元代码；当前不支持三角形/四边形混合块或高阶单元。
减缩积分单元 (Q4R) 与非协调模式单元 (Q4I) 均已支持。

> ⚠️ **Q4R 适用限制**：单点减缩 + 沙漏稳定在规则网格上表现优异，但
> 在薄板/少行网格上不稳定 —— 单元长宽比 ≥ 50 时过刚（解只有解析值的
> ~2%），`L/h ≈ 10` 少行时过柔（约 5 倍），且沙漏能占比在这两种情况下
> 都 > 90%（占比不是可靠性指标）。**仅限规则网格、长宽比 < 10 的场合**；
> 一般问题默认推荐 Q4I（`CPS4I/CPE4I`），其全长宽比稳定。

## Q4 mesh generation

```bash
python run.py models/plate_q4.geo \
  --quad --fix left --traction right:1e6,0
```

`--quad` 只在从 `.geo` / `.txt` 生成新网格时生效。生成器不会修改原始
`.geo`；Gmsh 输出原生 `.msh`（含 PhysicalNames/实体语义），经 Gmsh API
导入并确认是纯四边形后再进入求解。不完整重组产生的三角形或三角/四边
混合网格会被明确拒绝。

四边形重组依赖 Gmsh。简单四边面通常可直接重组；带孔或复杂拓扑可能
需要先在 Gmsh 中分区，或使用 transfinite mesh。`models/README.md`
列出了 CPS4、CPE4 和 `.geo --quad` 的可执行示例。

> ⚠️ **.geo 是"可信、可执行式输入"**：Gmsh 脚本是编程语言，支持
> `SystemCall` 等可执行任意系统命令的指令（本程序已在清洗阶段对
> `SystemCall` 黑名单拦截并明确报错）。**只应运行自己编写的 .geo**（含
> 其 `Include` 引用的文件）；第三方来源的 .geo 请先人工审查。本程序不
> 承诺对 .geo 内容做沙箱隔离。

## 目录结构

```
FEM2D_project_Q4/
├── run.py                 # CLI 源码入口 (转发到 runner.main)
├── run_demo.py            # demo_complex 演示 (椭圆孔板, 复用正式 API)
├── pyproject.toml         # 依赖/打包/工具链配置单一来源
├── requirements.txt       # 兼容薄壳 (指向 pyproject)
├── README.md / ARCHITECTURE.md / architecture.tex+pdf / LICENSE
├── fem2d/                 # 求解器核心 (30 模块 + boundary/ 13 + element/ 7)
│   ├── mesh.py            # Mesh 容器 (只读几何 + validate_state)
│   ├── solver.py          # 求解主流程 (阶段函数: 求解/残差/平衡/检查)
│   ├── assembly.py        # 稀疏装配
│   ├── bc.py / bc_apply.py# 约束施加 / BC-载荷装配 (每载荷类型一阶段函数)
│   ├── loads_core.py      # 载荷积分 (3 点 Gauss, AST 白名单表达式)
│   ├── stress.py / spr.py / error_est.py   # 应力恢复 / SPR / Z2-残差指标
│   ├── element/           # 单元内核 (注册表模式, 加新单元零改动)
│   │   ├── base.py        # ElementKernel 协议 + registry + 退化度量 hook
│   │   └── cst.py / q4.py / q4r.py / q4i.py
│   ├── boundary/          # 边界系统 (环构建/嵌套/曲线分类/物理组映射)
│   ├── gmsh_adapter.py    # Gmsh API 读回 (import_msh 恢复物理组语义)
│   ├── input_source.py    # 输入解析链 (.geo/.txt/.msh/.spec → 网格)
│   ├── preprocess.py      # 网格校验 validate_mesh + .geo @FEM 配置
│   ├── config.py / cli.py / runner.py / reporting.py
│   ├── visualize.py       # 云图 (flat/gouraud/isoband/scalar_jump)
│   ├── verification.py    # 平面应力/应变解析对照 (--self-test)
│   └── topology_core.py / quality.py / patch_test.py / convergence.py / material.py
├── scripts/               # 工具层 (14 .py + 1 .sh)
│   ├── gmsh_runner.py     # 子进程 gmsh: .geo → .msh (300s 超时 + 原子发布)
│   ├── geo_spec.py        # 中文文本描述 → .geo 生成
│   ├── check_dead_code.py / check_imports_deep.py   # 自制静态检查器
│   └── convergence_study.py / test_complex.py / make_test_spec.sh
├── tests/                 # 142 个测试文件 + conftest.py, 1793 测试
│                         #   (本机 2026-08-06 实测 1793 passed + 0 skipped,
│                         #    覆盖率 98.0%;
│                         #    无 Gmsh 环境的 collected/skip 以实测为准)
├── models/                # 算例库: 21 .geo + 7 .spec + 3 .txt
└── tools/gmsh-4.15.2-Windows64/gmsh.exe   # Gmsh 可执行文件 (捆绑)
```

> 打包惯例 (用户约定): `scripts/make_release_zip.py` 默认产出单 zip
> `FEM2D_project_Q4_<版本>_<时间戳>_maintainable.zip` 放 Downloads
> (版本号来自 pyproject.toml, 单一源), 不含 `tools/`、缓存、测试产物与
> `*.egg-info`。`--split` 产出 4 个分包 (同一版本号/时间戳/顶层目录,
> 解压到同一目录即还原完整项目, 守恒: 4 包文件并集 == 原单包清单):
>   ..._source.zip           源码包 — fem2d/scripts/tests/docs, 可 `pip install .` + `pytest`
>   ..._runtime-win64.zip    运行包 — 运行必需代码 + tools/ (gmsh.exe), 解压即 `python run.py ...`
>   ..._models.zip           示例模型包 — models/ 全部 (.geo/.spec/.txt/.msh)
>   ..._testdata.zip         测试/基准数据包 — boundary_golden + 收敛/性能基准数据
> `--full` 单 zip 捆绑 `tools/gmsh-4.15.2-Windows64/gmsh.exe` (Windows 64
> 位; 约 86 MB), 须保留其 GPL v2+ 许可证与版权声明 (https://gmsh.info);
> 其余平台可删该目录并自行放置 gmsh 或设 `GMSH_PATH` (SHA256:
> `317c43391e5b1fab3a1dd80dc5245dad6e2d087910f4b8ebc234bd6d4b8f41a1`)。
> `pip install -e .[dev]` 后 `fem2d` console script 可用。

## Implementation map

| Module | Responsibility |
|---|---|
| `fem2d/element/base.py` | ElementKernel 协议 + 注册表 + 载荷求值/退化度量 hook |
| `fem2d/element/cst.py` | CST kernel（几何/刚度/响应/点定位） |
| `fem2d/element/q4.py` | Q4 形函数、Jacobian、B、2×2 Gauss、刚度与体力 |
| `fem2d/element/q4r.py` | Q4R 减缩积分 + 沙漏稳定 |
| `fem2d/element/q4i.py` | Q4I 非协调模式（含逐 Gauss 点退化检查） |
| `fem2d/element/registry.py` | 单元注册与别名解析 |
| `fem2d/material.py` | 本构矩阵与归一化 von Mises |
| `fem2d/assembly.py` | 与节点数无关的稀疏组装 |
| `fem2d/bc.py` | 位移约束的消去法与乘大数法 |
| `fem2d/loads.py` / `loads_core.py` | 等效节点载荷（体力/面力 3 点 Gauss）与表达式解析 |
| `fem2d/preprocess.py` | 网格校验 validate_mesh（12 个独立校验步骤）+ .geo @FEM 配置 |
| `fem2d/mesh.py` | Mesh 容器（只读几何 property + validate_state + 拓扑缓存） |
| `fem2d/topology_core.py` | 向量化拓扑（邻接/边表/点定位 locator） |
| `fem2d/quality.py` | 网格质量评分 |
| `fem2d/boundary/topology.py` | 边界闭环、内外层级、G1 分段与方向 |
| `fem2d/boundary/geometry.py` | 曲率、圆/椭圆拟合及基元分类 |
| `fem2d/boundary/predicates.py` | Shewchuk 自适应精度 `orient2d` |
| `fem2d/boundary/naming.py` | Physical Curve 与边名称解析 |
| `fem2d/boundary/registry_mapping.py` / `segment_builder.py` | Physical 组→边分段映射与段构建 |
| `fem2d/regions.py` | Physical Point/Curve/Surface 到节点/边/单元的映射 |
| `fem2d/gmsh_adapter.py` | Gmsh API 读回（import_msh 恢复物理组/CAD 语义） |
| `fem2d/stress.py` | kernel 响应、L2 投影、点/双侧应力查询 |
| `fem2d/spr.py` | CST 质心 / Q4 Gauss-Barlow 点 patch recovery |
| `fem2d/error_est.py` | Z2 与显式残差型指标（三点 Gauss 边界残差） |
| `fem2d/visualize.py` | 云图（flat/gouraud/isoband/scalar_jump 分支函数） |
| `fem2d/patch_test.py` | CST 与 Q4 三个独立常应力 patch tests |
| `fem2d/convergence.py` | CST/Q4/Q4R 悬臂梁收敛性研究 |
| `fem2d/cli.py` | CLI 参数定义、优先级合并、平面态判型、`--debug` 异常边界 |
| `fem2d/config.py` | AnalysisConfig 类型化配置（构造/合并后统一校验） |
| `fem2d/input_source.py` | 输入源解析：`.spec`/`.geo`/`.txt` → `.msh`（lc 临时副本 + Gmsh 生成 + import_msh） |
| `fem2d/bc_apply.py` | BC/载荷装配（固定/面力/压力/集中力/体力 — 每载荷类型独立阶段函数） |
| `fem2d/verification.py` | 平面应力/应变解析对照（--self-test 用，不依赖 tests 包） |
| `fem2d/reporting.py` | 中文结果摘要与物理建议警告（体积自锁/CST 弯曲/误差） |
| `fem2d/runner.py` | 主流程编排：输入 → 网格 → 边界 → BC/载荷 → 求解 → 报告 → 云图 |
| `scripts/gmsh_runner.py` | CLI .geo 生成主路径（子进程执行 + 300s 超时 + stripped 副本 + 原子发布） |

> 说明：`run.py` 只转发到 `fem2d/runner.py`（编码安全网在 runner.main 内，
> `fem2d` console script 与源码运行共用）。CLI 参数在 `fem2d/cli.py`，
> 配置合并/校验在 `fem2d/config.py`，输入解析在 `fem2d/input_source.py`。

## Verification

```bash
pytest -q
python run.py --self-test
python -m fem2d.convergence
```

`test_gmsh_topology_adapter.py` 使用确定性的 Gmsh API 替身验证完整映射，
不依赖本机 Gmsh。依赖真实可执行文件的端到端边界测试会在 `gmsh`
不在 `PATH` 时标记为跳过。

Q4 验证覆盖：不规则四单元 patch test（逐个检查全部 Gauss 点）、
仿射场四 Gauss 点精确性、
稀疏/参考组装一致性、体力合力、全局力矩平衡、CPS4/CPE4 读取、
CW 单元拒绝（方向校验）、Jacobian 质量检查、L2/SPR/Z2、
四边形绘图路径，以及使用精确抛物线面力积分的 Q4 网格收敛性。

## CLI 诊断参数

- `--debug`：顶层异常显示完整 traceback（默认只打印错误摘要）— 排查 Gmsh
  缺失/网格解析失败等运行时错误时使用。
- `--self-test`：不带网格时运行四单元 patch test + 平面应力/应变解析对照。
- `--list-boundaries`：仅列出边界分段后退出。

## 输出位置 (--output-dir)

默认情况下，`.geo`/`.txt` 生成的 `.msh` 与临时文件都写在**输入文件同目录**。
把模型放在 U 盘、共享目录或只读示例目录时，写入会失败。此时用
`--output-dir` 把生成物（`.msh`、临时几何/网格文件）指到可写目录：

```bash
python run.py models/plate_q4.geo --fix left --traction right:1e6,0 \
  --output-dir ./work
# .msh 写入 ./work/plate_q4.msh; 源 .geo 不被修改, 输入目录无残留
```

规则：

- `--output-dir` 不存在时自动创建；创建/写入失败（只读目录、权限拒绝）
  会给出清晰错误 "输出目录不可写 — 请用 --output-dir 指定可写位置"，
  不会以裸 traceback 结尾。
- `.txt` 输入的生成 `.geo` 也写入输出目录（与 `.msh` 同 basename）；
  `@FEM` 注解与源 `.geo` 路径语义不变。
- `.spec` 中可用 `output_dir = <路径>` 键（相对路径以 `.spec` 所在目录
  为基准）；CLI `--output-dir` 显式参数优先。
- 只对 `.geo`/`.txt` 网格生成生效；直接输入 `.msh` 时忽略并 WARN。
- 含相对 `Include` 的 `.geo`，其临时几何副本必须留在源目录（相对引用
  以所在目录解析），`--output-dir` 此时只作用于 `.msh` 输出并给出提示。

**同名 .msh 覆盖保护**：目标 `.msh` 已存在时，本程序生成的（带内部
标记）照常覆盖；无法确认来源的（手写/其他工具产物）不会覆盖 —
WARN 提示后改写到临时文件，原文件保留，需要时用 `--output-dir` 换目录。

## Boundary detection policy

从 `.geo` 运行时，CLI 主路径是**原生 Gmsh 可执行文件**（子进程方式，
`scripts/gmsh_runner.py`，stripped 副本 → 拓扑校验 → 原子发布，且避免
Windows 下高密度重组经 Python API 的栈限制）。Gmsh 自己解释
OpenCASCADE、Boolean、Spline、宏和实体编号，生成的原生 `.msh`
（含 PhysicalNames/实体语义）经 Gmsh API 导入后在边界映射层恢复
三类有维度的区域：

- `Physical Point` → 网格节点，可直接作为 `--force` 的唯一目标；
- `Physical Curve` → 有序边链及节点，`--fix`、`--fix-ux`、
  `--fix-uy`、`--traction` 和压力均通过这些精确边施加；
- `Physical Surface` → 单元集合，并报告由 FEM 积分得到的区域面积。

同一条边可以同时属于细粒度组和汇总组（例如 `hole_1` 与
`all_holes`）。映射层保存重叠成员关系，但载荷积分拓扑中每条边只出现
一次。子进程路径强制 `Mesh.SaveAll=1`，`.msh` 中的多值 Physical
Curves 由 API 导入保留重叠成员关系。程序会报告恢复的组数和语义边
覆盖率。需要禁止纯几何降级时使用
`--require-physical-groups`；需要把内部 Physical Curve、CAD 实体重叠、
缺边等语义问题直接升级为致命错误时使用 `--strict-boundary`。
报告采用“已映射/已声明”的组数；即使 Physical Curve 的某些边随后
因不属于位移网格而被丢弃，组名和失败原因也会保留，严格模式不会再
把它静默忘掉。

自动检测仍保留为无物理组的 `.geo`/外部网格的兜底：先从无向边构造
闭环，再用稳健的 `orient2d` 射线法确定环的嵌套层级。边界策略为：

1. 语义恢复路径（Gmsh API `generate_from_geo`，或子进程 `.msh` 经
   `import_msh` 导入的物理组）先保存所有活动 Surface 的 CAD Curve
   entity，即使实体
   没有 Physical Group。CAD entity tag 是不可跨越的硬分段键，因而两条
   相邻样条、圆弧或直线不会被重新粘接后再靠曲率阈值猜测。
2. 存在 `Physical Curve` 时，以每条网格边的完整物理组成员集合为语义
   成员键；成员集合变化也会切段。相邻的 `load_a` /
   `load_b` 不会因为共线而合并，`hole_1` / `all_holes` 等重叠组也不会
   重复生成积分边。
3. CAD entity type 只描述来源，不强迫有限元直边链显示为某种基元；
   `line/arc/ellipse/curve` 仍由网格坐标的全点检查决定。这样既保留 CAD
   分段，又不会把粗糙或退化的 Circle 网格误当作精确圆。
4. 没有 CAD 实体图的降级路径，只在尖角、持续共线区间和显著
   曲率阶跃处分段；
   普通曲率极值与光滑拐点不是断点。
5. 直线、圆和椭圆只有通过严格的全点残差检查才使用解析标签；其余
   形状以 `curve` 保存完整有序节点链，并提供长度、曲率和拐点信息。

边界角色不再从“外边/内孔”显示文字推断。拓扑闭环显式携带
`loop_id`、`loop_depth` 和 `is_outer`；Physical Curve 的名称即使含有
“内孔”二字也不能改变它的真实拓扑角色。导入还会检查：

- 边界节点度数不是 2 的开放/非流形拓扑；
- 同一闭环自交、非相邻相触或共线重叠；
- 不同闭环之间相交、相切或重叠；
- Physical Curve 落在 2-D 网格内部、部分丢失、包含未网格化 CAD
  entity，或引用不存在的边；
- 完整 Gmsh API 模式下，每条外边界网格边必须有且仅有一个外部 CAD
  Curve entity；共享 CAD 接口必须保持为内部网格边；
- 一条边同时被多个 CAD Curve entity 声称；
- Physical Curve 名称仅大小写不同、纯数字、含控制字符，或包含 CLI
  保留分隔符 `,;:`。

完整 API 注册表中的 CAD/网格拓扑矛盾属于不可恢复错误，即使没有指定
`--strict-boundary` 也会阻断求解，程序不会悄悄改用几何猜测。未参与
二维位移网格的构造 Surface 会被排除，不会产生幽灵边界。Physical
Curve 的一般语义错误仍由 `--strict-boundary` 或
`--require-physical-groups` 升级为致命错误。

CLI 中的 Physical Curve 名称默认采用不区分大小写的完整匹配。例如
`load` 不会扩大为 `load_a` 与 `load_b`。需要查找旧标签时可显式使用
`~关键词`；若命中多个不同候选，程序会要求改用完整名称。
`Load` 与 `load` 同时出现会在导入阶段直接判为歧义，而不是把两组约束
或载荷悄悄合并。

线性 `:l` 和抛物线 `:p` 面力按 Physical Curve 所选有序边链的累计弧长
参数施加，不再按全局 x/y 范围猜测。因此斜边、圆弧、回折线和一般样条
的分布一致。一个名称跨越多个相邻 CAD entity 时会先重建连通链，再沿
整条链参数化；不同连通分量分别参数化。闭合链上的线性分布在接缝处不
唯一，程序会拒绝并要求拆成开放 Physical Curves。边链重建使用邻接表，
确定性排序后的时间复杂度为 O(E log E)，密集曲线不会退化为二次扫描。

闭环状态来自边界邻接图或 Gmsh 有序边链，并显式传给几何分类；首尾坐标
距离只用于没有拓扑上下文的独立几何调用。因此大坐标/大网格圆环不会因
浮点闭合容差漏检，接近整圆的开放圆弧也不会被误判为闭环。

8 个以上采样基元组成的粗离散圆可恢复为闭合圆弧；如果每条弦又包含
多个共线网格节点，会先提取弦端点再拟合，避免半径向内偏小。由于仅凭
坐标无法区分“刻意设计的正八边形”和“8 点离散圆”，这类语义有歧义
时应优先在 Gmsh 中定义 `Physical Curve`，其名称和分组优先于几何猜测。

“通用曲线”表示求解器忠实保留网格上的离散边链，并不声称能从
网格坐标唯一反推出原始 CAD 的 NURBS/Bezier 参数。需要精确 CAD 语义时应
保留 Gmsh `Physical Curve`；没有语义信息时，保真退化优先于错误拟合。
当前 CST/Q4 都是直边一阶单元，因此曲线长度、压力法向和面积采用离散
网格几何；若需要 CAD 参数域上的精确曲线积分，应使用高阶曲边单元
（T6/Q8/Q9）或另行实现 CAD 参数映射，不能把拟合标签当成高阶几何。

推荐在 `.geo` 中把物理意义写成名称，而不是依赖坐标容差：

```c
Physical Curve("fixed") = {leftCurve};
Physical Curve("traction") = {rightCurve};
Physical Surface("domain") = {surfaceTag};
Physical Point("load_tip") = {tipPoint};
```

对应命令可以直接写为：

```bash
python run.py model.geo \
  --fix fixed --traction traction:1e6,0 \
  --force load_tip,0,-1000 --no-plot
```

## References

- Bathe, K. J. *Finite Element Procedures*, 2nd ed., §§4.3.6, 5.3
- Zienkiewicz, O. C. & Zhu, J. Z. (1987), superconvergent patch recovery
- Sussman, T. & Bathe, K. J. (1987), stress-band mesh assessment
