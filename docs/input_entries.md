# FEM2D 输入入口对照表

> 版本: 2026-08-05 (D 易用性轮)。本文件回答: "我该用哪个入口？" —
> 5 条输入入口各一句话 + 典型命令。程序内引导: `run.py --help` 尾部
> 追加同一指南; 输入报错 (`.inp`/未知扩展名) 引用本表。

## 1. 对照表 (一句话一个入口)

| 入口 | 一句话 | 典型命令 |
|------|--------|----------|
| `run.py` | 标准入口 — 任意模型文件 (`.geo`/`.msh`/`.txt`/`.spec`), 走完整求解流程 | `python run.py models/plate_q4.geo --fix left --body 0,-78000` |
| `run_demo.py` | 演示入口 — 内置示例, 免模型文件一键看结果 | `python run_demo.py` |
| `fem2d` CLI | 已安装包入口 — pip 安装后与 `run.py` 等价的命令行 | `fem2d models/plate_q4.geo` |
| `.spec` 文件 | 参数化入口 — 材料/载荷/网格写进配置文件, 多工况可复现 | `python run.py batch1.spec` |
| gmsh `.geo` | 网格生成入口 — 只用 Gmsh 出网格, 再交给 `run.py` | `gmsh model.geo -2 -o model.msh` → `python run.py model.msh` |

## 2. 按场景选择

| 场景 | 选哪条 |
|------|--------|
| 第一次使用 / 只想快速看结果 | `python run_demo.py` 或 `python run.py <文件>` |
| 有现成 `.geo` (手写或 Gmsh 画) | `python run.py <文件.geo>` (自动调 Gmsh 网格化) |
| 只有 `.msh` (网格已生成) | `python run.py <文件.msh>` (直接导入, 不再网格化) |
| 中文描述建模 ("矩形 长=1 高=1") | `python run.py <文件.txt>` |
| 多工况 / 需要复现同一组参数 | 写 `.spec`, `python run.py <文件.spec>` |
| 只想要网格, 不解算 | `gmsh <文件.geo> -2 -o <文件.msh>` (或 Gmsh GUI 导出) |
| 已 pip 安装, 不用源码 | `fem2d <文件>` (与 `python run.py` 等价) |

## 3. 常见入口误用 (报错即引导)

| 误用 | 报错含引导 |
|------|-----------|
| `.inp` 文件传入 (Abaqus 输入口已移除) | 提示改传 `.geo`/`.msh`/`.spec` + 入口指南 |
| `.inp` 内容其实是 Gmsh `.geo` 脚本 | 提示"该文件内容是 .geo 脚本 — 用 run.py 或 gmsh" |
| 未知扩展名 (如 `.dat`/`.fem`) | 提示仅支持 `.spec/.geo/.txt/.msh` + 入口指南 |

## 4. 输出编码策略 (Windows 中文)

- 程序输出统一 **UTF-8** (交互控制台本就 UTF-8; 重定向/管道输出由
  `run.py` 强制 UTF-8, 不随系统代码页退化)。
- 若在 cmd/PowerShell 中重定向后用 `type` 查看出现乱码: 控制台代码页
  是 GBK 而文件是 UTF-8 — 先 `chcp 65001` 再查看, 或直接用编辑器
  (记事本/VS Code) 打开。
