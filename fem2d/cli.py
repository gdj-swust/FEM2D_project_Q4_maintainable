"""CLI 参数定义、判型与交互提示 — 原 run.py 拆分出来的参数层。

只承担"命令行界面"职责: argparse 定义、Abaqus 单元码 → 平面应力/
应变判型、EOF 安全的交互提示。程序默认值在 AnalysisConfig (config.py),
CLI 未指定的字段由 config 默认填充。
"""
import argparse
import sys

from .errors import CliError

# ── 平面态与单元码的对应 (Abaqus 命名: CPE=plane strain, CPS=plane stress) ──
_STRAIN_ELEMENT_TYPES = frozenset({"CPE3", "CPE4", "CPE4R", "CPE4I"})
_STRESS_ELEMENT_TYPES = frozenset(
    {"CPS3", "CPS4", "CPS4R", "CPS4I", "C2D3"})

def _resolve_plane_type(elem_type, requested_plane=None):
    """Resolve and validate plane behavior from an Abaqus element code."""
    element_code = str(elem_type).strip().upper()
    if "," in element_code:
        raise ValueError(
            f"Mesh contains mixed element types: {elem_type}. "
            "Use one homogeneous element code per analysis.")
    if requested_plane is None:
        return (
            "strain"
            if element_code in _STRAIN_ELEMENT_TYPES
            else "stress"
        )
    if element_code in _STRAIN_ELEMENT_TYPES and requested_plane == "stress":
        raise ValueError(
            f"{element_code} (plane-strain) mesh cannot use --plane stress. "
            f"Use --plane strain or omit --plane for auto-detection.")
    if element_code in _STRESS_ELEMENT_TYPES and requested_plane == "strain":
        raise ValueError(
            f"{element_code} (plane-stress) mesh cannot use --plane strain. "
            f"Use --plane stress or omit --plane for auto-detection.")
    return requested_plane


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='FEM2D — CST/Q4 二维有限元分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python run.py demo.spec
  python run.py models/plate_q4.geo --fix left --body 0,-78000
  python run.py models/plate_q4.geo --fix left --traction right:1e6,0
  python run.py models/plate_q4.geo --fix "left,right" --body 0,-78000 --no-plot
        ''')
    p.add_argument('mesh', nargs='?', default=None,
                   help='几何文件 (.geo) / 网格 (.msh) / 描述 (.txt) 或 规格文件 (.spec)')
    p.add_argument('--wizard', action='store_true', default=None,
                   help='交互式建模向导 (无参数且终端可用时自动进入)')
    p.add_argument('--plane', '-p', choices=['stress', 'strain'], default=None,
                   help='平面应力/平面应变 (不指定时按网格单元码自动判定: '
                        'CPE→strain, CPS/C2D→stress)')
    p.add_argument('--E', type=float, default=None, help='弹性模量 [Pa] (默认: 2.10e11)')
    p.add_argument('--nu', type=float, default=None, help='泊松比 (默认: 0.3)')
    p.add_argument('--lc', type=float, default=None, help='网格密度 (覆盖 .geo 中的 lc 值)')
    p.add_argument('--thickness', '-t', type=float, default=None, help='厚度 [m] (默认: 0.01)')
    p.add_argument('--body', '-b', default=None,
                   help='体力 bx,by [N/m3] 支持含x/y的表达式 (例: --body "0,-78000" 或 --body "0,-1000*(1-y/2)") ')
    p.add_argument('--fix', default=None,
                   help='固定边 (例: --fix left 或 --fix "left,right" 或 --fix 1,3)')
    p.add_argument('--fix-ux', default=None, help='Ux=0 的边')
    p.add_argument('--fix-uy', default=None, help='Uy=0 的边')
    p.add_argument('--traction', default=None,
                   help='面力 (例: --traction right:1e6,0 或 --traction 2:1e6,0)')
    p.add_argument('--force', default=None,
                   help='集中力 target,fx,fy；target 可为 Gmsh 节点号或 Physical Point 名称')
    p.add_argument('--no-plot', action='store_true', default=None, help='不显示交互式云图')
    p.add_argument('--keep-open', action='store_true', default=None,
                   help=(
                       '批处理命令 (CLI 传 BC) 计算完成后仍进入交互云图窗口 '
                       '(窗口停留, 键盘切换云图, q 退出; 可配 --save 先保存'
                       '再交互; 与 --no-plot 互斥; 无 BC 参数时本就交互, '
                       '无变化)'))
    p.add_argument('--list-boundaries', action='store_true', default=None,
                   help='仅列出边界分段后退出 (不求解)')
    p.add_argument(
        '--require-physical-groups', action='store_true', default=None,
        help='要求至少恢复一个 Physical Curve；禁止退化为纯几何边界')
    p.add_argument(
        '--strict-boundary', action='store_true', default=None,
        help='边界/CAD 语义存在内部边、缺边、重叠实体时立即终止')
    p.add_argument('--check-cond', action='store_true', default=None, help='估计刚度矩阵条件数 (Bathe §8.2.6)')
    p.add_argument(
        '--linear-solver',
        choices=['auto', 'direct', 'cg', 'cg-block', 'ilu'],
        default=None,
        help=(
            '线性求解器：auto(大型模型自动PCG)、direct(SuperLU)、'
            'cg/cg-block(Jacobi-PCG)、ilu(ILU-PCG)'
        ),
    )
    p.add_argument(
        '--error-method',
        choices=['auto', 'spr', 'l2', 'weighted'],
        default=None,
        help='误差恢复：auto(大网格用快速面积加权)、spr、l2 或 weighted',
    )
    p.add_argument('--debug', action='store_true', default=None,
                   help='顶层异常显示完整 traceback (默认只打印错误摘要)')
    p.add_argument('--self-test', action='store_true', default=None,
                   help='求解前运行当前单元的 patch test (仅开发/CI)')
    p.add_argument('--save', '-o', default=None, help='保存云图到文件')
    p.add_argument(
        '--output-dir', default=None,
        help='生成物输出目录 (.msh 与临时文件; 默认: 输入文件同目录)。'
             '只读示例目录/共享目录可用此参数指定可写位置')
    p.add_argument('--band-min', type=float, default=None, help='Isoband 应力带下限 (Pa)')
    p.add_argument('--band-max', type=float, default=None, help='Isoband 应力带上限 (Pa)')
    p.add_argument('--band-step', type=float, default=None, help='Isoband 应力带步长 (Pa)')
    p.add_argument('--band-tag', default=None,
                   choices=('vm', 'sx', 'sy', 'txy', 's1', 's2', 'taumax'),
                   help='固定带宽适用的应力分量 (默认: vm)')
    p.add_argument('--quad', action='store_true', default=None,
                   help='用 Gmsh 重组生成四边形网格（仅 .geo/.txt 输入）')
    p.add_argument('--jump-ref', type=float, default=None,
                   help='Traction Jump 固定参考应力 [Pa]; 跨网格对比时传入同一值')
    p.add_argument('--elem-type', default=None,
                   choices=['CST','Q4','Q4R','Q4I'],
                   help='覆盖单元类型: CST=常应变三角(教学基础) | '
                        'Q4=全积分四边(稳健保守) | '
                        'Q4I=非协调四边(综合最佳, 默认推荐) | '
                        'Q4R=减缩积分四边(专用: 规则网格/长宽比<10/'
                        '膜主导或 CPS4R 兼容)')
    return p.parse_args(argv)
def ask(prompt: str) -> str:
    """EOF/Ctrl-C 安全的交互提示 — 不泄漏 traceback。

    脚本/管道/CI 中以 pty 运行且无输入时, 裸 input() 会以
    EOFError 崩溃; 各调用点对空串已有"跳过/默认"语义。
    Ctrl-C 与 EOF 对称: 向导 banner 承诺"随时 Ctrl-C 退出", 裸
    KeyboardInterrupt 会泄漏整段 traceback 且退出码 130。
    """
    try:
        return input(prompt).strip()
    except EOFError:
        print("  [exit] 标准输入已关闭 — 按未输入处理。")
        return ""
    except OSError:
        # GUI 嵌入 (exe 双击, 无控制台) 时 stdin 句柄无效 —
        # input() 抛 OSError(9 Bad fd) 而非 EOFError, 与 EOF 同样
        # 按未输入处理 (曾静默泄漏导致 GUI 识别线程"没有结果")
        print("  [exit] 标准输入不可用 — 按未输入处理。")
        return ""
    except KeyboardInterrupt:
        raise CliError("\n  [INFO] 已退出 (Ctrl-C)", exit_code=0)


def is_batch_mode(config) -> bool:
    """批处理判定: CLI 提供了任何 BC/载荷 或 stdin 不是终端 → 不交互.

    接收 AnalysisConfig (类型化配置) 而非 argparse Namespace —
    执行层不感知 CLI 参数表.
    """
    try:
        tty = sys.stdin.isatty()
    except (OSError, ValueError, AttributeError):
        # stdin 无效/缺失 (exe 双击无控制台) — 无法交互 → 按批处理,
        # 否则调用方会走 ask() 的 input() 抛 OSError 静默失败
        tty = False
    return (not tty
            or bool(config.fix or config.fix_ux or config.fix_uy
                    or config.traction or config.force or config.body))
