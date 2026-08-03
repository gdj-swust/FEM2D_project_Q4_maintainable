"""AnalysisConfig — 一次分析的完整设置 (类型化配置对象).

一流求解器 (Abaqus/CalculiX) 的 CLI 层与执行层通过"分析任务对象"
解耦: CLI/.spec/.geo 配置统一合并进一个配置对象, 执行层 (runner/
bc_apply/reporting) 只消费配置对象, 不感知 argparse 或文件格式。

来源优先级 (与 run.py 历史语义一致):
  程序默认 < .spec/.geo 配置 < CLI 显式参数

``AnalysisConfig.from_args`` 把 argparse Namespace 转为配置对象:
args 中未指定 (None) 的字段保持本类的程序默认值。
"""
import numbers
from dataclasses import dataclass, field, fields

import numpy as np

from .checks import require_finite_positive, require_nu_valid


@dataclass
class AnalysisConfig:
    """一次 FEM 分析的全部设置 — CLI 参数的类型化载体.

    字段默认值 = 程序默认;
    ``plane`` 默认 None = 读网格后按 CPE/CPS 单元码自动判型。
    """

    # ── 输入与网格 ──
    mesh: str = ""                       # 输入文件 (.geo/.msh/.txt/.spec)
    wizard: bool = False                 # 交互式建模向导 (无参数+终端时自动)
    lc: float = None                     # 网格密度覆盖 (仅 .geo 输入)
    quad: bool = False                   # Gmsh 四边形重组
    elem_type: str = None                # 单元类型覆盖 (CST/Q4/Q4R/Q4I)

    # ── 材料与物理 ──
    E: float = 2.10e11                   # 弹性模量 [Pa]
    nu: float = 0.3                      # 泊松比
    thickness: float = 0.01              # 厚度 [m]
    plane: str = None                    # stress/strain; None = 自动判型

    # ── 边界条件与载荷 ──
    fix: str = ""                        # 固定边 (Ux=Uy=0)
    fix_ux: str = ""                     # 仅 Ux=0 的边
    fix_uy: str = ""                     # 仅 Uy=0 的边
    traction: str = ""                   # 面力规格 (edge:tx,ty[:p|l|n])
    force: str = ""                      # 集中力 (target,fx,fy)
    body: object = None                  # 体力 (bx,by) 或表达式字符串

    # ── 求解器 ──
    linear_solver: str = "auto"          # auto/direct/cg/cg-block/ilu
    check_cond: bool = False             # 估计刚度矩阵条件数
    error_method: str = "auto"           # auto/SPR/L2/weighted
    self_test: bool = False              # 求解前运行 patch test + 验证
    jump_ref: float = None               # 牵引跳跃固定参考应力

    # ── 边界语义 ──
    require_physical_groups: bool = False
    strict_boundary: bool = False

    # ── 诊断 ──
    debug: bool = False                  # 顶层异常显示完整 traceback

    # ── 输出 ──
    no_plot: bool = False                # 抑制交互云图窗口
    save: str = ""                       # 保存云图到文件
    list_boundaries: bool = False        # 仅列出边界后退出
    band_min: float = None               # Isoband 固定带宽下限
    band_max: float = None               # Isoband 固定带宽上限
    band_step: float = None              # Isoband 固定带宽步长
    band_tag: str = "vm"                 # 固定带宽分量

    # 内部: from_args 记录 CLI 显式指定的字段 (用于配置优先级判断)
    _explicit: frozenset = field(default=frozenset(), init=False, repr=False)

    def __post_init__(self):
        self.validate()

    def validate(self):
        """配置校验 — 非法值在分析开始前明确报错.

        构造时自动调用; 合并 (如 .spec 覆盖) 后需显式再调 —
        曾因合并发生在构造之后而绕过校验, 负 E/非法 ν 进入求解.
        """
        self._validate_scalar_params()
        self._validate_body()
        self._validate_solver_and_post()
        self._validate_band_args()

    def _validate_scalar_params(self):
        if self.E is not None:
            require_finite_positive(self.E, "E")
        if self.nu is not None:
            require_nu_valid(self.nu, "nu")
        if self.thickness is not None:
            require_finite_positive(self.thickness, "thickness")
        if self.lc is not None:
            require_finite_positive(self.lc, "lc")
        if self.jump_ref is not None:
            require_finite_positive(self.jump_ref, "jump_ref")
        if self.plane not in (None, "stress", "strain"):
            raise ValueError(
                f"plane='{self.plane}' — 仅支持 stress 或 strain")

    def _validate_body(self):
        if self.body is None or isinstance(self.body, str) \
                or callable(self.body):
            return
        # 程序化体力: 必须恰好 2 个分量, 每分量 callable 或有限数值
        if not isinstance(self.body, (tuple, list)) or len(self.body) != 2:
            raise ValueError(
                f"body={self.body!r} — 必须为 (bx, by) 二元组或表达式字符串")
        for b in self.body:
            if callable(b):
                continue
            # numbers.Real 涵盖 np.float32/np.int64 等标量 (isinstance
            # (int, float) 会误拒); bool 是 Real 子类, 先排除
            if isinstance(b, bool) or not isinstance(b, numbers.Real) \
                    or not np.isfinite(b):
                raise ValueError(
                    f"body 分量 {b!r} — 必须为有限数值或 callable "
                    "(拒绝 NaN/Inf/布尔)")

    def _validate_solver_and_post(self):
        if self.linear_solver not in ("auto", "direct", "cg", "cg-block", "ilu"):
            raise ValueError(
                f"linear_solver='{self.linear_solver}' — 仅支持 "
                "auto/direct/cg/cg-block/ilu")
        if self.error_method not in ("auto", "spr", "l2", "weighted"):
            raise ValueError(
                f"error_method='{self.error_method}' — 仅支持 auto/spr/l2/weighted")
        if self.band_tag not in ("vm", "sx", "sy", "txy", "s1", "s2", "taumax"):
            raise ValueError(f"band_tag='{self.band_tag}' — 非法应力分量")

    def _validate_band_args(self):
        # Isoband 固定带宽: 三参数全有或全无 (曾只在绘图路径校验,
        # --no-plot 时静默忽略无效参数)
        band_args = (self.band_min, self.band_max, self.band_step)
        if any(x is not None for x in band_args) and not all(
                x is not None for x in band_args):
            raise ValueError(
                "--band-min, --band-max and --band-step must be specified "
                "together")
        if not all(x is not None for x in band_args):
            return
        if not all(np.isfinite(x) for x in band_args):
            raise ValueError("--band-min/--band-max/--band-step 必须有限")
        if self.band_step <= 0.0:
            raise ValueError("--band-step 必须为正")
        if self.band_max <= self.band_min:
            raise ValueError("--band-max 必须大于 --band-min")
        if self.band_step > self.band_max - self.band_min:
            raise ValueError(
                f"--band-step ({self.band_step:.3g}) 必须 <= "
                f"--band-max - --band-min ({self.band_max - self.band_min:.3g})")
        # 预计层数上限: np.arange 曾无界分配 (step=1e-12 时万亿层
        # 耗尽内存 / 1e-320 时除出 inf 再 int() 抛 OverflowError;
        # )。正常报告几十到几百层足够。
        ratio = (self.band_max - self.band_min) / self.band_step
        if not np.isfinite(ratio) or ratio > 10000:
            raise ValueError(
                f"--band-step ({self.band_step:.3g}) 产生超过 10000 个"
                "应力带, 超过上限 — 请增大 step")
        # 区间必须能被 step 整除: runner 生成等距 levels, 非整除组合
        # 会在求解成功后的绘图阶段才抛 ValueError, 白跑
        if abs(ratio - round(ratio)) > 1e-9 * max(ratio, 1.0):
            raise ValueError(
                f"(--band-max - --band-min) / --band-step = {ratio:.6g} "
                f"非整数 — 区间 {self.band_max - self.band_min:.6g} 必须能被 "
                f"步长 {self.band_step:.6g} 整除; 请调整 band_max 或 band_step")

    def to_dict(self) -> dict:
        """导出为 dict — 配置即数据: 可序列化/复现/对比."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if not f.name.startswith("_")}

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisConfig":
        """从 dict 构建 — 与 to_dict 互逆.

        未知键打印 WARN — 曾静默忽略, 配置拼写错误悄悄用默认值
       。保留前向兼容 (不拒绝)。
        """
        valid = {f.name for f in fields(cls) if not f.name.startswith("_")}
        unknown = sorted(set(data) - valid)
        if unknown:
            print(f"  [WARN] AnalysisConfig.from_dict 忽略未知键: "
                  f"{', '.join(unknown)} — 可用键: {sorted(valid)}")
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_args(cls, args) -> "AnalysisConfig":
        """从 argparse Namespace 构建配置 — 未指定 (None) 字段保持默认.

        CLI 显式参数 (argparse 中 default=None 的可覆盖参数) 覆盖
        程序默认; 其余字段直接复制.
        """
        config = cls()
        explicit = []
        for f in fields(cls):
            if f.name.startswith("_") or not hasattr(args, f.name):
                continue
            value = getattr(args, f.name)
            if value is not None:
                setattr(config, f.name, value)
                explicit.append(f.name)
        config._explicit = frozenset(explicit)
        config.validate()  # setattr 发生在构造之后 — 必须重新校验
        return config
