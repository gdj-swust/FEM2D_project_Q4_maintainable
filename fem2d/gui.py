# -*- coding: utf-8 -*-
"""FEM2D 图形界面 (tkinter) — 选模型文件 + 拓扑识别 + 边界/载荷.

交互顺序:
  1. 【浏览…】选 .geo/.txt/.msh/.spec 模型文件 (或手输路径) →
     「① 识别边界」: 拓扑识别出全部边界段 (编号/类型/点数),
     边名下拉 = 识别出的段, 可下拉选或手输 (编号 / 组名如 内孔 /
     @段名) — 与 CLI 边界选择器完全同源 (fem2d.boundary.
     _resolve_edge_indices);
  2. 边界条件分两栏逐条添加:
     - 边界位移 (约束): 固定 / 仅Ux / 仅Uy / **位移值 Ux,Uy 可输入**
     - 载荷: 压力 p / 面力 tx,ty
     同一几何边约束与载荷重叠会被拒绝 (历史教训: 重叠 Physical
     Curve → 约束吞载荷);
  3. 体力 → 「② 求解」→ 窗口内嵌云图 (PNG)。

设计要点:
- 数值逻辑零改动 — GUI 只做前端; 求解线程复用 runner 各阶段函数
  (runner._resolve_input/_build_model/_analyze_and_report/_plot),
  边界用 mesh 公共 API 施加 (fix_node/add_pressure/add_traction,
  与 bc_apply 同款调用, 支持任意位移值), 不修改任何求解器代码
- 本文件是独立新增文件 — run.py / wizard.py / 求解器均未改动,
  CLI 交互流程保持原样
- 求解/识别在后台线程执行 (gmsh API 的 signal 注册在非主线程需
  no-op 化), 输出经 queue 轮询刷新, 窗口不冻结
- matplotlib 固定 Agg: 云图经 --save 的 PNG 内嵌显示, 不弹外部
  窗口 (exe 下 plt.show() 曾闪退)
"""
import io
import math
import os
import re
import sys
import queue
import threading
import tempfile
import time
import tkinter as tk
import numpy as np
from tkinter import ttk, filedialog, messagebox, scrolledtext

_DEFAULTS = {"E": "2.1e11", "nu": "0.3", "thickness": "0.01"}
_NUM = re.compile(r"[-+]?\d*\.?\d+(e[-+]?\d+)?")
_SEG_TAG = {"line": "━", "arc": "⌒", "curve": "~", "ellipse": "O"}
# 云图弹窗放大内存保护: 单帧像素上限 (16MP ≈ 64MB RGBA)
_MAX_ZOOM_PIX = 16_000_000


def parse_bc_value(kind, raw):
    """位移/载荷值解析 → 规范化原文 (显示原样, 全角逗号归一); 抛 ValueError.

    位移: 1 值 = 只给 Ux, 2 值 = Ux,Uy — 每分量数值 或 x/y 表达式.
    载荷: 1 值 = 压力 p, 2 值 = 面力 tx,ty — 每分量数值 或 x/y 表达式.
    """
    raw = (raw or "").replace("，", ",").strip()
    if not raw:
        raise ValueError("值不能为空")
    parts = [p.strip() for p in raw.split(",")]
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"{'位移' if kind == '位移' else '载荷'}需要 1-2 "
                         f"个分量 (逗号分隔), 得到 {len(parts)} 个")
    hint = "位移" if kind == "位移" else "载荷"
    comps = ("Ux", "Uy") if kind == "位移" else ("p", "ty")
    for p, c in zip(parts, comps):
        _parse_scalar_component(p, c, hint)
    return raw


def _parse_scalar_component(raw, name, hint="体力"):
    """单个标量分量 → float 或 lambda x,y (与 CLI 同语义, AST 白名单).

    纯数字 → float (isfinite 拒绝 1e999 溢出); 含 x/y 变量 → 编译为
    lambda x,y; 其他 → ValueError (中文, 含可用函数提示).
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError(f"{hint} {name} 不能为空")
    if raw.lower() in ("nan", "+nan", "-nan", "inf", "+inf", "-inf",
                       "infinity", "+infinity", "-infinity"):
        raise ValueError(f"{hint} {name}: '{raw}' 不是有限数值 "
                         f"(NaN/Inf 被拒绝)")
    try:
        value = float(raw)
    except ValueError:
        pass
    else:
        if not math.isfinite(value):
            raise ValueError(f"{hint} {name}: '{raw}' 溢出为 Inf, "
                             f"不是有限数值")
        return value
    from fem2d.loads_core import _compile_expr, _expr_has_spatial_names
    if not _expr_has_spatial_names(raw):
        raise ValueError(
            f"{hint} {name}: '{raw}' 既不是数值也不含 x/y 变量 — "
            f"支持 x/y 表达式, 函数: sin cos tan exp sqrt log abs pi "
            f"(例: 1000*exp(x/2), 1e6, -0.001*x*y)")
    try:
        return _compile_expr(raw)
    except ValueError as error:
        raise ValueError(f"{hint} {name}: {error}") from error


def parse_body(bx_raw, by_raw):
    """体力输入 → (bx, by) 或 None (两者都空时跳过).

    分量支持: 纯数字 (如 -78000), 或含 x/y 的函数表达式 (如
    1000*exp(x/2) — x,y 为全局坐标; 函数: sin/cos/tan/exp/sqrt/log/
    abs/pi; AST 白名单, 与 CLI 体力同源 loads_core._compile_expr).
    只填一个分量时另一个补 0 — 只填 by 表达重力 (0, -78000) 符合直觉。
    """
    bx_raw = (bx_raw or "").strip()
    by_raw = (by_raw or "").strip()
    if not bx_raw and not by_raw:
        return None
    return [_parse_body_comp(bx_raw, "bx"),
            _parse_body_comp(by_raw, "by")]


def _parse_body_comp(raw, name):
    """单分量体力: 纯数字 → float; 含 x/y → lambda (AST 白名单). 空 → 0."""
    if not raw or not raw.strip():
        return 0.0
    return _parse_scalar_component(raw, name, "体力")


def _fmt_body(v):
    """体力分量打印: callable 显示为函数, 常数保留数值."""
    if callable(v):
        return "<x/y 函数>"
    return f"{v:g}"


class FEM2DGUI:
    """主窗口: 模型文件 + 边界/载荷 + 右上几何预览 + 右下云图/日志."""

    def __init__(self, root):
        self.root = root
        root.title("FEM2D 交互建模")
        root.geometry("1200x880")
        root.minsize(1050, 780)

        self._result_q = queue.Queue()
        # 后台阶段进度消息 (识别/求解线程 → 主线程 status)
        self._progress_q = queue.Queue()
        self._identify_started = 0.0
        self._solve_started = 0.0
        self._stage_shown_at = 0.0
        self._solving = False
        self._identifying = False
        self._last_png = None
        # 识别结果 (拓扑边界) — 与当前模型文件一一对应
        self._segs = None
        self._mesh = None
        self._region_registry = None
        self._geo_path = None
        self._geo_label = ""
        # 边界条件: [(边选择器, 类型key, 值字符串, 已解析段索引tuple)]
        self._fix_items = []
        self._load_items = []
        # 几何预览视图状态 (滚轮缩放/左键拖动) — None=未识别
        self._pv = None
        # 云图弹窗单例 (防连点开多个)
        self._img_win = None

        # 全局兜底: tk 事件/after 回调的裸异常默认只进 stderr — exe
        # 隐藏控制台后完全不可见, 功能静默失效。覆写路由到日志面板,
        # 并复位后台任务标志与按钮 (防永久卡死)
        root.report_callback_exception = self._on_tk_error
        self._build_widgets()

    # ── 界面构建 ──
    def _build_widgets(self):
        left = ttk.Frame(self.root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.root, padding=(0, 10, 10, 10))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ── 模型文件 (第一步) ──
        file_box = ttk.LabelFrame(left, text="① 模型文件", padding=6)
        file_box.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.identify_btn = ttk.Button(file_box, text="① 识别边界",
                                       command=self._on_identify)
        self.identify_btn.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.path_var = tk.StringVar()
        # 路径变化自动识别 — 粘贴/手输路径也走"选完自动识别", 与
        # 浏览按钮同行为 (用户: 选了路径就该自动出结果, 不该再点按钮)
        self.path_var.trace_add("write", self._on_path_trace)
        self._path_after = None
        ttk.Entry(file_box, textvariable=self.path_var, width=26).grid(
            row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(file_box, text="浏览…", width=7,
                   command=self._on_open_file).grid(row=0, column=2, sticky="e")
        self.file_hint = ttk.Label(file_box, text="选 .geo/.txt/.msh/.spec — "
                                   "选完自动识别边界",
                                   foreground="#666", wraplength=330)
        self.file_hint.grid(row=1, column=0, columnspan=3, sticky="w",
                            pady=(2, 0))

        # ── 材料 ──
        mat = ttk.LabelFrame(left, text="材料", padding=6)
        mat.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.entry = {}
        fields = [("E", "弹性模量 E [Pa]"), ("nu", "泊松比 nu"),
                  ("thickness", "厚度 [m]")]
        for i, (key, label) in enumerate(fields):
            r, c = divmod(i, 2)
            ttk.Label(mat, text=label + ":").grid(
                row=r * 2, column=c * 2, sticky="w", padx=(0, 4))
            var = tk.StringVar(value=_DEFAULTS[key])
            ttk.Entry(mat, textvariable=var, width=14).grid(
                row=r * 2 + 1, column=c * 2, sticky="w", padx=(0, 10))
            self.entry[key] = var

        # ── 边界条件区 ──
        self.bc_frame = ttk.LabelFrame(
            left, text="边界条件 (边名下拉 = 识别出的边)", padding=6)
        self.bc_frame.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        self.edge_combo = ttk.Combobox(self.bc_frame, width=27)
        self.edge_combo.grid(row=0, column=0, sticky="w")
        self.edge_combo.bind("<<ComboboxSelected>>", self._on_edge_selected)
        self.edge_hint = ttk.Label(self.bc_frame, text="点击「① 识别边界」"
                                   "拓扑识别出全部边", foreground="#666",
                                   wraplength=340)
        self.edge_hint.grid(row=1, column=0, sticky="w", pady=(2, 0))

        # ── 边界位移 (约束): 选边 → 输入位移值 → + 追加 ──
        fix_box = ttk.LabelFrame(self.bc_frame, text="边界位移 (约束)", padding=4)
        fix_box.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(fix_box, text="位移 Ux,Uy [m] (支持 x/y 表达式):").grid(
            row=0, column=0, sticky="w")
        self.fix_value = ttk.Entry(fix_box, width=12)
        self.fix_value.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Button(fix_box, text="＋ 添加", width=7,
                   command=lambda: self._add_bc("fix")).grid(
            row=0, column=2, sticky="e", padx=(6, 0))
        self.fix_list = tk.Listbox(fix_box, height=3, width=46,
                                   font=("Consolas", 9))
        self.fix_list.grid(row=1, column=0, columnspan=3, sticky="ew",
                           pady=(4, 0))
        ttk.Button(fix_box, text="删除选中", width=8,
                   command=lambda: self._del_bc("fix")).grid(
            row=2, column=2, sticky="e", pady=(2, 0))

        # ── 载荷: 选边 → 输入值 → + 追加 ──
        load_box = ttk.LabelFrame(self.bc_frame, text="载荷 (单值=压力, "
                                  "两值=面力 tx,ty — 支持 x/y 表达式)",
                                  padding=4)
        load_box.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        ttk.Label(load_box, text="值 [Pa]:").grid(row=0, column=0, sticky="w")
        self.load_value = ttk.Entry(load_box, width=12)
        self.load_value.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Button(load_box, text="＋ 添加", width=7,
                   command=lambda: self._add_bc("load")).grid(
            row=0, column=2, sticky="e", padx=(6, 0))
        self.load_list = tk.Listbox(load_box, height=3, width=46,
                                    font=("Consolas", 9))
        self.load_list.grid(row=1, column=0, columnspan=3, sticky="ew",
                            pady=(4, 0))
        ttk.Button(load_box, text="删除选中", width=8,
                   command=lambda: self._del_bc("load")).grid(
            row=2, column=2, sticky="e", pady=(2, 0))

        # ── 体力 ──
        body = ttk.LabelFrame(left, text="体力 [N/m³] (支持 x/y 表达式: "
                               "sin cos tan exp sqrt log abs pi, 可留空)",
                               padding=6)
        body.grid(row=3, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(body, text="bx:").grid(row=0, column=0, sticky="w")
        self.body_bx = tk.StringVar()
        ttk.Entry(body, textvariable=self.body_bx, width=9).grid(
            row=0, column=1, sticky="w", padx=(0, 10))
        ttk.Label(body, text="by:").grid(row=0, column=2, sticky="w")
        self.body_by = tk.StringVar()
        ttk.Entry(body, textvariable=self.body_by, width=9).grid(
            row=0, column=3, sticky="w")

        # ── 按钮 ──
        btns = ttk.Frame(left)
        btns.grid(row=4, column=0, sticky="ew", pady=(0, 4))
        self.run_btn = ttk.Button(btns, text="② 求解", command=self._on_build)
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.status = tk.Label(left, text="就绪 — 第一步: 选择模型文件并"
                               "①识别边界", anchor="w", fg="#444")
        self.status.grid(row=5, column=0, sticky="ew")

        # ── 右上: 几何预览 (识别出的真实边界 + 坐标轴 + 编号) ──
        self.preview_frame = ttk.LabelFrame(right, text="几何预览"
                                            " (识别出的边界段)", padding=4)
        self.preview_frame.pack(fill=tk.X)
        self.preview = tk.Canvas(self.preview_frame, width=660, height=360,
                                 bg="white", highlightthickness=1,
                                 highlightbackground="#bbb")
        # 交互缩放: 滚轮缩放 (锚点=鼠标) / 左键拖动平移
        self.preview.bind("<MouseWheel>", self._on_preview_wheel)
        self.preview.bind("<ButtonPress-1>", self._on_preview_press)
        self.preview.bind("<B1-Motion>", self._on_preview_drag)
        self.preview.pack()

        # ── 右下: 云图 ──
        self.img_frame = ttk.LabelFrame(right, text="云图 (位移 / 应力)",
                                        padding=4)
        self.img_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.img_label = tk.Label(self.img_frame, text="求解后此处显示云图"
                                  " (点击放大: 滚轮缩放/拖动)",
                                  bg="#f0f0f0", fg="#888",
                                  width=100, height=20)
        self.img_label.bind("<Button-1>", self._on_img_click)
        self.img_label.pack(fill=tk.BOTH, expand=True)
        bar = ttk.Frame(right)
        bar.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(bar, text="保存云图为 PNG",
                   command=self._on_save_png).pack(side=tk.LEFT)
        self.state_lbl = ttk.Label(bar, text="")
        self.state_lbl.pack(side=tk.RIGHT)

        # ── 日志 ──
        ttk.Label(right, text="求解日志:").pack(anchor="w", pady=(6, 0))
        self.log = scrolledtext.ScrolledText(right, height=8, state="disabled",
                                             font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH)

    def _on_tk_error(self, exc_type, exc, tb):
        """report_callback_exception 兜底: 异常写日志面板 + 复位状态."""
        import traceback
        try:
            self._append_log("[GUI] 回调异常:\n" + "".join(
                traceback.format_exception(exc_type, exc, tb)) + "\n")
            # 后台任务中断后复位标志与按钮 — 否则识别/求解"永久进行中"
            if self._identifying or self._solving:
                self._identifying = False
                self._solving = False
                self.identify_btn.config(state="normal")
                self.run_btn.config(state="normal")
        except Exception:
            pass

    # ── ① 模型文件 + 拓扑边界识别 ──
    def _on_open_file(self):
        """浏览选择模型文件 → 路径入框 → 自动识别边界."""
        if self._solving:
            messagebox.showinfo("求解中", "正在求解 — 请稍候")
            return
        if self._identifying:
            messagebox.showinfo("识别中", "正在识别边界 — 稍候")
            return
        path = filedialog.askopenfilename(
            title="选择模型文件 (.geo/.txt/.msh/.spec)",
            filetypes=[("模型文件", "*.geo *.txt *.msh *.spec"),
                       ("所有文件", "*.*")])
        if not path:
            return
        # set 触发 _on_path_trace → 500ms 防抖后自动识别 (统一入口)
        self.path_var.set(path)
        self._append_log(f"已选择文件: {path}\n")

    def _on_path_trace(self, *_args):
        """路径框内容变化 → 500ms 防抖后自动识别 (浏览/粘贴/手输统一)."""
        if self._identifying or self._solving:
            return
        path = self.path_var.get().strip()
        if not path or not os.path.isfile(path):
            return
        # 防抖: 取消上次调度 — 连续编辑只识别最终路径
        if self._path_after is not None:
            try:
                self.root.after_cancel(self._path_after)
            except Exception:
                pass
        self._path_after = self.root.after(
            500, lambda: self._start_identify(path))

    def _on_identify(self):
        """对当前路径框中的模型文件识别边界."""
        if self._solving:
            messagebox.showinfo("求解中", "正在求解 — 请稍候")
            return
        if self._identifying:
            messagebox.showinfo("识别中", "正在识别边界 — 识别完成后"
                                "边名下拉会列出全部边")
            return
        path = self.path_var.get().strip()
        if not path:
            messagebox.showwarning("未选择文件", "先选择模型文件: 点"
                                   "【浏览…】或在路径框输入")
            return
        if not os.path.isfile(path):
            messagebox.showerror("文件不存在", f"文件不存在:\n{path}")
            return
        self._append_log(f"已选择文件: {path}\n")
        self._start_identify(path)

    def _start_identify(self, geo_path):
        """后台识别边界 → 填充边名下拉 + 真实预览."""
        from fem2d import runner  # noqa: F401 — 预热, 识别线程内导入也可

        # 材料非法提前拦截 (线程内 float() 失败只显示泛化"识别失败")
        err = self._validate_material()
        if err:
            messagebox.showerror("输入有误", err)
            return

        self._geo_path = geo_path
        self._geo_label = os.path.basename(geo_path)
        self._identify_aborted = False
        self._identify_started = time.monotonic()
        self._identifying = True
        self.identify_btn.config(state="disabled")
        self.run_btn.config(state="disabled")
        self.edge_combo.config(values=["识别中…"])
        self.edge_combo.set("识别中…")
        self.status.config(text="识别边界中…", fg="#b00")
        self._append_log(f"\n▶ 识别边界: {geo_path}\n")
        argv = self._identify_args(geo_path)
        self.root.after(100, self._poll_result)
        threading.Thread(target=self._identify_thread,
                         args=(argv,), daemon=True).start()

    def _identify_args(self, geo_path):
        """识别无需材料, 但 AnalysisConfig 校验需要合法数值 — 用表单值."""
        return [geo_path,
                "--E", self.entry["E"].get().strip() or "2.1e11",
                "--nu", self.entry["nu"].get().strip() or "0.3",
                "--thickness", self.entry["thickness"].get().strip()
                or "0.01"]

    def _identify_thread(self, argv):
        """后台: runner 管线识别拓扑边界 (与 CLI 完全同一条链)."""
        from fem2d.config import AnalysisConfig
        from fem2d import runner

        buf = io.StringIO()
        old = (sys.stdout, sys.stderr)
        sys.stdout = sys.stderr = buf
        # gmsh Python API 的 initialize() 在调用线程注册 SIGINT handler
        # (gmsh.py: signal.signal(SIGINT, SIG_DFL)) — 后台线程 (Py3.13)
        # 抛 ValueError. 线程内 no-op 化 signal.signal, 结束恢复。
        import signal as _signal
        _orig_signal = _signal.signal

        def _thread_signal(num, handler):
            try:
                return _orig_signal(num, handler)
            except ValueError:
                return _signal.SIG_DFL

        _signal.signal = _thread_signal
        try:
            self._progress_q.put("识别: 解析模型/网格化中 — "
                                 "复杂模型需 10~60 秒…")
            config = AnalysisConfig(mesh=argv[0], E=float(argv[2]),
                                    nu=float(argv[4]), thickness=float(argv[6]))
            resolved = runner._resolve_input(config)
            if resolved is None:
                raise ValueError("无法解析输入文件")
            self._progress_q.put("识别: 网格就绪 — 正在导入网格…")
            model = runner._build_model(config, resolved)
            self._progress_q.put(f"识别: 正在提取边界 (共 {len(model.segs)}"
                                 " 条边)…")
            result = (model.segs, model.mesh, model.region_registry)
        except SystemExit as error:
            # 漏捕会线程静默死 + _result_q 永空 → 按钮永久禁用卡死
            buf.write(f"[GUI] 识别线程被 SystemExit 终止 (code={error.code!r})\n")
            result = None
        except Exception as error:
            buf.write(f"[GUI] 识别失败: {error!r}\n")
            result = None
        finally:
            _signal.signal = _orig_signal
            sys.stdout, sys.stderr = old
        self._result_q.put((result, buf.getvalue()))

    def _on_edge_selected(self, _event=None):
        """下拉选中 → "3 | ━ 底部 (11点)" → 选择器 "3";
        "≡ 内孔 (8条)" → 中文组名 "内孔"."""
        raw = self.edge_combo.get().strip()
        m = re.match(r"^\s*(\d+)\s*\|", raw)
        if m:
            self.edge_combo.set(m.group(1))
            return
        m = re.match(r"^≡\s*([^（(]+)", raw)
        if m:
            self.edge_combo.set(m.group(1).strip())

    def _apply_identification(self, result):
        """识别完成: 填充边名下拉 (识别出的全部段) + 真实预览."""
        # 换模型 (或重新识别): 旧 BC 存的段索引映射新模型会静默错边 —
        # 必须清空, 否则显示"完成"但约束加在错误边上
        if self._fix_items or self._load_items:
            self._fix_items = []
            self._load_items = []
            self.fix_list.delete(0, tk.END)
            self.load_list.delete(0, tk.END)
            messagebox.showinfo("边界条件已清空",
                                "模型已更换/重新识别 — 原边界条件已清空, "
                                "请重新配置")
        self._identifying = False
        self.identify_btn.config(state="normal")
        self.run_btn.config(state="normal")
        segs, mesh, region_registry = result
        self._segs = segs
        self._mesh = mesh
        self._region_registry = region_registry

        options = []
        # 组条目优先: Physical Curve 组一键全选整组 (如 ≡ 椭圆孔 (20条))
        # — 80+ 条边时下拉不挤爆, 用户按名字选组而不是逐个数编号
        from fem2d.boundary.naming import segment_physical_names
        group_counts = {}
        for seg in segs:
            for name in segment_physical_names(seg):
                key = name.casefold()
                if key not in group_counts:
                    group_counts[key] = [name, 0]
                group_counts[key][1] += 1
        for orig, count in sorted(
                (v for v in group_counts.values()), key=lambda v: v[0]):
            options.append(f"≡ {orig} ({count}条)")
        # 拓扑孔分组 (无 Physical Curve 的外部网格也有一键全选孔)
        from fem2d.boundary.segment_utils import segment_is_outer
        hole_nums = [i + 1 for i, s in enumerate(segs)
                     if not segment_is_outer(s)]
        if hole_nums:
            options.append(f"≡ 内孔 ({len(hole_nums)}条)")
        # 段条目: 缩短 label (复合 label 首段 = 物理名/描述, 80+ 段可读)
        for i, seg in enumerate(segs):
            tag = _SEG_TAG.get(seg.get("type", ""), "?")
            label = str(seg.get("label", "")).split("|")[0].strip()
            options.append(f"{i + 1} | {tag} {label} ({len(seg['nodes'])}点)")
        self.edge_combo.config(values=options)
        self.edge_combo.set("")

        # 组汇总 (与 CLI 边界表一致): 外边/内孔/直边/圆弧/曲线
        from fem2d.boundary.segment_utils import segment_is_outer
        groups = []
        for name, cond in (
                ("外边", lambda s: segment_is_outer(s)),
                ("内孔", lambda s: not segment_is_outer(s)),
                ("直边", lambda s: s.get("type") == "line"),
                ("圆弧", lambda s: s.get("type") == "arc"),
                ("曲线", lambda s: s.get("type") == "curve")):
            nums = [i + 1 for i, s in enumerate(segs) if cond(s)]
            if nums:
                groups.append(f"{name}{len(nums)}条={_fmt_nums(nums)}")
        hint = (f"识别出 {len(segs)} 条边 ({', '.join(groups)}) — "
                "下拉选编号, 或选「≡组」整组全选, 或手输: "
                "编号(如 3) / 组名(如 椭圆孔) / @组名")
        self.edge_hint.config(text=hint, foreground="#666")

        self.status.config(
            text=f"已识别 {len(segs)} 条边 ({self._geo_label}) — "
                 "在下方添加约束/载荷并求解", fg="#080")
        # 识别完成弹窗 — 用户期望"选了路径就弹出识别结果"的直观反馈
        try:
            messagebox.showinfo(
                "识别完成",
                f"已识别 {len(segs)} 条边界 ({self._geo_label})\n\n"
                + ("\n".join(f"  {g}" for g in groups) + "\n\n" if groups
                   else "")
                + "在下方添加约束/载荷后点「② 求解」\n"
                  "(下拉选编号 / 选 ≡组 整组全选 / 手输 @组名)")
        except Exception:
            pass
        # 新模型视图基准作废 — 沿用旧 cx0/cy0/scale 会让新几何画出屏
        self._pv = None
        self._draw_preview()

    def _identify_failed(self, text):
        self._identifying = False
        self.identify_btn.config(state="normal")
        self.run_btn.config(state="normal")
        self.edge_combo.config(values=[])
        self.edge_combo.set("")
        # 状态全重置: 旧 _segs/_mesh 残留会让人对识别失败的新文件
        # 用旧 BC 求解 (静默错边) — 一并清空 BC 与预览
        self._segs = None
        self._mesh = None
        self._region_registry = None
        self._pv = None
        self._fix_items = []
        self._load_items = []
        self.fix_list.delete(0, tk.END)
        self.load_list.delete(0, tk.END)
        self.preview.delete("all")
        self.preview.create_text(330, 180, text="识别失败 — 见日志",
                                 fill="#b00")
        self.status.config(text="识别失败 — 见日志", fg="#b00")

    # ── 边界条件逐条添加/删除 (选择器与 CLI 同源) ──
    def _add_bc(self, fix_or_load):
        if self._identifying:
            messagebox.showinfo("识别中", "正在识别边界 — 完成后才能添加"
                                "约束/载荷 (识别期间旧模型边名无效)")
            return
        if not self._segs:
            messagebox.showwarning("未识别", "先选择模型文件并点击"
                                   "「① 识别边界」 — 识别出的边会列在"
                                   "下拉里供选择")
            return
        raw = self.edge_combo.get().strip()
        # 下拉格式 "3 | ━ 底部 (11点)" → 选择器 "3" (编号)
        m = re.match(r"^\s*(\d+)\s*\|", raw)
        if m:
            raw = m.group(1)
        # "≡ 内孔 (8条)" → 中文组名 "内孔"
        m = re.match(r"^≡\s*([^（(]+)", raw)
        if m:
            raw = m.group(1).strip()
        if not raw:
            messagebox.showwarning("边名", "先选择/输入边名 "
                                   "(编号 或 组名 或 @段名)")
            return

        items, listbox = (self._fix_items, self.fix_list) \
            if fix_or_load == "fix" else (self._load_items, self.load_list)
        value_entry = self.fix_value if fix_or_load == "fix" \
            else self.load_value
        value = value_entry.get().strip()
        try:
            # 仅做校验: 值可含 x/y 表达式, 实际求值在 _apply_model_bcs
            parse_bc_value(
                "位移" if fix_or_load == "fix" else "载荷", value)
        except ValueError as error:
            messagebox.showerror("数值有误", str(error))
            return
        # 载荷: 单值 = 压力 (法向), 两值 = 面力分量 (按分量数, 值可为表达式)
        key = "位移" if fix_or_load == "fix" else (
            "压力" if "," not in value else "拉力")

        # 解析选择器 — 与 CLI _resolve_edge_indices 同一条解析链
        # (region_registry 传给 @组名 路径: 椭圆孔等 Physical Curve 组)
        from fem2d.boundary import _resolve_edge_indices
        try:
            indices = tuple(_resolve_edge_indices(
                raw, self._segs, region_registry=self._region_registry))
        except ValueError as error:
            messagebox.showerror("边名有误",
                                 f"{error}\n\n可用边: 见下拉或输入编号")
            return
        if not indices:
            messagebox.showerror("边名有误",
                                 f"无法识别边名 '{raw}' — "
                                 "用下拉里的编号, 或组名 (如 内孔), 或 @段名")
            return
        # 同一边重复配置 → 重叠约束/载荷危险 (历史教训), 拒绝
        new_set = set(indices)
        for item in self._fix_items + self._load_items:
            if new_set & set(item[3]):
                conflict = [i + 1 for i in new_set & set(item[3])]
                messagebox.showwarning(
                    "边重叠", f"边 '{raw}' (编号 {_fmt_nums(conflict)}) 与已"
                    "配置的项重叠 — 同一几何边只能配置一次 "
                    "(重叠约束/载荷危险)")
                return

        label = f"{raw:<10} {key}  {value}"
        listbox.insert(tk.END, label)
        items.append((raw, key, value, indices))
        self.edge_combo.set("")
        value_entry.delete(0, tk.END)
        self._draw_preview()

    def _del_bc(self, fix_or_load):
        items, listbox = (self._fix_items, self.fix_list) \
            if fix_or_load == "fix" else (self._load_items, self.load_list)
        sel = listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        listbox.delete(idx)
        del items[idx]
        self._draw_preview()

    # ── 几何预览: 识别出的真实边界段 + 坐标轴 + 编号 (颜色反馈) ──
    def _draw_preview(self):
        c = self.preview
        c.delete("all")
        W, H = 660, 360
        if not self._segs:
            c.create_text(W // 2, H // 2,
                          text="点击「① 识别边界」后显示真实轮廓",
                          fill="#888")
            return

        coords_all = np.concatenate([self._seg_coords(seg)
                                     for seg in self._segs])
        if coords_all.size == 0 or not np.isfinite(coords_all).all():
            # 空/NaN/Inf 坐标 → np.min/max 裸抛 ValueError, 主线程回调
            # 异常在隐藏控制台后不可见 — 显式防护 + 占位文案
            self._append_log("[GUI] 预览: 段坐标为空或含 NaN/Inf, "
                             "跳过绘制\n")
            c.create_text(W // 2, H // 2, text="模型坐标异常 — 无法预览",
                          fill="#b00")
            return
        xs, xs_max = float(coords_all[:, 0].min()), float(coords_all[:, 0].max())
        ys_min, ys_max = float(coords_all[:, 1].min()), float(coords_all[:, 1].max())
        cx0, cy0 = (xs + xs_max) / 2, (ys_min + ys_max) / 2
        span = max(xs_max - xs, ys_max - ys_min) * 0.6 + 1e-9

        # 视图状态: 首次识别按适配存基准, 之后保留用户缩放/拖动
        if self._pv is None:
            s0 = min((W - 70) / (2 * span), (H - 70) / (2 * span))
            self._pv = {"scale0": s0, "scale": s0, "pan_x": 0.0,
                        "pan_y": 0.0, "cx0": cx0, "cy0": cy0,
                        "W": W, "H": H}
        pv = self._pv
        scale, pan_x, pan_y = pv["scale"], pv["pan_x"], pv["pan_y"]

        def px(x):
            return (x - cx0) * scale + W / 2 + pan_x

        def py(y):
            return -(y - cy0) * scale + H / 2 + pan_y

        # 坐标轴 + 刻度 (世界坐标 tick — 放大后自动变稀, 不重叠)
        ax_len = span * 6
        c.create_line(px(-ax_len), py(0), px(ax_len), py(0), fill="#999")
        c.create_line(px(0), py(-ax_len), px(0), py(ax_len), fill="#999")
        raw = span / 4
        k = int(math.floor(math.log10(raw)))
        base = 10.0 ** k
        step = base
        for m in (1, 2, 5, 10):
            if base * m >= raw:
                step = base * m
                break
        t = -span
        while t <= span + step * 0.5:
            if abs(t) > 1e-9:
                c.create_line(px(t), py(0) - 3, px(t), py(0) + 3, fill="#999")
                c.create_text(px(t), py(0) + 12, text=_fmt(t),
                              font=("TkDefaultFont", 7), fill="#666")
                c.create_line(px(0) - 3, py(t), px(0) + 3, py(t), fill="#999")
                c.create_text(px(0) - 10, py(t), text=_fmt(t),
                              font=("TkDefaultFont", 7), fill="#666")
            t += step
        c.create_text(W - 14, H - 20, text="x", fill="#666")
        c.create_text(12, 14, text="y", fill="#666")
        # 交互提示 (固定角落, 不随缩放)
        c.create_text(W - 6, 6, anchor="ne", text="滚轮缩放 · 左键拖动",
                      font=("TkDefaultFont", 8), fill="#999")

        # 配置状态 → 边颜色 (蓝=约束, 红=载荷)
        fix_idx = {i for item in self._fix_items for i in item[3]}
        load_idx = {i for item in self._load_items for i in item[3]}

        # 画每条段 + 编号
        for i, seg in enumerate(self._segs):
            coords = self._seg_coords(seg)
            pts = [(px(float(x)), py(float(y))) for x, y in coords]
            color = "#d33" if i in load_idx else (
                "#24c" if i in fix_idx else "#333")
            c.create_line(pts, fill=color, width=3)
            mx, my = coords[:, 0].mean(), coords[:, 1].mean()
            c.create_oval(px(mx) - 11, py(my) - 11, px(mx) + 11, py(my) + 11,
                          fill="white", outline=color, width=1)
            c.create_text(px(mx), py(my), text=str(i + 1), fill=color,
                          font=("TkDefaultFont", 10))

    # ── 预览交互: 滚轮缩放 (锚点=鼠标位置) ──
    def _on_preview_wheel(self, event):
        if not self._segs or not self._pv:
            return
        pv = self._pv
        old = pv["scale"]
        new = max(pv["scale0"] * 0.4,
                  min(pv["scale0"] * 60.0, old * 1.15 ** (event.delta / 120.0)))
        if new == old:
            return
        W, H = pv["W"], pv["H"]
        # 鼠标指向的世界坐标 — 缩放前后保持在该鼠标位置
        wx = (event.x - W / 2 - pv["pan_x"]) / old + pv["cx0"]
        wy = pv["cy0"] - (event.y - H / 2 - pv["pan_y"]) / old
        pv["scale"] = new
        pv["pan_x"] = event.x - W / 2 - (wx - pv["cx0"]) * new
        pv["pan_y"] = event.y - H / 2 + (wy - pv["cy0"]) * new
        self._draw_preview()

    # ── 预览交互: 左键拖动平移 ──
    def _on_preview_press(self, event):
        if self._pv:
            self._pv["drag"] = (event.x, event.y)

    def _on_preview_drag(self, event):
        if not self._pv or "drag" not in self._pv:
            return
        x0, y0 = self._pv["drag"]
        self._pv["pan_x"] += event.x - x0
        self._pv["pan_y"] += event.y - y0
        self._pv["drag"] = (event.x, event.y)
        self._draw_preview()

    def _seg_coords(self, seg):
        """段节点坐标 (mesh.nodes[seg['nodes']] 兜底 — coords 缺失时)."""
        if "coords" in seg and seg["coords"] is not None:
            return np.asarray(seg["coords"])
        return np.asarray(self._mesh.nodes)[seg["nodes"]]

    # ── 输入收集 + 求解 ──
    def _validate_material(self):
        """E/nu/thickness 数值校验 (识别/求解共用). 返回 None 或错误消息."""
        for key in ("E", "nu", "thickness"):
            raw = self.entry[key].get().strip()
            if not _NUM.fullmatch(raw):
                return f"'{key}' 需要数值 (得到 '{raw}')"
        if not (-1 < float(self.entry["nu"].get()) < 0.5):
            return "nu 需要在 (-1, 0.5) 区间"
        return None

    def _collect_inputs(self):
        """收集材料/边界/体力 → extras dict. 校验失败抛 ValueError."""
        err = self._validate_material()
        if err:
            raise ValueError(err)

        fix_rows = [(raw, key, parse_bc_value("位移", value))
                    for raw, key, value, _i in self._fix_items]
        load_rows = [(raw, key, parse_bc_value("载荷", value))
                     for raw, key, value, _i in self._load_items]
        body = parse_body(self.body_bx.get(), self.body_by.get())
        return {"fix_rows": fix_rows, "load_rows": load_rows, "body": body}

    def _on_build(self):
        if self._solving:
            return
        if self._identifying:
            messagebox.showinfo("识别中", "正在识别边界 — 完成后即可求解")
            return
        if not self._segs:
            messagebox.showwarning("未识别", "第一步: 选择模型文件并点"
                                   "击「① 识别边界」 — 识别出的边会列在"
                                   "下拉里供选择, 然后再配置边界与求解")
            return
        # 防静默: 路径框被改过而识别结果还是旧文件的 → 必须重新识别
        cur = self.path_var.get().strip()
        if not cur or not os.path.isfile(cur):
            messagebox.showwarning("路径无效", "模型文件路径无效 — "
                                   "请重新选择文件")
            return
        if cur != self._geo_path:
            messagebox.showwarning("路径已修改",
                                   "模型文件路径已修改但未重新识别 — "
                                   "请先点击「① 识别边界」")
            return
        try:
            extras = self._collect_inputs()
        except ValueError as error:
            messagebox.showerror("输入有误", str(error))
            return
        self._launch(extras=extras)

    def _launch(self, extras=None):
        """后台线程求解: 自建管线 (识别+施加+求解+云图).

        几何 = 识别时的同一文件 (self._geo_path), 边界/体力由求解
        线程通过 mesh 公共 API 直接施加, 不写入文件.
        """
        geo_path = self._geo_path
        label = self._geo_label or "模型"
        png = os.path.join(tempfile.gettempdir(), "fem2d_gui_result.png")
        self._last_png = png
        try:
            if os.path.exists(png):
                os.unlink(png)
        except OSError:
            pass
        extras = extras or {}
        self._solve_aborted = False
        self._solve_started = time.monotonic()
        self._solving = True
        self.run_btn.config(state="disabled")
        self.identify_btn.config(state="disabled")
        self.status.config(text=f"求解中 ({label})…", fg="#b00")
        self._append_log(f"\n▶ 求解: {geo_path}\n")
        # 材料值在主线程读取 — tkinter StringVar 后台线程读取会抛
        # TclError (main thread is not in main loop)
        args = (geo_path, png,
                extras.get("fix_rows", []), extras.get("load_rows", []),
                extras.get("body", None),
                self.entry["E"].get().strip(),
                self.entry["nu"].get().strip(),
                self.entry["thickness"].get().strip())
        self.root.after(100, self._poll_result)
        threading.Thread(target=self._solve_thread, args=(args,),
                         daemon=True).start()

    # ── 求解线程: 复用 runner 阶段函数 + mesh 公共 API ──
    def _solve_thread(self, args):
        """后台求解 — 与 CLI 同一管线, 边界用 mesh 公共 API 施加
        (支持任意位移值), 不修改任何求解器代码."""
        geo_path, png, fix_rows, load_rows, body, \
            e_str, nu_str, t_str = args
        buf = io.StringIO()
        old = (sys.stdout, sys.stderr)
        sys.stdout = sys.stderr = buf
        # 同 _identify_thread: gmsh API 的 signal 注册非主线程 no-op 化
        import signal as _signal
        _orig_signal = _signal.signal

        def _thread_signal(num, handler):
            try:
                return _orig_signal(num, handler)
            except ValueError:
                return _signal.SIG_DFL

        _signal.signal = _thread_signal
        try:
            from fem2d.config import AnalysisConfig
            from fem2d import runner

            self._progress_q.put("求解: 解析模型/网格化中 — "
                                 "复杂模型需 10~60 秒…")
            config = AnalysisConfig(
                mesh=geo_path,
                E=float(e_str), nu=float(nu_str), thickness=float(t_str),
                no_plot=True, save=png)
            resolved = runner._resolve_input(config)
            if resolved is None:
                raise ValueError("无法解析输入文件")
            self._progress_q.put("求解: 网格就绪 — 正在组装刚度矩阵…")
            model = runner._build_model(config, resolved)
            self._apply_model_bcs(model, fix_rows, load_rows, body)
            self._progress_q.put("求解: 线性方程组求解中…")
            result, _z2, _q, scale = runner._analyze_and_report(
                config, model)
            self._progress_q.put("求解: 绘制云图中…")
            runner._plot(config, model.mesh, result, scale)
            rc = 0
        except SystemExit as error:
            # 非 int code (SystemExit("msg")/SystemExit()) 算失败, 不能 rc=0
            rc = error.code if isinstance(error.code, int) else 2
        except Exception as error:
            buf.write(f"[GUI] 求解异常: {error!r}\n")
            rc = 2
        finally:
            _signal.signal = _orig_signal
            sys.stdout, sys.stderr = old
        self._result_q.put((rc, buf.getvalue()))

    def _apply_model_bcs(self, model, fix_rows, load_rows, body):
        """在 runner 构建的模型上施加边界 — mesh 公共 API, 与 bc_apply
        交互模式同款调用 (fix_node/add_pressure/add_traction)."""
        from fem2d.boundary import _resolve_edge_indices
        mesh = model.mesh
        segs = model.segs
        registry = model.region_registry
        avail = ", ".join(str(i + 1) for i in range(len(segs)))

        def _resolve(raw):
            try:
                indices = tuple(_resolve_edge_indices(
                    raw, segs, region_registry=registry))
            except ValueError as error:
                raise ValueError(f"边 '{raw}': {error}") from error
            if not indices:
                raise ValueError(f"未找到边 '{raw}' — 可用: {avail}")
            return indices

        for raw, key, value in fix_rows:
            for idx in _resolve(raw):
                ns = segs[idx]['nodes']
                if key == "位移":
                    # 分量预编译一次 (float|callable); fix_node 不收
                    # callable — 函数位移必须逐节点在节点坐标求值
                    parts = [p.strip() for p in str(value).split(",")]
                    vx = _parse_scalar_component(parts[0], "Ux", "位移")
                    vy = (_parse_scalar_component(parts[1], "Uy", "位移")
                          if len(parts) > 1 else None)
                for n in ns:
                    n = int(n)
                    if key == "固定":
                        mesh.fix_node(n, 'both', 0.0)
                    elif key == "fix_ux":
                        mesh.fix_node(n, 'x', 0.0)
                    elif key == "fix_uy":
                        mesh.fix_node(n, 'y', 0.0)
                    elif key == "位移":
                        x = float(mesh.nodes[n][0])
                        y = float(mesh.nodes[n][1])
                        val_x = vx(x, y) if callable(vx) else float(vx)
                        if not math.isfinite(val_x):
                            raise ValueError(
                                f"节点 {n} 位移 Ux 求值非法: {val_x!r} — "
                                f"表达式必须返回有穷数值")
                        mesh.fix_node(n, 'x', val_x)
                        if vy is not None:
                            val_y = vy(x, y) if callable(vy) else float(vy)
                            if not math.isfinite(val_y):
                                raise ValueError(
                                    f"节点 {n} 位移 Uy 求值非法: {val_y!r} "
                                    f"— 表达式必须返回有穷数值")
                            mesh.fix_node(n, 'y', val_y)
                print(f"  {key}: [{idx+1}] {segs[idx]['label']}  {value if key=='位移' else ''}")

        for raw, key, value in load_rows:
            if key == "压力":
                # callable 压力: 组装端 p(xg,yg) 逐 Gauss 点求值, 方向=外法向
                p_val = _parse_scalar_component(str(value), "p", "载荷")
            elif key == "拉力":
                parts = [p.strip() for p in str(value).split(",")]
                tx = _parse_scalar_component(parts[0], "tx", "载荷")
                ty = (_parse_scalar_component(parts[1], "ty", "载荷")
                      if len(parts) > 1 else 0.0)
            for idx in _resolve(raw):
                ns = segs[idx]['nodes']
                for a, b in zip(ns, ns[1:]):
                    a, b = int(a), int(b)
                    if key == "压力":
                        mesh.add_pressure(a, b, p_val)
                    elif key == "拉力":
                        mesh.add_traction(a, b, tx, ty)
                print(f"  {key}: [{idx+1}] {segs[idx]['label']}  {value}")

        if body:
            # 分量可为常数或 x/y 函数 callable — 内核逐高斯点求值
            mesh.body_force = (body[0], body[1])
            print(f"  体力: ({_fmt_body(body[0])}, {_fmt_body(body[1])})")

    def _poll_result(self):
        # 识别/求解期间: 拉取后台阶段消息 → status + 日志;
        # 等待秒数实时递增, 超阈值附加提示 (大型模型 gmsh 网格化
        # 可达 1-5 分钟 — 有反馈才知道没卡死)
        saw_stage = False
        if self._identifying or self._solving:
            while True:
                try:
                    msg = self._progress_q.get_nowait()
                except queue.Empty:
                    break
                saw_stage = True
                self._append_log(msg + "\n")
                self.status.config(text=msg, fg="#b00")
            started = (self._identify_started if self._identifying
                       else self._solve_started)
            elapsed = int(time.monotonic() - started)
            # ── 看门狗: 超时强制失败, 不再无限"识别中" ──
            # gmsh 可能被杀毒/OneDrive 按需下载/文件占用挂起 —
            # 超时后复位界面并给出可操作的原因, 让用户知道发生了什么
            if (self._identifying and elapsed > 180
                    and not getattr(self, "_identify_aborted", False)):
                self._identify_aborted = True
                while True:
                    try:
                        self._result_q.get_nowait()
                    except queue.Empty:
                        break
                self._append_log(
                    "[GUI] 识别超时 (>180s) — 已停止等待。\n"
                    "  可能原因: 杀毒软件拦截 gmsh / OneDrive 云端文件"
                    "同步中 / 文件被占用。请检查后重新识别。\n")
                self._identify_failed("识别超时 (>180s) — 已停止, 见日志")
                return
            elif (self._solving and elapsed > 600
                  and not getattr(self, "_solve_aborted", False)):
                self._solve_aborted = True
                while True:
                    try:
                        self._result_q.get_nowait()
                    except queue.Empty:
                        break
                self._append_log(
                    "[GUI] 求解超时 (>600s) — 已停止等待。\n"
                    "  可能原因: 杀毒拦截 / OneDrive 同步 / 模型过大。"
                    "请检查后重试。\n")
                self._solving = False
                self.run_btn.config(state="normal")
                self.identify_btn.config(state="normal")
                self.status.config(text="求解超时 (>600s) — 见日志",
                                   fg="#b00")
                return
            if saw_stage:
                # 阶段消息至少停留 2 秒再显示秒数 — 用户能看出识别
                # 在推进 (同轮秒数覆盖会把阶段消息闪没)
                self._stage_shown_at = time.monotonic()
            elif (elapsed >= 2 and
                  time.monotonic() - self._stage_shown_at >= 2.0):
                base = "识别边界中" if self._identifying else "求解中"
                extra = ""
                if elapsed > 120:
                    extra = " — 仍在处理 (网格化最长等待 5 分钟), 请耐心"
                elif elapsed > 30:
                    extra = " — 模型较大, 网格化/计算较久, 请耐心"
                self.status.config(
                    text=f"{base}… (已等待 {elapsed}s){extra}", fg="#b00")
        if saw_stage and not self._result_q.empty():
            # 最后阶段消息刚显示 (如"提取边界…") — 延迟一帧处理结果,
            # 让用户看到该阶段 (不能先 get 出 result — 会丢!)
            self.root.after(1500, self._poll_result)
            return
        try:
            rc, text = self._result_q.get_nowait()
        except queue.Empty:
            if self._solving or self._identifying:
                self.root.after(100, self._poll_result)
            return
        try:
            if self._identifying:
                # 识别线程: rc 是 (segs, mesh) 元组或 None
                self._append_log(text)
                if rc is not None:
                    self._apply_identification(rc)
                else:
                    self._identify_failed(text)
                return
            self._solving = False
            self.run_btn.config(state="normal")
            self.identify_btn.config(state="normal")
            self._append_log(text)
            self.status.config(
                text=f"完成 (退出码 {rc})" if rc == 0
                else f"失败 (退出码 {rc})",
                fg="#080" if rc == 0 else "#b00")
            self.state_lbl.config(text=f"rc={rc}")
            if rc == 0 and self._last_png and os.path.exists(self._last_png):
                self._show_image(self._last_png)
            elif rc != 0:
                self.img_label.config(text="求解失败 — 见日志")
        except Exception:
            # 双保险: 内部异常 (已由 _on_tk_error 记日志) 不得跳过
            # 状态复位 — 否则识别/求解标志残留, 按钮永久禁用
            self._identifying = False
            self._solving = False
            self.identify_btn.config(state="normal")
            self.run_btn.config(state="normal")

    def _show_image(self, png):
        try:
            photo = tk.PhotoImage(file=png)
        except tk.TclError as error:
            self._append_log(f"[GUI] 图片显示失败: {error}\n")
            return
        # 缩放适配: 宽度目标 ~640px, 整数倍 subsample
        factor = max(1, photo.width() // 640)
        if factor > 1:
            photo = photo.subsample(factor, factor)
        self.img_label.config(image=photo, text="")
        self.img_label.image = photo   # 防 GC

    def _on_img_click(self, _event=None):
        """点击云图 → 弹窗: 滚轮缩放 / 左键拖动 (PIL 重采样放大)."""
        if not self._last_png or not os.path.exists(self._last_png):
            return
        # 防重开: 已开则前置, 不重复创建 (连点/双击会叠多个弹窗)
        win_old = getattr(self, "_img_win", None)
        if win_old is not None and win_old.winfo_exists():
            win_old.lift()
            return
        try:
            from PIL import Image as PILImage
            base = PILImage.open(self._last_png)
            base.load()
        except Exception as error:
            messagebox.showerror("显示失败", f"无法显示云图: {error}")
            return
        win = tk.Toplevel(self.root)
        self._img_win = win
        win.protocol("WM_DELETE_WINDOW",
                     lambda: (win.destroy(),
                              setattr(self, "_img_win", None)))
        win.title("云图 — 滚轮缩放 / 左键拖动 (关闭窗口返回)")
        win.geometry("1000x760")
        canvas = tk.Canvas(win, bg="white", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        cw, ch = win.winfo_screenwidth() - 200, win.winfo_screenheight() - 200
        st = {"base": base, "scale": 1.0, "ix": 0.0, "iy": 0.0,
              "item": None, "img": None, "drag": None, "cw": cw, "ch": ch}
        st["scale"] = min(1.0, cw / base.width, ch / base.height)
        st["ix"] = (cw - base.width * st["scale"]) / 2
        st["iy"] = (ch - base.height * st["scale"]) / 2

        def _render():
            from PIL import ImageTk
            w = max(1, int(round(base.width * st["scale"])))
            h = max(1, int(round(base.height * st["scale"])))
            if w * h > _MAX_ZOOM_PIX:
                # 大 PNG 放大 5x 内存爆炸 — 按总像素上限截断
                k = math.sqrt(_MAX_ZOOM_PIX / (w * h))
                w, h = max(1, int(w * k)), max(1, int(h * k))
                st["scale"] = min(st["scale"], w / base.width)
            try:
                photo = ImageTk.PhotoImage(
                    base.resize((w, h), PILImage.LANCZOS))
            except MemoryError:
                photo = ImageTk.PhotoImage(
                    base.resize((base.width, base.height), PILImage.LANCZOS))
            if st["item"] is None:
                st["item"] = canvas.create_image(st["ix"], st["iy"],
                                                 anchor="nw")
            else:
                canvas.coords(st["item"], st["ix"], st["iy"])
            canvas.itemconfig(st["item"], image=photo)
            st["img"] = photo   # 防 GC

        def _wheel(event):
            old = st["scale"]
            new = max(0.05, min(5.0, old * 1.2 ** (event.delta / 120.0)))
            if new == old:
                return
            # 鼠标指向的图像点缩放前后保持不动
            rx = (event.x - st["ix"]) / (base.width * old)
            ry = (event.y - st["iy"]) / (base.height * old)
            st["scale"] = new
            st["ix"] = event.x - rx * base.width * new
            st["iy"] = event.y - ry * base.height * new
            _render()

        def _press(event):
            st["drag"] = (event.x, event.y)

        def _motion(event):
            if st["drag"] is None or st["item"] is None:
                return
            x0, y0 = st["drag"]
            st["ix"] += event.x - x0
            st["iy"] += event.y - y0
            st["drag"] = (event.x, event.y)
            canvas.coords(st["item"], st["ix"], st["iy"])

        canvas.bind("<MouseWheel>", _wheel)
        canvas.bind("<ButtonPress-1>", _press)
        canvas.bind("<B1-Motion>", _motion)
        _render()

    def _on_save_png(self):
        if not self._last_png or not os.path.exists(self._last_png):
            messagebox.showinfo("无图片", "还没有可保存的云图 — 先求解一次")
            return
        path = filedialog.asksaveasfilename(
            title="保存云图", defaultextension=".png",
            filetypes=[("PNG", "*.png")], initialfile="fem2d_result.png")
        if not path:
            return
        try:
            import shutil
            shutil.copyfile(self._last_png, path)
            self._append_log(f"[GUI] 已保存: {path}\n")
        except OSError as error:
            messagebox.showerror("保存失败", str(error))

    def _append_log(self, text):
        self.log.config(state="normal")
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.config(state="disabled")


def _fmt(x):
    """刻度标签: 0.5/1/2.5 这类短数字, 省去长尾."""
    return f"{x:g}"


def _fmt_nums(nums):
    """编号列表 → "1,2,3,5-8" 压缩显示 (连续段用 -)."""
    out = []
    run = []
    for n in sorted(nums):
        if run and n == run[-1] + 1:
            run.append(n)
        else:
            if run:
                out.append(f"{run[0]}" if len(run) == 1 else
                           f"{run[0]}-{run[-1]}")
            run = [n]
    if run:
        out.append(f"{run[0]}" if len(run) == 1 else f"{run[0]}-{run[-1]}")
    return ",".join(out)


def _hide_console():
    """隐藏附属控制台窗口 (PyInstaller console exe 双击进 GUI 的场景).

    双击 exe → GUI 窗口 + 后面带个黑控制台窗口不优雅; 失败静默 (无
    控制台/非 Windows/无权限)。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)   # SW_HIDE
    except Exception:
        pass


def _verify_runtime_deps():
    """matplotlib/PIL 兜底 — 打包遗漏时双击可见报错 (在 _hide_console
    前调用, 否则 stderr 已隐藏, 双击 exe 无声无息)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from fem2d import visualize  # noqa: F401 — 主线程预热 matplotlib
        import PIL.Image  # noqa: F401
        import PIL.ImageTk  # noqa: F401
    except Exception as error:
        return (f"FEM2D 启动失败: 缺少运行依赖\n{error}\n\n"
                "请重新安装: pip install matplotlib pillow")
    return None


def _show_fatal(msg):
    """启动失败双路报错: stderr (控制台还在时) + MessageBoxW (隐藏后)."""
    try:
        print(msg)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, msg, "FEM2D 启动失败", 0x10)   # MB_ICONERROR
        except Exception:
            pass


def main():
    """GUI 入口: 无显示环境 (CI) 友好退出."""
    err = _verify_runtime_deps()
    if err:
        _show_fatal(err)
        return 1
    _hide_console()
    try:
        root = tk.Tk()
    except tk.TclError as error:
        _show_fatal(f"FEM2D 无法创建窗口 (无显示环境?):\n{error}")
        return 1
    FEM2DGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
