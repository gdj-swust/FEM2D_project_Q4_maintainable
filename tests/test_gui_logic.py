# -*- coding: utf-8 -*-
"""GUI 逻辑层测试 — 无 tk 窗口依赖 (CI 无显示环境可跑).

覆盖: 解析函数全矩阵 / _apply_model_bcs 函数载荷落盘 / 线程 SystemExit
兜底 / 材料校验 / 识别失败状态重置 / 预览空坐标防护.
"""
import math
import queue
from types import SimpleNamespace

import numpy as np
import pytest

from fem2d.gui import (FEM2DGUI, _parse_scalar_component, parse_bc_value,
                       parse_body)
from fem2d.mesh import Mesh


class Stub:
    """最小 tk 控件替身: 任意方法调用记录, 返回自身."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _m(*a, **k):
            self.calls.append((name, a, k))
            return self
        return _m


def _bare_gui():
    """不经 __init__ 的实例 (只跑纯逻辑方法, 不建窗口)."""
    return object.__new__(FEM2DGUI)


def _make_model():
    """矩形 4 节点 2 单元 + 4 条边 — 各边均真实边界边."""
    mesh = Mesh(nodes=[[0., 0.], [2., 0.], [2., 1.], [0., 1.]],
                elements=[[0, 1, 2], [0, 2, 3]],
                thickness=0.01, E=2.1e11, nu=0.3)
    segs = [{"nodes": np.array([0, 1]), "label": "底", "type": "line"},
            {"nodes": np.array([1, 2]), "label": "右", "type": "line"},
            {"nodes": np.array([2, 3]), "label": "顶", "type": "line"},
            {"nodes": np.array([3, 0]), "label": "左", "type": "line"}]
    return SimpleNamespace(mesh=mesh, segs=segs, region_registry=None)


# ── 解析函数全矩阵 ──

def test_parse_bc_constant_forms():
    assert parse_bc_value("位移", "0.001,0.005") == "0.001,0.005"
    assert parse_bc_value("位移", "0.001") == "0.001"
    assert parse_bc_value("载荷", "1e5") == "1e5"
    assert parse_bc_value("载荷", "1e6,0") == "1e6,0"
    # 全角逗号归一 (CLI parse_vec2 同款契约)
    assert parse_bc_value("位移", "0.001，0.005") == "0.001,0.005"


def test_parse_bc_function_forms():
    # 函数分量通过 (返回原文, 显示原样)
    assert parse_bc_value("载荷", "1000*exp(x/2)") == "1000*exp(x/2)"
    assert parse_bc_value("载荷", "1e6, -500*y") == "1e6, -500*y"
    assert parse_bc_value("位移", "0.001*x, 0.002*y") == "0.001*x, 0.002*y"


@pytest.mark.parametrize("kind,raw", [
    ("位移", ""),
    ("载荷", "1e999"),          # 溢出 → inf 拒绝
    ("载荷", "nan"),
    ("载荷", "inf"),
    ("位移", "0.001,inf"),
    ("载荷", "sinh(x)"),        # 白名单外函数
    ("位移", "a,b,c"),           # 3 分量
    ("载荷", "exp(1)"),          # 常数表达式无 x/y — 拒绝 (与 CLI 契约一致)
    ("载荷", "0.001, 0.002, 0.003"),
])
def test_parse_bc_rejects(kind, raw):
    with pytest.raises(ValueError):
        parse_bc_value(kind, raw)


def test_parse_scalar_component():
    assert _parse_scalar_component("5", "t", "载荷") == 5.0
    v = _parse_scalar_component("1000*exp(x/2)", "t", "载荷")
    assert callable(v)
    assert abs(v(2.0, 0.0) - 1000 * math.exp(1.0)) < 1e-9
    with pytest.raises(ValueError):
        _parse_scalar_component("1e999", "t", "载荷")
    with pytest.raises(ValueError):
        _parse_scalar_component("", "t", "载荷")


def test_parse_body_regression():
    assert parse_body("", "-78000") == [0.0, -78000.0]
    assert parse_body("", "") is None
    with pytest.raises(ValueError):
        parse_body("1e999", "")
    with pytest.raises(ValueError):
        parse_body("exp(1)", "")     # 无 x/y 的常数表达式


# ── _apply_model_bcs 函数载荷/位移落盘 (直连, 不求解) ──

def test_apply_displacement_func_per_node():
    """边1 = 节点(0,0)-(2,0): Ux=0.001*x, Uy=0.002*y 逐节点求值."""
    gui = _bare_gui()
    model = _make_model()
    gui._apply_model_bcs(model, [("1", "位移", "0.001*x, 0.002*y")],
                         [], None)
    pv = model.mesh.prescribed_vals
    assert pv[2 * 0] == pytest.approx(0.0)      # 节点0 (0,0) Ux
    assert pv[2 * 0 + 1] == pytest.approx(0.0)  # 节点0 (0,0) Uy
    assert pv[2 * 1] == pytest.approx(0.002)    # 节点1 (2,0) Ux
    assert pv[2 * 1 + 1] == pytest.approx(0.0)  # 节点1 (2,0) Uy
    # 不在该边上的节点不约束
    assert 2 * 2 not in pv
    assert 2 * 3 + 1 not in pv


def test_apply_pressure_func_callable():
    gui = _bare_gui()
    model = _make_model()
    gui._apply_model_bcs(model, [], [("2", "压力", "1000*exp(x/2)")], None)
    tr = model.mesh.surface_tractions
    assert len(tr) == 1 and tr[0]["is_pressure"]
    p = tr[0]["traction"][0]
    assert callable(p)
    assert abs(p(2.0, 0.0) - 1000 * math.exp(1.0)) < 1e-9


def test_apply_traction_mixed_tuple():
    gui = _bare_gui()
    model = _make_model()
    gui._apply_model_bcs(model, [], [("3", "拉力", "1e6, -500*y")], None)
    tx, ty = model.mesh.surface_tractions[0]["traction"]
    assert tx == 1e6
    assert callable(ty)
    assert ty(1.0, 0.5) == -250.0


def test_apply_body_func_callable():
    gui = _bare_gui()
    model = _make_model()
    bx = _parse_scalar_component("1000*exp(x/2)", "bx", "体力")
    gui._apply_model_bcs(model, [], [], [bx, -78000.0])
    assert callable(model.mesh.body_force[0])
    assert model.mesh.body_force[1] == -78000.0


def test_apply_constant_regression():
    """常数路径与改造前逐位一致."""
    gui = _bare_gui()
    model = _make_model()
    gui._apply_model_bcs(model,
                         [("1", "位移", "0.001,0.005")],
                         [("2", "压力", "1e5"), ("3", "拉力", "1e6,0")],
                         [0.0, -78000.0])
    pv = model.mesh.prescribed_vals
    assert pv[2 * 0] == 0.001 and pv[2 * 0 + 1] == 0.005
    tr = model.mesh.surface_tractions
    assert tr[0]["traction"] == (1e5,) and tr[0].get("is_pressure")
    assert tr[1]["traction"] == (1e6, 0.0)
    assert not tr[1].get("is_pressure")
    assert model.mesh.body_force == (0.0, -78000.0)


# ── 线程 SystemExit 兜底 (防永久卡死) ──

def test_identify_thread_system_exit(monkeypatch):
    import fem2d.runner as runner

    def boom(*a, **k):
        raise SystemExit(3)
    monkeypatch.setattr(runner, "_resolve_input", boom)
    gui = _bare_gui()
    gui._progress_q = queue.Queue()
    gui._result_q = queue.Queue()
    gui._identify_thread(["x.geo", "--E", "2.1e11", "--nu", "0.3",
                          "--thickness", "0.01"])
    result, text = gui._result_q.get_nowait()
    assert result is None                    # 失败路径 → _identify_failed
    assert "SystemExit" in text              # 日志有原因


def test_solve_thread_system_exit_code_not_int(monkeypatch):
    import fem2d.runner as runner

    def boom(*a, **k):
        raise SystemExit("boom")             # 非 int code
    monkeypatch.setattr(runner, "_resolve_input", boom)
    gui = _bare_gui()
    gui._progress_q = queue.Queue()
    gui._result_q = queue.Queue()
    gui._solve_thread(("x.geo", "png.png", [], [], None,
                       "2.1e11", "0.3", "0.01"))
    rc, _text = gui._result_q.get_nowait()
    assert rc == 2                           # 不得误报 rc=0"完成"


def test_solve_thread_system_exit_int_code(monkeypatch):
    import fem2d.runner as runner

    def boom(*a, **k):
        raise SystemExit(3)
    monkeypatch.setattr(runner, "_resolve_input", boom)
    gui = _bare_gui()
    gui._progress_q = queue.Queue()
    gui._result_q = queue.Queue()
    gui._solve_thread(("x.geo", "png.png", [], [], None,
                       "2.1e11", "0.3", "0.01"))
    rc, _text = gui._result_q.get_nowait()
    assert rc == 3


# ── 材料校验 / 状态重置 / 预览防护 ──

def test_validate_material_rejects_bad():
    gui = _bare_gui()
    gui.entry = {"E": SimpleNamespace(get=lambda: "abc"),
                 "nu": SimpleNamespace(get=lambda: "0.3"),
                 "thickness": SimpleNamespace(get=lambda: "0.01")}
    assert gui._validate_material() == "'E' 需要数值 (得到 'abc')"


def test_validate_material_nu_range():
    gui = _bare_gui()
    gui.entry = {"E": SimpleNamespace(get=lambda: "2.1e11"),
                 "nu": SimpleNamespace(get=lambda: "0.9"),
                 "thickness": SimpleNamespace(get=lambda: "0.01")}
    assert "nu" in gui._validate_material()


def test_validate_material_ok():
    gui = _bare_gui()
    gui.entry = {"E": SimpleNamespace(get=lambda: "2.1e11"),
                 "nu": SimpleNamespace(get=lambda: "0.3"),
                 "thickness": SimpleNamespace(get=lambda: "0.01")}
    assert gui._validate_material() is None


def test_identify_failed_full_reset():
    """识别失败后旧模型/旧 BC 不得残留 (防静默错边)."""
    gui = _bare_gui()
    gui._identifying = True
    gui._segs = [object()]      # 非空残留
    gui._mesh = object()
    gui._pv = {"scale": 1.0}
    gui._fix_items = [(1, 2, 3, ())]
    gui._load_items = [(1, 2, 3, ())]
    for attr in ("identify_btn", "run_btn", "edge_combo", "fix_list",
                 "load_list", "preview", "status"):
        setattr(gui, attr, Stub())
    gui._identify_failed("失败原因")
    assert gui._segs is None and gui._mesh is None and gui._pv is None
    assert gui._fix_items == [] and gui._load_items == []
    assert gui._identifying is False
    assert ("delete", ("all",), {}) in gui.preview.calls


def test_draw_preview_nan_coords_no_crash():
    gui = _bare_gui()
    gui._segs = [{"nodes": [0], "coords": np.array([[np.nan, 0.0]])}]
    gui.preview = Stub()
    gui._append_log = lambda *a, **k: None
    gui._draw_preview()          # 不得抛 (历史: np.min 对 NaN 裸抛)
    assert ("delete", ("all",), {}) in gui.preview.calls


# ── 按名字全选整组 (Physical Curve 组 / @组名) ──

def _seg_with_phys(phys_name, label, n=5):
    """带物理组信息的段 (GUI 识别结果同构)."""
    return {"nodes": np.arange(n), "coords": np.zeros((n, 2)),
            "label": label, "type": "line",
            "info": {"physical_names": (phys_name,)}}


class _FakeRegistry:
    """by_name 命中 → 非空组记录 (组展开路径只需非 None 判真)."""

    def by_name(self, name, dimension=1):
        return {"椭圆孔": object()} if name == "椭圆孔" else None


def _gui_attrs(gui):
    for attr in ("identify_btn", "run_btn", "edge_combo", "fix_list",
                 "load_list", "preview", "status", "edge_hint",
                 "state_lbl"):
        setattr(gui, attr, Stub())
    gui._fix_items = []
    gui._load_items = []
    gui._pv = None
    gui._geo_label = "测试模型"


def test_apply_identification_group_entries(monkeypatch):
    """下拉: Physical Curve 组条目优先 (≡ 椭圆孔 (2条)), 段条目缩短.

    80+ 条边时用户按组名全选, 不用逐个数编号; 长复合 label 不进下拉.
    注: _apply_identification 会 showinfo("识别完成") — 无 root 时 tkinter
    Dialog 自动创建 Tk 并弹真实模态窗 (本机有显示 → 测试永久卡死; CI 无
    显示抛 TclError 被 try/except 吞 → 通过). 必须 patch 掉 messagebox.
    """
    gui = _bare_gui()
    _gui_attrs(gui)
    monkeypatch.setattr("fem2d.gui.messagebox", Stub())
    segs = [
        _seg_with_phys("椭圆孔", "椭圆孔 | 内孔 直边 (0.1,0.2)→(0.3,0.4)"),
        _seg_with_phys("椭圆孔", "椭圆孔 | 内孔 直边 (0.3,0.4)→(0.5,0.6)"),
        _seg_with_phys("底部", "底部 | 外边 直边 (0,0)→(1,0)"),
        {"nodes": np.arange(4), "coords": np.zeros((4, 2)),
         "label": "无组 | 内孔 直边 (0,0)→(1,1)", "type": "line",
         "info": {"is_outer": False}},
    ]
    gui._apply_identification((segs, object(), _FakeRegistry()))
    values = [c[2]["values"]
              for c in gui.edge_combo.calls if c[0] == "config"][0]
    # 组条目在前 (按名字排序): 底部 + 椭圆孔 + 拓扑孔组
    assert "≡ 椭圆孔 (2条)" in values
    assert "≡ 底部 (1条)" in values
    # 无 is_outer 标记的段默认非外边 → 全部计入拓扑孔组 (4 段),
    # 无物理组的网格仍可整组选孔
    assert "≡ 内孔 (4条)" in values
    assert values.index("≡ 底部 (1条)") < values.index("≡ 椭圆孔 (2条)")
    # 段条目缩短: 复合 label 的 "|" 不进下拉
    seg_rows = [v for v in values if v[0].isdigit()]
    assert len(seg_rows) == 4
    assert all("(0.1,0.2)" not in v for v in seg_rows)
    assert seg_rows[0].startswith("1 | ")
    # registry 存下供 @组名 路径使用
    assert gui._region_registry is not None


def test_add_bc_at_group_name_full_select():
    """@椭圆孔 全选整组 — region_registry 必须传进选择器 (曾传 None)."""
    gui = _bare_gui()
    _gui_attrs(gui)
    gui._identifying = False
    gui._segs = [_seg_with_phys("椭圆孔", "椭圆孔 | 内孔 直边 A"),
                 _seg_with_phys("椭圆孔", "椭圆孔 | 内孔 直边 B")]
    gui._region_registry = _FakeRegistry()
    gui.edge_combo.get = lambda: "@椭圆孔"
    gui.fix_value = Stub()
    gui.fix_value.get = lambda: "0.001,0.005"
    gui._draw_preview = lambda: None
    gui._add_bc("fix")
    assert len(gui._fix_items) == 1
    assert gui._fix_items[0][3] == (0, 1)    # 两段全选


def test_add_bc_plain_group_name_no_registry():
    """无 registry 时 (外部网格) 输入组名仍按 label/物理名精确匹配."""
    gui = _bare_gui()
    _gui_attrs(gui)
    gui._identifying = False
    gui._segs = [_seg_with_phys("椭圆孔", "椭圆孔 | 内孔 直边 A"),
                 _seg_with_phys("底部", "底部 | 外边 直边 B")]
    gui._region_registry = None
    gui.edge_combo.get = lambda: "椭圆孔"
    gui.fix_value = Stub()
    gui.fix_value.get = lambda: "0.001,0.005"
    gui._draw_preview = lambda: None
    gui._add_bc("fix")
    assert gui._fix_items[0][3] == (0,)     # 精确匹配物理名, 不落到别边


# ── 路径自动识别 (trace) ──

class _RootStub:
    """root 替身: after/after_cancel 记录, 可手动触发回调."""

    def __init__(self):
        self.after_calls = []
        self.cancelled = set()
        self._id = 0

    def after(self, ms, cb):
        self._id += 1
        self.after_calls.append((self._id, ms, cb))
        return self._id

    def after_cancel(self, task_id):
        self.cancelled.add(task_id)

    def fire(self, task_id):
        for tid, _, cb in self.after_calls:
            if tid == task_id:
                cb()
                return


def test_on_path_trace_auto_identify():
    """路径框变化 → 500ms 防抖后自动识别 (粘贴/手输同浏览行为)."""
    import os
    gui = _bare_gui()
    gui._identifying = gui._solving = False
    gui._path_after = None
    gui.root = _RootStub()
    started = []
    gui._start_identify = lambda path: started.append(path)
    here = os.path.abspath(__file__)      # 真实存在的文件
    # 初始空路径 → 不调度
    gui.path_var = SimpleNamespace(get=lambda: "  ")
    gui._on_path_trace()
    assert not gui.root.after_calls
    # 变为有效路径 → 调度 (500ms 防抖)
    gui.path_var = SimpleNamespace(get=lambda: here)
    gui._on_path_trace()
    # 连续变化: 第二次取消第一次, 只保留最后一次调度
    gui._on_path_trace()
    scheduled = [c for c in gui.root.after_calls
                 if c[0] not in gui.root.cancelled]
    assert len(scheduled) == 1
    tid, ms, cb = scheduled[0]
    assert ms == 500
    gui.root.fire(tid)
    assert started == [here]


def test_on_path_trace_skips_busy():
    """识别/求解进行中不重复调度."""
    gui = _bare_gui()
    gui._identifying = True
    gui._solving = False
    gui._path_after = None
    gui.root = _RootStub()
    gui.path_var = SimpleNamespace(get=lambda: r"C:\models\demo.geo")
    gui._on_path_trace()
    assert not gui.root.after_calls


def test_on_path_trace_empty_path_skipped():
    """空路径/无文件不调度."""
    gui = _bare_gui()
    gui._identifying = gui._solving = False
    gui._path_after = None
    gui.root = _RootStub()
    gui.path_var = SimpleNamespace(get=lambda: "   ")
    gui._on_path_trace()
    assert not gui.root.after_calls


# ── 识别/求解看门狗 ──

def test_poll_watchdog_identify_timeout():
    """识别 >180s 无结果 → 强制失败并复位 (不再无限'识别中')."""
    gui = _bare_gui()
    gui._gui_attrs_for_poll = True
    _gui_attrs(gui)
    gui._identifying = True
    gui._solving = False
    gui._identify_aborted = False
    gui._identify_started = 0.0          # 很久以前 (模拟超时)
    gui._stage_shown_at = 0.0
    gui._progress_q = queue.Queue()      # 无阶段消息
    gui._result_q = queue.Queue()        # 无结果
    failed = []
    gui._identify_failed = lambda text: failed.append(text)
    gui._append_log = lambda *a: None
    gui._poll_result()
    assert failed and "180s" in failed[0]


def test_poll_watchdog_solve_timeout():
    """求解 >600s 无结果 → 强制复位 (按钮恢复可用)."""
    gui = _bare_gui()
    _gui_attrs(gui)
    gui._identifying = False
    gui._solving = True
    gui._solve_aborted = False
    gui._solve_started = 0.0
    gui._stage_shown_at = 0.0
    gui._progress_q = queue.Queue()
    gui._result_q = queue.Queue()
    gui._append_log = lambda *a: None
    gui._poll_result()
    assert not gui._solving
    # run_btn/identify_btn 复位为 normal
    btn_calls = [c for c in gui.run_btn.calls if c[0] == "config"]
    assert btn_calls and btn_calls[-1][2].get("state") == "normal"


def test_poll_watchdog_identify_recent_no_abort():
    """识别刚开始 (<180s) 不看门狗触发 — 正常轮询继续."""
    gui = _bare_gui()
    _gui_attrs(gui)
    import time
    gui.root = _RootStub()
    gui._identifying = True
    gui._solving = False
    gui._identify_aborted = False
    gui._identify_started = time.monotonic()
    gui._stage_shown_at = time.monotonic()
    gui._progress_q = queue.Queue()
    gui._result_q = queue.Queue()
    gui._append_log = lambda *a: None
    failed = []
    gui._identify_failed = lambda text: failed.append(text)
    gui._poll_result()
    assert not failed
    assert gui._identify_aborted is False
