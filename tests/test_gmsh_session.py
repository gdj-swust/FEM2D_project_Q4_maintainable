"""pkg11 A6 — _gmsh_session 生命周期判别性测试.

gmsh 是进程级单例: 会话创建/复用顺序错误 (双重 initialize /
双重 finalize) 会抛底层异常或破坏外部会话。本测试用假模块
(fake gmsh) 锁定契约, 不需要真实 gmsh 环境:
  1. 未初始化 → 进入时 initialize 恰好一次, 退出时 finalize 恰好一次
  2. 已初始化 (外部持有) → 不得 initialize 也不得 finalize
  3. 顺序复用: 连续两次会话各自成对, 不双重 finalize
"""
import pytest

from fem2d.gmsh_adapter import _gmsh_session


class _FakeGmsh:
    """记录生命周期调用的假 gmsh 模块."""

    def __init__(self, initialized=False):
        self.calls = []
        self._init = initialized

    def isInitialized(self):
        return self._init

    def initialize(self):
        self.calls.append("initialize")
        self._init = True

    def finalize(self):
        self.calls.append("finalize")
        self._init = False

    @property
    def model(self):
        raise AssertionError("business logic must not run in lifecycle tests")


def test_session_creates_and_finalizes_exactly_once():
    """未初始化模块: initialize 进入一次, finalize 退出一次."""
    fake = _FakeGmsh(initialized=False)
    with _gmsh_session(fake) as active:
        assert active is fake
        assert fake.calls == ["initialize"]
        assert fake.isInitialized()
    assert fake.calls == ["initialize", "finalize"]
    assert not fake.isInitialized()


def test_session_reuses_external_session_without_finalize():
    """已初始化模块 (外部持有): 不得 initialize, 也不得 finalize.

    判别性: 外部会话在 with 退出后必须仍可继续使用 — 曾重复
    finalize 抛异常且破坏外部会话.
    """
    fake = _FakeGmsh(initialized=True)
    with _gmsh_session(fake) as active:
        assert active is fake
        assert fake.calls == []
    assert fake.calls == []
    assert fake.isInitialized()  # 外部会话原样保留


def test_session_sequence_pairs_initialize_and_finalize():
    """连续两次会话: 各自成对, 禁止双重 finalize."""
    fake = _FakeGmsh(initialized=False)
    with _gmsh_session(fake):
        with _gmsh_session(fake):
            pass  # 已初始化 → 第二层复用, 不重复 initialize
        assert fake.calls == ["initialize"]   # 外层持有, 内层不 finalize
    assert fake.calls == ["initialize", "finalize"]
    # 第二次独立会话
    with _gmsh_session(fake):
        pass
    assert fake.calls == ["initialize", "finalize", "initialize", "finalize"]


def test_session_finalizes_when_body_raises():
    """异常退出也要 finalize — 泄漏的会话会让后续测试互相污染."""
    fake = _FakeGmsh(initialized=False)
    with pytest.raises(RuntimeError, match="boom"):
        with _gmsh_session(fake):
            raise RuntimeError("boom")
    assert fake.calls == ["initialize", "finalize"]


def test_session_module_loaded_lazily(monkeypatch):
    """gmsh_module=None → _load_gmsh_module 惰性加载 (不在 import 时)."""
    loaded = []

    def _fake_load():
        loaded.append(True)
        return _FakeGmsh(initialized=False)

    monkeypatch.setattr(
        "fem2d.gmsh_adapter._load_gmsh_module", _fake_load)
    with _gmsh_session() as active:
        assert loaded == [True]
        assert active.isInitialized()
