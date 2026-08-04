"""runner._ensure_patch_test 缓存契约锁定 (组 D 判别性测试).

契约: patch test **通过**才缓存该 (单元, 平面) 组合 (每进程跑一次);
**失败**不缓存 — 同进程第二次调用必须重新运行并再次抛 CliError,
不能因缓存污染而静默放行。

判别性: test_patch_test_failure_not_cached 放回旧实现 (先
_patch_checked.add(key) 再校验) 第二次调用直接 return, 断言必失败。
"""
import pytest

import fem2d.runner as runner_mod
from fem2d.errors import CliError


@pytest.fixture(autouse=True)
def _fresh_cache():
    """每个测试独立缓存 — 避免跨测试/跨文件残留污染."""
    runner_mod._patch_checked.clear()
    yield
    runner_mod._patch_checked.clear()


def _counting_patch(result):
    """返回 (count 容器, fake run_patch_test)."""
    calls = {"n": 0}

    def fake(*args, **kwargs):
        calls["n"] += 1
        return dict(result)

    return calls, fake


def test_patch_test_failure_not_cached(monkeypatch):
    """失败后同进程第二次调用必须仍然失败 (旧实现第二次静默成功).

    失败不缓存: 修复单元代码后同进程重试应重新运行, 而非被
    第一次失败留下的缓存条目放行。
    """
    calls, fake = _counting_patch({"all_passed": False})
    monkeypatch.setattr(runner_mod, "run_patch_test", fake)

    with pytest.raises(CliError):
        runner_mod._ensure_patch_test()
    # 第二次调用: 缓存若已污染则直接返回 — 必须仍然失败
    with pytest.raises(CliError):
        runner_mod._ensure_patch_test()
    assert calls["n"] == 2


def test_patch_test_success_cached_once(monkeypatch):
    """通过后缓存生效: 同进程第二次调用不再重复运行."""
    calls, fake = _counting_patch({"all_passed": True})
    monkeypatch.setattr(runner_mod, "run_patch_test", fake)

    runner_mod._ensure_patch_test()
    runner_mod._ensure_patch_test()
    assert calls["n"] == 1


def test_patch_test_cache_keyed_by_pair(monkeypatch):
    """缓存按 (单元, 平面) 组合隔离 — 不同组合互不干扰."""
    calls = {"n": 0}

    def fake(*args, **kwargs):
        calls["n"] += 1
        return {"all_passed": True}

    monkeypatch.setattr(runner_mod, "run_patch_test", fake)

    runner_mod._ensure_patch_test("CPS3", "stress")
    runner_mod._ensure_patch_test("CPS3", "stress")   # 命中缓存
    runner_mod._ensure_patch_test("CPS4", "stress")   # 新组合, 必须重跑
    runner_mod._ensure_patch_test("CPS4", "strain")   # 新组合, 必须重跑
    assert calls["n"] == 3


def test_patch_test_failure_does_not_poison_other_keys(monkeypatch):
    """失败只属于本次组合 — 其余组合不受影响, 同组合修复后重试成功."""
    fail = {"CPS3"}  # 模拟 CPS3 单元代码损坏, CPS4 正常

    def fake(*args, **kwargs):
        elem_type = kwargs.get("elem_type", "CPS3")
        return {"all_passed": elem_type not in fail}

    monkeypatch.setattr(runner_mod, "run_patch_test", fake)

    with pytest.raises(CliError):
        runner_mod._ensure_patch_test("CPS3", "stress")
    with pytest.raises(CliError):
        runner_mod._ensure_patch_test("CPS3", "stress")

    # 失败期间另一组合正常通过并缓存
    runner_mod._ensure_patch_test("CPS4", "stress")
    runner_mod._ensure_patch_test("CPS4", "stress")

    # CPS3 单元代码修复 (从 fail 集合移除) — 必须重新运行并放行
    fail.discard("CPS3")
    runner_mod._ensure_patch_test("CPS3", "stress")
