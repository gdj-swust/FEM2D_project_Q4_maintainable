"""D3 判别性测试 — Windows GBK 乱码 (输出编码策略).

判别性 (回滚改动必须红):
  - GBK 流 (中文 Windows 重定向的 locale 编码) 强制为 UTF-8 —
    中文与非 GBK 字形 (⁻/ℓ) 完整保留, 不替换成 '?', 不崩溃
  - run.py 子进程重定向输出可被 UTF-8 解码 (修复前是 GBK 字节)

定位结论 (写入提交说明): 根因不在 runner.main 的 reconfigure_streams
调用本身, 而是 errors.py 的 reconfigure_streams 只设 errors=replace
不设 encoding — 重定向 stdout 按 locale (cp936) 编码。errors.py/
runner.py 属 B 结构轮边界 (禁碰), D 轮在 run.py 脚本入口补齐;
run_demo.py 与 console script 入口需要同款处理, 转交 B 轮。
"""
import io
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gbk_stream():
    """模拟中文 Windows 重定向: locale 编码 cp936 的 TextIOWrapper."""
    raw = io.BytesIO()
    return io.TextIOWrapper(raw, encoding="gbk", errors="replace"), raw


def test_gbk_stream_forced_to_utf8(monkeypatch):
    """判别性核心: GBK 流重配为 UTF-8 — 中文 + 非 GBK 字形完整保留."""
    import run as run_entry

    gbk_stream, raw = _gbk_stream()
    monkeypatch.setattr(sys, "stdout", gbk_stream)
    monkeypatch.setattr(sys, "stderr", gbk_stream)
    run_entry.force_utf8_streams()
    print("中文输出 → × ⁻ ℓ ≈")      # ⁻/ℓ 不在 GBK — 旧行为替换成 '?' 或崩溃
    gbk_stream.flush()
    payload = raw.getvalue()
    text = payload.decode("utf-8")    # 旧行为: GBK 字节 → UnicodeDecodeError
    assert text.strip() == "中文输出 → × ⁻ ℓ ≈"
    assert "?" not in text


def test_gbk_stream_keeps_reconfigure_failure_silent(monkeypatch):
    """不支持 reconfigure 的流 (StringIO): 静默保持原样, 不崩溃."""
    import run as run_entry

    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buffer)
    monkeypatch.setattr(sys, "stderr", buffer)
    run_entry.force_utf8_streams()
    print("保持原流")
    assert buffer.getvalue() == "保持原流\n"


def test_run_py_subprocess_redirected_output_is_utf8():
    """e2e: run.py 子进程 (stdout=管道) 输出可被 UTF-8 解码.

    修复前中文 Windows 下管道输出为 GBK 字节 → 本测试红.
    """
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)   # 排除外部设置干扰
    completed = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env, timeout=120)
    payload = completed.stdout
    text = payload.decode("utf-8")      # 修复前: GBK 字节 → 解码失败
    assert "入口选择指南" in text
