"""平面应力/应变验证 — 兼容薄壳 (实现已迁至 fem2d/verification.py).

wheel 打包不含 tests/, --self-test 从 tests 导入会在正式安装后失效。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fem2d.verification import run_plane_verification

if __name__ == "__main__":
    p, f = run_plane_verification()
    print("\n" + "=" * 55)
    print(f"  {p} PASS, {f} FAIL")
    print("=" * 55)
    sys.exit(0 if f == 0 else 1)
