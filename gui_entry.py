# -*- coding: utf-8 -*-
"""FEM2D 图形界面 — 独立 exe 入口 (PyInstaller 打包用).

本文件是 GUI 应用的**独立入口**, 仅供打包 exe 给朋友/同学使用:
双击 exe → 直接打开图形建模窗口。

与 CLI 完全隔离: run.py 及其交互向导流程**不被修改** — 命令行
用户继续用 `python run.py <文件.geo>`; GUI 用户双击 exe。

打包:
  pyinstaller --onefile --name FEM2D_Q4 --clean --noconfirm \
    --collect-all gmsh \
    --add-binary "tools/gmsh-4.15.2-Windows64/gmsh.exe;tools/gmsh-4.15.2-Windows64" \
    gui_entry.py
"""
import sys


def main():
    from fem2d.gui import main as gui_main
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
