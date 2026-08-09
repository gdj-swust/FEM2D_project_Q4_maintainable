@echo off
chcp 936 >nul
cd /d "%~dp0" 2>nul || (
    echo [错误] 无法进入脚本目录: %~dp0
    echo        请确认压缩包已完整解压，且未修改目录结构。
    pause
    exit /b 1
)
if not exist "python\python.exe" (
    echo [错误] 缺少 python\python.exe - 请确认压缩包完整解压。
    echo        不要把单个文件拷走，必须保留整个目录。
    pause
    exit /b 1
)
echo ==================================================
echo   FEM2D Q4 v9.28.0 一键演示 (便携版, 免安装)
echo   演示模型: models/demo_complex.geo (带孔板)
echo ==================================================
echo.
"python\python.exe" run_demo.py
echo.
pause
