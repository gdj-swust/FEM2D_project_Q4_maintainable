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
if not "%~1"=="" goto runargs

:interactive
echo ==================================================
echo   FEM2D Q4 v9.28.0 交互终端
echo   输入命令回车运行, 可连续输入; 输入 exit 退出。
echo ==================================================
echo   常用命令:
echo     run.py                    进入交互建模向导
echo     run.py models\demo_complex.geo --fix left --body 0,-78000
echo     run.py models\demo_complex.geo --fix left --traction right:1e6,0
echo     run_demo.py               一键演示
echo     ask_model.bat             菜单式向导
echo     scripts\verify_all.py --fast   环境自检
echo.
:loop
set CMD=
set /p CMD=FEM2D^> 
if not defined CMD goto loop
if "%CMD%"=="exit" goto end
if "%CMD%"=="quit" goto end
if "%CMD%"=="cls" cls
if "%CMD%"=="cls" goto loop
if "%CMD%"=="help" echo 输入 run.py 等命令后回车; exit 退出
if "%CMD%"=="help" goto loop
"python\python.exe" %CMD%
echo.
goto loop
:end
exit /b 0

:runargs
"python\python.exe" %*
echo.
pause
