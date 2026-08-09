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

:menu
echo ==================================================
echo   FEM2D Q4 v9.28.0 交互向导
echo ==================================================
echo  1) 一键演示 demo_complex.geo (固定左端 + 均布载荷)
echo  2) 孔板参数化建模 (输入尺寸, 自动生成模型并求解)
echo  3) 指定模型文件求解 (需要 .geo 或 .msh)
echo  4) 完整自定义 (模型 + 边界 + 载荷逐项输入)
echo  0) 退出
echo ==================================================
set /p CHOICE=请输入数字后回车:

if "%CHOICE%"=="1" goto demo
if "%CHOICE%"=="2" goto holeplate
if "%CHOICE%"=="3" goto userfile
if "%CHOICE%"=="4" goto custom
if "%CHOICE%"=="0" exit /b 0
echo [无效输入] 请输入 0-4 的数字。
goto menu

:demo
echo.
echo 正在运行 demo_complex.geo 演示...
"python\python.exe" run_demo.py
echo.
pause
goto menu

:holeplate
echo.
echo [孔板参数化建模] 直接回车使用默认值
set W=2.0
set H=1.0
set CX=1.0
set CY=0.5
set R=0.3
set LC=0.05
set /p W=板宽 W   [2.0]:
set /p H=板高 H   [1.0]:
set /p CX=孔心x   [1.0]:
set /p CY=孔心y   [0.5]:
set /p R=孔半径 R [0.3]:
set /p LC=网格密度 [0.05]:
del /q params.geo >nul 2>nul
echo W=%W%; H=%H%; CX=%CX%; CY=%CY%; R=%R%; lc=%LC%;> params.geo
echo Point(1)={0,0,0,lc};  Point(2)={W,0,0,lc};>> params.geo
echo Point(3)={W,H,0,lc};  Point(4)={0,H,0,lc};>> params.geo
echo Line(1)={1,2};  Line(2)={2,3};>> params.geo
echo Line(3)={3,4};  Line(4)={4,1};>> params.geo
echo Curve Loop(1)={1,2,3,4};>> params.geo
echo n=16;>> params.geo
echo For i In {0:n-1}>> params.geo
echo   ang=2*Pi*i/n;>> params.geo
echo   Point(100+i)={CX+R*Cos(ang), CY+R*Sin(ang), 0, lc};>> params.geo
echo EndFor>> params.geo
echo For i In {0:n-1}>> params.geo
echo   Line(200+i)={100+i, 100+((i+1)%%n)};>> params.geo
echo EndFor>> params.geo
echo Curve Loop(2)={200:200+n-1};>> params.geo
echo Plane Surface(1)={1,2};>> params.geo
echo Physical Surface("domain",200)={1};>> params.geo
echo Physical Curve("left",101)={4};>> params.geo
echo Physical Curve("right",102)={2};>> params.geo
echo Physical Curve("top",103)={3};>> params.geo
echo Physical Curve("bottom",104)={1};>> params.geo
echo Recombine Surface{1};>> params.geo
echo Mesh 2;>> params.geo
echo Save "params.msh";>> params.geo
echo.
echo [已生成] params.geo  W=%W% H=%H% 孔(%CX%,%CY%) R=%R% lc=%LC%
echo 默认工况: 左边固定 + 均布载荷 0,-78000  (边界标签见模型)
set /p EXTRA=附加参数(回车用默认 --fix left --body 0,-78000):
if "%EXTRA%"=="" set EXTRA=--fix left --body 0,-78000
"python\python.exe" run.py params.geo %EXTRA%
echo.
pause
goto menu

:userfile
echo.
set /p F=输入模型文件路径(如 models\demo_complex.geo):
if "%F%"=="" (
    echo [错误] 未输入文件名。
    goto userfile
)
if not exist "%F%" (
    echo [错误] 文件不存在: %F%
    pause
    goto menu
)
"python\python.exe" run.py %F%
echo.
pause
goto menu

:custom
echo.
set /p F=模型文件(如 models\demo_complex.geo):
if not exist "%F%" (
    echo [错误] 文件不存在: %F%
    pause
    goto menu
)
set /p EXTRA=边界+载荷参数(如 --fix left --body 0,-78000):
"python\python.exe" run.py %F% %EXTRA%
echo.
pause
goto menu
