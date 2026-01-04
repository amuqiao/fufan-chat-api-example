@echo off
REM 启动服务的批处理文件，避免MinGW环境下的TP_NUM_C_BUFS错误

REM 设置环境变量
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=1

REM 启动服务
python startup.py

REM 等待用户输入，以便查看输出
pause