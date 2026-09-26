@echo off
REM M5 smoke: 1 epoch only, to measure s/epoch.
REM M4 verified this extrapolation: smoke 52 s/epoch vs actual 55 s/epoch.
REM Artifacts go to _smoke_m5 dirs (isolated, zero overwrite).
REM !! THIS FILE MUST STAY PURE ASCII !!  CMD decodes a UTF-8 .bat as ANSI/GBK;
REM    non-ASCII comments would be mangled and could break parsing.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\anaconda3\envs\pytorch_env\python.exe

echo ==== M5 smoke started %DATE% %TIME% ==== >> logs_v7_m5_chain.log
"%PY%" task_extension6_deep_robust_train.py --dataset cifar100 --model wide_cnn --epochs 1 --seed 42 --tag _smoke_m5 >> logs_v7_m5_smoke.log 2>&1
echo   smoke exit=%ERRORLEVEL% >> logs_v7_m5_chain.log
echo ==== M5 smoke finished %DATE% %TIME% ==== >> logs_v7_m5_chain.log
