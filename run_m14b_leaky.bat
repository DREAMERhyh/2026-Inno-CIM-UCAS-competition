@echo off
REM M14b supplementary arms (only if time permits) -- LeakyReLU is a WEAK manipulation:
REM negative-value energy fraction is 1.0e-04 vs GELU's 6.4e-03 (63x).
REM If the leaky arm shows no effect, it must NOT be used to refute the hypothesis
REM (cannot tell "hypothesis wrong" from "manipulation too weak").
REM !! KEEP THIS FILE PURE ASCII !! CMD decodes a UTF-8 .bat as ANSI/GBK.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\anaconda3\envs\pytorch_env\python.exe

echo ==== M14b leaky started %DATE% %TIME% ==== >> logs_v7_m14b_chain.log

echo ---- 4/5 Leaky-2pool ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn --act leaky --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 42 --tag _m14b >> logs_v7_m14b_leaky2.log 2>&1
echo   4/5 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ---- 5/5 Leaky-5pool ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn_mp --act leaky --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 42 --tag _m14b >> logs_v7_m14b_leaky5.log 2>&1
echo   5/5 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ==== M14b leaky finished %DATE% %TIME% ==== >> logs_v7_m14b_chain.log
