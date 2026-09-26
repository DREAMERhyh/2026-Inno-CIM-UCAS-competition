@echo off
REM M14b core (3 runs): ReLU-2pool same-protocol baseline + GELU 2pool + GELU 5pool.
REM Criterion: P = drop_5max - drop_2max (positive = pooling protection);
REM   P(relu) - P(gelu) >= +4 pp -> support ; <= -4 pp -> refute ; in between -> not resolvable.
REM Driver is v3_methods_simplecnn.py (requires the --act 12-line diff, already applied).
REM !! KEEP THIS FILE PURE ASCII !! CMD decodes a UTF-8 .bat as ANSI/GBK.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\anaconda3\envs\pytorch_env\python.exe

echo ==== M14b core started %DATE% %TIME% ==== >> logs_v7_m14b_chain.log

echo ---- 1/3 ReLU-2pool (same-protocol baseline) ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn --act relu --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 42 --tag _m14b >> logs_v7_m14b_baseline.log 2>&1
echo   1/3 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ---- 2/3 GELU-2pool (main arm) ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn --act gelu --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 42 --tag _m14b >> logs_v7_m14b_gelu2.log 2>&1
echo   2/3 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ---- 3/3 GELU-5pool (main arm) ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn_mp --act gelu --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 42 --tag _m14b >> logs_v7_m14b_gelu5.log 2>&1
echo   3/3 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ==== M14b core finished %DATE% %TIME% ==== >> logs_v7_m14b_chain.log
