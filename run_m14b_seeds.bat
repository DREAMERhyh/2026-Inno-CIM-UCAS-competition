@echo off
REM M14b seed supplementation: 3 configs x seeds 43,44 (s42 already done).
REM WHY: single-seed noise on D is ~2.2 pp, so D=+2.57 falls inside the +-4 pp
REM band => "not resolvable". Adding 2 more seeds per config cuts the noise to
REM ~1.27 pp, which would make +2.57 exceed 2 sigma => resolvable.
REM relu-5max already has 3 seeds (18.52/22.20/21.09, mean 20.60); the other
REM five config-seed cells are what we need.
REM --tag carries the seed suffix so nothing overwrites the s42 artifacts.
REM !! KEEP THIS FILE PURE ASCII !! CMD decodes a UTF-8 .bat as ANSI/GBK.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\anaconda3\envs\pytorch_env\python.exe

echo ==== M14b seeds started %DATE% %TIME% ==== >> logs_v7_m14b_chain.log

echo ---- relu-2pool s43 ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn --act relu --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 43 --tag _m14b_s43 >> logs_v7_m14b_seeds.log 2>&1
echo   relu-2 s43 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ---- relu-2pool s44 ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn --act relu --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 44 --tag _m14b_s44 >> logs_v7_m14b_seeds.log 2>&1
echo   relu-2 s44 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ---- gelu-2pool s43 ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn --act gelu --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 43 --tag _m14b_s43 >> logs_v7_m14b_seeds.log 2>&1
echo   gelu-2 s43 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ---- gelu-2pool s44 ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn --act gelu --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 44 --tag _m14b_s44 >> logs_v7_m14b_seeds.log 2>&1
echo   gelu-2 s44 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ---- gelu-5pool s43 ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn_mp --act gelu --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 43 --tag _m14b_s43 >> logs_v7_m14b_seeds.log 2>&1
echo   gelu-5 s43 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ---- gelu-5pool s44 ---- >> logs_v7_m14b_chain.log
"%PY%" v3_methods_simplecnn.py --arch simple_cnn_mp --act gelu --mode clean --profile uniform --dataset cifar100 --epochs 120 --seed 44 --tag _m14b_s44 >> logs_v7_m14b_seeds.log 2>&1
echo   gelu-5 s44 exit=%ERRORLEVEL% >> logs_v7_m14b_chain.log

echo ==== M14b seeds finished %DATE% %TIME% ==== >> logs_v7_m14b_chain.log
