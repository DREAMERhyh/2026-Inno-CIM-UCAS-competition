@echo off
REM M5 full chain: train 120ep -> rank probe -> readout x3 seeds
REM Smoke measured 37.1 s/epoch => 120ep ~= 75 min (faster than VGG-11's 110 min).
REM Readout runs 3 seeds (not 1): the three recovery rates 72.2/70.2/36.9% only
REM reproduce under a 3-seed average (43.31/59.99, 47.79/68.05, 26.82/72.69);
REM see docs/verify/M5_MODEL_PREP.md sec 7.
REM !! KEEP THIS FILE PURE ASCII !! CMD decodes a UTF-8 .bat as ANSI/GBK.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\anaconda3\envs\pytorch_env\python.exe
set CKPT=checkpoints_cifar100\Exp2_Calib+Layerwise_widecnn_m5\best_model.pth

echo ==== M5 full chain started %DATE% %TIME% ==== >> logs_v7_m5_chain.log

echo ---- M5 train (120 ep) ---- >> logs_v7_m5_chain.log
"%PY%" task_extension6_deep_robust_train.py --dataset cifar100 --model wide_cnn --epochs 120 --seed 42 --tag _m5 >> logs_v7_m5_train.log 2>&1
echo   train exit=%ERRORLEVEL% >> logs_v7_m5_chain.log

echo ---- M5 rank probe (forward only) ---- >> logs_v7_m5_chain.log
"%PY%" v5_probe_rank_collapse.py --dataset cifar100 --arch robust_wide_cnn --ckpt "%CKPT%" --tag _widecnn_m5 >> logs_v7_m5_rank.log 2>&1
echo   rank exit=%ERRORLEVEL% >> logs_v7_m5_chain.log

echo ---- M5 readout seed 42 ---- >> logs_v7_m5_chain.log
"%PY%" v3_readout_repair.py --arch robust_wide_cnn --ckpt "%CKPT%" --backbone_tag widecnn_m5_ep24 --dataset cifar100 --alpha 0.3 --epochs_nat 24 --n_train 20000 --seed 42 >> logs_v7_m5_readout.log 2>&1
echo   readout 42 exit=%ERRORLEVEL% >> logs_v7_m5_chain.log

echo ---- M5 readout seed 43 ---- >> logs_v7_m5_chain.log
"%PY%" v3_readout_repair.py --arch robust_wide_cnn --ckpt "%CKPT%" --backbone_tag widecnn_m5_ep24 --dataset cifar100 --alpha 0.3 --epochs_nat 24 --n_train 20000 --seed 43 >> logs_v7_m5_readout.log 2>&1
echo   readout 43 exit=%ERRORLEVEL% >> logs_v7_m5_chain.log

echo ---- M5 readout seed 44 ---- >> logs_v7_m5_chain.log
"%PY%" v3_readout_repair.py --arch robust_wide_cnn --ckpt "%CKPT%" --backbone_tag widecnn_m5_ep24 --dataset cifar100 --alpha 0.3 --epochs_nat 24 --n_train 20000 --seed 44 >> logs_v7_m5_readout.log 2>&1
echo   readout 44 exit=%ERRORLEVEL% >> logs_v7_m5_chain.log

echo ==== M5 full chain finished %DATE% %TIME% ==== >> logs_v7_m5_chain.log
