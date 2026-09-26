@echo off
REM ===================================================================
REM  M8c: LP-FT vs pure readout adaptation  (three backbones, serial)
REM  Detached launcher -- survives session death. Do NOT use
REM  run_in_background for long trainings (it is a child of the session).
REM  Path is derived from %~dp0 (CMD resolves it correctly even when the
REM  repo path contains non-ASCII), never hard-coded.
REM ===================================================================
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PY=D:\anaconda3\envs\pytorch_env\python.exe

echo ==== M8c chain started %DATE% %TIME% ==== >> logs_v7_m8c_chain.log

echo ---- 1/3 SimpleCNN (clean backbone) ---- >> logs_v7_m8c_chain.log
"%PY%" v7_m8c_lpft.py --arch simple_cnn --ckpt checkpoints_cifar100\simple_cnn\best_model.pth --backbone_tag clean_simplecnn --lp_epochs 24 --ft_epochs 30 --n_train 20000 >> logs_v7_m8c_simplecnn.log 2>&1
echo   1/3 exit=%ERRORLEVEL% >> logs_v7_m8c_chain.log

echo ---- 2/3 VGG-11 (Exp2) ---- >> logs_v7_m8c_chain.log
"%PY%" v7_m8c_lpft.py --arch robust_vgg11 --ckpt "checkpoints_cifar100\Exp2_Calib+Layerwise_vgg11\best_model.pth" --backbone_tag exp2_vgg11 --lp_epochs 24 --ft_epochs 30 --n_train 20000 >> logs_v7_m8c_vgg11.log 2>&1
echo   2/3 exit=%ERRORLEVEL% >> logs_v7_m8c_chain.log

echo ---- 3/3 ResNet-18 (Exp3) ---- >> logs_v7_m8c_chain.log
"%PY%" v7_m8c_lpft.py --arch robust_resnet18 --ckpt checkpoints_cifar100\Exp3_FullRobust_resnet18\best_model.pth --backbone_tag exp3_resnet18 --lp_epochs 24 --ft_epochs 30 --n_train 20000 >> logs_v7_m8c_resnet18.log 2>&1
echo   3/3 exit=%ERRORLEVEL% >> logs_v7_m8c_chain.log

echo ==== M8c chain finished %DATE% %TIME% ==== >> logs_v7_m8c_chain.log
