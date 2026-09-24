# =====================================================================
#  M4 训练链守护进程（脱离会话运行）
# =====================================================================
#  立此脚本的直接原因：M4 训练链挂在总管会话的进程树下，
#  上一个总管会话上下文占满被杀时，把它一起带走了（见 docs/HANDOFF.md 2 节）。
#  重启后用的还是同一机制 => 同一个事故还会再犯一次。
#
#  设计要点：**零损失**——不杀正在跑的训练，只在"训练进程真的消失了
#  且还有没跑完的 run"时接手补齐。
#
#  !! 本文件刻意不含任何非 ASCII 字符 !!
#  PowerShell 5.1 读无 BOM 的 UTF-8 .ps1 时按 ANSI 解码，中文路径会被读成乱码
#  => 所有路径一律从 $PSScriptRoot 推导出来，不写字面量。改动时请保持这一点。
#
#  启动方式（必须 Start-Process 才能脱离会话）：
#    powershell -NoProfile -Command "Start-Process -FilePath 'powershell.exe' `
#      -ArgumentList '-NoProfile','-File','<本文件的绝对路径>' -WindowStyle Hidden"
#  执行策略：本机 CurrentUser = RemoteSigned，本地脚本直接可跑，**不需要** Bypass。
# =====================================================================

$ErrorActionPreference = 'Continue'

# ---- 路径全部推导，不写中文 ----
$repo = Split-Path -Parent $PSScriptRoot          # 本文件在 <repo>\tools\ 下
$py   = 'D:\anaconda3\envs\pytorch_env\python.exe'
$trainScript = 'task_extension6_deep_robust_train.py'   # 相对名，配合 -WorkingDirectory
$outRoot     = Join-Path $repo 'outputs_cifar100\extension6_deep_robust'
$log         = Join-Path $repo 'logs_v7_m4_watchdog.log'

# 本次要跑的 run（1/4、2/4 的 vgg11 已完成；余下这两个）
$runs = @(
    @{ seed = 43; dir = 'resnet18_exp3_s43' },
    @{ seed = 44; dir = 'resnet18_exp3_s44' }
)

function Write-Log($msg) {
    $line = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '  ' + $msg
    try   { Add-Content -Path $log -Value $line -Encoding utf8 -ErrorAction Stop }
    catch { }
}

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*task_extension6_deep_robust_train.py*' }
}

Write-Log ('=== watchdog started (pid ' + $PID + ') ; repo=' + $repo + ' ===')

# 连续缺席计数：会话链是 `run3 && run4`，两条命令之间有一瞬间没有 python 进程。
# 若只看 1 次采样，watchdog 可能趁那一秒插进去启动 run4，
# 而会话链同时也会启动 run4 => 两个训练同时跑 = 红线 1（GPU 串行）。
# 链路切换的间隙 <1 秒，5 分钟的阈值远大于它 => 不会误判。
$lastState   = ''
$absentCount = 0
$ABSENT_NEEDED = 5

while ($true) {
    # (1) 全部跑完 -> 退出
    $missing = @($runs | Where-Object { -not (Test-Path (Join-Path $outRoot ($_.dir + '\metrics.json'))) })
    if ($missing.Count -eq 0) {
        Write-Log 'all runs have metrics.json; watchdog exits normally'
        break
    }

    # (2) 有训练在跑 -> 不干预
    $alive = @(Get-TrainingProcess)
    if ($alive.Count -gt 0) {
        $absentCount = 0
        $state = 'waiting (training pid ' + $alive[0].ProcessId + ' alive; ' + $missing.Count + ' run(s) left)'
        if ($state -ne $lastState) { Write-Log $state; $lastState = $state }
        Start-Sleep -Seconds 60
        continue
    }

    # (3) 没看到训练进程 -> 先等够 5 分钟再动手
    $absentCount++
    if ($absentCount -lt $ABSENT_NEEDED) {
        Write-Log ('no training process seen (' + $absentCount + '/' + $ABSENT_NEEDED + ') - keep watching')
        Start-Sleep -Seconds 60
        continue
    }

    # (4) 连续缺席确认 -> 接手补齐
    Write-Log ('!! training gone for ' + $ABSENT_NEEDED + ' consecutive checks; ' + $missing.Count + ' run(s) left => taking over')
    $absentCount = 0
    Start-Sleep -Seconds 30

    foreach ($r in $runs) {
        $mf = Join-Path $outRoot ($r.dir + '\metrics.json')
        if (Test-Path $mf) {
            Write-Log ('  skip seed ' + $r.seed + ' (metrics.json exists)')
            continue
        }
        Write-Log ('  launching seed ' + $r.seed + ' -> ' + $r.dir)
        $argList = @(
            $trainScript,
            '--dataset', 'cifar100',
            '--model',   'resnet18',
            '--variant', 'exp3',
            '--epochs',  '200',
            '--seed',    "$($r.seed)",
            '--tag',     "_s$($r.seed)"
        )
        $env:PYTHONIOENCODING = 'utf-8'
        # !! 必须重定向输出 !! （2026-09-24 实修）
        # 首版漏了这一步，后果：16:37 那次真实接手里，重启的训练输出全部丢失，
        # `logs_v7_m4_train_part2.log` 停在崩溃前的内容 —— 监控者会以为训练没在跑。
        # 训练本身没受影响（GPU 96%、best_model.pth 在更新），但可见性没了。
        $runLog = Join-Path $repo ('logs_v7_m4_restart_s' + $r.seed + '.log')
        $runErr = Join-Path $repo ('logs_v7_m4_restart_s' + $r.seed + '.err')
        # 用 Start-Process 而不是 & 调用：让训练不在 watchdog 的进程树里，
        # 这样 watchdog 万一被杀，训练也活着（这正是本脚本存在的理由）
        try {
            $spArgs = @{
                FilePath               = $py
                ArgumentList           = $argList
                WorkingDirectory       = $repo
                WindowStyle            = 'Hidden'
                Wait                   = $true
                ErrorAction            = 'Stop'
                RedirectStandardOutput = $runLog
                RedirectStandardError  = $runErr
            }
            Start-Process @spArgs
            Write-Log ('  seed ' + $r.seed + ' finished (stdout -> ' + $runLog + ')')
        } catch {
            Write-Log ('  !! launch failed for seed ' + $r.seed + ': ' + $_.Exception.Message)
        }
    }
    Write-Log 'catch-up chain done; back to watching'
    Start-Sleep -Seconds 60
}
Write-Log '=== watchdog exited ==='
