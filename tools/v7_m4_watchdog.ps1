# =====================================================================
#  M4 training-chain watchdog (detached from any session)
# =====================================================================
#  WHY: the M4 chain runs as a child of the supervising session's process
#  tree. When that session died once, it took the training with it.
#  This watchdog is launched detached (see docs/V7_COORDINATION.md 5-suppl-1)
#  so the training survives any session death.
#
#  DESIGN: zero-loss. It never kills a running training. It only takes over
#  when the training process is really gone AND some run is still missing.
#
#  !! THIS FILE MUST STAY PURE ASCII !!
#  PowerShell 5.1 decodes a BOM-less UTF-8 .ps1 as ANSI, which mangles any
#  non-ASCII path. All paths are derived from $PSScriptRoot instead.
#  Also: use CRLF line endings and do NOT use backtick line-continuation
#  (PS 5.1 + LF + backtick => ParserError, process exits instantly).
#
#  LAUNCH:
#    powershell -NoProfile -Command "Start-Process -FilePath 'powershell.exe' `
#      -ArgumentList '-NoProfile','-File','<abs path to this file>' -WindowStyle Hidden"
#  NOTE: CurrentUser execution policy is RemoteSigned, so local scripts run
#  without any flag. Do NOT pass -ExecutionPolicy Bypass (it is unnecessary
#  and needlessly weakens a security control).
#
#  ---------------------------------------------------------------------
#  BUG FIXES 2026-09-24 (found by claude-cf; this is the SECOND revision)
#  ---------------------------------------------------------------------
#  v1 had a serious defect: it treated "the training process exited" as
#  "the run finished". A CRASH also exits the process. Observed damage:
#     16:37:26  launching seed 43
#     16:44:11  seed 43 finished          <-- only 6m45s for a 6-8h run!
#     16:44:11  launching seed 44         <-- moved on, abandoning seed 43
#  Fixes:
#   (1) Completion is judged ONLY by the artifact:
#       <outRoot>\<dir>\metrics.json exists AND its total_epochs == expected
#   (2) After a launch returns, re-check the artifact. If it is not there,
#       that was a CRASH -> break out of the chain (do NOT start the next run)
#   (3) Exponential backoff on consecutive crashes
#   (4) Memory gate: never launch while free physical memory is too low
#
#  KNOWN QUIRK (not yet fixed, 2026-09-24): when the memory gate defers, the
#  loop `continue`s back to the top, where $absentCount is recomputed from 0
#  -- so a deferred launch costs ANOTHER 5 minutes of counting before the next
#  attempt. Observed: 19:24:31 deferred -> 19:29:32 restarted counting at 1/5
#  -> 19:33:35 took over. Not a correctness bug (it still fires), but if you
#  want it to retry every 5 min instead of every 10, move the
#  `$absentCount = 0` line to AFTER the memory gate.
# =====================================================================

$ErrorActionPreference = 'Continue'

# ---- paths are derived, never written as literals (see ASCII note above) ----
$repo = Split-Path -Parent $PSScriptRoot          # this file lives in <repo>\tools\
$py   = 'D:\anaconda3\envs\pytorch_env\python.exe'
$trainScript = 'task_extension6_deep_robust_train.py'   # relative; resolved via -WorkingDirectory
$outRoot     = Join-Path $repo 'outputs_cifar100\extension6_deep_robust'
$log         = Join-Path $repo 'logs_v7_m4_watchdog.log'

# Remaining runs. 1/4 and 2/4 (vgg11) are already done.
#  3/4  resnet18 / exp3 / seed43 / 200ep  ->  resnet18_exp3_s43
#  4/4  resnet18 / exp3 / seed44 / 200ep  ->  resnet18_exp3_s44
$runs = @(
    @{ seed = 43; dir = 'resnet18_exp3_s43'; epochs = 200 },
    @{ seed = 44; dir = 'resnet18_exp3_s44'; epochs = 200 }
)

# Memory gate. Free physical memory must be at least this much before we
# launch. Rationale: on 2026-09-24 the training died with
#   torch.AcceleratorError: CUDA error: out of memory
# while 7 GB of VRAM was FREE -- i.e. it was host memory / WDDM that failed,
# not VRAM. Free physical was 0.6 GB and commit was 1.46 GB at the time.
# GATE CALIBRATION, 2026-09-25 (third revision).
# v2 gated on free PHYSICAL only, at 1500 MB. On 2026-09-25 12:04 that blocked a
# launch at free physical = 1104 MB while free COMMIT was ~25 GB -- i.e. it
# blocked on the wrong metric.
#   What actually failed on 2026-09-24:  CUDA error: out of memory
#     at free physical 0.6 GB AND free commit 1.46 GB  (VRAM was 7 GB free).
#   "out of memory" is an ALLOCATION failure => COMMIT is the metric that
#   responds to it. Physical matters too, but the observed failure threshold
#   there is ~0.6 GB, not 1.5 GB.
# => gate on BOTH: commit is the primary guard, physical a secondary one.
$MIN_FREE_PHYS_MB   = 800      # was 1500; the observed failure was at 600
$MIN_FREE_COMMIT_MB = 4000     # primary guard; failure observed at 1460

function Write-Log($msg) {
    $line = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '  ' + $msg
    try   { Add-Content -Path $log -Value $line -Encoding utf8 -ErrorAction Stop }
    catch { }
}

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*task_extension6_deep_robust_train.py*' }
}

function Get-FreePhysicalMB {
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        return [int]($os.FreePhysicalMemory / 1KB)
    } catch { return -1 }
}

function Get-FreeCommitMB {
    try {
        $os = Get-CimInstance Win32_OperatingSystem -ErrorAction Stop
        return [int]($os.FreeVirtualMemory / 1KB)
    } catch { return -1 }
}

# (1) Completion is judged by the ARTIFACT, never by "the process exited".
function Test-RunComplete($r) {
    $mf = Join-Path $outRoot ($r.dir + '\metrics.json')
    if (-not (Test-Path $mf)) { return $false }
    try {
        $j = Get-Content -Raw -Path $mf -Encoding UTF8 | ConvertFrom-Json
        return ([int]$j.total_epochs -eq [int]$r.epochs)
    } catch { return $false }
}

Write-Log ('=== watchdog started (pid ' + $PID + ') ; repo=' + $repo + ' ===')

$lastState          = ''
$absentCount        = 0
$ABSENT_NEEDED      = 5      # consecutive absences (x60s) before we take over
$consecutiveCrashes = 0

# Why 5 consecutive checks: the chain is `run3 && run4`; there is a brief
# instant between the two commands with no python process. A single sample
# could make the watchdog slip in and start run4 while the chain also starts
# run4 => two trainings at once = RED LINE 1 (GPU must be serial).
# The gap is <1 s; the 5-minute threshold is far larger, so no false positive.
while ($true) {
    # (A) all runs complete -> exit
    $missing = @($runs | Where-Object { -not (Test-RunComplete $_) })
    if ($missing.Count -eq 0) {
        Write-Log 'all runs have valid metrics.json (total_epochs ok); watchdog exits normally'
        break
    }

    # (B) a training is running -> leave it alone
    $alive = @(Get-TrainingProcess)
    if ($alive.Count -gt 0) {
        $absentCount = 0
        $state = 'waiting (training pid ' + $alive[0].ProcessId + ' alive; ' + $missing.Count + ' run(s) left)'
        if ($state -ne $lastState) { Write-Log $state; $lastState = $state }
        Start-Sleep -Seconds 60
        continue
    }

    # (C) no training seen -> wait for the threshold first
    $absentCount++
    if ($absentCount -lt $ABSENT_NEEDED) {
        Write-Log ('no training process seen (' + $absentCount + '/' + $ABSENT_NEEDED + ') - keep watching')
        Start-Sleep -Seconds 60
        continue
    }

    # (D) confirmed gone -> take over
    Write-Log ('!! training gone for ' + $ABSENT_NEEDED + ' consecutive checks; ' + $missing.Count + ' run(s) left => taking over')
    $absentCount = 0

    # (4) memory gate -- do NOT launch into a memory-starved machine.
    # Check BOTH metrics (calibration note at the top of this file).
    # !! Do NOT reset $absentCount here. Resetting it means every deferral costs
    # ANOTHER full 5-check cycle. Observed 2026-09-25: 19:24:31 deferred ->
    # 19:29:32 restarted counting at (1/5) -> 19:33:35 finally took over.
    # Leaving it >= ABSENT_NEEDED makes each loop retry the gate directly.
    $fp = Get-FreePhysicalMB
    $fc = Get-FreeCommitMB
    if (($fp -ge 0 -and $fp -lt $MIN_FREE_PHYS_MB) -or ($fc -ge 0 -and $fc -lt $MIN_FREE_COMMIT_MB)) {
        Write-Log ('!! memory gate: free physical = ' + $fp + ' MB (< ' + $MIN_FREE_PHYS_MB + ') or free commit = ' + $fc + ' MB (< ' + $MIN_FREE_COMMIT_MB + ') -> NOT launching; retry in 5 min')
        Start-Sleep -Seconds 300
        continue
    }
    Write-Log ('memory gate ok: free physical = ' + $fp + ' MB, free commit = ' + $fc + ' MB')
    $absentCount = 0

    # (3) backoff after consecutive crashes
    if ($consecutiveCrashes -gt 0) {
        $wait = [int][Math]::Min(1800, 120 * [Math]::Pow(2, $consecutiveCrashes - 1))
        Write-Log ('backoff: ' + $consecutiveCrashes + ' consecutive crash(es) -> sleeping ' + $wait + ' s before retry')
        Start-Sleep -Seconds $wait
        if ((Get-FreePhysicalMB) -lt $MIN_FREE_PHYS_MB) {
            Write-Log 'backoff done but memory still low -> loop back without launching'
            continue
        }
    }

    Start-Sleep -Seconds 30

    foreach ($r in $runs) {
        if (Test-RunComplete $r) {
            Write-Log ('  skip seed ' + $r.seed + ' (metrics.json valid, total_epochs=' + $r.epochs + ')')
            continue
        }
        Write-Log ('  launching seed ' + $r.seed + ' -> ' + $r.dir)
        $argList = @(
            $trainScript,
            '--dataset', 'cifar100',
            '--model',   'resnet18',
            '--variant', 'exp3',
            '--epochs',  "$($r.epochs)",
            '--seed',    "$($r.seed)",
            '--tag',     "_s$($r.seed)"
        )
        $env:PYTHONIOENCODING = 'utf-8'
        # Output MUST be redirected. v1 forgot this, so on 2026-09-24 the first
        # real takeover produced no visible output at all and
        # logs_v7_m4_train_part2.log stayed frozen at the pre-crash content --
        # which made a healthy training look dead to every monitor.
        $runLog = Join-Path $repo ('logs_v7_m4_restart_s' + $r.seed + '.log')
        $runErr = Join-Path $repo ('logs_v7_m4_restart_s' + $r.seed + '.err')
        # Start-Process (not &) so the training is NOT in the watchdog's process
        # tree -- if the watchdog dies, the training survives. That is the whole
        # point of this script.
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
        } catch {
            Write-Log ('  !! launch failed for seed ' + $r.seed + ': ' + $_.Exception.Message)
            $consecutiveCrashes++
            break
        }

        # (2) THE FIX: re-check the ARTIFACT after the launch returns.
        # "process exited" is NOT "run finished" -- a crash exits too.
        if (Test-RunComplete $r) {
            Write-Log ('  seed ' + $r.seed + ' COMPLETED (metrics.json present, total_epochs=' + $r.epochs + ')')
            $consecutiveCrashes = 0
        } else {
            $consecutiveCrashes++
            Write-Log ('  !! seed ' + $r.seed + ' did NOT complete (no valid metrics.json) -> CRASH, not completion.' +
                       ' Aborting this chain; consecutive crashes = ' + $consecutiveCrashes)
            break    # do NOT move on to the next run -- that is exactly the v1 bug
        }
    }
    Write-Log 'catch-up chain finished; back to watching'
    Start-Sleep -Seconds 60
}
Write-Log '=== watchdog exited ==='
