# =====================================================================
#  M4 readout runner (detached from any session)
# =====================================================================
#  WHY: the M4 training chain finishes by itself -- watchdog v3 exits once
#  every run has a valid metrics.json. Four readout runs must then follow
#  on the same GPU, serially. If nobody is at the keyboard when training
#  ends, the GPU sits idle for hours.
#
#  DESIGN: gate first, never guess.
#   (1) GATE. A backbone's readout is allowed ONLY if that backbone's
#       metrics.json exists AND its total_epochs equals the expected value
#       (vgg11/exp2 = 120, resnet18/exp3 = 200).
#       Reason -- the 2026-09-24 incident: a crash left behind a
#       best_model.pth from a 6m45s run. torch's load_state_dict accepts
#       it silently (shapes match), so a readout on it produces a
#       clean-looking but WRONG number. That is the silent failure this
#       project fears most.
#   (2) Never overwrite. The four output CSVs use bs43/bs44 names; they
#       were checked absent on 2026-09-25. Existing files are skipped.
#   (3) No judging. This script stops after the pairing check. Computing
#       the backbone-level std and deciding branch (1)/(2)/(3) is a human
#       step (docs/V7_M4_RUNBOOK.md section 3); branch (3) means STOP and
#       report to the user.
#
#  SOURCE OF THE COMMANDS: docs/V7_M4_RUNBOOK.md section 2 and
#  outputs_cifar100/v7_M4_audit/M4_EXECUTION_PLAN.md section 1.
#  Parameters re-verified against v3_readout_repair.py on 2026-09-25:
#    --arch accepts robust_vgg11 / robust_resnet18          (lines 58-61)
#    --ckpt / --backbone_tag / --epochs_nat / --n_train     (lines 217-230)
#    output name = readout_repair_{ds}_{bb}_alpha{+.2f}_{tag}.csv (line 295)
#    tag = n{n_train}, with _s{seed} appended only when seed != 42 (line 293)
#    (-seed 42 is deliberate: this experiment measures the BACKBONE's
#     variance, so the readout side is held fixed.)
#
#  !! THIS FILE MUST STAY PURE ASCII, WITH CRLF LINE ENDINGS !!
#  PowerShell 5.1 decodes a BOM-less UTF-8 .ps1 as ANSI, which mangles any
#  non-ASCII path. Paths are derived from $PSScriptRoot instead. Do NOT use
#  backtick line-continuation (PS 5.1 + LF + backtick => ParserError).
#
#  LAUNCH (do NOT pass -ExecutionPolicy Bypass; CurrentUser is RemoteSigned):
#    powershell -NoProfile -Command "Start-Process -FilePath 'powershell.exe' `
#      -ArgumentList '-NoProfile','-File','<abs path to this file>' -WindowStyle Hidden"
# =====================================================================

$ErrorActionPreference = 'Continue'

$repo   = Split-Path -Parent $PSScriptRoot
$py     = 'D:\anaconda3\envs\pytorch_env\python.exe'
$outDir = Join-Path $repo 'outputs_cifar100\v3_readout_repair'
$log    = Join-Path $repo 'logs_v7_m4_readout.log'

# Memory gate, same calibration as v7_m4_watchdog.ps1: "out of memory" is
# an ALLOCATION failure, so free COMMIT is the primary guard; the observed
# physical failure threshold was ~0.6 GB, so 800 MB is the secondary one.
$MIN_FREE_PHYS_MB   = 800
$MIN_FREE_COMMIT_MB = 4000
$WAIT_MAX_MINUTES   = 480

# The four backbones to re-read. mdir/ep feed the gate; ckpt/tag feed the
# readout command. Each hashtable stays on ONE line: no backtick continuation.
$runs = @(
    @{ arch = 'robust_vgg11';    ckpt = 'checkpoints_cifar100\Exp2_Calib+Layerwise_vgg11_s43\best_model.pth'; tag = 'exp2_vgg11_bs43_ep24';    mdir = 'outputs_cifar100\extension6_deep_robust\vgg11_s43';        ep = 120 },
    @{ arch = 'robust_vgg11';    ckpt = 'checkpoints_cifar100\Exp2_Calib+Layerwise_vgg11_s44\best_model.pth'; tag = 'exp2_vgg11_bs44_ep24';    mdir = 'outputs_cifar100\extension6_deep_robust\vgg11_s44';        ep = 120 },
    @{ arch = 'robust_resnet18'; ckpt = 'checkpoints_cifar100\Exp3_FullRobust_resnet18_s43\best_model.pth';   tag = 'exp3_resnet18_bs43_ep24'; mdir = 'outputs_cifar100\extension6_deep_robust\resnet18_exp3_s43'; ep = 200 },
    @{ arch = 'robust_resnet18'; ckpt = 'checkpoints_cifar100\Exp3_FullRobust_resnet18_s44\best_model.pth';   tag = 'exp3_resnet18_bs44_ep24'; mdir = 'outputs_cifar100\extension6_deep_robust\resnet18_exp3_s44'; ep = 200 }
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

function Get-FreePhysicalMB {
    try { return [int]((Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).FreePhysicalMemory / 1KB) }
    catch { return -1 }
}

function Get-FreeCommitMB {
    try { return [int]((Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).FreeVirtualMemory / 1KB) }
    catch { return -1 }
}

# The M4 gate: completeness is judged by the artifact, never by "the
# process exited" (a crash exits too).
function Test-BackboneReady($r) {
    $mf = Join-Path $repo ($r.mdir + '\metrics.json')
    if (-not (Test-Path $mf)) { return $false }
    try {
        $j = Get-Content -Raw -Path $mf -Encoding UTF8 | ConvertFrom-Json
        return ([int]$j.total_epochs -eq [int]$r.ep)
    } catch { return $false }
}

function Get-MetricsAcc($r) {
    $mf = Join-Path $repo ($r.mdir + '\metrics.json')
    $j = Get-Content -Raw -Path $mf -Encoding UTF8 | ConvertFrom-Json
    return [double]$j.best_test_acc
}

# Pairing gate: the frozen original FC head cannot be moved by the readout
# seed, so a_original_fc at alpha=0 must reproduce the backbone's own
# best_test_acc. A mismatch means the readout was wired to the WRONG
# backbone. Tolerance 0.01 = 1 test image out of 10000, a known
# cross-device artifact (exp3_resnet18_s42: csv 72.69 vs metrics 72.70).
function Get-CsvOriginalFc($csv) {
    $row = Import-Csv -Path $csv | Where-Object { $_.readout -eq 'a_original_fc' }
    if ($null -eq $row) { return $null }
    return [double]$row.'0.0'
}

Write-Log ('=== M4 readout runner started (pid ' + $PID + ') ===')

# ---- phase 1: wait until every backbone has a valid metrics.json ----
$waited = 0
while ($true) {
    $notReady = @($runs | Where-Object { -not (Test-BackboneReady $_) })
    if ($notReady.Count -eq 0) { break }
    if ($waited -ge $WAIT_MAX_MINUTES) {
        $names = ($notReady | ForEach-Object { $_.tag }) -join ', '
        Write-Log ('!! gate timeout after ' + $WAIT_MAX_MINUTES + ' min; still not ready: ' + $names + ' -> NOT running readouts')
        Write-Log '=== readout runner exited (gate timeout) ==='
        exit 1
    }
    if ($waited % 10 -eq 0) {
        Write-Log ('waiting on gate (' + $waited + ' min); not ready: ' + (($notReady | ForEach-Object { $_.tag }) -join ', '))
    }
    Start-Sleep -Seconds 60
    $waited++
}

Write-Log 'gate satisfied: all four backbones have metrics.json with the expected total_epochs'

# Let the training watchdog notice completion and exit before we touch the GPU.
Start-Sleep -Seconds 60
$alive = @(Get-TrainingProcess)
if ($alive.Count -gt 0) {
    Write-Log ('!! a training process is still alive (pid ' + $alive[0].ProcessId + ') after the gate passed -> NOT running readouts; a human must look at this')
    Write-Log '=== readout runner exited (training still alive) ==='
    exit 1
}
Write-Log 'no training process alive; GPU is free'

# ---- phase 2: the four readouts, strictly serial ----
$i = 0
foreach ($r in $runs) {
    $i++
    # +0.30 in the name comes from "{0:+.2f}" formatting in the source script.
    $csv = Join-Path $outDir ('readout_repair_c100_' + $r.tag + '_alpha+0.30_n20000.csv')

    if (Test-Path $csv) {
        Write-Log ('  skip ' + $r.tag + ' (csv already exists: ' + $csv + ')')
        continue
    }

    $fp = Get-FreePhysicalMB
    $fc = Get-FreeCommitMB
    if (($fp -ge 0 -and $fp -lt $MIN_FREE_PHYS_MB) -or ($fc -ge 0 -and $fc -lt $MIN_FREE_COMMIT_MB)) {
        Write-Log ('!! memory gate before ' + $r.tag + ': free physical = ' + $fp + ' MB, free commit = ' + $fc + ' MB -> NOT launching; a human must look at this')
        Write-Log '=== readout runner exited (memory gate) ==='
        exit 1
    }

    $runLog = Join-Path $repo ('logs_v7_m4_readout_' + $r.tag + '.log')
    $runErr = Join-Path $repo ('logs_v7_m4_readout_' + $r.tag + '.err')
    Write-Log ('launching readout ' + $i + '/4: ' + $r.tag + ' (' + $r.arch + ')')
    $argList = @(
        'v3_readout_repair.py',
        '--arch', $r.arch,
        '--ckpt', $r.ckpt,
        '--backbone_tag', $r.tag,
        '--dataset', 'cifar100',
        '--alpha', '0.3',
        '--epochs_nat', '24',
        '--n_train', '20000',
        '--seed', '42'
    )
    $env:PYTHONIOENCODING = 'utf-8'
    try {
        Start-Process -FilePath $py -ArgumentList $argList -WorkingDirectory $repo -WindowStyle Hidden -Wait -ErrorAction Stop -RedirectStandardOutput $runLog -RedirectStandardError $runErr
    } catch {
        Write-Log ('  !! launch failed for ' + $r.tag + ': ' + $_.Exception.Message)
        Write-Log '=== readout runner exited (launch failure) ==='
        exit 1
    }

    if (-not (Test-Path $csv)) {
        Write-Log ('  !! readout ' + $i + '/4 produced no csv (' + $csv + ') -> CRASH, not completion. Stopping here.')
        Write-Log '=== readout runner exited (readout crash) ==='
        exit 1
    }
    Write-Log ('  readout ' + $i + '/4 COMPLETED: ' + (Split-Path -Leaf $csv))

    $got  = Get-CsvOriginalFc $csv
    $want = Get-MetricsAcc $r
    if ($null -eq $got) {
        Write-Log ('  !! pairing check could not read a_original_fc from ' + (Split-Path -Leaf $csv) + ' -> stopping.')
        Write-Log '=== readout runner exited (pairing unreadable) ==='
        exit 1
    }
    # Tolerance is expressed in TEST IMAGES, not in percentage points.
    # The runbook's operating line: a 0.01 offset (= 1 image out of 10000) is
    # a KNOWN cross-device phenomenon and passes; >= 0.02 means stop.
    # Comparing floats against 0.01 directly is unreliable -- observed
    # 2026-09-25 on the very first live run:
    #   73.09 - 73.08 = 0.010000000000005116  (> 0.01  => FALSE FAIL)
    #   72.79 - 72.80 = 0.0099999999999909    (< 0.01  => lucky PASS)
    # Both are the same 1-image phenomenon; only the rounding direction
    # differed. Rounding the difference to whole images removes the tie.
    $diffImages = [int][math]::Round([math]::Abs($got - $want) * 100)
    if ($diffImages -le 1) {
        Write-Log ('  pairing PASS: a_original_fc@0.0 = ' + $got + ' vs metrics best_test_acc = ' + $want + ' (diff = ' + $diffImages + ' test image(s); <= 1 is the known offset)')
    } else {
        Write-Log ('  !! pairing FAIL: a_original_fc@0.0 = ' + $got + ' vs metrics best_test_acc = ' + $want + ' (diff = ' + $diffImages + ' test images > 1) -> WRONG BACKBONE? Stopping; do NOT trust this csv.')
        Write-Log '=== readout runner exited (pairing fail) ==='
        exit 1
    }
    Start-Sleep -Seconds 20
}

Write-Log 'all four readouts done and pairing-checked.'
Write-Log 'HUMAN STEP NEXT: compute the backbone-level std (population, ddof=0) for clean / alpha=+0.3 / drop@0.3,'
Write-Log '  main criterion on the e_readout_NAT row, also report the a_original_fc row, take the conservative'
Write-Log '  direction if they disagree, then decide branch 1 / 2 / 3 (docs/V7_M4_RUNBOOK.md section 3).'
Write-Log '  Branch 3 (>= 3 pp) means STOP and report to the user -- do not continue.'
Write-Log '=== readout runner exited normally ==='
