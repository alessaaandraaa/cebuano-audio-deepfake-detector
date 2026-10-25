<#
    run_all_tests.ps1 -- run the whole VC-corpus test suite unattended.

    Usage:
        .\run_all_tests.ps1
        .\run_all_tests.ps1 -RunDir outputs\some_other_run

    Runs every test, copies results.json immediately after each one (it is
    overwritten by the next test -- this has cost this project two result
    sets), then scores everything. Keeps going if one step fails, so a
    single failure does not waste the night. Everything is logged.
#>

param(
    [string]$RunDir = "",
    [string]$Configs = "training\configs_vc"
)

if (-not $RunDir) {
    $RunDir = (Get-ChildItem outputs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}

$Ckpt = Join-Path $RunDir "best_model.pth"
$Log  = Join-Path $RunDir "test_suite.log"
$Pred = Join-Path $RunDir "results\predictions"

function Say($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $Log -Value $line
}

if (-not (Test-Path $Ckpt)) { Write-Host "NO CHECKPOINT AT $Ckpt"; exit 1 }

$size = (Get-Item $Ckpt).Length
Say "run        : $RunDir"
Say "checkpoint : $Ckpt  ($size bytes)"
if ($size -lt 3000000000) {
    Say "WARNING: checkpoint is under 3 GB -- it may be a truncated write. Expected ~3.79 GB."
}

$results = @()

function RunTest($name) {
    Say "---- $name : starting"
    $cfg = Join-Path $Configs "$name.yaml"
    if (-not (Test-Path $cfg)) { Say "---- $name : CONFIG MISSING, skipped"; return }

    $src = Join-Path $RunDir "results.json"
    if (Test-Path $src) { Remove-Item $src -Force }

    & python training\run_training.py --test --config $cfg --checkpoint $Ckpt 2>&1 |
        Tee-Object -FilePath $Log -Append

    if (Test-Path $src) {
        $dst = Join-Path $RunDir "results_$name.json"
        Copy-Item $src $dst -Force
        Say "---- $name : done -> results_$name.json"
        $script:results += $name
    } else {
        Say "---- $name : FAILED (no results.json produced)"
    }
}

# val first -- everything else needs the threshold it produces
RunTest "test_val"
RunTest "test_clean"
RunTest "test_per_source"
RunTest "test_opus"
RunTest "test_amr"

Say "================ SCORING ================"

$preds = @()
foreach ($p in @("clean","test_opus","test_amr","elevenlabs","voice-conv")) {
    $f = Join-Path $Pred "${p}_predictions.txt"
    if (Test-Path $f) { $preds += "--predictions"; $preds += $f }
}

if ($preds.Count -gt 0) {
    Say "per-source EER:"
    & python training\per_source_eer.py --splits-dir manifests\splits_vc @preds 2>&1 |
        Tee-Object -FilePath $Log -Append
} else {
    Say "no prediction files found for per_source_eer"
}

$val = Join-Path $Pred "val_predictions.txt"
$evals = @()
foreach ($p in @("clean","test_opus","test_amr")) {
    $f = Join-Path $Pred "${p}_predictions.txt"
    if (Test-Path $f) { $evals += "--evaluate"; $evals += $f }
}

if ((Test-Path $val) -and $evals.Count -gt 0) {
    Say "calibrated scoring (threshold from val):"
    & python training\score_predictions.py --calibrate $val @evals 2>&1 |
        Tee-Object -FilePath $Log -Append
} else {
    Say "skipping calibration -- val predictions or eval files missing"
}

Say "================ SUMMARY ================"
Say ("completed: " + ($results -join ", "))
Say "full log : $Log"
Say "done."
