<#
    run_loso_eval.ps1 -- full evaluation of a finished LOSO run.

    Usage:
        .\run_loso_eval.ps1
        .\run_loso_eval.ps1 -RunDir outputs\ceb_loso_holdout_elevenlabs_20261001_095507
        .\run_loso_eval.ps1 -Fold voice-conv

    Runs six cells for one checkpoint -- the attack axis crossed with the
    channel axis:

                        clean     Opus      AMR
        matched (seen)      .        .        .
        unseen             .        .        .

    plus the fold's OWN validation split, which supplies the decision
    threshold. A LOSO model is trained on a different distribution from
    the main model, so it cannot borrow the main run's threshold.

    results.json is copied after every single test -- it is overwritten
    by the next one, and that has cost this project two result sets.
#>

param(
    [string]$RunDir  = "",
    [string]$Fold    = "elevenlabs",
    [string]$Configs = "training\configs_vc"
)

if (-not $RunDir) {
    $RunDir = (Get-ChildItem outputs -Directory |
               Where-Object { $_.Name -like "*loso*" } |
               Sort-Object LastWriteTime -Descending |
               Select-Object -First 1).FullName
}

$Ckpt = Join-Path $RunDir "best_model.pth"
$Log  = Join-Path $RunDir "loso_eval.log"
$Pred = Join-Path $RunDir "results\predictions"

function Say($m) {
    $l = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
    Write-Host $l; Add-Content -Path $Log -Value $l
}

if (-not (Test-Path $Ckpt)) { Write-Host "NO CHECKPOINT AT $Ckpt"; exit 1 }

$size = (Get-Item $Ckpt).Length
Say "run        : $RunDir"
Say "fold       : holdout_$Fold   (trained WITHOUT $Fold)"
Say "checkpoint : $size bytes"
if ($size -lt 3000000000) {
    Say "WARNING: checkpoint under 3 GB -- likely a truncated write, expected ~3.79 GB."
}

$ran = @()

function RunTest($name) {
    $cfg = Join-Path $Configs "$name.yaml"
    if (-not (Test-Path $cfg)) { Say "---- $name : CONFIG MISSING, skipped"; return }

    Say "---- $name : starting"
    $src = Join-Path $RunDir "results.json"
    if (Test-Path $src) { Remove-Item $src -Force }

    & python training\run_training.py --test --config $cfg --checkpoint $Ckpt 2>&1 |
        Tee-Object -FilePath $Log -Append

    if (Test-Path $src) {
        Copy-Item $src (Join-Path $RunDir "results_$name.json") -Force
        Say "---- $name : done -> results_$name.json"
        $script:ran += $name
    } else {
        Say "---- $name : FAILED (no results.json produced)"
    }
}

# 1. the fold's own val split -- must run first, it supplies the threshold
RunTest "test_val_loso_$Fold"

# 2-4. both cells, under each channel condition
RunTest "test_holdout_$Fold"
RunTest "test_holdout_${Fold}_opus"
RunTest "test_holdout_${Fold}_amr"

Say "================ SCORING ================"

# the seen system is whichever one was NOT held out
if ($Fold -eq "elevenlabs") { $seen = "voice-conv" } else { $seen = "elevenlabs" }

$cells = @(
    "matched_${seen}",
    "matched_${seen}_opus",
    "matched_${seen}_amr",
    "unseen_${Fold}",
    "unseen_${Fold}_opus",
    "unseen_${Fold}_amr"
)

$evals = @()
foreach ($c in $cells) {
    $f = Join-Path $Pred "${c}_predictions.txt"
    if (Test-Path $f) { $evals += "--evaluate"; $evals += $f }
    else { Say "missing prediction file: ${c}_predictions.txt" }
}

if ($evals.Count -eq 0) { Say "no prediction files found -- nothing to score"; exit 1 }

Say "per-source EER:"
$pargs = @()
foreach ($c in $cells) {
    $f = Join-Path $Pred "${c}_predictions.txt"
    if (Test-Path $f) { $pargs += "--predictions"; $pargs += $f }
}
& python training\per_source_eer.py --splits-dir manifests\splits_vc @pargs 2>&1 |
    Tee-Object -FilePath $Log -Append

$val = Join-Path $Pred "loso_val_predictions.txt"
if (Test-Path $val) {
    Say "calibrated scoring (threshold from this fold's own val split):"
    & python training\score_predictions.py --calibrate $val @evals 2>&1 |
        Tee-Object -FilePath $Log -Append
} else {
    Say "NO val predictions -- falling back to ORACLE thresholds. Do NOT report the accuracy figures."
    & python training\score_predictions.py @evals 2>&1 |
        Tee-Object -FilePath $Log -Append
}

Say "================ DONE ================"
Say ("ran: " + ($ran -join ", "))
Say "Headline is EER(unseen) vs EER(matched) at each channel condition."
Say "Report EER. The matched row is a SEEN channel -- the model was augmented on both codecs."
Say "log: $Log"
