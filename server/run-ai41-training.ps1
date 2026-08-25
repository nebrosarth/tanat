$hadPreviousVariant = Test-Path -LiteralPath Env:TANAT_TRAINING_VARIANT
$previousVariant = $env:TANAT_TRAINING_VARIANT
$hadPreviousNoPause = Test-Path -LiteralPath Env:TANAT_TRAINING_NO_PAUSE
$previousNoPause = $env:TANAT_TRAINING_NO_PAUSE
$hadPreviousWrapped = Test-Path -LiteralPath Env:TANAT_TRAINING_WRAPPED
$previousWrapped = $env:TANAT_TRAINING_WRAPPED
$noPauseRequested = $args -contains "-NoPause"
$exitCode = 1

try {
    $env:TANAT_TRAINING_VARIANT = "ai41"
    # The shared launcher normally pauses on completion. AI-41 owns that pause
    # so invoking this wrapper does not require pressing Enter twice.
    $env:TANAT_TRAINING_NO_PAUSE = "1"
    $env:TANAT_TRAINING_WRAPPED = "1"
    $exitCode = & (Join-Path $PSScriptRoot "run-ai40-training.ps1") @args
} catch {
    Write-Host "AI-41 training launcher failed: $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
} finally {
    if ($hadPreviousVariant) {
        $env:TANAT_TRAINING_VARIANT = $previousVariant
    } else {
        Remove-Item -LiteralPath Env:TANAT_TRAINING_VARIANT -ErrorAction SilentlyContinue
    }
    if ($hadPreviousNoPause) {
        $env:TANAT_TRAINING_NO_PAUSE = $previousNoPause
    } else {
        Remove-Item -LiteralPath Env:TANAT_TRAINING_NO_PAUSE -ErrorAction SilentlyContinue
    }
    if ($hadPreviousWrapped) {
        $env:TANAT_TRAINING_WRAPPED = $previousWrapped
    } else {
        Remove-Item -LiteralPath Env:TANAT_TRAINING_WRAPPED -ErrorAction SilentlyContinue
    }
    if (-not $noPauseRequested) {
        Write-Host "AI-41 training finished with exit code $exitCode."
        [void](Read-Host "Press Enter to close this window")
    }
}

exit $exitCode
