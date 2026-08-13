[CmdletBinding()]
param(
    [string]$RunName = "long-001",
    [int]$Stages = 10,
    [int]$MirrorPerStage = 100,
    [int]$Ai30PerStage = 100,
    [int]$EvalMatches = 50,
    [int]$EvalMediumMatches = 200,
    [int]$EvalFinalMatches = 500,
    [double]$EvalMediumWinRate = 0.40,
    [double]$EvalFinalWinRate = 0.55,
    [int]$EvalWorkers = 64,
    [int]$Workers = 64,
    [int]$GroupSize = 32,
    [int]$Steps = 256,
    [int]$MaxSteps = 4500,
    [int]$MinibatchSize = 2048,
    [int]$EnvGoMaxProcs = 1,
    [string]$Device = "cuda",
    [double]$StopWinRate = 0.60,
    [double]$StopCiLow = 0.50,
    [string]$Resume = ".\ai40\checkpoints\mixed-100\latest.pt",
    [switch]$FromScratch,
    [switch]$NoPipeline,
    [switch]$SkipBuild,
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot "ai40\.venv\Scripts\python.exe"
$environment = Join-Path $PSScriptRoot "assaultenv.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}
if (-not $SkipBuild) {
    & go build -o $environment ./cmd/assaultenv
    if ($LASTEXITCODE -ne 0) { throw "Failed to build assaultenv.exe" }
}
if (-not (Test-Path -LiteralPath $environment -PathType Leaf)) {
    throw "Training environment not found: $environment"
}
if ($FromScratch) {
    $Resume = ""
}

$output = Join-Path $PSScriptRoot "ai40\checkpoints\$RunName"
$arguments = @(
    "-m", "tanat_ai40.train_campaign",
    "--env", $environment,
    "--output", $output,
    "--stages", $Stages,
    "--mirror-per-stage", $MirrorPerStage,
    "--ai30-per-stage", $Ai30PerStage,
    "--eval-matches", $EvalMatches,
    "--eval-medium-matches", $EvalMediumMatches,
    "--eval-final-matches", $EvalFinalMatches,
    "--eval-medium-win-rate", $EvalMediumWinRate.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--eval-final-win-rate", $EvalFinalWinRate.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--eval-workers", $EvalWorkers,
    "--workers", $Workers,
    "--group-size", $GroupSize,
    "--steps", $Steps,
    "--max-steps", $MaxSteps,
    "--minibatch-size", $MinibatchSize,
    "--env-gomaxprocs", $EnvGoMaxProcs,
    "--device", $Device,
    "--stop-win-rate", $StopWinRate.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--stop-ci-low", $StopCiLow.ToString([Globalization.CultureInfo]::InvariantCulture)
)
if ($Resume) {
    if (-not (Test-Path -LiteralPath $Resume -PathType Leaf)) {
        throw "Resume checkpoint not found: $Resume"
    }
    $arguments += @("--resume", (Resolve-Path -LiteralPath $Resume).Path)
}
if ($ExtraArgs) {
    $arguments += $ExtraArgs
}
if ($NoPipeline) {
    $arguments += "--no-pipeline"
}

Write-Host "AI-40 campaign: $output"
if ($Resume) {
    Write-Host "Initial checkpoint: $((Resolve-Path -LiteralPath $Resume).Path)"
} else {
    Write-Host "Initial checkpoint: random weights (from scratch)"
}
Write-Host "Training: workers=$Workers groups=$GroupSize rollout=$Steps max_steps=$MaxSteps minibatch=$MinibatchSize device=$Device"
Write-Host "Actor-learner pipeline: $(-not $NoPipeline)"
Write-Host "Evaluation: $EvalMatches -> $EvalMediumMatches at $EvalMediumWinRate -> $EvalFinalMatches at $EvalFinalWinRate; workers=$EvalWorkers"
Write-Host "Stop target: winrate=$StopWinRate CI-low=$StopCiLow"
Write-Host "Press Ctrl+C to stop. Run the same command again to resume."
if ($DryRun) {
    Write-Host "Dry run: no training process started."
    Write-Host ($python + " " + ($arguments -join " "))
    exit 0
}
& $python @arguments
exit $LASTEXITCODE
