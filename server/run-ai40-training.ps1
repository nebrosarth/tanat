[CmdletBinding()]
param(
    [string]$RunName = "long-001",
    [int]$Stages = 1000,
    [int]$MirrorPerStage = 100,
    [int]$Ai30PerStage = 100,
    [int]$HistoricalPerStage = 0,
    [string]$HistoricalAnchor = "",
    [int]$HistoricalPoolSize = 8,
    [int]$EvalMatches = 50,
    [int]$EvalMediumMatches = 200,
    [int]$EvalFinalMatches = 500,
    [double]$EvalMediumWinRate = 0.70,
    [double]$EvalFinalWinRate = 0.99,
    [int]$EvalWorkers = 64,
    [int]$Workers = 64,
    [int]$GroupSize = 64,
    [int]$Steps = 256,
    [int]$MaxSteps = 4500,
    [int]$MinibatchSize = 8192,
    [double]$LearningRate = 0.0003,
    [int]$PpoEpochs = 3,
    [double]$TargetKl = 0.0,
    [int]$EnvGoMaxProcs = 16,
    [string]$Device = "cuda",
    [double]$StopWinRate = 0.99,
    [double]$StopCiLow = 0.80,
    [int]$PromotionEvalMatches = 50,
    [double]$PromotionTolerance = 0.0,
    [double]$PromotionMaxCategoryRegression = 0.05,
    [double]$DiscountHorizonSeconds = 19.8998324946844,
    [double]$GaeHorizonSeconds = 3.26032220809386,
    [string]$Resume = ".\ai40\checkpoints\mixed-100\latest.pt",
    [switch]$FromScratch,
    [switch]$NoPipeline,
    [switch]$NoBf16,
    [switch]$NoCompile,
    [switch]$CompileLearner,
    [switch]$SkipBuild,
    [switch]$DryRun,
    [switch]$NoPause,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$exitCode = 1
try {

$trainingVariant = if ($env:TANAT_TRAINING_VARIANT) {
    $env:TANAT_TRAINING_VARIANT.ToLowerInvariant()
} else {
    "ai40"
}
$campaignModule = "tanat_ai40.train_campaign"
$modelLabel = "AI-40"

switch ($trainingVariant) {
    "ai40" { }
    "ai41" {
        $campaignModule = "tanat_ai40.ai41"
        $modelLabel = "AI-41"
        if (-not $PSBoundParameters.ContainsKey("RunName")) {
            # AI-30's assault policy now clears the local wave before committing
            # to structures. That is a material opponent-distribution change,
            # so train in a new campaign instead of mixing its trajectories with
            # the passive-teacher stages saved in stable-002. A bare invocation
            # resumes this campaign once it has been created.
            $RunName = "ai41-ai30-teacher-001"
        }
        if (-not $PSBoundParameters.ContainsKey("Stages")) {
            # A promotion attempt is meaningful only while it stays close to
            # its bootstrap. Repeatedly rebasing rejected candidates forever
            # hides failure instead of forcing a fresh diagnosis.
            $Stages = 5
        }
        # A single 64-match vector process presents 640 recurrent actor rows
        # per decision to CUDA.  The former 32-row default left the GPU mostly
        # launch-bound even though the Go simulation itself was fast enough.
        if (-not $PSBoundParameters.ContainsKey("EnvGoMaxProcs")) {
            $EnvGoMaxProcs = 16
        }
        if (-not $PSBoundParameters.ContainsKey("Resume")) {
            # First make the neural policy imitate the current aggressive AI-30
            # through a short supervised warm start. PPO then improves it under
            # exactly the same strategic/navigation contract.
            $Resume = Join-Path $PSScriptRoot "ai40\checkpoints\ai41-ai30-skill-clone-bootstrap\latest.pt"
        }
        if (-not $PSBoundParameters.ContainsKey("Steps")) {
            $Steps = 1024
        }
        # Candidate stages always restart from the last promoted checkpoint.
        # Make the updated scripted opponent half of each proposal: it is the
        # current deployment baseline, while mirror and frozen-history games
        # preserve self-play skill and guard against catastrophic forgetting.
        if (-not $PSBoundParameters.ContainsKey("MirrorPerStage")) {
            $MirrorPerStage = 100
        }
        if (-not $PSBoundParameters.ContainsKey("HistoricalAnchor")) {
            $HistoricalAnchor = Join-Path $PSScriptRoot "ai40\checkpoints\ai41-nav-001\checkpoints\stage-005.pt"
        }
        if (-not $PSBoundParameters.ContainsKey("HistoricalPerStage")) {
            $HistoricalPerStage = 100
        }
        if (-not $PSBoundParameters.ContainsKey("Ai30PerStage")) {
            $Ai30PerStage = 200
        }
        if (-not $PSBoundParameters.ContainsKey("LearningRate")) {
            # Warm-starting an existing policy needs adaptation, not a full
            # policy reset. This is deliberately below the generic PPO default
            # but leaves more room than the stalled 5e-5 campaign's effective
            # mixed-opponent schedule.
            $LearningRate = 0.00003
        }
        if (-not $PSBoundParameters.ContainsKey("PpoEpochs")) {
            $PpoEpochs = 1
        }
        if (-not $PSBoundParameters.ContainsKey("TargetKl")) {
            $TargetKl = 0.01
        }
        if (-not $PSBoundParameters.ContainsKey("PromotionMaxCategoryRegression")) {
            # Fifty matched games resolve score changes in two-point increments;
            # allow sampling noise but never accept a materially worse policy
            # against the current AI-30 baseline.
            $PromotionMaxCategoryRegression = 0.03
        }
        if (-not $PSBoundParameters.ContainsKey("DiscountHorizonSeconds")) {
            $DiscountHorizonSeconds = 1200.0
        }
        if (-not $PSBoundParameters.ContainsKey("GaeHorizonSeconds")) {
            $GaeHorizonSeconds = 180.0
        }
    }
    default {
        throw "Unsupported TANAT_TRAINING_VARIANT: $trainingVariant"
    }
}

# A campaign makes stage-level samples comparable by locking its rollout mix
# and PPO update contract in campaign.json.  When the caller did not ask for a
# particular value, resume with that contract instead of replacing it with a
# newer launcher default.  A new -RunName receives the current defaults above.
if ($trainingVariant -eq "ai41") {
    $campaignStatePath = Join-Path $PSScriptRoot "ai40\checkpoints\$RunName\campaign.json"
    if (Test-Path -LiteralPath $campaignStatePath -PathType Leaf) {
        try {
            $campaignState = Get-Content -LiteralPath $campaignStatePath -Raw |
                ConvertFrom-Json -ErrorAction Stop
        } catch {
            throw "Cannot read existing campaign state ${campaignStatePath}: $($_.Exception.Message)"
        }
        $restored = [System.Collections.Generic.List[string]]::new()
        $lockedValues = @(
            @{ Parameter = "MirrorPerStage"; State = "mirror_per_stage"; Type = "int" },
            @{ Parameter = "Ai30PerStage"; State = "ai30_per_stage"; Type = "int" },
            @{ Parameter = "HistoricalPerStage"; State = "historical_per_stage"; Type = "int" },
            @{ Parameter = "HistoricalAnchor"; State = "historical_anchor"; Type = "string" },
            @{ Parameter = "LearningRate"; State = "learning_rate"; Type = "double" },
            @{ Parameter = "PpoEpochs"; State = "ppo_epochs"; Type = "int" },
            @{ Parameter = "TargetKl"; State = "target_kl"; Type = "double" }
        )
        foreach ($locked in $lockedValues) {
            if ($PSBoundParameters.ContainsKey($locked.Parameter)) {
                continue
            }
            $value = $campaignState.($locked.State)
            if ($null -eq $value) {
                continue
            }
            switch ($locked.Type) {
                "int" { Set-Variable -Name $locked.Parameter -Value ([int]$value) }
                "double" { Set-Variable -Name $locked.Parameter -Value ([double]$value) }
                "string" { Set-Variable -Name $locked.Parameter -Value ([string]$value) }
            }
            $restored.Add($locked.Parameter)
        }
        if ($restored.Count -gt 0) {
            Write-Host "Using saved campaign contract: $($restored -join ', ')."
        }
    }
}

$venvDirectory = if ($env:TANAT_AI40_VENV) {
    $env:TANAT_AI40_VENV
} else {
    Join-Path $PSScriptRoot "ai40\.venv"
}
$python = Join-Path $venvDirectory "Scripts\python.exe"
if (-not $env:TANAT_AI40_VENV -and -not (Test-Path -LiteralPath $python -PathType Leaf)) {
    # The Ubuntu setup preserves an existing Windows venv under this name.
    $windowsVenv = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot "ai40") -Directory -Force |
        Where-Object {
            $_.Name -like ".venv.windows.*" -and
            (Test-Path -LiteralPath (Join-Path $_.FullName "Scripts\python.exe") -PathType Leaf)
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($windowsVenv) {
        $venvDirectory = $windowsVenv.FullName
        $python = Join-Path $venvDirectory "Scripts\python.exe"
    }
}
$environment = if ($env:TANAT_ASSAULTENV) {
    $env:TANAT_ASSAULTENV
} else {
    Join-Path $PSScriptRoot "assaultenv.exe"
}
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

$ai41CloneBootstrap = Join-Path $PSScriptRoot "ai40\checkpoints\ai41-ai30-skill-clone-bootstrap\latest.pt"
if (
    $trainingVariant -eq "ai41" -and
    $Resume -eq $ai41CloneBootstrap -and
    -not (Test-Path -LiteralPath $Resume -PathType Leaf)
) {
    $ai41CloneSource = Join-Path $PSScriptRoot "ai40\checkpoints\ai41-tanat-reward-stable-002\promoted.pt"
    if (-not (Test-Path -LiteralPath $ai41CloneSource -PathType Leaf)) {
        throw "AI-41 clone source checkpoint not found: $ai41CloneSource"
    }
    Write-Host "Bootstrapping AI-41 from aggressive AI-30: $Resume"
    & $python -u -m tanat_ai40.clone_ai30 --env $environment --resume $ai41CloneSource --output $Resume `
        --workers $Workers --steps 8192 --tbptt 8 --max-steps $MaxSteps --device $Device
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the AI-30 behavior-cloning bootstrap"
    }
}

$defaultAi41Resume = Join-Path $PSScriptRoot "ai40\checkpoints\ai41-tanat-reward-stable-bootstrap\latest.pt"
if (
    $trainingVariant -eq "ai41" -and
    $Resume -eq $defaultAi41Resume -and
    -not (Test-Path -LiteralPath $Resume -PathType Leaf)
) {
    $ai41Source = Join-Path $PSScriptRoot "ai40\checkpoints\ai41-nav-001\checkpoints\stage-005.pt"
    if (-not (Test-Path -LiteralPath $ai41Source -PathType Leaf)) {
        throw "Frozen stage-005 checkpoint not found: $ai41Source"
    }
    Write-Host "Migrating stage-005 weights to calibrated Tanat reward contract: $Resume"
    & $python -m tanat_ai40.migrate_ai41_strategic $ai41Source --output $Resume
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to migrate the AI-41 strategic checkpoint"
    }
}

$output = Join-Path $PSScriptRoot "ai40\checkpoints\$RunName"
$arguments = @(
    "-m", $campaignModule,
    "--env", $environment,
    "--output", $output,
    "--stages", $Stages,
    "--mirror-per-stage", $MirrorPerStage,
    "--ai30-per-stage", $Ai30PerStage,
    "--historical-per-stage", $HistoricalPerStage,
    "--historical-pool-size", $HistoricalPoolSize,
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
    "--learning-rate", $LearningRate.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--ppo-epochs", $PpoEpochs,
    "--target-kl", $TargetKl.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--env-gomaxprocs", $EnvGoMaxProcs,
    "--device", $Device,
    "--stop-win-rate", $StopWinRate.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--stop-ci-low", $StopCiLow.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--promotion-eval-matches", $PromotionEvalMatches,
    "--promotion-tolerance", $PromotionTolerance.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--promotion-max-category-regression", $PromotionMaxCategoryRegression.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--discount-horizon-seconds", $DiscountHorizonSeconds.ToString([Globalization.CultureInfo]::InvariantCulture),
    "--gae-horizon-seconds", $GaeHorizonSeconds.ToString([Globalization.CultureInfo]::InvariantCulture)
)
if ($HistoricalAnchor) {
    if (-not (Test-Path -LiteralPath $HistoricalAnchor -PathType Leaf)) {
        throw "Historical anchor not found: $HistoricalAnchor"
    }
    $arguments += @("--historical-anchor", (Resolve-Path -LiteralPath $HistoricalAnchor).Path)
}
if ($Resume) {
    if (-not (Test-Path -LiteralPath $Resume -PathType Leaf)) {
        throw "Resume checkpoint not found: $Resume"
    }
    $arguments += @("--resume", (Resolve-Path -LiteralPath $Resume).Path)
}
if ($NoPipeline) {
    $arguments += "--no-pipeline"
}
if ($NoBf16) {
    $arguments += "--no-bf16"
}
if ($NoCompile) {
    $arguments += "--no-compile"
}
if ($CompileLearner) {
    $arguments += "--compile-learner"
}
if ($ExtraArgs) {
    $arguments += $ExtraArgs
}

Write-Host "$modelLabel campaign: $output"
if ($Resume) {
    Write-Host "Initial checkpoint: $((Resolve-Path -LiteralPath $Resume).Path)"
} else {
    Write-Host "Initial checkpoint: random weights (from scratch)"
}
Write-Host "Training: workers=$Workers groups=$GroupSize rollout=$Steps max_steps=$MaxSteps minibatch=$MinibatchSize device=$Device"
Write-Host "PPO: learning_rate=$LearningRate epochs=$PpoEpochs target_kl=$TargetKl (0 disables early stop)"
Write-Host "Historical self-play: per_stage=$HistoricalPerStage pool=$HistoricalPoolSize anchor=$HistoricalAnchor"
Write-Host "Credit assignment: discount=${DiscountHorizonSeconds}s GAE=${GaeHorizonSeconds}s"
Write-Host "Actor-learner pipeline: $(-not $NoPipeline)"
Write-Host "BF16: $(-not $NoBf16); torch.compile actor=$(-not $NoCompile) learner=$($CompileLearner -and -not $NoCompile)"
Write-Host "Evaluation: $EvalMatches -> $EvalMediumMatches at $EvalMediumWinRate -> $EvalFinalMatches at $EvalFinalWinRate; workers=$EvalWorkers"
Write-Host "Stop target: winrate=$StopWinRate CI-low=$StopCiLow"
Write-Host "Press Ctrl+C to stop. Run the same command again to resume."
if ($DryRun) {
    Write-Host "Dry run: no training process started."
    Write-Host ($python + " " + ($arguments -join " "))
    $exitCode = 0
} else {
    # Keep the per-update PPO progress visible in an interactive console and
    # immediately available to redirected training logs.
    & $python "-u" @arguments
    $exitCode = $LASTEXITCODE
}
} catch {
    Write-Host "Training launcher failed: $($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
} finally {
    if (-not $NoPause -and $env:TANAT_TRAINING_NO_PAUSE -ne "1") {
        Write-Host "Training finished with exit code $exitCode."
        [void](Read-Host "Press Enter to close this window")
    }
}

if ($env:TANAT_TRAINING_WRAPPED -eq "1") {
    return $exitCode
}

exit $exitCode
