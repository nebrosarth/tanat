$ErrorActionPreference = "Stop"

$serverRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageRoot = Join-Path $serverRoot "ai40"
$sourceRoot = Join-Path $packageRoot "src"

function Get-NormalizedFlagName {
    param([Parameter(Mandatory = $true)][string]$Argument)

    $equalsIndex = $Argument.IndexOf("=")
    $flag = if ($equalsIndex -ge 0) {
        $Argument.Substring(0, $equalsIndex)
    } else {
        $Argument
    }
    if (-not $flag.StartsWith("-")) {
        return $flag
    }
    return "--" + ($flag -replace "^-+", "")
}

# These values are wrapper-owned or legacy worker controls. They must never
# reach Go from a caller, even when supplied with the `--flag=value` spelling.
$wrapperOwnedFlags = @(
    "--config",
    "--torch-python",
    "--worker-timeout",
    "--worker-command",
    "--worker-arg"
)
$singleValueControlFlags = @(
    "--dataset",
    "--dataset-hash",
    "--profile",
    "--profile-hash",
    "--warm-start",
    "--output",
    "--report",
    "--device"
)
$seenControlFlags = @{}
for ($index = 0; $index -lt $args.Count; $index++) {
    $argument = [string]$args[$index]
    $flag = Get-NormalizedFlagName $argument
    if ($wrapperOwnedFlags -contains $flag) {
        throw "The AI-42 preflight wrapper owns '$flag'; do not pass it explicitly."
    }
    if ($singleValueControlFlags -contains $flag) {
        if ($seenControlFlags.ContainsKey($flag)) {
            throw "Duplicate AI-42 preflight control flag '$flag'."
        }
        $seenControlFlags[$flag] = $true
    }
}

$goCommand = Get-Command go -CommandType Application -ErrorAction SilentlyContinue
if (-not $goCommand) {
    throw "Go toolchain not found. Install Go and make 'go' available on PATH for native AI-42 BC preflight."
}
$go = $goCommand.Path

$candidates = @((Join-Path $packageRoot ".venv\Scripts\python.exe"))
$candidates += Get-ChildItem -LiteralPath $packageRoot -Directory -Filter ".venv.windows.*" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    ForEach-Object { Join-Path $_.FullName "Scripts\python.exe" }
$python = $candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if (-not $python) {
    $pythonCommand = Get-Command python -CommandType Application -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "No Python interpreter found for the AI-42 Torch worker. Install Python/Torch or create ai40\.venv."
    }
    $python = $pythonCommand.Path
}
& $python -c "import torch" *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Python interpreter '$python' cannot import Torch for the AI-42 worker."
}

$previousPythonPath = $env:PYTHONPATH
$previousLocation = Get-Location
try {
    Set-Location -LiteralPath $serverRoot
    $env:PYTHONPATH = if ($previousPythonPath) {
        "$sourceRoot$([IO.Path]::PathSeparator)$previousPythonPath"
    } else { $sourceRoot }
    $config = Join-Path $packageRoot "config\ai42_bc_preflight.json"
    $workerTimeout = "5m"

    # Keep every user argument as its own native-process argument. Go owns the
    # production preflight and launches only the fixed torch_probe_worker_ai42
    # module with this exact interpreter; Python never owns data verification,
    # report publication, or a production hot path.
    $nativeArgs = @("run", "./cmd/ai42preflight", "--config", $config) + $args + @(
        "--torch-python", $python,
        "--worker-timeout", $workerTimeout
    )
    & $go @nativeArgs
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Set-Location -LiteralPath $previousLocation
}
