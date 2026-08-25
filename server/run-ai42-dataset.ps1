$ErrorActionPreference = "Stop"

$serverRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageRoot = Join-Path $serverRoot "ai40"
$sourceRoot = Join-Path $packageRoot "src"
$python = Join-Path $packageRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if (-not $command) { throw "No Python interpreter found for AI-42 dataset collection." }
    $python = $command.Source
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = if ($previousPythonPath) {
        "$sourceRoot$([IO.Path]::PathSeparator)$previousPythonPath"
    } else { $sourceRoot }
    & $python -m tanat_ai40.build_ai42_dataset_go @args
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
