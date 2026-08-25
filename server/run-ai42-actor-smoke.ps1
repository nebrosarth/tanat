$ErrorActionPreference = "Stop"

$serverRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageRoot = Join-Path $serverRoot "ai40"
$sourceRoot = Join-Path $packageRoot "src"

$candidates = @(
    (Join-Path $packageRoot ".venv\Scripts\python.exe")
)
$candidates += Get-ChildItem -LiteralPath $packageRoot -Directory -Filter ".venv.windows.*" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    ForEach-Object { Join-Path $_.FullName "Scripts\python.exe" }

$python = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $python) {
    $command = Get-Command python -ErrorAction SilentlyContinue
    if (-not $command) {
        throw "No Python interpreter found for AI-42 actor smoke."
    }
    $python = $command.Source
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = if ($previousPythonPath) {
        "$sourceRoot$([IO.Path]::PathSeparator)$previousPythonPath"
    } else {
        $sourceRoot
    }
    & $python -m tanat_ai40.smoke_ai42_actor @args
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
