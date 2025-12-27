param(
    [string]$SourcePath = "c:\\AlgoOrange Task\\RTSP_reader\\sample_video",
    [int]$Port = 8554,
    [string]$StreamName = "test"
)

$ErrorActionPreference = "Stop"

function Resolve-Python {
    try {
        $py = (Get-Command py -ErrorAction Stop).Source
        return "$py -3"
    } catch {
        try {
            $python = (Get-Command python -ErrorAction Stop).Source
            return $python
        } catch {
            Write-Error "Python not found. Please install Python 3.8+ and ensure it's in PATH."
            exit 1
        }
    }
}

function Invoke-Cmd($exe, $args) {
    Write-Host "> $exe $args" -ForegroundColor Cyan
    & $exe $args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $exe $args"
    }
}

$pythonCmd = Resolve-Python
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    if ($pythonCmd -like "* -3") {
        & py -3 -m venv $venvPath
    } else {
        & $pythonCmd -m venv $venvPath
    }
}

Write-Host "Installing dependencies..." -ForegroundColor Green
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Host "Starting RTSP streamer..." -ForegroundColor Green
$escapedSource = $SourcePath
& $venvPython (Join-Path $projectRoot "rtsp_streamer.py") --source "$escapedSource" --port $Port --stream-name "$StreamName"
